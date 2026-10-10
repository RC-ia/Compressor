#!/usr/bin/env python3
"""Apply a sparse NF4 correction sidecar during Transformers inference.

The original BNB checkpoint remains untouched. Forward hooks add E*x to each
matched Linear4bit output, where E contains only the FP16 residual entries
exported by compare_bnb_weights.py.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from bitsandbytes.nn import Linear4bit
except Exception:
    Linear4bit = None


def read_uvarint(data: bytes, position: int, end: int) -> tuple[int, int]:
    """Read unsigned LEB128 and return (value, updated_position)."""
    value = 0
    shift = 0
    while position < end:
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, position
        shift += 7
        if shift > 63:
            raise ValueError("Índice LEB128 excedeu 64 bits.")
    raise ValueError("Mapa truncado durante leitura do índice LEB128.")


def load_map(
    map_path: Path, manifest_path: Path
) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = map_path.read_bytes()
    expected_bytes = payload.get("map_bytes")
    if expected_bytes is not None and int(expected_bytes) != len(data):
        raise ValueError(
            f"Tamanho do mapa divergente: manifesto={expected_bytes}, arquivo={len(data)}."
        )

    records: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for entry in payload.get("tensor_manifest", []):
        tensor_name = entry["tensor"]
        if tensor_name in seen_names:
            raise ValueError(f"Tensor duplicado no manifesto: {tensor_name}")
        seen_names.add(tensor_name)

        shape = tuple(int(v) for v in entry["shape"])
        if len(shape) != 2:
            raise ValueError(
                f"{tensor_name}: esperado tensor Linear 2D, recebido shape={shape}."
            )
        num_weights = math.prod(shape)
        if num_weights != int(entry["num_weights"]):
            raise ValueError(f"{tensor_name}: num_weights não coincide com shape.")

        start = int(entry["encoding_offset_bytes"])
        length = int(entry["encoding_length_bytes"])
        count = int(entry["correction_count"])
        end = start + length
        if start < 0 or length < 0 or end > len(data):
            raise ValueError(f"{tensor_name}: offsets do mapa fora do arquivo.")

        position = start
        previous_index = -1
        indices: list[int] = []
        residuals: list[float] = []
        for _ in range(count):
            delta, position = read_uvarint(data, position, end)
            if delta <= 0:
                raise ValueError(f"{tensor_name}: delta de índice inválido.")
            flat_index = previous_index + delta
            if flat_index < 0 or flat_index >= num_weights:
                raise ValueError(
                    f"{tensor_name}: índice {flat_index} fora do tensor ({num_weights})."
                )
            if position + 2 > end:
                raise ValueError(f"{tensor_name}: falta resíduo FP16 no mapa.")
            residual = struct.unpack_from("<e", data, position)[0]
            position += 2
            indices.append(flat_index)
            residuals.append(float(residual))
            previous_index = flat_index

        if position != end:
            raise ValueError(
                f"{tensor_name}: {end - position} bytes sobrando na seção do tensor."
            )

        index_tensor = torch.tensor(indices, dtype=torch.long)
        residual_tensor = torch.tensor(residuals, dtype=torch.float32)
        out_features, in_features = shape
        records.append({
            "tensor": tensor_name,
            "module": tensor_name[:-7] if tensor_name.endswith(".weight") else tensor_name,
            "shape": shape,
            "out_features": out_features,
            "in_features": in_features,
            "rows": torch.div(index_tensor, in_features, rounding_mode="floor"),
            "cols": torch.remainder(index_tensor, in_features),
            "values": residual_tensor,
            "count": count,
        })

    if not records:
        raise ValueError("O manifesto não contém tensores com correções.")
    return records


def _device_cache(record: dict[str, Any], device: torch.device):
    key = str(device)
    cache = record.setdefault("_device_cache", {})
    if key not in cache:
        cache[key] = (
            record["rows"].to(device=device, non_blocking=True),
            record["cols"].to(device=device, non_blocking=True),
            record["values"].to(device=device, dtype=torch.float32, non_blocking=True),
        )
    return cache[key]


def install_hooks(model: nn.Module, records: list[dict[str, Any]], state: dict[str, bool],
                  max_temp_elements: int = 1_000_000) -> tuple[list[Any], dict[str, int]]:
    modules = dict(model.named_modules())
    handles = []
    stats = {
        "installed": 0, "missing": 0, "shape_mismatch": 0,
        "not_4bit_linear": 0, "alias_resolved": 0,
    }
    missing_names = []
    alias_examples = []

    for record in records:
        manifest_module_name = record["module"]
        # The sidecar was built from the text checkpoint's tensor keys, which can
        # retain the multimodal wrapper "model.language_model." even though
        # AutoModelForCausalLM exposes those modules under "model.".
        candidates = [manifest_module_name]
        if manifest_module_name.startswith("model.language_model."):
            candidates.append("model." + manifest_module_name[len("model.language_model."):])
        elif manifest_module_name.startswith("language_model."):
            candidates.append("model." + manifest_module_name[len("language_model."):])

        module_name = next((name for name in candidates if name in modules), None)
        module = modules.get(module_name) if module_name is not None else None
        if module is None:
            stats["missing"] += 1
            missing_names.append({
                "manifest_name": manifest_module_name,
                "tried_names": candidates,
            })
            continue

        if module_name != manifest_module_name:
            stats["alias_resolved"] += 1
            if len(alias_examples) < 20:
                alias_examples.append({
                    "manifest_name": manifest_module_name,
                    "resolved_module": module_name,
                })
        record["resolved_module"] = module_name
        if Linear4bit is not None and not isinstance(module, Linear4bit):
            stats["not_4bit_linear"] += 1
            continue
        if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
            stats["not_4bit_linear"] += 1
            continue
        if (
            int(module.out_features) != record["out_features"]
            or int(module.in_features) != record["in_features"]
        ):
            stats["shape_mismatch"] += 1
            continue

        def make_hook(rec: dict[str, Any]):
            def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any):
                if not state["enabled"]:
                    return output
                if not isinstance(output, torch.Tensor) or not inputs:
                    raise TypeError(f"{rec['module']}: saída/entrada inesperada no forward hook.")
                x = inputs[0]
                if not isinstance(x, torch.Tensor) or x.shape[-1] != rec["in_features"]:
                    raise ValueError(f"{rec['module']}: dimensão da entrada incompatível.")

                rows, cols, values = _device_cache(rec, x.device)
                flat_x = x.reshape(-1, rec["in_features"])
                result = output.contiguous()
                flat_out = result.view(-1, rec["out_features"])
                nnz = int(values.numel())
                tokens_per_chunk = max(1, max_temp_elements // max(1, nnz))

                # Process token chunks to bound the temporary [tokens, nnz] buffer.
                for begin in range(0, flat_x.shape[0], tokens_per_chunk):
                    finish = min(flat_x.shape[0], begin + tokens_per_chunk)
                    x_chunk = flat_x[begin:finish]
                    contributions = x_chunk.index_select(1, cols).float()
                    contributions.mul_(values.unsqueeze(0))
                    correction = torch.zeros(
                        (finish - begin, rec["out_features"]),
                        device=x.device,
                        dtype=torch.float32,
                    )
                    row_index = rows.unsqueeze(0).expand(finish - begin, -1)
                    correction.scatter_add_(1, row_index, contributions)
                    flat_out[begin:finish].add_(correction.to(dtype=flat_out.dtype))
                return result
            return hook

        handles.append(module.register_forward_hook(make_hook(record)))
        stats["installed"] += 1

    if stats["missing"] or stats["shape_mismatch"] or stats["not_4bit_linear"]:
        details = {
            "stats": stats,
            "missing_examples": missing_names[:20],
            "alias_examples": alias_examples,
        }
        for handle in handles:
            handle.remove()
        raise RuntimeError(
            "Nem todas as correções foram associadas a camadas Linear4bit compatíveis. "
            + json.dumps(details, ensure_ascii=False)
        )
    return handles, stats


def prompt_inputs(tokenizer: Any, prompt: str) -> dict[str, torch.Tensor]:
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt")


def choose_input_device(model: nn.Module) -> torch.device:
    embeddings = model.get_input_embeddings()
    if embeddings is not None:
        device = embeddings.weight.device
        if device.type != "meta":
            return device

    device_map = getattr(model, "hf_device_map", {})
    named = dict(model.named_modules())
    embed_name = None
    if embeddings is not None:
        embed_name = next((name for name, module in named.items() if module is embeddings), None)
    if embed_name is not None:
        candidates = [
            (name, value) for name, value in device_map.items()
            if embed_name == name or embed_name.startswith(name + ".")
        ]
        if candidates:
            value = max(candidates, key=lambda item: len(item[0]))[1]
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
            if isinstance(value, str) and value not in ("disk", "meta"):
                return torch.device(value)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compare_logits(model: nn.Module, inputs: dict[str, torch.Tensor],
                   state: dict[str, bool]) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    state["enabled"] = False
    with torch.inference_mode():
        baseline = model(**inputs, use_cache=False).logits[:, -1, :].float().cpu()
    state["enabled"] = True
    with torch.inference_mode():
        corrected = model(**inputs, use_cache=False).logits[:, -1, :].float().cpu()

    diff = corrected - baseline
    a = baseline.reshape(-1).double()
    b = corrected.reshape(-1).double()
    dot = float(torch.dot(a, b))
    norm = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    metrics = {
        "max_abs_logit_difference": float(diff.abs().max()),
        "rmse_logit_difference": float(torch.mean(diff.double() ** 2).sqrt()),
        "argmax_changed": int(baseline.argmax(dim=-1).item() != corrected.argmax(dim=-1).item()),
        "baseline_top_token_id": int(baseline.argmax(dim=-1).item()),
        "corrected_top_token_id": int(corrected.argmax(dim=-1).item()),
        "logit_cosine_similarity": dot / norm if norm else None,
    }
    return metrics, baseline, corrected


def compare_candidate_to_reference(
    candidate: torch.Tensor, reference: torch.Tensor
) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        raise ValueError(
            "Os logits têm dimensões diferentes: "
            f"candidato={tuple(candidate.shape)}, referência={tuple(reference.shape)}. "
            "Confirme se os dois checkpoints têm o mesmo vocabulário."
        )
    candidate64 = candidate.reshape(-1).double()
    reference64 = reference.reshape(-1).double()
    diff = candidate64 - reference64
    rmse = float(torch.mean(diff.square()).sqrt())
    reference_std = float(reference64.std(unbiased=False))
    candidate_norm = float(torch.linalg.vector_norm(candidate64))
    reference_norm = float(torch.linalg.vector_norm(reference64))
    denominator = candidate_norm * reference_norm
    cosine = float(torch.dot(candidate64, reference64) / denominator) if denominator else None
    candidate_top = int(candidate.argmax(dim=-1).item())
    reference_top = int(reference.argmax(dim=-1).item())
    return {
        "rmse_to_reference": rmse,
        "mae_to_reference": float(diff.abs().mean()),
        "max_abs_difference_to_reference": float(diff.abs().max()),
        "rmse_over_reference_logit_std": rmse / reference_std if reference_std else None,
        "cosine_similarity_to_reference": cosine,
        "top_token_id": candidate_top,
        "reference_top_token_id": reference_top,
        "top_token_matches_reference": candidate_top == reference_top,
    }


def generate_once(model: nn.Module, tokenizer: Any, inputs: dict[str, torch.Tensor],
                  state: dict[str, bool], enabled: bool, max_new_tokens: int) -> tuple[str, float]:
    state["enabled"] = enabled
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    generated_ids = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True), elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="techwithsergiu/Qwen3.5-text-4B-bnb-4bit")
    parser.add_argument("--map-file", required=True, help="correction_map_gt_*.bin gerado pelo comparador.")
    parser.add_argument("--manifest", required=True, help="correction_map_manifest.json correspondente.")
    parser.add_argument("--prompt", default="Explique brevemente por que o céu parece azul.")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--device-map", choices=["gpu", "auto"], default="gpu",
        help="'gpu' força o modelo inteiro na CUDA 0; 'auto' permite dispatch automático (pode falhar em BNB 4-bit)."
    )
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16", "float32"],
                        default="float16",
                        help="Precisão de cálculo das camadas NF4; float16 é o padrão para GPUs Pascal/GTX 10.")
    parser.add_argument("--max-temp-elements", type=int, default=1_000_000,
                        help="Limite aproximado para o buffer temporário tokens x correções.")
    parser.add_argument("--output", default="bnb_correction_runtime_test.json")
    parser.add_argument(
        "--reference-model", default=None,
        help="Checkpoint BF16 original do MESMO backbone textual (ex.: techwithsergiu/Qwen3.5-text-4B). "
             "Quando informado, mede se os logits corrigidos ficam mais próximos dele."
    )
    parser.add_argument(
        "--reference-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto",
        help="'auto' preserva o dtype declarado no checkpoint; use bfloat16 para forçar BF16."
    )
    parser.add_argument(
        "--reference-max-cpu-memory", default="4GiB",
        help="Limite de RAM reservado ao modelo de referência; o excedente será enviado para disco (padrão: 4GiB)."
    )
    parser.add_argument(
        "--reference-offload-folder", default=None,
        help="Pasta para pesos temporários do modelo BF16 enviados ao disco. Padrão: <pasta do relatório>/bf16_offload."
    )
    args = parser.parse_args()

    if Linear4bit is None:
        parser.error("bitsandbytes não está instalado ou Linear4bit não pôde ser importado.")
    if args.max_new_tokens < 1 or args.max_temp_elements < 1:
        parser.error("--max-new-tokens e --max-temp-elements precisam ser positivos.")

    map_path = Path(args.map_file).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not map_path.is_file() or not manifest_path.is_file():
        parser.error("Não encontrei o mapa ou o manifesto informado.")

    records = load_map(map_path, manifest_path)
    print(f"Mapa: {map_path} ({map_path.stat().st_size:,} bytes)", flush=True)
    print(f"Tensores no manifesto: {len(records)} | correções: {sum(r['count'] for r in records):,}", flush=True)

    print(f"Carregando modelo {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    compute_dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    # This repository already contains a BNB quantization_config. Modify that
    # loaded config in memory instead of passing a second quantization_config,
    # which Transformers intentionally ignores for an already-quantized model.
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    saved_quant_config = getattr(model_config, "quantization_config", None)
    if isinstance(saved_quant_config, dict):
        saved_quant_config = dict(saved_quant_config)
        saved_quant_config["bnb_4bit_compute_dtype"] = args.compute_dtype
        model_config.quantization_config = saved_quant_config
    elif saved_quant_config is not None and hasattr(saved_quant_config, "bnb_4bit_compute_dtype"):
        saved_quant_config.bnb_4bit_compute_dtype = compute_dtypes[args.compute_dtype]
    else:
        raise RuntimeError(
            "O checkpoint não expõe uma quantization_config bitsandbytes reconhecível."
        )

    selected_device_map = {"": 0} if args.device_map == "gpu" else "auto"
    if args.device_map == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("--device-map gpu exige torch.cuda.is_available() == True.")
    if args.device_map == "gpu":
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        print(
            f"CUDA 0: {free_bytes / 1024**3:.2f} GiB livres de "
            f"{total_bytes / 1024**3:.2f} GiB; carregamento será forçado para a GPU.",
            flush=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=model_config,
        device_map=selected_device_map,
        dtype=compute_dtypes[args.compute_dtype],
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    state = {"enabled": False}
    handles, hook_stats = install_hooks(
        model, records, state, max_temp_elements=args.max_temp_elements
    )
    print(f"Hooks instalados: {hook_stats['installed']}", flush=True)

    inputs = prompt_inputs(tokenizer, args.prompt)
    input_device = choose_input_device(model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    print("Comparando logits antes/depois ...", flush=True)
    logit_metrics, baseline_logits, corrected_logits = compare_logits(model, inputs, state)
    print(json.dumps(logit_metrics, ensure_ascii=False, indent=2), flush=True)

    baseline_text, baseline_seconds = generate_once(
        model, tokenizer, inputs, state, False, args.max_new_tokens
    )
    corrected_text, corrected_seconds = generate_once(
        model, tokenizer, inputs, state, True, args.max_new_tokens
    )

    quantized_device_map = getattr(model, "hf_device_map", {})
    reference_comparison = None

    if args.reference_model:
        print("\nLiberando o NF4 antes de carregar a referência BF16 ...", flush=True)
        reference_inputs_cpu = {key: value.detach().cpu() for key, value in inputs.items()}
        for handle in handles:
            handle.remove()
        for record in records:
            record.pop("_device_cache", None)
        del inputs
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        reference_dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        output_parent = Path(args.output).expanduser().resolve().parent
        offload_folder = (
            Path(args.reference_offload_folder).expanduser().resolve()
            if args.reference_offload_folder
            else output_parent / "bf16_offload"
        )
        offload_folder.mkdir(parents=True, exist_ok=True)
        print(
            f"Carregando referência {args.reference_model} com dtype={args.reference_dtype}; "
            f"limite de RAM={args.reference_max_cpu_memory}. "
            "Atenção: camadas excedentes serão temporariamente armazenadas em disco.",
            flush=True,
        )
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.reference_model,
            device_map="auto",
            max_memory={"cpu": args.reference_max_cpu_memory},
            offload_folder=str(offload_folder),
            offload_state_dict=True,
            dtype=reference_dtype_map[args.reference_dtype],
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        reference_model.eval()
        reference_device_map = getattr(reference_model, "hf_device_map", {})
        reference_input_device = choose_input_device(reference_model)
        reference_inputs = {
            key: value.to(reference_input_device)
            for key, value in reference_inputs_cpu.items()
        }
        print(f"Dispositivo de entrada da referência: {reference_input_device}", flush=True)
        with torch.inference_mode():
            reference_logits = (
                reference_model(**reference_inputs, use_cache=False)
                .logits[:, -1, :].float().cpu()
            )
        base_vs_reference = compare_candidate_to_reference(baseline_logits, reference_logits)
        corrected_vs_reference = compare_candidate_to_reference(corrected_logits, reference_logits)
        base_rmse = base_vs_reference["rmse_to_reference"]
        corrected_rmse = corrected_vs_reference["rmse_to_reference"]
        improvement_percent = (
            (base_rmse - corrected_rmse) / base_rmse * 100.0 if base_rmse else None
        )
        reference_input_dtype = str(reference_model.get_input_embeddings().weight.dtype)
        reference_comparison = {
            "reference_model": args.reference_model,
            "requested_reference_dtype": args.reference_dtype,
            "reference_embedding_dtype": reference_input_dtype,
            "reference_device_map": reference_device_map,
            "reference_offload_folder": str(offload_folder),
            "reference_max_cpu_memory": args.reference_max_cpu_memory,
            "comparison_scope": "logits do último token de entrada",
            "nf4_vs_reference": base_vs_reference,
            "corrected_vs_reference": corrected_vs_reference,
            "correction_rmse_improvement_percent": improvement_percent,
            "closer_to_reference": (
                corrected_rmse < base_rmse
            ),
            "interpretation": (
                "Positivo em correction_rmse_improvement_percent significa que a correção "
                "reduziu o RMSE dos logits em relação à referência; negativo significa piora."
            ),
        }
        print("\n=== COMPARAÇÃO CONTRA A REFERÊNCIA ===", flush=True)
        print(json.dumps(reference_comparison, ensure_ascii=False, indent=2), flush=True)
        del reference_inputs, reference_inputs_cpu, reference_logits, reference_model
        gc.collect()

    report = {
        "model": args.model,
        "map_file": str(map_path),
        "manifest_file": str(manifest_path),
        "map_bytes": map_path.stat().st_size,
        "correction_threshold": json.loads(manifest_path.read_text(encoding="utf-8")).get("threshold"),
        "tensor_count": len(records),
        "correction_count": sum(record["count"] for record in records),
        "hook_stats": hook_stats,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "compute_dtype": args.compute_dtype,
        "requested_device_map": args.device_map,
        "actual_device_map": quantized_device_map,
        "logit_comparison": logit_metrics,
        "reference_comparison": reference_comparison,
        "baseline_generation": baseline_text,
        "corrected_generation": corrected_text,
        "baseline_generation_seconds": baseline_seconds,
        "corrected_generation_seconds": corrected_seconds,
        "generation_seconds_ratio_corrected_over_baseline": (
            corrected_seconds / baseline_seconds if baseline_seconds else None
        ),
        "note": "Prototype only. It applies sparse residuals through PyTorch forward hooks; it does not rewrite the checkpoint.",
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for handle in handles:
        handle.remove()

    print("\n=== RESULTADO DA CORREÇÃO ===")
    print(f"Base:      {baseline_text}")
    print(f"Corrigido: {corrected_text}")
    print(f"Tempo base: {baseline_seconds:.2f}s | corrigido: {corrected_seconds:.2f}s")
    print(f"Relatório: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
