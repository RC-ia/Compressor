#!/usr/bin/env python3
"""Measure functional similarity between MLP neurons on real token activations.

Unlike analyze_neuron_similarity.py, this does not compare weight vectors. It runs
one text sample through the checkpoint, captures the input-dependent gated MLP
activations, then searches for units whose activation traces are highly correlated.
For each candidate, it estimates the layer-output error if one unit were merged into
the other by adjusting the retained unit's down-projection column.

This is diagnostic only: it neither changes nor prunes the checkpoint. A single
calibration text can miss similarities that only occur on other inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

import compress_tensor as base

THRESHOLDS = (0.90, 0.95, 0.98, 0.99)


class _StopAfterSelectedMlp(RuntimeError):
    """Internal sentinel used to stop the model forward after the selected MLP."""


DEFAULT_TEXT = (
    "A linguagem permite representar ideias, resolver problemas e explicar relações. "
    "Um modelo aprende padrões a partir de muitos exemplos, mas nem todas as unidades "
    "de uma rede neural respondem da mesma maneira a cada entrada. Algumas unidades "
    "podem reagir a padrões semelhantes, enquanto outras respondem a contextos distintos. "
    "Para medir isso, comparamos suas ativações durante o processamento de um texto real. "
    "Também podemos estimar como a saída da camada muda quando duas unidades são fundidas."
)


def discover_mlp_module(model: torch.nn.Module, layer_index: int, requested_name: str | None = None):
    candidates = []
    pattern = re.compile(rf"(?:^|\.)layers\.{layer_index}\.mlp$")
    for name, module in model.named_modules():
        if not all(hasattr(module, key) for key in ("gate_proj", "up_proj", "down_proj")):
            continue
        if requested_name and name != requested_name:
            continue
        if pattern.search(name):
            candidates.append((name, module))
    if not candidates:
        raise KeyError(
            f"Não encontrei MLP gated para layer {layer_index}"
            + (f" com nome {requested_name!r}." if requested_name else ".")
        )
    # Prefer the language model when a multimodal checkpoint has other MLPs.
    candidates.sort(key=lambda pair: ("language_model" not in pair[0], pair[0]))
    if len(candidates) > 1 and not requested_name:
        names = ", ".join(name for name, _ in candidates)
        raise RuntimeError(f"Há várias MLPs candidatas: {names}. Use --module-name.")
    return candidates[0]


def compare_activation_signatures(
    activations: np.ndarray,
    down_weight: np.ndarray,
    layer_output: np.ndarray,
    top_limit: int = 100,
    thresholds: tuple[float, ...] = THRESHOLDS,
    batch_rows: int = 128,
) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    """Compare activation traces and estimate one-unit-into-another merge error."""
    acts = np.asarray(activations, dtype=np.float64)
    down = np.asarray(down_weight, dtype=np.float64)
    output = np.asarray(layer_output, dtype=np.float64)
    if acts.ndim != 2:
        raise ValueError(f"Ativações devem ter forma [tokens, neurônios], recebido {acts.shape}")
    token_count, neuron_count = acts.shape
    if down.ndim != 2 or down.shape[1] != neuron_count:
        raise ValueError(f"down_weight deve ter shape [hidden, {neuron_count}], recebido {down.shape}")
    if output.shape != (token_count, down.shape[0]):
        raise ValueError(f"Saída MLP esperada {(token_count, down.shape[0])}, recebida {output.shape}")

    centered = acts - acts.mean(axis=0, keepdims=True)
    signature_norms = np.linalg.norm(centered, axis=0)
    valid = signature_norms > max(float(signature_norms.max(initial=0.0)) * 1e-8, 1e-12)
    normalized = np.zeros_like(centered, dtype=np.float32)
    normalized[:, valid] = (centered[:, valid] / signature_norms[valid]).astype(np.float32)

    threshold_counts = {f"pairs_abs_corr_ge_{threshold:.2f}": 0 for threshold in thresholds}
    candidates: dict[tuple[int, int], float] = {}
    all_ids = np.arange(neuron_count, dtype=np.int64)

    for start in range(0, neuron_count, batch_rows):
        end = min(neuron_count, start + batch_rows)
        correlations = normalized[:, start:end].T @ normalized
        correlations = np.clip(correlations, -1.0, 1.0)
        absolute = np.abs(correlations)
        row_ids = all_ids[start:end]
        upper_mask = all_ids[None, :] > row_ids[:, None]
        upper_valid = upper_mask & valid[None, start:end].T & valid[None, :]
        for threshold in thresholds:
            threshold_counts[f"pairs_abs_corr_ge_{threshold:.2f}"] += int(
                np.count_nonzero((absolute >= threshold) & upper_valid)
            )
        for local_index, global_index in enumerate(row_ids):
            if not valid[global_index]:
                continue
            row_scores = absolute[local_index].copy()
            row_scores[~valid] = -1.0
            row_scores[global_index] = -1.0
            k = min(4, neuron_count - 1)
            if k <= 0:
                continue
            selected = np.argpartition(row_scores, -k)[-k:]
            for other in selected:
                score = float(row_scores[other])
                if score <= 0 or global_index == int(other):
                    continue
                pair = tuple(sorted((int(global_index), int(other))))
                if score > candidates.get(pair, -1.0):
                    candidates[pair] = score

    ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:top_limit]
    results: list[dict[str, Any]] = []
    layer_output_norm = float(np.linalg.norm(output))
    epsilon = 1e-30
    for (i, j), corr_abs in ranked:
        # Predict activation i from j by least squares through the origin.
        aj = acts[:, j]
        ai = acts[:, i]
        denominator = float(np.dot(aj, aj))
        alpha = float(np.dot(aj, ai) / denominator) if denominator > epsilon else 0.0
        residual_activation = ai - alpha * aj
        down_i = down[:, i]
        down_j = down[:, j]
        pair_output = ai[:, None] * down_i[None, :] + aj[:, None] * down_j[None, :]
        estimated_residual_norm = float(np.linalg.norm(residual_activation) * np.linalg.norm(down_i))
        pair_output_norm = float(np.linalg.norm(pair_output))
        results.append({
            "neuron_a": i,
            "neuron_b": j,
            "activation_correlation_abs": corr_abs,
            "activation_correlation_signed": float(np.corrcoef(acts[:, i], acts[:, j])[0, 1])
                if np.std(acts[:, i]) > 0 and np.std(acts[:, j]) > 0 else 0.0,
            "activation_rms_a": float(np.sqrt(np.mean(ai * ai))),
            "activation_rms_b": float(np.sqrt(np.mean(aj * aj))),
            "merge_a_into_b_scale": alpha,
            "pair_contribution_relative_error_if_merged": estimated_residual_norm / max(pair_output_norm, epsilon),
            "whole_mlp_output_relative_error_estimate": estimated_residual_norm / max(layer_output_norm, epsilon),
            "note": "Estimate on calibration tokens only; merging also changes behavior on unseen inputs.",
        })

    activation_rms = np.sqrt(np.mean(acts * acts, axis=0))
    summary = {
        "token_positions": int(token_count),
        "neurons": int(neuron_count),
        "constant_or_near_constant_activation_count": int((~valid).sum()),
        "median_activation_rms": float(np.median(activation_rms)),
        "p95_activation_rms": float(np.percentile(activation_rms, 95)),
        "layer_output_norm": layer_output_norm,
        "method": "Pearson activation correlation; one-unit merge estimate using calibrated outputs",
    }
    return threshold_counts, results, summary


def resolve_device(model: torch.nn.Module) -> torch.device:
    try:
        device = model.get_input_embeddings().weight.device
    except Exception:
        device = torch.device("meta")
    if device.type != "meta":
        return device
    device_map = getattr(model, "hf_device_map", {})
    for name, mapped_device in device_map.items():
        if "embed_tokens" not in name and "word_embeddings" not in name:
            continue
        if mapped_device == "disk":
            return torch.device("cpu")
        if isinstance(mapped_device, int):
            return torch.device(f"cuda:{mapped_device}" if torch.cuda.is_available() else "cpu")
        return torch.device(mapped_device)
    return next((p.device for p in model.parameters() if p.device.type != "meta"), torch.device("cpu"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=base.DEFAULT_MODEL, help="Checkpoint local Safetensors/Transformers")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--module-name", default=None, help="Nome exato da MLP; necessário se várias MLPs coincidirem")
    parser.add_argument("--max-tokens", type=int, default=256, help="Máximo de tokens de calibração de um único texto")
    parser.add_argument("--text", default=None, help="Texto de calibração; se omitido usa um parágrafo padrão")
    parser.add_argument("--text-file", default=None, help="Arquivo de texto local de calibração (tem prioridade sobre --text)")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-rows", type=int, default=128)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--output-dir", default="neuron_activation_results")
    args = parser.parse_args()
    if args.max_tokens < 8 or args.batch_rows < 1 or args.top < 1:
        parser.error("max-tokens deve ser >= 8; batch-rows e top devem ser positivos.")

    started = time.perf_counter()
    import transformers
    try:
        from transformers import AutoTokenizer
    except Exception as error:
        print(f"[ERRO] Não consegui importar AutoTokenizer: {type(error).__name__}: {error}", file=sys.stderr)
        return 3

    files = base.resolve_model(args.model, args.cache_dir, args.revision)
    source_root = Path(args.model).expanduser()
    if not source_root.is_dir():
        source_root = files[0].parent
    tokenizer = AutoTokenizer.from_pretrained(str(source_root), local_files_only=True)

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text = args.text or DEFAULT_TEXT
    if getattr(tokenizer, "chat_template", None):
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except Exception:
            formatted = f"<|im_start|>user\n{text}\n<|im_end|>\n"
    else:
        formatted = f"<|im_start|>user\n{text}\n<|im_end|>\n"
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=args.max_tokens)
    token_count = int(inputs["input_ids"].shape[-1])
    if token_count < 8:
        parser.error(f"O texto gerou somente {token_count} tokens; forneça texto de calibração maior.")

    try:
        loader = getattr(transformers, "AutoModelForMultimodalLM", None)
        if loader is None:
            loader = getattr(transformers, "AutoModelForImageTextToText")
        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        print(f"Carregando modelo e tokenizer: {source_root}", flush=True)
        model = loader.from_pretrained(
            str(source_root), dtype=dtype, device_map=args.device_map,
            local_files_only=True, low_cpu_mem_usage=True,
        )
    except Exception as error:
        print(f"[ERRO] Falha ao carregar o modelo: {type(error).__name__}: {error}", file=sys.stderr)
        return 3

    try:
        module_name, mlp = discover_mlp_module(model, args.layer_index, args.module_name)
    except Exception as error:
        print(f"[ERRO] {error}", file=sys.stderr)
        print("Use --module-name com um dos módulos que termina em layers.N.mlp.", file=sys.stderr)
        return 2

    print(f"Camada MLP: {module_name}", flush=True)
    print(f"Texto de calibração: {token_count} tokens; não haverá geração de texto.", flush=True)
    captured: dict[str, torch.Tensor] = {}

    def save_output(key: str):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            captured[key] = output.detach().to(device="cpu", dtype=torch.float32).contiguous()
        return hook

    def capture_mlp_and_stop(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        captured["mlp_output"] = output.detach().to(device="cpu", dtype=torch.float32).contiguous()
        # Stop immediately after the selected MLP. Later transformer layers are
        # irrelevant to this layer-local analysis and can be heavily disk-offloaded.
        raise _StopAfterSelectedMlp()

    handles = [
        mlp.gate_proj.register_forward_hook(save_output("gate")),
        mlp.up_proj.register_forward_hook(save_output("up")),
        mlp.register_forward_hook(capture_mlp_and_stop),
    ]

    # Safetensors is authoritative for the down-projection column vectors; this
    # avoids reading a dispatched parameter after Accelerate returns it to meta.
    all_tensors = base.list_tensors(files)
    expected_down_name = f"{module_name}.down_proj.weight"
    down_item = next((item for item in all_tensors if item["name"] == expected_down_name), None)
    if down_item is None:
        suffix = f".layers.{args.layer_index}.mlp.down_proj.weight"
        possibilities = [item for item in all_tensors if item["name"].endswith(suffix)]
        if len(possibilities) == 1:
            down_item = possibilities[0]
        else:
            for handle in handles:
                handle.remove()
            names = [item["name"] for item in possibilities[:10]]
            print(f"[ERRO] Não consegui localizar {expected_down_name}. Candidatos: {names}", file=sys.stderr)
            return 2
    with safe_open(str(down_item["file"]), framework="pt", device="cpu") as sf:
        down_weight = sf.get_tensor(down_item["name"]).to(torch.float32).numpy().copy()

    device = resolve_device(model)
    model_prefix = getattr(model, "base_model_prefix", "model")
    backbone = getattr(model, model_prefix, None)
    if backbone is None:
        backbone = model
    model_inputs = {key: value.to(device) for key, value in inputs.items()}
    print("Executando o forward somente até a saída da MLP-alvo (as camadas seguintes serão puladas)...", flush=True)
    try:
        with torch.inference_mode():
            backbone(**model_inputs, use_cache=False, return_dict=True)
    except _StopAfterSelectedMlp:
        # Expected: the hook intentionally interrupts execution after target MLP.
        pass
    except Exception as error:
        for handle in handles:
            handle.remove()
        print(f"[ERRO] Forward falhou: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        for handle in handles:
            handle.remove()
        del model_inputs

    if not all(key in captured for key in ("gate", "up", "mlp_output")):
        print(f"[ERRO] A MLP não produziu todas as ativações esperadas. Capturado: {list(captured)}", file=sys.stderr)
        return 1

    gate = captured["gate"].reshape(-1, captured["gate"].shape[-1])
    up = captured["up"].reshape(-1, captured["up"].shape[-1])
    mlp_output = captured["mlp_output"].reshape(-1, captured["mlp_output"].shape[-1])
    if gate.shape != up.shape:
        print(f"[ERRO] Formas incompatíveis: gate={tuple(gate.shape)}, up={tuple(up.shape)}", file=sys.stderr)
        return 1
    act_fn = getattr(mlp, "act_fn", torch.nn.functional.silu)
    try:
        activations = (act_fn(gate) * up).numpy()
    except Exception:
        activations = (torch.nn.functional.silu(gate) * up).numpy()
    del captured, gate, up

    if down_weight.shape[1] != activations.shape[1]:
        print(f"[ERRO] down_proj={down_weight.shape}, ativações={activations.shape}", file=sys.stderr)
        return 1

    counts, candidates, summary = compare_activation_signatures(
        activations, down_weight, mlp_output.numpy(), top_limit=args.top, batch_rows=args.batch_rows,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": str(source_root),
        "layer_index": args.layer_index,
        "module_name": module_name,
        "calibration_tokens": token_count,
        "activation_samples": int(activations.shape[0]),
        "threshold_pair_counts": counts,
        "summary": summary,
        "top_candidates": candidates,
        "elapsed_seconds": time.perf_counter() - started,
        "note": "One calibration text only. Confirm candidates on more calibration inputs before pruning.",
    }
    report_path = output_dir / "neuron_activation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SIMILARIDADE FUNCIONAL DE NEURÔNIOS ===")
    print(f"Camada: {module_name}")
    print(f"Tokens de calibração: {token_count}")
    print(f"Amostras de ativação: {activations.shape[0]} | neurônios: {activations.shape[1]}")
    for threshold, count in ((f"{key}", value) for key, value in counts.items()):
        print(f"Pares com correlação de ativação {threshold.replace('pairs_abs_corr_ge_', '')}: {count:,}")
    if candidates:
        print("\nMelhores pares e erro estimado ao fundir:")
        for item in candidates[:min(10, len(candidates))]:
            print(
                f"  {item['neuron_a']} ↔ {item['neuron_b']} | "
                f"|corr|={item['activation_correlation_abs']:.4f} "
                f"alpha={item['merge_a_into_b_scale']:.4f} "
                f"erro_par={item['pair_contribution_relative_error_if_merged']:.4f} "
                f"erro_camada≈{item['whole_mlp_output_relative_error_estimate']:.6f}"
            )
    print(f"Relatório: {report_path}")
    print(f"Tempo total: {report['elapsed_seconds']:.1f}s")
    print("O script só analisa; não remove nem modifica neurônios.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
