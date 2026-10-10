#!/usr/bin/env python3
"""Compare a real bitsandbytes NF4 checkpoint with its BF16 source weights.

The quantized checkpoint is DEQUANTIZED from its stored NF4 codes and QuantState;
this script does not simulate quantization. It streams unquantized tensors in
chunks and dequantizes one quantized tensor at a time.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

import compress_tensor as base

try:
    import bitsandbytes.functional as bnb_F
except Exception:
    bnb_F = None

DEFAULT_SOURCE = "techwithsergiu/Qwen3.5-text-4B"
DEFAULT_QUANTIZED = "techwithsergiu/Qwen3.5-text-4B-bnb-4bit"
QUANT_STATE_MARKER = ".quant_state.bitsandbytes__"
DEFAULT_THRESHOLDS = (1.0, 0.5, 0.1, 0.05, 0.01)


def parse_thresholds(value: str) -> list[float]:
    try:
        values = [float(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use limites numéricos separados por vírgula.") from exc
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise argparse.ArgumentTypeError("Os limites devem ser finitos e não negativos.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Os limites não podem se repetir.")
    return sorted(values, reverse=True)


def index_tensors(files: list[Path]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in base.list_tensors(files)}


def inspect_quantized_weights(
    target_index: dict[str, dict[str, Any]],
) -> tuple[set[str], set[str], dict[str, list[str]]]:
    quantized_names = {
        name.split(QUANT_STATE_MARKER, 1)[0]
        for name in target_index
        if QUANT_STATE_MARKER in name
    }
    if not quantized_names:
        raise RuntimeError(
            "Nenhum peso BNB pré-quantizado foi encontrado. Verifique se o destino "
            "é um checkpoint bitsandbytes NF4 salvo em Safetensors."
        )

    auxiliary_names: set[str] = set()
    state_keys_by_weight: dict[str, list[str]] = {}
    for weight_name in quantized_names:
        prefix = weight_name + "."
        state_keys = sorted(
            name for name in target_index
            if name.startswith(prefix)
        )
        if not any(QUANT_STATE_MARKER in name for name in state_keys):
            raise RuntimeError(f"Estado de quantização ausente para {weight_name}")
        auxiliary_names.update(state_keys)
        state_keys_by_weight[weight_name] = state_keys
    return quantized_names, auxiliary_names, state_keys_by_weight


def source_name_for_target(
    target_name: str,
    source_index: dict[str, dict[str, Any]],
) -> str | None:
    """Match exact text-only names first, then the official Qwen VLM text prefix."""
    if target_name in source_index:
        return target_name

    # Official Qwen3.5-4B names the text backbone under model.language_model.
    # The selected text-only derivative removes that extra wrapper.
    if target_name.startswith("model."):
        alias = "model.language_model." + target_name[len("model."):]
        if alias in source_index:
            return alias

    return None


def open_safetensors(files: list[Path], stack: ExitStack) -> dict[Path, Any]:
    return {
        path: stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
        for path in sorted(set(files))
    }


def source_array(handle: Any, name: str, shape: tuple[int, ...], row_start: int, row_end: int) -> np.ndarray:
    if not shape:
        tensor = handle.get_tensor(name)
    else:
        tensor = handle.get_slice(name)[row_start:row_end]
    return tensor.detach().cpu().float().contiguous().reshape(-1).numpy()


def row_chunks(shape: tuple[int, ...], chunk_elements: int):
    if not shape:
        yield 0, 1, 0, 1
        return
    inner = math.prod(shape[1:]) if len(shape) > 1 else 1
    rows_per_chunk = max(1, chunk_elements // max(1, inner))
    for row_start in range(0, shape[0], rows_per_chunk):
        row_end = min(shape[0], row_start + rows_per_chunk)
        yield row_start, row_end, row_start * inner, row_end * inner


def new_metrics(thresholds: list[float]) -> dict[str, Any]:
    return {
        "numel": 0,
        "nonzero_error_count": 0,
        "sum_abs_error": 0.0,
        "sum_squared_error": 0.0,
        "max_abs_error": 0.0,
        "sum_original": 0.0,
        "sum_original_squared": 0.0,
        "sum_reconstructed_squared": 0.0,
        "dot_original_reconstructed": 0.0,
        "threshold_counts": {format(t, "g"): 0 for t in thresholds},
    }


def add_metrics(
    metrics: dict[str, Any],
    original: np.ndarray,
    reconstructed: np.ndarray,
    thresholds: list[float],
) -> np.ndarray:
    orig = np.asarray(original, dtype=np.float32).reshape(-1)
    recon = np.asarray(reconstructed, dtype=np.float32).reshape(-1)
    if orig.size != recon.size:
        raise ValueError(f"Quantidade de valores divergente: origem={orig.size}, destino={recon.size}")

    error = recon - orig
    abs_error = np.abs(error)
    orig64 = orig.astype(np.float64)
    recon64 = recon.astype(np.float64)
    err64 = error.astype(np.float64)

    metrics["numel"] += int(orig.size)
    metrics["nonzero_error_count"] += int(np.count_nonzero(error))
    metrics["sum_abs_error"] += float(np.sum(abs_error, dtype=np.float64))
    metrics["sum_squared_error"] += float(np.dot(err64, err64))
    metrics["max_abs_error"] = max(
        metrics["max_abs_error"], float(abs_error.max(initial=0.0))
    )
    metrics["sum_original"] += float(np.sum(orig64, dtype=np.float64))
    metrics["sum_original_squared"] += float(np.dot(orig64, orig64))
    metrics["sum_reconstructed_squared"] += float(np.dot(recon64, recon64))
    metrics["dot_original_reconstructed"] += float(np.dot(orig64, recon64))

    for threshold in thresholds:
        metrics["threshold_counts"][format(threshold, "g")] += int(
            np.count_nonzero(abs_error > threshold)
        )
    return abs_error


def finish_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    count = int(metrics["numel"])
    if count == 0:
        return {
            "numel": 0, "nonzero_error_count": 0, "mae": 0.0, "rmse": 0.0,
            "max_abs_error": 0.0, "source_weight_std": 0.0,
            "rmse_over_source_std": 0.0, "cosine_similarity": None,
            "threshold_counts": metrics["threshold_counts"],
        }

    mae = metrics["sum_abs_error"] / count
    rmse = math.sqrt(metrics["sum_squared_error"] / count)
    mean = metrics["sum_original"] / count
    variance = max(0.0, metrics["sum_original_squared"] / count - mean * mean)
    source_std = math.sqrt(variance)
    norm = math.sqrt(
        metrics["sum_original_squared"] * metrics["sum_reconstructed_squared"]
    )
    cosine = metrics["dot_original_reconstructed"] / norm if norm else None
    return {
        "numel": count,
        "nonzero_error_count": int(metrics["nonzero_error_count"]),
        "mae": mae,
        "rmse": rmse,
        "max_abs_error": float(metrics["max_abs_error"]),
        "source_weight_std": source_std,
        "rmse_over_source_std": rmse / source_std if source_std else 0.0,
        "cosine_similarity": cosine,
        "threshold_counts": metrics["threshold_counts"],
    }


def update_top_weights(
    heap: list[tuple[float, int, dict[str, Any]]],
    top_k: int,
    sequence: int,
    tensor_name: str,
    shape: tuple[int, ...],
    flat_offset: int,
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> int:
    if top_k <= 0 or original.size == 0:
        return sequence
    orig = np.asarray(original, dtype=np.float32).reshape(-1)
    recon = np.asarray(reconstructed, dtype=np.float32).reshape(-1)
    errors = recon - orig
    absolute = np.abs(errors)
    take = min(top_k, absolute.size)
    candidates = (
        np.argpartition(absolute, absolute.size - take)[-take:]
        if take < absolute.size else np.arange(absolute.size)
    )
    for local_raw in candidates:
        local = int(local_raw)
        magnitude = float(absolute[local])
        if magnitude == 0.0:
            continue
        if len(heap) >= top_k and magnitude <= heap[0][0]:
            continue

        index = flat_offset + local
        coords = (
            [int(v) for v in np.unravel_index(index, shape)]
            if shape else []
        )
        record = {
            "tensor": tensor_name,
            "shape": list(shape),
            "flat_index": index,
            "coordinates": coords,
            "source_value": float(orig[local]),
            "bnb_dequantized_value": float(recon[local]),
            "delta_bnb_minus_source": float(errors[local]),
            "absolute_error": magnitude,
        }
        sequence += 1
        node = (magnitude, sequence, record)
        if len(heap) < top_k:
            heapq.heappush(heap, node)
        elif magnitude > heap[0][0]:
            heapq.heapreplace(heap, node)
    return sequence


def dequantize_bnb_tensor(
    target_handle: Any,
    weight_name: str,
    state_keys: list[str],
    device: torch.device,
) -> torch.Tensor:
    quantized_data = target_handle.get_tensor(weight_name).to(device)
    prefix = weight_name + "."
    state_dict = {
        key[len(prefix):]: target_handle.get_tensor(key)
        for key in state_keys
    }
    try:
        quant_state = bnb_F.QuantState.from_dict(qs_dict=state_dict, device=device)
        restored = bnb_F.dequantize_4bit(
            quantized_data,
            quant_state=quant_state,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Falha ao desquantizar {weight_name} em {device}. "
            "Use uma instalação funcional de bitsandbytes/PyTorch e tente --device cuda."
        ) from exc
    return restored.detach().to("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compara pesos BF16 com os pesos reais reconstruídos de um checkpoint BNB NF4."
    )
    parser.add_argument("--source-model", default=DEFAULT_SOURCE,
                        help="Diretório Safetensors BF16 ou ID do modelo no Hugging Face.")
    parser.add_argument("--quantized-model", default=DEFAULT_QUANTIZED,
                        help="Diretório Safetensors BNB NF4 ou ID do modelo no Hugging Face.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--quantized-revision", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Backend usado somente para desquantizar um tensor por vez.")
    parser.add_argument("--chunk-elements", type=int, default=1_048_576,
                        help="Quantidade aproximada de elementos por bloco para tensores não quantizados.")
    parser.add_argument("--top-k", type=int, default=1000,
                        help="Quantidade de maiores desvios individuais a salvar.")
    parser.add_argument("--thresholds", type=parse_thresholds,
                        default=list(DEFAULT_THRESHOLDS),
                        help="Limites de erro absoluto para contar desvios; separados por vírgula.")
    parser.add_argument("--output-dir", default="bnb_weight_validation")
    args = parser.parse_args()

    if args.chunk_elements < 1:
        parser.error("--chunk-elements precisa ser positivo")
    if args.top_k < 0:
        parser.error("--top-k não pode ser negativo")
    if bnb_F is None:
        parser.error("bitsandbytes não está instalado. Instale com: pip install bitsandbytes")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda foi solicitado, mas torch.cuda.is_available() é False")
    device = torch.device(device_name)

    print(f"[source] Resolvendo {args.source_model}", flush=True)
    source_files = base.resolve_model(args.source_model, args.cache_dir, args.source_revision)
    print(f"[quantized] Resolvendo {args.quantized_model}", flush=True)
    target_files = base.resolve_model(args.quantized_model, args.cache_dir, args.quantized_revision)

    source_index = index_tensors(source_files)
    target_index = index_tensors(target_files)
    quantized_names, auxiliary_names, state_keys_by_weight = inspect_quantized_weights(target_index)
    target_parameter_names = set(target_index) - auxiliary_names

    matched: list[tuple[str, str]] = []
    target_unmatched: list[str] = []
    source_used: set[str] = set()
    for target_name in sorted(target_parameter_names):
        source_name = source_name_for_target(target_name, source_index)
        if source_name is None:
            target_unmatched.append(target_name)
        else:
            matched.append((target_name, source_name))
            source_used.add(source_name)

    source_only = sorted(set(source_index) - source_used)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tensor_csv = output / "tensor_comparison.csv"
    top_csv = output / "top_weight_deviations.csv"
    report_path = output / "report.json"

    if not matched:
        raise RuntimeError(
            "Nenhum tensor foi pareado entre os modelos. Verifique o checkpoint e os nomes."
        )

    overall = new_metrics(args.thresholds)
    quantized_metrics = new_metrics(args.thresholds)
    unquantized_metrics = new_metrics(args.thresholds)
    top_heap: list[tuple[float, int, dict[str, Any]]] = []
    sequence = 0
    tensor_rows: list[dict[str, Any]] = []
    quantized_tensor_count = 0
    unquantized_tensor_count = 0
    shape_mismatches: list[dict[str, Any]] = []
    processed = 0

    with ExitStack() as stack:
        source_handles = open_safetensors(source_files, stack)
        target_handles = open_safetensors(target_files, stack)

        for number, (target_name, source_name) in enumerate(matched, 1):
            target_item = target_index[target_name]
            source_item = source_index[source_name]
            source_shape = tuple(int(v) for v in source_item["shape"])
            quantized = target_name in quantized_names
            target_shape = tuple(int(v) for v in target_item["shape"])

            if not quantized and source_shape != target_shape:
                shape_mismatches.append({
                    "source_tensor": source_name, "target_tensor": target_name,
                    "source_shape": list(source_shape), "target_shape": list(target_shape),
                })
                continue

            print(
                f"[{number}/{len(matched)}] {'NF4' if quantized else 'direto'} "
                f"{target_name} shape={source_shape}",
                flush=True,
            )
            source_handle = source_handles[source_item["file"]]
            target_handle = target_handles[target_item["file"]]
            tensor_metrics = new_metrics(args.thresholds)
            source_dtype = "unknown"
            target_dtype = "unknown"

            dequantized = None
            if quantized:
                dequantized = dequantize_bnb_tensor(
                    target_handle, target_name, state_keys_by_weight[target_name], device
                )
                if dequantized.numel() != math.prod(source_shape):
                    raise RuntimeError(
                        f"{target_name}: desquantização retornou {dequantized.numel()} valores, "
                        f"mas a origem tem {math.prod(source_shape)} ({source_shape})."
                    )
                dequantized = dequantized.reshape(source_shape).contiguous()
                target_dtype = str(target_item["shape"]) + " / BNB NF4"
                quantized_tensor_count += 1
            else:
                unquantized_tensor_count += 1

            for row_start, row_end, flat_start, flat_end in row_chunks(
                source_shape, args.chunk_elements
            ):
                original = source_array(
                    source_handle, source_name, source_shape, row_start, row_end
                )
                if quantized:
                    reconstructed = (
                        dequantized.reshape(-1)[flat_start:flat_end]
                        .to(torch.float32).contiguous().numpy()
                    )
                else:
                    reconstructed = source_array(
                        target_handle, target_name, target_shape, row_start, row_end
                    )
                    target_dtype = target_dtype if target_dtype != "unknown" else "unquantized"
                # Capture the actual source dtype on the first chunk for the report.
                if source_dtype == "unknown":
                    source_dtype = str(
                        source_handle.get_slice(source_name).get_dtype()
                        if source_shape else source_handle.get_tensor(source_name).dtype
                    ).replace("torch.", "")
                add_metrics(tensor_metrics, original, reconstructed, args.thresholds)
                add_metrics(overall, original, reconstructed, args.thresholds)
                add_metrics(
                    quantized_metrics if quantized else unquantized_metrics,
                    original, reconstructed, args.thresholds
                )
                sequence = update_top_weights(
                    top_heap, args.top_k, sequence, target_name, source_shape,
                    flat_start, original, reconstructed,
                )

            tensor_result = finish_metrics(tensor_metrics)
            tensor_rows.append({
                "source_tensor": source_name,
                "quantized_tensor": target_name,
                "source_shape": json.dumps(list(source_shape), separators=(",", ":")),
                "source_dtype": source_dtype,
                "representation": "bnb_nf4_dequantized" if quantized else "unquantized_direct",
                **tensor_result,
            })
            processed += 1
            del dequantized

    tensor_rows.sort(key=lambda item: item["max_abs_error"], reverse=True)
    with tensor_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        fields = [
            "source_tensor", "quantized_tensor", "source_shape", "source_dtype",
            "representation", "numel", "nonzero_error_count", "mae", "rmse",
            "max_abs_error", "source_weight_std", "rmse_over_source_std",
            "cosine_similarity", "threshold_counts",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in tensor_rows:
            row_copy = dict(row)
            row_copy["threshold_counts"] = json.dumps(
                row_copy["threshold_counts"], separators=(",", ":")
            )
            writer.writerow(row_copy)

    top_records = [node[2] for node in sorted(top_heap, key=lambda node: node[0], reverse=True)]
    with top_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        fields = [
            "rank", "tensor", "shape", "flat_index", "coordinates", "source_value",
            "bnb_dequantized_value", "delta_bnb_minus_source", "absolute_error",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, record in enumerate(top_records, 1):
            writer.writerow({
                "rank": rank,
                **record,
                "shape": json.dumps(record["shape"], separators=(",", ":")),
                "coordinates": json.dumps(record["coordinates"], separators=(",", ":")),
            })

    report = {
        "source_model": args.source_model,
        "quantized_model": args.quantized_model,
        "device_used_for_dequantization": str(device),
        "quantization_method": "read stored bitsandbytes NF4 QuantState and dequantize actual checkpoint weights",
        "matched_tensor_count": len(matched),
        "processed_tensor_count": processed,
        "quantized_tensor_count": quantized_tensor_count,
        "unquantized_tensor_count": unquantized_tensor_count,
        "source_only_tensor_count": len(source_only),
        "source_only_tensor_examples": source_only[:100],
        "target_only_tensor_count": len(target_unmatched),
        "target_only_tensor_examples": target_unmatched[:100],
        "shape_mismatch_count": len(shape_mismatches),
        "shape_mismatches": shape_mismatches[:100],
        "all_matched_weights": finish_metrics(overall),
        "bnb_quantized_weights_only": finish_metrics(quantized_metrics),
        "unquantized_weights_only": finish_metrics(unquantized_metrics),
        "tensor_csv": str(tensor_csv),
        "top_deviations_csv": str(top_csv),
        "limitations": [
            "This compares stored BNB NF4 weights after actual dequantization; it does not re-quantize the source model.",
            "It measures weight-level numerical differences, not perplexity or response quality.",
            "The quantized model is text-only. When using the official multimodal source, visual tensors are expected to be source-only.",
            "For the official Qwen3.5-4B source, model.language_model.* names are automatically matched to the text-only model.* names.",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for title, key in [
        ("TODOS OS PESOS PAREADOS", "all_matched_weights"),
        ("SOMENTE PESOS NF4", "bnb_quantized_weights_only"),
        ("TENSORES QUE FICARAM SEM QUANTIZAR", "unquantized_weights_only"),
    ]:
        result = report[key]
        print(f"\n=== {title} ===")
        print(f"Pesos comparados: {result['numel']:,}")
        print(f"Pesos com diferença não zero: {result['nonzero_error_count']:,}")
        print(f"MAE: {result['mae']:.8g}")
        print(f"RMSE: {result['rmse']:.8g}")
        print(f"Erro máximo: {result['max_abs_error']:.8g}")
        print(f"RMSE / std da origem: {result['rmse_over_source_std']:.8g}")
        print(f"Contagens por limite: {result['threshold_counts']}")
    print(f"\nTensores comparados: {processed}/{len(matched)}")
    print(f"Tensores somente na origem: {len(source_only)}")
    print(f"Tensores somente no quantizado: {len(target_unmatched)}")
    print(f"Relatório: {report_path}")
    print(f"Por tensor: {tensor_csv}")
    print(f"Maiores desvios individuais: {top_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
