#!/usr/bin/env python3
"""Brute-force Safetensors weight analysis for numerical grouping experiments.

The script streams tensors in chunks; it does not instantiate the model with
Transformers and does not require a GPU. Compression ratios are estimates,
not a serialized compressed checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_GROUPS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}
FLOAT_DTYPES = {"F16", "BF16", "F32", "F64", "F8_E4M3", "F8_E5M2"}


def parse_groups(raw: str) -> list[int]:
    values = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not values or any(x < 2 for x in values):
        raise argparse.ArgumentTypeError("Use grupos >= 2, por exemplo: 16,32,64,256")
    return values


def resolve_model(model_or_path: str, cache_dir: str | None, revision: str | None) -> Path:
    candidate = Path(model_or_path).expanduser()
    if candidate.exists():
        root = candidate.resolve()
        if not list(root.rglob("*.safetensors")):
            raise FileNotFoundError(f"Nenhum arquivo .safetensors encontrado em: {root}")
        return root
    print(f"\n[download] Modelo: {model_or_path}")
    print("[download] O download pode ocupar vários GB; os pesos não serão carregados todos na RAM.")
    return Path(snapshot_download(
        repo_id=model_or_path, revision=revision, cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json"],
    ))


def read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    """Read the small JSON header only, without materializing tensor data."""
    with path.open("rb") as f:
        first = f.read(8)
        if len(first) != 8:
            raise ValueError(f"Cabeçalho Safetensors inválido: {path}")
        header_len = int.from_bytes(first, "little", signed=False)
        if header_len <= 0 or header_len > 128 * 1024 * 1024:
            raise ValueError(f"Tamanho de cabeçalho suspeito ({header_len}) em {path}")
        header = json.loads(f.read(header_len).decode("utf-8"))
    return {k: v for k, v in header.items() if k != "__metadata__"}


def is_float_dtype(name: str) -> bool:
    return name in FLOAT_DTYPES or name.startswith("F8_")


def allocate_samples(
    tensors: list[dict[str, Any]], budget: int, min_per_tensor: int = 128
) -> dict[tuple[str, str], int]:
    """Stratified sampling: ensure tensor coverage, then allocate remaining budget by size."""
    total = sum(t["numel"] for t in tensors)
    if total == 0:
        return {}
    budget = min(max(1, budget), total)
    sizes = [t["numel"] for t in tensors]

    base = [min(n, max(1, int(min_per_tensor))) for n in sizes]
    if sum(base) > budget:
        # If the requested floor is impossible, revert to proportional sampling.
        raw = [budget * n / total for n in sizes]
        alloc = [min(n, int(math.floor(v))) for n, v in zip(sizes, raw)]
        remain = budget - sum(alloc)
        order = sorted(range(len(tensors)), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
        for i in order:
            if remain <= 0:
                break
            if alloc[i] < sizes[i]:
                alloc[i] += 1
                remain -= 1
        return {(t["file"], t["name"]): n for t, n in zip(tensors, alloc)}

    alloc = base[:]
    remaining = budget - sum(alloc)
    while remaining > 0:
        capacity = [n - a for n, a in zip(sizes, alloc)]
        total_capacity = sum(capacity)
        if total_capacity <= 0:
            break
        raw = [remaining * c / total_capacity for c in capacity]
        additions = [min(capacity[i], int(math.floor(raw[i]))) for i in range(len(alloc))]
        used = sum(additions)
        alloc = [a + b for a, b in zip(alloc, additions)]
        remaining -= used
        if remaining <= 0:
            break
        order = sorted(
            (i for i, cap in enumerate(capacity) if alloc[i] < sizes[i]),
            key=lambda i: raw[i] - math.floor(raw[i]),
            reverse=True,
        )
        if not order:
            break
        for i in order:
            if remaining <= 0:
                break
            if alloc[i] < sizes[i]:
                alloc[i] += 1
                remaining -= 1
        if used == 0 and not order:
            break
    return {(t["file"], t["name"]): n for t, n in zip(tensors, alloc)}


def torch_dtype_to_bits_tensor(block: torch.Tensor) -> np.ndarray | None:
    """Return exact scalar bit patterns for supported 16/32/64-bit tensors."""
    try:
        if block.dtype in (torch.bfloat16, torch.float16):
            return block.contiguous().view(torch.uint16).cpu().numpy().reshape(-1)
        if block.dtype == torch.float32:
            return block.contiguous().view(torch.uint32).cpu().numpy().reshape(-1)
        if block.dtype == torch.float64:
            return block.contiguous().view(torch.uint64).cpu().numpy().reshape(-1)
    except (RuntimeError, TypeError):
        return None
    return None


def chunk_iter(safe_file: Any, name: str, shape: tuple[int, ...], chunk_elements: int):
    if len(shape) == 0:
        yield 0, safe_file.get_tensor(name).reshape(-1)
        return
    slice_obj = safe_file.get_slice(name)
    rows = shape[0]
    row_width = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    rows_per_chunk = max(1, chunk_elements // max(1, row_width))
    flat_offset = 0
    for start in range(0, rows, rows_per_chunk):
        end = min(rows, start + rows_per_chunk)
        slicer = (slice(start, end),) + (slice(None),) * (len(shape) - 1)
        block = slice_obj[slicer].reshape(-1)
        yield flat_offset, block
        flat_offset += block.numel()


def weighted_1d_kmeans(
    sample: np.ndarray,
    k: int,
    sample_weights: np.ndarray | None = None,
    max_iter: int = 30,
) -> tuple[np.ndarray, float, float, float]:
    """Weighted scalar k-means; sample_weights lets stratified samples remain population-weighted."""
    x = np.asarray(sample, dtype=np.float64).reshape(-1)
    if sample_weights is None:
        w = np.ones(x.size, dtype=np.float64)
    else:
        w = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if w.size != x.size:
            raise ValueError("sample_weights deve ter o mesmo tamanho da amostra")
    good = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[good], w[good]
    if x.size == 0:
        raise ValueError("Amostra sem pesos finitos")
    values, inverse = np.unique(x, return_inverse=True)
    counts = np.bincount(inverse, weights=w, minlength=values.size).astype(np.float64)
    k = max(1, min(int(k), values.size))

    if k == 1:
        centers = np.array([np.average(values, weights=counts)], dtype=np.float64)
    else:
        # Start at evenly spaced ranks when weighted quantiles collide.
        cumulative = np.cumsum(counts)
        targets = (np.arange(k, dtype=np.float64) + 0.5) * cumulative[-1] / k
        idx = np.searchsorted(cumulative, targets, side="left")
        centers = np.unique(values[np.minimum(idx, values.size - 1)])
        if centers.size < k:
            ranks = np.linspace(0, values.size - 1, k).round().astype(np.int64)
            centers = values[ranks].copy()

    for _ in range(max_iter):
        centers.sort()
        if centers.size <= 1:
            break
        boundaries = (centers[:-1] + centers[1:]) / 2.0
        assignment = np.searchsorted(boundaries, values, side="right")
        cluster_weight = np.bincount(assignment, weights=counts, minlength=centers.size)
        cluster_sum = np.bincount(assignment, weights=counts * values, minlength=centers.size)
        nonempty = cluster_weight > 0
        updated = (cluster_sum[nonempty] / cluster_weight[nonempty])
        if updated.size == centers.size and np.allclose(updated, centers, rtol=1e-7, atol=1e-12):
            centers = updated
            break
        centers = updated
    centers.sort()
    boundaries = (centers[:-1] + centers[1:]) / 2.0
    assignment = np.searchsorted(boundaries, values, side="right") if centers.size > 1 else np.zeros(values.size, dtype=np.int64)
    errors = values - centers[assignment]
    mse = float(np.average(errors * errors, weights=counts))
    mae = float(np.average(np.abs(errors), weights=counts))
    return centers.astype(np.float32), mse, mae, math.sqrt(max(mse, 0.0))


def cast_codebook(centers: np.ndarray, dtype: str) -> np.ndarray:
    """Round representatives to the requested storage dtype and remove merged centers."""
    values = np.asarray(centers, dtype=np.float32)
    if dtype == "fp16":
        values = values.astype(np.float16).astype(np.float32)
    elif dtype == "bf16":
        values = torch.from_numpy(values.copy()).to(torch.bfloat16).float().numpy()
    # Low-precision rounding can merge two centroids. Do not pay index bits for duplicates.
    return np.unique(values)


def evaluate_codebook(centers: np.ndarray, values: np.ndarray) -> tuple[float, float, float, int]:
    """Return sample MSE, MAE, max-absolute-error, and sample count."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0
    c = np.sort(np.asarray(centers, dtype=np.float64))
    if c.size == 1:
        reconstructed = np.full(x.shape, c[0], dtype=np.float64)
    else:
        boundaries = (c[:-1] + c[1:]) / 2.0
        indices = np.searchsorted(boundaries, x, side="right")
        reconstructed = c[indices]
    error = x - reconstructed
    return (
        float(np.mean(error * error)),
        float(np.mean(np.abs(error))),
        float(np.max(np.abs(error))),
        int(x.size),
    )


def group_key_for_tensor(name: str, scope: str) -> str:
    """Choose one shared codebook globally, per transformer layer, or per tensor."""
    if scope == "global":
        return "global"
    if scope == "tensor":
        return "tensor::" + name
    # Share by layer index when a conventional layer/block naming pattern exists.
    match = re.search(r"(?i)(?:^|\.)(?:layers|blocks|h|layer|block)\.(\d+)(?:\.|$)", name)
    if match:
        return "layer::" + name[:match.end()].rstrip(".")
    # Embeddings, final norms, and other unindexed tensors remain isolated.
    return "tensor::" + name


def parse_scopes(raw: str) -> list[str]:
    values = [x.strip().lower() for x in raw.split(",") if x.strip()]
    allowed = {"global", "layer", "tensor"}
    if not values or any(x not in allowed for x in values):
        raise argparse.ArgumentTypeError("Escopos válidos: global,layer,tensor")
    return list(dict.fromkeys(values))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo ID ou diretório local Safetensors")
    parser.add_argument("--revision", default=None, help="Commit/tag/branch do repositório Hugging Face")
    parser.add_argument("--cache-dir", default=None, help="Diretório de cache do Hugging Face")
    parser.add_argument("--output-dir", default="compressor_results", help="Onde salvar CSVs e JSON")
    parser.add_argument("--sample-size", type=int, default=150_000, help="Amostra proporcional ao número de pesos")
    parser.add_argument("--chunk-elements", type=int, default=1_000_000, help="Elementos máximos por bloco lido")
    parser.add_argument("--groups", type=parse_groups, default=DEFAULT_GROUPS, help="Grupos candidatos separados por vírgula")
    parser.add_argument("--scopes", type=parse_scopes, default=["global", "layer", "tensor"], help="Codebook: global, layer, tensor")
    parser.add_argument("--codebook-dtype", choices=["fp32", "fp16", "bf16"], default="fp32", help="Formato dos representantes, independente dos índices")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Fração da amostra reservada para validação (0.05-0.5)")
    parser.add_argument("--min-samples-per-tensor", type=int, default=128, help="Amostras mínimas por tensor para evitar ignorar tensores pequenos")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.sample_size < 1000:
        parser.error("--sample-size precisa ser >= 1000 para produzir estatísticas úteis")
    if args.chunk_elements < 1:
        parser.error("--chunk-elements precisa ser positivo")
    if not 0.05 <= args.validation_fraction <= 0.5:
        parser.error("--validation-fraction deve ficar entre 0.05 e 0.5")
    if args.min_samples_per_tensor < 1:
        parser.error("--min-samples-per-tensor deve ser positivo")

    start_time = time.time()
    root = resolve_model(args.model, args.cache_dir, args.revision)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"Nenhum .safetensors encontrado em {root}")

    tensors: list[dict[str, Any]] = []
    for file_path in files:
        header = read_safetensors_header(file_path)
        for name, info in header.items():
            dtype = str(info.get("dtype", "UNKNOWN"))
            shape = tuple(int(x) for x in info.get("shape", []))
            numel = int(np.prod(shape, dtype=np.int64)) if shape else 1
            if is_float_dtype(dtype) and numel:
                tensors.append({
                    "file": str(file_path), "name": name, "shape": shape,
                    "shape_text": "x".join(map(str, shape)) if shape else "scalar",
                    "dtype": dtype, "numel": numel,
                    "bytes": numel * DTYPE_BYTES.get(dtype, 2),
                })

    if not tensors:
        raise RuntimeError("Nenhum tensor de ponto flutuante foi encontrado nos arquivos Safetensors.")

    total_params = sum(t["numel"] for t in tensors)
    total_bytes = sum(t["bytes"] for t in tensors)
    sample_counts = allocate_samples(tensors, args.sample_size, args.min_samples_per_tensor)
    rng = np.random.default_rng(args.seed)
    sample_indices: dict[tuple[str, str], np.ndarray] = {}
    for t in tensors:
        count = sample_counts.get((t["file"], t["name"]), 0)
        if count:
            idx = rng.choice(t["numel"], size=count, replace=False)
            idx.sort()
            sample_indices[(t["file"], t["name"])] = idx

    # Exact bit-pattern histograms for 16-bit floating point formats.
    bit_histograms: dict[str, np.ndarray] = {}
    float32_patterns: set[int] | None = set()
    float32_pattern_cap = 5_000_000
    float32_patterns_exact = True
    tensor_stats: list[dict[str, Any]] = []
    tensor_samples: dict[tuple[str, str], np.ndarray] = {}
    sample_parts: list[np.ndarray] = []
    scanned = 0
    started_scan = time.time()

    print(f"\n[scan] Arquivos Safetensors: {len(files)}")
    print(f"[scan] Tensores flutuantes: {len(tensors):,}")
    print(f"[scan] Parâmetros flutuantes analisados: {total_params:,} ({total_bytes / 1e9:.3f} GB brutos estimados)")
    print(f"[scan] Amostra alvo: {min(args.sample_size, total_params):,}; blocos de até {args.chunk_elements:,} elementos")

    for file_index, file_path in enumerate(files, start=1):
        print(f"[scan] ({file_index}/{len(files)}) {file_path.name}")
        with safe_open(str(file_path), framework="pt", device="cpu") as sf:
            keys_here = set(sf.keys())
            these = [t for t in tensors if t["file"] == str(file_path) and t["name"] in keys_here]
            for t in these:
                key = (t["file"], t["name"])
                n = t["numel"]
                total_sum = 0.0
                total_sq = 0.0
                finite_count = 0
                zero_count = 0
                minimum = float("inf")
                maximum = float("-inf")
                nonfinite_count = 0
                sidx = sample_indices.get(key, np.empty(0, dtype=np.int64))
                t_sample_parts: list[np.ndarray] = []

                for flat_offset, block in chunk_iter(sf, t["name"], t["shape"], args.chunk_elements):
                    arr = block.float().cpu().numpy().reshape(-1)
                    scanned += arr.size
                    finite = np.isfinite(arr)
                    if finite.any():
                        vals = arr[finite].astype(np.float64, copy=False)
                        total_sum += float(vals.sum(dtype=np.float64))
                        total_sq += float(np.dot(vals, vals))
                        finite_count += int(vals.size)
                        zero_count += int(np.count_nonzero(vals == 0))
                        minimum = min(minimum, float(vals.min()))
                        maximum = max(maximum, float(vals.max()))
                    nonfinite_count += int(arr.size - finite.sum())

                    original_dtype = str(block.dtype).replace("torch.", "").upper()
                    if original_dtype in ("BFLOAT16", "FLOAT16"):
                        dtype_key = "BF16" if original_dtype == "BFLOAT16" else "F16"
                        if dtype_key not in bit_histograms:
                            bit_histograms[dtype_key] = np.zeros(65536, dtype=np.uint64)
                        hist = bit_histograms[dtype_key]
                        bits = torch_dtype_to_bits_tensor(block)
                        if bits is not None:
                            hist += np.bincount(bits.astype(np.int64, copy=False), minlength=65536).astype(np.uint64)
                    elif original_dtype == "FLOAT32" and float32_patterns_exact:
                        bits = torch_dtype_to_bits_tensor(block)
                        if bits is not None:
                            block_unique = np.unique(bits)
                            if len(float32_patterns) + block_unique.size > float32_pattern_cap:
                                float32_patterns = None
                                float32_patterns_exact = False
                            else:
                                float32_patterns.update(int(x) for x in block_unique)

                    if sidx.size:
                        local_left = int(np.searchsorted(sidx, flat_offset, side="left"))
                        local_right = int(np.searchsorted(sidx, flat_offset + arr.size, side="left"))
                        if local_right > local_left:
                            local = sidx[local_left:local_right] - flat_offset
                            part = arr[local].astype(np.float32, copy=False)
                            part = part[np.isfinite(part)]
                            if part.size:
                                sample_parts.append(part)
                                t_sample_parts.append(part)

                if finite_count:
                    mean = total_sum / finite_count
                    variance = max(0.0, total_sq / finite_count - mean * mean)
                    std = math.sqrt(variance)
                else:
                    mean, std, minimum, maximum = 0.0, 0.0, 0.0, 0.0
                t_sample = np.concatenate(t_sample_parts) if t_sample_parts else np.empty(0, dtype=np.float32)
                tensor_samples[key] = t_sample
                tensor_stats.append({
                    "tensor": t["name"], "file": file_path.name, "dtype": t["dtype"],
                    "shape": t["shape_text"], "numel": n, "estimated_bytes": t["bytes"],
                    "min": minimum, "max": maximum, "mean": mean, "std": std,
                    "zero_fraction": zero_count / finite_count if finite_count else 0.0,
                    "nonfinite_count": nonfinite_count,
                    "sample_count": int(t_sample.size),
                    "sample_unique_count": int(np.unique(t_sample).size) if t_sample.size else 0,
                })

    sample = np.concatenate(sample_parts).astype(np.float32, copy=False) if sample_parts else np.empty(0, dtype=np.float32)
    sample = sample[np.isfinite(sample)]
    if sample.size < 1000:
        raise RuntimeError(f"Amostra válida muito pequena: {sample.size}")

    exact_16bit = {dtype: int(np.count_nonzero(hist)) for dtype, hist in bit_histograms.items()}
    exact_32bit = len(float32_patterns) if float32_patterns_exact and float32_patterns is not None else None

    # Split each tensor's sample independently. This avoids train/validation leakage and
    # permits validation errors to be weighted by the actual parameter count of each tensor.
    rng_split = np.random.default_rng(args.seed + 1)
    train_samples: dict[tuple[str, str], np.ndarray] = {}
    val_samples: dict[tuple[str, str], np.ndarray] = {}
    tensor_by_key = {(t["file"], t["name"]): t for t in tensors}
    for key, values in tensor_samples.items():
        values = np.asarray(values, dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size <= 1:
            train_samples[key] = values
            val_samples[key] = np.empty(0, dtype=np.float32)
            continue
        perm = rng_split.permutation(values.size)
        n_val = max(1, min(values.size - 1, int(round(values.size * args.validation_fraction))))
        val_samples[key] = values[perm[:n_val]]
        train_samples[key] = values[perm[n_val:]]

    # Population-weighted standard deviation from the streamed full-tensor statistics.
    stat_weight = sum(s["numel"] for s in tensor_stats)
    model_mean = sum(s["numel"] * s["mean"] for s in tensor_stats) / max(1, stat_weight)
    model_second = sum(
        s["numel"] * (s["std"] * s["std"] + s["mean"] * s["mean"]) for s in tensor_stats
    ) / max(1, stat_weight)
    model_weight_std = math.sqrt(max(0.0, model_second - model_mean * model_mean))

    # Weight each sampled value by tensor_population / tensor_sample_count so that
    # forced coverage of small tensors does not distort the learned global distribution.
    groups_rows: list[dict[str, Any]] = []
    for scope in args.scopes:
        group_train_values: dict[str, list[np.ndarray]] = defaultdict(list)
        group_train_weights: dict[str, list[np.ndarray]] = defaultdict(list)
        tensor_group: dict[tuple[str, str], str] = {}
        for t in tensors:
            key = (t["file"], t["name"])
            gkey = group_key_for_tensor(t["name"], scope)
            tensor_group[key] = gkey
            values = train_samples.get(key, np.empty(0, dtype=np.float32))
            if values.size == 0:
                continue
            per_value_weight = float(t["numel"]) / float(values.size)
            group_train_values[gkey].append(values)
            group_train_weights[gkey].append(np.full(values.size, per_value_weight, dtype=np.float64))

        for requested_k in args.groups:
            centers_by_group: dict[str, np.ndarray] = {}
            requested_capped_groups = 0
            for gkey, parts in group_train_values.items():
                x = np.concatenate(parts).astype(np.float32, copy=False)
                weights = np.concatenate(group_train_weights[gkey]).astype(np.float64, copy=False)
                unique_count = int(np.unique(x).size)
                if unique_count < requested_k:
                    requested_capped_groups += 1
                centers, _, _, _ = weighted_1d_kmeans(x, requested_k, weights)
                centers_by_group[gkey] = cast_codebook(centers, args.codebook_dtype)

            squared_error_weighted = 0.0
            absolute_error_weighted = 0.0
            max_absolute_error = 0.0
            val_sample_count = 0
            val_parameter_coverage = 0
            train_squared_error_weighted = 0.0
            train_parameter_coverage = 0

            for t in tensors:
                key = (t["file"], t["name"])
                gkey = tensor_group[key]
                centers = centers_by_group.get(gkey)
                if centers is None:
                    continue
                numel = int(t["numel"])
                v = val_samples.get(key, np.empty(0, dtype=np.float32))
                vmse, vmae, vmax, vn = evaluate_codebook(centers, v)
                if vn:
                    squared_error_weighted += vmse * numel
                    absolute_error_weighted += vmae * numel
                    max_absolute_error = max(max_absolute_error, vmax)
                    val_sample_count += vn
                    val_parameter_coverage += numel
                tr = train_samples.get(key, np.empty(0, dtype=np.float32))
                tmse, _, _, tn = evaluate_codebook(centers, tr)
                if tn:
                    train_squared_error_weighted += tmse * numel
                    train_parameter_coverage += numel

            val_rmse = math.sqrt(squared_error_weighted / val_parameter_coverage) if val_parameter_coverage else 0.0
            val_mae = absolute_error_weighted / val_parameter_coverage if val_parameter_coverage else 0.0
            train_rmse = math.sqrt(train_squared_error_weighted / train_parameter_coverage) if train_parameter_coverage else 0.0
            rel_rmse = val_rmse / model_weight_std if model_weight_std > 0 else 0.0

            codebook_bytes_per_value = {"fp32": 4, "fp16": 2, "bf16": 2}[args.codebook_dtype]
            index_bytes = 0
            weighted_index_bits = 0.0
            min_index_bits = None
            max_index_bits = 0
            indexed_parameters = 0
            for t in tensors:
                key = (t["file"], t["name"])
                centers = centers_by_group.get(tensor_group[key])
                if centers is None:
                    continue
                n = int(t["numel"])
                k_actual = int(centers.size)
                bits = int(math.ceil(math.log2(k_actual))) if k_actual > 1 else 0
                index_bytes += (n * bits + 7) // 8
                weighted_index_bits += n * bits
                indexed_parameters += n
                min_index_bits = bits if min_index_bits is None else min(min_index_bits, bits)
                max_index_bits = max(max_index_bits, bits)
            codebook_values = sum(int(c.size) for c in centers_by_group.values())
            codebook_bytes = codebook_values * codebook_bytes_per_value
            estimated_bytes = index_bytes + codebook_bytes
            estimated_ratio = estimated_bytes / total_bytes if total_bytes else 0.0
            actual_groups_total = codebook_values
            groups_rows.append({
                "scope": scope,
                "requested_groups_per_codebook": requested_k,
                "codebook_count": len(centers_by_group),
                "actual_groups_total_across_codebooks": actual_groups_total,
                "groups_capped_by_sample_uniques": requested_capped_groups,
                "codebook_dtype": args.codebook_dtype,
                "index_bits_min": min_index_bits if min_index_bits is not None else 0,
                "index_bits_max": max_index_bits,
                "weighted_mean_index_bits_per_parameter": weighted_index_bits / indexed_parameters if indexed_parameters else 0.0,
                "train_rmse_population_weighted": train_rmse,
                "validation_rmse_population_weighted": val_rmse,
                "validation_mae_population_weighted": val_mae,
                "validation_rmse_over_full_model_weight_std": rel_rmse,
                "validation_max_abs_error_sample": max_absolute_error,
                "validation_sample_values": val_sample_count,
                "validation_parameter_coverage": val_parameter_coverage,
                "validation_parameter_coverage_pct": 100 * val_parameter_coverage / total_params if total_params else 0.0,
                "estimated_index_MB_ideal_packed": index_bytes / 1_000_000,
                "estimated_codebook_MB": codebook_bytes / 1_000_000,
                "estimated_total_MB_ideal_packed": estimated_bytes / 1_000_000,
                "estimated_size_ratio_vs_source": estimated_ratio,
                "estimated_savings_pct_vs_source": (1 - estimated_ratio) * 100,
                "note": "Estimativa ideal; codebooks compartilhados + mapa de índices. Não é ainda um checkpoint serializado.",
            })

    tensor_stats.sort(key=lambda x: x["estimated_bytes"], reverse=True)
    write_csv(output_dir / "tensor_stats.csv", tensor_stats, [
        "tensor", "file", "dtype", "shape", "numel", "estimated_bytes", "min", "max",
        "mean", "std", "zero_fraction", "nonfinite_count", "sample_count", "sample_unique_count",
    ])
    write_csv(output_dir / "group_analysis.csv", groups_rows, list(groups_rows[0].keys()))

    summary = {
        "model_or_path": args.model,
        "resolved_model_path": str(root),
        "safetensors_files": [p.name for p in files],
        "floating_tensor_count": len(tensors),
        "floating_parameter_count": total_params,
        "estimated_source_bytes": total_bytes,
        "estimated_source_decimal_GB": total_bytes / 1e9,
        "sample_target": min(args.sample_size, total_params),
        "valid_sample_count": int(sample.size),
        "sample_seed": args.seed,
        "sample_mean": float(np.mean(sample, dtype=np.float64)),
        "sample_std": float(np.std(sample, dtype=np.float64)),
        "full_model_weighted_mean": model_mean,
        "full_model_weighted_std": model_weight_std,
        "sample_min": float(np.min(sample)),
        "sample_max": float(np.max(sample)),
        "analysis_scopes": args.scopes,
        "codebook_dtype": args.codebook_dtype,
        "validation_fraction": args.validation_fraction,
        "min_samples_per_tensor": args.min_samples_per_tensor,
        "exact_unique_bit_patterns_16bit": exact_16bit,
        "exact_unique_float32_bit_patterns": exact_32bit,
        "float32_unique_count_is_exact": bool(float32_patterns_exact),
        "group_analysis": groups_rows,
        "top_20_tensors_by_size": tensor_stats[:20],
        "elapsed_seconds": round(time.time() - start_time, 2),
        "scan_seconds": round(time.time() - started_scan, 2),
        "caveats": [
            "A contagem de padrões binários é exata por formato (BF16/F16/F32), mas padrões de formatos distintos são contados separadamente.",
            "Os codebooks são ajustados na amostra de treino e avaliados em uma amostra de validação separada por tensor.",
            "O erro de validação é ponderado pelo número real de parâmetros de cada tensor; as amostras mínimas por tensor não dominam artificialmente a métrica.",
            "Escopo global, por camada e por tensor têm custos diferentes: quanto mais codebooks, mais representantes/metadados são necessários.",
            "O mapa de índices continua necessário para reconstruir qual representante corresponde a cada posição; estimativas assumem índices idealmente empacotados.",
            "O tamanho estimado não inclui cabeçalhos, alinhamento, checksum, descritores e detalhes do formato final, e não é um checkpoint serializado.",
            "Representantes podem ser FP32, FP16 ou BF16, independentemente do número de bits usados nos índices.",
            "A RMSE dos pesos não demonstra preservação de qualidade linguística; será necessário reconstruir matrizes e avaliar logits/perplexidade/respostas.",
            "Este programa ainda não grava checkpoint quantizado e não mede a sobrecarga de inferência de lookup/reconstrução.",
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== RESULTADO BRUTO ===")
    print(f"Parâmetros flutuantes examinados : {total_params:,}")
    print(f"Tamanho bruto estimado          : {total_bytes / 1_000_000:.2f} MB ({total_bytes / 1e9:.3f} GB)")
    print(f"Amostra efetiva                 : {sample.size:,} valores")
    print(f"Valores distintos exatos       : {exact_16bit}")
    if exact_32bit is not None:
        print(f"Padrões F32 distintos exatos    : {exact_32bit:,}")
    else:
        print("Padrões F32 distintos exatos    : limite atingido; veja o JSON")
    print("\nEscopo  | Grupos/codebook | Codebooks | RMSE validação | RMSE/desvio-padrão | Tamanho ideal | Economia")
    for row in groups_rows:
        print(f"{row['scope']:<7} | {row['requested_groups_per_codebook']:>14} | {row['codebook_count']:>9} | "
              f"{row['validation_rmse_population_weighted']:.6g} | "
              f"{row['validation_rmse_over_full_model_weight_std']:.6g} | "
              f"{row['estimated_total_MB_ideal_packed']:.2f} MB | "
              f"{row['estimated_savings_pct_vs_source']:.2f}%")
    print(f"\nArquivos gravados em: {output_dir.resolve()}")
    print("  - summary.json      (resumo completo e ressalvas)")
    print("  - group_analysis.csv (trade-off entre grupos, erro e tamanho estimado)")
    print("  - tensor_stats.csv   (estatísticas por tensor/camada)")
    print("\nNota: esta fase mede o potencial de representantes compartilhados. Ainda não produz um checkpoint reconstruído.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERRO: {exc}", file=sys.stderr)
        raise
