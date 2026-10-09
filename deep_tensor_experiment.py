#!/usr/bin/env python3
"""Deep comparison of scalar quantization, randomized low-rank and hybrid codecs.

Every candidate is written to a real NPZ, reloaded, decoded, and measured against
one source tensor. This is a research harness, not a Transformers checkpoint codec.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

import compress_tensor as base


def parse_int_list(value: str) -> list[int]:
    result = sorted(set(int(part.strip()) for part in value.split(",") if part.strip()))
    if not result or any(x < 1 for x in result):
        raise argparse.ArgumentTypeError("Informe uma lista de inteiros positivos separados por vírgulas.")
    return result


def parse_float_list(value: str) -> list[float]:
    result = sorted(set(float(part.strip()) for part in value.split(",") if part.strip()))
    if not result or any(not math.isfinite(x) or x <= 0 for x in result):
        raise argparse.ArgumentTypeError("Informe uma lista de números positivos separados por vírgulas.")
    return result


def randomized_svd(
    matrix: np.ndarray,
    max_rank: int,
    oversample: int = 32,
    power_iterations: int = 2,
    seed: int = 12345,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate the leading singular triplets without materializing a full SVD."""
    a = np.asarray(matrix, dtype=np.float32)
    m, n = a.shape
    width = min(min(m, n), max_rank + max(0, oversample))
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n, width), dtype=np.float32)
    q, _ = np.linalg.qr(a @ omega, mode="reduced")
    for _ in range(max(0, power_iterations)):
        z, _ = np.linalg.qr(a.T @ q, mode="reduced")
        q, _ = np.linalg.qr(a @ z, mode="reduced")
    small = q.T @ a
    ub, singular_values, vh = np.linalg.svd(small, full_matrices=False)
    u = q @ ub[:, :max_rank]
    return u.astype(np.float32), singular_values[:max_rank].astype(np.float32), vh[:max_rank].astype(np.float32)


def balanced_factors(
    u: np.ndarray, singular_values: np.ndarray, vh: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray]:
    """Store balanced U*sqrt(S) and V*sqrt(S) factors in FP16."""
    scale = np.sqrt(np.maximum(singular_values[:rank], 0.0)).astype(np.float32)
    left = (u[:, :rank] * scale[None, :]).astype(np.float16)
    right = (vh[:rank, :].T * scale[None, :]).astype(np.float16)
    return left, right


def make_uniform_indices(x: np.ndarray, levels: int) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(x, dtype=np.float32).reshape(-1)
    lo = np.float32(values.min())
    hi = np.float32(values.max())
    step = np.float32((float(hi) - float(lo)) / (levels - 1)) if hi > lo else np.float32(0.0)
    dtype = np.uint8 if levels <= 256 else np.uint16
    if step == 0:
        indices = np.zeros(values.size, dtype=dtype)
    else:
        indices = np.rint((values - lo) / step).clip(0, levels - 1).astype(dtype)
    return indices, {"minimum": float(lo), "step": float(step), "levels": int(levels)}


def make_log_indices(
    x: np.ndarray, levels: int, scale_multiplier: float
) -> tuple[np.ndarray, dict[str, float]]:
    """Signed logarithmic companding with independently tunable characteristic scale."""
    values = np.asarray(x, dtype=np.float32).reshape(-1)
    absolute = np.abs(values)
    median = float(np.median(absolute))
    nonzero = absolute[absolute > 0]
    if median <= float(np.finfo(np.float32).tiny):
        median = float(np.median(nonzero)) if nonzero.size else 1.0
    scale = np.float32(max(median * scale_multiplier, float(np.finfo(np.float32).tiny)))
    transformed = np.sign(values) * np.log1p(absolute / scale)
    zmax = np.float32(np.max(np.abs(transformed))) if transformed.size else np.float32(0.0)
    dtype = np.uint8 if levels <= 256 else np.uint16
    if zmax == 0:
        indices = np.zeros(values.size, dtype=dtype)
    else:
        encoded = (transformed / zmax + 1.0) * (0.5 * (levels - 1))
        indices = np.rint(encoded).clip(0, levels - 1).astype(dtype)
    return indices, {
        "scale": float(scale),
        "zmax": float(zmax),
        "levels": int(levels),
        "scale_multiplier": float(scale_multiplier),
    }


def pack_indices(indices: np.ndarray, actual_levels: int) -> tuple[np.ndarray, int]:
    bits = int(math.ceil(math.log2(actual_levels))) if actual_levels > 1 else 0
    return base.pack_indices_bitplanes(indices, bits), bits


def decode_from_archive(path: Path, method: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        shape = tuple(meta["shape"])
        numel = int(meta["numel"])
        kind = meta["kind"]
        if kind in ("kmeans", "uniform", "log", "hybrid"):
            if meta.get("index_encoding") == "zlib_symbols":
                raw = zlib.decompress(archive["zlib_indices"].astype(np.uint8, copy=False).tobytes())
                index_dtype = np.dtype(meta["index_storage_dtype"])
                index = np.frombuffer(raw, dtype=index_dtype, count=numel).astype(np.uint32)
                if index.size != numel:
                    raise ValueError(f"Mapa zlib inválido em {method}: quantidade de índices divergente")
            else:
                packed = archive["packed_indices"].astype(np.uint8, copy=False)
                index = base.unpack_indices_bitplanes(packed, numel, int(meta["index_bits"]))
            if kind == "kmeans":
                centers = archive["codebook"].astype(np.float32)
                return centers[index.astype(np.int64)]
            if kind == "uniform":
                params = archive["formula_parameters"].astype(np.float32)
                return (params[0] + index.astype(np.float32) * params[1]).reshape(shape)
            if kind == "log":
                params = archive["formula_parameters"].astype(np.float32)
                levels = int(meta["levels"])
                if params[1] == 0:
                    return np.zeros(numel, dtype=np.float32).reshape(shape)
                z = (index.astype(np.float32) / (levels - 1) * 2.0 - 1.0) * params[1]
                decoded = np.sign(z) * params[0] * np.expm1(np.abs(z))
                return decoded.astype(np.float32).reshape(shape)
            centers = archive["residual_codebook"].astype(np.float32)
            residual = centers[index.astype(np.int64)]
            left = archive["left"].astype(np.float32)
            right = archive["right"].astype(np.float32)
            return (left @ right.T + residual.reshape(shape)).astype(np.float32)
        if kind == "lowrank":
            left = archive["left"].astype(np.float32)
            right = archive["right"].astype(np.float32)
            return (left @ right.T).astype(np.float32)
    raise ValueError(f"Tipo de arquivo não suportado para {method}: {kind}")


def reconstruction_metrics(
    original: np.ndarray, rebuilt: np.ndarray, chunk_size: int = 1_000_000
) -> dict[str, float]:
    """Chunked metrics avoid several full-tensor float64 copies."""
    x = np.asarray(original, dtype=np.float32).reshape(-1)
    y = np.asarray(rebuilt, dtype=np.float32).reshape(-1)
    if x.size != y.size:
        raise ValueError("Tamanhos divergentes na reconstrução")
    squared_error = absolute_error = x2 = y2 = dot = 0.0
    max_abs = 0.0
    for start in range(0, x.size, chunk_size):
        end = min(x.size, start + chunk_size)
        a = x[start:end].astype(np.float64)
        b = y[start:end].astype(np.float64)
        diff = a - b
        squared_error += float(np.dot(diff, diff))
        absolute_error += float(np.abs(diff).sum())
        if diff.size:
            max_abs = max(max_abs, float(np.max(np.abs(diff))))
        x2 += float(np.dot(a, a))
        y2 += float(np.dot(b, b))
        dot += float(np.dot(a, b))
    mse = squared_error / max(1, x.size)
    rmse = math.sqrt(max(0.0, mse))
    std = float(np.std(x, dtype=np.float64))
    cosine = dot / math.sqrt(x2 * y2) if x2 > 0 and y2 > 0 else (1.0 if x2 == y2 else 0.0)
    return {
        "rmse": rmse,
        "mae": absolute_error / max(1, x.size),
        "max_abs_error": max_abs,
        "weight_std": std,
        "rmse_over_weight_std": rmse / std if std > 0 else 0.0,
        "cosine_similarity_flat_weights": cosine,
    }


def projection_probe(
    original: np.ndarray,
    rebuilt: np.ndarray,
    shape: tuple[int, int],
    batch: int,
    seed: int,
) -> dict[str, Any] | None:
    if batch < 1:
        return None
    w = original.reshape(shape)
    wr = rebuilt.reshape(shape)
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((shape[1], batch), dtype=np.float32)
    inputs /= math.sqrt(max(1, shape[1]))
    y = w @ inputs
    yr = wr @ inputs
    yd, yrd = y.astype(np.float64), yr.astype(np.float64)
    diff = yd - yrd
    error = float(np.sqrt(np.mean(diff * diff)))
    rms = float(np.sqrt(np.mean(yd * yd)))
    yn, yrn = float(np.linalg.norm(yd)), float(np.linalg.norm(yrd))
    cosine = float(np.sum(yd * yrd) / (yn * yrn)) if yn and yrn else 0.0
    return {
        "batch_size": int(batch),
        "output_rmse": error,
        "output_rmse_over_original_rms": error / rms if rms else 0.0,
        "output_cosine_similarity": cosine,
        "note": "Teste W@X com entrada gaussiana sintética; não substitui logits ou ativações reais.",
    }


def factorized_projection_probe(
    archive_path: Path,
    original: np.ndarray,
    shape: tuple[int, int],
    batch: int,
    seed: int,
    repeats: int = 5,
) -> dict[str, Any] | None:
    """Benchmark direct U(V^T X) without rebuilding W, using NumPy FP32 on CPU."""
    if batch < 1:
        return None
    with np.load(archive_path, allow_pickle=False) as archive:
        if "left" not in archive or "right" not in archive:
            return None
        left = archive["left"].astype(np.float32)
        right = archive["right"].astype(np.float32)
    matrix = np.asarray(original, dtype=np.float32).reshape(shape)
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((shape[1], batch), dtype=np.float32)
    inputs /= math.sqrt(max(1, shape[1]))
    reference = matrix @ inputs

    def dense_op() -> np.ndarray:
        return matrix @ inputs

    def factored_op() -> np.ndarray:
        return left @ (right.T @ inputs)

    # Warm up BLAS and separate timing from first-call initialization.
    for _ in range(2):
        dense_op()
        factored_op()
    dense_times: list[float] = []
    factor_times: list[float] = []
    factored_output = np.empty_like(reference)
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        dense_op()
        dense_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        factored_output = factored_op()
        factor_times.append(time.perf_counter() - t0)

    diff = reference.astype(np.float64) - factored_output.astype(np.float64)
    error = float(np.sqrt(np.mean(diff * diff)))
    rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    dense_ms = float(np.median(dense_times) * 1000.0)
    factored_ms = float(np.median(factor_times) * 1000.0)
    return {
        "backend": "NumPy FP32 CPU; timing is relative and not a model tokens/second benchmark",
        "batch_size": int(batch),
        "rank": int(left.shape[1]),
        "dense_W_times_X_median_ms": dense_ms,
        "direct_U_times_VTX_median_ms": factored_ms,
        "dense_over_factorized_speed_ratio": dense_ms / factored_ms if factored_ms > 0 else None,
        "output_rmse_over_original_rms": error / rms if rms else 0.0,
        "output_cosine_similarity": float(
            np.sum(reference.astype(np.float64) * factored_output.astype(np.float64))
            / max(1e-30, np.linalg.norm(reference.astype(np.float64)) * np.linalg.norm(factored_output.astype(np.float64)))
        ),
        "timing_repeats": int(max(1, repeats)),
        "note": "O caminho fatorado calcula U(V^T X) diretamente: não materializa nem recalcula a matriz W completa por token.",
    }


def save_and_measure(
    outdir: Path,
    name: str,
    kind: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    original: np.ndarray,
    source_bytes: int,
    shape: tuple[int, int],
    projection_batch: int,
    seed: int,
    timing_repeats: int = 5,
    map_codec: str = "auto",
) -> dict[str, Any]:
    """Write actual archives with a shared set of index codecs and select the smallest."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    path = outdir / f"{safe}.npz"
    base_meta = dict(metadata)
    base_meta.update({"method": name, "kind": kind, "shape": list(shape), "numel": int(original.size)})
    arrays = dict(arrays)
    raw_indices = arrays.pop("raw_indices", None)
    arrays.pop("packed_indices", None)
    candidate_sizes: dict[str, int] = {}
    candidate_paths: dict[str, Path] = {}
    candidate_codecs = (
        ["zip_deflate_bitplanes", "zlib_symbols", "raw_bitplanes"]
        if map_codec == "auto" else [map_codec]
    )

    def metadata_bytes(meta: dict[str, Any]) -> np.ndarray:
        return np.frombuffer(
            json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            dtype=np.uint8,
        )

    if raw_indices is None:
        # Low-rank-only candidates have no index map; only normal NPZ/Deflate applies.
        base_meta["map_codec"] = "npz_deflate_no_index_map"
        full = dict(arrays)
        full["metadata"] = metadata_bytes(base_meta)
        np.savez_compressed(path, **full)
        candidate_sizes["npz_deflate_no_index_map"] = path.stat().st_size
        selected_codec = "npz_deflate_no_index_map"
        selected_meta = base_meta
    else:
        raw_indices = np.asarray(raw_indices).reshape(-1)
        bits = int(base_meta.get("index_bits", 0))
        packed = base.pack_indices_bitplanes(raw_indices, bits)
        max_index = int(raw_indices.max()) if raw_indices.size else 0
        symbol_dtype = np.dtype("<u1" if max_index <= 255 else "<u2" if max_index <= 65535 else "<u4")
        symbol_bytes = raw_indices.astype(symbol_dtype, copy=False).tobytes()

        for codec in candidate_codecs:
            candidate_path = outdir / f".{safe}.{codec}.candidate.npz"
            candidate_meta = dict(base_meta)
            candidate_arrays = dict(arrays)
            if codec in ("zip_deflate_bitplanes", "raw_bitplanes"):
                candidate_meta["index_encoding"] = "bitplanes"
                candidate_meta["index_storage_dtype"] = "packed_bits"
                candidate_meta["map_codec"] = codec
                candidate_arrays["packed_indices"] = packed
                candidate_arrays["metadata"] = metadata_bytes(candidate_meta)
                if codec == "zip_deflate_bitplanes":
                    np.savez_compressed(candidate_path, **candidate_arrays)
                else:
                    np.savez(candidate_path, **candidate_arrays)
            elif codec == "zlib_symbols":
                compressed_symbols = zlib.compress(symbol_bytes, level=9)
                candidate_meta["index_encoding"] = "zlib_symbols"
                candidate_meta["index_storage_dtype"] = symbol_dtype.str
                candidate_meta["map_codec"] = codec
                candidate_arrays["zlib_indices"] = np.frombuffer(compressed_symbols, dtype=np.uint8)
                candidate_arrays["metadata"] = metadata_bytes(candidate_meta)
                np.savez(candidate_path, **candidate_arrays)
            else:
                raise ValueError(f"Codificador de índices desconhecido: {codec}")
            candidate_paths[codec] = candidate_path
            candidate_sizes[codec] = candidate_path.stat().st_size

        selected_codec = min(candidate_sizes, key=candidate_sizes.get)
        candidate_paths[selected_codec].replace(path)
        selected_meta = dict(base_meta)
        selected_meta["map_codec"] = selected_codec
        for temp_path in candidate_paths.values():
            if temp_path.exists():
                temp_path.unlink()

    file_bytes = path.stat().st_size
    # Decode from the actual bytes on disk so metrics include storage dtype rounding.
    decode_started = time.perf_counter()
    rebuilt = decode_from_archive(path, name)
    decode_seconds = time.perf_counter() - decode_started
    metrics = reconstruction_metrics(original, rebuilt)
    projection = projection_probe(original, rebuilt, shape, projection_batch, seed)
    direct_factorized = (
        factorized_projection_probe(path, original, shape, projection_batch, seed, timing_repeats)
        if kind == "lowrank" else None
    )
    result = {
        "method": name,
        "kind": kind,
        "artifact_path": str(path.resolve()),
        "artifact_bytes_actual": int(file_bytes),
        "artifact_MB_actual": file_bytes / 1_000_000,
        "source_tensor_bytes": int(source_bytes),
        "source_tensor_MB": source_bytes / 1_000_000,
        "savings_pct_vs_source": (1.0 - file_bytes / source_bytes) * 100.0 if source_bytes else 0.0,
        "selected_map_codec": selected_codec,
        "candidate_codec_sizes_bytes": candidate_sizes,
        "full_tensor_error": metrics,
        "linear_projection_probe": projection,
        "archive_decode_seconds_offline": decode_seconds,
        "direct_factorized_projection_benchmark": direct_factorized,
        "metadata": {k: v for k, v in selected_meta.items() if k not in ("shape", "numel", "kind", "method")},
    }
    (outdir / f"{safe}_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def load_tensor(args: argparse.Namespace) -> tuple[dict[str, Any], str, int, np.ndarray]:
    files = base.resolve_model(args.model, args.cache_dir, args.revision)
    if args.list_tensors:
        for item in base.list_tensors(files)[:100]:
            print(f"{item['name']}\tshape={item['shape']}\tnumel={item['numel']}\tfile={item['file'].name}")
        raise SystemExit(0)
    chosen = base.choose_tensor(files, args.tensor_name, args.max_auto_elements)
    if len(chosen["shape"]) != 2:
        raise ValueError(f"O teste exige uma matriz 2D; recebido {chosen['name']} {chosen['shape']}")
    with safe_open(str(chosen["file"]), framework="pt", device="cpu") as sf:
        tensor = sf.get_tensor(chosen["name"])
    source_dtype = str(tensor.dtype).replace("torch.", "")
    source_bytes = int(tensor.numel() * tensor.element_size())
    x = tensor.float().cpu().numpy().reshape(-1).copy()
    del tensor
    if not np.isfinite(x).all():
        raise ValueError("Tensor contém NaN ou infinito; interrompendo para evitar métricas inválidas.")
    return chosen, source_dtype, source_bytes, x


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=base.DEFAULT_MODEL, help="Repo HF, diretório local ou Safetensors")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--tensor-name", default=None)
    parser.add_argument("--max-auto-elements", type=int, default=20_000_000)
    parser.add_argument("--list-tensors", action="store_true")
    parser.add_argument("--output-dir", default="deep_tensor_results")
    parser.add_argument("--kmeans-samples", type=parse_int_list, default=parse_int_list("250000,1000000,2000000"))
    parser.add_argument("--kmeans-groups", type=parse_int_list, default=parse_int_list("64,256"))
    parser.add_argument("--residual-sample-size", type=int, default=500_000, help="Amostra para treinar codebooks dos resíduos híbridos")
    parser.add_argument("--log-levels", type=parse_int_list, default=parse_int_list("128,256"))
    parser.add_argument("--log-scales", type=parse_float_list, default=parse_float_list("0.25,0.5,0.75,1"))
    parser.add_argument("--ranks", type=parse_int_list, default=parse_int_list("64,128,256"))
    parser.add_argument("--hybrid-ranks", type=parse_int_list, default=parse_int_list("64,128"))
    parser.add_argument("--residual-groups", type=parse_int_list, default=parse_int_list("16,32,64"))
    parser.add_argument("--oversample", type=int, default=32)
    parser.add_argument("--power-iterations", type=int, default=2)
    parser.add_argument("--projection-batch", type=int, default=4)
    parser.add_argument("--map-codec", choices=["auto", "zip_deflate_bitplanes", "zlib_symbols", "raw_bitplanes"], default="auto", help="Auto mede o tamanho real de três codificadores compartilhados entre K-means, log e híbridos")
    parser.add_argument("--timing-repeats", type=int, default=5, help="Repetições para benchmark NumPy da projeção fatorada")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for values, label in (
        (args.kmeans_groups, "kmeans-groups"),
        (args.log_levels, "log-levels"),
        (args.residual_groups, "residual-groups"),
    ):
        if any(v < 2 or v > 65536 for v in values):
            parser.error(f"--{label} aceita valores entre 2 e 65536")
    if args.oversample < 0 or args.power_iterations < 0:
        parser.error("--oversample e --power-iterations não podem ser negativos")

    started = time.time()
    chosen, source_dtype, source_bytes, x = load_tensor(args)
    shape = tuple(int(v) for v in chosen["shape"])
    matrix = x.reshape(shape)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[tensor] {chosen['name']} shape={shape} dtype={source_dtype} numel={x.size}")
    print(f"[source] {source_bytes / 1_000_000:.3f} MB; preparando experimentos ...")

    rng = np.random.default_rng(args.seed)
    sample_counts = sorted(set(min(sample_count, x.size) for sample_count in args.kmeans_samples))
    max_sample = max(sample_counts)
    selected_positions = rng.choice(x.size, size=max_sample, replace=False)
    results: list[dict[str, Any]] = []

    # A. K-means with increasing sample sizes, using identical quantizer settings otherwise.
    for sample_count in sample_counts:
        # Prevent duplicate runs on small test fixtures after clamping sample count.
        for groups in args.kmeans_groups:
            name = f"kmeans_n{sample_count}_g{groups}"
            sample = x[selected_positions[:sample_count]]
            centers = np.sort(base.weighted_kmeans_1d(sample, groups))
            indices = base.assign_indices(x, centers, chunk_size=2_000_000)
            bits = int(math.ceil(math.log2(int(centers.size)))) if centers.size > 1 else 0
            results.append(save_and_measure(
                outdir, name, "kmeans",
                {"codebook": centers.astype(np.float32), "raw_indices": indices},
                {"requested_sample_count": int(sample_count), "requested_groups": int(groups),
                 "actual_groups": int(centers.size), "index_bits": bits,
                 "training_seed": int(args.seed)},
                x, source_bytes, shape, args.projection_batch, args.seed, args.timing_repeats, args.map_codec,
            ))

    # B. Formula-generated signed log codebook across levels and characteristic scales.
    for levels in args.log_levels:
        for scale_multiplier in args.log_scales:
            name = f"log_levels{levels}_scale{scale_multiplier:g}"
            indices, params = make_log_indices(x, levels, scale_multiplier)
            bits = int(math.ceil(math.log2(levels))) if levels > 1 else 0
            params_array = np.array([params["scale"], params["zmax"]], dtype=np.float32)
            results.append(save_and_measure(
                outdir, name, "log",
                {"raw_indices": indices, "formula_parameters": params_array},
                {"levels": int(levels), "index_bits": bits, **params},
                x, source_bytes, shape, args.projection_batch, args.seed, args.timing_repeats, args.map_codec,
            ))

    # C. One randomized SVD basis, reused at all requested ranks.
    requested_ranks = sorted(set(args.ranks + args.hybrid_ranks))
    max_rank = min(max(requested_ranks), min(shape))
    u, singular_values, vh = randomized_svd(
        matrix, max_rank, oversample=args.oversample,
        power_iterations=args.power_iterations, seed=args.seed,
    )
    print(f"[low-rank] randomized SVD ready through rank {max_rank}")

    factor_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for rank in requested_ranks:
        rank = min(rank, max_rank)
        if rank in factor_cache:
            continue
        left, right = balanced_factors(u, singular_values, vh, rank)
        lowrank = left.astype(np.float32) @ right.astype(np.float32).T
        factor_cache[rank] = (left, right, lowrank)

    for rank in args.ranks:
        rank = min(rank, max_rank)
        left, right, _ = factor_cache[rank]
        name = f"lowrank_r{rank}_fp16"
        results.append(save_and_measure(
            outdir, name, "lowrank",
            {"left": left, "right": right},
            {"rank": int(rank), "factor_dtype": "fp16", "oversample": int(args.oversample),
             "power_iterations": int(args.power_iterations),
             "factor_parameter_bytes": int(left.nbytes + right.nbytes)},
            x, source_bytes, shape, args.projection_batch, args.seed, args.timing_repeats, args.map_codec,
        ))

    # D. Hybrid: low-rank structure plus quantized residual, retaining no per-weight FP16 values.
    for rank in args.hybrid_ranks:
        rank = min(rank, max_rank)
        left, right, lowrank = factor_cache[rank]
        residual = (matrix - lowrank).reshape(-1)
        residual_sample_count = min(max(1, args.residual_sample_size), max_sample, x.size)
        residual_sample = residual[selected_positions[:residual_sample_count]]
        for groups in args.residual_groups:
            name = f"hybrid_r{rank}_residual_g{groups}"
            centers = np.sort(base.weighted_kmeans_1d(residual_sample, groups))
            indices = base.assign_indices(residual, centers, chunk_size=2_000_000)
            bits = int(math.ceil(math.log2(int(centers.size)))) if centers.size > 1 else 0
            results.append(save_and_measure(
                outdir, name, "hybrid",
                {"left": left, "right": right, "residual_codebook": centers.astype(np.float32),
                 "raw_indices": indices},
                {"rank": int(rank), "residual_groups_requested": int(groups),
                 "residual_groups_actual": int(centers.size), "index_bits": bits,
                 "factor_dtype": "fp16", "residual_codebook_dtype": "fp32",
                 "oversample": int(args.oversample), "power_iterations": int(args.power_iterations)},
                x, source_bytes, shape, args.projection_batch, args.seed, args.timing_repeats, args.map_codec,
            ))

    results.sort(key=lambda item: (item["full_tensor_error"]["rmse_over_weight_std"], item["artifact_bytes_actual"]))
    summary = {
        "status": "deep_tensor_experiment_completed",
        "model_or_path": args.model,
        "tensor_name": chosen["name"],
        "shape": list(shape),
        "source_dtype": source_dtype,
        "numel": int(x.size),
        "source_tensor_bytes": int(source_bytes),
        "source_tensor_MB": source_bytes / 1_000_000,
        "parameters": {
            "kmeans_samples": args.kmeans_samples, "kmeans_groups": args.kmeans_groups,
            "log_levels": args.log_levels, "log_scales": args.log_scales, "ranks": args.ranks,
            "hybrid_ranks": args.hybrid_ranks, "residual_groups": args.residual_groups,
            "map_codec": args.map_codec,
            "residual_sample_size": args.residual_sample_size,
            "oversample": args.oversample, "power_iterations": args.power_iterations,
        },
        "results_sorted_by_normalized_rmse": results,
        "limitations": [
            "Um único tensor por execução; ainda não é um checkpoint Transformers completo.",
            "A SVD é aleatória aproximada e usa uma mesma base de posto máximo para comparação eficiente.",
            "Os fatores e codebooks são armazenados em FP16/FP32 conforme indicado; métricas são calculadas após recarregar o NPZ.",
            "W@X usa entradas gaussianas sintéticas, não ativações reais nem perplexidade.",
            "O arquivo NPZ mede tamanho físico, mas não representa ainda um formato de inferência otimizado.",
        ],
        "elapsed_seconds": round(time.time() - started, 2),
    }
    report_path = outdir / "deep_comparison.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = outdir / "deep_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "artifact_MB", "savings_pct", "map_codec", "rmse_over_std", "weight_cosine", "W@X_relative_error", "offline_decode_seconds", "direct_factorized_ms", "dense_over_factorized_ratio"])
        for item in results:
            probe = item["linear_projection_probe"]
            writer.writerow([
                item["method"], f"{item['artifact_MB_actual']:.6f}",
                f"{item['savings_pct_vs_source']:.4f}", item["selected_map_codec"],
                f"{item['full_tensor_error']['rmse_over_weight_std']:.8f}",
                f"{item['full_tensor_error']['cosine_similarity_flat_weights']:.8f}",
                "" if probe is None else f"{probe['output_rmse_over_original_rms']:.8f}",
                f"{item['archive_decode_seconds_offline']:.6f}",
                "" if item["direct_factorized_projection_benchmark"] is None else f"{item['direct_factorized_projection_benchmark']['direct_U_times_VTX_median_ms']:.6f}",
                "" if item["direct_factorized_projection_benchmark"] is None else f"{item['direct_factorized_projection_benchmark']['dense_over_factorized_speed_ratio']:.6f}",
            ])

    print("\n=== DEEP TENSOR EXPERIMENT ===")
    print(f"{'MÉTODO':34s} {'ARQUIVO MB':>11s} {'REDUÇÃO':>9s} {'CODEC':>22s} {'RMSE/std':>10s} {'COSSENO':>10s} {'ERRO W@X':>10s} {'DECODE s':>9s}")
    for item in results:
        metric = item["full_tensor_error"]
        probe = item["linear_projection_probe"]
        werr = probe["output_rmse_over_original_rms"] if probe else float("nan")
        print(f"{item['method']:34s} {item['artifact_MB_actual']:11.3f} {item['savings_pct_vs_source']:8.2f}% {item['selected_map_codec']:>22s} {metric['rmse_over_weight_std']:10.5f} {metric['cosine_similarity_flat_weights']:10.6f} {werr:10.5f} {item['archive_decode_seconds_offline']:9.3f}")
    print(f"\nRelatório JSON: {report_path.resolve()}")
    print(f"Resumo CSV   : {csv_path.resolve()}")
    print(f"Tempo total  : {time.time() - started:.2f}s")
    print("Os melhores resultados por RMSE aparecem primeiro; avalie tamanho e erro em conjunto.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.", file=sys.stderr)
        raise SystemExit(130)
