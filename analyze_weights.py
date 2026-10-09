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


def allocate_samples(tensors: list[dict[str, Any]], budget: int) -> dict[tuple[str, str], int]:
    """Allocate a model-size-proportional sample, using largest remainders."""
    total = sum(t["numel"] for t in tensors)
    if total == 0:
        return {}
    budget = min(budget, total)
    raw = [budget * t["numel"] / total for t in tensors]
    alloc = [min(t["numel"], int(math.floor(v))) for t, v in zip(tensors, raw)]
    remain = budget - sum(alloc)
    order = sorted(range(len(tensors)), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
    for i in order:
        if remain <= 0:
            break
        if alloc[i] < tensors[i]["numel"]:
            alloc[i] += 1
            remain -= 1
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


def weighted_1d_kmeans(sample: np.ndarray, k: int, max_iter: int = 40) -> tuple[np.ndarray, float, float, float]:
    """Lloyd k-means for 1D scalar weights; sample duplicates are frequency weights."""
    x = np.asarray(sample, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Amostra sem pesos finitos")
    values, counts = np.unique(x, return_counts=True)
    counts = counts.astype(np.float64)
    k = min(int(k), len(values))
    if k == 1:
        centers = np.array([np.average(values, weights=counts)], dtype=np.float64)
    else:
        cumulative = np.cumsum(counts)
        targets = (np.arange(k, dtype=np.float64) + 0.5) * cumulative[-1] / k
        idx = np.searchsorted(cumulative, targets, side="left")
        centers = np.unique(values[np.minimum(idx, len(values) - 1)].astype(np.float64))
        if centers.size < k:
            q = np.linspace(0, len(values) - 1, k).round().astype(int)
            centers = np.unique(values[q]).astype(np.float64)

    for _ in range(max_iter):
        if centers.size <= 1:
            break
        centers.sort()
        boundaries = (centers[:-1] + centers[1:]) / 2.0
        assignment = np.searchsorted(boundaries, values, side="right")
        cluster_weight = np.bincount(assignment, weights=counts, minlength=centers.size)
        cluster_sum = np.bincount(assignment, weights=counts * values, minlength=centers.size)
        nonempty = cluster_weight > 0
        updated = centers.copy()
        updated[nonempty] = cluster_sum[nonempty] / cluster_weight[nonempty]
        updated = updated[nonempty]
        if updated.size == centers.size and np.allclose(updated, centers, rtol=1e-7, atol=1e-12):
            centers = updated
            break
        centers = updated

    centers.sort()
    assignment = np.searchsorted((centers[:-1] + centers[1:]) / 2.0, values, side="right") if centers.size > 1 else np.zeros(values.size, dtype=np.int64)
    errors = values - centers[assignment]
    mse = float(np.average(errors * errors, weights=counts))
    mae = float(np.average(np.abs(errors), weights=counts))
    rmse = math.sqrt(max(mse, 0.0))
    return centers.astype(np.float32), mse, mae, rmse


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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.sample_size < 1000:
        parser.error("--sample-size precisa ser >= 1000 para produzir estatísticas úteis")
    if args.chunk_elements < 1:
        parser.error("--chunk-elements precisa ser positivo")

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
    sample_counts = allocate_samples(tensors, args.sample_size)
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
                        hist = bit_histograms.setdefault(dtype_key, np.zeros(65536, dtype=np.uint64))
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

    groups_rows: list[dict[str, Any]] = []
    for requested_k in args.groups:
        centers, mse, mae, rmse = weighted_1d_kmeans(sample, requested_k)
        actual_k = int(centers.size)
        bits_per_index = int(math.ceil(math.log2(actual_k))) if actual_k > 1 else 0
        packed_index_bytes = (total_params * bits_per_index + 7) // 8
        codebook_bytes = actual_k * 4  # float32 representatives, conservatively
        estimated_bytes = packed_index_bytes + codebook_bytes
        sample_std = float(np.std(sample, dtype=np.float64))
        rel_rmse = rmse / sample_std if sample_std > 0 else 0.0
        groups_rows.append({
            "requested_groups": requested_k, "actual_groups": actual_k,
            "index_bits_per_weight": bits_per_index,
            "sample_rmse": rmse, "sample_mae": mae, "sample_mse": mse,
            "relative_rmse_vs_sample_std": rel_rmse,
            "estimated_index_MB_ideal_packed": packed_index_bytes / 1_000_000,
            "estimated_codebook_MB_fp32": codebook_bytes / 1_000_000,
            "estimated_total_MB_ideal_packed": estimated_bytes / 1_000_000,
            "estimated_size_ratio_vs_source": estimated_bytes / total_bytes if total_bytes else 0.0,
            "estimated_savings_pct_vs_source": (1 - estimated_bytes / total_bytes) * 100 if total_bytes else 0.0,
            "note": "Estimativa ideal por amostra; não inclui metadados/overhead e não é um checkpoint codificado",
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
        "sample_min": float(np.min(sample)),
        "sample_max": float(np.max(sample)),
        "exact_unique_bit_patterns_16bit": exact_16bit,
        "exact_unique_float32_bit_patterns": exact_32bit,
        "float32_unique_count_is_exact": bool(float32_patterns_exact),
        "group_analysis": groups_rows,
        "top_20_tensors_by_size": tensor_stats[:20],
        "elapsed_seconds": round(time.time() - start_time, 2),
        "scan_seconds": round(time.time() - started_scan, 2),
        "caveats": [
            "A quantidade de padrões exatos é separada por formato binário (BF16/F16/F32); formatos diferentes não compartilham a mesma contagem.",
            "A análise de grupos aproximados usa amostra uniforme proporcional ao número de parâmetros, não todos os valores para calcular a distorção.",
            "O codebook é global para todos os tensores; codebooks por tensor/camada podem reduzir o erro, mas adicionam metadados e custo de armazenamento.",
            "O tamanho comprimido é uma estimativa ideal com índices bit-packed; nenhum peso quantizado/comprimido é escrito nesta primeira fase.",
            "MSE/RMSE avaliam os valores dos pesos, não a qualidade linguística ou multimodal do modelo. É necessário validar as saídas antes de concluir que a compressão é útil.",
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
    print("\nGrupos | RMSE amostral | RMSE/desvio-padrão | Tamanho estimado | Economia estimada")
    for row in groups_rows:
        print(f"{row['actual_groups']:>6} | {row['sample_rmse']:.6g} | {row['relative_rmse_vs_sample_std']:.6g} | "
              f"{row['estimated_total_MB_ideal_packed']:.2f} MB | {row['estimated_savings_pct_vs_source']:.2f}%")
    print(f"\nArquivos gravados em: {output_dir.resolve()}")
    print("  - summary.json      (resumo completo e ressalvas)")
    print("  - group_analysis.csv (trade-off entre grupos, erro e tamanho estimado)")
    print("  - tensor_stats.csv   (estatísticas por tensor/camada)")
    print("\nNota: esta fase mede potencial. Ainda não produz um modelo comprimido funcional.")
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
