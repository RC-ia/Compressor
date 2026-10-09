#!/usr/bin/env python3
"""Encode one real Safetensors tensor as representatives + packed assignment map.

This is a measured prototype, not a drop-in Transformer checkpoint format. It
writes an NPZ containing one tensor's codebook and its packed indices, reloads
it through the decoder, and measures full-tensor reconstruction error and an
optional linear-projection probe.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def resolve_model(model_or_path: str, cache_dir: str | None, revision: str | None) -> list[Path]:
    candidate = Path(model_or_path).expanduser()
    if candidate.is_file() and candidate.suffix == ".safetensors":
        return [candidate.resolve()]
    if candidate.is_dir():
        files = sorted(candidate.rglob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"Nenhum arquivo .safetensors encontrado em {candidate}")
        return files
    print(f"[download] Resolvendo {model_or_path}; o checkpoint completo pode ocupar vários GB.")
    root = Path(snapshot_download(
        repo_id=model_or_path,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json"],
    ))
    files = sorted(root.rglob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"Hugging Face não retornou arquivos Safetensors para {model_or_path}")
    return files


def list_tensors(files: list[Path]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as sf:
            for name in sf.keys():
                tensor_slice = sf.get_slice(name)
                shape = tuple(int(x) for x in tensor_slice.get_shape())
                found.append({"file": path, "name": name, "shape": shape,
                              "numel": int(math.prod(shape)) if shape else 1})
    found.sort(key=lambda item: item["numel"], reverse=True)
    return found


def choose_tensor(files: list[Path], tensor_name: str | None, max_auto_elements: int = 20_000_000) -> dict[str, Any]:
    candidates = list_tensors(files)
    if tensor_name:
        matches = [item for item in candidates if item["name"] == tensor_name]
        if not matches:
            preview = "\n".join(
                f"  {x['name']}  shape={x['shape']}  file={x['file'].name}"
                for x in candidates[:25]
            )
            raise KeyError(f"Tensor {tensor_name!r} não encontrado. Maiores tensores:\n{preview}")
        return matches[0]
    matrices = [item for item in candidates if len(item["shape"]) == 2 and item["numel"] >= 1_000_000]
    if not matrices:
        matrices = [item for item in candidates if len(item["shape"]) == 2]
    if not matrices:
        raise RuntimeError("Não encontrei tensores bidimensionais para o teste automático.")
    bounded = [item for item in matrices if item["numel"] <= max_auto_elements]
    # Avoid accidentally selecting an enormous tied embedding/LM-head matrix.
    # Prefer the largest medium-sized matrix; fall back to the smallest available matrix.
    return max(bounded, key=lambda item: item["numel"]) if bounded else min(matrices, key=lambda item: item["numel"])


def weighted_kmeans_1d(sample: np.ndarray, k: int, max_iter: int = 25) -> np.ndarray:
    """Deterministic scalar k-means with multiple starts; trained on sampled weights."""
    x = np.asarray(sample, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("A amostra não contém pesos finitos")
    values, counts = np.unique(x, return_counts=True)
    counts = counts.astype(np.float64)
    k = min(max(int(k), 1), values.size)
    if k == 1:
        return np.array([np.average(values, weights=counts)], dtype=np.float32)

    cumulative = np.cumsum(counts)
    targets = (np.arange(k, dtype=np.float64) + 0.5) * cumulative[-1] / k
    q_idx = np.searchsorted(cumulative, targets, side="left")
    quantile_centers = np.unique(values[np.minimum(q_idx, values.size - 1)])
    while quantile_centers.size < k:
        sorted_centers = np.sort(quantile_centers)
        bounds = (sorted_centers[:-1] + sorted_centers[1:]) / 2.0
        assignment = np.searchsorted(bounds, values, side="right")
        error_contribution = counts * np.square(values - sorted_centers[assignment])
        error_contribution[np.isin(values, sorted_centers)] = -np.inf
        pick = int(np.argmax(error_contribution))
        if not np.isfinite(error_contribution[pick]):
            break
        quantile_centers = np.append(quantile_centers, values[pick])

    rank_idx = np.linspace(0, values.size - 1, k).round().astype(np.int64)
    starts = [quantile_centers, np.unique(values[rank_idx])]

    def fit(start: np.ndarray) -> tuple[np.ndarray, float]:
        centers = np.unique(np.asarray(start, dtype=np.float64))
        for _ in range(max_iter):
            centers.sort()
            bounds = (centers[:-1] + centers[1:]) / 2.0
            assignment = np.searchsorted(bounds, values, side="right")
            mass = np.bincount(assignment, weights=counts, minlength=centers.size)
            sums = np.bincount(assignment, weights=counts * values, minlength=centers.size)
            live = mass > 0
            new_centers = sums[live] / mass[live]
            if new_centers.size == centers.size and np.allclose(new_centers, centers, rtol=1e-7, atol=1e-12):
                centers = new_centers
                break
            centers = new_centers
        centers.sort()
        bounds = (centers[:-1] + centers[1:]) / 2.0
        assignment = np.searchsorted(bounds, values, side="right") if centers.size > 1 else np.zeros(values.size, dtype=np.int64)
        mse = float(np.average(np.square(values - centers[assignment]), weights=counts))
        return centers, mse

    fitted = [fit(seed) for seed in starts]
    fitted.sort(key=lambda item: item[1])
    return fitted[0][0].astype(np.float32)


def prepare_codebook(centers: np.ndarray, dtype: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Return archived codebook array, float32 reconstruction values, encoding tag."""
    c = np.asarray(centers, dtype=np.float32)
    if dtype == "fp16":
        archived = c.astype(np.float16)
        decoded = archived.astype(np.float32)
        tag = "float16"
    elif dtype == "bf16":
        bf16 = torch.from_numpy(c.copy()).to(torch.bfloat16).contiguous()
        archived = bf16.view(torch.uint16).numpy().copy()
        decoded = torch.from_numpy(archived.copy()).view(torch.bfloat16).float().numpy()
        tag = "bfloat16_bits_u16"
    else:
        archived = c.astype(np.float32)
        decoded = archived.copy()
        tag = "float32"
    # Rounding a higher-precision codebook can collapse neighboring centers.
    unique_decoded = np.unique(decoded)
    if unique_decoded.size != decoded.size:
        decoded = unique_decoded.astype(np.float32)
        if dtype == "fp16":
            archived = decoded.astype(np.float16)
        elif dtype == "bf16":
            archived = torch.from_numpy(decoded.copy()).to(torch.bfloat16).view(torch.uint16).numpy().copy()
        else:
            archived = decoded.astype(np.float32)
    return archived, decoded, tag


def assign_indices(values: np.ndarray, centers: np.ndarray, chunk_size: int) -> np.ndarray:
    """Assign all weights to the nearest one-dimensional codebook center."""
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    c = np.sort(np.asarray(centers, dtype=np.float32))
    if c.size == 1:
        return np.zeros(x.size, dtype=np.uint8)
    bounds = ((c[:-1].astype(np.float64) + c[1:].astype(np.float64)) * 0.5).astype(np.float32)
    dtype = np.uint8 if c.size <= 256 else (np.uint16 if c.size <= 65536 else np.uint32)
    indices = np.empty(x.size, dtype=dtype)
    for begin in range(0, x.size, chunk_size):
        end = min(x.size, begin + chunk_size)
        indices[begin:end] = np.searchsorted(bounds, x[begin:end], side="right").astype(dtype)
    return indices


def pack_indices_bitplanes(indices: np.ndarray, bits_per_index: int) -> np.ndarray:
    """Pack fixed-width symbols into LSB-first bitplanes (small, vectorizable, lossless)."""
    idx = np.asarray(indices).reshape(-1).astype(np.uint32, copy=False)
    if bits_per_index == 0:
        return np.empty(0, dtype=np.uint8)
    planes = [np.packbits(((idx >> bit) & 1).astype(np.uint8), bitorder="little") for bit in range(bits_per_index)]
    return np.concatenate(planes).astype(np.uint8, copy=False)


def unpack_indices_bitplanes(packed: np.ndarray, count: int, bits_per_index: int) -> np.ndarray:
    if bits_per_index == 0:
        return np.zeros(count, dtype=np.uint8)
    data = np.asarray(packed, dtype=np.uint8).reshape(-1)
    plane_bytes = (count + 7) // 8
    expected = plane_bytes * bits_per_index
    if data.size != expected:
        raise ValueError(f"Mapa inválido: esperado {expected} bytes, recebido {data.size}")
    indices = np.zeros(count, dtype=np.uint32)
    for bit in range(bits_per_index):
        start = bit * plane_bytes
        plane = np.unpackbits(data[start:start + plane_bytes], bitorder="little")[:count]
        indices |= plane.astype(np.uint32) << bit
    if len(indices) and int(indices.max()) >= (1 << bits_per_index):
        raise ValueError("Mapa contém índice além da faixa permitida")
    return indices


def decode_archive(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        meta_raw = archive["metadata"].astype(np.uint8, copy=False).tobytes()
        metadata = json.loads(meta_raw.decode("utf-8"))
        codebook_archived = archive["codebook"]
        if metadata["codebook_encoding"] == "bfloat16_bits_u16":
            codebook = torch.from_numpy(codebook_archived.astype(np.uint16, copy=True)).view(torch.bfloat16).float().numpy()
        else:
            codebook = codebook_archived.astype(np.float32)
        index_encoding = metadata.get("index_encoding", "bitplanes")
        if index_encoding == "zlib_symbols":
            compressed_symbols = archive["zlib_indices"].astype(np.uint8, copy=False).tobytes()
            raw_symbols = zlib.decompress(compressed_symbols)
            storage_dtype = np.dtype(metadata["index_storage_dtype"])
            indices = np.frombuffer(raw_symbols, dtype=storage_dtype, count=int(metadata["numel"])).astype(np.uint32)
            if indices.size != int(metadata["numel"]):
                raise ValueError("Mapa zlib inválido: quantidade de índices divergente")
            return codebook, indices, metadata
        packed = archive["packed_indices"].copy()
    if index_encoding != "bitplanes":
        raise ValueError(f"Codificação de índices desconhecida: {index_encoding}")
    indices = unpack_indices_bitplanes(packed, int(metadata["numel"]), int(metadata["index_bits"]))
    return codebook, indices, metadata


def analyze_index_map(indices: np.ndarray, group_count: int) -> dict[str, Any]:
    """Measure empirical entropy, frequency skew, and simple local-run structure."""
    idx = np.asarray(indices).reshape(-1).astype(np.uint32, copy=False)
    hist = np.bincount(idx, minlength=group_count).astype(np.int64)
    used = hist[hist > 0]
    probs = used.astype(np.float64) / max(1, idx.size)
    entropy = float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0
    used_groups = int(used.size)
    changes = np.flatnonzero(idx[1:] != idx[:-1]) + 1 if idx.size > 1 else np.empty(0, dtype=np.int64)
    run_count = int(changes.size + (1 if idx.size else 0))
    if idx.size:
        run_starts = np.concatenate((np.array([0], dtype=np.int64), changes))
        run_ends = np.concatenate((changes, np.array([idx.size], dtype=np.int64)))
        longest_run = int(np.max(run_ends - run_starts))
    else:
        longest_run = 0
    top_order = np.argsort(hist)[::-1][:min(10, hist.size)]
    frequency = [
        {"representative_index": int(i), "count": int(hist[i]), "fraction": float(hist[i] / idx.size)}
        for i in top_order if hist[i] > 0
    ]
    return {
        "used_representatives": used_groups,
        "symbol_entropy_bits_per_weight": entropy,
        "ideal_zero_order_entropy_MB": idx.size * entropy / 8 / 1_000_000,
        "fixed_width_index_bits_per_weight": int(math.ceil(math.log2(group_count))) if group_count > 1 else 0,
        "zero_order_entropy_vs_fixed_width_savings_pct": (1 - entropy / max(1, math.ceil(math.log2(group_count)))) * 100 if group_count > 1 else 0.0,
        "adjacent_transition_rate": float(changes.size / max(1, idx.size - 1)),
        "run_count": run_count,
        "average_run_length": float(idx.size / run_count) if run_count else 0.0,
        "longest_run": longest_run,
        "top_representative_frequencies": frequency,
    }


def _metadata_array(metadata: dict[str, Any]) -> np.ndarray:
    return np.frombuffer(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), dtype=np.uint8)


def write_archive_candidate(path: Path, codec: str, codebook: np.ndarray,
                            packed_indices: np.ndarray, indices: np.ndarray,
                            base_metadata: dict[str, Any], index_bits: int) -> None:
    meta = dict(base_metadata)
    if codec in ("zip_deflate_bitplanes", "raw_bitplanes"):
        meta["index_encoding"] = "bitplanes"
        meta["index_storage_dtype"] = "packed_bits"
        meta["index_bits"] = index_bits
        arrays = {"codebook": codebook, "packed_indices": packed_indices, "metadata": _metadata_array(meta)}
        if codec == "zip_deflate_bitplanes":
            np.savez_compressed(path, **arrays)
        else:
            np.savez(path, **arrays)
    elif codec == "zlib_symbols":
        dtype = np.dtype("<u1" if indices.size and int(np.max(indices)) <= 255 else "<u2" if indices.size and int(np.max(indices)) <= 65535 else "<u4")
        symbol_bytes = indices.astype(dtype, copy=False).tobytes()
        compressed = zlib.compress(symbol_bytes, level=9)
        meta["index_encoding"] = "zlib_symbols"
        meta["index_storage_dtype"] = dtype.str
        meta["index_bits"] = int(math.ceil(math.log2(base_metadata["actual_groups"]))) if base_metadata["actual_groups"] > 1 else 0
        np.savez(path, codebook=codebook, zlib_indices=np.frombuffer(compressed, dtype=np.uint8), metadata=_metadata_array(meta))
    else:
        raise ValueError(f"Codificador não suportado: {codec}")


def error_metrics(values: np.ndarray, indices: np.ndarray, centers: np.ndarray, chunk_size: int) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    mse_sum = 0.0
    mae_sum = 0.0
    max_abs = 0.0
    sum_x2 = 0.0
    sum_recon2 = 0.0
    dot = 0.0
    count = x.size
    for begin in range(0, count, chunk_size):
        end = min(count, begin + chunk_size)
        orig = x[begin:end].astype(np.float64)
        recon = centers[indices[begin:end]].astype(np.float64)
        err = orig - recon
        mse_sum += float(np.dot(err, err))
        mae_sum += float(np.abs(err).sum())
        if err.size:
            max_abs = max(max_abs, float(np.abs(err).max()))
        sum_x2 += float(np.dot(orig, orig))
        sum_recon2 += float(np.dot(recon, recon))
        dot += float(np.dot(orig, recon))
    mse = mse_sum / max(1, count)
    rmse = math.sqrt(max(0.0, mse))
    std = float(np.std(x, dtype=np.float64))
    cosine = dot / math.sqrt(sum_x2 * sum_recon2) if sum_x2 > 0 and sum_recon2 > 0 else 1.0
    return {
        "numel": int(count),
        "rmse": rmse,
        "mae": mae_sum / max(1, count),
        "max_abs_error": max_abs,
        "weight_std": std,
        "rmse_over_weight_std": rmse / std if std > 0 else 0.0,
        "cosine_similarity_flat_weights": cosine,
    }


def projection_probe(original: np.ndarray, indices: np.ndarray, centers: np.ndarray,
                     shape: tuple[int, ...], batch_size: int, seed: int) -> dict[str, Any] | None:
    if len(shape) != 2 or batch_size < 1:
        return None
    out_features, in_features = shape
    original_2d = np.asarray(original, dtype=np.float32).reshape(out_features, in_features)
    reconstructed = centers[indices].astype(np.float32, copy=False).reshape(out_features, in_features)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((in_features, batch_size), dtype=np.float32)
    x /= math.sqrt(max(1, in_features))
    y_original = original_2d @ x
    y_recon = reconstructed @ x
    diff = y_original.astype(np.float64) - y_recon.astype(np.float64)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom = float(np.sqrt(np.mean(y_original.astype(np.float64) ** 2)))
    cosine = float(np.sum(y_original.astype(np.float64) * y_recon.astype(np.float64)) / max(
        1e-30, np.linalg.norm(y_original.astype(np.float64)) * np.linalg.norm(y_recon.astype(np.float64))))
    return {
        "batch_size": int(batch_size),
        "output_rmse": rmse,
        "output_rmse_over_original_rms": rmse / denom if denom > 0 else 0.0,
        "output_cosine_similarity": cosine,
        "note": "Probe linear com entradas gaussianas sintéticas; não substitui avaliação de ativações ou perplexidade reais.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Repo HF, diretório local ou arquivo Safetensors")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--tensor-name", default=None, help="Nome exato do tensor; sem isso escolhe uma matriz 2D intermediária para limitar a memória")
    parser.add_argument("--max-auto-elements", type=int, default=20_000_000, help="Maior matriz selecionada automaticamente (não limita --tensor-name)")
    parser.add_argument("--list-tensors", action="store_true", help="Listar os maiores tensores e sair")
    parser.add_argument("--groups", type=int, default=256, help="Número máximo de representantes para este tensor")
    parser.add_argument("--sample-size", type=int, default=250_000, help="Amostra de pesos usada para aprender os representantes")
    parser.add_argument("--codebook-dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--output-dir", default="compressor_tensor_test")
    parser.add_argument("--assignment-chunk", type=int, default=2_000_000)
    parser.add_argument("--metric-chunk", type=int, default=1_000_000)
    parser.add_argument("--projection-batch", type=int, default=4, help="Largura do teste W@X para matriz 2D; 0 desativa")
    parser.add_argument("--map-codec", choices=["auto", "zip_deflate_bitplanes", "zlib_symbols", "raw_bitplanes"], default="auto", help="Codificação do mapa; auto escolhe o menor artefato medido")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-reconstructed-safetensors", action="store_true", help="Também salva o tensor reconstruído em Safetensors (formato original)")
    args = parser.parse_args()

    if args.groups < 2:
        parser.error("--groups precisa ser >= 2")
    if args.sample_size < 1000:
        parser.error("--sample-size precisa ser >= 1000")
    if args.assignment_chunk < 1 or args.metric_chunk < 1:
        parser.error("Os tamanhos de bloco precisam ser positivos")
    if args.projection_batch < 0:
        parser.error("--projection-batch não pode ser negativo")
    if args.max_auto_elements < 1:
        parser.error("--max-auto-elements precisa ser positivo")

    started = time.time()
    files = resolve_model(args.model, args.cache_dir, args.revision)
    candidates = list_tensors(files)
    if args.list_tensors:
        print("Maiores tensores Safetensors:")
        for item in candidates[:100]:
            print(f"{item['numel']:>14,}  shape={item['shape']}  {item['name']}  [{item['file'].name}]")
        return 0
    chosen = choose_tensor(files, args.tensor_name, args.max_auto_elements)
    print(f"[tensor] {chosen['name']} | shape={chosen['shape']} | elements={chosen['numel']:,} | file={chosen['file'].name}")
    with safe_open(str(chosen["file"]), framework="pt", device="cpu") as sf:
        source_tensor = sf.get_tensor(chosen["name"])
    source_dtype = str(source_tensor.dtype).replace("torch.", "")
    if not source_tensor.is_floating_point():
        raise TypeError(f"Tensor não é flutuante: {source_dtype}")
    original_shape = tuple(int(x) for x in source_tensor.shape)
    source_bytes = source_tensor.numel() * source_tensor.element_size()
    original = source_tensor.float().cpu().numpy().reshape(-1).copy()
    del source_tensor

    rng = np.random.default_rng(args.seed)
    sample_count = min(args.sample_size, original.size)
    if sample_count < original.size:
        sample_index = rng.choice(original.size, size=sample_count, replace=False)
        sample = original[sample_index]
    else:
        sample = original
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise RuntimeError("Nenhum valor finito disponível para treinar o codebook")

    train_centers = weighted_kmeans_1d(sample, args.groups)
    archived_codebook, centers, codebook_encoding = prepare_codebook(train_centers, args.codebook_dtype)
    if centers.size > args.groups:
        raise RuntimeError("O codebook final excedeu a quantidade de grupos solicitada")
    indices = assign_indices(original, centers, args.assignment_chunk)
    bits = int(math.ceil(math.log2(centers.size))) if centers.size > 1 else 0
    packed_indices = pack_indices_bitplanes(indices, bits)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    tensor_stem = chosen["name"].replace(".", "_").replace("/", "_")[-100:]
    artifact_path = outdir / f"{tensor_stem}_{centers.size}groups.npz"
    base_metadata = {
        "format": "compressor_tensor_v2",
        "source_model": args.model,
        "source_file": chosen["file"].name,
        "tensor_name": chosen["name"],
        "shape": list(original_shape),
        "numel": int(original.size),
        "source_dtype": source_dtype,
        "source_tensor_bytes": int(source_bytes),
        "requested_groups": int(args.groups),
        "actual_groups": int(centers.size),
        "codebook_dtype": args.codebook_dtype,
        "codebook_encoding": codebook_encoding,
        "index_bits": bits,
        "packed_layout": "lsb_first_bitplanes",
        "packed_index_bytes": int(packed_indices.size),
        "sample_size_used": int(sample.size),
        "seed": int(args.seed),
    }
    entropy_stats = analyze_index_map(indices, int(centers.size))
    raw_index_symbols = indices.astype(np.uint8 if centers.size <= 256 else np.uint16 if centers.size <= 65536 else np.uint32, copy=False).tobytes()
    raw_symbols_zlib = zlib.compress(raw_index_symbols, level=9)
    packed_zlib = zlib.compress(packed_indices.tobytes(), level=9)

    codec_candidates = [args.map_codec] if args.map_codec != "auto" else ["zip_deflate_bitplanes", "zlib_symbols", "raw_bitplanes"]
    measured_candidates: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="compressor-candidates-", dir=str(outdir)) as tmpdir:
        temp_root = Path(tmpdir)
        for codec in codec_candidates:
            temp_path = temp_root / f"candidate_{codec}.npz"
            write_archive_candidate(temp_path, codec, archived_codebook, packed_indices, indices, base_metadata, bits)
            measured_candidates.append({"codec": codec, "bytes": int(temp_path.stat().st_size), "path": str(temp_path)})
        measured_candidates.sort(key=lambda item: item["bytes"])
        selected_codec = measured_candidates[0]["codec"]
        shutil.copyfile(measured_candidates[0]["path"], artifact_path)

    # The test intentionally uses the saved artifact as input to the decoder.
    decoded_centers, decoded_indices, decoded_metadata = decode_archive(artifact_path)
    if decoded_indices.size != original.size or int(decoded_metadata["numel"]) != original.size:
        raise RuntimeError("Round-trip incompatível: contagem de índices incorreta")
    if not np.array_equal(decoded_indices, indices.astype(np.uint32)):
        raise RuntimeError("Round-trip falhou: índices reconstruídos não coincidem")
    metrics = error_metrics(original, decoded_indices, decoded_centers, args.metric_chunk)
    projection = projection_probe(original, decoded_indices, decoded_centers, original_shape,
                                  args.projection_batch, args.seed + 100) if args.projection_batch else None

    archive_bytes = artifact_path.stat().st_size
    ideal_map_bytes = (original.size * bits + 7) // 8
    raw_book_bytes = int(archived_codebook.nbytes)
    npz_gain = 100 * (1 - archive_bytes / source_bytes) if source_bytes else 0.0
    report = {
        "status": "encoded_decoded_measured",
        "source_model": args.model,
        "source_tensor": chosen["name"],
        "shape": list(original_shape),
        "source_dtype": source_dtype,
        "source_tensor_bytes": int(source_bytes),
        "source_tensor_MB_decimal": source_bytes / 1e6,
        "actual_groups": int(centers.size),
        "index_bits": bits,
        "selected_map_codec": selected_codec,
        "map_codec_candidates_measured": [{"codec": item["codec"], "artifact_bytes": item["bytes"]} for item in measured_candidates],
        "sample_used_for_codebook": int(sample.size),
        "codebook_dtype": args.codebook_dtype,
        "codebook_raw_bytes": raw_book_bytes,
        "index_map_fixed_bit_bytes": int(ideal_map_bytes),
        "index_map_bitplane_bytes_before_deflate": int(packed_indices.size),
        "index_map_bitplane_zlib9_bytes": int(len(packed_zlib)),
        "index_map_symbol_bytes_fixed_width": int(len(raw_index_symbols)),
        "index_map_symbols_zlib9_bytes": int(len(raw_symbols_zlib)),
        "index_map_entropy_analysis": entropy_stats,
        "artifact_npz_bytes_actual": int(archive_bytes),
        "artifact_npz_MB_actual": archive_bytes / 1e6,
        "artifact_over_source_ratio": archive_bytes / source_bytes if source_bytes else None,
        "actual_file_savings_pct_vs_source_tensor": npz_gain,
        "full_tensor_weight_reconstruction_error": metrics,
        "linear_projection_probe": projection,
        "archive_path": str(artifact_path.resolve()),
        "elapsed_seconds": round(time.time() - started, 2),
        "limitations": [
            "Este teste codifica somente um tensor, não o checkpoint inteiro.",
            "O artefato NPZ contém tabela e mapa real; seu tamanho vem de stat() após gravação com ZIP/DEFLATE ou armazenamento binário.",
            "A redução de tamanho não é necessariamente uma redução de latência; lookup e reconstrução requerem kernels apropriados.",
            "O teste W@X usa entradas gaussianas sintéticas e não substitui avaliação com ativações reais do Qwen nem perplexidade.",
            "O script mede entropia de ordem zero, transições locais e tamanho real de três variantes do mapa: bitplanes com ZIP/DEFLATE, símbolos com zlib e bitplanes sem compressão.",
            "A entropia de ordem zero é um limite ideal baseado apenas em frequências, não o tamanho garantido de um arquivo Huffman/aritimético; dependências sequenciais podem alterar os resultados.",
        ],
    }
    report_path = outdir / f"{tensor_stem}_{centers.size}groups_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.save_reconstructed_safetensors:
        original_dtype_torch = getattr(torch, source_dtype, torch.float32)
        rebuilt = torch.from_numpy(decoded_centers[decoded_indices.astype(np.int64)].reshape(original_shape).copy())
        rebuilt = rebuilt.to(dtype=original_dtype_torch).contiguous()
        rebuilt_path = outdir / f"{tensor_stem}_{centers.size}groups_reconstructed.safetensors"
        save_file({"weight": rebuilt}, str(rebuilt_path), metadata={"source_tensor": chosen["name"]})
        report["reconstructed_safetensors_path"] = str(rebuilt_path.resolve())
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== RECONSTRUÇÃO REAL DO TENSOR ===")
    print(f"Tensor                    : {chosen['name']}")
    print(f"Forma                     : {original_shape}")
    print(f"Formato original          : {source_dtype}")
    print(f"Representantes            : {centers.size}")
    print(f"Bits por índice           : {bits}")
    print(f"Tensor original           : {source_bytes / 1e6:.3f} MB")
    print(f"Mapa fixo teórico         : {ideal_map_bytes / 1e6:.3f} MB")
    print(f"Entropia índice           : {entropy_stats['symbol_entropy_bits_per_weight']:.4f} bits/peso")
    print(f"Limite entropia ordem-zero: {entropy_stats['ideal_zero_order_entropy_MB']:.3f} MB")
    print(f"Mapa bitplanes + zlib     : {len(packed_zlib) / 1e6:.3f} MB")
    print(f"Mapa símbolos + zlib      : {len(raw_symbols_zlib) / 1e6:.3f} MB")
    print(f"Codebook                  : {raw_book_bytes / 1e6:.6f} MB")
    print(f"Codificação escolhida     : {selected_codec}")
    print(f"Artefato NPZ real         : {archive_bytes / 1e6:.3f} MB")
    print(f"Redução real deste tensor : {npz_gain:.2f}%")
    print(f"RMSE                      : {metrics['rmse']:.8g}")
    print(f"RMSE / desvio-padrão      : {metrics['rmse_over_weight_std']:.8g}")
    print(f"MAE                       : {metrics['mae']:.8g}")
    print(f"Similaridade cosseno      : {metrics['cosine_similarity_flat_weights']:.10f}")
    if projection:
        print(f"Erro relativo W@X         : {projection['output_rmse_over_original_rms']:.8g}")
        print(f"Cosseno das saídas W@X    : {projection['output_cosine_similarity']:.10f}")
    print(f"Relatório                 : {report_path.resolve()}")
    print("Nota: erro de pesos e teste linear sintético não confirmam a qualidade linguística do modelo inteiro.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.", file=sys.stderr)
        raise SystemExit(130)
