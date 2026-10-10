#!/usr/bin/env python3
"""Compare 16 matrix-wide representatives with blockwise GGML Q4_0.

The representative method stores a packed 4-bit index per weight plus one
FP16 codebook per matrix. The Q4_0 reference stores 32 4-bit codes and one FP16
scale per block (18 bytes per 32 weights, or 4.5 bits/weight). Both methods are
reconstructed from their packed payloads before numerical errors are measured.

This is a weight-reconstruction experiment, not an inference/quality benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
from safetensors import safe_open

import compress_tensor as base


DEFAULT_TENSOR = "model.language_model.layers.0.mlp.gate_proj.weight"
Q4_BLOCK = 32


def pack_nibbles(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    if values.size and int(values.max()) > 15:
        raise ValueError("Índices de representantes precisam estar entre 0 e 15.")
    if values.size % 2:
        values = np.concatenate((values, np.zeros(1, dtype=np.uint8)))
    return (values[0::2] | (values[1::2] << 4)).astype(np.uint8)


def unpack_nibbles(packed: np.ndarray, count: int) -> np.ndarray:
    data = np.asarray(packed, dtype=np.uint8).reshape(-1)
    values = np.empty(data.size * 2, dtype=np.uint8)
    values[0::2] = data & 0x0F
    values[1::2] = data >> 4
    return values[:count].copy()


def representative_reconstruction(
    packed_indices: np.ndarray,
    codebook: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    first_byte = start // 2
    last_byte = (end + 1) // 2
    packed = packed_indices[first_byte:last_byte]
    indices = np.empty(packed.size * 2, dtype=np.uint8)
    indices[0::2] = packed & 0x0F
    indices[1::2] = packed >> 4
    offset = start - first_byte * 2
    indices = indices[offset:offset + (end - start)]
    return codebook[indices]


def encode_q4_0(values: np.ndarray, chunk_blocks: int = 65_536) -> tuple[np.ndarray, np.ndarray]:
    """Encode the GGML Q4_0 block layout: FP16 scale + sixteen packed bytes."""
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size % Q4_BLOCK:
        raise ValueError(f"Q4_0 exige quantidade de pesos divisível por {Q4_BLOCK}.")
    block_count = flat.size // Q4_BLOCK
    packed = np.empty((block_count, Q4_BLOCK // 2), dtype=np.uint8)
    scales = np.empty(block_count, dtype=np.float16)

    for first in range(0, block_count, chunk_blocks):
        last = min(block_count, first + chunk_blocks)
        blocks = flat[first * Q4_BLOCK:last * Q4_BLOCK].reshape(-1, Q4_BLOCK)
        max_indices = np.argmax(np.abs(blocks), axis=1)
        signed_max = blocks[np.arange(blocks.shape[0]), max_indices]
        # GGML Q4_0 uses d = signed maximum / -8 and q = clamp(int(x/d + 8.5), 0, 15).
        d = signed_max / -8.0
        inverse = np.zeros_like(d, dtype=np.float32)
        np.divide(1.0, d, out=inverse, where=d != 0)
        quants = np.clip(np.floor(blocks * inverse[:, None] + 8.5), 0, 15).astype(np.uint8)
        packed[first:last] = quants[:, :16] | (quants[:, 16:] << 4)
        scales[first:last] = d.astype(np.float16)

    return packed, scales


def q4_0_reconstruction(
    packed: np.ndarray,
    scales: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    if start % Q4_BLOCK or end % Q4_BLOCK:
        raise ValueError("Os limites de reconstrução Q4_0 devem alinhar com blocos de 32.")
    first = start // Q4_BLOCK
    last = end // Q4_BLOCK
    encoded = packed[first:last]
    quantized = np.empty((last - first, Q4_BLOCK), dtype=np.uint8)
    quantized[:, :16] = encoded & 0x0F
    quantized[:, 16:] = encoded >> 4
    scale_values = scales[first:last].astype(np.float32)
    return ((quantized.astype(np.float32) - 8.0) * scale_values[:, None]).reshape(-1)


def measure_reconstruction(
    original: np.ndarray,
    reconstruct: Callable[[int, int], np.ndarray],
    chunk_elements: int,
) -> dict[str, float]:
    squared_error = 0.0
    absolute_error = 0.0
    dot = 0.0
    norm_original = 0.0
    norm_reconstructed = 0.0
    sum_original = 0.0
    sum_original_squared = 0.0
    maximum_error = 0.0
    count = int(original.size)

    for start in range(0, count, chunk_elements):
        end = min(count, start + chunk_elements)
        source = original[start:end]
        approx = np.asarray(reconstruct(start, end), dtype=np.float32).reshape(-1)
        difference = approx - source
        squared_error += float(np.sum(difference * difference, dtype=np.float64))
        absolute_error += float(np.sum(np.abs(difference), dtype=np.float64))
        dot += float(np.sum(source * approx, dtype=np.float64))
        norm_original += float(np.sum(source * source, dtype=np.float64))
        norm_reconstructed += float(np.sum(approx * approx, dtype=np.float64))
        sum_original += float(np.sum(source, dtype=np.float64))
        sum_original_squared += float(np.sum(source * source, dtype=np.float64))
        if difference.size:
            maximum_error = max(maximum_error, float(np.max(np.abs(difference))))

    rmse = math.sqrt(squared_error / max(1, count))
    mean = sum_original / max(1, count)
    original_std = math.sqrt(max(0.0, sum_original_squared / max(1, count) - mean * mean))
    cosine = (
        dot / math.sqrt(norm_original * norm_reconstructed)
        if norm_original > 0 and norm_reconstructed > 0
        else 0.0
    )
    return {
        "rmse": rmse,
        "rmse_over_original_std": rmse / original_std if original_std else 0.0,
        "relative_l2_error": math.sqrt(squared_error / norm_original) if norm_original else 0.0,
        "cosine_similarity": cosine,
        "mean_absolute_error": absolute_error / max(1, count),
        "max_absolute_error": maximum_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=base.DEFAULT_MODEL,
                        help="Diretório local, arquivo Safetensors ou ID do Hugging Face")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--tensor-name", default=DEFAULT_TENSOR,
                        help="Matriz 2D a comparar (padrão: gate_proj da camada 0)")
    parser.add_argument("--list-tensors", action="store_true")
    parser.add_argument("--representatives", type=int, default=16,
                        help="Quantidade de representantes por matriz; máximo 16")
    parser.add_argument("--sample-size", type=int, default=250_000,
                        help="Pesos amostrados para aprender o codebook")
    parser.add_argument("--assignment-chunk", type=int, default=2_000_000)
    parser.add_argument("--evaluation-chunk", type=int, default=1_048_576,
                        help="Tamanho de cada lote de avaliação; deve ser divisível por 32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="representatives_vs_q4_results")
    args = parser.parse_args()

    if not 2 <= args.representatives <= 16:
        parser.error("--representatives deve estar entre 2 e 16 (mapa de 4 bits).")
    if args.sample_size < args.representatives:
        parser.error("--sample-size precisa ser maior ou igual ao número de representantes.")
    if args.evaluation_chunk <= 0 or args.evaluation_chunk % Q4_BLOCK:
        parser.error("--evaluation-chunk precisa ser positivo e divisível por 32.")

    started = time.perf_counter()
    files = base.resolve_model(args.model, args.cache_dir, args.revision)
    if args.list_tensors:
        for item in base.list_tensors(files):
            print(f"{item['name']}\tshape={item['shape']}\tparams={item['numel']:,}\t{item['file'].name}")
        return 0

    selected = base.choose_tensor(files, args.tensor_name, 50_000_000)
    if len(selected["shape"]) != 2:
        parser.error(f"A matriz deve ser 2D; recebido {selected['shape']}.")
    if selected["numel"] % Q4_BLOCK:
        parser.error(f"A matriz tem {selected['numel']} pesos; Q4_0 exige múltiplo de {Q4_BLOCK}.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Matriz: {selected['name']}", flush=True)
    print(f"Dimensões: {selected['shape']} | pesos: {selected['numel']:,}", flush=True)
    with safe_open(str(selected["file"]), framework="pt", device="cpu") as source_file:
        tensor = source_file.get_tensor(selected["name"])
        source_dtype = str(tensor.dtype).replace("torch.", "")
        source_bytes = int(tensor.numel() * tensor.element_size())
        original = tensor.float().numpy().reshape(-1).copy()
        del tensor

    rng = np.random.default_rng(args.seed)
    sample_count = min(int(args.sample_size), original.size)
    sample_positions = rng.choice(original.size, size=sample_count, replace=False)
    print(f"Aprendendo {args.representatives} representantes em {sample_count:,} pesos...", flush=True)
    centers = base.weighted_kmeans_1d(original[sample_positions], args.representatives)
    centers_fp16 = np.unique(np.asarray(centers, dtype=np.float16))
    centers = centers_fp16.astype(np.float32)
    if centers.size < 2 or centers.size > 16:
        raise RuntimeError(f"Codebook inválido após conversão FP16: {centers.size} representantes.")
    del sample_positions

    print("Construindo o mapa de índices de 4 bits...", flush=True)
    indices = base.assign_indices(original, centers, args.assignment_chunk)
    packed_representatives = pack_nibbles(indices)
    round_trip = unpack_nibbles(packed_representatives, original.size)
    if not np.array_equal(indices, round_trip):
        raise RuntimeError("O mapa de representantes não passou na verificação de ida e volta.")
    del indices, round_trip

    print("Codificando a mesma matriz no formato de blocos Q4_0...", flush=True)
    packed_q4, q4_scales = encode_q4_0(original)

    representative_map_bytes = int(packed_representatives.nbytes)
    representative_table_bytes = int(centers_fp16.nbytes)
    representative_payload_bytes = representative_map_bytes + representative_table_bytes
    q4_codes_bytes = int(packed_q4.nbytes)
    q4_scales_bytes = int(q4_scales.nbytes)
    q4_payload_bytes = q4_codes_bytes + q4_scales_bytes

    print("Medindo o erro a partir dos payloads empacotados...", flush=True)
    rep_metrics = measure_reconstruction(
        original,
        lambda start, end: representative_reconstruction(
            packed_representatives, centers, start, end
        ),
        args.evaluation_chunk,
    )
    q4_metrics = measure_reconstruction(
        original,
        lambda start, end: q4_0_reconstruction(packed_q4, q4_scales, start, end),
        args.evaluation_chunk,
    )

    report = {
        "tensor_name": selected["name"],
        "shape": list(selected["shape"]),
        "num_weights": int(original.size),
        "source_dtype": source_dtype,
        "source_tensor_bytes": source_bytes,
        "source_tensor_MB": source_bytes / 1_000_000,
        "source_tensor_MiB": source_bytes / (1024 ** 2),
        "representative_method": {
            "format": "matrix-wide FP16 codebook + packed 4-bit indices",
            "requested_representatives": int(args.representatives),
            "actual_representatives": int(centers.size),
            "sample_size": int(sample_count),
            "map_bytes": representative_map_bytes,
            "codebook_bytes": representative_table_bytes,
            "payload_bytes": representative_payload_bytes,
            "payload_MB": representative_payload_bytes / 1_000_000,
            "payload_MiB": representative_payload_bytes / (1024 ** 2),
            "bits_per_weight_including_codebook": representative_payload_bytes * 8 / original.size,
            **rep_metrics,
        },
        "q4_0_method": {
            "format": "GGML Q4_0 block layout (32 weights, FP16 scale, 16 code bytes)",
            "block_size": Q4_BLOCK,
            "code_bytes": q4_codes_bytes,
            "scale_bytes": q4_scales_bytes,
            "payload_bytes": q4_payload_bytes,
            "payload_MB": q4_payload_bytes / 1_000_000,
            "payload_MiB": q4_payload_bytes / (1024 ** 2),
            "bits_per_weight_including_scales": q4_payload_bytes * 8 / original.size,
            **q4_metrics,
        },
        "representative_payload_saving_vs_q4_0_percent": (
            100.0 * (1.0 - representative_payload_bytes / q4_payload_bytes)
            if q4_payload_bytes else 0.0
        ),
        "representative_rmse_relative_to_q4_0": (
            rep_metrics["rmse"] / q4_metrics["rmse"] if q4_metrics["rmse"] else None
        ),
        "representative_rmse_std_relative_to_q4_0": (
            rep_metrics["rmse_over_original_std"] / q4_metrics["rmse_over_original_std"]
            if q4_metrics["rmse_over_original_std"] else None
        ),
        "seed": int(args.seed),
        "elapsed_seconds": time.perf_counter() - started,
        "scope_note": (
            "Compares numerical weight reconstruction only. Q4_0 is a standard blockwise "
            "baseline; this is not a comparison against Q4_K_M, model perplexity, or inference speed."
        ),
    }
    report_path = output_dir / "representatives_vs_q4_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== REPRESENTANTES POR MATRIZ vs Q4_0 ===")
    print(f"Codebook: {centers.size} representantes FP16")
    print(f"Mapa + tabela: {representative_payload_bytes:,} bytes "
          f"({representative_payload_bytes / 1_000_000:.3f} MB; "
          f"{representative_payload_bytes * 8 / original.size:.4f} bits/peso)")
    print(f"Q4_0: {q4_payload_bytes:,} bytes "
          f"({q4_payload_bytes / 1_000_000:.3f} MB; "
          f"{q4_payload_bytes * 8 / original.size:.4f} bits/peso)")
    print(f"Economia de tamanho dos representantes: "
          f"{report['representative_payload_saving_vs_q4_0_percent']:.2f}%")
    for label, metrics in (("16 representantes", rep_metrics), ("Q4_0", q4_metrics)):
        print(f"\n{label}:")
        print(f"  RMSE/std: {metrics['rmse_over_original_std']:.6f}")
        print(f"  Erro L2 relativo: {metrics['relative_l2_error']:.6f}")
        print(f"  Similaridade cosseno: {metrics['cosine_similarity']:.8f}")
        print(f"  MAE: {metrics['mean_absolute_error']:.8f}")
    print(f"Relatório: {report_path}")
    print(f"Tempo total: {report['elapsed_seconds']:.1f}s")
    print("Nota: Q4_0 é o baseline Q4 em blocos; isso não representa Q4_K_M.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
