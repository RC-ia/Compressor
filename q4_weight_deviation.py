#!/usr/bin/env python3
"""Locate individual BF16 -> Q4_0 weight deviations; avoids aggregate-only metrics."""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

import compress_tensor as base

BLOCK = 32


def q4_0_reconstruct_blocks(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Q4_0 reconstructed weights and decoded FP16 block scales."""
    blocks = np.asarray(source, dtype=np.float32).reshape(-1, BLOCK)
    max_pos = np.argmax(np.abs(blocks), axis=1)
    signed_max = blocks[np.arange(blocks.shape[0]), max_pos]
    scale_fp32 = signed_max / -8.0
    inverse = np.zeros_like(scale_fp32, dtype=np.float32)
    np.divide(1.0, scale_fp32, out=inverse, where=scale_fp32 != 0)
    codes = np.clip(
        np.floor(blocks * inverse[:, None] + 8.5), 0, 15
    ).astype(np.uint8)
    # Q4_0 stores the scale as FP16. Reconstruct from that stored value.
    scale_stored = scale_fp32.astype(np.float16)
    scale_decoded = scale_stored.astype(np.float32)
    reconstructed = (
        (codes.astype(np.float32) - 8.0) * scale_decoded[:, None]
    ).reshape(-1)
    return reconstructed, np.abs(scale_decoded)


def coordinate(index: int, shape: tuple[int, ...]) -> str:
    if not shape:
        return "[]"
    coords = np.unravel_index(index, shape)
    return "[" + ",".join(str(int(x)) for x in coords) + "]"


def weight_record(
    tensor_name: str,
    shape: tuple[int, ...],
    source_dtype: str,
    flat_index: int,
    original: float,
    reconstructed: float,
    signed_error: float,
    scale_step: float,
) -> dict[str, Any]:
    return {
        "tensor": tensor_name,
        "shape": list(shape),
        "source_dtype": source_dtype,
        "flat_index": int(flat_index),
        "coordinates": coordinate(flat_index, shape),
        "block_index": int(flat_index // BLOCK),
        "position_in_block": int(flat_index % BLOCK),
        "bf16_original": float(original),
        "q4_reconstructed": float(reconstructed),
        "delta_q4_minus_bf16": float(signed_error),
        "absolute_error": float(abs(signed_error)),
        "relative_error_percent": (
            float(abs(signed_error) / abs(original) * 100.0)
            if original != 0 else None
        ),
        "q4_scale_step": float(scale_step),
        "error_in_q4_steps": (
            float(abs(signed_error) / scale_step) if scale_step else 0.0
        ),
    }


WEIGHT_FIELDS = [
    "rank", "tensor", "shape", "source_dtype", "flat_index", "coordinates",
    "block_index", "position_in_block", "bf16_original", "q4_reconstructed",
    "delta_q4_minus_bf16", "absolute_error", "relative_error_percent",
    "q4_scale_step", "error_in_q4_steps",
    "block_start_flat_index", "block_start_coordinates",
]


def csv_weight_row(rank: int, record: dict[str, Any]) -> dict[str, Any]:
    row = dict(record)
    row["rank"] = rank
    row["shape"] = json.dumps(row["shape"], separators=(",", ":"))
    return row


def write_weight_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=WEIGHT_FIELDS)
        writer.writeheader()
        for rank, record in enumerate(records, 1):
            writer.writerow(csv_weight_row(rank, record))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize BF16 Safetensors weights to blockwise GGML Q4_0 and locate "
            "the exact weights/blocks with the largest reconstruction errors."
        )
    )
    parser.add_argument(
        "--model", default=base.DEFAULT_MODEL,
        help="Checkpoint directory, Safetensors file, or Hugging Face model ID",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--tensor-name", default=None,
        help="Analyze only this exact tensor name; default analyzes all eligible tensors",
    )
    parser.add_argument("--top-k", type=int, default=1000,
                        help="How many worst individual weights and blocks to save")
    parser.add_argument("--chunk-elements", type=int, default=1_048_576,
                        help="Approximate number of source values loaded per chunk")
    parser.add_argument("--output-dir", default="q4_weight_deviation_results")
    parser.add_argument(
        "--dump-all-weights", action="store_true",
        help="Write every weight/error to CSV; requires --tensor-name (can create a large file)",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.chunk_elements < BLOCK:
        parser.error("--chunk-elements must be at least 32")
    if args.dump_all_weights and not args.tensor_name:
        parser.error("--dump-all-weights requires --tensor-name to avoid enormous full-model CSVs")

    files = base.resolve_model(args.model, args.cache_dir, args.revision)
    inventory = base.list_tensors(files)
    if args.tensor_name:
        inventory = [item for item in inventory if item["name"] == args.tensor_name]
        if not inventory:
            raise KeyError(f"Tensor não encontrado: {args.tensor_name}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Min-heaps retain the globally largest absolute errors without holding all weights in RAM.
    worst_weights: list[tuple[float, int, dict[str, Any]]] = []
    worst_blocks: list[tuple[float, int, dict[str, Any]]] = []
    tensor_maxima: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total_quantized_weights = 0
    total_nonzero_errors = 0
    sequence = 0
    dump_stream = None
    dump_writer = None
    dump_path = output_dir / "all_weight_errors.csv"

    if args.dump_all_weights:
        dump_stream = dump_path.open("w", newline="", encoding="utf-8-sig")
        dump_writer = csv.writer(dump_stream)
        dump_writer.writerow([
            "flat_index", "coordinates", "block_index", "position_in_block",
            "bf16_original", "q4_reconstructed", "delta_q4_minus_bf16",
            "absolute_error", "relative_error_percent", "q4_scale_step",
            "error_in_q4_steps",
        ])

    try:
        for tensor_i, item in enumerate(inventory, 1):
            name = item["name"]
            shape = tuple(int(x) for x in item["shape"])
            numel = int(item["numel"])
            row_width = shape[-1] if shape else 0
            reason = None
            if not shape or numel % BLOCK:
                reason = "tensor size is not divisible by Q4_0 block size 32"
            elif len(shape) > 1 and row_width % BLOCK:
                reason = "last dimension is not divisible by 32; Q4_0 row blocks cannot be aligned safely"
            if reason:
                skipped.append({
                    "tensor": name,
                    "shape": list(shape),
                    "num_weights": numel,
                    "reason": reason,
                })
                continue

            print(
                f"[{tensor_i}/{len(inventory)}] {name} | shape={shape} | "
                f"weights={numel:,}",
                flush=True,
            )
            tensor_worst: dict[str, Any] | None = None
            tensor_max_error = -1.0
            tensor_nonzero = 0
            seen = 0
            carry = np.empty(0, dtype=np.float32)
            source_dtype = "unknown"

            with safe_open(str(item["file"]), framework="pt", device="cpu") as sf:
                sliced = sf.get_slice(name)
                first_dim = shape[0]
                inner = int(math.prod(shape[1:])) if len(shape) > 1 else 1
                rows_per_chunk = max(1, args.chunk_elements // max(1, inner))

                for row_start in range(0, first_dim, rows_per_chunk):
                    row_end = min(first_dim, row_start + rows_per_chunk)
                    part = sliced[row_start:row_end]
                    source_dtype = str(part.dtype).replace("torch.", "")
                    values = part.detach().cpu().float().numpy().reshape(-1)
                    combined_start = seen - carry.size
                    combined = np.concatenate((carry, values)) if carry.size else values
                    seen += values.size
                    usable = (combined.size // BLOCK) * BLOCK
                    if usable == 0:
                        carry = combined.copy()
                        continue

                    source = combined[:usable].astype(np.float32, copy=False)
                    carry = combined[usable:].copy()
                    reconstructed, scales = q4_0_reconstruct_blocks(source)
                    errors = reconstructed - source
                    abs_errors = np.abs(errors)
                    block_source = source.reshape(-1, BLOCK)
                    block_recon = reconstructed.reshape(-1, BLOCK)
                    block_errors = errors.reshape(-1, BLOCK)
                    block_abs_errors = abs_errors.reshape(-1, BLOCK)
                    block_scales = scales.reshape(-1)

                    total_quantized_weights += int(source.size)
                    tensor_nonzero += int(np.count_nonzero(abs_errors))
                    total_nonzero_errors += int(np.count_nonzero(abs_errors))

                    # Optional exact per-weight file for one selected tensor.
                    if dump_writer is not None and name == args.tensor_name:
                        for local in range(source.size):
                            flat_index = combined_start + local
                            err = float(errors[local])
                            scale = float(scales[local // BLOCK])
                            orig = float(source[local])
                            dump_writer.writerow([
                                flat_index, coordinate(flat_index, shape),
                                flat_index // BLOCK, flat_index % BLOCK,
                                orig, float(reconstructed[local]), err, abs(err),
                                abs(err) / abs(orig) * 100.0 if orig else "",
                                scale, abs(err) / scale if scale else 0.0,
                            ])

                    # Retain the strongest per-weight deviations from this chunk.
                    take = min(args.top_k, abs_errors.size)
                    candidate_indices = (
                        np.argpartition(abs_errors, abs_errors.size - take)[-take:]
                        if take < abs_errors.size else np.arange(abs_errors.size)
                    )
                    for local in candidate_indices:
                        local_i = int(local)
                        flat_index = combined_start + local_i
                        block_i = local_i // BLOCK
                        rec = weight_record(
                            name, shape, source_dtype, flat_index,
                            float(source[local_i]), float(reconstructed[local_i]),
                            float(errors[local_i]), float(scales[block_i]),
                        )
                        err_value = rec["absolute_error"]
                        sequence += 1
                        node = (err_value, sequence, rec)
                        if len(worst_weights) < args.top_k:
                            heapq.heappush(worst_weights, node)
                        elif err_value > worst_weights[0][0]:
                            heapq.heapreplace(worst_weights, node)

                    # Retain blocks ranked by their worst individual weight, not by their mean.
                    block_max_pos = np.argmax(block_abs_errors, axis=1)
                    block_max_errors = block_abs_errors[
                        np.arange(block_abs_errors.shape[0]), block_max_pos
                    ]
                    take_blocks = min(args.top_k, block_max_errors.size)
                    chosen_blocks = (
                        np.argpartition(block_max_errors, block_max_errors.size - take_blocks)[-take_blocks:]
                        if take_blocks < block_max_errors.size
                        else np.arange(block_max_errors.size)
                    )
                    for b in chosen_blocks:
                        b_i = int(b)
                        pos = int(block_max_pos[b_i])
                        flat_index = combined_start + b_i * BLOCK + pos
                        rec = weight_record(
                            name, shape, source_dtype, flat_index,
                            float(block_source[b_i, pos]),
                            float(block_recon[b_i, pos]),
                            float(block_errors[b_i, pos]),
                            float(block_scales[b_i]),
                        )
                        rec["block_start_flat_index"] = int(combined_start + b_i * BLOCK)
                        rec["block_start_coordinates"] = coordinate(
                            combined_start + b_i * BLOCK, shape
                        )
                        sequence += 1
                        node = (rec["absolute_error"], sequence, rec)
                        if len(worst_blocks) < args.top_k:
                            heapq.heappush(worst_blocks, node)
                        elif node[0] > worst_blocks[0][0]:
                            heapq.heapreplace(worst_blocks, node)

                    local_flat = int(np.argmax(abs_errors))
                    local_max = float(abs_errors[local_flat])
                    if local_max > tensor_max_error:
                        tensor_max_error = local_max
                        flat_index = combined_start + local_flat
                        tensor_worst = weight_record(
                            name, shape, source_dtype, flat_index,
                            float(source[local_flat]), float(reconstructed[local_flat]),
                            float(errors[local_flat]), float(scales[local_flat // BLOCK]),
                        )

            if carry.size:
                raise RuntimeError(
                    f"Tensor {name} terminou com {carry.size} valores fora de um bloco Q4_0; "
                    "a contagem deveria ser múltipla de 32."
                )
            if tensor_worst is not None:
                tensor_maxima.append({
                    "tensor": name,
                    "shape": list(shape),
                    "source_dtype": source_dtype,
                    "num_weights": numel,
                    "nonzero_error_weights": tensor_nonzero,
                    "max_absolute_error": tensor_max_error,
                    "worst_weight_flat_index": tensor_worst["flat_index"],
                    "worst_weight_coordinates": tensor_worst["coordinates"],
                    "bf16_original": tensor_worst["bf16_original"],
                    "q4_reconstructed": tensor_worst["q4_reconstructed"],
                    "delta_q4_minus_bf16": tensor_worst["delta_q4_minus_bf16"],
                    "relative_error_percent": tensor_worst["relative_error_percent"],
                    "q4_scale_step": tensor_worst["q4_scale_step"],
                    "error_in_q4_steps": tensor_worst["error_in_q4_steps"],
                })
    finally:
        if dump_stream is not None:
            dump_stream.close()

    top_weights = [item[2] for item in sorted(worst_weights, key=lambda x: x[0], reverse=True)]
    top_blocks = [item[2] for item in sorted(worst_blocks, key=lambda x: x[0], reverse=True)]
    write_weight_csv(output_dir / "top_weight_deviations.csv", top_weights)
    write_weight_csv(output_dir / "top_block_deviations.csv", top_blocks)

    with (output_dir / "worst_weight_per_tensor.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fields = [
            "tensor", "shape", "source_dtype", "num_weights", "nonzero_error_weights",
            "max_absolute_error", "worst_weight_flat_index", "worst_weight_coordinates",
            "bf16_original", "q4_reconstructed", "delta_q4_minus_bf16",
            "relative_error_percent", "q4_scale_step", "error_in_q4_steps",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(tensor_maxima, key=lambda r: r["max_absolute_error"], reverse=True):
            clean = dict(row)
            clean["shape"] = json.dumps(clean["shape"], separators=(",", ":"))
            writer.writerow(clean)

    report = {
        "reference": "Original Safetensors values; BF16 values are converted to FP32 for exact numerical comparison.",
        "quantizer": "Blockwise GGML-style Q4_0 simulation: 32 weights, 4-bit codes, FP16 scale; compare each reconstructed weight directly to its source value.",
        "scope": "All eligible tensors unless --tensor-name selects a single tensor.",
        "aggregates_intentionally_omitted": [
            "No mean error, RMSE, standard deviation, cosine similarity, or average is used to rank the losses."
        ],
        "top_k": int(args.top_k),
        "quantized_tensor_count": len(tensor_maxima),
        "skipped_tensor_count": len(skipped),
        "quantized_weight_count": total_quantized_weights,
        "weights_with_nonzero_absolute_error": total_nonzero_errors,
        "files": {
            "top_weight_deviations.csv": "Worst individual weights across the analyzed model/tensor, ranked by absolute BF16-to-Q4 difference.",
            "top_block_deviations.csv": "Worst 32-weight Q4_0 blocks, ranked by the largest single weight error in each block.",
            "worst_weight_per_tensor.csv": "The single most changed weight in every analyzed tensor.",
            "skipped_tensors.json": "Tensors that cannot be aligned to Q4_0 blocks of 32 without crossing row boundaries.",
            "all_weight_errors.csv": "Present only with --dump-all-weights and --tensor-name; exact per-weight detail for that selected tensor."
        },
        "skipped_tensors": skipped,
        "all_weight_errors_csv": str(dump_path) if args.dump_all_weights else None,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "skipped_tensors.json").write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== MAIORES DESVIOS INDIVIDUAIS BF16 -> Q4_0 ===")
    print(f"Tensores quantizados: {len(tensor_maxima)}")
    print(f"Pesos analisados: {total_quantized_weights:,}")
    print(f"Pesos com diferença numérica: {total_nonzero_errors:,}")
    print(f"Tensores fora do Q4_0 (tamanho não divisível por 32): {len(skipped)}")
    print(f"Top pesos: {output_dir / 'top_weight_deviations.csv'}")
    print(f"Top blocos: {output_dir / 'top_block_deviations.csv'}")
    print(f"Pior peso por tensor: {output_dir / 'worst_weight_per_tensor.csv'}")
    print(f"Relatório: {output_dir / 'report.json'}")
    if args.dump_all_weights:
        print(f"Todos os pesos do tensor escolhido: {dump_path}")
    if top_weights:
        print("\nTop 10:")
        for rank, rec in enumerate(top_weights[:10], 1):
            print(
                f"{rank:>2}. {rec['tensor']} {rec['coordinates']} | "
                f"BF16={rec['bf16_original']:.8g} | Q4={rec['q4_reconstructed']:.8g} | "
                f"delta={rec['delta_q4_minus_bf16']:+.8g} | "
                f"|delta|={rec['absolute_error']:.8g} | "
                f"{rec['error_in_q4_steps']:.3f} passos Q4"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
