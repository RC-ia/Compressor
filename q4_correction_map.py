#!/usr/bin/env python3
"""Build sparse per-weight correction maps for simulated Q4_0 from source Safetensors."""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

import compress_tensor as base

BLOCK = 32
DEFAULT_THRESHOLDS = (1.0, 0.8, 0.6, 0.5)


def q4_0_reconstruct_blocks(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocks = np.asarray(source, dtype=np.float32).reshape(-1, BLOCK)
    max_pos = np.argmax(np.abs(blocks), axis=1)
    signed_max = blocks[np.arange(blocks.shape[0]), max_pos]
    scale = signed_max / -8.0
    inverse = np.zeros_like(scale, dtype=np.float32)
    np.divide(1.0, scale, out=inverse, where=scale != 0)
    codes = np.clip(np.floor(blocks * inverse[:, None] + 8.5), 0, 15).astype(np.uint8)
    decoded_scale = scale.astype(np.float16).astype(np.float32)
    restored = ((codes.astype(np.float32) - 8.0) * decoded_scale[:, None]).reshape(-1)
    return restored, np.repeat(np.abs(decoded_scale), BLOCK)


def write_uvarint(stream, value: int) -> int:
    """Write unsigned LEB128; return encoded byte count."""
    if value < 0:
        raise ValueError("varint must be non-negative")
    count = 0
    while value >= 0x80:
        stream.write(bytes([(value & 0x7F) | 0x80]))
        value >>= 7
        count += 1
    stream.write(bytes([value]))
    return count + 1


def parse_thresholds(value: str) -> list[float]:
    values = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise argparse.ArgumentTypeError("thresholds must be comma-separated finite non-negative numbers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("thresholds must be unique")
    return sorted(values, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare source weights against reconstructed Q4_0 and write sparse FP16-residual maps."
    )
    parser.add_argument("--model", default=base.DEFAULT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--thresholds", type=parse_thresholds,
                        default=list(DEFAULT_THRESHOLDS),
                        help="Absolute-error thresholds, comma-separated (default: 1.0,0.8,0.6,0.5)")
    parser.add_argument("--chunk-elements", type=int, default=1_048_576)
    parser.add_argument("--output-dir", default="q4_correction_map_results")
    args = parser.parse_args()

    if args.chunk_elements < BLOCK:
        parser.error("--chunk-elements must be at least 32")

    files = base.resolve_model(args.model, args.cache_dir, args.revision)
    inventory = base.list_tensors(files)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    thresholds = args.thresholds
    streams = {}
    map_paths = {}
    stats = {
        t: {
            "threshold": t, "correction_count": 0, "blocks_affected": 0,
            "tensors_affected": 0, "map_bytes": 0, "index_bytes": 0,
            "residual_bytes": 0, "max_error_before": 0.0,
            "max_error_after_fp16_residual": 0.0,
            "corrected_weights_still_nonzero_error": 0,
        } for t in thresholds
    }
    tensor_manifest = []
    skipped = []
    global_offset = 0
    previous_global_index = {t: -1 for t in thresholds}
    affected_tensor_ids = {t: set() for t in thresholds}
    affected_blocks = {t: set() for t in thresholds}
    corrected_error_max = {t: 0.0 for t in thresholds}
    corrected_nonzero = {t: 0 for t in thresholds}

    for t in thresholds:
        tag = str(t).replace(".", "p")
        path = output / f"correction_map_gt_{tag}.bin"
        map_paths[t] = path
        streams[t] = path.open("wb")

    try:
        for tensor_id, item in enumerate(inventory):
            name = item["name"]
            shape = tuple(int(x) for x in item["shape"])
            numel = int(item["numel"])
            tensor_offset = global_offset
            global_offset += numel

            eligible = bool(shape) and numel % BLOCK == 0 and not (len(shape) > 1 and shape[-1] % BLOCK)
            tensor_manifest.append({
                "tensor_id": tensor_id, "tensor": name, "shape": list(shape),
                "num_weights": numel, "global_offset": tensor_offset,
                "source_dtype": "recorded_during_scan",
                "q4_0_eligible": eligible,
            })
            if not eligible:
                skipped.append({
                    "tensor_id": tensor_id, "tensor": name, "shape": list(shape),
                    "num_weights": numel,
                    "global_offset": tensor_offset,
                    "reason": "cannot align Q4_0 blocks of 32 within tensor rows",
                })
                continue

            tensor_counts = {t: 0 for t in thresholds}
            tensor_blocks = {t: set() for t in thresholds}
            max_before = 0.0
            max_after = {t: 0.0 for t in thresholds}
            nonzero_after = {t: 0 for t in thresholds}
            print(f"[tensor {tensor_id + 1}/{len(inventory)}] {name} {shape}", flush=True)

            with safe_open(str(item["file"]), framework="pt", device="cpu") as sf:
                sliced = sf.get_slice(name)
                first_dim = shape[0]
                inner = math.prod(shape[1:]) if len(shape) > 1 else 1
                rows_per_chunk = max(1, args.chunk_elements // max(1, inner))
                seen = 0
                carry = np.empty(0, dtype=np.float32)

                for row_start in range(0, first_dim, rows_per_chunk):
                    row_end = min(first_dim, row_start + rows_per_chunk)
                    part = sliced[row_start:row_end]
                    source_dtype = str(part.dtype).replace("torch.", "")
                    values = part.detach().cpu().float().numpy().reshape(-1)
                    chunk_global_start = tensor_offset + seen - carry.size
                    combined = np.concatenate((carry, values)) if carry.size else values
                    seen += values.size
                    usable = (combined.size // BLOCK) * BLOCK
                    if not usable:
                        carry = combined.copy()
                        continue
                    source = combined[:usable].astype(np.float32, copy=False)
                    carry = combined[usable:].copy()
                    q4, _ = q4_0_reconstruct_blocks(source)
                    delta = source - q4
                    abs_error = np.abs(delta)
                    max_before = max(max_before, float(abs_error.max(initial=0.0)))

                    for t in thresholds:
                        selected = np.flatnonzero(abs_error > t)
                        if selected.size == 0:
                            continue
                        tensor_counts[t] += int(selected.size)
                        for local_raw in selected:
                            local = int(local_raw)
                            global_index = chunk_global_start + local
                            residual = float(delta[local])
                            # Store the residual in FP16; measure the actual residual quantization error.
                            residual16 = np.float16(residual)
                            post_error = float(abs(float(source[local]) - (float(q4[local]) + float(residual16))))
                            corrected_error_max[t] = max(corrected_error_max[t], post_error)
                            max_after[t] = max(max_after[t], post_error)
                            if post_error != 0.0:
                                corrected_nonzero[t] += 1
                                nonzero_after[t] += 1
                            block_global = global_index // BLOCK
                            if block_global not in affected_blocks[t]:
                                affected_blocks[t].add(block_global)
                                tensor_blocks[t].add(block_global)
                            # Sparse record: delta-coded global weight index (unsigned LEB128) + FP16 residual.
                            index_delta = global_index - previous_global_index[t]
                            index_bytes = write_uvarint(streams[t], index_delta)
                            streams[t].write(struct.pack("<e", residual))
                            previous_global_index[t] = global_index
                            stats[t]["correction_count"] += 1
                            stats[t]["index_bytes"] += index_bytes
                            stats[t]["residual_bytes"] += 2
                            stats[t]["map_bytes"] += index_bytes + 2
                        affected_tensor_ids[t].add(tensor_id)

            if carry.size:
                raise RuntimeError(f"{name}: leftover {carry.size} values not aligned to Q4_0 block")
            tensor_manifest[-1]["source_dtype"] = source_dtype if numel else "unknown"
            for t in thresholds:
                if tensor_counts[t]:
                    stats[t]["blocks_affected"] += len(tensor_blocks[t])
                    stats[t]["tensors_affected"] += 1
                stats[t]["max_error_before"] = max(stats[t]["max_error_before"], max_before)
                stats[t]["max_error_after_fp16_residual"] = max(
                    stats[t]["max_error_after_fp16_residual"], max_after[t]
                )
                stats[t]["corrected_weights_still_nonzero_error"] += nonzero_after[t]

    finally:
        for stream in streams.values():
            stream.close()

    # File sizes are read from disk, not estimated.
    for t in thresholds:
        stats[t]["map_bytes"] = map_paths[t].stat().st_size
        stats[t]["map_mib"] = stats[t]["map_bytes"] / (1024 ** 2)
        stats[t]["correction_fraction_percent"] = (
            stats[t]["correction_count"] / max(1, sum(int(x["numel"]) for x in inventory)) * 100
        )
        stats[t]["map_format"] = "sorted delta-coded global flat index (unsigned LEB128) + FP16 residual"
        stats[t]["map_file"] = str(map_paths[t])

    manifest = {
        "source": "Safetensors checkpoint; inspect source_dtype in tensor_manifest.json to confirm BF16",
        "base_quantization": "GGML-style Q4_0 simulation, 32 weights per block, FP16 scale",
        "selection_rule": "Create an entry when abs(source_weight - reconstructed_Q4_weight) > threshold",
        "correction": "FP16 residual = source_weight - reconstructed_Q4_weight",
        "application": "corrected_weight = Q4_weight + FP16_residual at each mapped global flattened index",
        "index_format": "unsigned LEB128 delta from previous global index; first delta is global_index + 1",
        "global_index_layout": "concatenation of tensors in tensor_manifest.json order; each tensor offset and shape are recorded",
        "thresholds": stats,
        "quantized_tensor_count": len(tensor_manifest),
        "skipped_tensor_count": len(skipped),
        "skipped_tensors": skipped,
        "tensor_manifest_file": "tensor_manifest.json",
        "caveat": "This evaluates simulated Q4_0 reconstructed from source tensors; it does not load a separately quantized GGUF file. The residual map's file size excludes the base Q4 model and manifest metadata.",
    }
    (output / "tensor_manifest.json").write_text(
        json.dumps(tensor_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "report.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== MAPA ESPARSO DE CORREÇÕES Q4_0 ===")
    print("Cada entrada guarda índice global delta-coded + resíduo FP16.")
    for t in thresholds:
        s = stats[t]
        print(
            f"> {t:g}: {s['correction_count']:,} pesos | "
            f"{s['blocks_affected']:,} blocos | {s['tensors_affected']:,} tensores | "
            f"mapa={s['map_bytes']:,} bytes ({s['map_mib']:.3f} MiB) | "
            f"erro máximo após correção FP16={s['max_error_after_fp16_residual']:.8g} | "
            f"pesos corrigidos ainda com erro={s['corrected_weights_still_nonzero_error']:,}"
        )
    print(f"Relatório: {output / 'report.json'}")
    print(f"Manifesto de tensores: {output / 'tensor_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
