#!/usr/bin/env python3
"""Test whether a tiny predictive model can reduce a 4-bit representative map.

The script quantizes one real 2-D weight tensor to up to 16 learned representatives.
It compares a direct 4-bit map with a predictive residual map. The predictor is only
a tiny conditional table: it predicts each code from the matching code immediately
above (or to the left, when the tensor is transposed). Prediction is vectorized by
row during decode. This is a fast baseline before attempting a neural micro-decoder.

All compressed payloads are measured from actual bytes. Decoded indices must match
the original quantized map exactly; weight approximation error is measured separately.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

import compress_tensor as base


def pack_nibbles(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    if values.size and int(values.max()) > 15:
        raise ValueError("Packed nibble map only supports values 0..15")
    if values.size % 2:
        values = np.concatenate((values, np.zeros(1, dtype=np.uint8)))
    return (values[0::2] | (values[1::2] << 4)).astype(np.uint8)


def unpack_nibbles(packed: bytes | np.ndarray, count: int) -> np.ndarray:
    data = np.frombuffer(packed, dtype=np.uint8) if isinstance(packed, bytes) else np.asarray(packed, dtype=np.uint8)
    out = np.empty(data.size * 2, dtype=np.uint8)
    out[0::2] = data & 0x0F
    out[1::2] = data >> 4
    return out[:count].copy()


def shannon_entropy(values: np.ndarray, alphabet_size: int) -> float:
    histogram = np.bincount(np.asarray(values, dtype=np.int64).reshape(-1), minlength=alphabet_size).astype(np.float64)
    probabilities = histogram[histogram > 0] / max(1, histogram.sum())
    return float(-np.sum(probabilities * np.log2(probabilities))) if probabilities.size else 0.0


def build_predictor(labels: np.ndarray, group_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit one next-code table from the symbol directly above each position."""
    rows, cols = labels.shape
    previous = np.empty_like(labels, dtype=np.uint8)
    previous[0, :] = group_count  # extra state: start-of-stream/row
    previous[1:, :] = labels[:-1, :]
    contexts = previous.astype(np.int64).reshape(-1)
    targets = labels.astype(np.int64).reshape(-1)
    counts = np.bincount(contexts * group_count + targets,
                         minlength=(group_count + 1) * group_count)
    counts = counts.reshape(group_count + 1, group_count)
    predictor = counts.argmax(axis=1).astype(np.uint8)

    predicted = np.empty_like(labels, dtype=np.uint8)
    predicted[0, :] = predictor[group_count]
    predicted[1:, :] = predictor[labels[:-1, :]]
    return predictor, predicted


def make_residuals(labels: np.ndarray, predictor: np.ndarray, group_count: int) -> tuple[np.ndarray, float]:
    predicted = np.empty_like(labels, dtype=np.uint8)
    predicted[0, :] = predictor[group_count]
    predicted[1:, :] = predictor[labels[:-1, :]]
    accuracy = float(np.mean(predicted == labels))
    residuals = ((labels.astype(np.int16) - predicted.astype(np.int16)) % group_count).astype(np.uint8)
    return residuals, accuracy


def decode_predictive_map(residuals: np.ndarray, predictor: np.ndarray, group_count: int) -> np.ndarray:
    """Decode each row in parallel; only rows depend sequentially on the previous row."""
    result = np.empty_like(residuals, dtype=np.uint8)
    for row in range(result.shape[0]):
        prior = np.full(result.shape[1], predictor[group_count], dtype=np.uint8) if row == 0 else predictor[result[row - 1]]
        result[row] = ((prior.astype(np.uint16) + residuals[row].astype(np.uint16)) % group_count).astype(np.uint8)
    return result


def _predictive_candidate(labels: np.ndarray, group_count: int) -> dict[str, Any]:
    predictor, predicted = build_predictor(labels, group_count)
    accuracy = float(np.mean(predicted == labels))
    residuals = ((labels.astype(np.int16) - predicted.astype(np.int16)) % group_count).astype(np.uint8)
    packed_residuals = pack_nibbles(residuals)
    payload = zlib.compress(packed_residuals.tobytes(), level=9)
    return {
        "predictor": predictor,
        "residuals": residuals,
        "payload": payload,
        "prediction_accuracy": accuracy,
        "residual_entropy_bits_per_weight": shannon_entropy(residuals, group_count),
    }


def save_archive(
    path: Path,
    codebook: np.ndarray,
    predictor: np.ndarray,
    payload: bytes,
    metadata: dict[str, Any],
) -> int:
    np.savez(
        path,
        codebook=np.asarray(codebook, dtype=np.float16),
        predictor=np.asarray(predictor, dtype=np.uint8),
        payload=np.frombuffer(payload, dtype=np.uint8),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return path.stat().st_size


def load_archive(path: Path) -> tuple[np.ndarray, np.ndarray, bytes, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        codebook = archive["codebook"].astype(np.float32)
        predictor = archive["predictor"].astype(np.uint8)
        payload = archive["payload"].astype(np.uint8).tobytes()
        metadata = json.loads(str(archive["metadata_json"].item()))
    return codebook, predictor, payload, metadata


def decode_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    codebook, predictor, payload, metadata = load_archive(path)
    count = int(metadata["num_weights"])
    oriented_shape = tuple(int(value) for value in metadata["oriented_shape"])
    group_count = int(metadata["group_count"])
    residual_bytes = zlib.decompress(payload)
    residuals = unpack_nibbles(residual_bytes, count).reshape(oriented_shape)
    decoded_oriented = decode_predictive_map(residuals, predictor, group_count)
    if metadata["orientation"] == "horizontal":
        indices = decoded_oriented.T.copy()
    else:
        indices = decoded_oriented
    return indices, codebook


def measure_reconstruction(original: np.ndarray, indices: np.ndarray, codebook: np.ndarray, batch_rows: int = 64) -> dict[str, float]:
    count = int(original.size)
    squared_error = 0.0
    absolute_error = 0.0
    dot = 0.0
    norm_original = 0.0
    norm_reconstructed = 0.0
    maximum_error = 0.0
    for start in range(0, original.shape[0], batch_rows):
        end = min(original.shape[0], start + batch_rows)
        source = np.asarray(original[start:end], dtype=np.float64)
        reconstructed = codebook[indices[start:end]].astype(np.float64)
        difference = reconstructed - source
        squared_error += float(np.sum(difference * difference))
        absolute_error += float(np.sum(np.abs(difference)))
        dot += float(np.sum(source * reconstructed))
        norm_original += float(np.sum(source * source))
        norm_reconstructed += float(np.sum(reconstructed * reconstructed))
        maximum_error = max(maximum_error, float(np.max(np.abs(difference))))
    rmse = math.sqrt(squared_error / max(1, count))
    std = float(np.std(original, dtype=np.float64))
    cosine = dot / math.sqrt(norm_original * norm_reconstructed) if norm_original and norm_reconstructed else 0.0
    return {
        "rmse": rmse,
        "rmse_over_original_std": rmse / std if std else 0.0,
        "relative_l2_error": math.sqrt(squared_error / norm_original) if norm_original else 0.0,
        "cosine_similarity": cosine,
        "mean_absolute_error": absolute_error / max(1, count),
        "max_absolute_error": maximum_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=base.DEFAULT_MODEL, help="Diretório local, arquivo Safetensors ou ID do Hugging Face")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--tensor-name", default=None, help="Nome exato da matriz 2D")
    parser.add_argument("--max-auto-elements", type=int, default=20_000_000)
    parser.add_argument("--list-tensors", action="store_true")
    parser.add_argument("--groups", type=int, default=16, help="Representantes compartilhados; este teste exige no máximo 16")
    parser.add_argument("--sample-size", type=int, default=250_000)
    parser.add_argument("--assignment-chunk", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="predictive_map_results")
    args = parser.parse_args()
    if not 2 <= args.groups <= 16:
        parser.error("--groups deve estar entre 2 e 16 para usar um mapa de 4 bits.")

    started = time.perf_counter()
    files = base.resolve_model(args.model, args.cache_dir, args.revision)
    if args.list_tensors:
        for item in base.list_tensors(files):
            print(f"{item['name']}\tshape={item['shape']}\tparams={item['numel']:,}\t{item['file'].name}")
        return 0

    selected = base.choose_tensor(files, args.tensor_name, args.max_auto_elements)
    if len(selected["shape"]) != 2:
        parser.error(f"O tensor escolhido deve ser 2D; recebido {selected['shape']}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tensor: {selected['name']}", flush=True)
    print(f"Dimensões: {selected['shape']} | pesos: {selected['numel']:,}", flush=True)
    with safe_open(str(selected["file"]), framework="pt", device="cpu") as source:
        tensor = source.get_tensor(selected["name"])
        source_dtype = str(tensor.dtype).replace("torch.", "")
        source_bytes = int(tensor.numel() * tensor.element_size())
        original = tensor.float().numpy().copy()
        del tensor

    rng = np.random.default_rng(args.seed)
    sample_size = min(int(args.sample_size), original.size)
    sample_positions = rng.choice(original.size, size=sample_size, replace=False)
    centers = base.weighted_kmeans_1d(original.reshape(-1)[sample_positions], args.groups)
    centers = np.sort(np.asarray(centers, dtype=np.float32))
    if centers.size < 2 or centers.size > 16:
        raise RuntimeError(f"Número de representantes inesperado: {centers.size}")
    groups = int(centers.size)
    print(f"Aprendendo {groups} representantes e atribuindo todos os pesos...", flush=True)
    quantize_started = time.perf_counter()
    flat_indices = base.assign_indices(original.reshape(-1), centers, args.assignment_chunk)
    labels = flat_indices.reshape(original.shape).astype(np.uint8, copy=False)
    quantize_seconds = time.perf_counter() - quantize_started
    codebook = centers.astype(np.float16).astype(np.float32)

    print("Medindo o mapa direto e os preditores vertical/horizontal...", flush=True)
    direct_payload = zlib.compress(pack_nibbles(labels).tobytes(), level=9)
    raw_byte_payload = zlib.compress(labels.tobytes(), level=9)
    direct_decode_started = time.perf_counter()
    direct_decoded = unpack_nibbles(zlib.decompress(direct_payload), labels.size).reshape(labels.shape)
    direct_decode_seconds = time.perf_counter() - direct_decode_started
    if not np.array_equal(direct_decoded, labels):
        raise RuntimeError("A decodificação do mapa direto falhou.")

    candidates: list[dict[str, Any]] = []
    for orientation, source_labels in (("vertical", labels), ("horizontal", labels.T.copy())):
        candidate = _predictive_candidate(source_labels, groups)
        candidate["orientation"] = orientation
        candidates.append(candidate)
        print(
            f"  {orientation}: acerto do preditor={candidate['prediction_accuracy'] * 100:.2f}% | "
            f"entropia do resíduo={candidate['residual_entropy_bits_per_weight']:.4f} bits/peso | "
            f"payload={len(candidate['payload']):,} bytes",
            flush=True,
        )
    best = min(candidates, key=lambda item: len(item["payload"]))
    orientation = str(best["orientation"])
    oriented_shape = labels.shape if orientation == "vertical" else labels.T.shape
    metadata = {
        "format": "RC-IA predictive representative map",
        "version": 1,
        "tensor_name": selected["name"],
        "original_shape": list(labels.shape),
        "oriented_shape": list(oriented_shape),
        "source_dtype": source_dtype,
        "source_tensor_bytes": source_bytes,
        "num_weights": int(labels.size),
        "group_count": groups,
        "orientation": orientation,
        "predictor_type": "conditional mode of code immediately above (horizontal mode operates on transposed map)",
        "index_bits": 4,
    }
    artifact_path = output_dir / "predictive_representative_map.npz"
    artifact_bytes = save_archive(artifact_path, codebook, best["predictor"], best["payload"], metadata)

    print("Verificando decodificação a partir do arquivo salvo...", flush=True)
    decode_started = time.perf_counter()
    decoded_indices, decoded_codebook = decode_archive(artifact_path)
    predictive_decode_seconds = time.perf_counter() - decode_started
    if not np.array_equal(decoded_indices, labels):
        raise RuntimeError("Mapa preditivo não foi reconstruído exatamente; arquivo rejeitado.")
    metrics = measure_reconstruction(original, decoded_indices, decoded_codebook)
    residual_entropy = float(best["residual_entropy_bits_per_weight"])
    report: dict[str, Any] = {
        "tensor_name": selected["name"],
        "shape": list(labels.shape),
        "num_weights": int(labels.size),
        "source_dtype": source_dtype,
        "source_tensor_bytes": source_bytes,
        "group_count": groups,
        "codebook_bytes_fp16": int(codebook.astype(np.float16).nbytes),
        "index_entropy_bits_per_weight": shannon_entropy(labels, groups),
        "direct_map_packed_nibbles_zlib_bytes": len(direct_payload),
        "direct_map_byte_symbols_zlib_bytes": len(raw_byte_payload),
        "predictive_orientation": orientation,
        "predictor_accuracy": float(best["prediction_accuracy"]),
        "predictive_residual_entropy_bits_per_weight": residual_entropy,
        "predictive_payload_bytes": len(best["payload"]),
        "predictor_table_bytes": int(best["predictor"].nbytes),
        "artifact_path": str(artifact_path),
        "artifact_bytes": artifact_bytes,
        "artifact_bits_per_weight": artifact_bytes * 8 / labels.size,
        "reduction_vs_source_tensor_percent": 100.0 * (1.0 - artifact_bytes / source_bytes) if source_bytes else 0.0,
        "predictive_payload_saving_vs_direct_map_percent": 100.0 * (1.0 - len(best["payload"]) / len(direct_payload)) if direct_payload else 0.0,
        "direct_decode_seconds": direct_decode_seconds,
        "direct_decode_million_indices_per_second": labels.size / max(direct_decode_seconds, 1e-12) / 1_000_000,
        "predictive_decode_seconds": predictive_decode_seconds,
        "predictive_decode_million_indices_per_second": labels.size / max(predictive_decode_seconds, 1e-12) / 1_000_000,
        "map_round_trip_exact": True,
        "quantize_seconds": quantize_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        **metrics,
    }
    report_path = output_dir / "predictive_map_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== RESULTADO DO MAPA PREDITIVO ===")
    print(f"Arquivo final: {artifact_path} ({artifact_bytes:,} bytes)")
    print(f"Mapa direto comprimido: {len(direct_payload):,} bytes")
    print(f"Mapa preditivo comprimido: {len(best['payload']):,} bytes ({orientation})")
    print(f"Economia adicional no mapa: {report['predictive_payload_saving_vs_direct_map_percent']:.2f}%")
    print(f"Bits efetivos por peso incluindo tabela: {report['artifact_bits_per_weight']:.4f}")
    print(f"Acerto do preditor: {best['prediction_accuracy'] * 100:.2f}%")
    print(f"Entropia do mapa: {report['index_entropy_bits_per_weight']:.4f} bits/peso")
    print(f"Entropia do resíduo: {residual_entropy:.4f} bits/peso")
    print(f"RMSE/std dos pesos: {metrics['rmse_over_original_std']:.6f}")
    print(f"Cosseno dos pesos: {metrics['cosine_similarity']:.6f}")
    print(f"Decodificação direta: {report['direct_decode_million_indices_per_second']:.2f} milhões de índices/s")
    print(f"Decodificação preditiva: {report['predictive_decode_million_indices_per_second']:.2f} milhões de índices/s")
    print(f"Relatório: {report_path}")
    print(f"Tempo total: {report['elapsed_seconds']:.1f}s")
    print("A verificação confirma que o mapa de índices é recuperado sem perdas; a aproximação dos pesos vem dos representantes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
