#!/usr/bin/env python3
"""Full-checkpoint logarithmic weight codec with streaming encode/decode and metrics.

The compressed archive stores every tensor in a single .rccomp ZIP64 container.
Large floating tensors use formula-generated signed-log quantization; small or
non-floating tensors are preserved exactly. Index payloads are zlib-compressed
inside ZIP_STORED entries to avoid double compression. No full model is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
import zlib
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from huggingface_hub import snapshot_download

import compress_tensor as base

FORMAT_NAME = "RC-IA formula-compressed weights"
FORMAT_VERSION = 1
DEFAULT_LEVELS = 256
DEFAULT_SCALE_MULTIPLIER = 0.75
VERIFY_CHUNK_BYTES = 4 * 1024 * 1024


class MetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.squared_error = 0.0
        self.absolute_error = 0.0
        self.max_abs_error = 0.0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.dot_xy = 0.0

    def update(self, original: np.ndarray, reconstructed: np.ndarray) -> None:
        x = np.asarray(original, dtype=np.float32).reshape(-1).astype(np.float64)
        y = np.asarray(reconstructed, dtype=np.float32).reshape(-1).astype(np.float64)
        if x.size != y.size:
            raise ValueError("Tamanhos divergentes ao calcular métricas.")
        diff = x - y
        self.count += int(x.size)
        self.squared_error += float(np.dot(diff, diff))
        self.absolute_error += float(np.abs(diff).sum())
        if diff.size:
            self.max_abs_error = max(self.max_abs_error, float(np.max(np.abs(diff))))
        self.sum_x += float(x.sum())
        self.sum_x2 += float(np.dot(x, x))
        self.sum_y2 += float(np.dot(y, y))
        self.dot_xy += float(np.dot(x, y))

    def result(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0, "rmse": 0.0, "mae": 0.0, "rmse_over_std": 0.0,
                    "cosine_similarity": 1.0, "max_abs_error": 0.0}
        mse = max(0.0, self.squared_error / self.count)
        mean = self.sum_x / self.count
        variance = max(0.0, self.sum_x2 / self.count - mean * mean)
        std = math.sqrt(variance)
        rmse = math.sqrt(mse)
        denom = math.sqrt(max(0.0, self.sum_x2) * max(0.0, self.sum_y2))
        cosine = self.dot_xy / denom if denom > 0 else 1.0
        return {
            "count": int(self.count),
            "rmse": rmse,
            "mae": self.absolute_error / self.count,
            "max_abs_error": self.max_abs_error,
            "source_weight_std": std,
            "rmse_over_std": rmse / std if std > 0 else 0.0,
            "cosine_similarity": cosine,
        }


def tensor_numel(shape: tuple[int, ...]) -> int:
    return int(math.prod(shape)) if shape else 1


def iter_tensor_chunks(
    sf: Any, tensor_name: str, shape: tuple[int, ...], chunk_elements: int
) -> Iterator[torch.Tensor]:
    """Slice along dimension zero to cap RAM usage for large checkpoint tensors."""
    if not shape:
        yield sf.get_tensor(tensor_name).reshape(-1)
        return
    if any(dimension == 0 for dimension in shape):
        return
    tensor_slice = sf.get_slice(tensor_name)
    trailing_elements = int(math.prod(shape[1:])) if len(shape) > 1 else 1
    rows = max(1, chunk_elements // max(1, trailing_elements))
    for begin in range(0, shape[0], rows):
        yield tensor_slice[begin:min(shape[0], begin + rows)].reshape(-1)


def tensor_dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def cast_reconstruction(values: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    """Model runtime weights are restored to the original per-tensor dtype."""
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(dtype=dtype).float().numpy()


def formula_parameters(
    sf: Any,
    tensor_name: str,
    shape: tuple[int, ...],
    chunk_elements: int,
    sample_size: int,
    multiplier: float,
    seed: int,
) -> dict[str, float]:
    """Stream the tensor, using a deterministic stratified sample for its scale."""
    count = tensor_numel(shape)
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    max_abs = 0.0
    for chunk in iter_tensor_chunks(sf, tensor_name, shape, chunk_elements):
        x = chunk.float().cpu().numpy().reshape(-1)
        if not x.size:
            continue
        if not np.isfinite(x).all():
            raise ValueError(f"Tensor {tensor_name} contém NaN ou infinito; não é seguro quantizá-lo.")
        absolute = np.abs(x)
        max_abs = max(max_abs, float(np.max(absolute)))
        quota = min(x.size, max(1, int(round(sample_size * x.size / max(1, count)))))
        if quota >= x.size:
            selected = absolute
        else:
            selected = absolute[rng.choice(x.size, size=quota, replace=False)]
        samples.append(np.asarray(selected, dtype=np.float32))
    if not samples:
        return {"scale": 1.0, "zmax": 0.0, "sample_median_abs": 0.0, "max_abs": 0.0}
    sample = np.concatenate(samples)
    if sample.size > sample_size:
        sample = sample[rng.choice(sample.size, size=sample_size, replace=False)]
    median_abs = float(np.median(sample))
    if median_abs <= float(np.finfo(np.float32).tiny):
        nonzero = sample[sample > 0]
        if nonzero.size:
            median_abs = float(np.median(nonzero))
        elif max_abs > 0:
            median_abs = max_abs / 100.0
        else:
            median_abs = 1.0
    scale = float(np.float32(max(
        median_abs * multiplier, float(np.finfo(np.float32).tiny)
    )))
    zmax = float(np.float32(math.log1p(max_abs / scale))) if max_abs > 0 else 0.0
    return {"scale": scale, "zmax": zmax, "sample_median_abs": float(np.median(sample)),
            "max_abs": max_abs, "sample_count": int(sample.size)}


def encode_log_indices(values: np.ndarray, scale: float, zmax: float, levels: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if zmax <= 0:
        return np.zeros(x.size, dtype=np.uint8)
    scale32 = np.float32(scale)
    zmax32 = np.float32(zmax)
    transformed = np.sign(x) * np.log1p(np.abs(x) / scale32)
    encoded = np.rint((transformed / zmax32 + 1.0) * (0.5 * (levels - 1)))
    return np.clip(encoded, 0, levels - 1).astype(np.uint8)


def decode_log_indices(indices: np.ndarray, scale: float, zmax: float, levels: int) -> np.ndarray:
    q = np.asarray(indices, dtype=np.uint8).reshape(-1).astype(np.float32)
    if zmax <= 0:
        return np.zeros(q.size, dtype=np.float32)
    z = (q / np.float32(levels - 1) * 2.0 - 1.0) * np.float32(zmax)
    result = np.sign(z) * np.float32(scale) * np.expm1(np.abs(z))
    return result.astype(np.float32)


def original_tensor_bytes(chunk: torch.Tensor) -> bytes:
    """Get exact native storage bytes, including BF16 (not supported by NumPy arrays)."""
    return chunk.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def write_zlib_payload(
    archive: zipfile.ZipFile,
    payload_name: str,
    sf: Any,
    tensor_name: str,
    shape: tuple[int, ...],
    kind: str,
    dtype: torch.dtype,
    levels: int,
    scale: float,
    zmax: float,
    chunk_elements: int,
    metrics: MetricAccumulator | None,
    local_metrics: MetricAccumulator | None,
) -> tuple[str, int, int]:
    """Write a zlib stream to a ZIP_STORED entry and return its uncompressed digest/size."""
    info = zipfile.ZipInfo(payload_name)
    info.compress_type = zipfile.ZIP_STORED
    compressor = zlib.compressobj(level=9)
    digest = hashlib.sha256()
    uncompressed_bytes = 0
    with archive.open(info, mode="w", force_zip64=True) as target:
        for chunk in iter_tensor_chunks(sf, tensor_name, shape, chunk_elements):
            if kind == "log":
                x = chunk.float().cpu().numpy().reshape(-1)
                if not np.isfinite(x).all():
                    raise ValueError(f"Tensor {tensor_name} contém NaN ou infinito; codec interrompido.")
                indices = encode_log_indices(x, scale, zmax, levels)
                raw = indices.tobytes()
                if metrics is not None:
                    reconstructed = cast_reconstruction(
                        decode_log_indices(indices, scale, zmax, levels), dtype
                    )
                    metrics.update(x, reconstructed)
                if local_metrics is not None:
                    local_metrics.update(x, reconstructed)
            else:
                raw = original_tensor_bytes(chunk)
                if metrics is not None and chunk.is_floating_point():
                    x = chunk.float().cpu().numpy().reshape(-1)
                    metrics.update(x, x)
                if local_metrics is not None and chunk.is_floating_point():
                    x = chunk.float().cpu().numpy().reshape(-1)
                    local_metrics.update(x, x)
            digest.update(raw)
            uncompressed_bytes += len(raw)
            compressed = compressor.compress(raw)
            if compressed:
                target.write(compressed)
        tail = compressor.flush()
        if tail:
            target.write(tail)
    return digest.hexdigest(), uncompressed_bytes, archive.getinfo(payload_name).file_size


def iter_zlib_payload(archive: zipfile.ZipFile, payload_name: str,
                      max_output_bytes: int = VERIFY_CHUNK_BYTES) -> Iterator[bytes]:
    """Decompress an inner zlib member with a hard bound on each returned chunk."""
    decoder = zlib.decompressobj()
    with archive.open(payload_name, "r") as source:
        while True:
            compressed = source.read(1024 * 1024)
            if not compressed:
                break
            pending = compressed
            while True:
                decoded = decoder.decompress(pending, max_output_bytes)
                pending = decoder.unconsumed_tail
                if decoded:
                    yield decoded
                if pending:
                    continue
                if len(decoded) == max_output_bytes:
                    pending = b""
                    continue
                break
        tail = decoder.flush()
        if tail:
            yield tail


def verify_payloads(archive_path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify each stored zlib stream end-to-end by its uncompressed length and SHA-256."""
    started = time.perf_counter()
    total_bytes = 0
    with zipfile.ZipFile(archive_path, "r") as archive:
        for number, record in enumerate(records, 1):
            digest = hashlib.sha256()
            count = 0
            for chunk in iter_zlib_payload(archive, record["payload"]):
                digest.update(chunk)
                count += len(chunk)
            if count != record["payload_uncompressed_bytes"]:
                raise ValueError(
                    f"Payload {record['tensor_name']} tem {count} bytes descomprimidos; "
                    f"esperados {record['payload_uncompressed_bytes']}."
                )
            if digest.hexdigest() != record["payload_sha256"]:
                raise ValueError(f"SHA-256 divergente no payload {record['tensor_name']}.")
            if number % 100 == 0:
                print(f"[verify] {number}/{len(records)} tensores verificados", flush=True)
    return {
        "status": "passed",
        "payloads_verified": len(records),
        "uncompressed_payload_bytes_verified": int(sum(r["payload_uncompressed_bytes"] for r in records)),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def compress_full_model(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    source_files = base.resolve_model(args.model, args.cache_dir, args.revision)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if any(output == source.resolve() for source in source_files):
        raise ValueError("O arquivo de saída não pode substituir um arquivo Safetensors de origem.")

    all_metrics = MetricAccumulator()
    records: list[dict[str, Any]] = []
    kind_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    raw_tensor_bytes = 0
    raw_file_bytes = sum(path.stat().st_size for path in source_files)
    tensor_count = 0
    floating_parameter_count = 0
    output_tmp = output.with_name(output.name + ".partial")
    if output_tmp.exists():
        output_tmp.unlink()

    print(f"[source] {len(source_files)} arquivos Safetensors; arquivo(s) fonte: {raw_file_bytes / 1e9:.3f} GB")
    print(f"[config] níveis={args.levels}, escala={args.scale_multiplier}, "
          f"amostra de escala/tensor={args.scale_sample_size:,}, "
          f"bloco de leitura={args.chunk_elements:,}, preservar exatamente <= {args.preserve_small_elements} elementos")

    try:
        with zipfile.ZipFile(output_tmp, mode="w", allowZip64=True) as archive:
            for file_index, path in enumerate(source_files):
                with safe_open(str(path), framework="pt", device="cpu") as sf:
                    for tensor_name in sf.keys():
                        tensor_slice = sf.get_slice(tensor_name)
                        shape = tuple(int(v) for v in tensor_slice.get_shape())
                        numel = tensor_numel(shape)
                        first_chunk = next(iter_tensor_chunks(
                            sf, tensor_name, shape, args.chunk_elements
                        ), None)
                        if first_chunk is None:
                            first_chunk = sf.get_tensor(tensor_name).reshape(-1)
                        dtype_name = tensor_dtype_name(first_chunk)
                        dtype = first_chunk.dtype
                        element_size = int(first_chunk.element_size())
                        source_bytes = int(numel * element_size)
                        is_float = bool(first_chunk.is_floating_point())
                        should_quantize = is_float and numel > args.preserve_small_elements
                        params = {"scale": 1.0, "zmax": 0.0}
                        if should_quantize:
                            params = formula_parameters(
                                sf, tensor_name, shape, args.chunk_elements,
                                args.scale_sample_size, args.scale_multiplier,
                                args.seed + tensor_count,
                            )
                            kind = "log"
                        else:
                            kind = "raw"

                        payload = f"payloads/{tensor_count:08d}.z"
                        local_metrics = MetricAccumulator() if is_float else None
                        global_for_tensor = all_metrics if is_float else None
                        sha256, raw_payload_bytes, payload_zip_bytes = write_zlib_payload(
                            archive, payload, sf, tensor_name, shape, kind, dtype,
                            args.levels, float(params["scale"]), float(params["zmax"]),
                            args.chunk_elements, global_for_tensor, local_metrics,
                        )
                        per_tensor_metrics = local_metrics.result() if local_metrics else None
                        record = {
                            "id": int(tensor_count),
                            "tensor_name": tensor_name,
                            "source_file": path.name,
                            "payload": payload,
                            "kind": kind,
                            "dtype": dtype_name,
                            "shape": list(shape),
                            "numel": int(numel),
                            "element_size": element_size,
                            "source_tensor_bytes": source_bytes,
                            "payload_uncompressed_bytes": raw_payload_bytes,
                            "payload_zlib_bytes": int(payload_zip_bytes),
                            "payload_sha256": sha256,
                            "metrics": per_tensor_metrics,
                        }
                        if kind == "log":
                            record.update({
                                "scale": float(params["scale"]),
                                "zmax": float(params["zmax"]),
                                "scale_sample_median_abs": float(params["sample_median_abs"]),
                                "scale_sample_count": int(params["sample_count"]),
                                "levels": int(args.levels),
                                "scale_multiplier": float(args.scale_multiplier),
                                "index_dtype": "uint8",
                            })
                            floating_parameter_count += numel
                        records.append(record)
                        tensor_count += 1
                        raw_tensor_bytes += source_bytes
                        kind_counts[kind] += 1
                        dtype_counts[dtype_name] += 1
                        dtype_bytes[dtype_name] += source_bytes
                        if tensor_count % args.progress_every == 0:
                            print(
                                f"[encode] {tensor_count} tensores; "
                                f"{raw_tensor_bytes / 1e9:.3f} GB de pesos lidos",
                                flush=True,
                            )

            manifest = {
                "format": FORMAT_NAME,
                "format_version": FORMAT_VERSION,
                "model_source": args.model,
                "quantization": {
                    "method": "signed logarithmic companding",
                    "levels": int(args.levels),
                    "scale_multiplier": float(args.scale_multiplier),
                    "scale_strategy": "per-tensor median absolute weight estimated from a deterministic stratified sample",
                    "index_codec": "zlib level 9, inner payload; outer ZIP entry is stored without a second compression pass",
                    "preserve_exact_if": f"non-floating dtype or tensor numel <= {args.preserve_small_elements}",
                },
                "source": {
                    "safetensors_file_count": len(source_files),
                    "safetensors_file_bytes": int(raw_file_bytes),
                    "tensor_storage_bytes": int(raw_tensor_bytes),
                    "tensor_count": int(tensor_count),
                    "floating_parameter_count_quantized": int(floating_parameter_count),
                    "source_files": [p.name for p in source_files],
                },
                "counts": {"by_representation": dict(kind_counts), "by_dtype": dict(dtype_counts),
                           "source_bytes_by_dtype": dict(dtype_bytes)},
                "global_floating_weight_metrics": all_metrics.result(),
                "tensors": records,
            }
            manifest_bytes = json.dumps(
                manifest, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            archive.writestr("manifest.json", manifest_bytes, compress_type=zipfile.ZIP_DEFLATED)
        output_tmp.replace(output)
    except Exception:
        if output_tmp.exists():
            output_tmp.unlink()
        raise

    archive_bytes = output.stat().st_size
    verification = None
    if not args.skip_verify:
        print("[verify] validando os payloads comprimidos do arquivo completo...", flush=True)
        verification = verify_payloads(output, records)

    report = {
        "status": "full_checkpoint_compressed",
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "model_source": args.model,
        "archive_path": str(output),
        "archive_bytes_actual": int(archive_bytes),
        "archive_GB_decimal": archive_bytes / 1e9,
        "source_safetensors_file_bytes": int(raw_file_bytes),
        "source_safetensors_file_GB": raw_file_bytes / 1e9,
        "source_tensor_storage_bytes": int(raw_tensor_bytes),
        "source_tensor_storage_GB": raw_tensor_bytes / 1e9,
        "saved_GB_vs_tensor_storage": (raw_tensor_bytes - archive_bytes) / 1e9,
        "savings_pct_vs_tensor_storage": (1.0 - archive_bytes / raw_tensor_bytes) * 100.0 if raw_tensor_bytes else 0.0,
        "tensor_count": tensor_count,
        "quantized_tensor_count": int(kind_counts["log"]),
        "exact_tensor_count": int(kind_counts["raw"]),
        "quantized_parameter_count": floating_parameter_count,
        "counts_by_dtype": dict(dtype_counts),
        "source_bytes_by_dtype": dict(dtype_bytes),
        "global_floating_weight_metrics": all_metrics.result(),
        "archive_payload_verification": verification,
        "per_tensor_metrics": [
            {
                "tensor_name": r["tensor_name"],
                "kind": r["kind"],
                "dtype": r["dtype"],
                "numel": r["numel"],
                "payload_MB": r["payload_zlib_bytes"] / 1e6,
                "rmse": None if r["metrics"] is None else r["metrics"]["rmse"],
                "rmse_over_std": None if r["metrics"] is None else r["metrics"]["rmse_over_std"],
                "cosine_similarity": None if r["metrics"] is None else r["metrics"]["cosine_similarity"],
            }
            for r in records
        ],
        "limitations": [
            "O .rccomp contém todos os tensores no formato experimental; não é carregável diretamente pelo Transformers.",
            "A avaliação mede fidelidade numérica dos pesos; não prova perplexidade ou qualidade linguística preservada.",
            "O modelo precisa ser reconstruído para Safetensors ou receber kernels de inferência específicos.",
            "A escala logarítmica é estimada por tensor usando amostragem determinística; não são armazenados representantes individuais.",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = report["global_floating_weight_metrics"]
    print("\n=== COMPRESSÃO DO CHECKPOINT COMPLETO ===")
    print(f"Tensores processados          : {tensor_count}")
    print(f"Tensores quantizados por fórmula: {kind_counts['log']}")
    print(f"Tensores preservados exatos   : {kind_counts['raw']}")
    print(f"Pesos originais               : {raw_tensor_bytes / 1e9:.3f} GB")
    print(f"Arquivo comprimido real       : {archive_bytes / 1e9:.3f} GB")
    print(f"Redução no armazenamento      : {report['savings_pct_vs_tensor_storage']:.2f}%")
    print(f"RMSE global dos pesos         : {metrics['rmse']:.8g}")
    print(f"RMSE / desvio-padrão global   : {metrics['rmse_over_std']:.8g}")
    print(f"Cosseno global dos pesos      : {metrics['cosine_similarity']:.8f}")
    if verification:
        print(f"Verificação do arquivo        : {verification['status']} ({verification['elapsed_seconds']:.1f}s)")
    print(f"Relatório completo            : {report_path}")
    print(f"Tempo total                   : {time.perf_counter() - started:.1f}s")
    return 0


def iter_decoded_tensor(
    archive: zipfile.ZipFile, record: dict[str, Any], max_values: int = 1_000_000
) -> torch.Tensor:
    """Decode a single tensor without materializing its full compressed index stream."""
    dtype = getattr(torch, record["dtype"])
    shape = tuple(int(v) for v in record["shape"])
    numel = int(record["numel"])
    output = torch.empty(numel, dtype=dtype)
    position = 0
    if record["kind"] == "log":
        levels = int(record["levels"])
        scale = float(record["scale"])
        zmax = float(record["zmax"])
        for raw in iter_zlib_payload(archive, record["payload"]):
            indices = np.frombuffer(raw, dtype=np.uint8)
            for start in range(0, indices.size, max_values):
                selected = indices[start:start + max_values]
                values = decode_log_indices(selected, scale, zmax, levels)
                decoded = torch.from_numpy(values.copy()).to(dtype=dtype)
                end = position + decoded.numel()
                if end > numel:
                    raise ValueError(f"Índices excedem a forma declarada de {record['tensor_name']}.")
                output[position:end] = decoded
                position = end
    else:
        element_size = int(record["element_size"])
        leftover = b""
        for raw in iter_zlib_payload(archive, record["payload"]):
            buffer = leftover + raw
            aligned_size = (len(buffer) // element_size) * element_size
            aligned = buffer[:aligned_size]
            leftover = buffer[aligned_size:]
            if aligned:
                values = torch.frombuffer(bytearray(aligned), dtype=dtype).clone()
                end = position + values.numel()
                if end > numel:
                    raise ValueError(f"Bytes excedem a forma declarada de {record['tensor_name']}.")
                output[position:end] = values
                position = end
        if leftover:
            raise ValueError(f"Payload raw desalinhado para {record['tensor_name']}.")
    if position != numel:
        raise ValueError(
            f"Tensor {record['tensor_name']} reconstruído com {position} elementos; esperados {numel}."
        )
    return output.reshape(shape).contiguous()


def source_auxiliary_root(source_model: str, revision: str | None, cache_dir: str | None) -> Path | None:
    candidate = Path(source_model).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if candidate.is_file():
        return candidate.resolve().parent
    try:
        downloaded = snapshot_download(
            repo_id=source_model,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=[
                "*.json", "*.model", "*.tiktoken", "*.txt", "*.jinja", "*.py",
                "*.yaml", "*.yml", "*.merges", "*.vocab",
            ],
        )
        return Path(downloaded)
    except Exception as error:
        print(f"[decode] aviso: não consegui obter os arquivos auxiliares do modelo: {error}", file=sys.stderr)
        return None


def copy_auxiliary_files(source_root: Path | None, output_dir: Path) -> list[str]:
    copied: list[str] = []
    if source_root is None or not source_root.is_dir():
        return copied
    for source in source_root.rglob("*"):
        if not source.is_file():
            continue
        if any(part in {".git", ".cache", "__pycache__"} for part in source.parts):
            continue
        if source.suffix in {".safetensors", ".rccomp"}:
            continue
        if source.name.endswith(".safetensors.index.json") or source.name == "model.safetensors.index.json":
            continue
        try:
            relative = source.relative_to(source_root)
        except ValueError:
            continue
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(str(relative))
    return copied


def decode_full_model(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    archive_path = Path(args.archive).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    with zipfile.ZipFile(archive_path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        records = manifest["tensors"]
        source_model = args.source_model or manifest.get("model_source")
    source_candidate = Path(source_model).expanduser() if source_model else None
    if source_candidate is not None and source_candidate.is_dir() and source_candidate.resolve() == output_dir:
        raise ValueError("O diretório reconstruído precisa ser diferente do diretório do checkpoint original.")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale output shards from earlier decoding attempts, but never touch source weights.
    for stale in output_dir.glob("model-*-of-*.safetensors"):
        stale.unlink()
    for stale in output_dir.glob("model.safetensors"):
        stale.unlink()
    stale_index = output_dir / "model.safetensors.index.json"
    if stale_index.exists():
        stale_index.unlink()
    with zipfile.ZipFile(archive_path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        records = manifest["tensors"]
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_bytes = 0
        max_bytes = int(args.shard_max_mb * 1_000_000)
        for record in records:
            size = int(record["source_tensor_bytes"])
            if current and current_bytes + size > max_bytes:
                groups.append(current)
                current = []
                current_bytes = 0
            current.append(record)
            current_bytes += size
        if current:
            groups.append(current)

        weight_map: dict[str, str] = {}
        total_size = 0
        shard_count = len(groups)
        for shard_index, group in enumerate(groups, 1):
            shard_name = (
                "model.safetensors" if shard_count == 1
                else f"model-{shard_index:05d}-of-{shard_count:05d}.safetensors"
            )
            tensors: dict[str, torch.Tensor] = {}
            shard_size = 0
            for record in group:
                tensor = iter_decoded_tensor(archive, record)
                tensors[record["tensor_name"]] = tensor
                shard_size += int(record["source_tensor_bytes"])
                weight_map[record["tensor_name"]] = shard_name
            shard_path = output_dir / shard_name
            save_file(tensors, str(shard_path), metadata={"source_codec": FORMAT_NAME})
            total_size += shard_size
            del tensors
            print(f"[decode] shard {shard_index}/{shard_count}: {shard_name} ({shard_size / 1e6:.1f} MB)", flush=True)

    if len(groups) > 1:
        index = {"metadata": {"total_size": int(total_size)}, "weight_map": weight_map}
        (output_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    auxiliary_root = source_auxiliary_root(source_model, args.revision, args.cache_dir) if source_model else None
    copied_files = copy_auxiliary_files(auxiliary_root, output_dir)
    report = {
        "status": "full_checkpoint_decoded",
        "archive": str(archive_path),
        "output_dir": str(output_dir),
        "tensor_count": len(records),
        "safetensors_shard_count": len(groups),
        "decoded_tensor_storage_bytes": int(total_size),
        "copied_auxiliary_file_count": len(copied_files),
        "copied_auxiliary_files": copied_files,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "note": "Pesos reconstruídos no dtype original. Execute avaliação de inferência/perplexidade para medir preservação da qualidade.",
    }
    report_path = output_dir / "decode_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== CHECKPOINT RECONSTRUÍDO ===")
    print(f"Tensores                  : {len(records)}")
    print(f"Shards Safetensors        : {len(groups)}")
    print(f"Pesos reconstruídos       : {total_size / 1e9:.3f} GB")
    print(f"Arquivos auxiliares copiados: {len(copied_files)}")
    print(f"Diretório de saída        : {output_dir}")
    print(f"Relatório                 : {report_path}")
    print(f"Tempo total               : {time.perf_counter() - started:.1f}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode = subparsers.add_parser("compress", help="Comprimir todos os tensores do checkpoint")
    encode.add_argument("--model", required=True, help="Repo Hugging Face, diretório local ou Safetensors")
    encode.add_argument("--output", required=True, help="Arquivo .rccomp de saída")
    encode.add_argument("--revision", default=None)
    encode.add_argument("--cache-dir", default=None)
    encode.add_argument("--levels", type=int, choices=[128, 256], default=DEFAULT_LEVELS)
    encode.add_argument("--scale-multiplier", type=float, default=DEFAULT_SCALE_MULTIPLIER)
    encode.add_argument("--scale-sample-size", type=int, default=262_144)
    encode.add_argument("--chunk-elements", type=int, default=2_000_000)
    encode.add_argument("--preserve-small-elements", type=int, default=256)
    encode.add_argument("--seed", type=int, default=42)
    encode.add_argument("--progress-every", type=int, default=25)
    encode.add_argument("--skip-verify", action="store_true",
                        help="Pular a leitura de verificação de todos os payloads após a codificação")
    encode.set_defaults(func=compress_full_model)

    decode = subparsers.add_parser("decode", help="Reconstruir todos os tensores em Safetensors")
    decode.add_argument("--archive", required=True, help="Arquivo .rccomp produzido por compress")
    decode.add_argument("--output-dir", required=True, help="Diretório do checkpoint reconstruído")
    decode.add_argument("--source-model", default=None, help="Diretório local ou ID HF para copiar config/tokenizer")
    decode.add_argument("--revision", default=None)
    decode.add_argument("--cache-dir", default=None)
    decode.add_argument("--shard-max-mb", type=int, default=1024,
                        help="Tamanho alvo máximo por shard; tensores individuais grandes permanecem inteiros")
    decode.set_defaults(func=decode_full_model)

    args = parser.parse_args()
    if getattr(args, "levels", DEFAULT_LEVELS) not in (128, 256):
        parser.error("--levels deve ser 128 ou 256 para este formato uint8.")
    if getattr(args, "scale_multiplier", DEFAULT_SCALE_MULTIPLIER) <= 0:
        parser.error("--scale-multiplier precisa ser positivo.")
    if getattr(args, "chunk_elements", 1) < 1 or getattr(args, "scale_sample_size", 1) < 1:
        parser.error("--chunk-elements e --scale-sample-size precisam ser positivos.")
    return int(args.func(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.", file=sys.stderr)
        raise SystemExit(130)
