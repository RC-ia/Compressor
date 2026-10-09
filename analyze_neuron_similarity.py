#!/usr/bin/env python3
"""Find potentially redundant MLP neurons by comparing their three weight vectors.

For a gated MLP, neuron i is characterized by:
  gate_proj[i, :], up_proj[i, :], down_proj[:, i].
The script performs exhaustive pairwise cosine comparisons on all neurons by
default when CUDA is available. A candidate duplicate needs a similar gate vector,
similar up/down directions (allowing up and down to both flip signs), and comparable
scales. This is a weight-only screening test, not proof of activation equivalence.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

import compress_tensor as base

SUFFIXES = {
    "gate": ".gate_proj.weight",
    "up": ".up_proj.weight",
    "down": ".down_proj.weight",
}
THRESHOLDS = (0.90, 0.95, 0.98, 0.99)


def discover_mlp_layers(files: list[Path]) -> list[dict[str, Any]]:
    tensors = base.list_tensors(files)
    grouped: dict[str, dict[str, Any]] = {}
    for item in tensors:
        name = item["name"]
        for kind, suffix in SUFFIXES.items():
            if name.endswith(suffix):
                prefix = name[:-len(suffix)]
                grouped.setdefault(prefix, {})[kind] = item
                break

    result = []
    for prefix, members in grouped.items():
        if set(members) != set(SUFFIXES):
            continue
        layer_match = re.search(r"(?:^|\.)layers\.(\d+)\.mlp(?:\.|$)", prefix)
        result.append({
            "prefix": prefix,
            "layer_index": int(layer_match.group(1)) if layer_match else None,
            "members": members,
        })
    result.sort(key=lambda item: (item["layer_index"] is None, item["layer_index"] or -1, item["prefix"]))
    return result


def load_normalized_neurons(layer: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], tuple[int, int]]:
    """Load and normalize each neuron's weight vectors, one tensor at a time."""
    normalized: dict[str, torch.Tensor] = {}
    norms: dict[str, torch.Tensor] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for kind in ("gate", "up", "down"):
        item = layer["members"][kind]
        with safe_open(str(item["file"]), framework="pt", device="cpu") as sf:
            tensor = sf.get_tensor(item["name"]).to(torch.float32)
        shapes[kind] = tuple(tensor.shape)
        if kind == "down":
            tensor = tensor.transpose(0, 1).contiguous()
        if tensor.ndim != 2:
            raise ValueError(f"{kind} não é uma matriz 2D: {shapes[kind]}")
        norm = torch.linalg.vector_norm(tensor, ord=2, dim=1).clamp_min(1e-12)
        normalized[kind] = (tensor / norm[:, None]).to(device)
        norms[kind] = norm.to(device)
        del tensor
    if shapes["gate"] != shapes["up"]:
        raise ValueError(f"gate_proj {shapes['gate']} e up_proj {shapes['up']} têm formas diferentes.")
    if shapes["down"][1] != shapes["gate"][0] or shapes["down"][0] != shapes["gate"][1]:
        raise ValueError(
            "Formas inesperadas: gate/up devem ser [intermediate, hidden] e "
            f"down [hidden, intermediate]. Recebido gate={shapes['gate']}, down={shapes['down']}."
        )
    return normalized, norms, shapes["gate"]


def cosine_block(
    normalized: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    rows: torch.Tensor,
    all_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return component cosines and scale-agreement ratios for a pair block."""
    gate = normalized["gate"][rows] @ normalized["gate"].T
    up = normalized["up"][rows] @ normalized["up"].T
    down = normalized["down"][rows] @ normalized["down"].T

    gate_norm = norms["gate"]
    effective_norm = norms["up"] * norms["down"]
    gi = gate_norm[rows, None]
    gj = gate_norm[None, :]
    ei = effective_norm[rows, None]
    ej = effective_norm[None, :]
    gate_ratio = torch.minimum(gi, gj) / torch.maximum(gi, gj).clamp_min(1e-12)
    effective_ratio = torch.minimum(ei, ej) / torch.maximum(ei, ej).clamp_min(1e-12)
    return gate, up, down, gate_ratio, effective_ratio


def analyze_similarity(
    normalized: dict[str, torch.Tensor],
    norms: dict[str, torch.Tensor],
    neuron_ids: np.ndarray,
    batch_rows: int,
    scale_tolerance: float,
    top_limit: int,
) -> tuple[dict[str, int], list[dict[str, Any]], float]:
    n = int(normalized["gate"].shape[0])
    counts = {f"pairs_all_components_ge_{int(t * 100)}pct": 0 for t in THRESHOLDS}
    candidates: list[dict[str, Any]] = []
    started = time.perf_counter()
    all_ids = torch.arange(n, device=normalized["gate"].device)

    with torch.inference_mode():
        for start in range(0, n, batch_rows):
            end = min(n, start + batch_rows)
            row_ids = all_ids[start:end]
            cg, cu, cd, gate_ratio, effective_ratio = cosine_block(normalized, norms, row_ids, all_ids)

            upper = all_ids[None, :] > row_ids[:, None]
            sign_compatible = (cu * cd) > 0
            scale_compatible = (gate_ratio >= 1.0 - scale_tolerance) & (
                effective_ratio >= 1.0 - scale_tolerance
            )
            valid_base = upper & sign_compatible & scale_compatible & (cg > 0)
            abs_up, abs_down = cu.abs(), cd.abs()

            for threshold in THRESHOLDS:
                match = (
                    valid_base
                    & (cg >= threshold)
                    & (abs_up >= threshold)
                    & (abs_down >= threshold)
                )
                counts[f"pairs_all_components_ge_{int(threshold * 100)}pct"] += int(match.sum().item())

            # Keep a small shortlist of the best plausible pairs, even if no pair
            # reaches the duplicate threshold. The weakest component defines score.
            quality = torch.minimum(cg, torch.minimum(abs_up, abs_down))
            quality = quality.masked_fill(~valid_base, -2.0)
            k = min(4, n)
            values, indices = torch.topk(quality, k=k, dim=1)
            for local_row in range(end - start):
                for rank in range(k):
                    score = float(values[local_row, rank].item())
                    col = int(indices[local_row, rank].item())
                    if score <= -1.0:
                        continue
                    r = start + local_row
                    # Similarity is symmetric; retain only a unique pair.
                    if r >= col:
                        continue
                    candidates.append({
                        "neuron_a": int(neuron_ids[r]),
                        "neuron_b": int(neuron_ids[col]),
                        "minimum_component_cosine": score,
                        "gate_cosine": float(cg[local_row, col].item()),
                        "up_cosine": float(cu[local_row, col].item()),
                        "down_cosine": float(cd[local_row, col].item()),
                        "gate_norm_ratio": float(gate_ratio[local_row, col].item()),
                        "up_down_product_norm_ratio": float(effective_ratio[local_row, col].item()),
                    })
            # Keep candidate storage bounded while processing thousands of neurons.
            if len(candidates) > top_limit * 10:
                candidates.sort(key=lambda item: item["minimum_component_cosine"], reverse=True)
                candidates = candidates[:top_limit]

    candidates.sort(key=lambda item: item["minimum_component_cosine"], reverse=True)
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for candidate in candidates:
        pair = (candidate["neuron_a"], candidate["neuron_b"])
        if pair not in seen:
            seen.add(pair)
            deduplicated.append(candidate)
        if len(deduplicated) >= top_limit:
            break
    return counts, deduplicated, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=base.DEFAULT_MODEL, help="Diretório local, arquivo Safetensors ou ID do Hugging Face")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--layer-index", type=int, default=0, help="Índice de camada MLP; padrão 0")
    parser.add_argument("--list-layers", action="store_true", help="Listar camadas que contêm gate/up/down projections")
    parser.add_argument("--batch-rows", type=int, default=128, help="Número de neurônios por bloco de comparação")
    parser.add_argument("--max-neurons", type=int, default=0,
                        help="Limitar análise a uma amostra aleatória (0 = todos na GPU; na CPU, padrão automático 1024)")
    parser.add_argument("--scale-tolerance", type=float, default=0.10,
                        help="Diferença relativa máxima nas escalas para considerar dois neurônios candidatos")
    parser.add_argument("--top", type=int, default=100, help="Número de pares candidatos a salvar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="neuron_similarity_results")
    args = parser.parse_args()

    if args.batch_rows < 1 or args.top < 1 or not 0 <= args.scale_tolerance < 1:
        parser.error("batch-rows e top precisam ser positivos; scale-tolerance deve estar em [0,1).")

    started = time.perf_counter()
    files = base.resolve_model(args.model, args.cache_dir, args.revision)
    layers = discover_mlp_layers(files)
    if not layers:
        print("Não encontrei conjuntos completos de gate_proj/up_proj/down_proj. Use --list-layers para inspecionar.", file=sys.stderr)
        return 2
    if args.list_layers:
        for layer in layers:
            gate_shape = tuple(layer["members"]["gate"]["shape"])
            print(f"layer={layer['layer_index']} prefix={layer['prefix']} gate_shape={gate_shape}")
        return 0

    matches = [layer for layer in layers if layer["layer_index"] == args.layer_index]
    if not matches:
        print(f"Camada MLP {args.layer_index} não encontrada. Camadas disponíveis:", file=sys.stderr)
        for layer in layers:
            print(f"  layer={layer['layer_index']} prefix={layer['prefix']}", file=sys.stderr)
        return 2
    layer = matches[0]

    requested_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalized, norms, shape = load_normalized_neurons(layer, requested_device)
    total_neurons = int(shape[0])
    neuron_ids = np.arange(total_neurons, dtype=np.int64)

    max_neurons = args.max_neurons
    if max_neurons < 0:
        parser.error("--max-neurons não pode ser negativo")
    if max_neurons == 0 and requested_device.type == "cpu" and total_neurons > 1024:
        max_neurons = 1024
        print("[AVISO] CPU detectada: para limitar o cálculo, a análise usará uma amostra de 1024 neurônios.", flush=True)
        print("Use --max-neurons 0 com CUDA disponível para analisar todos os neurônios.", flush=True)
    if max_neurons and max_neurons < total_neurons:
        rng = np.random.default_rng(args.seed)
        neuron_ids = np.sort(rng.choice(total_neurons, size=max_neurons, replace=False))
        row_index = torch.as_tensor(neuron_ids, dtype=torch.long, device=requested_device)
        normalized = {key: value[row_index] for key, value in normalized.items()}
        norms = {key: value[row_index] for key, value in norms.items()}
        print(f"[AVISO] Analisando amostra de {len(neuron_ids)} entre {total_neurons} neurônios.", flush=True)

    n = len(neuron_ids)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Camada: {layer['prefix']}", flush=True)
    print(f"Tensor gate/up: {shape} | neurônios analisados: {n:,} de {total_neurons:,}", flush=True)
    print(f"Dispositivo: {requested_device} | bloco: {args.batch_rows}", flush=True)
    print("Comparando cossenos de gate/up/down, compatibilidade de escala e sinal...", flush=True)

    counts, top_pairs, compare_seconds = analyze_similarity(
        normalized, norms, neuron_ids, args.batch_rows, args.scale_tolerance, args.top
    )
    report = {
        "layer_prefix": layer["prefix"],
        "layer_index": layer["layer_index"],
        "hidden_size": int(shape[1]),
        "intermediate_size": int(shape[0]),
        "total_neurons_in_layer": total_neurons,
        "neurons_compared": int(n),
        "sampling_used": bool(n < total_neurons),
        "device": str(requested_device),
        "batch_rows": args.batch_rows,
        "scale_tolerance": args.scale_tolerance,
        "threshold_pair_counts": counts,
        "top_candidate_pairs": top_pairs,
        "comparison_seconds": compare_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "method": "weight-vector screening; not activation-level functional equivalence",
    }
    report_path = output_dir / "neuron_similarity_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SIMILARIDADE ENTRE NEURÔNIOS ===")
    print(f"Neurônios comparados: {n:,}/{total_neurons:,}")
    for threshold in THRESHOLDS:
        count = counts[f"pairs_all_components_ge_{int(threshold * 100)}pct"]
        print(f"Pares com gate/up/down >= {threshold:.2f} e escalas compatíveis: {count:,}")
    if top_pairs:
        print("\nMelhores candidatos (cossenos independentes):")
        for item in top_pairs[:min(10, len(top_pairs))]:
            print(
                f"  {item['neuron_a']} ↔ {item['neuron_b']} | min={item['minimum_component_cosine']:.4f} "
                f"gate={item['gate_cosine']:.4f} up={item['up_cosine']:.4f} down={item['down_cosine']:.4f} "
                f"escala_gate={item['gate_norm_ratio']:.3f} escala_up*down={item['up_down_product_norm_ratio']:.3f}"
            )
    else:
        print("Nenhum par candidato passou pelos filtros preliminares.")
    print(f"Comparação: {compare_seconds:.1f}s")
    print(f"Relatório: {report_path}")
    print("Nota: similaridade entre pesos é um filtro inicial; confirmar equivalência exige comparar ativações em dados reais.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
