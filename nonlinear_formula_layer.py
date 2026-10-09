#!/usr/bin/env python3
"""Fit a learned nonlinear function that generates one matrix of model weights.

Each weight is predicted from a learned row embedding and column embedding:
    W_hat[r,c] = mean + std * (MLP(row_code[r], col_code[c]) + row_bias[r] + col_bias[c])

No per-weight indices or per-weight corrections are stored. The saved artifact contains
only the shared row/column codes, biases, MLP parameters, and normalization metadata.
This is a compression experiment; low weight error does not automatically guarantee
that a language model retains its quality.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from torch import nn

import compress_tensor as base


class NonlinearWeightFormula(nn.Module):
    """A learned nonlinear function shared by all positions in one matrix."""

    def __init__(self, rows: int, cols: int, embedding_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.row_embedding = nn.Embedding(rows, embedding_dim)
        self.col_embedding = nn.Embedding(cols, embedding_dim)
        self.row_bias = nn.Embedding(rows, 1)
        self.col_bias = nn.Embedding(cols, 1)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.row_embedding.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.col_embedding.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.row_bias.weight)
        nn.init.zeros_(self.col_bias.weight)
        nn.init.normal_(self.network[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, row_ids: torch.Tensor, col_ids: torch.Tensor | None = None) -> torch.Tensor:
        if col_ids is None:
            col_ids = torch.arange(self.cols, device=row_ids.device)
        row_code = self.row_embedding(row_ids)
        col_code = self.col_embedding(col_ids)
        row_grid = row_code[:, None, :].expand(-1, col_ids.numel(), -1)
        col_grid = col_code[None, :, :].expand(row_ids.numel(), -1, -1)
        pair_codes = torch.cat((row_grid, col_grid), dim=-1)
        prediction = self.network(pair_codes).squeeze(-1)
        prediction = prediction + self.row_bias(row_ids).squeeze(-1)[:, None]
        prediction = prediction + self.col_bias(col_ids).squeeze(-1)[None, :]
        return prediction


def save_formula(
    path: Path,
    model: NonlinearWeightFormula,
    metadata: dict[str, Any],
) -> int:
    arrays: dict[str, np.ndarray] = {
        f"param__{name}": parameter.detach().to(device="cpu", dtype=torch.float16).numpy()
        for name, parameter in model.state_dict().items()
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(path, **arrays)
    return path.stat().st_size


def load_formula(path: Path, device: torch.device) -> tuple[NonlinearWeightFormula, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        model = NonlinearWeightFormula(
            rows=int(metadata["shape"][0]),
            cols=int(metadata["shape"][1]),
            embedding_dim=int(metadata["embedding_dim"]),
            hidden_dim=int(metadata["hidden_dim"]),
        )
        state: dict[str, torch.Tensor] = {}
        for key in archive.files:
            if key.startswith("param__"):
                name = key[len("param__"):]
                state[name] = torch.from_numpy(archive[key].astype(np.float32))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, metadata


def reconstruct_and_measure(
    model: NonlinearWeightFormula,
    original: np.ndarray,
    mean: float,
    std: float,
    device: torch.device,
    batch_rows: int,
    reconstructed_path: Path | None = None,
) -> dict[str, float]:
    rows, cols = original.shape
    output_map = (
        np.lib.format.open_memmap(reconstructed_path, mode="w+", dtype=np.float32, shape=(rows, cols))
        if reconstructed_path else None
    )
    count = 0
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    dot_sum = 0.0
    original_sq_sum = 0.0
    reconstructed_sq_sum = 0.0
    maximum_absolute_error = 0.0
    model.eval()
    all_rows = torch.arange(rows)
    with torch.inference_mode():
        for start in range(0, rows, batch_rows):
            row_cpu = all_rows[start:start + batch_rows]
            prediction = model(row_cpu.to(device)).float().cpu().numpy()
            prediction = prediction * std + mean
            reference = original[start:start + len(row_cpu)].astype(np.float32, copy=False)
            difference = prediction - reference
            if output_map is not None:
                output_map[start:start + len(row_cpu)] = prediction
            diff64 = difference.astype(np.float64)
            ref64 = reference.astype(np.float64)
            pred64 = prediction.astype(np.float64)
            squared_error_sum += float(np.sum(diff64 * diff64))
            absolute_error_sum += float(np.sum(np.abs(diff64)))
            dot_sum += float(np.sum(ref64 * pred64))
            original_sq_sum += float(np.sum(ref64 * ref64))
            reconstructed_sq_sum += float(np.sum(pred64 * pred64))
            maximum_absolute_error = max(maximum_absolute_error, float(np.max(np.abs(diff64))))
            count += difference.size
    if output_map is not None:
        output_map.flush()
        del output_map
    original_std = float(np.std(original, dtype=np.float64))
    rmse = math.sqrt(squared_error_sum / max(count, 1))
    cosine = dot_sum / math.sqrt(original_sq_sum * reconstructed_sq_sum) if original_sq_sum and reconstructed_sq_sum else 0.0
    return {
        "rmse": rmse,
        "rmse_over_original_std": rmse / original_std if original_std else 0.0,
        "relative_l2_error": math.sqrt(squared_error_sum / original_sq_sum) if original_sq_sum else 0.0,
        "cosine_similarity": cosine,
        "mean_absolute_error": absolute_error_sum / max(count, 1),
        "max_absolute_error": maximum_absolute_error,
        "original_std": original_std,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=base.DEFAULT_MODEL, help="Diretório local, arquivo Safetensors ou ID do Hugging Face")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--tensor-name", default=None, help="Nome exato do tensor 2D; sem isso escolhe uma matriz automaticamente")
    parser.add_argument("--max-auto-elements", type=int, default=20_000_000)
    parser.add_argument("--list-tensors", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Dimensão dos códigos compartilhados por linha/coluna")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Largura das camadas internas da fórmula não linear")
    parser.add_argument("--epochs", type=int, default=5, help="Passadas completas sobre todas as linhas da matriz")
    parser.add_argument("--batch-rows", type=int, default=16, help="Linhas processadas em cada lote")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="nonlinear_formula_layer_results")
    parser.add_argument("--save-reconstructed", action="store_true", help="Salvar também a matriz prevista em NPY (tamanho próximo ao original FP32)")
    args = parser.parse_args()

    if args.embedding_dim < 1 or args.hidden_dim < 1 or args.epochs < 1 or args.batch_rows < 1:
        parser.error("embedding-dim, hidden-dim, epochs e batch-rows precisam ser positivos.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.perf_counter()
    files = base.resolve_model(args.model, args.cache_dir, args.revision)
    if args.list_tensors:
        for item in base.list_tensors(files):
            print(f"{item['name']}\tshape={item['shape']}\tparams={item['numel']:,}\t{item['file'].name}")
        return 0

    selected = base.choose_tensor(files, args.tensor_name, args.max_auto_elements)
    if len(selected["shape"]) != 2:
        print(f"[ERRO] O tensor selecionado não é 2D: {selected['name']} {selected['shape']}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Tensor: {selected['name']}", flush=True)
    print(f"Arquivo: {selected['file']}", flush=True)
    print(f"Dimensões: {selected['shape']} | pesos: {selected['numel']:,}", flush=True)

    with safe_open(str(selected["file"]), framework="pt", device="cpu") as source:
        source_tensor = source.get_tensor(selected["name"])
        source_dtype = str(source_tensor.dtype).replace("torch.", "")
        source_tensor_bytes = int(source_tensor.numel() * source_tensor.element_size())
        original = source_tensor.to(dtype=torch.float32).cpu().numpy().copy()
        del source_tensor

    mean = float(np.mean(original, dtype=np.float64))
    std = float(np.std(original, dtype=np.float64))
    if std <= 0.0 or not math.isfinite(std):
        print("[ERRO] O tensor tem desvio-padrão inválido; não é possível normalizar.", file=sys.stderr)
        return 2
    target_normalized = torch.from_numpy((original - mean) / std)
    rows, cols = original.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NonlinearWeightFormula(rows, cols, args.embedding_dim, args.hidden_dim).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes_fp16 = parameter_count * 2
    source_bits_per_weight = source_tensor_bytes * 8 / original.size

    print(f"Dispositivo de treino: {device}", flush=True)
    print(f"Fórmula: MLP não linear + códigos aprendidos de linha/coluna", flush=True)
    print(f"Parâmetros aprendidos: {parameter_count:,} (~{parameter_bytes_fp16 / 1_000_000:.3f} MB em FP16)", flush=True)
    print(f"Tamanho original do tensor: {source_tensor_bytes:,} bytes ({source_bits_per_weight:.2f} bits/peso)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    for epoch in range(args.epochs):
        model.train()
        row_order = torch.randperm(rows)
        epoch_loss_sum = 0.0
        epoch_elements = 0
        for start in range(0, rows, args.batch_rows):
            row_ids_cpu = row_order[start:start + args.batch_rows]
            target = target_normalized[row_ids_cpu].to(device)
            prediction = model(row_ids_cpu.to(device))
            loss = nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += float(loss.detach().cpu()) * target.numel()
            epoch_elements += target.numel()
        print(f"Época {epoch + 1}/{args.epochs}: MSE normalizada={epoch_loss_sum / epoch_elements:.6f}", flush=True)

    formula_path = output_dir / "nonlinear_weight_formula.npz"
    metadata = {
        "format": "RC-IA learned nonlinear weight formula",
        "version": 1,
        "tensor_name": selected["name"],
        "shape": [rows, cols],
        "source_dtype": source_dtype,
        "normalization_mean": mean,
        "normalization_std": std,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "formula": "W_hat[r,c] = mean + std * (MLP(row_embedding[r], col_embedding[c]) + row_bias[r] + col_bias[c])",
        "parameter_count": parameter_count,
    }
    artifact_bytes = save_formula(formula_path, model, metadata)
    del model

    # Re-open the actual saved artifact and measure its FP16-stored parameters.
    restored_model, _ = load_formula(formula_path, device)
    print("Medindo a fórmula recarregada sobre todos os pesos...", flush=True)
    reconstructed_path = output_dir / "weights_nonlinear_reconstructed.npy" if args.save_reconstructed else None
    metrics = reconstruct_and_measure(
        restored_model, original, mean, std, device, args.batch_rows, reconstructed_path
    )
    del restored_model

    report: dict[str, Any] = {
        "tensor_name": selected["name"],
        "source_file": str(selected["file"]),
        "shape": [rows, cols],
        "num_weights": int(original.size),
        "source_dtype": source_dtype,
        "original_tensor_bytes": source_tensor_bytes,
        "original_tensor_bits_per_weight": source_bits_per_weight,
        "formula_path": str(formula_path),
        "formula_bytes": artifact_bytes,
        "formula_bits_per_weight": artifact_bytes * 8 / original.size,
        "compression_percent_vs_source_tensor": 100.0 * (1.0 - artifact_bytes / source_tensor_bytes),
        "parameter_count": parameter_count,
        "parameter_bytes_fp16_before_container_compression": parameter_bytes_fp16,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "batch_rows": args.batch_rows,
        "device": str(device),
        "reconstructed_weights_path": str(reconstructed_path) if reconstructed_path else None,
        **metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output_dir / "nonlinear_formula_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== RESULTADO DA FÓRMULA NÃO LINEAR ===")
    print(f"Artefato: {formula_path} ({artifact_bytes:,} bytes)")
    print(f"Bits efetivos/peso: {report['formula_bits_per_weight']:.5f}")
    print(f"Redução vs. tensor original: {report['compression_percent_vs_source_tensor']:.2f}%")
    print(f"RMSE/std: {metrics['rmse_over_original_std']:.6f}")
    print(f"Erro L2 relativo: {metrics['relative_l2_error']:.6f}")
    print(f"Cosseno: {metrics['cosine_similarity']:.6f}")
    print(f"Relatório: {report_path}")
    print(f"Tempo total: {report['elapsed_seconds']:.1f}s")
    print("A matriz inteira não é armazenada no artefato: os pesos são previstos pela função compartilhada.")
    print("Nota: erro dos pesos não equivale diretamente à perda de qualidade de geração.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
