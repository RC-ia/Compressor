from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import deep_tensor_experiment as deep  # noqa: E402


class DeepTensorExperimentTests(unittest.TestCase):
    def test_randomized_svd_reconstructs_low_rank_matrix(self) -> None:
        rng = np.random.default_rng(123)
        left = rng.standard_normal((64, 4), dtype=np.float32)
        right = rng.standard_normal((48, 4), dtype=np.float32)
        matrix = left @ right.T
        u, singular_values, vh = deep.randomized_svd(
            matrix, max_rank=4, oversample=8, power_iterations=1, seed=7
        )
        stored_left, stored_right = deep.balanced_factors(u, singular_values, vh, rank=4)
        reconstructed = stored_left.astype(np.float32) @ stored_right.astype(np.float32).T
        relative_rmse = float(
            np.sqrt(np.mean((matrix - reconstructed) ** 2)) / np.std(matrix)
        )
        self.assertEqual(stored_left.shape, (64, 4))
        self.assertEqual(stored_right.shape, (48, 4))
        self.assertLess(relative_rmse, 0.01)

    def test_log_scale_parameter_changes_quantizer(self) -> None:
        rng = np.random.default_rng(456)
        values = np.concatenate([
            rng.normal(0.0, 0.02, 2048),
            np.array([-0.8, 0.6], dtype=np.float32),
        ]).astype(np.float32)
        idx_a, p_a = deep.make_log_indices(values, levels=64, scale_multiplier=0.5)
        idx_b, p_b = deep.make_log_indices(values, levels=64, scale_multiplier=2.0)
        self.assertNotEqual(p_a["scale"], p_b["scale"])
        self.assertFalse(np.array_equal(idx_a, idx_b))

    def test_chunked_metrics_are_finite(self) -> None:
        rng = np.random.default_rng(789)
        original = rng.normal(size=10_003).astype(np.float32)
        rebuilt = original + rng.normal(scale=0.01, size=original.size).astype(np.float32)
        metrics = deep.reconstruction_metrics(original, rebuilt, chunk_size=512)
        self.assertTrue(np.isfinite(metrics["rmse"]))
        self.assertTrue(np.isfinite(metrics["cosine_similarity_flat_weights"]))
        self.assertLess(metrics["rmse_over_weight_std"], 0.02)

    def test_cli_creates_and_decodes_all_codec_families(self) -> None:
        rng = np.random.default_rng(999)
        matrix = rng.normal(0.0, 0.03, size=(64, 64)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "tiny_matrix.safetensors"
            save_file({"model.test.proj.weight": torch.from_numpy(matrix)}, str(source))
            output = root / "deep_out"
            process = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "deep_tensor_experiment.py"),
                    "--model", str(source),
                    "--tensor-name", "model.test.proj.weight",
                    "--output-dir", str(output),
                    "--kmeans-samples", "256",
                    "--kmeans-groups", "8",
                    "--log-levels", "8,16",
                    "--log-scales", "0.5,1",
                    "--ranks", "4,8",
                    "--hybrid-ranks", "4",
                    "--residual-groups", "8",
                    "--oversample", "4",
                    "--power-iterations", "1",
                    "--projection-batch", "2",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, msg=process.stdout + "\n" + process.stderr)
            summaries = list(output.glob("deep_comparison.json"))
            archives = list(output.glob("*.npz"))
            self.assertEqual(len(summaries), 1)
            self.assertEqual(len(archives), 8)
            summary = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "deep_tensor_experiment_completed")
            self.assertEqual(len(summary["results_sorted_by_normalized_rmse"]), 8)
            methods = {item["kind"] for item in summary["results_sorted_by_normalized_rmse"]}
            self.assertEqual(methods, {"kmeans", "log", "lowrank", "hybrid"})
            for result in summary["results_sorted_by_normalized_rmse"]:
                self.assertGreater(result["artifact_bytes_actual"], 0)
                self.assertTrue(np.isfinite(result["full_tensor_error"]["rmse_over_weight_std"]))
                self.assertIsNotNone(result["linear_projection_probe"])
                if result["kind"] in {"kmeans", "log", "hybrid"}:
                    self.assertIn(
                        result["selected_map_codec"],
                        {"zip_deflate_bitplanes", "zlib_symbols", "raw_bitplanes"},
                    )
                    self.assertEqual(
                        set(result["candidate_codec_sizes_bytes"]),
                        {"zip_deflate_bitplanes", "zlib_symbols", "raw_bitplanes"},
                    )
                else:
                    self.assertEqual(result["selected_map_codec"], "npz_deflate_no_index_map")


if __name__ == "__main__":
    unittest.main(verbosity=2)
