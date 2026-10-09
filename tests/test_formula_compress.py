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
import formula_compress as formulas  # noqa: E402


class FormulaCompressionTests(unittest.TestCase):
    def test_arithmetic_progression_quantizer_has_bounded_error(self) -> None:
        x = np.linspace(-1.0, 2.0, 1001, dtype=np.float32)
        q, params = formulas.uniform_quantize(x, 64)
        rebuilt = formulas.uniform_reconstruct(q, params)
        max_error = float(np.max(np.abs(x - rebuilt)))
        self.assertEqual(q.dtype, np.uint8)
        self.assertLessEqual(max_error, float(params["step"]) / 2.0 + 1e-6)
        self.assertEqual(rebuilt.shape, x.shape)

    def test_log_compander_reconstructs_finite_values(self) -> None:
        rng = np.random.default_rng(123)
        x = np.concatenate([rng.normal(0.0, 0.03, 4096), np.array([-0.9, 0.8])]).astype(np.float32)
        q, params = formulas.log_compand_quantize(x, 256)
        rebuilt = formulas.log_compand_reconstruct(q, params)
        self.assertEqual(q.dtype, np.uint8)
        self.assertTrue(np.isfinite(rebuilt).all())
        self.assertEqual(rebuilt.shape, x.shape)
        self.assertGreater(float(np.max(np.abs(rebuilt))), 0.0)

    def test_block_polynomial_reconstructs_piecewise_linear_signal(self) -> None:
        block_size = 64
        parts = []
        for i in range(12):
            t = np.linspace(-1.0, 1.0, block_size, dtype=np.float32)
            parts.append((i * 0.05 + (i + 1) * 0.01 * t).astype(np.float32))
        x = np.concatenate(parts)
        coeffs, params = formulas.fit_polynomial_blocks(x, block_size, degree=1, coefficient_dtype="fp32")
        rebuilt = formulas.reconstruct_polynomial_blocks(coeffs, params)
        self.assertEqual(coeffs.shape, (12, 2))
        self.assertLess(float(np.sqrt(np.mean((x - rebuilt) ** 2))), 1e-6)
        self.assertLess(coeffs.nbytes, x.nbytes)

    def test_formula_cli_writes_real_artifacts_from_synthetic_tensor(self) -> None:
        rng = np.random.default_rng(456)
        matrix = rng.normal(0.0, 0.02, size=(64, 64)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "synthetic.safetensors"
            save_file({"test.proj.weight": torch.from_numpy(matrix)}, str(source))
            output = root / "formula_out"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "formula_compress.py"),
                    "--model",
                    str(source),
                    "--tensor-name",
                    "test.proj.weight",
                    "--levels",
                    "16",
                    "--sample-size",
                    "2048",
                    "--block-size",
                    "32",
                    "--projection-batch",
                    "2",
                    "--output-dir",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
            summaries = list(output.glob("*_formula_comparison.json"))
            artifacts = list(output.glob("*.npz"))
            self.assertEqual(len(summaries), 1)
            self.assertEqual(len(artifacts), 5)
            summary = json.loads(summaries[0].read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "formula_experiments_completed")
            self.assertEqual(len(summary["results"]), 5)
            for item in summary["results"]:
                self.assertGreater(item["artifact_npz_bytes_actual"], 0)
                self.assertGreaterEqual(item["savings_pct_vs_source_tensor"], -100.0)
                self.assertTrue(np.isfinite(item["full_tensor_weight_reconstruction_error"]["rmse"]))
                self.assertIsNotNone(item["linear_projection_probe"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
