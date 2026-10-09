from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from nonlinear_formula_layer import NonlinearWeightFormula, load_formula, save_formula


class NonlinearFormulaTests(unittest.TestCase):
    def test_formula_artifact_round_trip(self) -> None:
        torch.manual_seed(7)
        model = NonlinearWeightFormula(rows=5, cols=7, embedding_dim=3, hidden_dim=8)
        model.eval()
        row_ids = torch.arange(5)
        expected = model(row_ids).detach()

        metadata = {
            "shape": [5, 7],
            "embedding_dim": 3,
            "hidden_dim": 8,
            "normalization_mean": 0.0,
            "normalization_std": 1.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "formula.npz"
            size = save_formula(path, model, metadata)
            restored, restored_metadata = load_formula(path, torch.device("cpu"))
            actual = restored(row_ids).detach()

        self.assertGreater(size, 0)
        self.assertEqual(tuple(actual.shape), (5, 7))
        self.assertEqual(restored_metadata["shape"], [5, 7])
        # Parameters are deliberately stored as FP16 in the compact artifact.
        self.assertTrue(torch.allclose(expected, actual, atol=1e-3, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
