from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from formula_fit_layer import fit_fourier_formula, reconstruct_formula, save_formula


class FourierFormulaTests(unittest.TestCase):
    def test_round_trip_compact_formula_on_smooth_matrix(self) -> None:
        rows, cols = 32, 40
        row = np.arange(rows, dtype=np.float32)[:, None]
        col = np.arange(cols, dtype=np.float32)[None, :]
        source = (
            0.7 * np.cos(2.0 * np.pi * 3.0 * row / rows)
            + 0.2 * np.sin(2.0 * np.pi * 2.0 * col / cols)
            + 0.1 * np.cos(2.0 * np.pi * (row / rows + col / cols))
        ).astype(np.float32)

        formula = fit_fourier_formula(source, cutoff=4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "formula.npz"
            size = save_formula(path, formula, "synthetic.weight")
            restored, metadata = reconstruct_formula(path)

        self.assertGreater(size, 0)
        self.assertEqual(restored.shape, source.shape)
        self.assertEqual(metadata["tensor_name"], "synthetic.weight")
        self.assertLess(float(np.sqrt(np.mean((source - restored) ** 2))), 1e-3)


if __name__ == "__main__":
    unittest.main()
