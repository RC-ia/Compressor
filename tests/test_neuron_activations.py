from __future__ import annotations

import unittest

import numpy as np

from analyze_neuron_activations import compare_activation_signatures


class ActivationSimilarityTests(unittest.TestCase):
    def test_detects_correlated_activations_and_estimates_merge(self) -> None:
        x = np.linspace(-1.0, 1.0, 64, dtype=np.float64)
        activation_a = x + 1.0
        activation_b = 2.0 * activation_a
        activation_c = np.sin(np.arange(64, dtype=np.float64) * 1.731)
        activations = np.column_stack((activation_a, activation_b, activation_c))
        down_weight = np.array([
            [0.6, -0.2, 0.5],
            [0.1, 0.9, -0.7],
        ], dtype=np.float64)
        layer_output = activations @ down_weight.T

        counts, candidates, summary = compare_activation_signatures(
            activations,
            down_weight,
            layer_output,
            top_limit=10,
            batch_rows=2,
        )

        self.assertGreaterEqual(counts["pairs_abs_corr_ge_0.99"], 1)
        candidate = next(
            item for item in candidates
            if {item["neuron_a"], item["neuron_b"]} == {0, 1}
        )
        self.assertAlmostEqual(candidate["activation_correlation_abs"], 1.0, places=6)
        self.assertAlmostEqual(candidate["merge_a_into_b_scale"], 0.5, places=6)
        self.assertLess(candidate["pair_contribution_relative_error_if_merged"], 1e-10)
        self.assertEqual(summary["token_positions"], 64)


if __name__ == "__main__":
    unittest.main()
