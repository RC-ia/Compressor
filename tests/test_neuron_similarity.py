from __future__ import annotations

import unittest

import torch

from analyze_neuron_similarity import analyze_similarity


class NeuronSimilarityTests(unittest.TestCase):
    def test_finds_duplicate_gated_neuron_with_joint_up_down_sign_flip(self) -> None:
        # Neurons 0 and 1 share the same gate direction. Their up and down
        # vectors both flip sign, so (up*x) * down produces the same contribution.
        gate = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        up = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        down = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        normalized = {"gate": gate, "up": up, "down": down}
        norms = {
            "gate": torch.tensor([1.0, 1.0, 1.0]),
            "up": torch.tensor([2.0, 2.0, 1.0]),
            "down": torch.tensor([0.5, 0.5, 1.0]),
        }

        counts, candidates, _ = analyze_similarity(
            normalized, norms, neuron_ids=[0, 1, 2], batch_rows=2,
            scale_tolerance=0.1, top_limit=10,
        )

        self.assertEqual(counts["pairs_all_components_ge_99pct"], 1)
        self.assertTrue(any({item["neuron_a"], item["neuron_b"]} == {0, 1} for item in candidates))


if __name__ == "__main__":
    unittest.main()
