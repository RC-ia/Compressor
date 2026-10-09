from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from predictive_map_experiment import (
    _predictive_candidate,
    decode_predictive_map,
    pack_nibbles,
    save_archive,
    load_archive,
    unpack_nibbles,
)


class PredictiveMapTests(unittest.TestCase):
    def test_pack_and_unpack_nibbles(self) -> None:
        values = np.array([0, 15, 7, 2, 9], dtype=np.uint8)
        restored = unpack_nibbles(pack_nibbles(values), len(values))
        np.testing.assert_array_equal(restored, values)

    def test_predictor_round_trip_on_repeated_rows(self) -> None:
        base_row = np.array([0, 1, 2, 3, 3, 2, 1, 0], dtype=np.uint8)
        labels = np.vstack([base_row for _ in range(20)])
        candidate = _predictive_candidate(labels, group_count=4)
        payload = candidate["payload"]
        residuals = unpack_nibbles(__import__("zlib").decompress(payload), labels.size).reshape(labels.shape)
        decoded = decode_predictive_map(residuals, candidate["predictor"], group_count=4)
        np.testing.assert_array_equal(decoded, labels)
        self.assertGreater(candidate["prediction_accuracy"], 0.9)

    def test_saved_predictive_archive(self) -> None:
        labels = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]], dtype=np.uint8)
        candidate = _predictive_candidate(labels, group_count=2)
        codebook = np.array([-0.25, 0.25], dtype=np.float32)
        metadata = {
            "num_weights": int(labels.size),
            "oriented_shape": list(labels.shape),
            "group_count": 2,
            "orientation": "vertical",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "map.npz"
            save_archive(path, codebook, candidate["predictor"], candidate["payload"], metadata)
            decoded_codebook, predictor, payload, restored_meta = load_archive(path)
        residuals = unpack_nibbles(__import__("zlib").decompress(payload), labels.size).reshape(labels.shape)
        decoded = decode_predictive_map(residuals, predictor, 2)
        np.testing.assert_array_equal(decoded, labels)
        np.testing.assert_array_equal(decoded_codebook, codebook)
        self.assertEqual(restored_meta["orientation"], "vertical")


if __name__ == "__main__":
    unittest.main()
