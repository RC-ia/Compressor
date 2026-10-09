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
import compress_tensor as codec  # noqa: E402


class CodecUnitTests(unittest.TestCase):
    def test_pack_unpack_roundtrip_various_widths(self) -> None:
        rng = np.random.default_rng(123)
        for bits in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
            indices = rng.integers(0, 1 << bits, size=10_003, dtype=np.uint32)
            packed = codec.pack_indices_bitplanes(indices, bits)
            decoded = codec.unpack_indices_bitplanes(packed, indices.size, bits)
            np.testing.assert_array_equal(decoded, indices)

    def test_single_symbol_group_uses_zero_bits(self) -> None:
        indices = np.zeros(17, dtype=np.uint8)
        packed = codec.pack_indices_bitplanes(indices, 0)
        decoded = codec.unpack_indices_bitplanes(packed, indices.size, 0)
        self.assertEqual(packed.size, 0)
        np.testing.assert_array_equal(decoded, indices)

    def test_kmeans_improves_over_single_mean(self) -> None:
        rng = np.random.default_rng(456)
        values = np.concatenate([
            rng.normal(-2.0, 0.12, 5_000),
            rng.normal(0.3, 0.08, 3_000),
            rng.normal(3.5, 0.2, 2_000),
        ]).astype(np.float32)
        centers = codec.weighted_kmeans_1d(values, 16)
        self.assertGreaterEqual(centers.size, 2)
        reconstructed = centers[np.searchsorted((centers[:-1] + centers[1:]) / 2, values, side="right")]
        self.assertLess(float(np.sqrt(np.mean((values - reconstructed) ** 2))), float(np.std(values)))

    def test_index_entropy_identifies_skewed_distribution(self) -> None:
        indices = np.array([0] * 9000 + [1] * 1000, dtype=np.uint8)
        stats = codec.analyze_index_map(indices, 16)
        self.assertLess(stats["symbol_entropy_bits_per_weight"], 1.0)
        self.assertEqual(stats["used_representatives"], 2)
        self.assertGreater(stats["average_run_length"], 1.0)

    def test_synthetic_safetensors_encode_decode_measures_real_file(self) -> None:
        rng = np.random.default_rng(789)
        matrix = np.concatenate([
            rng.normal(-0.02, 0.01, (128, 128)),
            rng.normal(0.02, 0.01, (128, 128)),
        ], axis=0).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "synthetic.safetensors"
            save_file({"model.layers.0.test_proj.weight": torch.from_numpy(matrix)}, str(source))
            for map_codec in ("auto", "zip_deflate_bitplanes", "zlib_symbols", "raw_bitplanes"):
                with self.subTest(map_codec=map_codec):
                    outdir = tmp_path / ("out_" + map_codec)
                    result = subprocess.run([
                        sys.executable, str(ROOT / "compress_tensor.py"),
                        "--model", str(source),
                        "--tensor-name", "model.layers.0.test_proj.weight",
                        "--groups", "16",
                        "--sample-size", "10000",
                        "--codebook-dtype", "fp32",
                        "--output-dir", str(outdir),
                        "--projection-batch", "3",
                        "--map-codec", map_codec,
                    ], capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
                    reports = list(outdir.glob("*_report.json"))
                    archives = list(outdir.glob("*.npz"))
                    self.assertEqual(len(reports), 1)
                    self.assertEqual(len(archives), 1)
                    report = json.loads(reports[0].read_text(encoding="utf-8"))
                    self.assertEqual(report["status"], "encoded_decoded_measured")
                    self.assertGreater(report["actual_groups"], 1)
                    self.assertLess(report["full_tensor_weight_reconstruction_error"]["rmse_over_weight_std"], 0.5)
                    self.assertIsNotNone(report["linear_projection_probe"])
                    self.assertGreater(report["artifact_npz_bytes_actual"], 0)
                    self.assertEqual(report["selected_map_codec"], map_codec if map_codec != "auto" else report["selected_map_codec"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
