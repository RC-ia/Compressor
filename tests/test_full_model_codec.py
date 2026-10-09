from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


class FullModelCodecTests(unittest.TestCase):
    def test_full_checkpoint_compress_verify_and_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source_model"
            source.mkdir()
            original_weight = torch.randn(96, 128, dtype=torch.float32).to(torch.bfloat16)
            original_bias = torch.linspace(-0.1, 0.1, 32, dtype=torch.float32).to(torch.bfloat16)
            original_integer = torch.tensor([1, 2, 3, 255], dtype=torch.int64)
            original_zeros = torch.zeros(24, dtype=torch.bfloat16)
            save_file(
                {
                    "model.layers.0.proj.weight": original_weight.contiguous(),
                    "model.layers.0.norm.weight": original_bias.contiguous(),
                    "model.test.integer_buffer": original_integer.contiguous(),
                    "model.test.zero_buffer": original_zeros.contiguous(),
                },
                str(source / "model.safetensors"),
            )
            (source / "config.json").write_text(
                json.dumps({"model_type": "synthetic", "hidden_size": 128}),
                encoding="utf-8",
            )
            archive_path = root / "full_model.rccomp"
            output_dir = root / "rebuilt_model"

            compress = subprocess.run(
                [
                    sys.executable, str(ROOT / "full_model_codec.py"), "compress",
                    "--model", str(source),
                    "--output", str(archive_path),
                    "--levels", "256",
                    "--scale-multiplier", "0.75",
                    "--scale-sample-size", "256",
                    "--chunk-elements", "512",
                    "--preserve-small-elements", "64",
                    "--progress-every", "1",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(compress.returncode, 0, msg=compress.stdout + "\n" + compress.stderr)
            self.assertTrue(archive_path.exists())
            report_path = archive_path.with_suffix(archive_path.suffix + ".report.json")
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "full_checkpoint_compressed")
            self.assertEqual(report["tensor_count"], 4)
            self.assertEqual(report["archive_payload_verification"]["status"], "passed")
            self.assertGreater(report["global_floating_weight_metrics"]["cosine_similarity"], 0.99)

            with zipfile.ZipFile(archive_path, "r") as archive:
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            self.assertEqual(len(manifest["tensors"]), 4)
            kinds = {record["tensor_name"]: record["kind"] for record in manifest["tensors"]}
            self.assertEqual(kinds["model.layers.0.proj.weight"], "log")
            self.assertEqual(kinds["model.layers.0.norm.weight"], "raw")
            self.assertEqual(kinds["model.test.integer_buffer"], "raw")

            decode = subprocess.run(
                [
                    sys.executable, str(ROOT / "full_model_codec.py"), "decode",
                    "--archive", str(archive_path),
                    "--output-dir", str(output_dir),
                    "--source-model", str(source),
                    "--shard-max-mb", "1",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(decode.returncode, 0, msg=decode.stdout + "\n" + decode.stderr)
            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue((output_dir / "model.safetensors").exists())
            with safe_open(str(output_dir / "model.safetensors"), framework="pt", device="cpu") as sf:
                rebuilt_weight = sf.get_tensor("model.layers.0.proj.weight")
                rebuilt_bias = sf.get_tensor("model.layers.0.norm.weight")
                rebuilt_int = sf.get_tensor("model.test.integer_buffer")
                rebuilt_zeros = sf.get_tensor("model.test.zero_buffer")
            self.assertEqual(rebuilt_weight.dtype, torch.bfloat16)
            self.assertEqual(rebuilt_bias.dtype, torch.bfloat16)
            torch.testing.assert_close(rebuilt_bias, original_bias, rtol=0, atol=0)
            torch.testing.assert_close(rebuilt_int, original_integer, rtol=0, atol=0)
            torch.testing.assert_close(rebuilt_zeros, original_zeros, rtol=0, atol=0)
            self.assertLess(
                float(torch.sqrt(torch.mean((rebuilt_weight.float() - original_weight.float()) ** 2))),
                0.01,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
