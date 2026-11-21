"""Training produces ONNX by default; existing policies convert without retraining."""

import hashlib
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from conftest import workspace_root

from aiops_maze_training.common.datasets import prepare_dataset, save_dataset


class OnnxCliTests(unittest.TestCase):
    def test_training_and_conversion_preserve_json_and_refuse_overwrite(self) -> None:
        root = workspace_root()
        temporary = root / "tmp" / "odymaze-onnx-cli"
        temporary.mkdir(parents=True, exist_ok=True)
        for lesson in ("fixed-map-v1", "random-map-v1", "local-map-v1", "memory-map-v1"):
            with self.subTest(lesson=lesson), tempfile.TemporaryDirectory(dir=temporary) as folder:
                output = Path(folder) / "trained"
                dataset_path = Path(folder) / "maps.jsonl"
                save_dataset(prepare_dataset(lesson, 7, 16), dataset_path)
                command = [
                    sys.executable,
                    "-m",
                    "aiops_maze_training.runtime.train",
                    "--lesson",
                    lesson,
                    "--epochs",
                    "0",
                    "--output",
                    str(output),
                    "--dataset",
                    str(dataset_path),
                ]
                if lesson != "fixed-map-v1":
                    command += ["--maps", "16"]
                trained = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
                self.assertEqual(trained.returncode, 0, trained.stderr)
                for name in ("model.json", "samples.json", "report.json", "model.onnx", "model.manifest.toml"):
                    self.assertTrue((output / name).is_file(), name)
                self.assertFalse((output / "dataset.jsonl").exists())
                self.assertFalse((output / "dataset.toml").exists())
                before = (output / "model.json").read_bytes()
                manifest = tomllib.loads((output / "model.manifest.toml").read_text())
                self.assertEqual(manifest["lesson"], lesson)
                self.assertNotIn("maze", manifest)
                self.assertIn("dataset", manifest)
                self.assertEqual(manifest["sha256"], hashlib.sha256((output / "model.onnx").read_bytes()).hexdigest())
                converted = Path(folder) / "converted" / "policy.onnx"
                convert = [
                    sys.executable,
                    "-m",
                    "aiops_maze_training.runtime.export",
                    "--input",
                    str(output / "model.json"),
                    "--output",
                    str(converted),
                ]
                result = subprocess.run(convert, check=False, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(converted.is_file())
                self.assertTrue(converted.with_suffix(".manifest.toml").is_file())
                self.assertEqual((output / "model.json").read_bytes(), before)
                binary_before = converted.read_bytes()
                refused = subprocess.run(convert, check=False, capture_output=True, text=True, timeout=30)
                self.assertEqual(refused.returncode, 2, refused.stderr)
                self.assertEqual(converted.read_bytes(), binary_before)

    def test_converter_rejects_invalid_json_and_extension(self) -> None:
        temporary = workspace_root() / "tmp" / "odymaze-onnx-cli"
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as folder:
            source = Path(folder) / "model.json"
            source.write_text('{"lesson":"future"}')
            command = [sys.executable, "-m", "aiops_maze_training.runtime.export", "--input", str(source)]
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertFalse((Path(folder) / "model.onnx").exists())
            result = subprocess.run(
                command + ["--output", str(Path(folder) / "bad.bin")],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 2)
            self.assertFalse((Path(folder) / "bad.bin").exists())


if __name__ == "__main__":
    unittest.main()
