"""Visual policies have their own protocol and never silently train on CPU."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from conftest import workspace_root

import jax
import numpy as np
from aiops_go1_training.vision.artifacts import load_policy, save_policy
from aiops_go1_training.vision.configuration import VisionConfig
from aiops_go1_training.vision.policy import initialize
from aiops_go1_training.runtime.devices import select_device


class ArtifactTests(unittest.TestCase):
    def test_pickle_free_roundtrip_and_checksum_and_source_rejection(self):
        root = workspace_root() / "tmp" / "go1-visual-artifacts"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            directory = Path(folder)
            config = VisionConfig()
            params = initialize(jax.random.PRNGKey(8), config.height, config.width)
            save_policy(directory, params, config, "a" * 64)
            restored, loaded = load_policy(directory, "a" * 64)
            self.assertEqual(loaded, config)
            self.assertTrue(all(np.array_equal(params[name], restored[name]) for name in params))
            with self.assertRaises(ValueError):
                load_policy(directory, "b" * 64)
            manifest = directory / "model.manifest.toml"
            original = manifest.read_text()
            manifest.write_text(original.replace("actions = 12", "actions = 13"))
            with self.assertRaisesRegex(ValueError, "protocol"):
                load_policy(directory, "a" * 64)
            manifest.write_text(original)
            (directory / "policy.npz").write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_policy(directory, "a" * 64)

    def test_missing_gpu_is_refused_instead_of_cpu_fallback(self):
        with (
            patch("aiops_go1_training.runtime.devices.jax.devices", return_value=[]),
            self.assertRaisesRegex(RuntimeError, "no CPU fallback"),
        ):
            select_device("cuda")
