"""Residual initialization must retain the imported actor and freeze it during PPO."""

import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from aiops_go1_training.upstream import artifacts, policy
from aiops_go1_training.vision.artifacts import load_policy, save_policy
from aiops_go1_training.vision.configuration import VisionConfig
from aiops_go1_training.vision.policy import adam, forward, initialize, update


class ResidualTests(unittest.TestCase):
    def test_zero_residual_and_ppo_freeze_and_portable_roundtrip(self):
        base = {name: jnp.full(shape, 0.01) for name, shape in policy.SHAPES.items()}
        params = initialize(jax.random.PRNGKey(1), 16, 16, base=base)
        depth, body = jnp.ones((8, 16, 16, 1)), jnp.ones((8, 45))
        expected, _ = policy.forward(base, body)
        mean, _ = forward(params, depth, body)
        np.testing.assert_allclose(mean, expected, atol=1e-6)
        params, _, _ = update(
            params, adam(params), depth, body, mean + 0.01, jnp.zeros(8), jnp.arange(8.0), jnp.ones(8), 0.0003
        )
        for name, value in base.items():
            np.testing.assert_array_equal(params["base." + name], value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts.save(root / "input", base, "a" * 64, "fixture")
            _, manifest = artifacts.load(root / "input", "a" * 64)
            config = VisionConfig(
                format="go1-depth-residual/v2", height=16, width=16, base_policy_sha256=str(manifest["policy_sha256"])
            )
            (root / "output").mkdir()
            save_policy(root / "output", params, config, "a" * 64, base_directory=root / "input")
            loaded, restored = load_policy(root / "output", "a" * 64)
            self.assertEqual(restored.format, "go1-depth-residual/v2")
            for name, value in params.items():
                np.testing.assert_array_equal(loaded[name], value)
            (root / "output" / "base" / "policy.npz").write_bytes(b"bad")
            with self.assertRaises(ValueError):
                load_policy(root / "output", "a" * 64)
