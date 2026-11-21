"""Real SB3 is an oracle, not a training dependency of the adapter."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import cast

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import torch
from aiops_go1_training.upstream.artifacts import load, save
from aiops_go1_training.upstream.convert import convert
from aiops_go1_training.upstream.policy import forward, log_probability, observation
from stable_baselines3 import PPO


class Fixture(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (45,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (12,), dtype=np.float32)


class UpstreamTests(unittest.TestCase):
    def test_real_sb3_actor_critic_distribution_and_manifest_round_trip(self):
        model = PPO("MlpPolicy", Fixture(), device="cpu", n_steps=8, batch_size=8, seed=7)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model.save(root / "input.zip")
            convert(root / "input.zip", root / "base", "a" * 64, "fixture")
            params, manifest = load(root / "base", "a" * 64)
            raw = np.random.default_rng(1).normal(size=(16, 45)).astype(np.float32)
            raw[-1] *= 1000
            obs = np.asarray(observation(jnp.asarray(raw)))
            with torch.no_grad():
                tensor = torch.as_tensor(obs.copy())
                distribution = cast(torch.distributions.Normal, model.policy.get_distribution(tensor).distribution)
                expected = distribution.mean.numpy()
                values = model.policy.predict_values(tensor).numpy().reshape(-1)
            actual, actual_values = forward(params, jnp.asarray(raw))
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(actual_values, values, rtol=2e-5, atol=2e-6)
            chosen = jnp.asarray(expected) + 0.1
            np.testing.assert_allclose(
                log_probability(params, actual, chosen),
                distribution.log_prob(torch.as_tensor(np.asarray(chosen).copy())).sum(-1).numpy(),
                atol=3e-5,
            )
            np.testing.assert_allclose(np.clip(actual, -1, 1), np.clip(expected, -1, 1), atol=2e-6)
            np.testing.assert_array_equal(params["log_std"], model.policy.log_std.detach().numpy())
            self.assertEqual(manifest["observation_protocol"], "go1-sb3-observation/v1")
            save(root / "copy", params, "a" * 64, "copy", parent=manifest["policy_sha256"])
            load(root / "copy", "a" * 64)
            (root / "copy" / "policy.npz").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                load(root / "copy", "a" * 64)

    def test_observation_scaling_does_not_mutate_input(self):
        raw = np.ones((2, 45), dtype=np.float32)
        scaled = np.asarray(observation(jnp.asarray(raw)))
        np.testing.assert_array_equal(scaled[:, 12:15], [[2, 2, 0.25]] * 2)
        np.testing.assert_array_equal(scaled[:, 30:33], [[2, 2, 0.25]] * 2)
        np.testing.assert_array_equal(raw, np.ones((2, 45)))

    def test_reject_custom_policy_before_loading_any_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with zipfile.ZipFile(root / "bad.zip", "w") as archive:
                archive.writestr("data", json.dumps({"policy_kwargs": {"activation_fn": "unknown"}}))
                archive.writestr("policy.pth", b"not a checkpoint")
            with self.assertRaisesRegex(ValueError, "default"):
                convert(root / "bad.zip", root / "out", "a" * 64, "bad")
            self.assertFalse((root / "out").exists())

    def test_opaque_pickled_metadata_is_never_decoded_and_tensor_payloads_fail_closed(self):
        import io

        model = PPO("MlpPolicy", Fixture(), device="cpu", n_steps=8, batch_size=8, seed=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model.save(root / "original.zip")
            with zipfile.ZipFile(root / "original.zip") as archive:
                members = {name: archive.read(name) for name in archive.namelist()}
            metadata = json.loads(members["data"])
            metadata["policy_class"][":serialized:"] = "invalid pickle: must never be decoded"
            members["data"] = json.dumps(metadata).encode()
            with zipfile.ZipFile(root / "opaque.zip", "w") as archive:
                for name, data in members.items():
                    archive.writestr(name, data)
            convert(root / "opaque.zip", root / "okay", "a" * 64, "opaque")
            bad = dict(model.policy.state_dict())
            bad["log_std"] = torch.full((12,), float("nan"))
            buffer = io.BytesIO()
            torch.save(bad, buffer)
            members["policy.pth"] = buffer.getvalue()
            with zipfile.ZipFile(root / "nonfinite.zip", "w") as archive:
                for name, data in members.items():
                    archive.writestr(name, data)
            with self.assertRaises(ValueError):
                convert(root / "nonfinite.zip", root / "rejected", "a" * 64, "nonfinite")
            self.assertFalse((root / "rejected").exists())

    def test_reject_non_tensor_checkpoint_without_unrestricted_pickle_fallback(self):
        import io
        from types import SimpleNamespace

        model = PPO("MlpPolicy", Fixture(), device="cpu", n_steps=8, batch_size=8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model.save(root / "input.zip")
            with zipfile.ZipFile(root / "input.zip") as archive:
                members = {name: archive.read(name) for name in archive.namelist()}
            buffer = io.BytesIO()
            torch.save({"payload": SimpleNamespace(value=1)}, buffer)
            members["policy.pth"] = buffer.getvalue()
            with zipfile.ZipFile(root / "bad.zip", "w") as archive:
                for name, data in members.items():
                    archive.writestr(name, data)
            with self.assertRaisesRegex(ValueError, "numeric tensor"):
                convert(root / "bad.zip", root / "out", "a" * 64, "bad")
            self.assertFalse((root / "out").exists())
