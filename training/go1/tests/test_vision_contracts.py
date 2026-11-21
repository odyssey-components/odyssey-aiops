"""Bounded visual input, independent crossing outcomes and real CNN gradients."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from aiops_go1_training.vision.configuration import VisionConfig
from aiops_go1_training.vision.evaluation import Crossing
from aiops_go1_training.vision.policy import adam, forward, initialize, update


class VisionContracts(unittest.TestCase):
    def test_rejects_invalid_geometry_and_overlapping_seed_sets(self):
        for patch in (
            {"height_min": -0.1},
            {"height_max": 0.5},
            {"camera_hz": 60},
            {"width": 0},
            {"train_seeds": [1], "test_seeds": [1]},
        ):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                VisionConfig.model_validate(patch)

    def test_whole_body_crossing_requires_stability_and_cannot_bypass(self):
        judge = Crossing(end_x=1.2, corridor=0.3, stable_steps=3)
        self.assertEqual(judge.observe(1.4, 0, 0.3, 1, False), "running")
        self.assertEqual(judge.observe(1.7, 0, 0.3, 1, False), "running")
        self.assertEqual(judge.observe(1.7, 0, 0.3, 1, False), "running")
        self.assertEqual(judge.observe(1.7, 0, 0.3, 1, False), "success")
        self.assertEqual(Crossing(1.2, 0.3, 3).observe(2, 1, 0.3, 1, False), "bypass")
        self.assertEqual(Crossing(1.2, 0.3, 3).observe(2, 0, 0.3, 1, True), "collision")
        self.assertEqual(Crossing(1.2, 0.3, 3).observe(2, 0, 0.1, 0.1, False), "fallen")

    def test_crossing_rejects_rear_legs_still_before_obstacle_and_nonfinite_state(self):
        judge = Crossing(1.2, 0.3, 1)
        self.assertEqual(judge.observe(2, 0, 0.3, 1, False, rear_x=1.0), "running")
        self.assertEqual(judge.observe(2, 0, 0.3, 1, False, rear_x=1.3), "success")
        with self.assertRaises(ValueError):
            Crossing(1.2, 0.3, 1).observe(float("nan"), 0, 0.3, 1, False)

    def test_visual_features_affect_policy_and_receive_ppo_gradients(self):
        cfg = VisionConfig()
        params = initialize(jax.random.PRNGKey(7), cfg.height, cfg.width)
        depth = jnp.stack([jnp.zeros((cfg.height, cfg.width, 1)), jnp.ones((cfg.height, cfg.width, 1))])
        state = jnp.zeros((2, 45))
        mean, values = forward(params, depth, state)
        self.assertEqual(mean.shape, (2, 12))
        self.assertEqual(values.shape, (2,))
        self.assertFalse(np.allclose(np.asarray(mean[0]), np.asarray(mean[1])))
        updated, _, metrics = update(
            params,
            adam(params),
            depth,
            state,
            mean + jnp.array(0.1),
            jnp.zeros(2),
            jnp.array([1.0, -1.0]),
            jnp.array([1.0, 0.0]),
            0.0003,
        )
        self.assertTrue(all(np.isfinite(np.asarray(value)).all() for value in updated.values()))
        self.assertGreater(float(jnp.max(jnp.abs(updated["c1"] - params["c1"]))), 0)
        self.assertTrue(np.isfinite(float(metrics["loss"])))
