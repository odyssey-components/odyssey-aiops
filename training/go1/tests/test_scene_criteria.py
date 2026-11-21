"""Configured task success must reflect physical progress, never mere job completion."""

import unittest

from aiops_go1_training.simulation.configuration import SimulationConfig
from aiops_go1_training.vision.configuration import VisionConfig
from aiops_go1_training.vision.goals import Goal


class CriteriaTests(unittest.TestCase):
    def test_turn_requires_configured_exit_distance_and_heading(self):
        cfg = VisionConfig(goal_kind="turn", success_exit_distance_m=2, success_heading_tolerance_deg=5, stable_steps=2)
        goal = Goal(cfg, 1.2)
        for _ in range(100):
            self.assertEqual(goal.observe(0, 0, 0.3, 1, False), "running")
        goal.observe(1.2, 0, 0.3, 1, False)
        self.assertEqual(goal.observe(1.2, 1, 0.3, 1, False, heading=1.570796), "running")
        self.assertEqual(goal.observe(1.2, 2.1, 0.3, 1, False, heading=1.570796), "running")
        self.assertEqual(goal.observe(1.2, 2.1, 0.3, 1, False, heading=1.570796), "success")

    def test_simulation_uses_scene_criteria_even_when_model_criteria_differ(self):
        config = SimulationConfig.model_validate(
            {
                "steps": 500,
                "scene": {
                    "kind": "turn",
                    "parameters": {"angle_deg": 90, "road_width_m": 0.8, "distance_m": 1.2},
                    "criteria": {"stable_seconds": 2, "exit_distance_m": 1.5, "heading_tolerance_deg": 5},
                },
            }
        )
        cfg = config.environment(VisionConfig(stable_steps=3))
        self.assertEqual(cfg.stable_steps, 100)
        self.assertEqual(cfg.success_exit_distance_m, 1.5)
        self.assertEqual(cfg.success_heading_tolerance_deg, 5)

    def test_slope_clearance_and_stability_cannot_be_bypassed_by_crouching(self):
        cfg = VisionConfig(goal_kind="slope", success_clearance_m=0.5, stable_steps=1)
        goal = Goal(cfg, 2)
        self.assertEqual(goal.observe(2.6, 0, 0.3, 1, False, rear_x=2.1), "running")
        self.assertEqual(goal.observe(3, 0, 0.18, 1, False, rear_x=2.6), "fallen")

    def test_legacy_scene_resets_criteria_to_legacy_defaults(self):
        cfg = SimulationConfig.model_validate(
            {"scene": {"kind": "uturn", "parameters": {"road_width_m": 0.8, "turn_at_m": 1.2}}}
        ).environment(VisionConfig(stable_steps=3, success_return_tolerance_m=0.02))
        self.assertEqual(cfg.stable_steps, 50)
        self.assertEqual(cfg.success_return_tolerance_m, 0.15)

    def test_uturn_overshooting_origin_is_not_inside_return_region(self):
        goal = Goal(VisionConfig(goal_kind="uturn", turn_angle_deg=180, stable_steps=1), 1.2)
        goal.observe(1.2, 0, 0.3, 1, False)
        self.assertEqual(goal.observe(-1, 0, 0.3, 1, False, heading=3.14159265), "running")
        self.assertEqual(goal.observe(0.1, 0, 0.3, 1, False, heading=3.14159265), "success")

    def test_legacy_empty_toml_criteria_uses_defaults(self):
        cfg = SimulationConfig.model_validate(
            {
                "scene": {
                    "kind": "turn",
                    "parameters": {"angle_deg": 90, "road_width_m": 0.8, "distance_m": 1.2},
                    "criteria": {},
                }
            }
        ).environment()
        self.assertEqual(cfg.stable_steps, 50)
        self.assertEqual(cfg.success_exit_distance_m, 0.8)
