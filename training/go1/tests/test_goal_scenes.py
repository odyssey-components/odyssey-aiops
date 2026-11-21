"""Target goals differ from straight walking and cannot shortcut their sequence."""

import math
import unittest

from aiops_go1_training.vision.configuration import VisionConfig
from aiops_go1_training.vision.goals import Goal


class GoalTests(unittest.TestCase):
    def test_turn_requires_reaching_corner_then_rotating_and_continuing(self):
        goal = Goal(VisionConfig(goal_kind="turn", stable_steps=2), 1.2)
        self.assertEqual(goal.observe(0.9, 0, 0.3, 1, False), "running")
        self.assertEqual(goal.observe(1.2, 0, 0.3, 1, False), "running")
        self.assertEqual(goal.command(0)[0], 0)
        self.assertGreater(goal.command(0)[2], 0)
        self.assertEqual(goal.observe(1.2, 0.9, 0.3, 1, False, heading=math.pi / 2), "running")
        self.assertEqual(goal.observe(1.2, 0.9, 0.3, 1, False, heading=math.pi / 2), "success")
        straight = Goal(VisionConfig(goal_kind="turn"), 1.2)
        self.assertEqual(straight.observe(2, 0, 0.3, 1, False), "bypass")

    def test_uturn_requires_outbound_leg_and_return_with_correct_heading(self):
        config = VisionConfig(goal_kind="uturn", turn_angle_deg=180, stable_steps=1)
        goal = Goal(config, 1.2)
        self.assertEqual(goal.observe(0, 0, 0.3, 1, False, heading=math.pi), "running")
        goal.observe(1.2, 0, 0.3, 1, False)
        self.assertEqual(goal.observe(0.1, 0, 0.3, 1, False, heading=0), "running")
        self.assertEqual(goal.observe(0.1, 0, 0.3, 1, False, heading=math.pi), "success")
        self.assertEqual(Goal(config, 1.2).observe(0.5, 1, 0.3, 1, False), "bypass")

    def test_slope_preserves_full_robot_clearance_and_failure_checks(self):
        goal = Goal(VisionConfig(goal_kind="slope", stable_steps=1), 2)
        self.assertEqual(goal.observe(2.6, 0, 0.6, 1, False, rear_x=1.9), "running")
        self.assertEqual(goal.observe(2.6, 0, 0.6, 1, False, rear_x=2.1), "success")
        with self.assertRaises(ValueError):
            VisionConfig(goal_kind="uturn", turn_angle_deg=90)

    def test_beams_require_the_matching_maneuver_before_clearance(self):
        for kind, height, airborne in (("duck", 0.21, False), ("jump", 0.43, True)):
            config = VisionConfig.model_validate(
                {"goal_kind": kind, "stable_steps": 1, "beam_height_m": 0.28 if kind == "duck" else 0.20}
            )
            goal = Goal(config, 1.28)
            self.assertEqual(goal.observe(2, 0, 0.3, 1, False, rear_x=1.5), "running")
            goal.observe(1.2, 0, height, 1, False, airborne=airborne)
            self.assertEqual(goal.observe(2, 0, 0.3, 1, False, rear_x=1.5), "success")
            wrong = Goal(config, 1.28)
            wrong.observe(1.2, 0, 0.35, 1, False, airborne=False)
            self.assertEqual(wrong.observe(2, 0, 0.3, 1, False, rear_x=1.5, timeout=True), "timeout")
