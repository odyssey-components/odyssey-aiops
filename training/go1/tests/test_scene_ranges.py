"""Scene dimensions are not capped at the old short-course distances."""

import tempfile
import unittest
from pathlib import Path
from conftest import workspace_root

import mujoco
import numpy as np
from aiops_go1_training.vision.configuration import VisionConfig
from aiops_go1_training.vision.scene import make_scene


class SceneRangeTests(unittest.TestCase):
    def test_large_dimensions_and_nonfinite_rejection(self) -> None:
        for kind in ("turn", "uturn", "slope"):
            with self.subTest(kind=kind):
                config = VisionConfig(
                    goal_kind=kind,
                    turn_angle_deg=180 if kind == "uturn" else 90,
                    road_width_m=100,
                    target_distance_m=1000,
                )
                self.assertEqual(config.target_distance_m, 1000)
        for name in ("road_width_m", "target_distance_m"):
            for value in (float("nan"), float("inf"), -float("inf"), 0, -1):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    VisionConfig.model_validate({name: value})

    def test_wide_turns_and_long_slopes_compile_without_rendering(self) -> None:
        root = workspace_root() / "tmp/go1-scene-ranges"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            source = Path(directory)
            (source / "unitree_go1").mkdir()
            joints = "".join(
                f'<body name="leg{i}" pos="0 0 -.1"><joint name="j{i}"/><geom size=".01"/></body>' for i in range(12)
            )
            motors = "".join(f'<motor joint="j{i}" ctrllimited="true" ctrlrange="-1 1"/>' for i in range(12))
            (source / "unitree_go1/scene_torque.xml").write_text(
                '<mujoco><worldbody><geom type="plane" size="0 0 .1"/>'
                '<body name="trunk" pos="0 0 .3"><freejoint/><geom type="box" size=".1 .05 .05" mass="1"/>'
                f"{joints}</body></worldbody><actuator>{motors}</actuator></mujoco>"
            )
            for kind in ("turn", "uturn", "slope"):
                config = VisionConfig(
                    goal_kind=kind,
                    turn_angle_deg=180 if kind == "uturn" else 90,
                    road_width_m=100,
                    target_distance_m=1000,
                )
                model, _ = make_scene(source, config, 7)
                self.assertTrue(np.isfinite(model.geom_size).all())
                for geom in range(len(model.geom_size)):
                    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
                        self.assertTrue((model.geom_size[geom] > 0).all())

            for kind, height in (("duck", 0.28), ("jump", 0.20)):
                config = VisionConfig.model_validate({"goal_kind": kind, "beam_height_m": height})
                model, obstacle = make_scene(source, config, 7)
                beam = model.geom("vision_beam")
                self.assertAlmostEqual(obstacle.end_x, 1.28)
                if kind == "duck":
                    self.assertAlmostEqual(beam.pos[2] - beam.size[2], height)
                    self.assertEqual(model.geom("vision_beam_post_1").pos[1], 0.56)
                else:
                    self.assertAlmostEqual(beam.pos[2] + beam.size[2], height)
                self.assertNotEqual(beam.contype[0], 0)
