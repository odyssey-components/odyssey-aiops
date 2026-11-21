"""A physical ray must hit the outer corner instead of escaping through a gap."""

import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import cast
from conftest import workspace_root

import mujoco
import numpy as np

from aiops_go1_training.vision.configuration import VisionConfig
from aiops_go1_training.vision.scene import make_scene


class TurnWallTests(unittest.TestCase):
    def test_outer_corner_is_closed_in_actual_collision_geometry(self) -> None:
        root = workspace_root() / "tmp/aiops-turn-walls"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            source = Path(folder)
            (source / "unitree_go1").mkdir()
            joints = "".join(
                f'<body name="leg{i}" pos="0 0 -.1"><joint name="j{i}"/><geom size=".01"/></body>' for i in range(12)
            )
            motors = "".join(f'<motor joint="j{i}" ctrllimited="true" ctrlrange="-1 1"/>' for i in range(12))
            (source / "unitree_go1/scene_torque.xml").write_text(
                '<mujoco><worldbody><body name="trunk" pos="0 0 .3"><freejoint/>'
                '<geom type="sphere" size=".02"/>'
                + joints
                + "</body></worldbody><actuator>"
                + motors
                + "</actuator></mujoco>"
            )
            for sign in (-1, 1):
                model, _ = make_scene(source, VisionConfig(goal_kind="turn", turn_angle_deg=sign * 90), 7)
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                rays = (
                    ([1.4, 0, 0.2], [0, -sign, 0]),
                    ([1.2, 0, 0.2], [1, 0, 0]),
                    ([1.618, -sign * 0.418, 0.6], [0, 0, -1]),
                    ([0.818, sign * 0.382, 0.6], [0, 0, -1]),
                )
                for point, direction in rays:
                    geom = np.array([-1], dtype=np.int32)
                    ray = cast(Callable[..., float], getattr(mujoco, "mj_ray"))
                    distance = ray(model, data, np.array(point), np.array(direction), None, True, -1, geom)
                    self.assertGreaterEqual(distance, 0, "outer wall has an open corner")
                    self.assertTrue(
                        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom[0])) or "").startswith("vision_")
                    )
