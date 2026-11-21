"""Real offscreen camera observations using a synthetic, readable model."""

import tempfile
import unittest
from pathlib import Path
from typing import Protocol, cast
from conftest import workspace_root

import mujoco
import numpy as np
from aiops_go1_training.vision.configuration import VisionConfig
from aiops_go1_training.vision.scene import DepthCamera, body_collision, make_scene, minimum_robot_x
from numpy.typing import NDArray


class _ModelGeometry(Protocol):
    geom_pos: NDArray[np.float64]


class SensorTests(unittest.TestCase):
    def test_obstacles_change_pixels_and_frames_follow_simulation_time(self):
        root = workspace_root() / "tmp" / "go1-visual-sensor"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as name:
            source = Path(name)
            (source / "unitree_go1").mkdir()
            joints = "".join(
                f'<body name="leg{i}" pos="0 0 -.1"><joint name="j{i}"/><geom size=".01"/></body>' for i in range(12)
            )
            motors = "".join(f'<motor joint="j{i}" ctrllimited="true" ctrlrange="-1 1"/>' for i in range(12))
            xml = f'<mujoco><option timestep=".002"/><worldbody><geom type="plane" size="5 5 .1"/><body name="trunk" pos="0 0 .3"><freejoint/><geom type="box" size=".1 .05 .05"/>{joints}</body></worldbody><actuator>{motors}</actuator></mujoco>'
            (source / "unitree_go1" / "scene_torque.xml").write_text(xml)
            cfg = VisionConfig(height_min=0.02, height_max=0.02)
            model, _ = make_scene(source, cfg, 7)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            with DepthCamera(model, cfg) as camera:
                first = camera.observe(np.asarray(data.qpos), np.asarray(data.qvel), 0)
                self.assertIs(first, camera.observe(np.asarray(data.qpos), np.asarray(data.qvel), 0.02))
                self.assertEqual(first.depth.shape, (48, 64, 1))
                self.assertTrue(np.isfinite(first.depth).all())
                self.assertEqual(camera.observe(np.asarray(data.qpos), np.asarray(data.qvel), 0.1).time, 0.1)
                with self.assertRaises(ValueError):
                    camera.observe(np.asarray(data.qpos), np.asarray(data.qvel), 0)
                camera.reset()
                obstacle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "vision_obstacle")
                cast(_ModelGeometry, model).geom_pos[obstacle, 2] = 0.15
                model.geom_size[obstacle, 2] = 0.15
                changed = camera.observe(np.asarray(data.qpos), np.asarray(data.qvel), 0)
                self.assertGreater(float(np.abs(first.depth - changed.depth).max()), 0.01)
            model, obstacle_spec = make_scene(source, cfg, 7)
            data = mujoco.MjData(model)
            data.qpos[0] = obstacle_spec.x - obstacle_spec.length / 2 - 0.05
            data.qpos[2] = 0.03
            mujoco.mj_forward(model, data)
            self.assertGreater(data.ncon, 0)
            self.assertTrue(body_collision(model, data))
            self.assertLess(minimum_robot_x(model, data), float(data.qpos[0]))
            with self.assertRaises(RuntimeError):
                camera.observe(np.asarray(data.qpos), np.asarray(data.qvel), 0.2)
