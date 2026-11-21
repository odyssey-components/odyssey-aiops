"""GPU MJX physics synchronized with a MuJoCo EGL depth camera."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from numpy.typing import NDArray

from .configuration import VisionConfig
from .evaluation import Outcome
from .goals import Goal, yaw
from .scene import DepthCamera, body_collision, make_scene, minimum_robot_x, robot_airborne


class VisualEnvironment:
    def __init__(self, source: Path, config: VisionConfig, seed: int, device: jax.Device):
        if device.platform not in ("cpu", "gpu"):
            raise ValueError("visual training requires a JAX CPU or GPU")
        self.device = device
        self.config = config
        self.model, self.obstacle = make_scene(source, config, seed)
        self.host = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.host, 0)
        mujoco.mj_forward(self.model, self.host)
        self.default = np.asarray(self.host.qpos[7:]).copy()
        self.data: mjx.Data | None = None
        self.initial_data: mjx.Data | None = None
        self.advance: Callable[[mjx.Data, jax.Array], mjx.Data] | None = None
        self.camera = DepthCamera(self.model, config)
        _, renderer = self.camera.graphics_identity()
        if device.platform == "gpu" and any(name in renderer.lower() for name in ("llvmpipe", "softpipe", "software")):
            self.camera.close()
            raise RuntimeError("visual training refuses a software camera renderer")
        self.last = np.zeros(12, dtype=np.float32)
        self.previous_last = self.last.copy()
        self.steps = 0
        self.judge = Goal(config, self.obstacle.end_x)

        if device.platform == "gpu":
            gpu_model = mjx.put_model(self.model, device=device)
            self.data = mjx.put_data(self.model, self.host, device=device)
            self.initial_data = self.data

            def advance(data: mjx.Data, action: jax.Array) -> mjx.Data:
                data = data.replace(ctrl=action)

                def tick(current: mjx.Data, _: None) -> tuple[mjx.Data, None]:
                    return mjx.step(gpu_model, current), None

                result, _ = jax.lax.scan(tick, data, None, length=10)
                return result

            self.advance = cast(Callable[[mjx.Data, jax.Array], mjx.Data], jax.jit(advance))  # pyright: ignore[reportUnknownMemberType]

    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.host, 0)
        mujoco.mj_forward(self.model, self.host)
        self.data = self.initial_data
        self.last = np.zeros(12, dtype=np.float32)
        self.previous_last = self.last.copy()
        self.steps = 0
        self.judge = Goal(self.config, self.obstacle.end_x)
        self.camera.reset()

    def observe(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        frame = self.camera.observe(
            np.asarray(self.host.qpos), np.asarray(self.host.qvel), self.steps / self.config.control_hz
        )
        body = np.concatenate(
            (
                self.host.qpos[7:] - self.default,
                self.host.qvel,
                np.array(self.judge.command(yaw(tuple(float(q) for q in self.host.qpos[3:7])))),
                self.previous_last if self.config.base_policy_sha256 else self.last,
            )
        ).astype(np.float32)
        if body.shape != (45,) or not np.isfinite(body).all():
            raise ValueError("invalid visual policy proprioception")
        return frame.depth, body

    def step(self, action: NDArray[np.float32]) -> tuple[float, Outcome]:
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("invalid control action")
        previous_progress = self.judge.progress(float(self.host.qpos[0]), float(self.host.qpos[1]))
        bounded = np.clip(action, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]).astype(
            np.float32
        )
        beam = self.config.goal_kind in ("duck", "jump")
        hit_beam = False
        if self.device.platform == "cpu":
            self.host.ctrl[:] = bounded
            for _ in range(10):
                mujoco.mj_step(self.model, self.host)
                if beam and body_collision(self.model, self.host):
                    hit_beam = True
        else:
            if self.data is None or self.advance is None:
                raise RuntimeError("MJX state was not initialized")
            self.data = self.advance(self.data, jnp.asarray(bounded))
            self.host.qpos[:] = np.asarray(self.data.qpos)
            self.host.qvel[:] = np.asarray(self.data.qvel)
        if not np.isfinite(self.host.qpos).all() or not np.isfinite(self.host.qvel).all():
            raise ValueError("nonfinite physics state")
        self.previous_last = self.last.copy()
        self.last = bounded
        self.steps += 1
        self.host.time = self.steps / self.config.control_hz
        mujoco.mj_forward(self.model, self.host)
        upright = 1.0 - 2.0 * float(self.host.qpos[4] ** 2 + self.host.qpos[5] ** 2)
        outcome = self.judge.observe(
            float(self.host.qpos[0]),
            float(self.host.qpos[1]),
            float(self.host.qpos[2]),
            upright,
            hit_beam or body_collision(self.model, self.host),
            self.steps >= self.config.episode_steps,
            airborne=self.config.goal_kind == "jump" and robot_airborne(self.model, self.host),
            rear_x=minimum_robot_x(self.model, self.host),
            heading=yaw(tuple(float(q) for q in self.host.qpos[3:7])),
        )
        score = (
            10.0 * (self.judge.progress(float(self.host.qpos[0]), float(self.host.qpos[1])) - previous_progress)
            + 0.02 * upright
            - 0.00002 * float(np.sum(bounded**2))
        )
        score += 10.0 if outcome == "success" else -5.0 if outcome not in ("running", "success") else 0.0
        return score, outcome

    def close(self) -> None:
        self.camera.close()
