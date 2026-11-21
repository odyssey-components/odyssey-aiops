"""Versioned command training environment: upstream observations, corrected pose/contact health."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from numpy.typing import NDArray


class Environment:
    def __init__(self, source: Path, device: jax.Device, seed: int, speed: float, random_commands: int = 0):
        self.model = mujoco.MjModel.from_xml_path(str(source / "unitree_go1/scene_torque.xml"))
        if (self.model.nq, self.model.nv, self.model.nu) != (19, 18, 12) or abs(self.model.opt.timestep - 0.002) > 1e-9:
            raise ValueError("unexpected Go1 model")
        self.host = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.host, 0)
        self.default = np.asarray(self.host.qpos).copy()
        self.rng = np.random.default_rng(seed)
        self.command = np.array([speed, 0.0, 0.0])
        self.random_commands = random_commands
        self.feet = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("FR", "FL", "RR", "RL")]
        if min(self.feet) < 0:
            raise ValueError("Go1 foot sites missing")
        self.advance: Callable[[mjx.Data, jax.Array], mjx.Data] | None = None
        self.data: mjx.Data | None = None
        if device.platform == "gpu":
            model = mjx.put_model(self.model, device=device)

            def advance(data: mjx.Data, control: jax.Array) -> mjx.Data:
                def tick(current: mjx.Data, _: None) -> tuple[mjx.Data, None]:
                    return mjx.step(model, current), None

                result, _ = jax.lax.scan(tick, data.replace(ctrl=control), None, length=10)
                return result

            self.advance = cast(Callable[[mjx.Data, jax.Array], mjx.Data], jax.jit(advance))  # pyright: ignore[reportUnknownMemberType]
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.host, 0)
        self.host.qpos[7:] += self.rng.uniform(-0.025, 0.025, 12)
        mujoco.mj_forward(self.model, self.host)
        self.data = mjx.put_data(self.model, self.host) if self.advance else None
        self.last = np.zeros(12, dtype=np.float32)
        self.previous_last = self.last.copy()
        self.air = np.zeros(4)
        self.contacts = np.zeros(4, dtype=np.bool_)
        self.steps = 0
        if self.random_commands:
            self.sample_command()

    def sample_command(self) -> None:
        if self.random_commands == 2:
            speed = float(self.rng.choice([0.0, 0.0, 0.3]))
            yaw_rate = 0.0
        elif self.random_commands == 3:
            speed = 0.3
            yaw_rate = float(self.rng.choice([-0.8, -0.4, 0.0, 0.4, 0.8]))
        else:
            speed = float(self.rng.choice([0.0, 0.3, 0.6]))
            yaw_rate = float(self.rng.choice([-0.8, -0.4, 0.0, 0.4, 0.8]))
        self.command[:] = (speed, 0.0, yaw_rate)

    def observe(self) -> NDArray[np.float32]:
        # Match upstream's observation-before-last_action-update order, not just its dimensions.
        return np.concatenate(
            (self.host.qpos[7:] - self.default[7:], self.host.qvel, self.command, self.previous_last)
        ).astype(np.float32)

    def step(self, action: NDArray[np.float32]) -> tuple[float, bool]:
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("invalid base action")
        bounded = np.clip(action, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        if self.advance:
            if self.data is None:
                raise RuntimeError("MJX state missing")
            self.data = self.advance(self.data, jnp.asarray(bounded))
            self.host.qpos[:] = np.asarray(self.data.qpos)
            self.host.qvel[:] = np.asarray(self.data.qvel)
        else:
            self.host.ctrl[:] = bounded
            for _ in range(10):
                mujoco.mj_step(self.model, self.host)
        mujoco.mj_forward(self.model, self.host)
        if not np.isfinite(self.host.qpos).all() or not np.isfinite(self.host.qvel).all():
            raise ValueError("nonfinite physics")
        q = self.host.qpos[3:7]
        roll = np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        pitch = np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1, 1))
        healthy = 0.19 <= self.host.qpos[2] <= 0.65 and abs(roll) <= np.pi / 12 and abs(pitch) <= np.pi / 12
        touching: set[int] = set()
        for i in range(self.host.ncon):
            item = self.host.contact[i]
            if self.model.geom_bodyid[item.geom1] == 0:
                touching.add(item.geom2)
            if self.model.geom_bodyid[item.geom2] == 0:
                touching.add(item.geom1)
        contact = np.array([foot in touching for foot in self.feet], dtype=np.bool_)
        filtered = contact | self.contacts
        first = (self.air > 0) & filtered
        self.air += 0.02
        air_reward = float(np.sum((self.air - 0.5) * first)) * (np.linalg.norm(self.command[:2]) > 0.1)
        self.air *= ~filtered
        self.contacts = contact
        vel = self.host.qvel
        tracking = 2 * np.exp(-np.sum((self.command[:2] - vel[:2]) ** 2) / 0.25) + 0.5 * np.exp(
            -((self.command[2] - vel[5]) ** 2) / 0.25
        )
        if self.random_commands:
            # A tighter yaw term distinguishes holding still from slow commanded turns.
            tracking += 2.0 * np.exp(-((self.command[2] - vel[5]) ** 2) / 0.09)
        if self.random_commands == 2 and self.command[0] == 0:
            tracking += 4.0 * np.exp(-np.sum(vel[:2] ** 2) / 0.10)
        if self.random_commands == 3 and self.command[2] != 0:
            # Match the commanded rate and penalize turning against it.
            tracking += 3.0 * np.exp(-((self.command[2] - vel[5]) ** 2) / 0.16)
            tracking += 1.5 * np.clip(np.sign(self.command[2]) * vel[5], -1.0, abs(self.command[2]))
        limits = self.model.jnt_range[1:]
        center, half = limits.mean(axis=1), (limits[:, 1] - limits[:, 0]) * 0.45
        joint_cost = np.maximum(np.abs(self.host.qpos[7:] - center) - half, 0).sum()
        costs = (
            0.0002 * np.sum(bounded**2)
            + 0.5 * vel[2] ** 2
            + 0.02 * np.sum(vel[3:5] ** 2)
            + 0.01 * np.sum((self.last - bounded) ** 2)
            + 10 * joint_cost
        )
        score = float((tracking + air_reward - costs) * 0.02) - (0 if healthy else 2)
        self.previous_last = self.last.copy()
        self.last = bounded.astype(np.float32)
        self.steps += 1
        if self.random_commands and self.steps % 150 == 0:
            self.sample_command()
        return score, not healthy or self.steps >= 750
