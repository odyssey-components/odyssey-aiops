"""CPU adapter for the pinned Walk These Ways Go1 policy and position controller."""

import hashlib
import json
from itertools import pairwise
from pathlib import Path
from typing import Protocol, cast

import mujoco
import numpy as np
from numpy.typing import NDArray

FORMAT = "go1-walk-these-ways/v1"
UPSTREAM = "0e7236bdc81ce855cbe3d70345a7899452bdeb1c"
JOINTS = tuple(f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf"))
DEFAULT = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.float32)
LAYERS = {"adaptation": (2100, 256, 128, 2), "body": (2102, 512, 256, 128, 12)}
DIGESTS = {
    "adaptation": "f0adb113954058a37d2596282f041a3bb128365558ecf64405bbc1353d5dc1b0",
    "body": "6c14b59ca28550a012de3bf5d21fc7d6aa6b6d38eab4dc31e172519f12a19ece",
}


class JointView(Protocol):
    id: int
    qposadr: NDArray[np.int32]


class BodyView(Protocol):
    id: int


class ModelLayout(Protocol):
    actuator_gear: NDArray[np.float64]

    def joint(self, name: str) -> JointView: ...
    def body(self, name: str) -> BodyView: ...


class BodyOrientation(Protocol):
    xmat: NDArray[np.float64]


def weights(path: Path) -> dict[str, NDArray[np.float32]]:
    expected = {
        f"{name}.{2 * i}.{key}": (out, inp) if key == "weight" else (out,)
        for name, dims in LAYERS.items()
        for i, (inp, out) in enumerate(pairwise(dims))
        for key in ("weight", "bias")
    }
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 8 << 20:
        raise ValueError("invalid Walk These Ways weights")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("unexpected Walk These Ways tensors")
        result = {name: np.asarray(archive[name], dtype=np.float32) for name in expected}
    if any(value.shape != expected[name] or not np.isfinite(value).all() for name, value in result.items()):
        raise ValueError("invalid Walk These Ways tensor")
    return result


class PositionController:
    def __init__(self, model: mujoco.MjModel) -> None:
        layout = cast(ModelLayout, model)
        self.order = np.asarray([int(layout.joint(name).qposadr[0]) - 7 for name in JOINTS], dtype=np.intp)
        self.inverse = np.argsort(self.order)
        expected = [layout.joint(JOINTS[i]).id for i in self.inverse]
        if model.actuator_trnid[:, 0].tolist() != expected:
            raise ValueError("Walk These Ways requires matching Go1 joint/actuator ordering")
        self.gear = layout.actuator_gear[:, 0].copy()
        self.trunk = layout.body("trunk").id

    def reset_pose(self, data: mujoco.MjData) -> None:
        data.qpos[7:] = DEFAULT[self.inverse]

    def motor_targets(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        offset: NDArray[np.float32],
        kp: float,
        torque_limit: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        target = (DEFAULT + offset)[self.inverse]
        torque = kp * (target - data.qpos[7:]) - 0.5 * data.qvel[6:]
        if torque_limit is not None:
            torque = np.clip(torque, -torque_limit, torque_limit)
        return np.asarray(
            np.clip(torque / self.gear, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]),
            dtype=np.float32,
        )


class WalkTheseWays(PositionController):
    def __init__(self, directory: Path, source: str, model: mujoco.MjModel) -> None:
        super().__init__(model)
        manifest = json.loads((directory / "public-policy.json").read_text())
        path = directory / "policy.npz"
        self.params = weights(path)
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            manifest.get("format") != FORMAT
            or manifest.get("source_sha256") != source
            or manifest.get("policy_sha256") != self.digest
        ):
            raise ValueError("Walk These Ways model/source checksum mismatch")
        self.history = np.zeros((30, 70), dtype=np.float32)
        self.previous = np.zeros(12, dtype=np.float32)
        self.last = self.previous.copy()
        self.steps = 0
        self.network_state = np.zeros(2102, dtype=np.float32)

    def forward(self, name: str, input_: NDArray[np.float32]) -> NDArray[np.float32]:
        value = input_
        final = 2 * (len(LAYERS[name]) - 2)
        for index in range(0, final + 1, 2):
            prefix = f"{name}.{index}"
            value = self.params[f"{prefix}.weight"] @ value + self.params[f"{prefix}.bias"]
            if index != final:
                value = np.where(value >= 0, value, np.expm1(np.minimum(value, 0)))
        return value

    def observe(self, data: mujoco.MjData, command: tuple[float, float, float]) -> NDArray[np.float32]:
        rotation = cast(BodyOrientation, data).xmat[self.trunk].reshape(3, 3)
        gravity = -rotation[2]
        vx, vy, yaw = command
        commands = [2 * vx, 2 * vy, 0.25 * yaw, 0, 3, 0.5, 0, 0, 0.5, 0.08 * 0.15, 0, 0, 0.25, 0.38, 0]
        clocks = np.sin(2 * np.pi * (self.steps * 0.06 + np.array([0.5, 0, 0, 0.5])))
        self.history[:-1] = self.history[1:]
        self.history[-1] = np.concatenate(
            (
                gravity,
                commands,
                data.qpos[7:][self.order] - DEFAULT,
                data.qvel[6:][self.order] * 0.05,
                self.last,
                self.previous,
                clocks,
            )
        )
        inputs = self.history.reshape(-1)
        latent = self.forward("adaptation", inputs)
        self.network_state = np.concatenate((inputs, latent))
        raw = self.forward("body", self.network_state)
        self.previous, self.last = self.last, np.clip(raw, -10, 10)
        self.steps += 1
        return raw

    def motor(self, model: mujoco.MjModel, data: mujoco.MjData) -> NDArray[np.float32]:
        offset = self.last * 0.25
        offset[::3] *= 0.5
        return self.motor_targets(model, data, offset, 20)
