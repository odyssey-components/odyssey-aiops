"""CPU inference for the pinned Robot Parkour Learning Go1 student, with a calibrated depth input."""

import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Protocol, cast

import mujoco
import numpy as np
from numpy.typing import NDArray

from .walk_these_ways import DEFAULT, BodyOrientation, PositionController

FORMAT = "go1-robot-parkour/v1"
UPSTREAM = "5e1f4136f7fec9dac2741add63ca58d32090666f"
CHECKPOINT_SHA256 = "f896b72602a0e89560d3eab7fb10694fcb7cd495dbca524a404faa7d83709232"
LOW = np.tile(np.array([-1.894, -2.526, -2.042, -1.494, -2.526, -2.042], dtype=np.float32), 2)
LOW[[7, 10]] = -2.926
HIGH = np.tile(np.array([1.494, 3.932, 0.926, 1.894, 3.932, 0.926], dtype=np.float32), 2)
HIGH[[7, 10]] = 3.532
TORQUE = np.tile(np.array([20, 20, 25], dtype=np.float32), 4)
LAYERS = {
    "visual_encoder.conv.conv.0": (16, 1, 5, 5),
    "visual_encoder.conv.conv.3": (32, 16, 4, 4),
    "visual_encoder.conv.conv.6": (32, 32, 3, 3),
    "visual_encoder.head.model.0": (128, 2464),
    "visual_encoder.head.model.2": (128, 128),
    "actor.0": (512, 256),
    "actor.2": (256, 512),
    "actor.4": (128, 256),
    "actor.6": (12, 128),
}
SHAPES = {
    **{
        f"{layer}.{name}": shape if name == "weight" else (shape[0],)
        for layer, shape in LAYERS.items()
        for name in ("weight", "bias")
    },
    "memory_a.rnn.weight_ih_l0": (768, 176),
    "memory_a.rnn.weight_hh_l0": (768, 256),
    "memory_a.rnn.bias_ih_l0": (768,),
    "memory_a.rnn.bias_hh_l0": (768,),
}


class CameraView(Protocol):
    pos: NDArray[np.float64]
    quat: NDArray[np.float64]
    fovy: NDArray[np.float64]


class RotationFunctions(Protocol):
    def mju_mat2Quat(self, output: NDArray[np.float64], matrix: NDArray[np.float64]) -> None: ...


class CameraLayout(Protocol):
    def camera(self, name: str) -> CameraView: ...


def configure_camera(model: mujoco.MjModel) -> None:
    camera = cast(CameraLayout, model).camera("go1_depth")
    pitch = 0.52
    camera.pos[:] = [0.272, 0.0075, 0.092]
    axes = np.array([[0, math.sin(pitch), -math.cos(pitch)], [-1, 0, 0], [0, math.cos(pitch), math.sin(pitch)]])
    cast(RotationFunctions, mujoco).mju_mat2Quat(camera.quat, axes.reshape(-1))
    camera.fovy[0] = math.degrees(2 * math.atan(math.tan(math.radians(43)) * 60 / 106))


def pooling_axis(source: int, target: int) -> NDArray[np.float32]:
    result = np.zeros((target, source), dtype=np.float32)
    for index in range(target):
        start = index * source // target
        end = ((index + 1) * source + target - 1) // target
        result[index, start:end] = 1 / (end - start)
    return result


VERTICAL = pooling_axis(60, 48)
HORIZONTAL = pooling_axis(80, 64)


def weights(path: Path) -> dict[str, NDArray[np.float32]]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(SHAPES):
            raise ValueError("unexpected Robot Parkour Learning tensors")
        arrays = {name: np.asarray(archive[name], dtype=np.float32) for name in SHAPES}
    if any(value.shape != SHAPES[name] or not np.isfinite(value).all() for name, value in arrays.items()):
        raise ValueError("invalid Robot Parkour Learning tensor")
    return arrays


class Parkour(PositionController):
    def __init__(self, directory: Path, source: str, model: mujoco.MjModel) -> None:
        super().__init__(model)
        manifest = json.loads((directory / "public-policy.json").read_text())
        path = directory / "policy.npz"
        if path.is_symlink() or path.stat().st_size > 8 << 20:
            raise ValueError("invalid Robot Parkour Learning weights")
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            manifest.get("format") != FORMAT
            or manifest.get("source_sha256") != source
            or manifest.get("policy_sha256") != self.digest
        ):
            raise ValueError("Robot Parkour Learning model/source checksum mismatch")
        self.params = weights(path)
        self.hidden = np.zeros(256, dtype=np.float32)
        self.last = np.zeros(12, dtype=np.float32)
        self.network_state = np.zeros(176, dtype=np.float32)
        self.proprio: deque[NDArray[np.float32]] = deque(maxlen=3)
        self.vision: deque[NDArray[np.float32]] = deque(maxlen=15)
        self.last_image: NDArray[np.float32] | None = None
        self.last_latent = np.zeros(128, dtype=np.float32)

    def linear(self, x: NDArray[np.float32], name: str) -> NDArray[np.float32]:
        return self.params[name + ".weight"] @ x + self.params[name + ".bias"]

    def encode(self, depth: NDArray[np.float32]) -> NDArray[np.float32]:
        x = depth.reshape(1, 48, 64)
        for index in (0, 3, 6):
            key = f"visual_encoder.conv.conv.{index}"
            weight = self.params[key + ".weight"]
            kernel = weight.shape[-1]
            windows = np.lib.stride_tricks.sliding_window_view(x, (kernel, kernel), axis=(-2, -1))
            cols = windows.transpose(1, 2, 0, 3, 4).reshape(-1, int(np.prod(weight.shape[1:])))
            width, height = windows.shape[2], windows.shape[1]
            value = cols @ weight.reshape(weight.shape[0], -1).T + self.params[key + ".bias"]
            x = np.maximum(value, value * 0.01).reshape(height, width, -1).transpose(2, 0, 1)
            if index < 6:
                c, h, w = x.shape
                x = x[:, : h // 2 * 2, : w // 2 * 2].reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))
        head = self.linear(x.reshape(-1), "visual_encoder.head.model.0")
        return self.linear(np.maximum(head, head * 0.01), "visual_encoder.head.model.2")

    def advance(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        self.network_state = features
        prefix = "memory_a.rnn."
        incoming = self.params[prefix + "weight_ih_l0"] @ features + self.params[prefix + "bias_ih_l0"]
        recurrent = self.params[prefix + "weight_hh_l0"] @ self.hidden + self.params[prefix + "bias_hh_l0"]
        ir, iz, in_ = np.split(incoming, 3)
        hr, hz, hn = np.split(recurrent, 3)
        reset = 1 / (1 + np.exp(-np.clip(ir + hr, -80, 80)))
        update = 1 / (1 + np.exp(-np.clip(iz + hz, -80, 80)))
        self.hidden = (1 - update) * np.tanh(in_ + reset * hn) + update * self.hidden
        x = self.hidden
        for index in (0, 2, 4):
            x = self.linear(np.asarray(x, dtype=np.float32), f"actor.{index}")
            x = np.where(x >= 0, x, np.expm1(np.minimum(x, 0)))
        return self.linear(np.asarray(x, dtype=np.float32), "actor.6")

    def observe(
        self, data: mujoco.MjData, command: tuple[float, float, float], depth: NDArray[np.float32],
        *, preprocessed: bool = False,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        rotation = cast(BodyOrientation, data).xmat[self.trunk].reshape(3, 3)
        vx, vy, yaw = command
        proprio = np.concatenate(
            (
                np.zeros(3),
                data.qvel[3:6] * 0.25,
                -rotation[2],
                [vx * 2, vy * 2, yaw * 0.25],
                data.qpos[7:][self.order] - DEFAULT,
                data.qvel[6:][self.order] * 0.05,
                self.last,
            )
        )
        self.proprio.append(proprio.astype(np.float32))
        buffered = self.proprio[0].copy()
        buffered[-12:] = self.last
        if preprocessed:
            self.depth = depth
            self.last_latent = self.encode(depth)
        else:
            self.vision.append(depth)
            image = self.vision[0]
            if image is not self.last_image:
                self.depth = np.asarray((VERTICAL @ image[:, 15:95, 0] @ HORIZONTAL.T)[..., None], dtype=np.float32)
                self.last_latent = self.encode(self.depth)
                self.last_image = image
        raw = self.advance(np.concatenate((buffered, self.last_latent)))
        self.last = np.clip(raw, LOW, HIGH)
        return raw, self.depth

    def motor(self, model: mujoco.MjModel, data: mujoco.MjData) -> NDArray[np.float32]:
        return self.motor_targets(model, data, self.last * 0.5, 40, TORQUE)
