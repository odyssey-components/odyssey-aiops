"""Immutable Go1 source overlays and an actual robot-mounted depth sensor."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import mujoco
import numpy as np
from numpy.typing import NDArray
from OpenGL import GL

from .configuration import VisionConfig


@dataclass(frozen=True)
class Obstacle:
    x: float
    length: float
    height: float
    kind: str = "block"

    @property
    def end_x(self) -> float:
        return self.x + self.length / 2


def make_scene(source: Path, config: VisionConfig, seed: int) -> tuple[mujoco.MjModel, Obstacle]:
    spec = mujoco.MjSpec.from_file(str(source / "unitree_go1/scene_torque.xml"))
    trunk = next((body for body in spec.bodies if body.name == "trunk"), None)
    if trunk is None:
        raise ValueError("Go1 source is missing the trunk body")
    angle = math.radians(15)
    trunk.add_camera(
        name="go1_depth", pos=[0.30, 0, 0.03], xyaxes=[0, -1, 0, math.sin(angle), 0, math.cos(angle)], fovy=65
    )
    height = float(np.random.default_rng(seed).uniform(config.height_min, config.height_max))
    kind = ("step" if seed % 2 else "block") if config.scene_kind == "mixed" else config.scene_kind
    length = 1.0 if kind == "step" else 0.18
    obstacle = Obstacle(config.obstacle_x + length / 2, length, height, kind)
    if config.goal_kind in ("turn", "uturn"):
        distance = config.target_distance_m
        half = config.road_width_m / 2

        wall_half_thickness = 0.025

        def wall(name: str, cx: float, cy: float, length: float, angle: float = 0.0) -> None:
            spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[cx, cy, 0.2],
                size=[length / 2, wall_half_thickness, 0.2],
                rgba=[0.4, 0.4, 0.4, 1],
                contype=1,
                conaffinity=1,
                friction=[0.8, 0.005, 0.0001],
                quat=[math.cos(angle / 2), 0, 0, math.sin(angle / 2)],
            )

        if config.goal_kind == "uturn":
            for side in (-1, 1):
                wall(f"vision_side_{side}", (distance - 0.2) / 2, side * half, distance + 1.0)
            wall("vision_end", distance + 0.6, 0, config.road_width_m, math.pi / 2)
        else:
            angle = math.radians(config.turn_angle_deg)
            bend = half * math.tan(angle / 2)
            # Extend to the exposed face intersection to fill the notch between box ends.
            overlap = wall_half_thickness * abs(math.tan(angle / 2))
            entrance = min(-0.8, distance - abs(bend) - 0.3)
            end = max(1.3, abs(bend) + 0.3)
            for side in (-1, 1):
                # Meet at the boundary-line intersection, rather than clipping both sides equally.
                corner = distance - side * bend
                length = corner + overlap - entrance
                wall(f"vision_approach_{side}", entrance + length / 2, side * half, length)
                start = side * bend - overlap
                length = end - start
                offset = (start + end) / 2
                wall(
                    f"vision_exit_{side}",
                    distance + math.cos(angle) * offset - math.sin(angle) * side * half,
                    math.sin(angle) * offset + math.cos(angle) * side * half,
                    length,
                    angle,
                )
        obstacle = Obstacle(distance, 0.0, 0.0, config.goal_kind)
    elif config.goal_kind in ("duck", "jump"):
        duck = config.goal_kind == "duck"
        x, height = config.target_distance_m, config.beam_height_m
        spec.worldbody.add_geom(
            name="vision_beam",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, 0, height + 0.04 if duck else height / 2],
            size=[0.08, 0.6, 0.04 if duck else height / 2],
            rgba=[0.5, 0.4, 0.3, 1],
        )
        if duck:
            for side in (-1, 1):
                spec.worldbody.add_geom(
                    name=f"vision_beam_post_{side}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[x, side * 0.56, height / 2],
                    size=[0.08, 0.04, height / 2],
                    rgba=[0.5, 0.4, 0.3, 1],
                )
        obstacle = Obstacle(x, 0.16, height, config.goal_kind)
    elif config.goal_kind == "slope":
        angle = math.radians(config.slope_angle_deg)
        length = config.target_distance_m
        rise = math.sin(angle) * length
        spec.worldbody.add_geom(
            name="vision_ramp",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[config.obstacle_x + math.cos(angle) * length / 2, 0, rise / 2],
            size=[length / 2, 0.6, 0.025],
            rgba=[0.5, 0.4, 0.3, 1],
            contype=1,
            conaffinity=1,
            friction=[0.8, 0.005, 0.0001],
            quat=[math.cos(angle / 2), 0, -math.sin(angle / 2), 0],
        )
        edge = config.obstacle_x + math.cos(angle) * length
        spec.worldbody.add_geom(
            name="vision_plateau",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[edge + 1.0, 0, rise / 2],
            size=[1.0, 0.6, rise / 2],
            rgba=[0.5, 0.4, 0.3, 1],
            contype=1,
            conaffinity=1,
            friction=[0.8, 0.005, 0.0001],
        )
        obstacle = Obstacle(config.obstacle_x + math.cos(angle) * length / 2, math.cos(angle) * length, rise, "slope")
    elif config.goal_kind != "flat":
        spec.worldbody.add_geom(
            name="vision_obstacle",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[obstacle.x, 0, height / 2],
            size=[obstacle.length / 2, 0.6, height / 2],
            rgba=[0.5, 0.4, 0.3, 1],
            contype=1,
            conaffinity=1,
            friction=[0.8, 0.005, 0.0001],
        )
    model = spec.compile()
    if (model.nq, model.nv, model.nu) != (19, 18, 12):
        raise ValueError("visual task requires the original 12 actuator Go1")
    return model, obstacle


@dataclass(frozen=True)
class DepthFrame:
    time: float
    depth: NDArray[np.float32]


class DepthCamera:
    def __init__(self, model: mujoco.MjModel, config: VisionConfig):
        self.model, self.config = model, config
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=config.height, width=config.width)
        self.renderer.enable_depth_rendering()
        self.frame: DepthFrame | None = None
        self.closed = False

    def observe(self, qpos: NDArray[np.float64], qvel: NDArray[np.float64], time: float) -> DepthFrame:
        if self.closed:
            raise RuntimeError("depth camera is closed")
        if not np.isfinite(time) or time < 0 or (self.frame and time < self.frame.time - 1e-9):
            raise ValueError("nonmonotonic camera time; reset the episode explicitly")
        if (
            qpos.shape != (self.model.nq,)
            or qvel.shape != (self.model.nv,)
            or not np.isfinite(qpos).all()
            or not np.isfinite(qvel).all()
        ):
            raise ValueError("invalid physics state")
        if self.frame and time - self.frame.time < 1 / self.config.camera_hz - 1e-9:
            return self.frame
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.time = time
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="go1_depth")
        depth = np.asarray(self.renderer.render(), dtype=np.float32)
        if depth.shape != (self.config.height, self.config.width) or not np.isfinite(depth).all():
            raise ValueError("invalid depth frame")
        normalized = (np.clip(depth, self.config.near, self.config.far) / self.config.far)[..., None].copy()
        normalized.setflags(write=False)
        self.frame = DepthFrame(time, normalized)
        return self.frame

    def graphics_identity(self) -> tuple[str, str]:
        vendor = GL.glGetString(GL.GL_VENDOR)
        renderer = GL.glGetString(GL.GL_RENDERER)
        if not vendor or not renderer:
            raise RuntimeError("graphics device identity unavailable")
        return vendor.decode(), renderer.decode()

    def reset(self) -> None:
        self.frame = None

    def close(self) -> None:
        if not self.closed:
            self.renderer.close()
            self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def obstacle_contact(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[str, str] | None:
    for index in range(data.ncon):
        contact = data.contact[index]
        for obstacle, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
            obstacle_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(obstacle)) or ""
            if not obstacle_name.startswith("vision_"):
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(other)) or ""
            if obstacle_name.startswith("vision_beam") or name not in ("FR", "FL", "RR", "RL"):
                body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other])) or name
                return obstacle_name, body
    return None


def body_collision(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    return obstacle_contact(model, data) is not None


def minimum_robot_x(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    collidable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    robot = (model.geom_bodyid > 0) & collidable
    if not robot.any():
        raise ValueError("robot has no collision geometry")
    return float(np.min(data.geom_xpos[robot, 0] - model.geom_rbound[robot]))


def robot_airborne(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """No active contact between the robot and the world."""
    for contact in data.contact:
        a, b = int(contact.geom1), int(contact.geom2)
        if contact.efc_address >= 0 and (model.geom_bodyid[a] == 0) != (model.geom_bodyid[b] == 0):
            return False
    return True
