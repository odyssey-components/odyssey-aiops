"""Bounded real MuJoCo CPU simulation with no gradient updates."""

import hashlib
import json
import struct
from pathlib import Path
from typing import Protocol, cast

import mujoco
import numpy as np
from numpy.typing import NDArray

from ..locomotion.replay import write_replay
from ..runtime.provenance import model_digest
from ..vision.goals import Goal, yaw
from ..vision.scene import DepthCamera, body_collision, make_scene, minimum_robot_x, obstacle_contact, robot_airborne
from .configuration import SimulationConfig
from .parkour import FORMAT as PARKOUR_FORMAT
from .parkour import Parkour, configure_camera
from .policy import read_policy
from .recording import camera_contract, read_injected_depth, record
from .walk_these_ways import WalkTheseWays
from .walk_turn import WalkTurn


class _Keyframes(Protocol):
    nkey: int


def run(
    source: Path, model_directory: Path, output: Path, config: SimulationConfig, recording: Path | None = None
) -> dict[str, object]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite simulation output")
    source_hash = model_digest(source / "unitree_go1")
    network_state = np.empty(45, dtype=np.float32)

    def capture_state(state: NDArray[np.float32]) -> None:
        network_state[:] = state

    public_manifest = model_directory / "public-policy.json"
    external = public_manifest.is_file()
    parkour = external and json.loads(public_manifest.read_text()).get("format") == PARKOUR_FORMAT
    if external and not parkour and recording is not None:
        raise ValueError("public position policies are available for simulation only")
    action, base, policy_hash = (
        (None, None, "") if external else read_policy(model_directory, source_hash, capture_state)
    )
    environment = config.environment(base)
    if parkour:
        environment = environment.model_copy(update={"near": 0.02, "far": 2.0})
        if environment.goal_kind not in ("duck", "jump", "flat"):
            raise ValueError("Robot Parkour Learning uses forward obstacle scenes or a flat diagnostic")
    recorded_camera = camera_contract(environment)
    injection_camera = recorded_camera if parkour else camera_contract(base)
    injected = (
        read_injected_depth(recording, injection_camera, source_hash, parkour=parkour)
        if recording is not None
        else None
    )
    model, obstacle = make_scene(source, environment, config.seed)
    if (model.nq, model.nv, model.nu) != (19, 18, 12) or abs(model.opt.timestep - 0.002) > 1e-9:
        raise ValueError("unexpected Go1 simulation model")
    data = mujoco.MjData(model)
    if cast(_Keyframes, model).nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    controller = (
        Parkour(model_directory, source_hash, model)
        if parkour
        else WalkTheseWays(model_directory, source_hash, model)
        if external
        else None
    )
    camera_config = environment
    if parkour:
        configure_camera(model)
        camera_config = environment.model_copy(update={"width": 106, "height": 60})
    if controller:
        controller.reset_pose(data)
        policy_hash = controller.digest
    mujoco.mj_forward(model, data)
    default = np.asarray(data.qpos[7:]).copy()
    rng = np.random.default_rng(config.seed)
    data.qpos[2] += rng.uniform(-0.005, 0.005)
    data.qpos[7:] += rng.uniform(-0.025, 0.025, 12)
    mujoco.mj_forward(model, data)
    last = np.zeros(12, dtype=np.float32)
    goal = Goal(environment, obstacle.end_x)
    corner = (
        WalkTurn(goal)
        if isinstance(controller, WalkTheseWays)
        and environment.goal_kind == "turn"
        and environment.road_width_m >= 0.8
        and abs(environment.turn_angle_deg) <= 90
        else None
    )
    poses: list[NDArray[np.float64]] = []
    has_grounded = False
    first_airborne: dict[str, float] | None = None
    first_contact: dict[str, str | float] | None = None
    output.mkdir(parents=True)
    camera = None if injected is not None else DepthCamera(model, camera_config)
    limit = min(config.steps, len(injected)) if injected is not None else config.steps
    try:
        with record(
            output / "policy-input.mcap",
            {
                "policy_sha256": policy_hash,
                "source_sha256": source_hash,
                "camera": recorded_camera,
                **({"depth_stage": "parkour-encoder/v1"} if parkour else {}),
                "simulation": config.model_dump(mode="json"),
                "action_space": "joint-position-offset" if controller else "motor-control",
            },
            network_dimension=176 if parkour else 2102 if controller else 45,
        ) as append:
            for step in range(1, limit + 1):
                if injected is not None:
                    depth = injected[step - 1]
                else:
                    assert camera is not None
                    depth = camera.observe(np.asarray(data.qpos), np.asarray(data.qvel), (step - 1) / 50).depth
                heading = yaw(tuple(float(q) for q in data.qpos[3:7]))
                command = (
                    corner.command(float(data.qpos[0]), float(data.qpos[1]), heading)
                    if corner
                    else goal.command(heading)
                )
                body = np.concatenate((data.qpos[7:] - default, data.qvel, np.array(command), last)).astype(np.float32)
                if controller:
                    if isinstance(controller, Parkour):
                        control, depth = controller.observe(data, command, depth, preprocessed=injected is not None)
                    else:
                        control = controller.observe(data, command)
                    applied = controller.motor(model, data)
                    inputs = controller.network_state
                else:
                    assert action is not None
                    control = action(body, depth if base is not None else None)
                    applied = np.asarray(
                        np.clip(control, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]),
                        dtype=np.float32,
                    )
                    inputs = network_state
                if control.shape != (12,) or not np.isfinite(control).all():
                    raise ValueError("nonfinite or invalid simulation action")
                last = applied
                append(step - 1, body, inputs, depth, control, applied)
                beam = environment.goal_kind in ("duck", "jump")
                hit_beam = False
                for _ in range(10):
                    data.ctrl[:] = controller.motor(model, data) if controller else applied
                    mujoco.mj_step(model, data)
                    if beam:
                        contact = obstacle_contact(model, data)
                        if contact is not None:
                            hit_beam = True
                            if first_contact is None:
                                first_contact = {
                                    "time_seconds": float(data.time),
                                    "x_m": float(data.qpos[0]),
                                    "obstacle": contact[0],
                                    "robot_body": contact[1],
                                }
                        if environment.goal_kind == "jump" and first_airborne is None:
                            airborne = robot_airborne(model, data)
                            if not airborne:
                                has_grounded = True
                            elif has_grounded:
                                first_airborne = {"time_seconds": float(data.time), "x_m": float(data.qpos[0])}
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError("nonfinite simulation state")
                mujoco.mj_forward(model, data)
                poses.append(np.asarray(data.qpos).copy())
                heading = yaw(tuple(float(q) for q in data.qpos[3:7]))
                if corner:
                    corner.enter_corner(float(data.qpos[0]), heading)
                outcome = goal.observe(
                    float(data.qpos[0]),
                    float(data.qpos[1]),
                    float(data.qpos[2]),
                    1 - 2 * float(data.qpos[4] ** 2 + data.qpos[5] ** 2),
                    hit_beam or body_collision(model, data),
                    step == limit,
                    airborne=environment.goal_kind == "jump" and robot_airborne(model, data),
                    rear_x=minimum_robot_x(model, data),
                    heading=heading,
                )
                if step % 10 == 0 or outcome != "running":
                    print(json.dumps({"event": "simulation.progress", "step": step, "total": config.steps}), flush=True)
                if outcome != "running":
                    break
    finally:
        if camera is not None:
            camera.close()
    if not poses:
        raise ValueError("simulation produced no frames")
    write_replay(output / "replay.json", poses, [False] * (len(poses) - 1) + [True], 0.02)
    result: dict[str, object] = {
        "format": "go1-simulation-result/v1",
        "outcome": goal.result,
        "passed": goal.result == "success",
        "criteria": {
            "stable_seconds": environment.stable_steps / 50,
            "exit_distance_m": environment.success_exit_distance_m,
            "return_tolerance_m": environment.success_return_tolerance_m,
            "heading_tolerance_deg": environment.success_heading_tolerance_deg,
            "clearance_m": environment.success_clearance_m,
        },
        "final_progress_m": goal.progress(float(data.qpos[0]), float(data.qpos[1])),
        "steps": len(poses),
        "duration_seconds": len(poses) * 0.02,
        "seed": config.seed,
        "goal_kind": config.scene.kind,
        "backend": "mujoco-cpu",
        "policy_sha256": policy_hash,
        "source_sha256": source_hash,
        "minimum_height_m": min(float(p[2]) for p in poses),
        "final_position_m": [float(v) for v in poses[-1][:3]],
        "trained": False,
    }
    if first_airborne is not None:
        result["first_airborne"] = first_airborne
    if first_contact is not None:
        result["first_obstacle_contact"] = first_contact
    if injected is not None:
        result["vision_injected"] = True
        result["vision_input_exhausted"] = goal.result == "timeout" and len(poses) == len(injected)
        with (output / "injected-depth.bin").open("xb") as stream:
            stream.write(struct.pack("<II", environment.width, environment.height))
            for frame in injected[: len(poses)]:
                stream.write(frame.astype("<f4", copy=False).tobytes())
    if config.scene.kind == "flat":
        result["final_yaw_deg"] = float(np.degrees(yaw(poses[-1][3:7].tolist())))
    with (output / "policy-input.mcap").open("rb") as recording_file:
        result["recording_sha256"] = hashlib.file_digest(recording_file, "sha256").hexdigest()
    (output / "result.json").write_text(json.dumps(result, allow_nan=False) + "\n")
    return result
