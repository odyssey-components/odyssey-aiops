"""Independent Go1 depth PPO CLI; training fails closed without a JAX GPU."""

import argparse
import hashlib
import json
import os
import subprocess
import time
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from numpy.typing import NDArray
from odylab_training.continuous_ppo import generalized_advantage

from ..locomotion.replay import write_replay
from ..runtime.devices import MODES, configure_device, select_device
from ..runtime.local import COMMIT
from ..runtime.provenance import model_digest, require_source
from ..upstream.artifacts import load as load_base
from .artifacts import load_policy, save_policy, write_toml
from .configuration import VisionConfig
from .goals import Goal, yaw
from .policy import Adam, Params, adam, distribution_log_std, forward, initialize, log_probability, update
from .rollout import VisualEnvironment
from .scene import DepthCamera, body_collision, make_scene, minimum_robot_x, robot_airborne


def probe(source: Path, output: Path, config: VisionConfig, source_hash: str) -> None:
    model, obstacle = make_scene(source, config, config.test_seeds[0])
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    judge = Goal(config, obstacle.end_x)
    times: list[float] = []
    poses: list[NDArray[np.float64]] = []
    depths: list[NDArray[np.float32]] = []
    last_time = -1.0
    with DepthCamera(model, config) as camera:
        for index in range(min(config.episode_steps, 100)):
            time = index / config.control_hz
            frame = camera.observe(np.asarray(data.qpos), np.asarray(data.qvel), time)
            if frame.time != last_time:
                times.append(frame.time)
                depths.append(frame.depth.copy())
                poses.append(np.asarray(data.qpos).copy())
                last_time = frame.time
            data.ctrl[:] = 0
            for _ in range(10):
                mujoco.mj_step(model, data)
            upright = 1.0 - 2.0 * float(data.qpos[4] ** 2 + data.qpos[5] ** 2)
            result = judge.observe(
                float(data.qpos[0]),
                float(data.qpos[1]),
                float(data.qpos[2]),
                upright,
                body_collision(model, data),
                index == 99,
                rear_x=minimum_robot_x(model, data),
                airborne=obstacle.kind == "jump" and robot_airborne(model, data),
                heading=yaw(tuple(float(q) for q in data.qpos[3:7])),
            )
            if result != "running":
                break
        # Queried from the active rendering context, never inferred from PCI names.
        vendor, renderer = camera.graphics_identity()
    with (output / "sensor-probe.npz").open("xb") as stream:
        np.savez_compressed(stream, time=np.asarray(times), qpos=np.asarray(poses), depth=np.asarray(depths))
    write_toml(
        output / "probe.toml",
        {
            "format": "go1-depth-probe/v1",
            "source_sha256": source_hash,
            "mode": "sensor-probe-only",
            "physics": "mujoco-cpu",
            "graphics_vendor": vendor,
            "graphics_renderer": renderer,
            "depth_frames": len(times),
            "depth_height": config.height,
            "depth_width": config.width,
            "camera_hz": config.camera_hz,
            "scene_kind": obstacle.kind,
            "obstacle_height_m": obstacle.height,
            "outcome": judge.result,
            "trained": False,
        },
    )
    print(
        json.dumps(
            {
                "probe": "complete",
                "graphics_renderer": renderer,
                "depth_frames": len(times),
                "outcome": judge.result,
                "trained": False,
            }
        )
    )


def evaluate(
    source: Path, output: Path, config: VisionConfig, params: Params, device: jax.Device, policy_directory: Path
) -> None:
    outcomes: list[str] = []
    for seed in config.test_seeds:
        env = VisualEnvironment(source, config, seed, device)
        poses: list[NDArray[np.float64]] = []
        try:
            while env.judge.result == "running":
                depth, body = env.observe()
                mean, _ = forward(params, jnp.asarray(depth[None]), jnp.asarray(body[None]))
                env.step(np.asarray(mean[0], dtype=np.float32))
                poses.append(np.asarray(env.host.qpos).copy())
            outcomes.append(env.judge.result)
            if seed == config.test_seeds[0]:
                stride = max(1, (len(poses) + 499) // 500)
                selected = poses[::stride]
                write_replay(
                    output / "evaluation-replay.json",
                    selected,
                    [False] * (len(selected) - 1) + [True],
                    stride / config.control_hz,
                )
            with (output / f"evaluation-{seed}.npz").open("xb") as stream:
                np.savez_compressed(
                    stream, qpos=np.asarray(poses), time=np.arange(1, len(poses) + 1) / config.control_hz
                )
        finally:
            env.close()
    write_toml(
        output / "evaluation.toml",
        {
            "format": "go1-depth-evaluation/v1",
            "policy_directory": str(policy_directory.resolve()),
            "policy_sha256": hashlib.sha256((policy_directory / "policy.npz").read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256((policy_directory / "config.toml").read_bytes()).hexdigest(),
            "source_sha256": model_digest(source / "unitree_go1"),
            "seeds": config.test_seeds,
            "outcomes": outcomes,
            "successes": outcomes.count("success"),
            "total": len(outcomes),
            "success_rate": outcomes.count("success") / len(outcomes),
        },
    )
    print(json.dumps({"evaluation_outcomes": outcomes, "successes": outcomes.count("success"), "total": len(outcomes)}))


def train(
    source: Path,
    output: Path,
    config: VisionConfig,
    source_hash: str,
    device: jax.Device,
    base_directory: Path | None = None,
) -> None:
    key = jax.random.PRNGKey(config.train_seeds[0])
    key, initial = jax.random.split(key)
    base = None
    if base_directory is not None:
        base, manifest = load_base(base_directory, source_hash)
        if manifest["policy_sha256"] != config.base_policy_sha256:
            raise ValueError("base policy differs from task snapshot")
    elif config.base_policy_sha256:
        raise ValueError("base policy directory required")
    params = initialize(initial, config.height, config.width, base=base)
    optimizer = adam(params)
    predict = cast(Callable[[Params, jax.Array, jax.Array], tuple[jax.Array, jax.Array]], jax.jit(forward))  # pyright: ignore[reportUnknownMemberType]
    optimize = cast(Callable[..., tuple[Params, Adam, dict[str, jax.Array]]], jax.jit(update))  # pyright: ignore[reportUnknownMemberType]
    started = time.monotonic()
    episode = 0
    env = VisualEnvironment(source, config, config.train_seeds[0], device)
    environments = {config.train_seeds[0]: env}
    try:
        with (output / "training.jsonl").open("x") as curve:
            for iteration in range(config.iterations):
                images: list[NDArray[np.float32]] = []
                states: list[NDArray[np.float32]] = []
                actions: list[NDArray[np.float32]] = []
                logps: list[float] = []
                values: list[float] = []
                rewards: list[float] = []
                dones: list[float] = []
                for _ in range(config.rollout_steps):
                    image, body = env.observe()
                    mean, value = predict(params, jnp.asarray(image[None]), jnp.asarray(body[None]))
                    key, sub = jax.random.split(key)
                    action = mean + jnp.exp(distribution_log_std(params)) * jax.random.normal(sub, mean.shape)
                    logp = log_probability(params, mean, action)
                    reward, outcome = env.step(np.asarray(action[0], dtype=np.float32))
                    images.append(image)
                    states.append(body)
                    actions.append(np.asarray(action[0], dtype=np.float32))
                    logps.append(float(logp[0]))
                    values.append(float(value[0]))
                    rewards.append(reward)
                    dones.append(float(outcome != "running"))
                    if outcome != "running":
                        episode += 1
                        seed = config.train_seeds[episode % len(config.train_seeds)]
                        if seed not in environments:
                            environments[seed] = VisualEnvironment(source, config, seed, device)
                        env = environments[seed]
                        env.reset()
                image, body = env.observe()
                _, bootstrap = predict(params, jnp.asarray(image[None]), jnp.asarray(body[None]))
                advantages = generalized_advantage(
                    jnp.asarray(rewards), jnp.asarray(values), jnp.asarray(dones), bootstrap[0]
                )
                params, optimizer, metrics = optimize(
                    params,
                    optimizer,
                    jnp.asarray(np.stack(images)),
                    jnp.asarray(np.stack(states)),
                    jnp.asarray(np.stack(actions)),
                    jnp.asarray(logps),
                    advantages,
                    advantages + jnp.asarray(values),
                    config.learning_rate,
                )
                if any(not np.isfinite(np.asarray(value)).all() for value in params.values()):
                    raise ValueError("nonfinite visual policy update")
                record = {
                    "iteration": iteration + 1,
                    "timesteps": (iteration + 1) * config.rollout_steps,
                    "total_timesteps": config.iterations * config.rollout_steps,
                    "mean_height_m": float(env.host.qpos[2]),
                    "episode_ends": sum(dones),
                    "elapsed_seconds": time.monotonic() - started,
                    "loss": float(metrics["loss"]),
                    "mean_reward": float(np.mean(rewards)),
                    "episodes": episode,
                }
                curve.write(json.dumps(record, allow_nan=False) + "\n")
                curve.flush()
                print(json.dumps({"event": "training.progress", "attributes": record}), flush=True)
    finally:
        for environment in environments.values():
            environment.close()
    save_policy(output, params, config, source_hash, base_directory=base_directory)
    evaluate(source, output, config, params, device, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=MODES, default=os.environ.get("AIOPS_GO1_DEVICE", "cpu"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base-policy", type=Path)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-output", type=Path)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--evaluate", type=Path)
    args = parser.parse_args()
    try:
        configure_device(args.device)
        source_hash = require_source(args.source)
        if args.validate_output:
            load_policy(args.validate_output, source_hash)
            print(json.dumps({"valid": True}))
            return
        if args.check:
            device = select_device(args.device)
            if device.platform == "gpu":
                check_config = VisionConfig(**tomllib.loads(args.config.read_text())) if args.config else VisionConfig()
                check_model, _ = make_scene(args.source, check_config, check_config.train_seeds[0])
                mjx.put_model(check_model, device=device)
            print(
                json.dumps(
                    {
                        "ready": True,
                        "devices": [str(device)],
                        "errors": [],
                        "source_commit": COMMIT,
                        "source_sha256": source_hash,
                    }
                )
            )
            return
        config = VisionConfig(**tomllib.loads(args.config.read_text())) if args.config else VisionConfig()
        if args.probe:
            output = args.output or Path.home() / ".odyssey/go1/vision" / datetime.now(UTC).strftime(
                "%Y%m%dT%H%M%S%f-probe"
            )
            output.mkdir(parents=True, exist_ok=False)
            probe(args.source, output, config, source_hash)
            return
        device = select_device(args.device)  # Fail before creating any training output.
        output = args.output or Path.home() / ".odyssey/go1/vision" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        with jax.default_device(device):
            if args.evaluate:
                params, config = load_policy(args.evaluate, source_hash)
                output.mkdir(parents=True, exist_ok=False)
                evaluate(args.source, output, config, params, device, args.evaluate)
            else:
                output.mkdir(parents=True, exist_ok=False)
                train(args.source, output, config, source_hash, device, args.base_policy)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        if args.check:
            print(json.dumps({"ready": False, "errors": [str(error)], "devices": [], "source_commit": COMMIT}))
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
