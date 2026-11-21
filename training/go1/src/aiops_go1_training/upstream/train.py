"""Policy warm start with fresh optimizer, bounded rollouts and held-out CPU/GPU evaluation."""

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from odylab_training.continuous_ppo import generalized_advantage

from ..locomotion.replay import write_replay
from ..runtime.toml import write_toml
from ..vision.policy import Adam, adam
from .artifacts import digest, load, save
from .environment import Environment
from .optimize import update
from .policy import Params, forward, log_probability


def run(
    source: Path, output: Path, config: dict[str, float], base_directory: Path, source_hash: str, device: jax.Device
) -> None:
    params, manifest = load(base_directory, source_hash)
    initial_hash = digest(base_directory / "policy.npz")
    optimizer = adam(params)  # Deliberate warm start, never advertised as exact optimizer resume.
    key = jax.random.PRNGKey(int(config["seed"]))
    predict = cast(Callable[[Params, jax.Array], tuple[jax.Array, jax.Array]], jax.jit(forward))  # pyright: ignore[reportUnknownMemberType]
    optimize = cast(Callable[..., tuple[Params, Adam, dict[str, jax.Array]]], jax.jit(update))  # pyright: ignore[reportUnknownMemberType]
    random_commands = int(config.get("random_commands", 0))
    teacher = params if random_commands in (2, 3) else None
    env = Environment(source, device, int(config["seed"]), config["target_speed_m_s"], random_commands)
    started = time.monotonic()
    curve: list[dict[str, float]] = []
    first_mean: list[float] = []
    # Output is created only by save after successful training; never overwrite an existing model.
    if output.exists():
        raise FileExistsError("warm-start output exists")
    for iteration in range(int(config["iterations"])):
        states: list[NDArray[np.float32]] = []
        actions: list[NDArray[np.float32]] = []
        logs: list[float] = []
        values: list[float] = []
        rewards: list[float] = []
        dones: list[float] = []
        for _ in range(int(config["rollout_steps"])):
            body = env.observe()
            mean, value = predict(params, jnp.asarray(body[None]))
            if not first_mean:
                first_mean = np.asarray(mean[0]).tolist()
            key, sub = jax.random.split(key)
            action = mean + jnp.exp(params["log_std"]) * jax.random.normal(sub, mean.shape)
            score, done = env.step(np.asarray(action[0], dtype=np.float32))
            states.append(body)
            actions.append(np.asarray(action[0], dtype=np.float32))
            logs.append(float(log_probability(params, mean, action)[0]))
            values.append(float(value[0]))
            rewards.append(score)
            dones.append(float(done))
            if done:
                env.reset()
        _, bootstrap = predict(params, jnp.asarray(env.observe()[None]))
        advantages = generalized_advantage(jnp.asarray(rewards), jnp.asarray(values), jnp.asarray(dones), bootstrap[0])
        params, optimizer, metrics = optimize(
            params,
            optimizer,
            jnp.asarray(np.stack(states)),
            jnp.asarray(np.stack(actions)),
            jnp.asarray(logs),
            advantages,
            advantages + jnp.asarray(values),
            config["learning_rate"],
            teacher,
        )
        record = {
            "iteration": float(iteration + 1),
            "timesteps": float((iteration + 1) * config["rollout_steps"]),
            "total_timesteps": config["iterations"] * config["rollout_steps"],
            "loss": float(metrics["loss"]),
            "mean_reward": float(np.mean(rewards)),
            "mean_height_m": float(env.host.qpos[2]),
            "episode_ends": sum(dones),
            "elapsed_seconds": time.monotonic() - started,
        }
        curve.append(record)
        print(json.dumps({"event": "training.progress", "attributes": record}, allow_nan=False), flush=True)
    save(output, params, source_hash, str(manifest["title"]), parent=initial_hash)
    write_toml(output / "config.toml", dict(config))
    with (output / "model.manifest.toml").open("a") as stream:
        version = {
            0: "go1-command/v2",
            1: "go1-command-random/v1",
            2: "go1-command-stop/v1",
            3: "go1-command-turn/v1",
        }[random_commands]
        stream.write(f'environment_version = "{version}"\n')
        stream.write('config_sha256 = "' + digest(output / "config.toml") + '"\n')
    (output / "training.jsonl").write_text("".join(json.dumps(v, allow_nan=False) + "\n" for v in curve))
    write_toml(
        output / "warm-start.toml",
        {"initial_policy_sha256": initial_hash, "first_action_mean": first_mean, "optimizer_restored": False},
    )
    evaluate = Environment(source, device, int(config["eval_seed"]), config["target_speed_m_s"])
    poses: list[NDArray[np.float64]] = []
    score = 0.0
    for _ in range(int(config["eval_steps"])):
        mean, _ = predict(params, jnp.asarray(evaluate.observe()[None]))
        reward, done = evaluate.step(np.asarray(mean[0], dtype=np.float32))
        score += reward
        poses.append(np.asarray(evaluate.host.qpos).copy())
        if done:
            break
    write_replay(output / "evaluation-replay.json", poses, [False] * (len(poses) - 1) + [True], 0.02)
    write_toml(
        output / "evaluation.toml",
        {
            "format": "go1-command-evaluation/v2",
            "seed": int(config["eval_seed"]),
            "steps": len(poses),
            "reward": score,
            "final_height_m": float(evaluate.host.qpos[2]),
            "capability_verified": False,
        },
    )
