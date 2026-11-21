"""Experimental Go1 MJX PPO training; requires CUDA and a pinned Go1 source."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, Protocol, cast

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from odylab_training import Metric, event_record, local_reporter
from odylab_training import continuous_ppo as ppo
from odylab_training.worker import WorkerRun, training_stage

from ..runtime.devices import select_device
from ..runtime.provenance import model_digest
from ..runtime.resource_monitor import ResourceMonitor
from .checkpoint import write_checkpoint
from .configuration import (
    TRAINING_LOADER,
    TrainingConfig,
    TrainingMetric,
    create_monitor_event,
    load_training_config,
    monitor_due,
    resolve_monitor_config,
    validate_config,
)
from .environment import create_environment
from .evaluation import summarize
from .evaluation_conditions import ConditionsInput, code_digest, conditions, conditions_digest
from .model_manifest import create_manifest, validate_policy
from .replay import write_replay

if TYPE_CHECKING:
    from jax import Array
    from numpy.typing import NDArray

    from .environment import EnvironmentState


class TrainingMetadata(ConditionsInput):
    model: str
    limitations: list[str]
    evaluation_conditions: NotRequired[dict[str, object]]
    evaluation_conditions_sha256: NotRequired[str]


if TYPE_CHECKING:
    type TrainingCarry = tuple[EnvironmentState, Array]
    type TrainingRollout = tuple[TrainingCarry, tuple[Array, Array, Array, Array, Array, Array, Array]]
    type EvaluationRollout = tuple[EnvironmentState, tuple[Array, Array, Array, Array]]


class _Compile(Protocol):
    def __call__[**P, R](self, function: Callable[P, R], /) -> Callable[P, R]: ...


def validate_packaged_source(provenance: object, actual_digest: str) -> None:
    expected = "f4fb0d12ac68842c1567bf1de6f771ce8e6671b5"
    if not isinstance(provenance, dict):
        raise ValueError("packaged Go1 model does not match its declared source")  # noqa: TRY004 - preserve provenance validation's ValueError contract.
    metadata = cast(Mapping[str, object], provenance)
    if metadata.get("commit") != expected or metadata.get("model_sha256") != actual_digest:
        raise ValueError("packaged Go1 model does not match its declared source")


def run(source: Path, output: Path, config: TrainingConfig, provenance: object = None) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monitor_metrics, monitor_interval = resolve_monitor_config(config)
    if provenance is not None:
        validate_packaged_source(provenance, model_digest(source / "unitree_go1"))
    else:
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if commit != "f4fb0d12ac68842c1567bf1de6f771ce8e6671b5":
            raise ValueError("unexpected Go1 source commit")
        if subprocess.run(
            ["git", "-C", str(source), "diff", "--quiet", "HEAD", "--", "unitree_go1"], check=False
        ).returncode:
            raise ValueError("Go1 model has local modifications")
    device = select_device()
    reporter = local_reporter("go1", "ppo", device=str(device), name="Go1 PPO")
    model = mujoco.MjModel.from_xml_path(str(source / "unitree_go1/scene_torque.xml"))
    initial = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, initial, 0)
    mujoco.mj_forward(model, initial)
    reset, observe, step = create_environment(
        model, mjx.put_model(model, device=device), initial, config["target_speed_m_s"]
    )
    state = reset()
    key = jax.random.PRNGKey(config["seed"])
    key, init_key = jax.random.split(key)
    params = ppo.init_policy(init_key, observe(state[0], state[1]).shape[0], 12)
    optimizer = ppo.adam_init(params)
    output.mkdir(parents=True, exist_ok=True)
    metadata: TrainingMetadata = {
        "model": "unitree-go1",
        "model_sha256": model_digest(source / "unitree_go1"),
        "source_commit": "f4fb0d12ac68842c1567bf1de6f771ce8e6671b5",
        "device": str(device),
        "jax": jax.__version__,
        "mujoco": mujoco.__version__,
        "config": config,
        "limitations": [
            "experimental simplified reward and observation",
            "random initial policy",
            "not validated for real robot",
        ],
    }
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def flatten_array_tree(tree: Mapping[str, object], prefix: str = "") -> dict[str, NDArray[np.generic]]:
        arrays: dict[str, NDArray[np.generic]] = {}
        for key in sorted(tree):
            value = tree[key]
            if isinstance(value, dict):
                arrays.update(flatten_array_tree(cast(Mapping[str, object], value), prefix + key + "/"))
            elif isinstance(value, jax.Array):
                arrays[prefix + key] = np.asarray(value)
            else:
                raise TypeError(f"expected a JAX array at {prefix + key}")
        return arrays

    def collect_rollout(params: ppo.PolicyParams, state: EnvironmentState, key: Array) -> TrainingRollout:
        def tick(carry: TrainingCarry, _: None) -> TrainingRollout:
            state, key = carry
            key, sub = jax.random.split(key)
            obs = observe(state[0], state[1])
            mean, value = ppo.policy(params, obs)
            action = mean + jnp.exp(jnp.clip(params["log_std"], -5.0, 2.0)) * jax.random.normal(sub, mean.shape)
            logp = ppo.log_probability(params, mean, action)
            next_state, r, done, qpos = step(state, action)
            return (next_state, key), (obs, action, logp, value, r, done, qpos)

        return jax.lax.scan(tick, (state, key), None, length=config["rollout_steps"])

    # JAX's compiled callables preserve these signatures but its public type omits them.
    compile_function = cast(_Compile, jax.jit)
    collect = compile_function(collect_rollout)
    update = compile_function(ppo.update)
    start = time.monotonic()
    next_monitor_step = monitor_interval
    last_rollout: tuple[Array, Array, Array] | None = None
    last_metric: TrainingMetric | None = None
    with (output / "training-curve.jsonl").open("w") as curve:
        for iteration in range(1, config["iterations"] + 1):
            (state, key), (obs, actions, logp, values, rewards, dones, qpos) = collect(params, state, key)
            _, bootstrap = ppo.policy(params, observe(state[0], state[1]))
            adv = ppo.generalized_advantage(rewards, values, dones.astype(jnp.float32), bootstrap)
            batch: ppo.Batch = {
                "observations": obs,
                "actions": actions,
                "old_logp": logp,
                "advantages": adv,
                "returns": adv + values,
            }
            losses = None
            for _ in range(4):
                params, optimizer, losses = update(params, optimizer, batch, config["learning_rate"])
            if losses is None:
                raise RuntimeError("PPO update did not produce a loss record")
            params["l1"]["weight"].block_until_ready()
            devices = (params["l1"]["weight"].device, state[0].qpos.device)
            if any(
                isinstance(array_device, jax.sharding.Sharding) or array_device.platform != device.platform
                for array_device in devices
            ):
                raise RuntimeError("physics and PPO must run on the selected device")
            policy_arrays = cast(list[jax.Array], jax.tree_util.tree_leaves(params))
            if not all(np.isfinite(np.asarray(x)).all() for x in (rewards, qpos, *policy_arrays)):
                raise RuntimeError("non-finite training values")
            metric: TrainingMetric = {
                "iteration": iteration,
                "timesteps": iteration * config["rollout_steps"],
                "total_timesteps": config["iterations"] * config["rollout_steps"],
                "mean_reward": float(rewards.mean()),
                "episode_ends": int(dones.sum()),
                "mean_height_m": float(qpos[:, 2].mean()),
                "loss": float(losses["loss"]),
                "elapsed_seconds": time.monotonic() - start,
            }
            last_rollout = (qpos, rewards, dones)
            last_metric = metric
            curve.write(json.dumps(metric, allow_nan=False) + "\n")
            curve.flush()
            progress = {**metric, "monitor_metrics": [f"go1.{item}" for item in monitor_metrics]}
            print(
                json.dumps(
                    event_record("training.progress", "Go1 PPO 训练", progress), ensure_ascii=False, allow_nan=False
                ),
                flush=True,
            )
            if monitor_due(metric["timesteps"], next_monitor_step, metric["total_timesteps"]):
                values = {
                    "timesteps": metric["timesteps"],
                    "mean_reward": metric["mean_reward"],
                    "loss": metric["loss"],
                    "mean_height_m": metric["mean_height_m"],
                    "episode_ends": metric["episode_ends"],
                    "elapsed_seconds": metric["elapsed_seconds"],
                }
                print(
                    json.dumps(
                        create_monitor_event(metric["timesteps"], monitor_metrics, values),
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                    flush=True,
                )
                if reporter:
                    reporter.emit(
                        "ppo",
                        metric["timesteps"],
                        [
                            Metric("common.elapsed_seconds", "训练耗时", metric["elapsed_seconds"], "秒", "line"),
                            Metric("go1.mean_reward", "平均奖励", metric["mean_reward"], "", "line"),
                            Metric("go1.loss", "PPO 损失", metric["loss"], "", "line"),
                        ],
                        quantity={"value": metric["timesteps"], "unit": "step", "label": "训练步数"},
                        costs={"ppo": metric["elapsed_seconds"]},
                    )
                while next_monitor_step <= metric["timesteps"]:
                    next_monitor_step += monitor_interval
            if iteration % config["checkpoint_interval"] == 0 or iteration == config["iterations"]:
                arrays = {
                    **flatten_array_tree(params, "params/"),
                    **flatten_array_tree(optimizer, "optimizer/"),
                    "iteration": np.asarray(iteration),
                    "random_key": np.asarray(key),
                    "qpos": np.asarray(state[0].qpos),
                    "qvel": np.asarray(state[0].qvel),
                    "ctrl": np.asarray(state[0].ctrl),
                    "last_action": np.asarray(state[1]),
                    "episode_step": np.asarray(state[2]),
                }
                write_checkpoint(output / f"checkpoint-{iteration:06d}.npz", arrays)
    if last_rollout is None or last_metric is None:
        raise RuntimeError("training completed without any PPO iterations")
    qpos, rewards, dones = last_rollout
    metric = last_metric
    # Evaluate once from a held-out perturbed initial state; this is not a pass/fail criterion.
    eval_initial = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, eval_initial, 0)
    rng = np.random.default_rng(config["eval_seed"])
    eval_initial.qpos[2] += rng.uniform(-0.005, 0.005)
    eval_initial.qpos[7:] += rng.uniform(-0.025, 0.025, 12)
    mujoco.mj_forward(model, eval_initial)
    eval_reset, eval_observe, eval_step = create_environment(
        model, mjx.put_model(model, device=device), eval_initial, config["target_speed_m_s"]
    )

    def evaluate_rollout(params: ppo.PolicyParams) -> EvaluationRollout:
        def tick(state: EnvironmentState, _: None) -> EvaluationRollout:
            action, _value = ppo.policy(params, eval_observe(state[0], state[1]))
            next_state, score, done, position = eval_step(state, action)
            return next_state, (position, action, score, done)

        return jax.lax.scan(tick, eval_reset(), None, length=config["eval_steps"])

    evaluate_policy = compile_function(evaluate_rollout)
    eval_state, eval_frames = evaluate_policy(params)
    eval_state[0].qpos.block_until_ready()
    eval_qpos, eval_action, eval_reward, eval_done = (np.asarray(frame) for frame in eval_frames)
    if not all(np.isfinite(x).all() for x in (eval_qpos, eval_action, eval_reward)):
        raise RuntimeError("non-finite evaluation values")
    evaluation = summarize(eval_qpos[:, 2], eval_done, eval_qpos[:, 0], 0.02, config["eval_seed"])
    np.savez_compressed(
        output / "evaluation.npz",
        time=np.arange(1, len(eval_qpos) + 1) * 0.02,
        qpos=eval_qpos,
        action=eval_action,
        reward=eval_reward,
        done=eval_done,
    )
    write_replay(output / "evaluation-replay.json", eval_qpos, eval_done, 0.02)
    policy_weights = flatten_array_tree(params)
    if "allow_pickle" in policy_weights:
        raise ValueError("policy archive uses a reserved NumPy keyword")
    np.savez_compressed(output / "policy.npz", allow_pickle=False, **policy_weights)  # type: ignore[reportArgumentType]  # NumPy stubs treat dynamic array keys as the allow_pickle option.
    eval_conditions = conditions(
        metadata, code_digest(Path(__file__).resolve().parent.parent, Path(ppo.__file__).resolve().parent)
    )
    metadata["evaluation_conditions"] = eval_conditions
    metadata["evaluation_conditions_sha256"] = conditions_digest(eval_conditions)
    validate_policy(output / "policy.npz")
    manifest = create_manifest(output / "policy.npz", metadata, evaluation)
    (output / "model-manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        output / "last-rollout.npz", qpos=np.asarray(qpos), reward=np.asarray(rewards), done=np.asarray(dones)
    )
    final_metric = {**metric, "status": "completed", "evaluation": evaluation}
    (output / "metrics.json").write_text(json.dumps(final_metric, indent=2, allow_nan=False) + "\n")
    if reporter:
        reporter.finish()


__all__ = [
    "create_monitor_event",
    "load_training_config",
    "main",
    "monitor_due",
    "resolve_monitor_config",
    "run",
    "validate_config",
    "validate_packaged_source",
]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--odylab", action="store_true")
    parser.add_argument("--profile")
    TRAINING_LOADER.bind_arguments(parser)
    args = parser.parse_args(argv)
    overrides = {key: value for key, value in vars(args).items() if key.startswith("training.")}
    if not args.odylab:
        if not args.source or not args.output:
            parser.error("--source and --output are required")
        if args.config is None and not os.environ.get("ODYLAB_GO1_CONFIG_FILE"):
            parser.error("--config or ODYLAB_GO1_CONFIG_FILE is required")
    try:
        with training_stage("preparation", stderr=not args.odylab):
            if args.odylab:
                worker = WorkerRun.from_environment()
                source, output = worker.scene_dir, worker.output_dir
                provenance = json.loads((source / "source.json").read_text())
                config = worker.load_config(
                    load_training_config, args.config, profile=args.profile, overrides=overrides
                )
            else:
                source, output, provenance = args.source, args.output, None
                config = load_training_config(args.config, profile=args.profile, overrides=overrides)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    with training_stage("execution", stderr=not args.odylab):
        output.mkdir(parents=True, exist_ok=args.odylab)
        with ResourceMonitor(output):
            run(source.expanduser().resolve(), output.expanduser().resolve(), config, provenance)


if __name__ == "__main__":
    main()
