"""Go1 PPO configuration and structured progress contracts."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

from odyssey_config import Loader
from pydantic import BaseModel as SettingsModel
from pydantic import Field as SettingsField

from .evaluation_conditions import ConditionsConfig


class TrainingConfig(ConditionsConfig):
    iterations: int
    rollout_steps: int
    seed: int
    learning_rate: float
    checkpoint_interval: int
    monitor_metrics: Sequence[str]
    monitor_interval: int


class MonitorMetric(TypedDict):
    key: str
    label: str
    unit: str
    display: str
    value: float


class MonitorAttributes(TypedDict):
    step: int
    metrics: list[MonitorMetric]


class MonitorEvent(TypedDict):
    event: str
    attributes: MonitorAttributes


class TrainingMetric(TypedDict):
    iteration: int
    timesteps: int
    total_timesteps: int
    mean_reward: float
    episode_ends: int
    mean_height_m: float
    loss: float
    elapsed_seconds: float


class Go1TrainingConfig(SettingsModel):
    iterations: int = 20
    rollout_steps: int = 64
    seed: int = 42
    target_speed_m_s: float = 0.0
    learning_rate: float = 0.0003
    eval_seed: int = 10001
    eval_steps: int = 100
    checkpoint_interval: int = 10
    monitor_metrics: list[str] = ["mean_reward", "loss", "mean_height_m", "episode_ends"]
    monitor_interval: int = 1000


class Go1Config(SettingsModel):
    training: Go1TrainingConfig = SettingsField(default_factory=Go1TrainingConfig)


DEFAULTS: Mapping[str, object] = Go1TrainingConfig().model_dump()
MONITOR_METRICS = {
    "timesteps": ("训练步数", "步", "value"),
    "mean_reward": ("平均奖励", "", "line"),
    "loss": ("PPO 损失", "", "line"),
    "mean_height_m": ("平均机身高度", "m", "line"),
    "episode_ends": ("本次采样回合结束", "次", "value"),
    "elapsed_seconds": ("训练用时", "秒", "line"),
}
DEFAULT_MONITOR_METRICS = ("mean_reward", "loss", "mean_height_m", "episode_ends")


def resolve_monitor_config(config: Mapping[str, object]) -> tuple[Sequence[str], int]:
    metrics = config.get("monitor_metrics", DEFAULT_MONITOR_METRICS)
    items = cast(Sequence[object], metrics) if isinstance(metrics, Sequence) else ()
    if (
        not items
        or any(not isinstance(item, str) or item not in MONITOR_METRICS for item in items)
        or len(items) != len(set(items))
    ):
        raise ValueError("training.monitor_metrics contains an unknown or duplicate metric")
    interval = config.get("monitor_interval", 1000)
    if type(interval) is not int:
        raise ValueError("training.monitor_interval must be an integer")
    if not 100 <= interval <= 100000:
        raise ValueError("training.monitor_interval must be between 100 and 100000")
    return cast(Sequence[str], metrics), interval


def create_monitor_event(step: int, selected: Sequence[str], values: Mapping[str, float]) -> MonitorEvent:
    definitions: list[MonitorMetric] = []
    for metric_id in selected:
        if metric_id not in MONITOR_METRICS or metric_id not in values:
            raise ValueError("unknown or unavailable monitor metric")
        value = values[metric_id]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("monitor metric must be finite")
        label, unit, display = MONITOR_METRICS[metric_id]
        definitions.append(
            {"key": f"go1.{metric_id}", "label": label, "unit": unit, "display": display, "value": value}
        )
    return {"event": "monitor.metrics", "attributes": {"step": step, "metrics": definitions}}


def monitor_due(step: int, next_step: int, total_steps: int) -> bool:
    return step >= next_step or step >= total_steps


def validate_config(values: object) -> TrainingConfig:
    if not isinstance(values, dict):
        raise ValueError("unknown training parameters")  # noqa: TRY004 - preserve config validation's ValueError contract.
    specified = cast(Mapping[str, object], values)
    if specified.keys() - DEFAULTS.keys():
        raise ValueError("unknown training parameters")
    c = {**DEFAULTS, **specified}
    for name, low, high in (
        ("iterations", 1, 100000),
        ("rollout_steps", 8, 4096),
        ("seed", 0, 2**31 - 1),
        ("eval_seed", 0, 2**31 - 1),
        ("eval_steps", 2, 500),
        ("checkpoint_interval", 1, 100000),
    ):
        if type(value := c[name]) is not int or not low <= value <= high:
            raise ValueError("invalid " + name)
    for name in ("target_speed_m_s", "learning_rate"):
        if (
            type(value := c[name]) not in (int, float)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("invalid " + name)
    if c["eval_seed"] == c["seed"]:
        raise ValueError("evaluation seed must differ from training seed")
    learning_rate = cast(float, c["learning_rate"])
    target_speed = cast(float, c["target_speed_m_s"])
    if not 0 < learning_rate <= 0.01 or abs(target_speed) > 2:
        raise ValueError("training parameter outside allowed range")
    resolve_monitor_config(c)
    return cast(TrainingConfig, c)


TRAINING_LOADER = Loader(Go1Config, env_prefix="ODYLAB_GO1")


def load_training_config(
    path: str | Path | None,
    *,
    profile: str | None = None,
    overrides: Mapping[str, object] | None = None,
    default_path: Path | None = None,
) -> TrainingConfig:
    if (
        "GO1_TRAIN_CONFIG" in os.environ
        or "TRAINING_MONITOR_METRICS" in os.environ
        or "TRAINING_MONITOR_INTERVAL" in os.environ
    ):
        raise ValueError("legacy Go1 task configuration variables are no longer supported")
    document = TRAINING_LOADER.load(path, profile=profile, overrides=overrides, default_path=default_path).training
    return validate_config(document.model_dump())
