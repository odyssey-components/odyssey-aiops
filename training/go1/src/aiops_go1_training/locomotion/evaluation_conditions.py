"""Record the independent evaluation conditions used for model comparison."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from typing import ReadOnly

FILES = (
    "locomotion/train.py",
    "locomotion/environment.py",
    "locomotion/configuration.py",
    "runtime/provenance.py",
    "runtime/devices.py",
    "locomotion/evaluation.py",
    "__init__.py",
    "continuous_ppo.py",
    "events.py",
)


def code_digest(directory: str | Path, shared_package_path: str | Path | None = None) -> str:
    root = Path(directory)
    digest = hashlib.sha256()
    for name in FILES:
        path = (
            Path(shared_package_path) / name
            if name in {"__init__.py", "continuous_ppo.py", "events.py"} and shared_package_path is not None
            else root / name
        )
        content = path.read_bytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(content).digest())

    for source in sorted(files("odyssey_config").iterdir(), key=lambda item: item.name):
        if source.name.endswith(".py") and source.is_file():
            digest.update(f"odyssey_config/{source.name}".encode() + b"\0")
            digest.update(hashlib.sha256(source.read_bytes()).digest())
    return digest.hexdigest()


class ConditionsConfig(TypedDict):
    eval_seed: int
    eval_steps: int
    target_speed_m_s: float


class ConditionsInput(TypedDict):
    model_sha256: str
    source_commit: str
    config: ReadOnly[ConditionsConfig]
    jax: str
    mujoco: str
    device: str


def conditions(metadata: ConditionsInput, code_sha256: str) -> dict[str, object]:
    config = metadata["config"]
    return {
        "model_sha256": metadata["model_sha256"],
        "source_commit": metadata["source_commit"],
        "evaluation_seed": config["eval_seed"],
        "evaluation_steps": config["eval_steps"],
        "target_speed_m_s": config["target_speed_m_s"],
        "physics_step_seconds": 0.002,
        "control_step_seconds": 0.02,
        "initial_state": "keyframe-0-with-seeded-height-and-joint-perturbation",
        "runtime": {"jax": metadata["jax"], "mujoco": metadata["mujoco"], "device": metadata["device"]},
        "code_sha256": code_sha256,
    }


def conditions_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
