"""Build a descriptive model manifest from a saved numeric Go1 policy."""

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast

import numpy as np


class ModelManifest(TypedDict):
    kind: str
    policy_file: str
    policy_sha256: str
    observation_dimension: int
    action_dimension: int
    policy_format: str
    source: dict[str, object]
    runtime: dict[str, object]
    training: Mapping[str, object]
    evaluation: Mapping[str, object]
    evaluation_conditions: object
    evaluation_conditions_sha256: object
    limitations: list[str]


def create_manifest(policy: str | Path, environment: object, evaluation: object) -> ModelManifest:
    metadata: Mapping[str, object] = cast(Mapping[str, object], environment) if isinstance(environment, dict) else {}
    config = metadata.get("config")
    if not metadata.get("source_commit") or not metadata.get("model_sha256") or not isinstance(config, dict):
        raise ValueError("model source and training configuration are required")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be an object")  # noqa: TRY004 - external JSON validation uses a stable ValueError contract.
    policy = Path(policy)
    with policy.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "kind": "go1-mjx-ppo",
        "policy_file": policy.name,
        "policy_sha256": digest,
        "observation_dimension": 45,
        "action_dimension": 12,
        "policy_format": "numpy-npz",
        "source": {"commit": metadata["source_commit"], "model_sha256": metadata["model_sha256"]},
        "runtime": {"jax": metadata.get("jax"), "mujoco": metadata.get("mujoco"), "device": metadata.get("device")},
        "training": cast(Mapping[str, object], config),
        "evaluation": cast(Mapping[str, object], evaluation),
        "evaluation_conditions": metadata.get("evaluation_conditions"),
        "evaluation_conditions_sha256": metadata.get("evaluation_conditions_sha256"),
        "limitations": [
            "experimental policy",
            "no real-robot validation",
            "single perturbed evaluation is not a generalization guarantee",
        ],
    }


def validate_policy(policy: str | Path) -> None:
    shapes = {
        "l1/weight": (45, 64),
        "l1/bias": (64,),
        "l2/weight": (64, 64),
        "l2/bias": (64,),
        "actor/weight": (64, 12),
        "actor/bias": (12,),
        "critic/weight": (64, 1),
        "critic/bias": (1,),
        "log_std": (12,),
    }
    with np.load(policy, allow_pickle=False) as data:
        if set(data.files) != set(shapes):
            raise ValueError("unexpected policy arrays")
        for name, shape in shapes.items():
            array = data[name]
            if array.shape != shape or array.dtype.kind != "f" or not np.isfinite(array).all():
                raise ValueError(f"invalid policy array: {name}")
