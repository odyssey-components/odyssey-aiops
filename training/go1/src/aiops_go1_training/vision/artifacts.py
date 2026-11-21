"""Small TOML contracts and pickle-free visual-policy checkpoints."""

import hashlib
import shutil
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from ..runtime.local import COMMIT
from ..runtime.toml import write_toml
from ..upstream.artifacts import digest
from ..upstream.artifacts import load as load_base
from ..upstream.policy import Params as BaseParams
from .configuration import VisionConfig
from .policy import Params, initialize


def save_policy(
    output: Path, params: Params, config: VisionConfig, source_hash: str, *, base_directory: Path | None = None
) -> None:
    base_fields: dict[str, object] = {}
    if config.base_policy_sha256:
        if base_directory is None:
            raise ValueError("residual model requires base artifacts")
        base, base_manifest = load_base(base_directory, source_hash)
        if base_manifest["policy_sha256"] != config.base_policy_sha256:
            raise ValueError("base policy changed")
        for name, value in base.items():
            if not np.array_equal(np.asarray(params.get("base." + name)), np.asarray(value)):
                raise ValueError("frozen base policy was modified")
        (output / "base").mkdir()
        for name in ("policy.npz", "model.manifest.toml", "config.toml"):
            if name == "config.toml" and not (base_directory / name).exists():
                continue
            shutil.copyfile(base_directory / name, output / "base" / name)
        base_fields = {
            "base_policy_sha256": config.base_policy_sha256,
            "base_manifest_sha256": digest(output / "base" / "model.manifest.toml", 65536),
        }
    target = output / "policy.npz"
    with target.open("xb") as stream:
        save = cast(Callable[..., None], np.savez_compressed)
        save(stream, **{name: np.asarray(value) for name, value in params.items() if not name.startswith("base.")})
    write_toml(output / "config.toml", cast(dict[str, object], config.model_dump()))
    write_toml(
        output / "model.manifest.toml",
        {
            "format": config.format,
            "policy_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "source_sha256": source_hash,
            "source_commit": COMMIT,
            "config_sha256": hashlib.sha256((output / "config.toml").read_bytes()).hexdigest(),
            "policy": "depth-base-residual-ppo/v2" if config.base_policy_sha256 else "depth-cnn-proprioception-ppo/v1",
            **base_fields,
            "proprioception": 45,
            "actions": 12,
            "depth_height": config.height,
            "depth_width": config.width,
            "camera_hz": config.camera_hz,
            "control_hz": config.control_hz,
            "physics": "mujoco-cpu"
            if next(iter(next(iter(params.values())).devices())).platform == "cpu"
            else "mjx-gpu",
            "renderer": "mujoco-egl",
            "camera": "trunk-forward-depth-15deg/v1",
        },
    )


def load_policy(directory: Path, source_hash: str) -> tuple[Params, VisionConfig]:
    digest(directory / "model.manifest.toml", 65536)
    manifest = tomllib.loads((directory / "model.manifest.toml").read_text())
    if (
        manifest.get("format") not in ("go1-depth/v1", "go1-depth-residual/v2")
        or manifest.get("policy") not in ("depth-cnn-proprioception-ppo/v1", "depth-base-residual-ppo/v2")
        or manifest.get("source_sha256") != source_hash
    ):
        raise ValueError("visual policy/source protocol mismatch")
    for name, key in [("policy.npz", "policy_sha256"), ("config.toml", "config_sha256")]:
        if digest(directory / name) != manifest.get(key):
            raise ValueError("visual artifact checksum mismatch")
    config = VisionConfig(**tomllib.loads((directory / "config.toml").read_text()))
    required = {
        "proprioception": 45,
        "actions": 12,
        "depth_height": config.height,
        "depth_width": config.width,
        "camera_hz": config.camera_hz,
        "control_hz": config.control_hz,
        "renderer": "mujoco-egl",
        "camera": "trunk-forward-depth-15deg/v1",
    }
    if manifest.get("physics") not in {"mujoco-cpu", "mjx-cpu", "mjx-gpu"} or any(
        manifest.get(name) != value for name, value in required.items()
    ):
        raise ValueError("visual camera/control protocol mismatch")
    base: BaseParams | None = None
    if config.base_policy_sha256:
        if manifest.get("format") != "go1-depth-residual/v2" or manifest.get("policy") != "depth-base-residual-ppo/v2":
            raise ValueError("residual policy protocol mismatch")
        if digest(directory / "base" / "model.manifest.toml", 65536) != manifest.get("base_manifest_sha256"):
            raise ValueError("base manifest checksum mismatch")
        base, base_manifest = load_base(directory / "base", source_hash)
        if (
            config.base_policy_sha256 != base_manifest["policy_sha256"]
            or manifest.get("base_policy_sha256") != config.base_policy_sha256
        ):
            raise ValueError("base policy checksum mismatch")
    elif manifest.get("format") != "go1-depth/v1" or manifest.get("policy") != "depth-cnn-proprioception-ppo/v1":
        raise ValueError("unexpected residual policy")
    expected = initialize(jax.random.PRNGKey(0), config.height, config.width)
    with np.load(directory / "policy.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("visual policy parameter names differ")
        params: Params = {}
        for name, reference in expected.items():
            value = np.asarray(archive[name])
            if value.shape != reference.shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("visual policy parameter shape/dtype/value mismatch")
            params[name] = jnp.asarray(value)
    if base is not None:
        params.update({"base." + name: value for name, value in base.items()})
    return params, config
