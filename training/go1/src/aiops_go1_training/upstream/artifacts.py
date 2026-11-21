"""Bounded numeric policy artifacts with an explicit upstream contract."""

import hashlib
import re
import tomllib
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax.numpy as jnp
import numpy as np

from ..runtime.local import COMMIT
from ..runtime.toml import write_toml
from .policy import SHAPES, Params

FORMAT = "go1-upstream-policy/v1"
PROTOCOL = "go1-sb3-observation/v1"


def digest(path: Path, limit: int = 4 << 20) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("policy file missing, symlinked or too large")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(params: Params) -> None:
    if set(params) != set(SHAPES):
        raise ValueError("upstream policy parameter names mismatch")
    for name, shape in SHAPES.items():
        value = np.asarray(params[name])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("upstream policy shape/dtype/value mismatch: " + name)
    if np.any(np.abs(np.asarray(params["log_std"])) > 20):
        raise ValueError("unsupported upstream log_std")


def save(
    directory: Path, params: Params, source_hash: str, title: str, *, parent: object = "", origin: str = ""
) -> None:
    validate(params)
    if not re.fullmatch("[0-9a-f]{64}", source_hash) or not 1 <= len(title.strip()) <= 80:
        raise ValueError("invalid model source or title")
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "policy.npz").open("xb") as stream:
        save_arrays = cast(Callable[..., None], np.savez_compressed)
        save_arrays(stream, **{k: np.asarray(v) for k, v in params.items()})
    write_toml(
        directory / "model.manifest.toml",
        {
            "format": FORMAT,
            "policy": "sb3-default-64x64-tanh/v1",
            "policy_sha256": digest(directory / "policy.npz"),
            "source_commit": COMMIT,
            "source_sha256": source_hash,
            "title": title.strip(),
            "parent_sha256": str(parent),
            "origin_sha256": origin,
            "observation_protocol": PROTOCOL,
            "actions": 12,
            "proprioception": 45,
            "control_hz": 50,
            "action_low": [-1.0] * 12,
            "action_high": [1.0] * 12,
            "last_action_timing": "before-previous-step",
            "optimizer_restored": False,
        },
    )


def load(directory: Path, source_hash: str) -> tuple[Params, dict[str, object]]:
    digest(directory / "model.manifest.toml", 65536)
    manifest = tomllib.loads((directory / "model.manifest.toml").read_text())
    expected: dict[str, object] = {
        "format": FORMAT,
        "policy": "sb3-default-64x64-tanh/v1",
        "source_commit": COMMIT,
        "source_sha256": source_hash,
        "observation_protocol": PROTOCOL,
        "actions": 12,
        "proprioception": 45,
        "control_hz": 50,
        "action_low": [-1.0] * 12,
        "action_high": [1.0] * 12,
        "last_action_timing": "before-previous-step",
        "optimizer_restored": False,
    }
    if any(manifest.get(k) != v for k, v in expected.items()) or manifest.get("policy_sha256") != digest(
        directory / "policy.npz"
    ):
        raise ValueError("upstream policy protocol/source/checksum mismatch")
    if "config_sha256" in manifest and digest(directory / "config.toml", 65536) != manifest["config_sha256"]:
        raise ValueError("upstream config checksum mismatch")
    with zipfile.ZipFile(directory / "policy.npz") as archive:
        if sum(item.file_size for item in archive.infolist()) > 4 << 20:
            raise ValueError("oversized numeric policy")
    with np.load(directory / "policy.npz", allow_pickle=False) as archive:
        if len(archive.files) != len(SHAPES) or set(archive.files) != set(SHAPES):
            raise ValueError("upstream policy arrays mismatch")
        for name in archive.files:
            value = archive[name]
            if value.shape != SHAPES[name] or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("upstream numeric array mismatch")
        params = {k: jnp.asarray(archive[k]) for k in archive.files}
    validate(params)
    return params, manifest
