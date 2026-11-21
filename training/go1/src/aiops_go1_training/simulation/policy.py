"""Read-only, checksum-verified NumPy and depth policy inference."""

import hashlib
import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ..locomotion.model_manifest import validate_policy
from ..runtime.local import COMMIT
from ..upstream.artifacts import FORMAT as BASE_FORMAT
from ..upstream.artifacts import load as load_base
from ..upstream.policy import forward as base_forward
from ..vision.artifacts import load_policy
from ..vision.configuration import VisionConfig
from ..vision.policy import forward

Action = Callable[[NDArray[np.float32], NDArray[np.float32] | None], NDArray[np.float32]]


def locomotion_action(params: dict[str, NDArray[np.float64]], body: NDArray[np.float32]) -> NDArray[np.float32]:
    if body.shape != (45,) or not np.isfinite(body).all():
        raise ValueError("simulation observation must have 45 finite values")
    hidden = np.tanh(body @ params["l1/weight"] + params["l1/bias"])
    hidden = np.tanh(hidden @ params["l2/weight"] + params["l2/bias"])
    return np.asarray(hidden @ params["actor/weight"] + params["actor/bias"], dtype=np.float32)


def read_policy(
    directory: Path,
    source_hash: str,
    observe_state: Callable[[NDArray[np.float32]], None] | None = None,
) -> tuple[Action, VisionConfig | None, str]:
    path = directory / "policy.npz"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("simulation policy file invalid")
    vision = directory / "model.manifest.toml"
    manifest_path = vision if vision.exists() else directory / "model-manifest.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 65536:
        raise ValueError("simulation manifest invalid")
    manifest = tomllib.loads(manifest_path.read_text()) if vision.exists() else json.loads(manifest_path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest.get("policy_sha256") != digest:
        raise ValueError("simulation policy checksum mismatch")
    if vision.exists() and manifest.get("format") == BASE_FORMAT:
        params, _ = load_base(directory, source_hash)
        previous = np.zeros(12, dtype=np.float32)

        def imported(body: NDArray[np.float32], _depth: NDArray[np.float32] | None) -> NDArray[np.float32]:
            nonlocal previous
            state = body.copy()
            state[33:] = previous
            previous = body[33:].copy()
            if observe_state is not None:
                observe_state(state)
            mean, _ = base_forward(params, jnp.asarray(state[None]))
            return np.asarray(mean[0], dtype=np.float32)

        return imported, None, digest
    if vision.exists():
        if manifest.get("source_commit") != COMMIT or jax.default_backend() != "cpu":
            raise ValueError("simulation requires pinned source and JAX CPU")
        params, config = load_policy(directory, source_hash)

        previous = np.zeros(12, dtype=np.float32)

        def visual(body: NDArray[np.float32], depth: NDArray[np.float32] | None) -> NDArray[np.float32]:
            nonlocal previous
            if depth is None:
                raise ValueError("depth observation required")
            state = body.copy()
            if config.base_policy_sha256:
                state[33:] = previous
                previous = body[33:].copy()
            if observe_state is not None:
                observe_state(state)
            mean, _ = forward(params, jnp.asarray(depth[None]), jnp.asarray(state[None]))
            return np.asarray(mean[0], dtype=np.float32)

        return visual, config, digest
    source = manifest.get("source", {})
    if (
        manifest.get("kind") != "go1-mjx-ppo"
        or source.get("commit") != COMMIT
        or source.get("model_sha256") != source_hash
    ):
        raise ValueError("simulation model/source mismatch")
    validate_policy(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}

    def locomotion(body: NDArray[np.float32], _depth: NDArray[np.float32] | None) -> NDArray[np.float32]:
        if observe_state is not None:
            observe_state(body)
        return locomotion_action(cast(dict[str, NDArray[np.float64]], arrays), body)

    return locomotion, None, digest
