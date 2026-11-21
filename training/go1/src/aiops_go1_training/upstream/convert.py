"""Convert a declared pinned Go1 default SB3 PPO zip without loading its pickled config."""

import argparse
import io
import json
import pickle
import uuid
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import jax.numpy as jnp

from ..runtime.provenance import require_source
from .artifacts import PROTOCOL, digest, save
from .policy import SHAPES, Params


def convert(archive_path: Path, output: Path, source_hash: str, title: str) -> None:
    import torch

    origin = digest(archive_path, 16 << 20)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > 16 or sum(i.file_size for i in archive.infolist()) > 32 << 20:
            raise ValueError("invalid or oversized SB3 archive")
        if archive.getinfo("data").file_size > 1 << 20 or archive.getinfo("policy.pth").file_size > 4 << 20:
            raise ValueError("oversized SB3 policy/config")
        data = json.loads(archive.read("data"))
        policy_class = data.get("policy_class", {})
        if (
            data.get("policy_kwargs") != {}
            or data.get("use_sde") is not False
            or policy_class.get("__module__") != "stable_baselines3.common.policies"
        ):
            raise ValueError("only default SB3 PPO MlpPolicy supported")
        for key, shape in (("observation_space", [45]), ("action_space", [12])):
            space = data.get(key, {})
            if space.get("_shape") != shape or "gymnasium.spaces.box.Box" not in space.get(":type:", ""):
                raise ValueError("unsupported Go1 spaces")
        action = data["action_space"]
        # Use readable metadata only, never deserialize gym spaces or arbitrary callbacks.
        for name, expected in (("low", -1.0), ("high", 1.0)):
            import numpy as np

            bound = np.fromstring(action.get(name, "").strip("[]"), sep=" ")
            if bound.shape != (12,) or not np.all(bound == expected):
                raise ValueError("unsupported action bounds")
        if any("vecnormalize" in name.lower() for name in names):
            raise ValueError("VecNormalize unsupported")
        # weights_only rejects arbitrary globals; no retry with unrestricted pickle.
        try:
            state = cast(
                object, torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True)
            )
        except (pickle.UnpicklingError, RuntimeError) as error:
            raise ValueError("policy checkpoint must contain only numeric tensor weights") from error
    params: Params = {}
    mapping: dict[str, str] = {"log_std": "log_std"}
    for branch, sb in (("pi", "policy_net"), ("vf", "value_net")):
        for i in range(3):
            source = f"mlp_extractor.{sb}.{i * 2}" if i < 2 else "action_net" if branch == "pi" else "value_net"
            for part in ("weight", "bias"):
                mapping[f"{branch}{i}.{part}"] = f"{source}.{part}"
    if not isinstance(state, Mapping):
        raise ValueError("unsupported upstream state dictionary")  # noqa: TRY004 - stable external artifact rejection contract.
    state = cast(Mapping[str, object], state)
    if set(state) != set(mapping.values()):
        raise ValueError("unsupported upstream parameter keys")
    for name, original in mapping.items():
        tensor = state[original]
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise ValueError("only float32 tensor weights supported")
        array = tensor.detach().cpu().numpy()
        if name.endswith(".weight"):
            array = array.T
        if array.shape != SHAPES[name]:
            raise ValueError("unsupported upstream network shape")
        params[name] = jnp.asarray(array)
    save(output, params, source_hash, title, origin=origin)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--observation-protocol",
        choices=[PROTOCOL],
        required=True,
        help="Attest pinned upstream observation semantics and absence of VecNormalize; zip alone cannot prove these.",
    )
    args = parser.parse_args()
    source_hash = require_source(args.source)
    identifier = str(uuid.uuid4())
    convert(args.archive, args.data_dir / "bases" / identifier, source_hash, args.title)
    print(json.dumps({"id": identifier, "title": args.title}))


if __name__ == "__main__":
    main()
