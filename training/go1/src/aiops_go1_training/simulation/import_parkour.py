"""Convert the pinned Robot Parkour Learning Go1 student to inert numerical weights."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import cast

import numpy as np

from ..runtime.provenance import model_digest
from .import_walk import Tensor
from .parkour import CHECKPOINT_SHA256, FORMAT, SHAPES, UPSTREAM, weights


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("archive differs from the pinned Go1 student")
    with zipfile.ZipFile(args.archive) as archive:
        model = next(name for name in archive.namelist() if name.endswith("/model_674000.pt"))
        state = cast(
            dict[str, dict[str, Tensor]],
            torch.load(io.BytesIO(archive.read(model)), map_location="cpu", weights_only=True),
        )["model_state_dict"]
    args.output.mkdir(parents=True)
    path = args.output / "policy.npz"
    np.savez_compressed(path, allow_pickle=False, **{name: state[name].cpu().numpy() for name in SHAPES})
    weights(path)
    manifest = {
        "format": FORMAT,
        "title": "Robot Parkour Learning · Go1（试验）",
        "source_sha256": model_digest(args.source / "unitree_go1"),
        "upstream_commit": UPSTREAM,
        "policy_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (args.output / "public-policy.json").write_text(json.dumps(manifest, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
