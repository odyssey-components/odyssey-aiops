"""Convert the pinned Walk These Ways release to inert numerical weights."""

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from ..runtime.provenance import model_digest
from .walk_these_ways import DIGESTS, FORMAT, UPSTREAM, WalkTheseWays, weights


class Tensor(Protocol):
    def cpu(self) -> Tensor: ...
    def numpy(self) -> NDArray[np.float32]: ...


class PolicyModule(Protocol):
    def state_dict(self) -> dict[str, Tensor]: ...
    def __call__(self, value: Tensor) -> Tensor: ...


class TorchJIT(Protocol):
    def load(self, path: str, *, map_location: str) -> PolicyModule: ...


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    jit = cast(TorchJIT, torch.jit)
    from_numpy = cast(Callable[[NDArray[np.float32]], Tensor], torch.from_numpy)  # pyright: ignore[reportUnknownMemberType]
    concatenate = cast(Callable[[tuple[Tensor, Tensor]], Tensor], torch.cat)
    modules: dict[str, PolicyModule] = {}
    tensors: dict[str, NDArray[np.float32]] = {}
    for name, filename in [("adaptation", "adaptation_module_latest.jit"), ("body", "body_latest.jit")]:
        path = args.checkpoints / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != DIGESTS[name]:
            raise ValueError("checkpoint differs from the pinned release")
        module = jit.load(str(path), map_location="cpu")
        modules[name] = module
        tensors.update({f"{name}.{key}": value.cpu().numpy() for key, value in module.state_dict().items()})
    args.output.mkdir(parents=True)
    path = args.output / "policy.npz"
    np.savez_compressed(path, allow_pickle=False, **tensors)
    adapter = WalkTheseWays.__new__(WalkTheseWays)
    adapter.params = weights(path)
    rng = np.random.default_rng(7)
    with torch.inference_mode():
        for _ in range(10):
            sample = rng.normal(size=2100).astype(np.float32)
            sample_tensor = from_numpy(sample)
            reference = modules["body"](concatenate((sample_tensor, modules["adaptation"](sample_tensor)))).numpy()
            prediction = adapter.forward("body", np.concatenate((sample, adapter.forward("adaptation", sample))))
            np.testing.assert_allclose(prediction, reference, rtol=1e-4, atol=3e-5)
    manifest = {
        "format": FORMAT,
        "title": "Walk These Ways · Go1",
        "source_sha256": model_digest(args.source / "unitree_go1"),
        "upstream_commit": UPSTREAM,
        "policy_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (args.output / "public-policy.json").write_text(json.dumps(manifest) + "\n")


if __name__ == "__main__":
    main()
