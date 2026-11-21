"""Fixed local simulation command: pinned source, CPU inference and immutable inputs."""

import argparse
import importlib
import os
import subprocess
import tomllib
from pathlib import Path
from typing import cast

from .local import COMMIT


def run_regression(source: Path, model: Path, output: Path, recording: Path) -> None:
    import json

    import numpy as np

    from ..simulation.policy import read_policy
    from ..simulation.recording import FloatArray, camera_contract, compare, vector
    from .provenance import model_digest

    digest = model_digest(source / "unitree_go1")
    state = np.empty(45, dtype=np.float32)

    def observe(value: FloatArray) -> None:
        state[:] = value

    action, config, policy_hash = read_policy(model, digest, observe)
    camera = camera_contract(config)
    output.mkdir()

    def capture(step: int, depth: str | None) -> list[float]:
        if depth is not None:
            (output / f"depth-{step}.json").write_text(
                json.dumps(
                    {
                        "width": camera["width"],
                        "height": camera["height"],
                        "f32le": depth,
                    }
                )
            )
        return vector(state, 45).tolist()

    report = compare(recording, action, camera, digest, capture=capture)
    report["candidate_sha256"] = policy_hash
    (output / "result.json").write_text(json.dumps(report, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--inject-vision", action="store_true")
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("MUJOCO_GL", "egl")
    commit = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True, timeout=5).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(args.source), "status", "--porcelain", "--", "unitree_go1"], text=True, timeout=5
    )
    if commit != COMMIT or dirty.strip():
        raise ValueError("simulation requires the pinned unmodified Go1 source")
    if args.recording and not args.inject_vision:
        run_regression(args.source, args.model, args.output, args.recording)
        return
    run_simulation(
        args.source,
        args.model,
        args.output,
        args.config,
        args.recording if args.inject_vision else None,
        inject_cases=args.inject_vision and args.recording is None,
    )


def run_simulation(
    source: Path,
    model: Path,
    output: Path,
    config_path: Path,
    recording: Path | None = None,
    *,
    inject_cases: bool = False,
) -> None:
    import json

    module = importlib.import_module("aiops_go1_training.simulation.run")
    config_module = importlib.import_module("aiops_go1_training.simulation.configuration")
    document = tomllib.loads(config_path.read_text())
    if "cases" not in document:
        if inject_cases:
            raise ValueError("missing visual injection input")
        module.run(source, model, output, config_module.SimulationConfig.model_validate(document), recording)
        return
    cases = document["cases"]
    if recording is not None:
        raise ValueError("visual injection requires a single scene")
    if set(document) != {"cases"} or not isinstance(cases, list):
        raise ValueError("simulation set must contain 1–16 scene configurations")
    documents = cast(list[object], cases)
    if not 1 <= len(documents) <= 16:
        raise ValueError("simulation set must contain 1–16 scene configurations")
    configs = [config_module.SimulationConfig.model_validate(case) for case in documents]
    output.mkdir()
    for index, config in enumerate(configs, start=1):
        input_recording = config_path.parent / f"input-case-{index}.mcap" if inject_cases else None
        module.run(source, model, output / f"case-{index}", config, input_recording)
        print(json.dumps({"event": "simulation.case-complete", "step": index, "total": len(configs)}), flush=True)


if __name__ == "__main__":
    main()
