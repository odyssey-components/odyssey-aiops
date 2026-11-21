"""Upstream-compatible warm-start CLI; shared worker lifecycle and explicit device."""

import argparse
import json
import os
import tomllib
from pathlib import Path

import jax

from ..runtime.provenance import require_source
from ..upstream.artifacts import load
from ..upstream.train import run
from .devices import MODES, configure_device, select_device
from .local import environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", choices=MODES, default=os.environ.get("AIOPS_GO1_DEVICE", "cpu"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--validate-output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base-policy", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configure_device(args.device)
    source_hash = require_source(args.source)
    if args.validate_output:
        load(args.validate_output, source_hash)
        print(json.dumps({"valid": True}))
        return
    if args.check:
        print(json.dumps(environment(args.source, args.device)))
        return
    if not args.config or not args.base_policy or not args.output:
        parser.error("--config, --base-policy and --output required")
    config = tomllib.loads(args.config.read_text())["training"]
    from ..locomotion.train import validate_config

    random_commands = config.pop("random_commands", 0)
    if type(random_commands) is not int or random_commands not in (0, 1, 2, 3):
        parser.error("random_commands must be 0, 1, 2, or 3")
    validated = validate_config(config)
    numeric = {
        k: float(v)
        for k, v in validated.items()
        if k not in ("monitor_metrics", "monitor_interval") and isinstance(v, (int, float))
    }
    if random_commands:
        numeric["random_commands"] = float(random_commands)
    device = select_device(args.device)
    with jax.default_device(device):
        run(args.source, args.output, numeric, args.base_policy, source_hash, device)


if __name__ == "__main__":
    main()
