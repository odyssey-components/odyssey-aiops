"""Run a Go1 command without importing training engines at package startup."""

import argparse
import importlib
import sys

ENTRIES = {
    "locomotion": "aiops_go1_training.runtime.local",
    "vision": "aiops_go1_training.vision.train",
    "simulation": "aiops_go1_training.runtime.simulation",
    "worker": "aiops_go1_training.runtime.worker",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Go1 training commands")
    parser.add_argument("command", choices=tuple(ENTRIES))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    module = ENTRIES[args.command]
    sys.argv = [module, *args.args]
    importlib.import_module(module).main()


if __name__ == "__main__":
    main()
