"""Run a Maze command without importing training engines at startup."""

import argparse
import importlib
import sys

ENTRIES = {
    "train": "aiops_maze_training.runtime.train",
    "prepare": "aiops_maze_training.common.datasets",
    "export": "aiops_maze_training.runtime.export",
    "worker": "aiops_maze_training.runtime.worker",
    "migrate-run": "aiops_maze_training.runtime.migrate_run",
    "migrate-storage": "aiops_maze_training.runtime.migrate_storage",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Maze training commands")
    parser.add_argument("command", choices=tuple(ENTRIES))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    module = ENTRIES[args.command]
    sys.argv = [module, *args.args]
    importlib.import_module(module).main()


if __name__ == "__main__":
    main()
