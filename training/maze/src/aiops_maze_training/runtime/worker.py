"""Maze child lifecycle entry with a project-owned module allowlist."""

from odylab_training.local_worker import main as run_worker


def main() -> None:
    run_worker(("aiops_maze_training.common.datasets", "aiops_maze_training.runtime.train"))


if __name__ == "__main__":
    main()
