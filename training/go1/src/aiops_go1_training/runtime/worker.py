"""Go1 child lifecycle entry with a project-owned module allowlist."""

from odylab_training.local_worker import main as run_worker


def main() -> None:
    run_worker(
        (
            "aiops_go1_training.runtime.local",
            "aiops_go1_training.runtime.upstream",
            "aiops_go1_training.vision.train",
            "aiops_go1_training.runtime.simulation",
        )
    )


if __name__ == "__main__":
    main()
