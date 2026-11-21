"""The package root contains only initialization and the command entry."""

import importlib
from pathlib import Path

import aiops_go1_training


def test_root_contains_only_entries() -> None:
    root = Path(aiops_go1_training.__file__).parent
    assert {path.name for path in root.glob("*.py")} == {"__init__.py", "__main__.py"}


def test_canonical_subpackages_import() -> None:
    for name in (
        "locomotion.configuration",
        "locomotion.model_manifest",
        "runtime.local",
        "runtime.provenance",
        "runtime.worker",
        "vision.configuration",
        "vision.policy",
        "vision.artifacts",
    ):
        assert importlib.import_module(f"aiops_go1_training.{name}")
