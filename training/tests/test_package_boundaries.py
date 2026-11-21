"""Compatibility and project worker boundaries after the package split."""

import importlib
import subprocess
import sys
import unittest
from pathlib import Path


class PackageBoundaryTests(unittest.TestCase):
    def test_maze_root_contains_only_entries(self) -> None:
        package = importlib.import_module("aiops_maze_training")
        assert package.__file__ is not None
        root = Path(package.__file__).parent
        self.assertEqual({path.name for path in root.glob("*.py")}, {"__init__.py", "__main__.py"})

    def test_workers_reject_other_project_modules(self) -> None:
        for worker, target in (
            ("aiops_maze_training.runtime.worker", "aiops_go1_training.runtime.local"),
            ("aiops_go1_training.runtime.worker", "aiops_maze_training.runtime.train"),
            ("odylab_training.local_worker", "os"),
        ):
            result = subprocess.run(
                [sys.executable, "-m", worker, "--parent-pid", "2", "--", target],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsupported worker module", result.stderr)
