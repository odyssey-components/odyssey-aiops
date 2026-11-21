"""Local adapter checks use only synthetic files; never initialize a training GPU."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiops_go1_training.locomotion.configuration import load_training_config
from aiops_go1_training.runtime.local import COMMIT, environment


class LocalGo1Tests(unittest.TestCase):
    def test_missing_source_is_not_ready(self) -> None:
        with patch("importlib.util.find_spec", return_value=None):
            result = environment(Path("/nonexistent-go1-fixture"))
        self.assertFalse(result["ready"])
        self.assertEqual(result["source_commit"], COMMIT)
        self.assertTrue(result["errors"])

    def test_local_file_overrides_inherited_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.toml"
            path.write_text("[training]\niterations=1\nrollout_steps=8\n")
            with patch.dict(os.environ, {"ODYLAB_GO1_TRAINING_ITERATIONS": "999"}):
                config = load_training_config(path, overrides={"training.iterations": 1})
            self.assertEqual(config["iterations"], 1)
            self.assertEqual(config["rollout_steps"], 8)
