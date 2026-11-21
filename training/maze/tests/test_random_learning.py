"""Independent-map course contract and real policy rollouts."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from conftest import workspace_root

from aiops_maze_training.lessons.fixed_map_v1.environment import goal_distances, initial_state
from aiops_maze_training.lessons.random_map_v1.environment import encode, generate_maze, samples, split_maps
from aiops_maze_training.lessons.random_map_v1.train import train


class RandomLearningTests(unittest.TestCase):
    def test_generator_encoding_and_disjoint_content(self) -> None:
        maze = generate_maze(7)
        self.assertEqual(maze, generate_maze(7))
        self.assertIn((1, 1), goal_distances(maze))
        observation = encode(maze, initial_state(maze))
        self.assertEqual(len(observation), 151)
        self.assertEqual(sum(observation[49:98]), 1)
        self.assertEqual(sum(observation[98:147]), 1)
        self.assertEqual(samples(maze)[0].get("maze_id"), "odymaze-random-7")
        splits = split_maps(16, 7)
        contents = [tuple(maze["rows"]) for maps in splits.values() for maze in maps]
        self.assertEqual(len(contents), len(set(contents)))
        self.assertEqual(len(splits["test"]), 100)

    def test_zero_epoch_is_honest_baseline(self) -> None:
        policy, report, dataset = train(epochs=0, seed=7, maps=16)
        self.assertEqual(policy["format"], "odymaze-policy/v2")
        self.assertEqual(policy.get("lesson"), "random-map-v1")
        self.assertEqual(len(policy["layers"][0]["weight"]), 128)
        self.assertEqual(len(dataset[0]["observation"]), 151)
        self.assertEqual(report["selected_epoch"], 0)
        self.assertEqual(report["initial"]["success_rate"], report["final"]["success_rate"])
        self.assertEqual(len(report["final"]["episodes"]), 100)

    def test_invalid_parameters_are_rejected(self) -> None:
        for maps in (0, 15, 2049):
            with self.assertRaises(ValueError):
                split_maps(maps, 7)
        for seed in (-1, 2**32):
            with self.assertRaises(ValueError):
                generate_maze(seed)
        with self.assertRaises(ValueError):
            train(epochs=-1)

    def test_random_cli_exports_and_refuses_overwrite(self) -> None:
        temp_root = workspace_root() / "tmp" / "odymaze-2515"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as parent:
            output = Path(parent) / "run"
            command = [
                sys.executable,
                "-m",
                "aiops_maze_training.runtime.train",
                "--lesson",
                "random-map-v1",
                "--maps",
                "16",
                "--epochs",
                "0",
                "--output",
                str(output),
            ]
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            before = (output / "model.json").read_bytes()
            self.assertEqual(json.loads(before)["lesson"], "random-map-v1")
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(len(report["splits"]["test"]), 100)
            refused = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
            self.assertEqual(refused.returncode, 2)
            self.assertEqual((output / "model.json").read_bytes(), before)
            invalid = subprocess.run(
                command + ["--lesson", "unknown"], check=False, capture_output=True, text=True, timeout=30
            )
            self.assertEqual(invalid.returncode, 2)
