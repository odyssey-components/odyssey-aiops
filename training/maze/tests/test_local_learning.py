"""Local observations cannot reveal hidden layouts or use a full-map teacher."""

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from conftest import workspace_root

from aiops_maze_training.lessons.fixed_map_v1.environment import initial_state, step
from aiops_maze_training.lessons.local_map_v1.environment import encode, expert_action, generate_maze, observe, samples
from aiops_maze_training.lessons.local_map_v1.train import train
from aiops_maze_training.common.types import Maze


class LocalLearningTests(unittest.TestCase):
    def test_hidden_layout_invariance_and_reset(self) -> None:
        maze = generate_maze(7)
        state = initial_state(maze)
        changed: Maze = {**maze, "rows": maze["rows"][:5] + ["#####.#", maze["rows"][6]]}
        memory = observe(maze, state)
        self.assertEqual(memory, observe(changed, state))
        self.assertEqual(len(encode(memory, state)), 249)
        self.assertEqual(sum(memory["visited"]), 1)
        self.assertEqual(memory, observe(maze, state))
        with self.assertRaises(ValueError):
            expert_action(memory, replace(state, status="success"))

    def test_expert_rollouts_reach_diverse_maps(self) -> None:
        for seed in range(100):
            maze = generate_maze(seed)
            state = initial_state(maze)
            memory = observe(maze, state)
            while state.status not in ("success", "timeout"):
                state = step(maze, state, expert_action(memory, state))
                memory = observe(maze, state, memory)
            self.assertEqual(state.status, "success", seed)
        self.assertEqual(samples(generate_maze(7)), samples(generate_maze(7)))

    def test_zero_epoch_export(self) -> None:
        policy, report, dataset = train(epochs=0, maps=16)
        self.assertEqual(policy["format"], "odymaze-policy/v3")
        self.assertEqual(policy.get("lesson"), "local-map-v1")
        self.assertEqual(len(policy["layers"][0]["weight"]), 64)
        self.assertEqual(len(dataset[0]["observation"]), 249)
        self.assertIn("memory", dataset[0])
        self.assertEqual(report["initial"], report["final"])
        self.assertEqual(report["selected_epoch"], 0)
        repeated_policy, repeated_report, repeated_dataset = train(epochs=0, maps=16)
        self.assertEqual(policy, repeated_policy)
        self.assertEqual(report, repeated_report)
        self.assertEqual(dataset, repeated_dataset)

    def test_expert_uses_observed_routes(self) -> None:
        maze = generate_maze(7)
        state = initial_state(maze)
        memory = observe(maze, state)
        memory["known"] = ["???????", "?.G????", "???????", "???????", "???????", "???????", "???????"]
        self.assertEqual(expert_action(memory, state), 2)
        with self.assertRaises(ValueError):
            expert_action(memory, replace(state, status="timeout"))

    def test_cli_exports_readable_bounded_model(self) -> None:
        root = workspace_root() / "tmp" / "odymaze-local-tests"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as output:
            command = [
                sys.executable,
                "-m",
                "aiops_maze_training.runtime.train",
                "--lesson",
                "local-map-v1",
                "--maps",
                "16",
                "--epochs",
                "0",
                "--output",
                output,
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=40, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = (Path(output) / "model.json").read_bytes()
            self.assertLess(len(data), 1024 * 1024)
            self.assertEqual(json.loads(data)["lesson"], "local-map-v1")
            refused = subprocess.run(command, capture_output=True, text=True, timeout=40, check=False)
            self.assertEqual(refused.returncode, 2)
