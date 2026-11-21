"""Behavior tests: movement rules, expert routes, real training and artifacts."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from conftest import workspace_root

from aiops_maze_training.lessons.fixed_map_v1.environment import (
    State,
    encode,
    expert_action,
    load_maze,
    samples,
    step,
)
from aiops_maze_training.runtime.train import train


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.maze = load_maze()

    def test_turns_and_wall_hits_count_steps_without_moving(self):
        state = State(1, 1, 0)
        state = step(self.maze, state, 0)
        self.assertEqual((state.x, state.y, state.direction, state.steps), (1, 1, 3, 1))
        state = step(self.maze, state, 2)
        self.assertEqual((state.x, state.y, state.steps), (1, 1, 2))
        state = step(self.maze, state, 1)
        state = step(self.maze, state, 2)
        self.assertEqual((state.x, state.y, state.direction), (2, 1, 0))

    def test_success_and_timeout_are_terminal(self):
        success = step(self.maze, State(2, 5, 2, 99), 2)
        self.assertEqual(success.status, "success")
        self.assertEqual(step(self.maze, success, 1), success)
        timeout = step(self.maze, State(1, 1, 0, 99), 0)
        self.assertEqual(timeout.status, "timeout")
        self.assertEqual(step(self.maze, timeout, 2), timeout)

    def test_expert_takes_short_route_and_reaches_goal(self):
        state = State(1, 1, 0)
        self.assertEqual(expert_action(self.maze, state), 1)
        for _ in range(100):
            state = step(self.maze, state, expert_action(self.maze, state))
            if state.status == "success":
                break
        self.assertEqual((state.x, state.y, state.steps), (1, 5, 12))

    def test_encoding_has_separate_position_and_heading_features(self):
        features = encode(self.maze, State(2, 3, 1))
        self.assertEqual(len(features), 18)
        self.assertEqual([i for i, value in enumerate(features) if value], [2, 10, 15])
        dataset = samples(self.maze)
        states = {(row["state"]["x"], row["state"]["y"], row["state"]["direction"]) for row in dataset}
        self.assertEqual(len(states), len(dataset))
        self.assertNotIn((1, 5, 0), states)
        self.assertTrue(all(len(row["observation"]) == 18 for row in dataset))


class TrainingTests(unittest.TestCase):
    def test_real_training_reduces_loss_and_completes_maze(self):
        maze = load_maze()
        policy, report = train(maze, epochs=400, seed=7)
        self.assertTrue(report["final"]["success"])
        self.assertEqual(report["final"]["steps"], 12)
        self.assertGreaterEqual(report["final"]["accuracy"], 0.99)
        self.assertLess(report["history"][-1]["loss"], report["history"][0]["loss"])
        self.assertEqual(policy["layers"][0]["activation"], "tanh")
        self.assertEqual(len(policy["layers"][0]["weight"]), 32)

    def test_zero_epochs_and_cli_output_refusal(self):
        _, report = train(load_maze(), epochs=0, seed=7)
        self.assertEqual(report["history"], [])
        self.assertEqual(report["initial"], report["final"])
        temp_root = workspace_root() / "tmp" / "odymaze-2514"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as parent:
            output = Path(parent) / "run"
            command = [
                sys.executable,
                "-m",
                "aiops_maze_training.runtime.train",
                "--epochs",
                "0",
                "--output",
                str(output),
            ]
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            model = output / "model.json"
            before = model.read_bytes()
            self.assertEqual(json.loads(before)["metadata"]["epochs"], 0)
            refused = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
            self.assertEqual(refused.returncode, 2)
            self.assertEqual(model.read_bytes(), before)
            invalid = subprocess.run(
                command + ["--epochs", "-1"], check=False, capture_output=True, text=True, timeout=30
            )
            self.assertEqual(invalid.returncode, 2)
            self.assertEqual(model.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
