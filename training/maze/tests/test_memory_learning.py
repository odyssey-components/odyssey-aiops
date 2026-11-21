"""The fourth lesson preserves time and learns its own hidden state."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from conftest import workspace_root

import torch
from aiops_maze_training.lessons.fixed_map_v1.environment import initial_state, step
from aiops_maze_training.lessons.memory_map_v1.environment import encode, generate_maze, samples
from aiops_maze_training.lessons.memory_map_v1.model import make_model
from aiops_maze_training.lessons.memory_map_v1.train import pad_episodes, sequence_loss, train
from aiops_maze_training.common.types import Maze


class MemoryLearningTests(unittest.TestCase):
    def test_sensor_cannot_read_hidden_layout(self) -> None:
        maze = generate_maze(7)
        state = initial_state(maze)
        changed: Maze = {**maze, "rows": maze["rows"][:5] + ["#####.#", maze["rows"][6]]}
        self.assertEqual(encode(maze, state), encode(changed, state))
        observation = encode(maze, state)
        self.assertEqual(len(observation), 40)
        self.assertEqual(observation[36:], [1, 0, 0, 0])
        self.assertEqual(sum(observation[18:25]), 1)
        with self.assertRaises(ValueError):
            encode(maze, state, 3)

    def test_step_matches_sequence_and_reset(self) -> None:
        torch.manual_seed(7)  # pyright: ignore[reportUnknownMemberType]
        model = make_model()
        observations = torch.randn(1, 9, 40)
        hidden: torch.Tensor | None = None
        steps: list[torch.Tensor] = []
        with torch.no_grad():
            for observation in observations[0]:
                logits, hidden = model.step(observation, hidden)
                steps.append(logits)
            torch.testing.assert_close(torch.stack(steps), model(observations)[0])
            first, _ = model.step(observations[0, 0])
            torch.testing.assert_close(first, steps[0])
            torch.testing.assert_close(model(observations), model(observations))

    def test_ordered_episodes_padding_and_previous_action(self) -> None:
        episodes = [samples(generate_maze(seed)) for seed in (7, 9, 11)]
        for episode in episodes:
            maze = generate_maze(int(str(episode[0].get("maze_id")).rsplit("-", 1)[1]))
            state = initial_state(maze)
            previous = -1
            for row in episode:
                self.assertNotIn("memory", row)
                self.assertEqual(row.get("previous_action"), previous)
                self.assertEqual(row["observation"], encode(maze, state, previous))
                state = step(maze, state, row["action"])
                previous = row["action"]
        features, labels = pad_episodes(episodes)
        logits = make_model()(features)
        valid = labels != -100
        expected = torch.nn.functional.cross_entropy(logits[valid], labels[valid])
        torch.testing.assert_close(sequence_loss(logits, labels), expected)
        changed = logits.clone()
        changed[~valid] = 1000
        torch.testing.assert_close(sequence_loss(changed, labels), expected)
        for index, episode in enumerate(episodes):
            torch.testing.assert_close(
                features[index, : len(episode)], torch.tensor([r["observation"] for r in episode])
            )

    def test_zero_epoch_deterministic_export(self) -> None:
        policy, report, dataset = train(epochs=0, maps=16)
        self.assertEqual(policy["format"], "odymaze-policy/v4")
        self.assertEqual(policy.get("lesson"), "memory-map-v1")
        gru = policy.get("gru")
        assert gru is not None
        self.assertEqual(len(gru["weight_ih"]), 96)
        self.assertEqual(len(gru["weight_ih"][0]), 40)
        self.assertEqual(len(gru["weight_hh"][0]), 32)
        self.assertEqual(len(gru["bias_ih"]), 96)
        self.assertEqual(len(policy["layers"][0]["weight"]), 3)
        self.assertEqual(report["initial"], report["final"])
        self.assertEqual(report["selected_epoch"], 0)
        self.assertEqual((policy, report, dataset), train(epochs=0, maps=16))
        self.assertLess(len(json.dumps(policy)), 1024 * 1024)

    def test_cli_and_refusals(self) -> None:
        root = workspace_root() / "tmp" / "odymaze-memory-tests"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as output:
            command = [
                sys.executable,
                "-m",
                "aiops_maze_training.runtime.train",
                "--lesson",
                "memory-map-v1",
                "--maps",
                "16",
                "--epochs",
                "0",
                "--output",
                output,
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((Path(output) / "model.json").read_text())["lesson"], "memory-map-v1")
            self.assertEqual(subprocess.run(command, capture_output=True, timeout=60, check=False).returncode, 2)
            for invalid in (["--maps", "15"], ["--epochs", "-1"], ["--seed", "-1"]):
                result = subprocess.run(command + invalid, capture_output=True, timeout=60, check=False)
                self.assertEqual(result.returncode, 2)
