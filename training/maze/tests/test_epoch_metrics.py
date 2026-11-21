"""Every plotted epoch is a measured rollout, with validation-only selection."""

import io
import json
import unittest
from contextlib import redirect_stdout
from typing import cast
from unittest.mock import patch

from aiops_maze_training.lessons.fixed_map_v1.environment import load_maze
from aiops_maze_training.lessons.fixed_map_v1.train import train as train_fixed
from aiops_maze_training.lessons.local_map_v1.train import train as train_local
from aiops_maze_training.lessons.memory_map_v1.train import train as train_memory
from aiops_maze_training.lessons.random_map_v1.environment import split_maps
from aiops_maze_training.lessons.random_map_v1.train import Scores
from aiops_maze_training.lessons.random_map_v1.train import train as train_random
from aiops_maze_training.common.metrics import Metric, emit_metric


def metrics(output: str) -> list[Metric]:
    return [
        cast(Metric, json.loads(line.removeprefix("ODYMAZE_METRIC ")))
        for line in output.splitlines()
        if line.startswith("ODYMAZE_METRIC ")
    ]


class EpochMetricsTests(unittest.TestCase):
    def test_full_precision_and_flush(self) -> None:
        with patch("builtins.print") as output:
            emit_metric(2, 3, 1 / 3, 64, "validation", loss=0.123456789, accuracy=0.75)
        args, kwargs = output.call_args
        metric = metrics(str(args[0]))[0]
        self.assertEqual(metric["successRate"], 1 / 3)
        self.assertEqual(metric.get("loss"), 0.123456789)
        self.assertTrue(kwargs["flush"])

    def test_fixed_measures_each_updated_model(self) -> None:
        for epochs in (0, 3):
            with self.subTest(epochs=epochs), redirect_stdout(io.StringIO()) as output:
                _, report = train_fixed(load_maze(), epochs=epochs)
            rows = metrics(output.getvalue())
            self.assertEqual([row["epoch"] for row in rows], list(range(epochs + 1)))
            self.assertEqual(len(report["history"]), epochs)
            self.assertEqual(
                [row["successRate"] for row in rows], [row["success_rate"] for row in report.get("validation", [])]
            )
            self.assertEqual(rows[-1]["successRate"], float(report["final"]["success"]))
            for row in rows:
                self.assertEqual((row["total"], row["split"], row["evaluatedMaps"]), (epochs, "fixed", 1))
                self.assertIn(row["successRate"], (0.0, 1.0))

    def test_each_validation_epoch_selects_first_best_and_uses_prepared_maps(self) -> None:
        splits = split_maps(16, 11)
        for lesson, trainer in (
            ("random_map_v1", train_random),
            ("local_map_v1", train_local),
            ("memory_map_v1", train_memory),
        ):
            for epochs in (0, 3):
                with self.subTest(lesson=lesson, epochs=epochs):
                    scores = [1 / 64, 17 / 64, 42 / 64, 42 / 64][: epochs + 1]
                    evaluations: list[Scores] = [{"success_rate": score, "episodes": []} for score in scores]
                    heldout: Scores = {"success_rate": 0.0, "episodes": []}
                    evaluations.extend([heldout] * 3)
                    prefix = f"aiops_maze_training.lessons.{lesson}.train"
                    with (
                        patch(prefix + ".evaluate_maps", side_effect=evaluations) as evaluate,
                        patch(prefix + ".split_maps") as generate,
                        redirect_stdout(io.StringIO()) as output,
                    ):
                        _, report, _ = trainer(epochs=epochs, splits=splits)
                    generate.assert_not_called()
                    rows = metrics(output.getvalue())
                    self.assertEqual([row["epoch"] for row in rows], list(range(epochs + 1)))
                    self.assertEqual([row["successRate"] for row in rows], scores)
                    self.assertEqual([row["success_rate"] for row in report.get("validation", [])], scores)
                    self.assertEqual([row.get("success_rate") for row in report["history"]], scores[1:])
                    self.assertEqual(report["selected_epoch"], 2 if epochs else 0)
                    self.assertEqual(
                        report["splits"]["train"], [int(m["id"].rsplit("-", 1)[1]) for m in splits["train"]]
                    )
                    for index, row in enumerate(rows):
                        self.assertEqual((row["split"], row["evaluatedMaps"], row["total"]), ("validation", 64, epochs))
                        self.assertIs(evaluate.call_args_list[index].args[1], splits["validation"])
                        if index:
                            self.assertEqual(row.get("loss"), report["history"][index - 1]["loss"])
                            self.assertEqual(row.get("accuracy"), report["history"][index - 1]["accuracy"])
                    for call in evaluate.call_args_list[epochs + 1 :]:
                        self.assertIs(call.args[1], splits["test"])
