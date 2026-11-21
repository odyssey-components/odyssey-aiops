"""Numeric prepared maps are immutable training inputs, not regenerated seeds."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiops_maze_training.common.datasets import LESSONS, load_dataset, prepare_dataset, read_map, save_dataset
from aiops_maze_training.lessons.local_map_v1.train import train as train_local
from aiops_maze_training.lessons.memory_map_v1.train import train as train_memory
from aiops_maze_training.lessons.random_map_v1.train import train as train_random


class DatasetTests(unittest.TestCase):
    def test_numeric_roundtrip_and_disjoint_maps(self) -> None:
        for lesson in LESSONS:
            with self.subTest(lesson=lesson), tempfile.TemporaryDirectory() as directory:
                value = prepare_dataset(lesson, 7, 16, title='课堂 "A"', description="地图组\n说明")
                path = Path(directory) / "maps.jsonl"
                save_dataset(value, path)
                loaded = load_dataset(path, lesson)
                self.assertEqual(loaded["lesson"], lesson)
                self.assertEqual(loaded["splits"], value["splits"])
                self.assertEqual(loaded["title"], value["title"])
                self.assertEqual(loaded["description"], value["description"])
                raw = json.loads(path.read_text().splitlines()[0])
                self.assertTrue(path.with_suffix(".toml").is_file())
                self.assertEqual(
                    len(path.read_text().splitlines()), sum(len(part) for part in value["splits"].values())
                )
                cells = raw["cells"]
                self.assertEqual({cell for row in cells for cell in row}, {-1, 0, 1})
                with self.assertRaises(FileExistsError):
                    save_dataset(value, path)
                with self.assertRaises(ValueError):
                    load_dataset(path, "future")

    def test_job_reference_requires_the_expected_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "maps.jsonl"
            save_dataset(prepare_dataset("fixed-map-v1", 7, 16), path)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(load_dataset(path, "fixed-map-v1", expected_sha256=checksum)["lesson"], "fixed-map-v1")
            with self.assertRaisesRegex(ValueError, "job.*checksum"):
                load_dataset(path, "fixed-map-v1", expected_sha256="0" * 64)

    def test_invalid_map_and_overlap_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "maps.jsonl"
            save_dataset(prepare_dataset("random-map-v1", 7, 16), path)
            original = path.read_text()
            raw = json.loads(original.splitlines()[0])
            for bad in (2, True, "1"):
                item = json.loads(json.dumps(raw))
                item["cells"][0][0] = bad
                with self.assertRaises(ValueError):
                    read_map(item, "random-map-v1")
            path.write_text(original + original.splitlines()[0] + "\n")
            with self.assertRaises(ValueError):
                load_dataset(path, "random-map-v1")
            path.write_text(original)
            metadata = path.with_suffix(".toml")
            metadata.write_text(metadata.read_text().replace('lesson = "random-map-v1"', 'lesson = "local-map-v1"'))
            with self.assertRaises(ValueError):
                load_dataset(path, "random-map-v1")

    def test_duplicate_map_with_matching_checksum_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_dataset("random-map-v1", 7, 16)
            prepared["splits"]["test"][0] = prepared["splits"]["train"][0]
            path = Path(directory) / "overlap.jsonl"
            save_dataset(prepared, path)
            with self.assertRaises(ValueError):
                load_dataset(path, "random-map-v1")

    def test_trainers_never_generate_when_given_prepared_maps(self) -> None:
        for lesson, trainer in (
            ("random-map-v1", train_random),
            ("local-map-v1", train_local),
            ("memory-map-v1", train_memory),
        ):
            prepared = prepare_dataset(lesson, 7, 16)
            with (
                self.subTest(lesson=lesson),
                patch(
                    f"aiops_maze_training.lessons.{lesson.replace('-', '_')}.train.split_maps",
                    side_effect=AssertionError("generated during training"),
                ),
            ):
                model, report, _ = trainer(1, 7, 16, splits=prepared["splits"])
                self.assertEqual(model.get("lesson"), lesson)
                self.assertEqual(report["maps"], 16)
