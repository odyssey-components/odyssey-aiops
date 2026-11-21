"""Run migrations retain a reversible backup and refuse unregistered maps."""

import hashlib
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from conftest import workspace_root

from aiops_maze_training.common.datasets import prepare_dataset, save_dataset
from aiops_maze_training.runtime.migrate_run import migrate_job, migrate_run


class MigrationTests(unittest.TestCase):
    def test_migration_references_canonical_maps_and_preserves_backup(self) -> None:
        temporary = workspace_root() / "tmp" / "odymaze-migrate"
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as folder:
            root = Path(folder)
            run, datasets, backup = root / "runs" / "run", root / "datasets", root / "backup"
            run.mkdir(parents=True)
            datasets.mkdir()
            value = prepare_dataset("random-map-v1", 7, 16)
            save_dataset(value, datasets / "canonical.jsonl")
            save_dataset(value, run / "dataset.jsonl")
            binary = b"synthetic model"
            (run / "model.onnx").write_bytes(binary)
            manifest = {
                "lesson": "random-map-v1",
                "maze": value["splits"]["train"][0],
                "sha256": hashlib.sha256(binary).hexdigest(),
                "inputs": [{"shape": ["batch", 151]}],
                "outputs": [{"shape": ["batch", 3]}],
            }
            (run / "model.manifest.json").write_text(json.dumps(manifest))
            (run / "model.json").write_text(json.dumps({"lesson": "random-map-v1", "maze": manifest["maze"]}))
            # No canonical match must leave the entire run untouched.
            before = {path.name: path.read_bytes() for path in run.iterdir()}
            (datasets / "canonical.toml").rename(datasets / "hidden")
            with self.assertRaisesRegex(ValueError, "no matching"):
                migrate_run(run, datasets, backup)
            self.assertEqual(before, {path.name: path.read_bytes() for path in run.iterdir()})
            (datasets / "hidden").rename(datasets / "canonical.toml")
            target = migrate_run(run, datasets, backup)
            stored = tomllib.loads(target.read_text())
            self.assertEqual(stored["dataset"]["id"], "canonical")
            self.assertNotIn("maze", stored)
            self.assertEqual((run / "model.onnx").read_bytes(), binary)
            self.assertFalse((run / "dataset.jsonl").exists())
            self.assertFalse((run / "dataset.toml").exists())
            for name in ("model.manifest.json", "model.json", "dataset.jsonl", "dataset.toml"):
                self.assertEqual((backup / name).read_bytes(), before[name])

    def test_metadata_only_run_and_job_snapshots_migrate_without_inventing_provenance(self) -> None:
        temporary = workspace_root() / "tmp" / "odymaze-migrate"
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as folder:
            root = Path(folder)
            run, datasets, job = root / "run", root / "datasets", root / "job"
            for directory in (run, datasets, job):
                directory.mkdir()
            value = prepare_dataset("random-map-v1", 7, 16)
            save_dataset(value, datasets / "canonical.jsonl")
            save_dataset(value, job / "maps.jsonl")
            (job / "job.toml").write_text('course = "random-map-v1"\nstate = "failed"\naction = "train"\n')
            migrate_job(job, datasets, root / "job-backup")
            self.assertEqual(sorted(path.name for path in job.iterdir()), ["job.toml"])
            self.assertEqual(tomllib.loads((job / "job.toml").read_text())["dataset"]["id"], "canonical")
            self.assertTrue((root / "job-backup" / "maps.jsonl").exists())
            binary = b"synthetic model"
            (run / "model.onnx").write_bytes(binary)
            maze = value["splits"]["train"][0]
            manifest = {
                "lesson": "random-map-v1",
                "maze": maze,
                "sha256": hashlib.sha256(binary).hexdigest(),
                "inputs": [{"shape": ["batch", 151]}],
                "outputs": [{"shape": ["batch", 3]}],
            }
            (run / "model.manifest.json").write_text(json.dumps(manifest))
            (run / "model.json").write_text(json.dumps({"lesson": "random-map-v1", "maze": maze}))
            target = migrate_run(run, datasets, root / "run-backup")
            stored = tomllib.loads(target.read_text())
            self.assertNotIn("dataset", stored)
            self.assertEqual(stored["dataset_provenance"], "unrecorded")
            self.assertFalse((run / "model.manifest.json").exists())
            self.assertEqual((run / "model.onnx").read_bytes(), binary)
