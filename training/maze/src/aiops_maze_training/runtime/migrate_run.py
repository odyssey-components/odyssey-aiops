"""Explicit, backed-up migrations; legacy JSON is consumed only here."""

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import cast

from aiops_maze_training.common.datasets import load_dataset
from aiops_maze_training.common.manifest import dump_toml, resolve_policy
from aiops_maze_training.common.onnx_export import export_onnx
from aiops_maze_training.common.types import Policy


def dataset_reference(path: Path, datasets: Path, lesson: str) -> dict[str, str]:
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    load_dataset(path, lesson, expected_sha256=checksum)
    matches: list[Path] = []
    for metadata in datasets.glob("*.toml"):
        if metadata.is_symlink():
            continue
        value = tomllib.loads(metadata.read_text())
        if value.get("sha256") == checksum and value.get("lesson") == lesson and value.get("id") == metadata.stem:
            load_dataset(metadata.with_suffix(".jsonl"), lesson, expected_sha256=checksum)
            matches.append(metadata)
    if not matches:
        raise ValueError("no matching canonical dataset; preserve original files")
    return {"id": min(matches).stem, "sha256": checksum}


def originals_at(directory: Path, backup: Path) -> dict[str, bytes]:
    if directory.resolve() != directory.absolute() or not directory.is_dir():
        raise ValueError("migration requires a real directory")
    originals: dict[str, bytes] = {}
    for path in directory.iterdir():
        if path.is_symlink():
            raise ValueError("migration refuses symlinks")
        if path.is_file():
            originals[path.name] = path.read_bytes()
    backup.mkdir(parents=True, exist_ok=False)
    for name, content in originals.items():
        with (backup / name).open("xb") as stream:
            stream.write(content)
    for name, content in originals.items():
        if (directory / name).is_symlink() or (directory / name).read_bytes() != content:
            raise ValueError("directory changed during backup")
    return originals


def restore(directory: Path, originals: dict[str, bytes], created: set[str]) -> None:
    for name in created - originals.keys():
        (directory / name).unlink(missing_ok=True)
    for name, content in originals.items():
        (directory / name).write_bytes(content)


def migrate_run(run: Path, datasets: Path, backup: Path) -> Path:
    run, datasets, backup = run.absolute(), datasets.absolute(), backup.absolute()
    if datasets.resolve() != datasets or not datasets.is_dir():
        raise ValueError("migration requires a real datasets directory")
    if not (run / "model.json").is_file():
        raise ValueError("run has no model policy")
    policy = cast(dict[str, object], json.loads((run / "model.json").read_text()))
    lesson = str(policy.get("lesson", "fixed-map-v1"))
    reference: dict[str, str] | None = None
    if (run / "dataset.jsonl").exists():
        reference = dataset_reference(run / "dataset.jsonl", datasets, lesson)
    legacy = run / "model.manifest.json"
    target = run / "model.manifest.toml"
    manifest = (
        cast(dict[str, object], json.loads(legacy.read_text()))
        if legacy.exists()
        else (cast(dict[str, object], tomllib.loads(target.read_text())) if target.exists() else None)
    )
    if manifest is not None:
        binary = run / "model.onnx"
        if binary.is_symlink() or hashlib.sha256(binary.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("ONNX checksum mismatch")
        if manifest.get("lesson") != lesson:
            raise ValueError("manifest and policy lessons differ")
    for value in (policy, manifest):
        if value is None:
            continue
        if "maze" in value:
            maze = cast(dict[str, object], value.pop("maze"))
            value["map_ref"] = {"id": maze["id"]}
            reconstructed = resolve_policy(cast(Policy, {**policy, "map_ref": value["map_ref"]}))
            if reconstructed["maze"] != maze:
                raise ValueError("map reference cannot reproduce embedded maze")
        if reference is not None:
            value["dataset"] = reference
        elif "dataset" not in value:
            value["dataset_provenance"] = "unrecorded"
    if manifest is not None:
        for tensor in cast(list[dict[str, object]], manifest["inputs"]) + cast(
            list[dict[str, object]], manifest["outputs"]
        ):
            if isinstance(tensor["shape"], list):
                shape = cast(list[object], tensor["shape"])
                tensor["shape"] = {"batch": shape[0], "width": shape[1]}
    originals = originals_at(run, backup)
    created = {"model.onnx", "model.manifest.toml", "model.json.tmp", "model.manifest.toml.tmp"}
    try:
        if manifest is None:
            if (run / "model.onnx").exists():
                raise ValueError("unpaired ONNX file; refusing to replace")
            export_onnx(cast(Policy, policy), run / "model.onnx")
            manifest = cast(dict[str, object], tomllib.loads(target.read_text()))
            manifest["dataset_provenance"] = "unrecorded"
        encoded = dump_toml(manifest)
        tomllib.loads(encoded)
        for name, content in [
            ("model.manifest.toml", encoded),
            ("model.json", json.dumps(policy, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
        ]:
            temporary = run / (name + ".tmp")
            with temporary.open("x") as stream:
                stream.write(content)
            temporary.replace(run / name)
        for name in ("dataset.jsonl", "dataset.toml", "model.manifest.json"):
            (run / name).unlink(missing_ok=True)
    except BaseException:
        restore(run, originals, created)
        raise
    return target


def migrate_job(job: Path, datasets: Path, backup: Path) -> Path:
    metadata = job / "job.toml"
    value = cast(dict[str, object], tomllib.loads(metadata.read_text())) if metadata.exists() else {}
    if value.get("state") in ("running", "cancelling", "finalizing"):
        raise ValueError("refusing to migrate an active job")
    maps = job / "maps.jsonl"
    lesson = str(value.get("course") or tomllib.loads(maps.with_suffix(".toml").read_text())["lesson"])
    reference = dataset_reference(maps, datasets, lesson)
    value.update({"format": "odymaze-job/v1", "id": job.name, "course": lesson, "dataset": reference})
    if "state" not in value:
        value.update(
            {"state": "unknown", "action": "train", "migration_note": "original lifecycle metadata was not recorded"}
        )
    originals = originals_at(job, backup)
    try:
        with (job / "job.toml.tmp").open("x") as stream:
            stream.write(dump_toml(value))
        (job / "job.toml.tmp").replace(metadata)
        maps.unlink()
        maps.with_suffix(".toml").unlink()
    except BaseException:
        restore(job, originals, {"job.toml", "job.toml.tmp"})
        raise
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate one run with an explicit backup directory")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    print(migrate_run(args.run, args.datasets, args.backup))


if __name__ == "__main__":
    main()
