"""Migrate every historical run/job, retaining per-directory rollback backups."""

import argparse
import json
import tomllib
from pathlib import Path

from aiops_maze_training.runtime.migrate_run import migrate_job, migrate_run


def migrate_storage(base: Path, backup: Path) -> dict[str, object]:
    if base.resolve() != base.absolute() or not base.is_dir():
        raise ValueError("storage root must be a real directory")
    backup.mkdir(parents=True, exist_ok=False)
    migrated: list[str] = []
    unchanged: list[str] = []
    empty: list[str] = []
    failures: dict[str, str] = {}
    for category in ("runs", "jobs"):
        for directory in sorted((base / category).iterdir()):
            identity = f"{category}/{directory.name}"
            try:
                if directory.is_symlink() or not directory.is_dir():
                    raise ValueError("unexpected or linked storage entry")
                if not any(directory.iterdir()):
                    empty.append(identity)
                    continue
                if category == "runs":
                    policy = json.loads((directory / "model.json").read_text())
                    if (
                        (directory / "model.manifest.json").exists()
                        or (directory / "dataset.jsonl").exists()
                        or "maze" in policy
                    ):
                        migrate_run(directory, base / "datasets", backup / identity)
                        migrated.append(identity)
                    else:
                        if not (directory / "model.manifest.toml").is_file():
                            raise ValueError("missing TOML manifest")
                        tomllib.loads((directory / "model.manifest.toml").read_text())
                        unchanged.append(identity)
                elif (directory / "maps.jsonl").exists():
                    migrate_job(directory, base / "datasets", backup / identity)
                    migrated.append(identity)
                else:
                    if any(directory.rglob("*.manifest.json")) or any(directory.rglob("dataset.jsonl")):
                        raise ValueError("unfinished nested output requires an explicit migration")
                    unchanged.append(identity)
            except (OSError, ValueError, KeyError, TypeError) as error:
                failures[identity] = str(error)
    return {"migrated": migrated, "unchanged": unchanged, "empty": empty, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate all runs and jobs to TOML and map references")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    result = migrate_storage(args.base.absolute(), args.backup.absolute())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
