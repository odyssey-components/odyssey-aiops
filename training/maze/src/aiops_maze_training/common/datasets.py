"""Prepare numeric world maps once; training consumes validated stored maps."""

import argparse
import hashlib
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast

from aiops_maze_training.lessons.fixed_map_v1.environment import goal_distances, load_maze
from aiops_maze_training.lessons.local_map_v1.environment import split_maps as split_local
from aiops_maze_training.lessons.random_map_v1.environment import split_maps as split_random
from aiops_maze_training.common.types import Maze

LESSONS = ("fixed-map-v1", "random-map-v1", "local-map-v1", "memory-map-v1")
FORMAT = "odymaze-maps/v2"
VALUES = {-1: "#", 1: ".", 0: "G"}
MAX_BYTES = 4 * 1024 * 1024


class MapDataset(TypedDict):
    lesson: str
    seed: int
    title: str
    description: str
    created_at: str
    splits: dict[str, list[Maze]]


def prepare_dataset(lesson: str, seed: int, maps: int, *, title: str = "", description: str = "") -> MapDataset:
    if lesson not in LESSONS or type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("invalid lesson or seed")
    if type(maps) is not int or not 16 <= maps <= 2048:
        raise ValueError("map count must be 16..2048")
    if len(title) > 80 or len(description) > 500:
        raise ValueError("group title or description too long")
    splits = (
        {"train": [load_maze()], "validation": [], "test": []}
        if lesson == "fixed-map-v1"
        else (split_random if lesson == "random-map-v1" else split_local)(maps, seed)
    )
    return {
        "lesson": lesson,
        "seed": seed,
        "title": title.strip() or lesson + " 地图组",
        "description": description,
        "created_at": datetime.now(UTC).isoformat(),
        "splits": splits,
    }


def numeric_map(maze: Maze) -> dict[str, object]:
    values = {value: key for key, value in VALUES.items()}
    return {
        "id": maze["id"],
        "cells": [[values[cell] for cell in row] for row in maze["rows"]],
        "start": maze["start"],
        "max_steps": maze["max_steps"],
    }


def save_dataset(dataset: MapDataset, path: Path) -> None:
    if path.suffix != ".jsonl":
        raise ValueError("map groups require a .jsonl file and same-name .toml")
    metadata = path.with_suffix(".toml")
    if path.exists() or metadata.exists():
        raise FileExistsError("will not overwrite an existing map group")
    records = [
        json.dumps({"split": split, **numeric_map(maze)}, ensure_ascii=False, allow_nan=False)
        for split, maps in dataset["splits"].items()
        for maze in maps
    ]
    content = ("\n".join(records) + "\n").encode("utf-8")
    local = dataset["lesson"] in ("local-map-v1", "memory-map-v1")
    strings = {
        "format": FORMAT,
        "id": path.stem,
        "title": dataset["title"],
        "description": dataset["description"],
        "created_at": dataset["created_at"],
        "lesson": dataset["lesson"],
        "map_format": "numeric-grid/v1",
        "generator": "fixed-map/v1" if dataset["lesson"] == "fixed-map-v1" else "lcg-corridor-7/v1",
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    lines = [f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in strings.items()]
    lines += [
        f"seed = {dataset['seed']}",
        "rows = 7",
        "columns = 7",
        "wall_value = -1",
        "road_value = 1",
        "goal_value = 0",
        f"max_steps = {200 if local else 100}",
        "",
        "[split_counts]",
    ]
    lines += [f"{key} = {len(maps)}" for key, maps in dataset["splits"].items()]
    with path.open("xb") as stream:
        stream.write(content)
    # The companion description is the completion marker; incomplete groups stay invisible.
    with metadata.open("x") as stream:
        stream.write("\n".join(lines) + "\n")


def mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("expected object")
    if not all(isinstance(key, str) for key in cast(dict[object, object], value)):
        raise ValueError("expected object")
    return cast(dict[str, object], value)


def sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("expected array")
    return cast(list[object], value)


def read_map(value: object, lesson: str) -> Maze:
    data = mapping(value)
    cells = sequence(data.get("cells"))
    if len(cells) != 7:
        raise ValueError("map must be 7 by 7")
    rows: list[str] = []
    for source in cells:
        row = sequence(source)
        if len(row) != 7 or any(type(cell) is not int or cell not in VALUES for cell in row):
            raise ValueError("map cells must be integers -1, 1 or 0")
        rows.append("".join(VALUES[cast(int, cell)] for cell in row))
    local = lesson in ("local-map-v1", "memory-map-v1")
    if data.get("max_steps") != (200 if local else 100) or type(data.get("max_steps")) is not int:
        raise ValueError("invalid action limit")
    if data.get("start") != {"x": 1, "y": 1, "direction": 0}:
        raise ValueError("invalid start")
    # bool equals int in Python, so check the component types as well.
    if any(type(item) is not int for item in mapping(data["start"]).values()):
        raise ValueError("invalid start types")
    name = data.get("id")
    if not isinstance(name, str) or len(name) > 100:
        raise ValueError("invalid map id")
    if rows[0] != "#######" or rows[-1] != "#######" or any(row[0] != "#" or row[-1] != "#" for row in rows):
        raise ValueError("open map border")
    if sum(row.count("G") for row in rows) != 1 or rows[1][1] != ".":
        raise ValueError("invalid goal or start cell")
    maze: Maze = {
        "id": name,
        "rows": rows,
        "start": {"x": 1, "y": 1, "direction": 0},
        "max_steps": 200 if local else 100,
    }
    if (1, 1) not in goal_distances(maze):
        raise ValueError("unreachable goal")
    if lesson == "fixed-map-v1":
        if maze != load_maze():
            raise ValueError("first lesson requires its fixed map")
    else:
        if rows[5][5] != "G" or not name.startswith("odymaze-random-"):
            raise ValueError("invalid generated-map identity")
        suffix = name.removeprefix("odymaze-random-")
        if not suffix.isascii() or not suffix.isdecimal() or not 0 <= int(suffix) <= 0xFFFFFFFF:
            raise ValueError("invalid map seed")
    return maze


def load_dataset(path: Path, lesson: str, *, expected_sha256: str | None = None) -> MapDataset:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("invalid dataset file or size")
    with path.open("rb") as stream:
        content = stream.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError("dataset too large")
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("job dataset checksum mismatch")
    metadata_path = path.with_suffix(".toml")
    if (
        path.suffix != ".jsonl"
        or metadata_path.is_symlink()
        or not metadata_path.is_file()
        or metadata_path.stat().st_size > 16384
    ):
        raise ValueError("missing or invalid companion TOML")
    data = mapping(tomllib.loads(metadata_path.read_text()))
    if lesson not in LESSONS or data.get("lesson") != lesson or data.get("format") != FORMAT:
        raise ValueError("dataset format or lesson mismatch")
    if data.get("sha256") != hashlib.sha256(content).hexdigest():
        raise ValueError("map group checksum mismatch")
    for key, expected in {
        "rows": 7,
        "columns": 7,
        "wall_value": -1,
        "road_value": 1,
        "goal_value": 0,
        "max_steps": 200 if lesson in ("local-map-v1", "memory-map-v1") else 100,
    }.items():
        if type(data.get(key)) is not int or data[key] != expected:
            raise ValueError("invalid map specification")
    if data.get("map_format") != "numeric-grid/v1":
        raise ValueError("unsupported numeric grid format")
    expected_generator = "fixed-map/v1" if lesson == "fixed-map-v1" else "lcg-corridor-7/v1"
    if data.get("generator") != expected_generator:
        raise ValueError("unsupported map generator")
    title, description, created_at = data.get("title"), data.get("description"), data.get("created_at")
    if (
        not isinstance(title, str)
        or not title.strip()
        or len(title) > 80
        or not isinstance(description, str)
        or len(description) > 500
        or not isinstance(created_at, str)
    ):
        raise ValueError("invalid group description")
    if datetime.fromisoformat(created_at).tzinfo is None:
        raise ValueError("group date requires timezone")
    seed = data.get("seed")
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("invalid dataset seed")
    counts = mapping(data.get("split_counts"))
    raw: dict[str, list[object]] = {"train": [], "validation": [], "test": []}
    if set(counts) != set(raw):
        raise ValueError("invalid dataset splits")
    for line in content.splitlines():
        if not line.strip():
            continue
        item = mapping(json.loads(line))
        split = item.get("split")
        if not isinstance(split, str) or split not in raw:
            raise ValueError("unknown map split")
        raw[split].append(item)
    if any(type(counts[key]) is not int or counts[key] != len(items) for key, items in raw.items()):
        raise ValueError("map count differs from TOML specification")
    splits: dict[str, list[Maze]] = {}
    seen: set[tuple[str, ...]] = set()
    identities: set[str] = set()
    for key in ("train", "validation", "test"):
        items = sequence(raw[key])
        valid_count = (
            len(items) == (1 if key == "train" else 0)
            if lesson == "fixed-map-v1"
            else (16 <= len(items) <= 2048 if key == "train" else len(items) == (64 if key == "validation" else 100))
        )
        if not valid_count:
            raise ValueError("invalid split size")
        maps = [read_map(item, lesson) for item in items]
        for maze in maps:
            rows = tuple(maze["rows"])
            if rows in seen or maze["id"] in identities:
                raise ValueError("duplicate map across dataset")
            seen.add(rows)
            identities.add(maze["id"])
        splits[key] = maps
    return {
        "lesson": lesson,
        "seed": seed,
        "title": title,
        "description": description,
        "created_at": created_at,
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="预生成 AIOps 数值地图集，不训练模型")
    parser.add_argument("--lesson", choices=LESSONS, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--maps", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--description", default="")
    args = parser.parse_args()
    try:
        value = prepare_dataset(args.lesson, args.seed, args.maps, title=args.title, description=args.description)
        save_dataset(value, args.output)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    print("地图集已保存；训练将直接读取这些地图，不重新生成。", flush=True)


if __name__ == "__main__":
    main()
