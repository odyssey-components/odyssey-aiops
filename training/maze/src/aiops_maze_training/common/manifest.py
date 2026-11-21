"""TOML manifests and references to versioned maps, without embedded grids."""

import json
import math
from typing import cast

from aiops_maze_training.lessons.fixed_map_v1.environment import load_maze
from aiops_maze_training.lessons.random_map_v1.environment import generate_maze
from aiops_maze_training.common.types import Policy


def dump_toml(values: dict[str, object]) -> str:
    def encode(value: object) -> str:
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)) and math.isfinite(value):
            return repr(value)
        if isinstance(value, list):
            return "[" + ", ".join(encode(item) for item in cast(list[object], value)) + "]"
        if isinstance(value, dict):
            return (
                "{ "
                + ", ".join(
                    f"{json.dumps(key)} = {encode(item)}" for key, item in cast(dict[str, object], value).items()
                )
                + " }"
            )
        raise ValueError("manifest requires finite TOML values")

    return "\n".join(f"{json.dumps(key)} = {encode(value)}" for key, value in values.items()) + "\n"


def map_reference(policy: Policy) -> dict[str, str]:
    return {"id": policy["maze"]["id"]}


def resolve_policy(policy: Policy) -> Policy:
    data = cast(dict[str, object], policy)
    if "maze" in data or "map_ref" not in data:
        return policy
    reference = cast(dict[str, str], data["map_ref"])
    identity = reference["id"]
    if policy.get("lesson", "fixed-map-v1") == "fixed-map-v1":
        maze = load_maze()
    else:
        prefix = "odymaze-random-"
        if (
            not identity.startswith(prefix)
            or not identity[len(prefix) :].isascii()
            or not identity[len(prefix) :].isdecimal()
        ):
            raise ValueError("invalid map reference")
        seed = int(identity[len(prefix) :])
        if not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("invalid map reference seed")
        maze = generate_maze(seed)
        maze["max_steps"] = 100 if policy.get("lesson") == "random-map-v1" else 200
    if maze["id"] != identity:
        raise ValueError("map reference does not match lesson")
    return cast(Policy, {**data, "maze": maze})
