"""Encode recorded evaluation states for browser replay without simulation."""

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, SupportsFloat, TypedDict, TypeVar

_Item_co = TypeVar("_Item_co", covariant=True)


class _Samples(Iterable[_Item_co], Protocol):
    def __len__(self) -> int: ...


class _Frame(TypedDict):
    time: float
    position_m: list[float]
    orientation_wxyz: list[float]
    joint_position_rad: list[float]
    episode_end: bool


def write_replay(path: Path, qpos: _Samples[Iterable[SupportsFloat]], dones: _Samples[object], period: float) -> None:
    if not math.isfinite(period) or period <= 0 or len(qpos) != len(dones) or not 1 <= len(qpos) <= 500:
        raise ValueError("invalid evaluation replay")
    frames: list[_Frame] = []
    for index, (position, done) in enumerate(zip(qpos, dones)):
        values = [float(x) for x in position]
        if len(values) != 19 or not all(math.isfinite(x) for x in values):
            raise ValueError("invalid evaluation position")
        if not 0.99 <= math.sqrt(sum(x * x for x in values[3:7])) <= 1.01:
            raise ValueError("invalid evaluation orientation")
        frames.append(
            {
                "time": (index + 1) * period,
                "position_m": values[:3],
                "orientation_wxyz": values[3:7],
                "joint_position_rad": values[7:],
                "episode_end": bool(done),
            }
        )
    path.write_text(
        json.dumps({"kind": "go1-evaluation-replay", "frames": frames}, allow_nan=False, separators=(",", ":")) + "\n"
    )
