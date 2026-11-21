"""Summarize an experimental Go1 evaluation without inventing a verdict."""

import math
from collections.abc import Iterable
from typing import SupportsFloat, TypedDict

import numpy as np


class EvaluationSummary(TypedDict):
    evaluation_seed: int
    duration_seconds: float
    seconds_before_first_fall: float
    episode_ends: int
    minimum_height_m: float
    forward_displacement_m: float
    note: str


def summarize(
    heights: Iterable[SupportsFloat],
    dones: Iterable[object],
    positions_x: Iterable[float | np.floating],
    period: float,
    seed: int,
) -> EvaluationSummary:
    heights = list(heights)
    dones = list(dones)
    positions_x = list(positions_x)
    if (
        not heights
        or len(heights) != len(dones)
        or len(heights) != len(positions_x)
        or not math.isfinite(period)
        or period <= 0
    ):
        raise ValueError("invalid evaluation samples")
    if not all(math.isfinite(float(v)) for v in heights + positions_x):
        raise ValueError("non-finite evaluation samples")
    first = next((i for i, v in enumerate(dones) if v), None)
    return {
        "evaluation_seed": seed,
        "duration_seconds": len(heights) * period,
        "seconds_before_first_fall": (first + 1) * period if first is not None else len(heights) * period,
        "episode_ends": sum(bool(v) for v in dones),
        "minimum_height_m": min(float(v) for v in heights),
        "forward_displacement_m": float(np.subtract(positions_x[first if first is not None else -1], positions_x[0])),
        "note": "one perturbed initial state; experimental and not a generalization guarantee",
    }
