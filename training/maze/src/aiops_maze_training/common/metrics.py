"""Machine-readable measurements from actual policy rollouts."""

import json
from typing import Literal, NotRequired, TypedDict


class Metric(TypedDict):
    epoch: int
    total: int
    successRate: float
    evaluatedMaps: int
    split: Literal["fixed", "validation"]
    loss: NotRequired[float]
    accuracy: NotRequired[float]


def emit_metric(
    epoch: int,
    total: int,
    success_rate: float,
    evaluated_maps: int,
    split: Literal["fixed", "validation"],
    *,
    loss: float | None = None,
    accuracy: float | None = None,
) -> None:
    metric: Metric = {
        "epoch": epoch,
        "total": total,
        "successRate": success_rate,
        "evaluatedMaps": evaluated_maps,
        "split": split,
    }
    if loss is not None:
        metric["loss"] = loss
    if accuracy is not None:
        metric["accuracy"] = accuracy
    print("ODYMAZE_METRIC " + json.dumps(metric, allow_nan=False, separators=(",", ":")), flush=True)
