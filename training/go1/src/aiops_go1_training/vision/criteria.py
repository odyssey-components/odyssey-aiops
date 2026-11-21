"""Scene criteria translated into the same physical evaluator for training and simulation."""

import math

DEFAULTS: dict[str, dict[str, float]] = {
    "turn": {"stable_seconds": 1.0, "exit_distance_m": 0.8, "heading_tolerance_deg": 15.0},
    "uturn": {"stable_seconds": 1.0, "return_tolerance_m": 0.15, "heading_tolerance_deg": 15.0},
    "duck": {"stable_seconds": 1.0, "clearance_m": 0.0},
    "jump": {"stable_seconds": 1.0, "clearance_m": 0.0},
    "slope": {"stable_seconds": 1.0, "clearance_m": 0.0},
}


def criteria_config(kind: str, parameters: dict[str, float], criteria: dict[str, float] | None) -> dict[str, object]:
    values = dict(DEFAULTS[kind]) if not criteria else dict(criteria)
    if set(values) != set(DEFAULTS[kind]) or not all(math.isfinite(v) for v in values.values()):
        raise ValueError("scene pass criteria differ")
    if not 0.02 <= values["stable_seconds"] <= 10:
        raise ValueError("stable duration must be .02..10 seconds")
    if kind in ("turn", "uturn") and not 1 <= values["heading_tolerance_deg"] <= 45:
        raise ValueError("heading tolerance must be 1..45 degrees")
    if kind == "turn" and values["exit_distance_m"] < 0.1:
        raise ValueError("exit distance must be at least .1m")
    if kind == "uturn" and not 0.01 <= values["return_tolerance_m"] <= parameters["turn_at_m"] / 2:
        raise ValueError("return tolerance must be .01m..half the outbound distance")
    if kind in ("slope", "duck", "jump") and values["clearance_m"] < 0:
        raise ValueError("clearance must be nonnegative")
    return {
        "stable_steps": math.ceil(values["stable_seconds"] * 50 - 1e-9),
        "success_exit_distance_m": values.get("exit_distance_m", 0.8),
        "success_return_tolerance_m": values.get("return_tolerance_m", 0.15),
        "success_heading_tolerance_deg": values.get("heading_tolerance_deg", 15.0),
        "success_clearance_m": values.get("clearance_m", 0.0),
    }
