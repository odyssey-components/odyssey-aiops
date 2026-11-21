"""Physics outcomes independent of policy rewards or rendered appearance."""

import math
from dataclasses import dataclass
from typing import Literal

Outcome = Literal["running", "success", "fallen", "collision", "bypass", "timeout", "observed"]


@dataclass
class Crossing:
    end_x: float
    corridor: float
    stable_steps: int
    consecutive: int = 0
    result: Outcome = "running"

    def observe(
        self,
        x: float,
        y: float,
        height: float,
        upright: float,
        body_collision: bool,
        timeout: bool = False,
        rear_x: float | None = None,
        eligible: bool = True,
    ) -> Outcome:
        if (
            not all(math.isfinite(value) for value in (x, y, height, upright))
            or rear_x is not None
            and not math.isfinite(rear_x)
        ):
            raise ValueError("nonfinite crossing state")
        if self.result != "running":
            return self.result
        if height < 0.19 or height > 0.75 or upright < 0.85:
            self.result = "fallen"
        elif body_collision:
            self.result = "collision"
        elif abs(y) > self.corridor:
            self.result = "bypass"
        else:
            # Root must clear the far edge by the robot's half length plus margin.
            self.consecutive = (
                self.consecutive + 1
                if eligible and x > self.end_x + 0.45 and (rear_x is None or rear_x > self.end_x)
                else 0
            )
            if self.consecutive >= self.stable_steps:
                self.result = "success"
            elif timeout:
                self.result = "timeout"
        return self.result
