"""User task intent and independent turning/returning/slope outcomes."""

import math
from collections.abc import Sequence

from .configuration import VisionConfig
from .evaluation import Crossing, Outcome


def angle_error(target: float, current: float) -> float:
    return math.atan2(math.sin(target - current), math.cos(target - current))


def yaw(quaternion: Sequence[float]) -> float:
    if len(quaternion) != 4 or not all(math.isfinite(q) for q in quaternion):
        raise ValueError("invalid quaternion")
    w, x, y, z = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Goal:
    def __init__(self, config: VisionConfig, end_x: float):
        self.config = config
        self.stage = 0
        self.beam_maneuver = False
        self.stable = 0
        self.result: Outcome = "running"
        self.crossing = Crossing(end_x + config.success_clearance_m, config.corridor, config.stable_steps)

    def command(self, heading: float) -> tuple[float, float, float]:
        if self.config.goal_kind == "flat":
            return self.config.forward_m_s, 0.0, self.config.yaw_rate_rad_s
        target = (
            math.radians(self.config.turn_angle_deg)
            if self.stage and self.config.goal_kind in ("turn", "uturn")
            else 0.0
        )
        error = angle_error(target, heading)
        return (0.3 if abs(error) < 0.3 else 0.0, 0.0, max(-1.0, min(1.0, error * 2)))

    def progress(self, x: float, y: float) -> float:
        if not self.stage or self.config.goal_kind in ("crossing", "slope"):
            return x
        distance = self.config.target_distance_m
        if self.config.goal_kind == "uturn":
            return 2 * distance - x
        angle = math.radians(self.config.turn_angle_deg)
        return distance + (x - distance) * math.cos(angle) + y * math.sin(angle)

    def observe(
        self,
        x: float,
        y: float,
        height: float,
        upright: float,
        body_collision: bool,
        timeout: bool = False,
        rear_x: float | None = None,
        heading: float = 0.0,
        airborne: bool = False,
    ) -> Outcome:
        if self.result != "running":
            return self.result
        if not math.isfinite(heading):
            raise ValueError("invalid heading")
        if self.config.goal_kind == "flat":
            self.result = "observed" if timeout else "running"
            return self.result
        if self.config.goal_kind in ("duck", "jump"):
            if abs(x - self.config.target_distance_m) <= 0.08 and not body_collision:
                if self.config.goal_kind == "duck":
                    self.beam_maneuver |= height <= self.config.beam_height_m - 0.045
                else:
                    self.beam_maneuver |= airborne
            self.result = self.crossing.observe(
                x,
                y,
                height,
                upright,
                body_collision,
                timeout,
                rear_x,
                eligible=self.beam_maneuver and not airborne,
            )
            return self.result
        if self.config.goal_kind in ("crossing", "slope"):
            self.result = self.crossing.observe(x, y, height, upright, body_collision, timeout, rear_x)
            return self.result
        # First check state validity and physical failures, without claiming crossing success.
        guard = Crossing(float("inf"), 1e6, 1)
        status = guard.observe(x, y, height, upright, body_collision)
        if status != "running":
            self.result = status
            return status
        distance = self.config.target_distance_m
        if not self.stage and x >= distance and abs(angle_error(0.0, heading)) < math.radians(15):
            self.stage = 1
        lateral = abs(y)
        if self.stage and self.config.goal_kind == "turn":
            angle = math.radians(self.config.turn_angle_deg)
            lateral = abs(-(x - distance) * math.sin(angle) + y * math.cos(angle))
        if lateral > self.config.road_width_m / 2 - 0.15:
            self.result = "bypass"
            return self.result
        target = math.radians(self.config.turn_angle_deg)
        aligned = abs(angle_error(target, heading)) < math.radians(self.config.success_heading_tolerance_deg)
        arrived = (
            (abs(x) < self.config.success_return_tolerance_m and abs(y) < self.config.success_return_tolerance_m)
            if self.config.goal_kind == "uturn"
            else self.progress(x, y) > distance + self.config.success_exit_distance_m
        )
        self.stable = self.stable + 1 if self.stage and arrived and aligned else 0
        if self.stable >= self.config.stable_steps:
            self.result = "success"
        elif timeout:
            self.result = "timeout"
        return self.result
