"""Route command calibration for the public Go1 walking policy."""

import math

from ..vision.goals import Goal, angle_error


class WalkTurn:
    """Lead the turn before the centerline intersection, then track the exit."""

    def __init__(self, goal: Goal) -> None:
        self.goal = goal
        self.angle = math.radians(goal.config.turn_angle_deg)
        self.corner = goal.config.target_distance_m

    def command(self, x: float, y: float, heading: float) -> tuple[float, float, float]:
        if not self.goal.stage:
            return 1.3, 0.0, max(-1.0, min(1.0, -3 * heading))
        dx = x - self.corner
        across = -dx * math.sin(self.angle) + y * math.cos(self.angle)
        along = dx * math.cos(self.angle) + y * math.sin(self.angle)
        target = self.angle - math.atan2(across, 0.45)
        if along < 0.08 and dx < -0.1:
            target = math.copysign(min(abs(target), abs(self.angle) * 0.7), self.angle)
        turn = max(-1.7, min(1.7, 4 * angle_error(target, heading)))
        return (0.3 if along < 0.2 else 1.3), 0.0, turn

    def enter_corner(self, x: float, heading: float) -> None:
        # Preserve Goal's exit-distance, heading, collision and stability criteria.
        lead = min(0.18, self.goal.config.road_width_m / 2 - 0.16)
        if not self.goal.stage and lead > 0 and x >= self.corner - lead and abs(heading) < math.radians(15):
            self.goal.stage = 1
