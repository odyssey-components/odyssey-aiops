"""Bounded immutable simulation inputs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..vision.configuration import VisionConfig
from ..vision.criteria import criteria_config


class SceneInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    kind: Literal["turn", "uturn", "slope", "duck", "jump", "flat"]
    parameters: dict[str, float]
    criteria: dict[str, float] | None = None


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    seed: int = Field(default=7, ge=0, le=2147483647, strict=True)
    steps: int = Field(default=500, ge=50, le=500, strict=True)
    scene: SceneInput

    def environment(self, base: VisionConfig | None = None) -> VisionConfig:
        p = self.scene.parameters
        fields: dict[str, object] = {"goal_kind": self.scene.kind, "episode_steps": self.steps}
        if self.scene.kind == "flat":
            if set(p) != {"forward_m_s", "yaw_rate_rad_s"} or self.scene.criteria:
                raise ValueError("flat diagnostic parameters differ")
            return VisionConfig.model_validate({**(base or VisionConfig()).model_dump(), **fields, **p})
        if self.scene.kind == "turn":
            expected = {"angle_deg", "road_width_m", "distance_m"}
            fields.update(
                turn_angle_deg=p.get("angle_deg"),
                road_width_m=p.get("road_width_m"),
                target_distance_m=p.get("distance_m"),
            )
        elif self.scene.kind == "uturn":
            expected = {"road_width_m", "turn_at_m"}
            fields.update(
                turn_angle_deg=180.0, road_width_m=p.get("road_width_m"), target_distance_m=p.get("turn_at_m")
            )
        elif self.scene.kind in ("duck", "jump"):
            expected = {"distance_m", "height_m"}
            fields.update(target_distance_m=p.get("distance_m"), beam_height_m=p.get("height_m"))
        else:
            expected = {"angle_deg", "length_m"}
            fields.update(slope_angle_deg=p.get("angle_deg"), target_distance_m=p.get("length_m"))
        if set(p) != expected:
            raise ValueError("simulation scene parameters differ")
        fields.update(criteria_config(self.scene.kind, p, self.scene.criteria))
        return VisionConfig.model_validate({**(base or VisionConfig()).model_dump(), **fields})
