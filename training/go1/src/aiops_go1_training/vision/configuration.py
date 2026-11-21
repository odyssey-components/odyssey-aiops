"""Versioned, bounded depth-camera obstacle task configuration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    format: Literal["go1-depth/v1", "go1-depth-residual/v2"] = "go1-depth/v1"
    base_policy_sha256: str = ""
    goal_kind: Literal["crossing", "turn", "uturn", "slope", "duck", "jump", "flat"] = "crossing"
    forward_m_s: float = Field(default=0.3, ge=-2, le=2)
    yaw_rate_rad_s: float = Field(default=0.0, ge=-1, le=1)
    turn_angle_deg: float = Field(default=90, ge=-180, le=180)
    road_width_m: float = Field(default=0.8, ge=0.6)
    slope_angle_deg: float = Field(default=8, ge=3, le=15)
    target_distance_m: float = Field(default=1.2, ge=0.8)
    beam_height_m: float = 0.28
    width: int = Field(default=64, ge=16, le=128, strict=True)
    height: int = Field(default=48, ge=16, le=128, strict=True)
    camera_hz: int = Field(default=10, ge=1, le=50, strict=True)
    control_hz: Literal[50] = 50
    near: float = Field(default=0.1, ge=0.02, le=0.5)
    far: float = Field(default=3.0, ge=1.0, le=10.0)
    scene_kind: Literal["block", "step", "mixed"] = "mixed"
    height_min: float = Field(default=0.02, ge=0.01, le=0.08)
    height_max: float = Field(default=0.08, ge=0.01, le=0.08)
    obstacle_x: float = Field(default=0.8, ge=0.6, le=2.0)
    corridor: float = Field(default=0.25, ge=0.15, le=0.3)
    episode_steps: int = Field(default=750, ge=50, le=1500, strict=True)
    success_exit_distance_m: float = Field(default=0.8, ge=0.1)
    success_return_tolerance_m: float = Field(default=0.15, ge=0.01)
    success_heading_tolerance_deg: float = Field(default=15.0, ge=1, le=45)
    success_clearance_m: float = Field(default=0.0, ge=0)
    stable_steps: int = Field(default=50, ge=1, le=500, strict=True)
    iterations: int = Field(default=20, ge=1, le=1000, strict=True)
    rollout_steps: int = Field(default=128, ge=8, le=512, strict=True)
    learning_rate: float = Field(default=0.0003, gt=0, le=0.01)
    train_seeds: tuple[int, ...] = Field(default=(7, 8), max_length=8)
    test_seeds: tuple[int, ...] = Field(default=(10001, 10002), max_length=32)

    @model_validator(mode="after")
    def validate_contract(self) -> VisionConfig:
        import re

        if (
            bool(self.base_policy_sha256) != (self.format == "go1-depth-residual/v2")
            or self.base_policy_sha256
            and not re.fullmatch("[0-9a-f]{64}", self.base_policy_sha256)
        ):
            raise ValueError("residual format requires a valid base policy digest")
        if self.goal_kind == "duck" and not 0.26 <= self.beam_height_m <= 0.30:
            raise ValueError("duck beam clearance must be 0.26..0.30m")
        if self.goal_kind == "jump" and not 0.16 <= self.beam_height_m <= 0.24:
            raise ValueError("jump beam height must be 0.16..0.24m")
        if self.goal_kind == "uturn" and self.success_return_tolerance_m > self.target_distance_m / 2:
            raise ValueError("return tolerance exceeds half outbound distance")
        if self.goal_kind == "turn" and not 45 <= abs(self.turn_angle_deg) <= 135:
            raise ValueError("turn angle must be 45..135 degrees in either direction")
        if self.goal_kind == "uturn" and abs(self.turn_angle_deg) != 180:
            raise ValueError("U-turn requires a 180 degree target")
        if self.height_min > self.height_max or self.near >= self.far or self.control_hz % self.camera_hz:
            raise ValueError("invalid geometry or camera/control period")
        if not self.train_seeds or not self.test_seeds or set(self.train_seeds) & set(self.test_seeds):
            raise ValueError("training and test seeds must be nonempty and disjoint")
        if any(type(seed) is not int or not 0 <= seed <= 2147483647 for seed in self.train_seeds + self.test_seeds):
            raise ValueError("invalid scene seed")
        return self
