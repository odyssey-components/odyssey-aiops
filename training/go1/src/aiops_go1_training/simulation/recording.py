"""Versioned MCAP input and baseline action stream for open-loop comparisons."""

import base64
import json
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
from mcap.reader import SeekingReader
from mcap.writer import Writer
from numpy.typing import NDArray

from ..vision.configuration import VisionConfig
from .policy import Action

FORMAT = "go1-policy-input/v1"
TOPIC = "/go1/policy_step"
PERIOD_NS = 20_000_000
MAX_STEPS = 500
MAX_BYTES = 48 << 20


FloatArray = NDArray[np.float32]
AppendStep = Callable[[int, FloatArray, FloatArray, FloatArray | None, FloatArray, FloatArray], None]


class ComparisonStep(TypedDict):
    step: int
    time_seconds: float
    baseline_action: list[float]
    baseline_applied: list[float]
    candidate_action: list[float]
    action_delta: list[float]
    max_abs_delta: float
    candidate_state: NotRequired[list[float]]
    has_depth: NotRequired[bool]


def camera_contract(config: VisionConfig | None) -> dict[str, int | float]:
    if config is None:
        return {}
    return {key: getattr(config, key) for key in ("width", "height", "near", "far", "camera_hz")}


@contextmanager
def record(path: Path, metadata: dict[str, object], *, network_dimension: int = 45) -> Generator[AppendStep]:
    temporary = path.with_suffix(".partial")
    try:
        with temporary.open("xb") as stream:
            writer = Writer(stream, enable_data_crcs=True)
            writer.start(profile=FORMAT)
            schema = writer.register_schema(
                name=FORMAT,
                encoding="jsonschema",
                data=json.dumps(
                    {
                        "type": "object",
                        "properties": {
                            "depth_f32le_base64": {"type": ["string", "null"]},
                            "body": {"type": "array", "items": {"type": "number"}, "minItems": 45, "maxItems": 45},
                            "network_state": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": network_dimension,
                                "maxItems": network_dimension,
                            },
                            "action": {"type": "array", "items": {"type": "number"}, "minItems": 12, "maxItems": 12},
                            "applied": {"type": "array", "items": {"type": "number"}, "minItems": 12, "maxItems": 12},
                        },
                        "required": ["depth_f32le_base64", "body", "network_state", "action", "applied"],
                    }
                ).encode(),
            )
            channel = writer.register_channel(topic=TOPIC, message_encoding="json", schema_id=schema)
            writer.add_metadata(FORMAT, {"contract": json.dumps(metadata, allow_nan=False)})

            def append(
                step: int,
                body: FloatArray,
                network_state: FloatArray,
                depth: FloatArray | None,
                action: FloatArray,
                applied: FloatArray,
            ) -> None:
                timestamp = step * PERIOD_NS
                encoded = (
                    None if depth is None else base64.b64encode(depth.astype("<f4", copy=False).tobytes()).decode()
                )
                payload = {
                    "depth_f32le_base64": encoded,
                    "body": body.tolist(),
                    "network_state": network_state.tolist(),
                    "action": action.tolist(),
                    "applied": applied.tolist(),
                }
                writer.add_message(channel, timestamp, json.dumps(payload, allow_nan=False).encode(), timestamp, step)

            yield append
            writer.finish()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def vector(value: object, count: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError("invalid Go1 recording vector")
    return result


@contextmanager
def policy_steps(
    path: Path,
    camera: dict[str, int | float],
    source_sha256: str,
) -> Generator[tuple[dict[str, object], Generator[tuple[int, dict[str, object], FloatArray | None]]]]:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Go1 recording exceeds size limit")
    with path.open("rb") as stream:
        reader = SeekingReader(stream, validate_crcs=True, record_size_limit=MAX_BYTES)
        contracts = [item for item in reader.iter_metadata() if item.name == FORMAT]
        if len(contracts) != 1:
            raise ValueError("missing or duplicate Go1 recording contract")
        metadata = json.loads(contracts[0].metadata["contract"])
        recorded_camera = metadata["camera"]
        if metadata["source_sha256"] != source_sha256:
            raise ValueError("model and recording use different robot sources")
        if camera and not recorded_camera:
            raise ValueError("recording has no depth frames; run a new simulation for visual regression")
        if camera and recorded_camera != camera:
            raise ValueError(f"camera observation contracts differ: recording={recorded_camera}; model={camera}")

        def frames() -> Generator[tuple[int, dict[str, object], FloatArray | None]]:
            for step, (schema, channel, message) in enumerate(reader.iter_messages(TOPIC, log_time_order=False)):
                if (
                    step >= MAX_STEPS
                    or schema is None
                    or schema.name != FORMAT
                    or channel.message_encoding != "json"
                    or message.sequence != step
                    or message.log_time != step * PERIOD_NS
                ):
                    raise ValueError("invalid Go1 recording step")
                payload = json.loads(message.data)
                depth = None
                encoded = payload["depth_f32le_base64"]
                if camera:
                    if not isinstance(encoded, str):
                        raise ValueError("missing recorded depth")
                    depth = np.frombuffer(base64.b64decode(encoded, validate=True), dtype="<f4").reshape(
                        int(camera["height"]), int(camera["width"]), 1
                    )
                    if not np.isfinite(depth).all():
                        raise ValueError("nonfinite recorded depth")
                elif not recorded_camera and encoded is not None:
                    raise ValueError("unexpected recorded depth")
                yield step, payload, depth

        yield metadata, frames()


def compare(
    path: Path,
    policy: Action,
    camera: dict[str, int | float],
    source_sha256: str,
    *,
    capture: Callable[[int, str | None], list[float]] | None = None,
) -> dict[str, object]:
    rows: list[ComparisonStep] = []
    with policy_steps(path, camera, source_sha256) as (metadata, frames):
        if metadata.get("action_space", "motor-control") != "motor-control":
            raise ValueError("recording action space does not match the torque policy")
        for step, payload, depth in frames:
            vector(payload["network_state"], 45)
            baseline = vector(payload["action"], 12)
            applied = vector(payload["applied"], 12)
            candidate = vector(policy(vector(payload["body"], 45), depth), 12)
            delta = candidate.astype(np.float64) - baseline
            row: ComparisonStep = {
                "step": step + 1,
                "time_seconds": step / 50,
                "baseline_action": baseline.tolist(),
                "baseline_applied": applied.tolist(),
                "candidate_action": candidate.tolist(),
                "action_delta": delta.tolist(),
                "max_abs_delta": float(np.max(np.abs(delta))),
            }
            if capture is not None:
                encoded = payload["depth_f32le_base64"]
                row["candidate_state"] = capture(step + 1, encoded if camera and isinstance(encoded, str) else None)
                row["has_depth"] = bool(camera)
            rows.append(row)
    if not rows:
        raise ValueError("Go1 recording has no policy steps")
    worst = max(rows, key=lambda row: row["max_abs_delta"])
    return {
        "format": "go1-regression/v1",
        "baseline": metadata,
        "steps": rows,
        "max_abs_delta": worst["max_abs_delta"],
        "worst_step": worst["step"],
    }


def read_injected_depth(
    path: Path, camera: dict[str, int | float], source_sha256: str, *, parkour: bool = False
) -> list[FloatArray]:
    """Validate the recorded depth contract and order before driving physics."""
    if not camera:
        raise ValueError("visual injection requires a depth policy")
    frames: list[FloatArray] = []
    with policy_steps(path, camera, source_sha256) as (metadata, steps):
        # Legacy Parkour MCAPs predate the stage field but record the same encoder input and 176-value state.
        stage = metadata.get("depth_stage")
        if (parkour and stage not in (None, "parkour-encoder/v1")) or (not parkour and stage is not None):
            raise ValueError("recorded depth stage is incompatible with the selected model")
        if parkour and metadata.get("action_space") != "joint-position-offset":
            raise ValueError("Parkour visual injection requires a Parkour recording")
        for _, payload, depth in steps:
            if parkour:
                vector(payload["network_state"], 176)
            if depth is None:
                raise ValueError("missing injected depth observation")
            frames.append(depth)
    if not frames:
        raise ValueError("Go1 recording has no policy steps")
    return frames
