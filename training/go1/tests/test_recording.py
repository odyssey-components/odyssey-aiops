from pathlib import Path

import numpy as np
import pytest
from aiops_go1_training.simulation.recording import FloatArray, compare, read_injected_depth, record


def test_mcap_roundtrip_replays_recorded_inputs_and_compares_candidate_actions(tmp_path: Path) -> None:
    path = tmp_path / "policy-input.mcap"
    camera = {"width": 16, "height": 16, "near": 0.1, "far": 3.0, "camera_hz": 10}
    observation = np.zeros(45, dtype=np.float32)
    depth = np.linspace(0.1, 1, 256, dtype=np.float32).reshape(16, 16, 1)
    baseline = np.arange(12, dtype=np.float32)
    with record(path, {"source_sha256": "fixture", "camera": camera}) as append:
        for frame in range(2):
            observation[-1] = frame
            append(frame, observation, observation, depth, baseline, baseline.clip(0, 1))
    expected_frames = iter((0, 1))

    def candidate(body: FloatArray, image: FloatArray | None) -> FloatArray:
        step = next(expected_frames)
        assert body[-1] == step
        np.testing.assert_array_equal(image, depth)
        return baseline + (step + 1) / 8

    result = compare(path, candidate, camera, "fixture")
    assert result["worst_step"] == 2
    assert result["max_abs_delta"] == 0.25
    with pytest.raises(ValueError, match="observation contracts differ"):
        compare(path, candidate, {**camera, "far": 4.0}, "fixture")

@pytest.mark.parametrize("tagged", [False, True])
def test_parkour_injection_accepts_existing_encoder_frames(tmp_path: Path, tagged: bool) -> None:
    path = tmp_path / "parkour.mcap"
    camera = {"width": 64, "height": 48, "near": 0.02, "far": 2.0, "camera_hz": 10}
    metadata: dict[str, object] = {
        "source_sha256": "fixture",
        "camera": camera,
        "action_space": "joint-position-offset",
    }
    if tagged:
        metadata["depth_stage"] = "parkour-encoder/v1"
    image = np.linspace(0.02, 2.0, 48 * 64, dtype=np.float32).reshape(48, 64, 1)
    action = np.zeros(12, dtype=np.float32)
    with record(path, metadata, network_dimension=176) as append:
        append(0, np.zeros(45, dtype=np.float32), np.zeros(176, dtype=np.float32), image, action, action)
    frames = read_injected_depth(path, camera, "fixture", parkour=True)
    np.testing.assert_array_equal(frames[0], image)
    if tagged:
        with pytest.raises(ValueError, match="depth stage"):
            read_injected_depth(path, camera, "fixture")
