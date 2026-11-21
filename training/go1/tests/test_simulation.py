"""Read-only simulation contracts and actual synthetic CPU physics."""

import subprocess
import sys

import numpy as np
import pytest
from aiops_go1_training.runtime.simulation import main
from aiops_go1_training.simulation.configuration import SimulationConfig
from aiops_go1_training.simulation.policy import locomotion_action
from numpy.typing import NDArray


def test_simulation_rejects_unsafe_inputs() -> None:
    for value in ({"steps": 501}, {"seed": True}, {"command": "x"}):
        with pytest.raises(ValueError):
            SimulationConfig.model_validate({"scene": {"kind": "turn", "parameters": {}}, **value})
    with pytest.raises(ValueError):
        SimulationConfig.model_validate({"scene": {"kind": "turn", "parameters": {}}}).environment()


def test_locomotion_policy_inference() -> None:
    params: dict[str, NDArray[np.float64]] = {
        "l1/weight": np.zeros((45, 64)),
        "l1/bias": np.zeros(64),
        "l2/weight": np.zeros((64, 64)),
        "l2/bias": np.zeros(64),
        "actor/weight": np.zeros((64, 12)),
        "actor/bias": np.arange(12, dtype=np.float64),
    }
    np.testing.assert_equal(locomotion_action(params, np.zeros(45, dtype=np.float32)), np.arange(12, dtype=np.float64))


def test_worker_accepts_simulation_target_but_checks_parent() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aiops_go1_training.runtime.worker",
            "--parent-pid",
            "2",
            "--",
            "aiops_go1_training.runtime.simulation",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "training parent unavailable" in result.stderr
    assert "unsupported worker module" not in result.stderr


def test_simulation_cli_rejects_unpinned_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simulation",
            "--source",
            "/synthetic/source",
            "--model",
            "/synthetic/model",
            "--config",
            "/synthetic/config",
            "--output",
            "/synthetic/output",
        ],
    )

    def wrong_source(*_args: object, **_kwargs: object) -> str:
        return "wrong-source"

    monkeypatch.setattr(subprocess, "check_output", wrong_source)
    with pytest.raises(ValueError, match="pinned"):
        main()
