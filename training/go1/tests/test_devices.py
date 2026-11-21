"""Explicit backend selection never silently falls back to another device."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aiops_go1_training.runtime.devices import select_device


def test_cpu_selection_sets_cpu_backend() -> None:
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("jax.config.update"),
        patch("jax.devices", return_value=[SimpleNamespace(platform="cpu")]) as devices,
    ):
        assert select_device("cpu").platform == "cpu"
        assert os.environ["JAX_PLATFORMS"] == "cpu"
        devices.assert_called_once_with("cpu")


@pytest.mark.parametrize("mode", ["cuda", "rocm"])
def test_gpu_selection_rejects_cpu_fallback(mode: str) -> None:
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("jax.config.update"),
        patch("jax.devices", return_value=[SimpleNamespace(platform="cpu")]),
    ):
        with pytest.raises(RuntimeError):
            select_device(mode)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        select_device("arbitrary")
