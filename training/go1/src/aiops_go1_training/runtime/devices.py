"""Explicit JAX device selection for CPU, NVIDIA CUDA and AMD ROCm."""

import os
from collections.abc import Callable
from typing import cast

import jax

MODES = ("cpu", "cuda", "rocm")


def configure_device(mode: str | None = None) -> str:
    mode = mode or os.environ.get("AIOPS_GO1_DEVICE") or os.environ.get("JAX_PLATFORMS", "cpu").split(",")[0]
    if mode not in MODES:
        raise ValueError("device must be cpu, cuda or rocm")
    os.environ["AIOPS_GO1_DEVICE"] = mode
    os.environ["JAX_PLATFORMS"] = mode
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return mode


def select_device(mode: str | None = None) -> jax.Device:
    previous = {key: os.environ.get(key) for key in ("AIOPS_GO1_DEVICE", "JAX_PLATFORMS")}
    update_config = cast(Callable[[str, object], None], jax.config.update)
    previous_platforms = jax.config.jax_platforms
    mode = configure_device(mode)
    try:
        update_config("jax_platforms", mode)
        devices = jax.devices("cpu" if mode == "cpu" else "gpu")
        platform = "cpu" if mode == "cpu" else "gpu"
        if not devices or devices[0].platform != platform:
            raise RuntimeError("selected backend did not provide the requested device")
    except (RuntimeError, AssertionError, ValueError) as error:
        update_config("jax_platforms", previous_platforms)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        detail = str(error).split("(set JAX_PLATFORMS=", 1)[0].strip() or type(error).__name__
        suffix = "; no CPU fallback for an explicit GPU request" if mode != "cpu" else ""
        raise RuntimeError(f"{mode} device unavailable: {detail}{suffix}") from error
    return devices[0]
