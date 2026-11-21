"""Fixed Go1 local adapter: environment checks, bounded config, no Lab tasks."""

import argparse
import importlib.util
import json
import os
import subprocess
import tomllib
from pathlib import Path
from typing import TypedDict

from .devices import MODES, configure_device, select_device

COMMIT = "f4fb0d12ac68842c1567bf1de6f771ce8e6671b5"


class Environment(TypedDict):
    ready: bool
    errors: list[str]
    devices: list[str]
    source_commit: str


def environment(source: Path, device: str = "cpu") -> Environment:
    configure_device(device)
    errors: list[str] = []
    missing: list[str] = []
    for name in ("jax", "mujoco", "numpy", "mujoco.mjx"):
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except ModuleNotFoundError:
            missing.append(name)
    if missing:
        errors.append("缺少 Python 依赖：" + ", ".join(missing))
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
        if commit != COMMIT:
            errors.append("Go1 源码版本不匹配，需要 " + COMMIT)
        status = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--", "unitree_go1"], text=True, timeout=5
        )
        if status.strip() or not (source / "unitree_go1/scene_torque.xml").is_file():
            errors.append("Go1 模型文件缺失或存在修改")
    except OSError, subprocess.SubprocessError:
        errors.append("缺少固定版本 Go1 源码，请设置服务端 ODYSSEY_GO1_SOURCE")
    devices: list[str] = []
    if not missing:
        try:
            devices = [str(select_device(device))]
            if not devices:
                errors.append("未发现请求的训练设备")
        except Exception as error:  # noqa: BLE001 - optional native GPU backends fail closed with diagnostics.
            errors.append(f"JAX {device} 不可用：" + (str(error) or type(error).__name__)[:500])
    return {"ready": not errors, "errors": errors, "devices": devices, "source_commit": COMMIT}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=MODES, default=os.environ.get("AIOPS_GO1_DEVICE", "cpu"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-output", type=Path)
    args = parser.parse_args()
    if args.validate_output is not None:
        from ..locomotion.model_manifest import validate_policy

        validate_policy(args.validate_output / "policy.npz")
        print("validated")
        return
    if args.source is None:
        parser.error("--source required")
    configure_device(args.device)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    result = environment(args.source, args.device)
    if args.check:
        print(json.dumps(result, ensure_ascii=False))
        return
    if not result["ready"]:
        raise RuntimeError("; ".join(result["errors"]))
    if args.config is None or args.output is None:
        parser.error("--config and --output required")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite training output")
    from ..locomotion.train import load_training_config, run
    from .resource_monitor import ResourceMonitor

    parameters = tomllib.loads(args.config.read_text())["training"]
    config = load_training_config(
        args.config, overrides={f"training.{key}": value for key, value in parameters.items()}
    )
    args.output.mkdir(parents=True)
    with ResourceMonitor(args.output):
        run(args.source.resolve(), args.output.resolve(), config)


if __name__ == "__main__":
    main()
