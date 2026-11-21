"""Self-contained ONNX conversion of the four readable lesson policies."""

import hashlib
import io
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import onnx
import torch
from torch import nn

from aiops_maze_training.lessons.fixed_map_v1.environment import load_maze
from aiops_maze_training.lessons.memory_map_v1.model import MemoryModel
from aiops_maze_training.lessons.random_map_v1.environment import generate_maze
from aiops_maze_training.common.manifest import dump_toml, map_reference, resolve_policy
from aiops_maze_training.common.types import Policy

# Format, lesson, observation encoding, input width, hidden width.
CONTRACTS = (
    ("odymaze-policy/v1", "fixed-map-v1", "one-hot-xy-direction/v1", 18, 32),
    ("odymaze-policy/v2", "random-map-v1", "grid-walls-goal-agent-direction/v1", 151, 128),
    ("odymaze-policy/v3", "local-map-v1", "grid-known-walls-goal-visited-agent-direction/v1", 249, 64),
    ("odymaze-policy/v4", "memory-map-v1", "local-view-xy-direction-previous-action/v1", 40, 32),
)


class _Export(Protocol):
    def __call__(
        self,
        model: nn.Module,
        args: tuple[torch.Tensor, ...],
        output: io.BytesIO,
        /,
        *,
        input_names: list[str],
        output_names: list[str],
        dynamo: bool,
        opset_version: int,
        dynamic_axes: dict[str, dict[int, str]],
        external_data: bool,
    ) -> object: ...


class _RecurrentStep(nn.Module):
    def __init__(self, model: MemoryModel) -> None:
        super().__init__()
        self.model: MemoryModel = model

    def forward(self, observations: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, updated = self.model.gru(observations.unsqueeze(1), hidden.unsqueeze(0))
        return self.model.output(sequence[:, 0]), updated[0]


def _tensor(value: object, shape: tuple[int, ...]) -> torch.Tensor:
    def valid(item: object, dimensions: tuple[int, ...]) -> bool:
        if not dimensions:
            return (
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(item)
                and abs(item) <= 100
            )
        return (
            isinstance(item, list)
            and len(cast(list[object], item)) == dimensions[0]
            and all(valid(child, dimensions[1:]) for child in cast(list[object], item))
        )

    if not valid(value, shape):
        raise ValueError(f"invalid policy parameter; expected finite bounded values with shape {shape}")
    return torch.tensor(cast(list[float], value), dtype=torch.float32)


def _validate(policy: Policy) -> tuple[str, int, int]:
    contract = next((row for row in CONTRACTS if row[0] == policy.get("format")), None)
    if contract is None:
        raise ValueError("unsupported policy format")
    _, lesson, encoding, width, hidden = contract
    if policy.get("lesson", "fixed-map-v1") != lesson or policy.get("encoding") != encoding:
        raise ValueError("policy lesson/encoding does not match its format")
    if policy.get("actions") != ["left", "right", "forward"]:
        raise ValueError("policy actions must be left, right, forward")
    raw_metadata = cast(object, policy.get("metadata"))
    if not isinstance(raw_metadata, dict):
        raise TypeError("policy metadata must be an object")
    metadata = cast(dict[str, object], raw_metadata)
    if any(name not in metadata for name in ("seed", "epochs", "samples", "accuracy", "success")):
        raise ValueError("policy metadata is incomplete")
    for name in ("seed", "epochs", "samples"):
        value = metadata[name]
        maximum = 5000 if name == "epochs" else 2147483647
        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError(f"invalid metadata {name}")
    accuracy = metadata["accuracy"]
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(accuracy)
        or not 0 <= accuracy <= 1
    ):
        raise ValueError("invalid metadata accuracy")
    rate = metadata.get("test_success_rate", 0.0)
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("invalid metadata test_success_rate")
    if type(metadata["success"]) is not bool:
        raise ValueError("invalid metadata success")
    if lesson == "fixed-map-v1":
        expected = load_maze()
    else:
        if policy.get("generator") != "lcg-corridor-7/v1":
            raise ValueError("unsupported maze generator")
        maze = policy.get("maze")
        try:
            seed = int(maze["id"].removeprefix("odymaze-random-"))
            expected = generate_maze(seed)
            expected["max_steps"] = 100 if lesson == "random-map-v1" else 200
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ValueError("invalid generated maze") from error
    if policy.get("maze") != expected:
        raise ValueError("policy maze is incompatible with its lesson")
    try:
        json.dumps(policy, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("policy must contain finite JSON data") from error
    return lesson, width, hidden


def _restore(policy: Policy, width: int, hidden: int, recurrent: bool) -> nn.Module:
    layers = policy.get("layers", [])
    if len(layers) != (1 if recurrent else 2):
        raise ValueError("invalid number of policy layers")
    model: MemoryModel | nn.Sequential = (
        MemoryModel() if recurrent else nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 3))
    )
    with torch.no_grad():
        for index, layer in enumerate(layers):
            output_size, input_size = (3, hidden) if recurrent or index == 1 else (hidden, width)
            if layer.get("activation") != ("linear" if recurrent or index == 1 else "tanh"):
                raise ValueError("invalid layer activation")
            target = model.output if isinstance(model, MemoryModel) else cast(nn.Linear, model[index * 2])
            target.weight.copy_(_tensor(layer.get("weight"), (output_size, input_size)))
            target.bias.copy_(_tensor(layer.get("bias"), (output_size,)))
        if isinstance(model, MemoryModel):
            parameters = policy.get("gru")
            if not parameters or parameters.get("hidden_size") != 32:
                raise ValueError("invalid GRU parameters")
            for name, shape in (
                ("weight_ih", (96, 40)),
                ("weight_hh", (96, 32)),
                ("bias_ih", (96,)),
                ("bias_hh", (96,)),
            ):
                target = cast(torch.Tensor, getattr(model.gru, f"{name}_l0"))
                target.copy_(_tensor(parameters.get(name), shape))
    return (_RecurrentStep(model) if isinstance(model, MemoryModel) else model).eval()


def export_onnx(policy: Policy, output: Path) -> Path:
    """Write model + sibling manifest exclusively, without retraining or replacing files."""
    output = Path(output)
    manifest_path = output.with_suffix(".manifest.toml")
    if output == manifest_path:
        raise ValueError("model output must differ from its manifest")
    for path in (output, manifest_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to replace {path}")
    try:
        policy = resolve_policy(policy)
        lesson, width, hidden = _validate(policy)
        recurrent = lesson == "memory-map-v1"
        model = _restore(policy, width, hidden, recurrent)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("malformed policy data") from error
    inputs = ["observations", "hidden"] if recurrent else ["observations"]
    outputs = ["logits", "next_hidden"] if recurrent else ["logits"]
    args = (torch.zeros(1, width), torch.zeros(1, hidden)) if recurrent else (torch.zeros(1, width),)
    buffer = io.BytesIO()
    cast(_Export, torch.onnx.export)(
        model,
        args,
        buffer,
        input_names=inputs,
        output_names=outputs,
        dynamo=False,
        opset_version=17,
        dynamic_axes={name: {0: "batch"} for name in inputs + outputs},
        external_data=False,
    )
    data = buffer.getvalue()
    graph = onnx.load_model_from_string(data)
    cast(Callable[..., None], onnx.checker.check_model)(graph, full_check=True)
    if any(
        tensor.external_data or tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer
    ):
        raise ValueError("ONNX model must be self-contained")
    manifest: dict[str, object] = {
        "schema": "odymaze-onnx/v1",
        "filename": output.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "opset": 17,
        "policy_format": policy["format"],
        "lesson": lesson,
        "encoding": policy["encoding"],
        "actions": policy["actions"],
        "map_ref": map_reference(policy),
        "metadata": policy["metadata"],
        "outputs_are_logits": True,
        "inputs": [{"name": "observations", "dtype": "float32", "shape": ["batch", width]}],
        "outputs": [{"name": "logits", "dtype": "float32", "shape": ["batch", 3]}],
    }
    if "generator" in policy:
        manifest["generator"] = policy["generator"]
    if recurrent:
        cast(list[object], manifest["inputs"]).append({"name": "hidden", "dtype": "float32", "shape": ["batch", 32]})
        cast(list[object], manifest["outputs"]).append(
            {"name": "next_hidden", "dtype": "float32", "shape": ["batch", 32]}
        )
        manifest["recurrent"] = {"initial_hidden": [0.0] * 32, "reset": "per-maze", "update": "once-per-action"}
    if "dataset" in policy:
        manifest["dataset"] = cast(dict[str, object], policy)["dataset"]
    for tensor in cast(list[dict[str, object]], manifest["inputs"]) + cast(
        list[dict[str, object]], manifest["outputs"]
    ):
        shape = cast(list[object], tensor["shape"])
        tensor["shape"] = {"batch": "batch", "width": shape[1]}
    manifest_bytes = dump_toml(manifest).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        for path, content in ((output, data), (manifest_path, manifest_bytes)):
            with path.open("xb") as stream:
                created.append(path)
                stream.write(content)
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise
    return output
