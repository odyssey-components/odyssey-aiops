"""Lesson 3: a tiny network, an honest rollout, and plain-data export."""

from typing import cast

import torch
from torch import nn

from aiops_maze_training.common.types import Evaluation, Layer, Maze, Metadata, Policy
from aiops_maze_training.lessons.fixed_map_v1.environment import encode, initial_state, step


def make_model(input_size: int) -> nn.Sequential:
    # One hidden layer. No memory, search, reward function, or expert at inference.
    return nn.Sequential(nn.Linear(input_size, 32), nn.Tanh(), nn.Linear(32, 3))


def evaluate(model: nn.Sequential, maze: Maze, features: torch.Tensor, labels: torch.Tensor) -> Evaluation:
    """Dataset accuracy and game success answer two different questions."""
    with torch.no_grad():
        accuracy = float((model(features).argmax(dim=1) == labels).float().mean())
        state = initial_state(maze)
        actions: list[int] = []
        while state.status not in ("success", "timeout"):
            observation = torch.tensor(encode(maze, state), dtype=torch.float32)
            action = int(model(observation).argmax())
            actions.append(action)
            state = step(maze, state, action)
    return {"accuracy": accuracy, "success": state.status == "success", "steps": state.steps, "actions": actions}


def export_model(model: nn.Sequential, maze: Maze, metadata: Metadata) -> Policy:
    layers: list[Layer] = []
    for index, activation in ((0, "tanh"), (2, "linear")):
        layer = model[index]
        if not isinstance(layer, nn.Linear):
            raise TypeError("the maze model must contain two linear layers")
        layers.append(
            {
                "activation": activation,
                # PyTorch's tolist return type does not encode tensor rank.
                "weight": cast(list[list[float]], layer.weight.detach().tolist()),  # pyright: ignore[reportUnknownMemberType]
                "bias": cast(list[float], layer.bias.detach().tolist()),  # pyright: ignore[reportUnknownMemberType]
            }
        )
    return {
        "format": "odymaze-policy/v1",
        "lesson": "fixed-map-v1",
        "maze": maze,
        "encoding": "one-hot-xy-direction/v1",
        "actions": ["left", "right", "forward"],
        "layers": layers,
        "metadata": metadata,
    }
