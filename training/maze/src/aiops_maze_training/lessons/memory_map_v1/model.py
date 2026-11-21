"""A 40-input GRU, reset per episode, predicts three action logits."""

import torch
from torch import nn

from aiops_maze_training.lessons.fixed_map_v1.environment import initial_state, step
from aiops_maze_training.common.types import Maze, Metadata, Policy
from aiops_maze_training.lessons.local_map_v1.model import Rollout
from aiops_maze_training.lessons.memory_map_v1.environment import encode


class MemoryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gru: nn.GRU = nn.GRU(40, 32, batch_first=True)
        self.output: nn.Linear = nn.Linear(32, 3)

    def forward(self, observations: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        sequence, _ = self.gru(observations, hidden)
        return self.output(sequence)

    def step(self, observation: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, updated = self.gru(observation.reshape(1, 1, 40), hidden)
        return self.output(sequence)[0, 0], updated


def make_model() -> MemoryModel:
    return MemoryModel()


def rollout(model: MemoryModel, maze: Maze, reset_hidden: bool = False) -> Rollout:
    state = initial_state(maze)
    hidden: torch.Tensor | None = None
    previous_action = -1
    actions: list[int] = []
    with torch.no_grad():
        while state.status not in ("success", "timeout"):
            observation = torch.tensor(encode(maze, state, previous_action), dtype=torch.float32)
            logits, hidden = model.step(observation, None if reset_hidden else hidden)
            action = int(logits.argmax())
            actions.append(action)
            state = step(maze, state, action)
            previous_action = action
    return {"success": state.status == "success", "steps": state.steps, "actions": actions}


def export_model(model: MemoryModel, maze: Maze, metadata: Metadata) -> Policy:
    parameters = model.state_dict()
    return {
        "format": "odymaze-policy/v4",
        "lesson": "memory-map-v1",
        "generator": "lcg-corridor-7/v1",
        "maze": maze,
        "encoding": "local-view-xy-direction-previous-action/v1",
        "actions": ["left", "right", "forward"],
        "gru": {
            "hidden_size": 32,
            "weight_ih": [[float(value) for value in row] for row in parameters["gru.weight_ih_l0"]],
            "weight_hh": [[float(value) for value in row] for row in parameters["gru.weight_hh_l0"]],
            "bias_ih": [float(value) for value in parameters["gru.bias_ih_l0"]],
            "bias_hh": [float(value) for value in parameters["gru.bias_hh_l0"]],
        },
        "layers": [
            {
                "activation": "linear",
                "weight": [[float(value) for value in row] for row in model.output.weight.detach()],
                "bias": [float(value) for value in model.output.bias.detach()],
            }
        ],
        "metadata": metadata,
    }
