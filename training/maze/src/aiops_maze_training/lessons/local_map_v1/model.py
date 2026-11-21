"""Third course network: observed memory enters, three action scores leave."""

from typing import TypedDict

import torch
from torch import nn

from aiops_maze_training.lessons.fixed_map_v1.environment import initial_state, step
from aiops_maze_training.lessons.fixed_map_v1.model import export_model as export_fixed_model
from aiops_maze_training.common.types import Maze, Metadata, Policy
from aiops_maze_training.lessons.local_map_v1.environment import encode, observe


class Rollout(TypedDict):
    success: bool
    steps: int
    actions: list[int]


def make_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(249, 64), nn.Tanh(), nn.Linear(64, 3))


def rollout(model: nn.Sequential, maze: Maze) -> Rollout:
    state = initial_state(maze)
    memory = observe(maze, state)
    actions: list[int] = []
    with torch.no_grad():
        while state.status not in ("success", "timeout"):
            observation = torch.tensor(encode(memory, state), dtype=torch.float32)
            action = int(model(observation).argmax())
            actions.append(action)
            state = step(maze, state, action)
            memory = observe(maze, state, memory)
    return {"success": state.status == "success", "steps": state.steps, "actions": actions}


def export_model(model: nn.Sequential, maze: Maze, metadata: Metadata) -> Policy:
    policy = export_fixed_model(model, maze, metadata)
    policy.update(
        format="odymaze-policy/v3",
        lesson="local-map-v1",
        encoding="grid-known-walls-goal-visited-agent-direction/v1",
        generator="lcg-corridor-7/v1",
    )
    return policy
