"""Only the radius-one sensor reads layout; the teacher owns separate memory."""

from dataclasses import asdict

from aiops_maze_training.lessons.fixed_map_v1.environment import State, initial_state, step
from aiops_maze_training.common.types import Maze, Sample
from aiops_maze_training.lessons.local_map_v1.environment import expert_action, generate_maze, observe, split_maps


def encode(maze: Maze, state: State, previous_action: int = -1) -> list[float]:
    if previous_action not in (-1, 0, 1, 2):
        raise ValueError("previous_action must be none (-1), left (0), right (1), or forward (2)")
    cells = [
        maze["rows"][y][x] if 0 <= x < 7 and 0 <= y < 7 else "#"
        for y in range(state.y - 1, state.y + 2)
        for x in range(state.x - 1, state.x + 2)
    ]
    return (
        [float(cell == "#") for cell in cells]
        + [float(cell == "G") for cell in cells]
        + [float(index == state.x) for index in range(7)]
        + [float(index == state.y) for index in range(7)]
        + [float(index == state.direction) for index in range(4)]
        + [float(index == previous_action) for index in (-1, 0, 1, 2)]
    )


def samples(maze: Maze) -> list[Sample]:
    state = initial_state(maze)
    memory = observe(maze, state)
    previous_action = -1
    result: list[Sample] = []
    while state.status not in ("success", "timeout"):
        action = expert_action(memory, state)
        result.append(
            {
                "maze_id": maze["id"],
                "state": asdict(state),
                "observation": encode(maze, state, previous_action),
                "action": action,
                "previous_action": previous_action,
            }
        )
        state = step(maze, state, action)
        previous_action = action
        memory = observe(maze, state, memory)
    return result


__all__ = ["encode", "generate_maze", "samples", "split_maps"]
