"""A radius-one Chebyshev sensor reveals 3×3 cells without wall occlusion.

Only the sensor reads the true layout. The expert and encoder consume memory.
"""

from collections import deque
from dataclasses import asdict

from aiops_maze_training.lessons.fixed_map_v1.environment import DIRECTIONS, State, initial_state, step
from aiops_maze_training.common.types import Maze, Memory, Sample
from aiops_maze_training.lessons.random_map_v1.environment import generate_maze as generate_random_maze
from aiops_maze_training.lessons.random_map_v1.environment import split_maps as split_random_maps


def generate_maze(seed: int) -> Maze:
    return {**generate_random_maze(seed), "max_steps": 200}


def split_maps(maps: int, seed: int) -> dict[str, list[Maze]]:
    return {
        name: [{**maze, "max_steps": 200} for maze in mazes] for name, mazes in split_random_maps(maps, seed).items()
    }


def observe(maze: Maze, state: State, memory: Memory | None = None) -> Memory:
    known = [list(row) for row in memory["known"]] if memory else [["?"] * 7 for _ in range(7)]
    visited = memory["visited"][:] if memory else [0] * 49
    for y in range(max(0, state.y - 1), min(7, state.y + 2)):
        for x in range(max(0, state.x - 1), min(7, state.x + 2)):
            known[y][x] = maze["rows"][y][x]
    visited[state.y * 7 + state.x] = 1
    return {"known": ["".join(row) for row in known], "visited": visited}


def initial_memory(maze: Maze, state: State) -> Memory:
    return observe(maze, state)


def encode(memory: Memory, state: State) -> list[float]:
    cells = "".join(memory["known"])
    return (
        [float(cell != "?") for cell in cells]
        + [float(cell == "#") for cell in cells]
        + [float(cell == "G") for cell in cells]
        + [float(value) for value in memory["visited"]]
        + [float(index == state.y * 7 + state.x) for index in range(49)]
        + [float(direction == state.direction) for direction in range(4)]
    )


def expert_action(memory: Memory, state: State) -> int:
    if state.status in ("success", "timeout") or memory["known"][state.y][state.x] == "G":
        raise ValueError("the expert does not act in terminal states")
    has_goal = any("G" in row for row in memory["known"])
    queue = deque([(state.x, state.y, -1)])
    seen = {(state.x, state.y)}
    while queue:
        x, y, first = queue.popleft()
        target = memory["known"][y][x] == "G" if has_goal else not memory["visited"][y * 7 + x]
        if first >= 0 and target:
            turn = (first - state.direction) % 4
            return 2 if turn == 0 else 1 if turn == 1 else 0
        for direction, (dx, dy) in enumerate(DIRECTIONS):
            nx, ny = x + dx, y + dy
            if 0 <= nx < 7 and 0 <= ny < 7 and (nx, ny) not in seen and memory["known"][ny][nx] in (".", "G"):
                seen.add((nx, ny))
                queue.append((nx, ny, direction if first == -1 else first))
    raise ValueError("no reachable observed exploration target")


def samples(maze: Maze) -> list[Sample]:
    state = initial_state(maze)
    memory = observe(maze, state)
    result: list[Sample] = []
    while state.status not in ("success", "timeout"):
        action = expert_action(memory, state)
        result.append(
            {
                "maze_id": maze["id"],
                "state": asdict(state),
                "memory": memory,
                "observation": encode(memory, state),
                "action": action,
            }
        )
        state = step(maze, state, action)
        memory = observe(maze, state, memory)
    return result
