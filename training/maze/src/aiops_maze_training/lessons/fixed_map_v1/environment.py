"""Lesson 1–2: maze rules, observations, and a shortest-path teacher.

Coordinates: x goes right, y goes down. Headings: east, south, west, north.
Actions: 0=left, 1=right, 2=forward. A turn is one step too.
"""

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from aiops_maze_training.common.types import Maze, Sample

DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
DEFAULT_RUNS = Path.home() / ".odyssey/odymaze/runs"
# Wheels receive this resource from the canonical web/maze.json during packaging.
_PACKAGED_MAZE = Path(__file__).resolve().parents[2] / "resources" / "fixed_maze.json"
MAZE_PATH = _PACKAGED_MAZE if _PACKAGED_MAZE.is_file() else Path(__file__).resolve().parents[6] / "web" / "maze.json"


@dataclass(frozen=True)
class State:
    x: int
    y: int
    direction: int
    steps: int = 0
    status: str = "ready"


def load_maze() -> Maze:
    return cast(Maze, json.loads(MAZE_PATH.read_text()))


def initial_state(maze: Maze) -> State:
    return State(**maze["start"])


def walkable(maze: Maze, x: int, y: int) -> bool:
    rows = maze["rows"]
    return 0 <= y < len(rows) and 0 <= x < len(rows[0]) and rows[y][x] != "#"


def step(maze: Maze, state: State, action: int) -> State:
    """Execute exactly one action. Terminal states never change."""
    if action not in (0, 1, 2):
        raise ValueError("action must be left=0, right=1, or forward=2")
    if state.status in ("success", "timeout"):
        return state
    x, y, direction = state.x, state.y, state.direction
    if action != 2:
        direction = (direction + (1 if action == 1 else -1)) % 4
    else:
        dx, dy = DIRECTIONS[direction]
        if walkable(maze, x + dx, y + dy):
            x, y = x + dx, y + dy
    steps = state.steps + 1
    status = "success" if maze["rows"][y][x] == "G" else "timeout" if steps >= maze["max_steps"] else "running"
    return State(x, y, direction, steps, status)


def encode(maze: Maze, state: State) -> list[float]:
    """Three one-hot groups. No expert answer or goal distance is included."""
    width, height = len(maze["rows"][0]), len(maze["rows"])
    result = [0.0] * (width + height + 4)
    result[state.x] = 1.0
    result[width + state.y] = 1.0
    result[width + height + state.direction] = 1.0
    return result


def goal_distances(maze: Maze) -> dict[tuple[int, int], int]:
    """BFS spreads from the exit; every reachable cell gets its distance."""
    goal = next((x, y) for y, row in enumerate(maze["rows"]) for x, cell in enumerate(row) if cell == "G")
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            point = (x + dx, y + dy)
            if point not in distances and walkable(maze, *point):
                distances[point] = distances[x, y] + 1
                queue.append(point)
    return distances


def expert_action(maze: Maze, state: State) -> int:
    distances = goal_distances(maze)
    distance = distances[state.x, state.y]
    if distance == 0:
        raise ValueError("the expert does not act after reaching the exit")
    target = next(
        direction
        for direction, (dx, dy) in enumerate(DIRECTIONS)
        if distances.get((state.x + dx, state.y + dy)) == distance - 1
    )
    turn = (target - state.direction) % 4
    return 2 if turn == 0 else 1 if turn == 1 else 0


def samples(maze: Maze) -> list[Sample]:
    """Ask the expert at every reachable non-goal cell, in all four headings.

    These examples teach this one map. They are not evidence of generalization.
    """
    result: list[Sample] = []
    for (x, y), distance in sorted(goal_distances(maze).items()):
        if distance == 0:
            continue
        for direction in range(4):
            state = State(x, y, direction)
            result.append(
                {"state": asdict(state), "observation": encode(maze, state), "action": expert_action(maze, state)}
            )
    return result
