"""Full-map observations and reproducible, reachable synthetic maps."""

from dataclasses import asdict

from aiops_maze_training.lessons.fixed_map_v1.environment import State, expert_action, goal_distances
from aiops_maze_training.common.types import Maze, Sample


def generate_maze(seed: int) -> Maze:
    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("map seed must be uint32")
    state = seed

    def random() -> float:
        nonlocal state
        state = (1664525 * state + 1013904223) % (2**32)
        return state / (2**32)

    cells = [["#"] * 7 for _ in range(7)]
    for y in range(1, 6):
        for x in range(1, 6):
            cells[y][x] = "#" if random() < 0.28 else "."
    x, y = 1, 1
    while (x, y) != (5, 5):
        cells[y][x] = "."
        if x == 5:
            y += 1
        elif y == 5 or random() < 0.5:
            x += 1
        else:
            y += 1
    cells[5][5] = "G"
    return {
        "id": f"odymaze-random-{seed}",
        "rows": ["".join(row) for row in cells],
        "start": {"x": 1, "y": 1, "direction": 0},
        "max_steps": 100,
    }


def encode(maze: Maze, state: State) -> list[float]:
    cells = "".join(maze["rows"])
    return (
        [float(cell == "#") for cell in cells]
        + [float(cell == "G") for cell in cells]
        + [float(index == state.y * 7 + state.x) for index in range(49)]
        + [float(direction == state.direction) for direction in range(4)]
    )


def samples(maze: Maze) -> list[Sample]:
    result: list[Sample] = []
    for (x, y), distance in sorted(goal_distances(maze).items()):
        if distance:
            for direction in range(4):
                state = State(x, y, direction)
                result.append(
                    {
                        "maze_id": maze["id"],
                        "state": asdict(state),
                        "observation": encode(maze, state),
                        "action": expert_action(maze, state),
                    }
                )
    return result


def split_maps(maps: int, seed: int) -> dict[str, list[Maze]]:
    """Disjoint million-seed intervals; deduplicate complete rows across all sets."""
    if not 16 <= maps <= 2048 or not 0 <= seed <= 2147483647:
        raise ValueError("maps must be 16..2048 and seed 0..2147483647")
    seen: set[tuple[str, ...]] = set()
    result: dict[str, list[Maze]] = {}
    for offset, (name, count) in enumerate((("train", maps), ("validation", 64), ("test", 100))):
        selected: list[Maze] = []
        for candidate in range(seed + offset * 1000000, seed + (offset + 1) * 1000000):
            maze = generate_maze(candidate)
            content = tuple(maze["rows"])
            if content not in seen:
                seen.add(content)
                selected.append(maze)
            if len(selected) == count:
                break
        result[name] = selected
    return result
