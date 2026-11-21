"""Imitate many maps; choose on validation rollouts, test only after selection."""

from copy import deepcopy
from typing import TypedDict

import torch
from torch import nn

from aiops_maze_training.common.metrics import emit_metric
from aiops_maze_training.common.types import Epoch, Maze, Metadata, Policy, Sample
from aiops_maze_training.lessons.local_map_v1.environment import samples, split_maps
from aiops_maze_training.lessons.local_map_v1.model import export_model, make_model, rollout

DEFAULT_EPOCHS = 120
DEFAULT_MAPS = 128


class Episode(TypedDict):
    success: bool
    steps: int
    actions: list[int]
    seed: int
    rows: list[str]


class Scores(TypedDict):
    success_rate: float
    episodes: list[Episode]


class RandomReport(TypedDict):
    lesson: str
    seed: int
    epochs: int
    maps: int
    samples: int
    learning_rate: float
    selected_epoch: int
    history: list[Epoch]
    validation: list[dict[str, float]]
    splits: dict[str, list[int]]
    initial: Scores
    final: Scores


def evaluate_maps(model: nn.Sequential, mazes: list[Maze]) -> Scores:
    episodes: list[Episode] = []
    for maze in mazes:
        score = rollout(model, maze)
        episodes.append(
            {
                "success": score["success"],
                "steps": score["steps"],
                "actions": score["actions"],
                "seed": int(maze["id"].rsplit("-", 1)[1]),
                "rows": maze["rows"],
            }
        )
    return {"success_rate": sum(row["success"] for row in episodes) / len(episodes), "episodes": episodes}


def train(
    epochs: int = DEFAULT_EPOCHS,
    seed: int = 7,
    maps: int = DEFAULT_MAPS,
    *,
    splits: dict[str, list[Maze]] | None = None,
) -> tuple[Policy, RandomReport, list[Sample]]:
    if not 0 <= epochs <= 5000:
        raise ValueError("epochs must be 0..5000")
    splits = split_maps(maps, seed) if splits is None else splits
    maps = len(splits["train"])
    torch.set_num_threads(1)
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    dataset = [row for maze in splits["train"] for row in samples(maze)]
    features = torch.tensor([row["observation"] for row in dataset], dtype=torch.float32)
    labels = torch.tensor([row["action"] for row in dataset], dtype=torch.long)
    model = make_model()
    baseline = deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    selected = deepcopy(model)
    best_score = evaluate_maps(model, splits["validation"])["success_rate"]
    selected_epoch = 0
    history: list[Epoch] = []
    validation: list[dict[str, float]] = [{"epoch": 0, "success_rate": best_score}]
    emit_metric(0, epochs, best_score, len(splits["validation"]), "validation")
    for epoch in range(1, epochs + 1):
        # Mini-batches make a bounded CPU demonstration practical.
        indices = torch.randperm(len(dataset))
        for start in range(0, len(dataset), 512):
            batch = indices[start : start + 512]
            loss = nn.functional.cross_entropy(model(features[batch]), labels[batch])
            optimizer.zero_grad()
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        with torch.no_grad():
            logits = model(features)
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(nn.functional.cross_entropy(logits, labels)),
                    "accuracy": float((logits.argmax(dim=1) == labels).float().mean()),
                }
            )
        score = evaluate_maps(model, splits["validation"])["success_rate"]
        history[-1]["success_rate"] = score
        validation.append({"epoch": epoch, "success_rate": score})
        if score > best_score:
            best_score, selected_epoch, selected = score, epoch, deepcopy(model)
        emit_metric(
            epoch,
            epochs,
            score,
            len(splits["validation"]),
            "validation",
            loss=history[-1]["loss"],
            accuracy=history[-1]["accuracy"],
        )
        print(f"epoch {epoch}/{epochs}; validation success {score:.1%}", flush=True)
    # Neither test results nor test labels affect checkpoint selection.
    initial = evaluate_maps(baseline, splits["test"])
    final = evaluate_maps(selected, splits["test"])
    with torch.no_grad():
        accuracy = float((selected(features).argmax(dim=1) == labels).float().mean())
    metadata: Metadata = {
        "seed": seed,
        "epochs": epochs,
        "samples": len(dataset),
        "accuracy": accuracy,
        "success": final["success_rate"] == 1.0,
        "test_success_rate": final["success_rate"],
    }
    report: RandomReport = {
        "lesson": "local-map-v1",
        "seed": seed,
        "epochs": epochs,
        "maps": maps,
        "samples": len(dataset),
        "learning_rate": 0.003,
        "selected_epoch": selected_epoch,
        "history": history,
        "validation": validation,
        "splits": {name: [int(maze["id"].rsplit("-", 1)[1]) for maze in mazes] for name, mazes in splits.items()},
        "initial": initial,
        "final": final,
    }
    return export_model(selected, splits["train"][0], metadata), report, dataset
