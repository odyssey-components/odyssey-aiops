"""Lesson 4: generate expert answers → predict → compare → update → repeat.

From the repository root:
uv run --project . --no-sync odymaze-train
"""

import torch
from torch import nn

from aiops_maze_training.common.metrics import emit_metric
from aiops_maze_training.common.types import Epoch, Maze, Metadata, Policy, Report
from aiops_maze_training.lessons.fixed_map_v1.environment import samples
from aiops_maze_training.lessons.fixed_map_v1.model import evaluate, export_model, make_model


def train(maze: Maze, epochs: int = 400, seed: int = 7) -> tuple[Policy, Report]:
    if not 0 <= epochs <= 5000 or not 0 <= seed <= 2147483647:
        raise ValueError("epochs 必须为 0..5000，seed 必须为 0..2147483647")
    torch.set_num_threads(1)
    # These PyTorch signatures leave seed/closure/autograd arguments untyped.
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    dataset = samples(maze)
    features = torch.tensor([row["observation"] for row in dataset], dtype=torch.float32)
    labels = torch.tensor([row["action"] for row in dataset], dtype=torch.long)
    model = make_model(features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial = evaluate(model, maze, features, labels)
    history: list[Epoch] = []
    validation: list[dict[str, float]] = [{"epoch": 0, "success_rate": float(initial["success"])}]
    emit_metric(0, epochs, float(initial["success"]), 1, "fixed", accuracy=initial["accuracy"])
    for epoch in range(1, epochs + 1):
        # 1. Predict an action distribution for every example.
        logits = model(features)
        # 2. Compare the prediction with the expert's answer.
        loss = nn.functional.cross_entropy(logits, labels)
        # 3. Compute gradients, then update the SAME model's parameters.
        optimizer.zero_grad()
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        # Measure after the update, so this row describes the saved parameters.
        with torch.no_grad():
            updated = model(features)
            record: Epoch = {
                "epoch": epoch,
                "loss": float(nn.functional.cross_entropy(updated, labels)),
                "accuracy": float((updated.argmax(dim=1) == labels).float().mean()),
            }
        score = float(evaluate(model, maze, features, labels)["success"])
        record["success_rate"] = score
        history.append(record)
        validation.append({"epoch": epoch, "success_rate": score})
        emit_metric(epoch, epochs, score, 1, "fixed", loss=record["loss"], accuracy=record["accuracy"])
        print(
            f"轮次 {epoch:4d}/{epochs}  损失 {record['loss']:.6f}  专家动作正确率 {record['accuracy']:.1%}",
            flush=True,
        )
    final = evaluate(model, maze, features, labels)
    metadata: Metadata = {
        "seed": seed,
        "epochs": epochs,
        "samples": len(dataset),
        "accuracy": final["accuracy"],
        "success": final["success"],
    }
    policy = export_model(model, maze, metadata)
    report: Report = {
        **metadata,
        "learning_rate": 0.03,
        "history": history,
        "validation": validation,
        "initial": initial,
        "final": final,
    }
    return policy, report
