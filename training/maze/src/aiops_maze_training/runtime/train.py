"""Lesson 4: generate expert answers → predict → compare → update → repeat.

From the repository root:
uv run --project . --no-sync odymaze-train
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from aiops_maze_training.common.datasets import load_dataset, prepare_dataset, save_dataset
from aiops_maze_training.lessons.fixed_map_v1.environment import DEFAULT_RUNS, samples
from aiops_maze_training.lessons.fixed_map_v1.train import train
from aiops_maze_training.lessons.local_map_v1.train import DEFAULT_EPOCHS as LOCAL_EPOCHS
from aiops_maze_training.lessons.local_map_v1.train import DEFAULT_MAPS as LOCAL_MAPS
from aiops_maze_training.lessons.local_map_v1.train import train as train_local
from aiops_maze_training.lessons.memory_map_v1.train import DEFAULT_EPOCHS as MEMORY_EPOCHS
from aiops_maze_training.lessons.memory_map_v1.train import DEFAULT_MAPS as MEMORY_MAPS
from aiops_maze_training.lessons.memory_map_v1.train import train as train_memory
from aiops_maze_training.lessons.random_map_v1.train import DEFAULT_EPOCHS, DEFAULT_MAPS
from aiops_maze_training.lessons.random_map_v1.train import train as train_random
from aiops_maze_training.common.manifest import map_reference
from aiops_maze_training.common.onnx_export import export_onnx
from aiops_maze_training.common.types import Policy


def main() -> None:
    parser = argparse.ArgumentParser(description="AIOps：选择固定地图、完整观测或局部记忆的专家模仿课程")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数，0..5000；0 表示不训练，用于对照")
    parser.add_argument(
        "--lesson", choices=("fixed-map-v1", "random-map-v1", "local-map-v1", "memory-map-v1"), default="fixed-map-v1"
    )
    parser.add_argument("--maps", type=int, default=None, help="第二、三、四课地图数，16..2048")
    parser.add_argument("--seed", type=int, default=7, help="随机种子，0..2147483647")
    parser.add_argument("--output", type=Path, help="新建或空的输出目录；默认 ~/.odyssey/odymaze/runs/<时间>")
    parser.add_argument("--dataset", type=Path, help="预生成的本课数值地图集；提供后不再生成地图")
    parser.add_argument("--dataset-sha256", help="任务关联地图集的预期 SHA-256")
    args = parser.parse_args()
    if args.dataset_sha256 is not None and args.dataset is None:
        parser.error("dataset-sha256 必须与 dataset 一起使用")
    args.epochs = (
        args.epochs
        if args.epochs is not None
        else (
            MEMORY_EPOCHS
            if args.lesson == "memory-map-v1"
            else 400
            if args.lesson == "fixed-map-v1"
            else LOCAL_EPOCHS
            if args.lesson == "local-map-v1"
            else DEFAULT_EPOCHS
        )
    )
    if args.maps is not None and (args.lesson == "fixed-map-v1" or not 16 <= args.maps <= 2048):
        parser.error("maps 仅适用于第二、三、四课，且必须为 16..2048")
    if not 0 <= args.epochs <= 5000 or not 0 <= args.seed <= 2147483647:
        parser.error("epochs 必须为 0..5000，seed 必须为 0..2147483647")
    prepared = None
    dataset_path = args.dataset
    dataset_sha256 = args.dataset_sha256
    if args.dataset is not None:
        try:
            dataset_sha256 = dataset_sha256 or hashlib.sha256(args.dataset.read_bytes()).hexdigest()
            prepared = load_dataset(args.dataset, args.lesson, expected_sha256=dataset_sha256)
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"无法读取预生成地图：{error}")
    output = args.output or DEFAULT_RUNS / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    output = output.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            parser.error("输出目录非空，请更换目录；不会覆盖已有文件")
    except OSError as error:
        parser.error(f"无法使用输出目录：{error}")
    print(f"课程：{args.lesson}；设备：CPU；训练轮数：{args.epochs}", flush=True)
    if prepared is None:
        count = args.maps or (
            MEMORY_MAPS
            if args.lesson == "memory-map-v1"
            else LOCAL_MAPS
            if args.lesson == "local-map-v1"
            else DEFAULT_MAPS
        )
        prepared = prepare_dataset(args.lesson, args.seed, count)
        dataset_path = DEFAULT_RUNS.parent / "datasets" / f"{uuid4()}.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(prepared, dataset_path)
        dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert dataset_path is not None and dataset_sha256 is not None
    reference = {"id": dataset_path.stem, "sha256": dataset_sha256}
    print("读取已准备的地图集；训练阶段不生成地图。", flush=True)
    if args.lesson in ("random-map-v1", "local-map-v1", "memory-map-v1"):
        trainer = (
            train_memory
            if args.lesson == "memory-map-v1"
            else train_local
            if args.lesson == "local-map-v1"
            else train_random
        )
        policy, random_report, dataset = trainer(
            args.epochs,
            args.seed,
            len(prepared["splits"]["train"]),
            splits=prepared["splits"],
        )
        report_value = random_report
        print(
            f"独立测试通关：{random_report['initial']['success_rate']:.1%} → {random_report['final']['success_rate']:.1%}"
        )
    else:
        maze = prepared["splits"]["train"][0]
        dataset = samples(maze)
        policy, report = train(maze, args.epochs, args.seed)
        report_value = report
        for name, score in (("训练前", report["initial"]), ("训练后", report["final"])):
            print(f"{name}：专家动作正确率 {score['accuracy']:.1%}；通关 {score['success']}；步数 {score['steps']}")
    try:
        policy = cast(Policy, {**policy, "dataset": reference})
        export_onnx(policy, output / "model.onnx")
        report_value = {
            **report_value,
            "dataset": {
                "format": "odymaze-maps/v2",
                "sha256": dataset_sha256,
            },
        }
        readable = {key: value for key, value in policy.items() if key != "maze"}
        readable["map_ref"] = map_reference(policy)
        for name, value in (("model.json", readable), ("samples.json", dataset), ("report.json", report_value)):
            with (output / name).open("x") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
    except (OSError, ValueError) as error:
        parser.error(f"无法保存产物：{error}")
    print(
        f"ONNX 已保存：{output / 'model.onnx'}\n清单：{output / 'model.manifest.toml'}\n"
        f"教学 JSON：{output / 'model.json'}\n在网页刷新本课模型列表，选择即使用。网页只读取 ~/.odyssey/odymaze/runs/ 下的模型。"
    )


if __name__ == "__main__":
    main()

__all__ = ["main", "train"]
