"""Convert an existing teaching JSON policy without retraining or overwriting outputs."""

import argparse
import json
from pathlib import Path
from typing import cast

from aiops_maze_training.common.onnx_export import export_onnx
from aiops_maze_training.common.types import Policy

MAX_JSON_BYTES = 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description="AIOps：将已有教学 JSON 转换为 ONNX 和配套清单，无需重新训练")
    parser.add_argument("--input", type=Path, required=True, help="已有 model.json 路径")
    parser.add_argument("--output", type=Path, help="新的 .onnx 文件路径；默认与输入同目录的 model.onnx")
    args = parser.parse_args()
    output = args.output or args.input.parent / "model.onnx"
    if output.suffix != ".onnx":
        parser.error("output 必须以 .onnx 结尾")
    try:
        with args.input.expanduser().open("rb") as source:
            content = source.read(MAX_JSON_BYTES + 1)
        if len(content) > MAX_JSON_BYTES:
            raise ValueError("教学 JSON 超过 1 MiB")
        policy = cast(Policy, json.loads(content))
        result = export_onnx(policy, output.expanduser())
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(f"无法转换模型：{error}")
    print(f"ONNX 已保存：{result}\n清单：{result.with_suffix('.manifest.toml')}")


if __name__ == "__main__":
    main()
