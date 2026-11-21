# Maze 训练包

发行名称 `aiops-maze-training`，导入 `aiops_maze_training`。四课位于 `src/aiops_maze_training/lessons/`，每课保留独立环境、模型和训练代码。依赖 PyTorch、ONNX 与通用训练包，不依赖 Go1。

仓库内手动训练：`uv run --project . --no-sync aiops-maze-train --lesson random-map-v1`。

发行包构建及独立环境安装见 [产品说明](../../README.md)。`aiops-maze-export` 导出 ONNX，旧 `odymaze-train` / `odymaze-export` CLI别名继续兼容，Python调用统一使用子包路径。固定地图资源由构建脚本从 web/maze.json 打入 wheel。

根包仅保留 `__init__.py` / `__main__.py`；`lessons/` 是四课算法，`common/` 是数据、协议、指标与ONNX，`runtime/` 是训练编排、导出、迁移和worker。可用 `python -m aiops_maze_training train --help` 查看入口。
