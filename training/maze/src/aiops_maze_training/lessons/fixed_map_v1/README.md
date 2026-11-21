# 第一课：熟悉固定地图

本目录保留最初的训练方案，作为学习起点。

- `environment.py`：原固定地图规则、位置/朝向编码、BFS 专家与 68 条样本。
- `model.py`：18→32 tanh→3 网络、实际行走评估、JSON 导出。
- `train.py`：预测、交叉熵、反向传播、Adam 更新，默认 400 轮。

仓库根运行 `uv run --project . --no-sync aiops-maze-train`，或加 `--lesson fixed-map-v1`。`--epochs 0` 可生成初始对照。旧 v1 模型无课程字段仍能加载，旧 Python 模块导入继续有效。

详细解释见产品 `docs/learning.md`。本课只学习固定地图，不能用其正确率证明陌生地图能力。
