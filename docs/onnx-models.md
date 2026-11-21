# 将训练模型输出为ONNX

参考OdyLab，AIOps训练现在同时生成 `model.onnx` 和 `model.manifest.toml`。ONNX把网络结构和训练权重打包成标准推理文件，其他程序可以通过ONNX Runtime等运行时使用它，不必重写网络的矩阵计算。

## 训练命令保持不变

```bash
# 第一课
uv run --project . --no-sync aiops-maze-train

# 第二、三、四课
uv run --project . --no-sync aiops-maze-train --lesson random-map-v1
uv run --project . --no-sync aiops-maze-train --lesson local-map-v1
uv run --project . --no-sync aiops-maze-train --lesson memory-map-v1
```

每个新run包含：

| 文件 | 用途 |
| --- | --- |
| model.onnx | 网络结构与训练后选中的权重，自包含文件 |
| model.manifest.toml | 课程、输入编码、动作顺序、输入输出形状、SHA-256与训练摘要 |
| model.json | 可读教学参数，供离线讲解与转换 |
| samples.json | 训练用的专家习题与答案 |
| report.json | 训练记录和真实评估结果 |

模型输出默认仍在 `~/.odyssey/odymaze/runs/<时间>/`。训练拒绝非空输出目录，转换拒绝覆盖已有ONNX或清单。ONNX文件和运行数据不进入Git。

网页现在直接读取ONNX与清单，在自托管的ONNX Runtime WASM中推理；不会请求JSON网络权重或回退到原生矩阵计算。未配对、清单无效或hash不匹配的文件不会出现在模型列表。推理期间禁止新动作；过时请求不会覆盖重置、换图或新模型的状态。资源不依赖CDN。

## 已有模型不用重新训练

在仓库根目录运行，替换成实际模型路径：

```bash
uv run --project . --no-sync aiops-maze-export \
  --input "$HOME/.odyssey/odymaze/runs/<run>/model.json"
```

默认在同目录生成 `model.onnx` 和 `model.manifest.toml`，保留原JSON。也能指定新路径：

```bash
uv run --project . --no-sync aiops-maze-export \
  --input "$HOME/.odyssey/odymaze/runs/<run>/model.json" \
  --output "$HOME/.odyssey/odymaze/runs/<run>/exports/policy.onnx"
```

对应清单名为 `policy.manifest.toml`。已有旧v1模型缺少lesson字段时仍按第一课导出；不同课程或损坏参数会明确拒绝。

如果旧环境提示找不到新增的 `aiops-maze-export`，在已有根虚拟环境中刷新本地包入口：

```bash
uv pip install --python .venv/bin/python --no-deps -e training/maze
```

包声明了torch和onnx依赖；常规Linux环境仍按README使用锁文件同步。ONNX Runtime用于运行和数值验证，根工作区已有此依赖。

## 输入输出有什么区别？

所有张量为float32，支持动态batch，opset17。batch表示同时运行几局，不是把同一局的连续时间步堆在一起。

| 课时 | 输入 | 输出 |
| --- | --- | --- |
| 第一课 | observations[batch,18] | logits[batch,3] |
| 第二课 | observations[batch,151] | logits[batch,3] |
| 第三课 | observations[batch,249] | logits[batch,3] |
| 第四课 | observations[batch,40]、hidden[batch,32] | logits[batch,3]、next_hidden[batch,32] |

`logits`是左转、右转、前进三个动作分数。选择最高分即可决定动作；需要显示概率时再计算softmax。ONNX不包含地图生成、观察编码、探索专家或动作执行，这些仍由调用程序负责。

第四课每局开始时hidden全0。处理当前观察得到next_hidden，实际执行动作后构造下一份观察，再把上次的next_hidden作为新的hidden。每个观察处理一次；重置或换局时清零。训练权重保存在ONNX里，hidden是这一局的临时记忆，两者不同。

## 用ONNX Runtime检查一条样本

在run目录中运行下面的Python代码；第四课这段代码只演示初始观察，不代表可以每一步都重新清零hidden。

```python
import json

import numpy as np
import onnxruntime as ort

with open("samples.json") as stream:
    sample = json.load(stream)[0]

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
feeds = {"observations": np.asarray([sample["observation"]], dtype=np.float32)}
if any(item.name == "hidden" for item in session.get_inputs()):
    feeds["hidden"] = np.zeros((1, 32), dtype=np.float32)
outputs = dict(zip([item.name for item in session.get_outputs()], session.run(None, feeds)))
print("动作分数：", outputs["logits"])
if "next_hidden" in outputs:
    print("下一步需要携带的记忆：", outputs["next_hidden"])
```

浏览器还会校验清单、二进制SHA-256和实际会话输入输出。第四课每次执行动作后提交一次新的异步观察，模式切换与页面重绘不推进记忆；推理失败会暂停并提示，不用旧预测兜底。

导出过程校验ONNX图和自包含权重，清单记录模型字节的SHA-256。产品回归另外实际执行ONNX Runtime，比较四课的PyTorch输出、双样本batch和GRU多步隐藏状态；清单校验不代替运行时验证。

## 对照源码

- [导出与配套清单](../training/maze/src/aiops_maze_training/common/onnx_export.py)
- [已有JSON转换命令](../training/maze/src/aiops_maze_training/runtime/export.py)
- [训练产物入口](../training/maze/src/aiops_maze_training/runtime/train.py)

清单统一使用 TOML，`dataset` 保存地图组 ID/SHA-256，`map_ref.id` 引用已版本化地图标识，不内嵌网格。张量 `shape` 用 `{ batch = "batch", width = N }` 表示，HTTP 接口仍以 JSON 传输，加载层规范化后校验。运行时只读取 TOML 清单，旧 JSON 清单必须先迁移。训练输出不再包含 dataset.jsonl/toml；地图组只在 datasets 中保存。教学 model.json 同样使用地图引用。直接 CLI 训练先注册地图组，显式 --dataset 则复用该组。

历史 run 可显式迁移（先备份，找不到匹配的原始地图组则拒绝）：

```bash
uv run --no-sync python -m aiops_maze_training.runtime.migrate_run \
  --run /path/to/run --datasets /path/to/datasets \
  --backup /path/to/new-backup-directory
```

迁移只改写元信息、移除 run 内已匹配的地图副本，不修改 ONNX。备份目录保存原始 JSON 清单、教学模型和两份地图文件；运行时不再读取旧 JSON 清单。早期未记录地图组来源的模型仅保留已验证的地图标识，并标记 dataset_provenance 为 unrecorded，不猜测地图组。

全部历史 runs/jobs 可使用 `python -m aiops_maze_training.runtime.migrate_storage --base /path/to/odymaze --backup /path/to/new-backup-directory` 批量迁移。已有 ONNX 保持字节不变，JSON-only 模型只转换格式、不重新训练；空目录无可迁移产物，保留原样。
