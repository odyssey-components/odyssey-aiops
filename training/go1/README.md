# Go1 训练包

发行名称 `aiops-go1-training`，导入 `aiops_go1_training`。MJX PPO、环境、评估、检查点及回放属于此包。仅依赖通用训练包与基础配置/数值依赖；CPU训练使用 `cpu` extra，NVIDIA训练使用 `gpu` extra；ROCm另需AMD适配插件。

安装到仓库根 `.venv`：`bun scripts/setup-python.ts --release tmp/aiops-final-release --project go1`。Go服务默认使用同一个仓库根解释器；如需覆盖可通过 `--go1-python` 配置，使用 `--go1-source` 指向固定 commit 的模型源码。

`aiops-go1-train` 为本地适配入口；检查、参数与产物校验见 [使用说明](../../docs/local-training-modules.md)。统一使用 `aiops_go1_training` 导入，不再提供旧 Go1 包路径；共享包不包含 Go1 实现。

深度视觉越障的独立 CLI、相机/场景配置及设备配置及本机验证边界见 [VISION.md](VISION.md)。新视觉模型采用独立协议，原状态训练入口保持原用法。

包内视觉代码统一放在 `aiops_go1_training.vision`，使用 `configuration`、`policy`、`scene`、`rollout`、`evaluation`、`artifacts`、`goals`、`train` 模块；根目录不再保留 `vision_*` 兼容文件。

## 包结构

- `aiops_go1_training.locomotion`：运动控制 PPO、配置、环境、评估、检查点及模型清单。
- `aiops_go1_training.vision`：视觉课程的配置、策略、场景、采样、评估、产物和训练入口。
- `aiops_go1_training.runtime`：本地入口、项目 worker、资源监控和源码来源校验。

业务实现直接依赖所属子包。根目录仅保留 `__init__.py` 与 `__main__.py` 入口；命令行脚本和 worker 均指向对应子包。wheel 与 OdyLab 算法包递归包含子包，训练源码摘要使用 canonical 文件。

根包仅保留 `__init__.py` / `__main__.py`；`locomotion/`、`vision/` 保存各训练实现，`runtime/` 保存预检和进程支持，不再提供根模块兼容转发。可用 `python -m aiops_go1_training --help` 查看入口。

## 独立仿真任务

`simulation/` 实现只读策略仿真；`runtime.simulation` 是固定命令入口，也可用 `python -m aiops_go1_training simulation --help` 查看参数。仿真使用 MuJoCo CPU 物理，运动控制策略使用 NumPy 推理，视觉策略使用 JAX CPU 与真实深度相机，不执行优化器或更新模型。可安装本包的 `simulation` extra 获取 JAX CPU、MuJoCo 和 PyOpenGL 依赖。

Go API 从服务端固定源码目录读取机器人，要求固定 commit 和未修改模型文件；任务模型文件、清单和场景参数会单独保存为输入快照。返回的进程完成状态与目标结果分开表示，跌倒、碰撞、偏离道路和超时均保存为真实结果，不记作目标成功。

设备默认CPU，命令示例：

```bash
.venv/bin/python -m aiops_go1_training locomotion --device cpu --check --source "$HOME/.odyssey/go1/source"
.venv/bin/python -m aiops_go1_training locomotion --device cpu --source "$HOME/.odyssey/go1/source" --config /path/to/training.toml --output /path/to/new-run
.venv/bin/python -m aiops_go1_training vision --device cpu --source "$HOME/.odyssey/go1/source" --config /path/to/vision.toml --output /path/to/new-visual-run
```

`--device cuda` / `--device rocm` 显式选择对应GPU，失败不会改用CPU。完整说明见[本地训练模块](../../docs/local-training-modules.md)。
