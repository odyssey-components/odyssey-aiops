# Go1 深度视觉越障

独立入口 `aiops-go1-vision`（或 `python -m aiops_go1_training.vision.train`）使用前向机身深度相机、CNN 与45维身体状态融合策略、PPO，目标为跨越2–8厘米低障碍和台阶。旧状态策略及本地执行器原训练入口保持原协议；Go1 场景中保存目标后，可在创建训练任务时选择场景；选中场景走视觉训练，未选择场景仍使用原状态训练。任务保留场景快照，模型继承相同引用。

视觉策略不读取障碍物坐标。场景通过 MjSpec 在内存叠加到固定版本 Go1 模型；物理碰撞真实启用，原源码不被改写。按种子生成 block/step，训练和测试种子不能重叠。控制50Hz，相机10Hz，帧按仿真时间复用；新 episode 显式重置相机缓存。

## 环境要求与本机结果

训练支持显式CPU/CUDA/ROCm，默认CPU。CPU物理使用原生MuJoCo，策略与更新使用JAX CPU；GPU物理使用MJX，策略与更新使用相同GPU。渲染使用 MuJoCo EGL，与 MJX 状态同步，首版是单环境实现，不保证高吞吐；显式GPU请求失败不会静默回退CPU。

本机 AMD Navi14 独显可渲染深度图，但当前 JAX CUDA 后端初始化失败、只能使用 CPU，未安装 ROCm 用户态。独显能渲染不代表现有 JAX 能训练。不要通过伪装GPU、调整安全策略或把CPU单测当GPU训练绕过检查。

EGL device编号可能随进程加载的图形库改变；必须检查实际 `graphics_renderer`。本机完整CLI栈实测 `MUJOCO_EGL_DEVICE_ID=0` 为AMD，1为Intel，2为llvmpipe。其他机器不能照搬编号。

## 命令

来源为 `https://github.com/shinobi-wolf/go1-locomotion-rl.git`，固定提交 `f4fb0d12ac68842c1567bf1de6f771ce8e6671b5`，Go1 模型文件不得修改。源码目录由 `--source` 显式传入。

```bash
# 只检查来源和GPU，不创建训练输出
.venv/bin/python -m aiops_go1_training.vision.train --source /path/to/source --check

# 真实深度传感器/物理探针；不训练，不宣称越障成功
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
.venv/bin/python -m aiops_go1_training.vision.train \
  --source /path/to/source --probe --output /path/to/new-probe

# GPU训练 + 最终独立测试；配置示例见 examples/depth-obstacles.toml
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
.venv/bin/python -m aiops_go1_training.vision.train \
  --source /path/to/source --config training/go1/examples/depth-obstacles.toml \
  --output /path/to/new-run

# 加载视觉模型并在其配置的独立种子上评估，不重新训练
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
.venv/bin/python -m aiops_go1_training.vision.train \
  --source /path/to/source --evaluate /path/to/run --output /path/to/new-evaluation
```

省略输出时写入 `~/.odyssey/go1/vision/` 的新目录，不覆盖已有产物。训练产生 `policy.npz`、`model.manifest.toml`、`config.toml`、逐次训练指标，以及每个测试种子的物理轨迹和 `evaluation.toml`。检查点只加载纯数组，验证参数形状、有限性、模型/配置/源摘要；不兼容旧状态策略检查点，未提供恢复训练操作。深度帧与轨迹以仿真时间关联，模型关联源场景摘要与场景配置，不复制固定来源的模型资源。

评价成功要求机器人根位置超过障碍后沿，全部可碰撞机器人几何的保守后缘也已越过，并保持稳定1秒；躯干/腿碰撞障碍、跌倒、绕行和超时分别计为失败，四个命名脚部接触允许。最终测试不参与训练更新或模型选择。完成训练迭代不等于学会越障，必须查看真实测试成功数。

## 已验证与未验证

已验证：合成物理场景中的实际深度变化/采样时序/释放、CNN视觉敏感性和PPO梯度、模型纯数组往返及摘要拒绝、GPU缺失拒绝，以及固定来源真实Go1在本机AMD上的深度传感器探针。

传感器探针使用CPU物理、AMD图形渲染和零控制，实际结果为跌倒，`trained=false`；它证明视觉/场景通路，不证明越障能力。实际MJX GPU训练、GPU策略更新、训练后越障成功与独立测试成绩仍为 UNVERIFIED，需可用的JAX GPU后端解除。

## 应用场景目标

Go1 二级导航提供场景、训练任务、模型和评估回放。场景不可变保存为 TOML；修改参数应保存为新的场景 ID。拐弯配置角度/道路宽度/接近距离，窄路掉头配置宽度/掉头位置，小坡配置坡度/坡长。模型输出 scene.toml 与清单中的场景摘要绑定。

拐弯必须先沿来路到拐点，再按目标角度转向并前进；掉头必须先到指定位置，再转180°回到起点区域；小坡要求全部可碰撞几何越过坡顶。所有目标均要求无躯干碰撞、保持路径约束且稳定1秒。转弯通道、窄路侧墙和小坡均为真实碰撞实体。训练任务不能只带场景标题而执行原直线算法。当前GPU训练未实测，以上目标执行/评价链路不等于已训练获得对应能力。

2026-11-19 CPU实际训练：1次PPO迭代、8步、16×16深度图，生成策略及独立评估，参数最大变化0.000300005；独立评估跌倒。清单记录mujoco-cpu；该结果仅为流程验收。GPU的MJX圆柱/盒体碰撞限制尚未解除，不宣称GPU训练通过。

已有行走策略的转换、热启动与视觉残差接入见[上游策略适配](UPSTREAM.md)。选择基础模型后使用go1-depth-residual/v2，不改变已有go1-depth/v1策略语义。
