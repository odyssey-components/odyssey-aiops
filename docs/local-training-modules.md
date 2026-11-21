# Maze 与 Go1 本地训练

一级导航选择 Maze 或 Go1。Maze 保留四个课时、地图组、训练任务与模型；Go1 提供场景、训练任务、模型和仿真任务；评估结果与三维回放属于对应仿真任务。前端由 Vite 提供，统一调用本地 Go API，两个模块共享同一串行执行器。

## Go1 使用

1. 在 Go 服务端用 `--go1-python` / `ODYSSEY_GO1_PYTHON` 配置 Go1 Python，用 `--go1-source` / `ODYSSEY_GO1_SOURCE` 配置固定模型源码目录。Python 环境需要安装共享训练包、JAX、MuJoCo/MJX、NumPy 和配置依赖；训练不从浏览器传递命令或路径。
2. 源码必须处于 commit `f4fb0d12ac68842c1567bf1de6f771ce8e6671b5`，`unitree_go1` 无修改并包含 `scene_torque.xml`。Go1状态训练使用MJX PPO和45维观测、12维动作；视觉CPU使用原生MuJoCo物理与JAX PPO。
3. Go1「训练任务」选择CPU、NVIDIA CUDA或AMD ROCm，再点击「检查环境」。源码、依赖或所选设备不满足时显示原因，不启用创建按钮。就绪后点击「添加训练任务」，填写参数。
4. 同一时刻只运行一个训练进程。Maze 和 Go1 的新任务按顺序排队；排队与运行中的任务均可取消。Go1 单任务最多 100 万采样步、运行上限一小时。
5. 任务详情查看日志及平均奖励、PPO 损失、机身高度、采样回合结束数和用时。指标来自实际训练进度，不将进程成功表述为训练收敛。
6. 完整模型在「模型」列表中可查看 NPZ 摘要、训练参数、独立扰动评估与环境来源；点击“创建仿真任务”预选当前模型，再选择场景执行；“训练评估记录”保留训练结束时生成的历史三维轨迹，不重新计算物理。检查点不提供断点恢复功能。

## 仿真、评估与回放

在“仿真任务”选择模型、场景、种子和步数后执行。也可从训练模型或基础模型详情进入，预选模型并打开表单；仅打开或取消表单不会执行任务。

任务列表分别显示执行状态和评估结果；“已完成”表示运行完成，不表示达到场景目标。详情记录模型摘要、场景参数、成功条件和实际运行指标。完成后的三维回放展示任务保存的场景结构与机器狗轨迹，支持返回原任务；后续修改场景库不会改变历史任务快照。模型中的“训练评估记录”和旧回放链接仍保留，不在导航中提供另一套评估任务。

## 数据与兼容

Maze 继续使用 `~/.odyssey/odymaze/{datasets,jobs,runs}`；Go1 使用 `~/.odyssey/go1/{jobs,runs,scenes,simulations}`，不迁移或删除既有数据。Go1 任务原子保存 `job.json`、训练配置快照与有界日志/指标；成功模型与评估产物关联来源任务。服务重启后未结束任务标记中断，不自动重训。

Go1 算法的 canonical 实现在 `training/go1/src/aiops_go1_training/`；OdyLab 旧训练入口转发到该实现，算法包携带相同源码。当前应用不恢复 Lab Task/Attempt 写入，不调用其私有训练服务。

共享执行器、API 运行与 Python wheel 安装命令见 [Go 本地服务](go-local-service.md)。真实GPU训练需要对应的CUDA或ROCm环境与固定源码，合成接口或 CPU 回归不能替代该项验证。

## 设备与CPU实测

默认设备为CPU，预检和任务使用同一选择，任务详情记录设备。切换设备后需重新检查环境；GPU请求不可用时拒绝，不回退CPU。CLI使用 `--device cpu|cuda|rocm`，Go API的check/train使用独立`device`字段。

本机CPU验收：状态训练1迭代×8步并完成独立评估，实际权重变化；视觉训练1迭代×8步、16×16深度输入、独立种子评估，策略权重变化。页面→Go→CPU训练→模型校验发布→回放→刷新实测通过。这证明训练流程，不证明学习收敛；视觉短训评估结果为跌倒。

CPU视觉使用原生MuJoCo，保持固定Go1模型及真实碰撞，清单`physics=mujoco-cpu`。GPU保持MJX（`mjx-gpu`）；MJX当前对圆柱与盒体碰撞存在限制，GPU视觉完整训练尚未实测。

AMD GPU需JAX ROCm插件及受支持的GPU/系统；本机Navi14不在AMD官方Radeon支持矩阵中，实际ROCm检查不通过。未安装驱动或尝试不受支持的硬件覆盖。官方依据：[JAX安装](https://docs.jax.dev/en/latest/installation.html)、[AMD兼容矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html)。图形显卡能渲染深度图，不代表JAX能用它计算。

## 三维轨迹回放

模型的训练评估轨迹与独立仿真轨迹使用同一个Three.js视图。机器人为按Go1关节轴和连杆偏移构建的简化模型，使用已保存的wxyz朝向、三维位置和FR/FL/RR/RL十二关节角；浏览器不执行NPZ策略或物理仿真，不需要重新训练。

播放按真实时间戳推进，支持暂停、重置、0.25/0.5/1/2倍速、时间滑块、旋转/缩放/平移、跟随机器人和适配轨迹。同一回合内仅做显示插值；episode_end之间不插值，也不连接重置后的地面路径。切换来源、加载失败、卸载或页面隐藏会停止播放。单帧不可播放；WebGL创建失败或上下文丢失会明确显示错误并支持重新创建视图。

Three.js按需加载；进入Maze不加载三维引擎。截图与交互验收使用受控headless及合成姿态，验证真实WebGL draw calls、全部关节与相机变化、播放/拖动/暂停、旧请求、上下文恢复及窄屏。

## Go1 三维模型资产

场景详情与仿真回放共用训练源码固定版本的Go1原始STL视觉网格，按MJCF关节和局部视觉变换组装；场景显示home初始姿态，回放应用已保存的机身姿态和十二关节轨迹，不重新运行物理。原始STL、MJCF、BSD-3-Clause许可和来源说明随前端发布于web/public/models/go1，部署后不依赖本机训练源码目录或外部CDN。

五个网格约9.9MiB，仅进入三维页面后按需加载；两页共享字节缓存，Three几何/材质按视口独立释放。加载失败显示重试，不回退到简化模型。
