# 上游行走策略适配

来源：shinobi-wolf/go1-locomotion-rl，固定 f4fb0d12ac68842c1567bf1de6f771ce8e6671b5。
本模块保留JAX/MJX。导入支持上游默认SB3 PPO MlpPolicy：actor/critic各64/64 tanh、45状态、12维归一化电机控制[-1,1]（MJCF齿比23/23/35转换为关节力矩）、50Hz；仅转换数值state_dict，不读取pickle配置，不恢复SB3优化器。

## 导入

复用仓库根.venv。转换需PyTorch，可安装训练包的import-policy extra；SB3只用于policy-oracle验证，不是训练依赖。将参数中的路径替换为自己的现有模型、固定源码和数据目录：

```bash
.venv/bin/python -m aiops_go1_training.upstream.convert \
  --archive /path/to/final_model.zip \
  --source /path/to/go1-locomotion-rl \
  --data-dir /home/liudi/.odyssey/go1 \
  --title 'Go1 基础行走' \
  --observation-protocol go1-sb3-observation/v1
```

最后一个参数声明该模型由固定上游原始观测训练、未使用VecNormalize。SB3 zip不包含外置归一化器，无法仅靠zip自动证明这点；不确定时不要声明兼容。自定义网络、gSDE、其他动作范围或归一化模型不支持。

导入创建新的bases/UUID/policy.npz与model.manifest.toml，不覆盖原模型。网页刷新后在模型页和训练的“基础模型”选择中出现。无场景+基础模型进行行走热启动；场景+基础模型训练冻结基础actor上的视觉残差；选择“从头训练”保持原流程。训练任务保存不可变基础副本和摘要，排队期间修改原文件不影响任务。

## 协议与行为

原始身体状态经兼容层缩放：线速度与目标命令乘[2,2,.25]，观测clip ±100；保留上游观测先于last_action更新的时序。基础模型分别保存actor/critic，不能套入旧共享主干参数。

热启动重建优化器，环境go1-command/v2借鉴上游奖励项，修正姿态健康判定和足部地面接触，明确不声称完整恢复上游环境/训练状态。输出可再次作为基础模型。

视觉go1-depth-residual/v2使用基础actor+CNN残差，残差actor零初始化；基础参数冻结，输出附带base/数值策略、清单和必要配置及摘要。原go1-depth/v1模型语义不变。CPU独立仿真支持这两类新协议，不执行梯度更新。

固定上游仓库models目录提供了已训练权重，其中2024-04-16_10-11-57.../final_model.zip为500万步力矩控制模型。合成模型数值一致、CPU短训练和回放只验证软件链路，不证明稳定行走、越障或真实机器人能力。需要实际训练模型及独立多种子场景评估。

当前默认选择已登记基础模型；历史从头训练产物移出模型库后，训练任务历史仍保留并显示模型已移除。上游500万步模型的平地能力已在本机CPU以独立种子验证10秒：前进约10.25米，未触发跌倒；该结果不代表拐弯或越障已通过。

## 公开位置控制策略

模型库的公开策略目录为 `~/.odyssey/go1/public-policies/<UUID>/`，每个条目包含 `public-policy.json` 和数值 `policy.npz`。仿真运行时校验来源和权重摘要，并对每项任务保存不可变快照。策略输出关节位置偏移，位置控制器在每个 MuJoCo 子步计算限幅电机控制量。录制包含策略实际输入、原始动作和该控制步首个电机指令。位置策略支持闭环仿真，开环动作差比较仍适用于原有力矩策略。

- Walk These Ways 固定上游提交 `0e7236bdc81ce855cbe3d70345a7899452bdeb1c`。导入入口为 `python -m aiops_go1_training.simulation.import_walk CHECKPOINTS SOURCE OUTPUT`。导入校验两个官方文件的摘要，并核对转换前后的推理。
- Robot Parkour Learning 固定上游提交 `5e1f4136f7fec9dac2741add63ca58d32090666f` 的 Go1 674000 步学生权重。导入入口为 `python -m aiops_go1_training.simulation.import_parkour ARCHIVE SOURCE OUTPUT`。模型使用自己的深度相机安装位姿、裁剪和确定性延迟，并按控制步更新 GRU 状态。标记为试验模型，仅用于空旷场地、低横杆和跳跃仿真。

本机 MuJoCo 诊断使用种子 7。Walk These Ways 在 0.3 m/s、±0.4 rad/s 指令下运行 5 秒，最终朝向分别约 +27.4°、−28.3°。转向方向正确，实际转向幅度低于该指令对应的理论 114.6°。Robot Parkour Learning 在本产品固定低横杆和跳跃场景中分别在 1.62 秒与 0.78 秒跌倒，两项均未通过。两套模型的仿真回放和 MCAP 深度已从本机 API 读取验证。

拐角仿真对宽度至少 0.8 m、转角不大于 90° 的 Walk These Ways 场景使用提前减速转向和出口纠偏。默认宽 0.8 m、距拐点 1.2 m 的 ±90° 用例，种子 7、8、9 均通过。跳跃仿真结果保存首次离地和首次触碰障碍的时刻、前进位置及触碰部位。Parkour 在固定跳跃用例仍未通过，保留原有判定与试验标识。
