export const RANDOM_LESSONS = [
  {title:'认识环境',content:<><p>第二课使用 7 × 7 随机道路。角色从 (1, 1) 朝东出发，出口固定为 (5, 5)，100 步结束。生成器保证存在通路。</p><div className="experiment"><b>试一试</b> 改变种子再生成迷宫；同一种子总会得到同一张地图。</div></>},
  {title:'跟随专家',content:<><p>专家读取整张地图，用最短路径给每个可达位置和朝向标注动作。</p><div className="experiment"><b>试一试</b> 切换专家，观察当前地图中的转向与移动。不同地图需要不同的动作。</div><p>训练收集多张地图的专家样本；验证与测试使用独立且内容不重复的地图。</p></>},
  {title:'理解模型',content:<><p>输入 151 个数：墙、出口、角色各 49 个格子，再加朝向 4 个数，按平面逐行排列。</p><div className="experiment"><b>观察输入</b> 换地图时墙平面改变，角色移动时角色平面改变。</div><p>151 → 128 个 tanh 隐藏单元 → 3 个动作。模型预测不调用专家、不读取最短路径距离。</p></>},
  {title:'开始训练',content:<><p>用随机地图专家样本训练，再以独立验证集选择模型。</p><pre><code>uv run --project . --no-sync odymaze-train --lesson random-map-v1</code></pre><div className="experiment"><b>观察报告</b> 同时看损失、样本正确率、验证集和测试集的实际通关率。测试集不用于调参。</div></>},
  {title:'检验结果',content:<><p>切换到第二课会自动加载本课的第一个有效模型。同一个模型可在新生成的地图上继续推理。</p><div className="experiment"><b>做个对照</b> 使用 --epochs 0 训练未学习基线。比较多个新种子的成功、超时和轨迹，不能只看训练准确率。</div><p>随机道路训练不能保证解决所有陌生迷宫；尺寸、起点和终点仍然固定。</p></>},
];
