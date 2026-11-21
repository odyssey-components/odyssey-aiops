# AIOps · 本地训练实验室

按课程保留每一步训练方案：**第一课熟悉固定路线，第二课观察完整随机地图，第三课局部观察并记录探索过程，第四课用GRU学习历史记忆**。前两课用最短路径专家、第三和第四课用只读取已知地图的探索专家生成答案，通过模仿学习训练；不接入集群或训练服务，不覆盖之前的课程。

单用户账号和部署配置见 [AIOps 单用户登录](docs/access-control.md)。

## 启动服务与前端

在仓库根目录执行：

```bash
git submodule update --init --recursive
bun install --frozen-lockfile
# 首次构建并安装训练包
bun scripts/build.ts --wheels-only --output tmp/aiops-wheels
bun scripts/setup-python.ts --release tmp/aiops-wheels
# 终端一：Go API
cd server
go run ./cmd/aiops-server
# 终端二，在仓库根目录运行
bun --cwd web dev
```

打开 [AIOps 本地训练实验室](http://127.0.0.1:4175)，默认进入 Go1 训练任务。需要使用 Maze 课程时，在启动或构建前端时设置 `VITE_AIOPS_SHOW_MAZE=true`。启用后，前端使用 Vite、TanStack Router SPA 和 Odyssey UI；参考 OdyLab 复用 ProAdminShell 的一级「迷宫」菜单与二级课时菜单，操作页保留实验与训练控件，课程说明放在文档中，小屏使用共享抽屉菜单。种子输入、模型状态、提示和信息面板也使用共享组件。服务只监听回环地址，默认模型列表来自 `~/.odyssey/odymaze/runs/`。

开发时 Vite 将 `/api` 转发到 `127.0.0.1:4176`。Go 只提供 API，根路径返回 404。生产页面由 Nginx 提供，见 [部署配置](deploy/nginx.conf)。完整发行包：`bun scripts/build.ts --output tmp/aiops-release`。安装可用 `--venv` 指定环境，Go 使用 `--python /绝对路径/bin/python`；训练期间不需要 uv 或仓库源码。

先选课程，试试手动走、专家单步和自动运行。第二至四课可以输入地图种子，再点「生成新迷宫」。同一个种子在训练端和网页生成相同地图。课时、知识点、种子、模式、速度和可恢复的默认目录模型会同步到 URL，支持刷新和前进/后退；刷新保持暂停。模型目录固定为 `~/.odyssey/odymaze/runs/`；进入课时自动加载本课有效列表的第一个模型（目录名倒序），刷新列表后采用新的首项；保留模型下拉选择且无占位选项，没有目录选择、文件导入或额外加载按钮。

## 训练四课

请在另一个终端回到仓库根目录执行训练。训练包使用 Python 3.14。独立检出使用 AIOps 根目录的 `.venv`，在 Odyssey 联调时使用 Odyssey 根目录的 `.venv`。

- 按上面的构建和安装命令安装 Maze。Go1 使用同一环境并运行 `bun scripts/setup-python.ts --release tmp/aiops-wheels --project go1`。
- 共享依赖来自固定的 [Odyssey Python](https://github.com/odyssey-components/odyssey-python) 提交，Go API 使用版本化的 [Odyssey Go](https://github.com/odyssey-components/odyssey-go) 模块。

```bash
# 第一课：仍是原来的默认命令，400 轮
uv run --project . --no-sync aiops-maze-train

# 第二课：随机地图，默认 256 张训练图、160 轮
uv run --project . --no-sync aiops-maze-train --lesson random-map-v1

# 第三课：局部探索与程序记忆，默认128张训练图、120轮
uv run --project . --no-sync aiops-maze-train --lesson local-map-v1

# 第四课：模型记忆，默认256张训练图、160轮
uv run --project . --no-sync aiops-maze-train --lesson memory-map-v1
```

都使用 CPU。输出目录默认 `~/.odyssey/odymaze/runs/<时间>/`，不会覆盖非空目录：

| 产物 | 内容 |
| --- | --- |
| `model.onnx` | 标准推理网络和已选中的训练权重 |
| `model.manifest.toml` | ONNX输入输出、课程协议、SHA-256和训练摘要 |
| `model.json` | 可读教学参数，供离线讲解与转换 |
| `samples.json` | 专家习题与答案；第二课包含每条样本对应的地图身份 |
| `report.json` | 逐轮损失/正确率、初始与最终实际评估；第二课还保存数据划分、验证记录和选中轮次 |

训练完成后刷新网页模型列表，自动使用本课列表中的第一个有效模型。没有 `lesson` 字段的旧 v1 JSON转换时归第一课，前端只列出有效的ONNX与清单文件对。目录固定，不提供目录选择或单文件导入；自定义训练输出需放入固定目录的独立 run 子目录后才能由网页自动加载。

## ONNX导出

四课训练默认生成ONNX与配套清单，网页通过自托管ONNX Runtime WASM直接执行模型。已有模型无需重新训练：

```bash
uv run --project . --no-sync aiops-maze-export --input "$HOME/.odyssey/odymaze/runs/<run>/model.json"
```

转换在同目录新增 `model.onnx` 与 `model.manifest.toml`，不覆盖原文件。第四课导出显式的hidden输入/next_hidden输出；详见[ONNX使用说明](docs/onnx-models.md)。

## 做学习对照

```bash
# 第一课：完全未训练的网络
uv run --project . --no-sync aiops-maze-train --epochs 0

# 第二课：小规模、零轮对照
uv run --project . --no-sync aiops-maze-train --lesson random-map-v1 --maps 16 --epochs 0

# 查看参数
uv run --project . --no-sync aiops-maze-train --help
```

每次从初始网络开始，轮数不同是同一课的不同实验，不会创建新适配器，也不会自动续训。

## 课程区别与实测边界

| | 第一课 | 第二课 |
| --- | --- | --- |
| 课程 | `fixed-map-v1` | `random-map-v1` |
| 输入 | 位置与朝向，18 个数 | 整图墙/出口/角色及朝向，151 个数 |
| 网络 | 18 → 32 → 3 | 151 → 128 → 3 |
| 数据 | 一张图、68 条专家样本 | 多张随机可达图，训练/验证/测试内容互不重复 |
| 前端 | 原固定地图 | 按种子换图，继续使用同一个第二课模型 |

默认种子 7 的本地 CPU 实测：第一课 400 轮后 12 步通关；第二课使用 256 张训练图、64 张验证图，训练 160 轮，由验证集选中第 110 轮，在 100 张独立测试图上 **63 张通关**，未训练模型为 0 张。此结果只属于该次配置和地图生成规则，不能保证任意地图通关。模型推理不会调用专家，绕圈时会在 100 步停止。

第二课仍限定 7×7、固定起终点，并保证存在一条向东/向南的通路。训练集正确率和未知地图通关率是两个指标；网页和报告分开说明。

第三课使用3×3局部视野和程序保存的已知地图/访问记录，249 → 64 → 3，200动作上限；默认实训在100张独立测试地图上53张通关，未训练0张。模型没有规划器兜底，可能绕圈或超时。详见[第三课教程](docs/local-exploration.md)。

第四课只输入当前视野、位置朝向和上一动作，共40维；GRU保留32维隐藏状态。默认实训独立100图91张通关；同一模型每步清空隐藏状态为9张。详见[第四课教程](docs/model-memory.md)，不要把跨课成绩差异全部归因于记忆。

## 学习源码

- [第一课的五个学习点](docs/learning.md)：保留原来的完整解释。
- [两课比较与第二课教程](docs/courses.md)：输入为什么变化、随机生成和独立评估。
- [第一课代码](training/maze/src/aiops_maze_training/lessons/fixed_map_v1/)：原来的环境、模型和训练。
- [第二课代码](training/maze/src/aiops_maze_training/lessons/random_map_v1/)：随机环境、模型和训练。
- [第三课教程](docs/local-exploration.md)与[独立训练代码](training/maze/src/aiops_maze_training/lessons/local_map_v1/)：局部视野、程序记忆与真实探索评估。
- [第四课教程](docs/model-memory.md)与[独立训练代码](training/maze/src/aiops_maze_training/lessons/memory_map_v1/)：序列训练、GRU和隐藏状态对照。
- [前端课程适配器](web/src/lessons/)；[统一推理入口](web/core.ts)。

第一课环境、模型和训练统一位于 `aiops_maze_training.lessons.fixed_map_v1`。前端只接受已支持课程的纯数据 JSON，未知身份或不兼容输入不会被默认为第一课。

## 验证

```bash
uv run --project . --no-sync python -m unittest discover -s training/maze/tests -v
bun test ./web/tests/
bun web/tests/onnx-journey.mjs
bun web/tests/training-journey.mjs
bun check-python
bun run check-lines
```

前端目录中另外运行 `bun run typecheck` 和 `bun run build`，再从仓库根运行上述生产预览 Journey。

Journey 使用独立 HOME、真实训练模型与受控 Chromium headless；验证四课 ONNX WASM、两端概率一致、加载拒绝、逐轮曲线、训练取消和生命周期。只生成合成数据，不操作真实模型或集群。构建/模型/日志不提交到 Git。

设计与验收依据：[课程适配](../../specs/2515-odymaze-course-adapters/spec.md)与[Go 服务迁移](../../specs/2517-odymaze-go-local-service/spec.md)。

## 网页训练与地图组

四课都可在「地图组」tab 中查看已有地图组，点击右上角「添加地图组」弹出表单并预生成，再到「训练」tab 启动 Python 后台训练，查看日志或取消任务。每组全部地图保存为同一个 JSONL，并配同名 TOML，记录名称、说明、日期、课时、规格与 SHA-256；训练读取已准备地图，不在轮次中生成地图。操作、格式和本地边界见 [网页训练](docs/web-training.md)。

## 前端源码结构

`web/src/views/` 编排实验和训练页面；`components/` 按 common、layout、maze、models、training 拆分控件；`apis/` 集中模型与训练的浏览器请求；`utilities/` 保存状态、协议校验、选择器、图表计算和推理辅助。`routes/` 保留 TanStack 路由适配，产品 `server/` 提供 Go API，`lessons/` 保留独立课程实现。页面和组件通过 API 模块访问后台。

## 训练曲线

训练页显示逐轮真实通关率折线、最新/最佳/最佳轮次与初始差值，支持按轮次查看和CSV导出。新任务每轮实际运行评估；第一课为固定图，其他课为64张验证图。旧日志缺失轮次不补造，详见[训练曲线说明](docs/training-curves.md)。

## Python 包结构

- `training/maze`：`aiops-maze-training` / `aiops_maze_training`，四个迷宫课时。
- `training/go1`：`aiops-go1-training` / `aiops_go1_training`，Go1 MJX PPO。
- `repositories/odyssey-python/python/training`：现有共享 `odylab-training` 包，提供通用算法、事件与进程生命周期；不依赖 Maze 或 Go1 包。Go1统一使用 `aiops_go1_training`，已移除旧包路径。

本地开发默认使用仓库根 `.venv`，Maze 与 Go1 共用解释器，训练包仍独立发布。默认安装只安装 Maze 及共享依赖；安装Go1包使用 `bun scripts/setup-python.ts --release tmp/aiops-release --project go1`，同样默认安装到根 `.venv`；仅显式指定其他环境时配置 `--go1-python`。实际 GPU 训练另需安装该包的 `gpu` extra 与固定模型源码；安装 Maze 不会安装 CUDA/JAX。

旧根模块导入已迁移至子包；`odymaze-train` / `odymaze-export` CLI别名继续兼容。模型清单、数据格式、浏览器主题存储 key 与 `~/.odyssey/odymaze` 数据目录保留，避免改名后看不到已有数据。新的服务配置名称为 `aiops.yaml` / `AIOPS_*`。

Go1 使用 PostgreSQL 的 `aiops_go1` schema 和 Silo 的 `aiops-go1` 桶。参考
[配置样例](server/aiops.example.yaml)，把实际配置保存为 `aiops.yaml`，使用
`CONFIG_DIR=/home/liudi/devtools/aiops` 或 `--config-dir` 指向其目录。
数据库账号需要在所选数据库中创建或写入专用 schema 的权限，Silo 凭据需要读写专用桶。
配置文件保存凭据时应限制为当前用户可读。

Go1 的任务和场景从数据库恢复，完成后的权重、录制和回放从 Silo 恢复到进程专用工作目录。
Go1 基础模型从 PostgreSQL 目录和 Silo 对象恢复。Python 工作进程使用从 Silo 准备的临时文件。
每个数据库只运行一个 Go1 API 实例。

Go1 目录采用独立实体表和外键。场景参数、通过条件、集合成员、
训练参数、训练指标、仿真用例和 Silo 对象索引都有明细表。
新表不含 JSON 列，工作进程专用的任务快照存 Silo。
新产物写入 `go1/artifacts/`，新任务快照写入 `go1/snapshots/`。
读取时使用 PostgreSQL 记录的完整对象键，现有对象无需迁移。

## Dell 集群部署

Traefik 入口为 `/aiops/`，访问由 AIOps 账号会话控制。Argo CD 管理 Nginx、
仅包含 Go 的 API、Workflow 模板和各自的服务账号。API 容器设定
`AIOPS_EXECUTION=workflow` 后，Maze 和 Go1 训练由 `aiops-train-v1` 执行，
Go1 仿真由 `aiops-simulate-v1` 执行。Worker 从 Silo 读取输入和上传产物，
Go API 验证成功结果后才发布模型和回放。首次使用的 Worker 镜像较大，
拉取会比后续任务慢。API 重启会终止中断的执行；可在任务列表中重新提交。

本地默认使用 `local` 执行方式，通过仓库根 `.venv` 运行训练与仿真。
集群发布用 `bash deploy/kubernetes/build-images.sh`
构建三张镜像，GitOps 清单位于 `argocd-apps/dell/aiops`。
发布新版 Worker 镜像后，在 `aiops-train-v1` 和 `aiops-simulate-v1` 中同时更新镜像摘要、
将 `command` 改为 `[/usr/local/bin/aiops-worker]`，并将 `args` 改为
`["{{workflow.parameters.task-id}}"]`，然后同步 Argo CD。
