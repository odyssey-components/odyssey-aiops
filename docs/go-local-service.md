# Go 本地服务

开发链路为浏览器 → Vite:4175 → Go API:4176 → 已安装的 Python 包。生产链路为浏览器 → Nginx 静态 SPA / API 代理 → Go → Python。Go 不提供页面、静态文件或前端 fallback。

从仓库根构建发行包：

```bash
bun scripts/build.ts --output tmp/aiops-release
bun scripts/setup-python.ts --release tmp/aiops-release
```

发行目录包含 Go 可执行文件、web 静态目录、Nginx 配置样例、三份 wheel 和摘要清单。固定地图从唯一输入 `web/maze.json` 复制进 wheel 暂存区；不依赖运行时源码相对路径。安装阶段使用 uv，默认目标为仓库根 `.venv`；运行阶段直接调用 Python。独立发行安装使用显式 `--venv`，从仓库外启动Go时使用 `--python` 指定该环境。`--offline` 要求已有完整构建/依赖缓存；可用 `--torch-index` 指定安装源。

服务使用共享 `bootstrap.App` / `bootstrap.Main` 加载配置、处理退出信号并关闭训练进程。支持 `aiops.yaml`、`AIOPS_LISTEN` / `AIOPS_PYTHON` 等环境变量与现有命令行参数（命令行优先）；使用 `--help` 查看完整配置选项。Vite 仍须单独启动。

启动发行包的 `aiops-server`，默认监听 `127.0.0.1:4176`，默认 Python 为仓库根 `.venv/bin/python`，从启动工作目录向上识别 `go.work` 与 `pyproject.toml`。可设置 `--listen`、`--data-dir`、`--python`；Go1 默认复用同一个解释器，可用 `--go1-python`、`--go1-source` 配置。这些是管理员启动参数，浏览器不能提交命令或目录。

开发页面在 `web` 运行 `bun dev`；Vite 代理保留 Host。`AIOPS_API_ORIGIN`（兼容旧 `ODYMAZE_API_ORIGIN`） 可覆盖回环 API 地址。`bun preview` 是静态预览，只有显式设置该变量时才代理 API。

Nginx 使用 [配置样例](../deploy/nginx.conf)，将 root 改为发行包 web 目录，`/api/` 转发至 Go，其余路径使用 SPA fallback。样例只绑定回环，保持本阶段的本地信任边界；公网认证、TLS、集群执行器不在本次实现中。生产页面不依赖 Node。

Maze 与 Go1 使用同一 FIFO 执行器，同一时刻只运行一个子进程。取消或超时会终止进程组；服务重启将未完成任务标记失败，不自动重新训练。完整产物通过校验后才发布。原数据目录不迁移、不清理。

可运行 `bun web/tests/installed-wheel-training.mjs` 验证独立安装环境中的四课训练；该验证使用根目录 tmp 下的合成数据。GPU Go1 训练需另备 CUDA/JAX 环境。

Go1安装默认包含CPU依赖，`setup-python.ts --project go1 --device cpu`；NVIDIA可选择`--device cuda`安装CUDA extra，ROCm插件按AMD兼容文档单独配置。服务不在训练请求中安装依赖。
