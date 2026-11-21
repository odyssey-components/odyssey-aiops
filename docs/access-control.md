# AIOps 单用户登录

在 `aiops.yaml` 中配置唯一账号，用户名和密码均为必填项。

```yaml
account:
  username: admin
  password: "替换为自己的密码"
```

本地用 `go run .` 启动，打开 AIOps 后使用共享登录页。Go API 在处理业务请求前校验登录会话。退出登录、服务重启或会话到期后需要重新登录。健康探针可独立使用。

账号配置文件应仅对当前用户可读，并保留在 Git 之外。集群通过 Kubernetes Secret 挂载配置文件，并用 `--config-dir` 指定目录。集群入口使用 HTTPS，保留 Nginx 的来源检查和 Go API 的回环监听。Workflow 模式使用 Secure 会话 Cookie。
