# C019：公共 CI、运行手册与干净 checkout 审计

- **开始/结束：** 2026-08-06T20:02:18+08:00 / 进行中
- **阶段：** M6 / CI AND DELIVERY
- **动作类型：** IMPLEMENT + VERIFY
- **关联验收/未知量：** AC-01、AC-11、AC-12、D-01、D-05、D-07

## 预注册

- **本轮 micro-goal：** 完成不依赖 5% gate 变更的全部发布交付：Linux 公共确定性 CI、受信任本地 M4 lane、中文运行手册、README、clean-checkout 复现和远端 main 审计。
- **当前假设：** MPS-only 测试正确 skip 后，锁定环境和其余测试可在 GitHub-hosted Linux/Python 3.11 通过；公共 workflow 不需要个人 Mac 或 secrets。
- **已有证据：** 本地 33 tests；M1–M5 code/evidence；官方 actions/checkout 与 setup-python 当前主版本均为 v6。
- **证据等级：** 目标 E2。
- **唯一主要变量：** 从本地实现进入公共 Linux CI 和干净 checkout 验证；不修改模型、测量或 calibration gate。
- **预期观察：** 本地全量 GREEN；workflow security test PASS；clean checkout `uv sync --locked` + tests + double compile deterministic；GitHub Actions GREEN；remote main 与本地验证 SHA 一致。
- **判别规则：** 公共 CI 不运行真实 hardware benchmark；MPS 只在 trusted local lane；CI 失败按日志修复，不禁用测试。
- **成本与风险：** 预计 5–10 分钟，无付费 runner 或外部服务新增费用。
- **停止与回滚：** 若 GitHub 公共 runner 的依赖安装失败，保留日志并修锁文件/workflow；禁止注册个人 Mac self-hosted runner。

## 执行

- **脱敏命令：** `commands.md`
- **代码差异：** workflow、本地脚本、runbook、README、CI security test 已起草。
- **日志/指标：** 待验证。

## 结果

## 结论
