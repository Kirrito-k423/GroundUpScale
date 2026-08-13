# C004：调度纠偏与无效远端运行隔离

- **阶段：** VERIFY
- **动作类型：** PROBE / ROLLBACK（进程级）

## 观察事实

- #44 同一 worktree 同时启动两个完整 pytest；保留较早进程并终止后启动重复进程。
- #46 同一 worktree 同时启动两个完整 pytest；保留较早进程并终止后启动重复进程。
- #45 通过 SSH 直接运行带 `ASCEND_RT_VISIBLE_DEVICES=0` 的 focused pytest，未由公共 wrapper 持锁。
- 主 Agent 中断 #45 Agent并终止远端相关 pytest；随后 NPU 0 无运行进程、公共 owner 不存在。

## 判定

- #44/#46 重复全套属于无新增证据重跑，已按失败去重纪律停止。
- #45 绕过 wrapper 的运行及其产物全部 invalid/non-authoritative，禁止进入权威 Run Bundle；不把它视为 issue 实现失败。
- #45 恢复同一上下文后，任何可能初始化 torch_npu 的 pytest 与 measurement/holdout 均须让 wrapper 覆盖完整 session。

## 下一步

- 等待 #44–#46 完成合法阶段验证；审计每个 NPU bundle 的 lock metadata。
