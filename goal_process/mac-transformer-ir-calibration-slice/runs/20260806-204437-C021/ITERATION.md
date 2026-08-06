# C021：第三次受控环境阻塞审计

- **开始/结束：** 2026-08-06T20:44:37+08:00 / 2026-08-06T20:46:06+08:00
- **阶段：** M5 / BLOCKED AUDIT
- **动作类型：** PROBE
- **关联验收/未知量：** AC-06、AC-07、AC-08、AC-10、AC-12、H-06

## 预注册

- **本轮 micro-goal：** 用已冻结的 `local-apple-silicon-v1` policy 判定当前 Mac 是否已具备启动全新 fit/holdout cohort 的资格，并确认前两轮外部负载阻塞是否仍然存在。
- **当前假设：** 两个持续运行的 autoresearch board 服务和 macOS 媒体分析仍会使 normalized load 或单进程 CPU 超过 C020 预注册门槛。
- **已有证据：** C020 两次真实 preflight 均因 `load-above-policy` 与 `competing-process-above-policy` 拒绝；只读 `ps` 显示两个 board 服务已运行约 13 天。
- **证据等级：** E3（阻塞重复性）；对“环境合格后能获得 5 个有效 holdout”仍为 E1。
- **唯一主要变量：** 无；只读复核当前外部状态，不修改进程、policy 或 benchmark。
- **预期观察：** 若 preflight PASS，则立即开始全新 cohort；若同样 reason codes 再现且长期服务仍活跃，则满足第三个连续 Goal 回合的同一外部阻塞条件。
- **判别规则：** `eligible=true` 才允许 EXPERIMENT；`eligible=false` 时禁止采样。连续第三回合同一阻塞且需用户暂停/授权停止进程时，按 Goal 合同进入 BLOCKED 并形成 HANDOVER。
- **成本与风险：** 约 1 分钟，只读；不采集命令参数进入 Run Bundle，不终止或调度用户进程。
- **停止与回滚：** 同一错误签名不再无信息重跑；环境失败后只归档一次并升级。

## 执行

- **脱敏命令：** `commands.md`
- **日志/指标：** preflight exit 2；normalized load `0.407666>0.25`；最大竞争进程 `mediaanalysisd=89.4% CPU`；两个长期 board 服务在采样窗口内均为 `55.6% CPU`。

## 结果

- **观察事实：** Darwin/arm64、AC power、thermal nominal 继续 PASS；`load-above-policy` 与 `competing-process-above-policy` 第三回合重复。PID 18974/18975 已运行约 13 天 2 小时，PID 78312 已运行约 3 小时 34 分钟。
- **错误签名：** `EnvironmentValidityError + preflight + load-above-policy,competing-process-above-policy`。
- **推断：** 当前无法在不停止/影响用户长期服务的情况下建立合格 hardware cohort；继续抽样会违反 C020 policy 和 Goal 的 3% 前置门禁。
- **证据等级变化：** 外部环境阻塞 E2→E3（三个连续 Goal 回合重复）；环境合格后的 holdout 成功率仍未知。
- **信息增量：** 已满足 Goal 的严格 BLOCKED 判据，并确认阻塞不是短暂 GitHub CI/本轮测试负载。

## 结论

- **验收/交付更新：** AC-06/07/08 保持 BLOCKED，AC-10/12 连带未完成；新增 `HANDOVER.md`。
- **预算变化：** 本轮无正式 benchmark、无付费资源；只读探针约 5 秒。
- **下一 micro-goal：** 用户暂停 board 服务或授权临时停止后，等待 `groundupscale preflight` PASS 并恢复全新 cohort。
- **是否需决策：** 是。推荐临时停止 PID 18974/18975，并等待 `mediaanalysisd` 自然降载；未经授权不执行。
