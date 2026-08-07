# C023：可逆暂停残留 board 进程

- **开始/结束：** 2026-08-07T09:28:19+08:00 / 2026-08-07T09:29:53+08:00
- **阶段：** PROBE
- **动作类型：** PROBE
- **关联验收/未知量：** H-06、AC-06、AC-07、AC-08

## 预注册

- **本轮 micro-goal：** 用 SIGSTOP 可逆暂停已关闭 listener 但未退出的 PID 18974，确认其 CPU 归零，并等待负载达到 preflight 门槛。
- **当前假设：** 18974 的残留后台循环是当前唯一已识别的 board 竞争源；SIGSTOP 后它不再消耗 CPU，SIGCONT 可恢复。
- **已有证据：** C022：18975 正常退出；18974 listener 已关闭但进程仍间歇使用 25.6% CPU。
- **证据等级：** E2。
- **唯一主要变量：** 向精确 PID 18974 发送 SIGSTOP；不终止或暂停其他进程。
- **预期观察：** 进程 state 含 `T`、CPU 趋近 0，端口继续无 listener；1-minute load 随时间回落。
- **判别规则：** SIGSTOP 后先核对 state/端口，再运行 preflight；只有所有 check PASS 才恢复 Goal 实验。失败时按新 top competitor 定位，不调整 policy。
- **成本与风险：** 1–10 分钟；18974 保持内存但不运行。恢复命令是 SIGCONT，进程可能因 8766 对端已退出而需要后续重启。
- **停止与回滚：** 若 PID 已变化则不发信号；若暂停成功但本轮不能测量，发送 SIGCONT 或按用户授权恢复服务。

## 执行

- **脱敏命令：** `commands.md`
- **日志/指标：** 信号前复核发现 8765 已由新 PID 64290 监听、8766 由 64289 监听；两者父进程分别为长期 watchdog 50158/50161。未向旧 PID 发送 SIGSTOP。

## 结果

- **观察事实：** watchdog 每 10 秒检查并重启 worker；C022 的正常停止触发了新 worker，因此“只有一个残留进程”假设不成立。
- **错误签名：** `service-auto-restart + watchdog 50158/50161 + board listeners 8765/8766`。
- **推断：** 必须按服务拓扑先停止精确 watchdog，随后处理 worker；只操作 worker 不能维持测量窗口。
- **证据等级变化：** watchdog 根因 E0→E3（父子关系、命令、端口和重启时间直接对齐）。
- **信息增量：** 识别了服务真正的生命周期 owner，避免暂停陈旧 PID。

## 结论

- **验收/交付更新：** 未启动 benchmark，安全边界保持。
- **预算变化：** 约 2 分钟，无付费资源。
- **下一 micro-goal：** 精确停止两个 watchdog-owned board 拓扑，再验证 preflight。
- **是否需决策：** 用户对临时停止 board 服务的授权覆盖其 lifecycle owner；不处理其他 watchdog/服务。
