# C024：停止 watchdog-owned board 服务拓扑

- **开始/结束：** 2026-08-07T09:29:53+08:00 / 2026-08-07T09:33:52+08:00
- **阶段：** PROBE
- **动作类型：** PROBE
- **关联验收/未知量：** H-06、AC-06、AC-07、AC-08

## 预注册

- **本轮 micro-goal：** 先正常停止精确匹配端口/data-root 的 watchdog 50158/50161，再停止其当前 worker，使 8765/8766 在整个测量窗口保持关闭。
- **当前假设：** watchdog 是 C022 worker 自动复活的唯一 lifecycle owner；停止它们后 worker 不再重启，旧 18974 可用可逆 SIGSTOP 消除残留 CPU。
- **已有证据：** C023 的 PID/PPID、启动时间、命令和 listener 对齐。
- **证据等级：** E3。
- **唯一主要变量：** 临时停止这两个 board 服务拓扑；不操作其他 autoresearch、Docker、训练或 macOS 进程。
- **预期观察：** watchdog 与 64289/64290 退出，8765/8766 无 listener 且 15 秒后不复活；18974 若仍存活则 SIGSTOP 后 state=T、CPU=0。
- **判别规则：** 所有目标在信号前再次做完整命令匹配；先 TERM watchdog，观察子进程；残留 worker TERM；18974 仅 STOP、不 KILL。任何新 owner 出现则停止并重新调查。
- **成本与风险：** 约 2 分钟；两个 board 暂不可用。原 watchdog/worker 完整命令已记录，实验结束后恢复 watchdog 并以端口/HTTP 健康检查验收。
- **停止与回滚：** 不使用 SIGKILL；超出已识别拓扑不操作；测量未启动或结束后立即恢复服务。

## 执行

- **脱敏命令：** `commands.md`
- **日志/指标：** watchdog 50158/50161 与 worker 64289/64290 经 TERM 退出；旧 18974 SIGSTOP 后 state=Ts/CPU 0。约 2 分钟后 launchd 创建新 watchdog 64812/64814 和 worker 64816/64817，端口重新 LISTEN；preflight top=64816 96%、load=0.417。

## 结果

- **观察事实：** watchdog 不是最终 owner；两个用户 LaunchAgent 的 `KeepAlive|RunAtLoad` 负责重建 watchdog。精确标签与 plist 路径均已由 `launchctl print` 解析。
- **错误签名：** `launchd-keepalive + com.autoresearch.qwen35-24h-20260714/com.autoresearch.qwen3vl8b-mopd-24h-20260720 + board topology restart`。
- **推断：** 必须临时 bootout 精确 LaunchAgent；继续 TERM 子进程会无效循环。
- **证据等级变化：** lifecycle owner 从 watchdog 上溯到 launchd，E3。
- **信息增量：** 获得可逆且作用域最小的真正停止/恢复 seam：bootout/bootstrap 两个 plist。

## 结论

- **验收/交付更新：** 未启动 benchmark；旧 18974 已安全暂停。
- **预算变化：** 约 4 分钟，无付费资源。
- **下一 micro-goal：** bootout 两个精确 LaunchAgent，验证 30 秒不复活并让 preflight 回落。
- **是否需决策：** 已获临时停止 board 服务授权；恢复路径明确。
