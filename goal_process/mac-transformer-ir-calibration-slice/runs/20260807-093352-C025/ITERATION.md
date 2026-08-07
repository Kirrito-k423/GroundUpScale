# C025：临时卸载 board LaunchAgent

- **开始/结束：** 2026-08-07T09:33:52+08:00 / 2026-08-07T09:44:27+08:00
- **阶段：** PROBE
- **动作类型：** PROBE
- **关联验收/未知量：** H-06、AC-06、AC-07、AC-08

## 预注册

- **本轮 micro-goal：** 用 launchctl bootout 临时卸载两个精确 board LaunchAgent，使 watchdog/worker 在测量窗口不再复活，并保留 plist 供实验后 bootstrap 恢复。
- **当前假设：** LaunchAgent 是最终 lifecycle owner；bootout 后端口和 board CPU 竞争会保持消失。
- **已有证据：** C024；`launchctl print gui/502/<label>` 显示 state=running、keepalive、runatload 和权威 plist path。
- **证据等级：** E3。
- **唯一主要变量：** bootout `com.autoresearch.qwen35-24h-20260714` 与 `com.autoresearch.qwen3vl8b-mopd-24h-20260720`；不 disable、不删除/编辑 plist。
- **预期观察：** 两标签不再存在于 gui domain；watchdog/worker 退出；8765/8766 至少 30 秒无 listener；旧 18974 继续 Ts/0%。
- **判别规则：** bootout 只使用 exact label；操作后用 launchctl、ps、lsof 四重验证。若仍复活，停止并调查其他 owner；不扩大到宽泛 pkill。
- **成本与风险：** 约 2–10 分钟；board 暂不可用。恢复命令为对原 plist 执行 `launchctl bootstrap gui/502 <path>`，随后检查端口和 HTTP。
- **停止与回滚：** 任一 label/path 不匹配则不执行；正式实验未开始而需要终止时立即 bootstrap 恢复。

## 执行

- **脱敏命令：** `commands.md`
- **日志/指标：** 两个 exact LaunchAgent bootout 成功；watchdog/worker 退出且 30 秒无复活；18974 state=Ts/CPU 0；端口无 listener。Storage 自然空闲后最终 preflight PASS：normalized load `0.247754`、最大竞争进程 `12.4%`。

## 结果

- **观察事实：** 不删除/修改 plist 即可获得稳定安静窗口；board 生命周期 owner 已正确控制。中间两次 preflight 因 Storage/Codex 短峰失败并原样保留，随后独立窗口全部检查通过。
- **错误签名：** 无；最终 `eligible=true`。
- **推断：** H-06 的外部阻塞已解除，可以按 E2 启动受控 MPS 实验；每个 Run 仍必须自己通过 preflight 与 3% noise gate。
- **证据等级变化：** 环境恢复路径 E1→E3。
- **信息增量：** 建立了可逆、精确、可复现的 board 停止/恢复方法。

## 结论

- **验收/交付更新：** Goal 可从 BLOCKED 恢复到 M5 EXPERIMENT。
- **预算变化：** 约 10 分钟等待，无付费资源。
- **下一 micro-goal：** 全新 MPS fit/holdout cohort。
- **是否需决策：** 无；实验结束必须 bootstrap 恢复两个 LaunchAgent。
