# C003：验证 MPS 稳定态 warmup

- **开始/结束：** 2026-08-06T17:40:11+08:00 / 2026-08-06T17:42:01+08:00
- **阶段：** PROBE
- **动作类型：** MEASURE
- **关联验收/未知量：** AC-06、AC-07、H-02、U-01

## 预注册

- **本轮 micro-goal：** 判定足量 warmup 后 CPU/MPS 的稳定态 `IQR/median` 是否均不超过 3%。
- **当前假设：** C002 的 MPS 超限来自 5 个单组 warmup 不足；后 12 个样本已显示约 282 μs 稳定平台。
- **已有证据：** C002 CPU 1.855%；MPS 7.288%，但 MPS 后段收敛。
- **证据等级：** H-02 E1；CPU 子命题 E2，MPS 子命题 E1。
- **唯一主要变量：** warmup 从 5 个单操作组提高到 500 个；`inner_iterations=50`、repeats=20、Shape、dtype、seed、设备和同步边界不变。
- **预期观察：** CPU/MPS `IQR/median <=3%`；MPS 首个 timed sample 不再经历大幅单调下降。
- **判别规则：** 两设备均通过则 H-02 升 E2，并将“足量 warmup + 扩大同步窗口”写入后续 Benchmark 协议；失败则保留证据并评估分设备 warmup/热状态控制。
- **成本与风险：** 预计小于 2 分钟；无外部费用。
- **停止与回滚：** 单命令 20 分钟停止；不得删除失败样本或只挑稳定子集过门禁。

## 执行

- **脱敏命令：** `commands.md`
- **配置/环境差异：** 仅 warmup=500。
- **代码差异：** 无。
- **日志/指标：** CPU median `465,935.42 ns`、IQR/median `3.335%`；MPS median `269,820.83 ns`、IQR/median `0.650%`。

## 结果

- **观察事实：** MPS 首个 timed sample 已处于稳定平台，噪声由 C002 的 7.288% 降至 0.650%。CPU 有两个完整 window 调度尖峰（约 628.5 μs、598.3 μs），整体 IQR/median 略超 3%。PyTorch 当前默认 intra-op threads=4、interop threads=10。
- **错误签名：** 无。
- **推断：** MPS warmup 假设成立。CPU 的一个 timed window 不能稳健代表一次 benchmark sample；需要显式保留多个原始 window 并以组内 median 形成 sample，最终门禁仍看 sample 间 IQR，不能删除尖峰。
- **证据等级变化：** H-02 保持 E1；MPS 稳定态子命题升至 E2。
- **信息增量：** 确立 MPS 至少 500 个同类操作组 warmup；识别 CPU 调度尖峰对单窗口样本的影响。

## 结论

- **验收/交付更新：** MPS benchmark 协议新增足量 warmup 约束。
- **预算变化：** 无。
- **下一 micro-goal：** C004 为每个统计 sample 保留并聚合 5 个 timed window，以组内 median 抵抗单窗口调度尖峰；其他配置保持 C003。
- **是否需决策：** 当前无。
