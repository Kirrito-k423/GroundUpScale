# C004：用组内 median 构造稳健统计 sample

- **开始/结束：** 2026-08-06T17:42:01+08:00 / 2026-08-06T17:44:37+08:00
- **阶段：** PROBE
- **动作类型：** MEASURE
- **关联验收/未知量：** AC-06、AC-07、H-02、U-01

## 预注册

- **本轮 micro-goal：** 验证每个统计 sample 由 5 个原始 timed window 的 median 构成时，CPU/MPS sample 间 `IQR/median` 是否均不超过 3%。
- **当前假设：** C003 CPU 超限来自偶发完整 window 调度尖峰；保留原始值并使用组内 median 能给出稳健而非删点的 run-level 代表值。
- **已有证据：** CPU 单窗口 C002 1.855%、C003 3.335%；MPS 稳定态 C003 0.650%。
- **证据等级：** H-02 E1；分设备子命题已有局部 E2。
- **唯一主要变量：** 每个统计 sample 的 timed windows 从 1 增加到 5，sample 值为这 5 个 window 的 median；warmup=500、inner_iterations=50、samples=20 及其他条件不变。
- **预期观察：** CPU/MPS sample 间 `IQR/median <=3%`；原始 window 继续完整输出。
- **判别规则：** 两设备均通过则 H-02 升 E2，M1 采用该稳健聚合协议；失败则按残余趋势决定是否冻结 CPU 线程口径。
- **成本与风险：** 预计小于 5 分钟；无外部费用；不删除或 winsorize 原始值。
- **停止与回滚：** 单命令 20 分钟停止；不得以挑选子集替代预注册聚合。

## 执行

- **脱敏命令：** `commands.md`
- **配置/环境差异：** 每个 sample 使用 5 个 window 的 median；其他实验参数不变。
- **代码差异：** CLI/探针新增 `windows_per_sample`，保留二维原始 window、归一化 window 和 sample median；契约测试 RED 后 GREEN（1 passed）。
- **日志/指标：** CPU median `472,224.16 ns`、IQR/median `3.152%`；MPS median `281,513.34 ns`、IQR/median `0.249%`。

## 结果

- **观察事实：** 组内 median 把 MPS 稳定性进一步提升至 0.249%；CPU 仍比 3% 门禁高 0.152 个百分点，sample median 在约 462–488 μs 间缓慢漂移。
- **错误签名：** 无。
- **推断：** 单个约 23 ms CPU window 仍不足以完全平均系统调度/频率变化；不应靠删异常点或放弃四线程，先扩大 timed window。
- **证据等级变化：** H-02 保持 E1；MPS 证据加固，CPU 未通过。
- **信息增量：** 证明组内 median 有效但非充分；量化了 CPU 剩余差距。

## 结论

- **验收/交付更新：** measurement schema 已能同时保留原始 window 和稳健 sample。
- **预算变化：** 无。
- **下一 micro-goal：** C005 只将 inner_iterations 从 50 提高到 500，把 CPU timed window 扩大到约 0.2 秒级。
- **是否需决策：** 当前无。
