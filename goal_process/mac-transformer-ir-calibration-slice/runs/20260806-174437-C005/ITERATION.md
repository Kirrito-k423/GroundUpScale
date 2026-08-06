# C005：扩大 CPU timed window 到稳定粒度

- **开始/结束：** 2026-08-06T17:44:37+08:00 / 2026-08-06T17:47:53+08:00
- **阶段：** PROBE
- **动作类型：** MEASURE
- **关联验收/未知量：** AC-06、AC-07、H-02、U-01

## 预注册

- **本轮 micro-goal：** 验证约 0.2 秒级 timed window 能否使默认四线程 CPU 与 MPS 的 sample 间 `IQR/median` 均不超过 3%。
- **当前假设：** C004 每个 CPU window 约 23 ms，仍对调度/频率漂移敏感；提高 10 倍工作量会降低非工作负载开销占比。
- **已有证据：** C004 CPU 3.152%、MPS 0.249%，且所有原始 window 已保留。
- **证据等级：** H-02 E1；MPS E2，CPU 接近但未过门禁。
- **唯一主要变量：** inner_iterations 从 50 提高到 500；warmup=500、windows_per_sample=5、samples=20、Shape、dtype、seed、设备、四线程默认值和同步边界不变。
- **预期观察：** CPU/MPS `IQR/median <=3%`。
- **判别规则：** 两设备均通过则 H-02 升 E2 并退出 M1 噪声诊断；CPU 仍失败则下一步必须显式冻结线程/并发口径，不能继续放大窗口。
- **成本与风险：** 预计小于 2 分钟；无外部费用；CPU/MPS 短时持续负载。
- **停止与回滚：** 单命令 20 分钟停止；不修改门禁或样本选择。

## 执行

- **脱敏命令：** `commands.md`
- **配置/环境差异：** 仅 inner_iterations=500。
- **代码差异：** 无。
- **日志/指标：** CPU median `476,684.54 ns`、IQR/median `1.611%`；MPS median `272,522.25 ns`、IQR/median `0.314%`。

## 结果

- **观察事实：** 两设备均低于 3% 门禁。CPU 原始 window 仍保留一个归一化 `1,074,540.58 ns` 尖峰（约正常值 2.25 倍），其所在 sample 的组内 median 为 `472,125.83 ns`，说明预注册稳健聚合按设计工作。MPS sample 稳定在约 265–273 μs。
- **错误签名：** 无。
- **推断：** 当前机器支持同时保留默认四线程 CPU 能力与 3% 噪声门禁；必要条件是足量 warmup、约 0.2 秒级 timed window、保留原始 window 并用组内 median 构造 sample。MPS 同样通过。
- **证据等级变化：** H-02 E1→E2（当前环境、Shape、协议的本机证据）。
- **信息增量：** M1 剩余噪声未知量关闭；为 M4/M5 的 Benchmark runner 提供了可执行采样协议。

## 结论

- **验收/交付更新：** M1 退出条件全部满足；AC-06/AC-07 的测量可判定性获得前置证据，但预测误差门禁仍待 M5。
- **预算变化：** 无。
- **下一 micro-goal：** M2/C006：定义最小 YAML bundle 的严格 Schema 与公开 compile seam，先产出层次化 ModelIR/WorkloadIR/SemanticIR。
- **是否需决策：** 当前无。
