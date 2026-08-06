# C010：独立验证单层/两层 Cost 总表并接入 CLI

- **开始/结束：** 2026-08-06T18:32:33+08:00 / 2026-08-06T18:36:36+08:00
- **阶段：** M3 / COST AGGREGATION
- **动作类型：** VERIFY
- **关联验收/未知量：** AC-03、AC-05、AC-10、D-03

## 预注册

- **本轮 micro-goal：** 以手工推导、硬编码 literal 验证单层与两层 FLOPs、逻辑/物化 bytes、参数、buffer、显式 activation/alias 总量；证明 CPU/MPS CostIR byte-identical，并由 CLI 输出 CostIR。
- **当前假设：** Region aggregation 是逐 op metrics 的纯加法；唯一 state bytes 与 per-invocation state read bytes 在本模型中相等，因为每个 state 每层只读一次。
- **已有证据：** C009 原子规则 E2；自动聚合初值未作为 expected 来源。
- **证据等级：** 原子 E2，聚合 E1。
- **唯一主要变量：** 新增独立总表断言和 CLI Cost artifact；不修改原子公式。
- **预期观察：** 两层总 FLOPs `9,710,850,048`；parameter `33,562,624 B`；buffer `2,097,152 B`；explicit activation `121,634,816 B`；alias `18,874,368 B`，以及其余 literal 全部精确相等。
- **判别规则：** 任一不等立即回到具体 region/op 定位，不调整 expected 迁就实现；CPU/MPS canonical CostIR 必须一致。
- **成本与风险：** 预计 10–15 分钟；确定性测试。
- **停止与回滚：** 测试 expected 不得调用生产公式或读取 CostIR 后再计算自身。

## 执行

- **脱敏命令：** `commands.md`
- **配置/环境差异：** 无。
- **代码差异：** 新增独立单层/两层 literal 总表测试、dependency DAG critical-path FLOPs、CLI CostIR artifact/summary，以及公式方法文档。
- **日志/指标：** C010 4 passed；全量 21 passed。CPU/MPS CostIR SHA-256 均为 `cc8c83d0b5ddd0a915b936692ae39ad982ad2e75302144bbaa9d3b3743bb2185`。

## 结果

- **观察事实：** 单层/两层所有预注册数字精确相等；两层串行工作 9,710,850,048 FLOPs，理想并行关键路径 6,489,624,576 FLOPs。parameter、buffer 与 workload artifact 均按唯一 State Artifact 去重；累计 activation/alias 单列。
- **推断：** Region 纯加法聚合与 producer dependency DAG 工作正常；同一 CostIR 可复用于 CPU/MPS Backend。关键路径是算法工作下界，不能当作 duration。
- **证据等级变化：** 聚合 Cost 与 AC-05 E1→E2。
- **信息增量：** M4 可以同时消费结构公式、依赖与状态大小，开始实现 reference runner/live-set/trace。

## 结论

- **验收/交付更新：** M3 完成；AC-03、AC-05 DONE；D-03 DONE。
- **预算变化：** 无硬件实验。
- **下一 micro-goal：** M4/C011：实现真实 PyTorch 两层 causal Transformer reference，并先完成 CPU/MPS E2E 数值一致性与无 fallback 证据。
- **是否需决策：** 无。
