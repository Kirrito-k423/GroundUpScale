# 决策记录

## D-001：TDD 只验证已确认的公开 seam

- **时间：** 2026-08-06T17:29:02+08:00
- **背景证据：** `GOAL.md`、ADR-0027、ADR-0030、ADR-0031、ADR-0032。
- **选项：** 测内部 pass / 通过公开编译与运行接口验证 / 全量 snapshot 内部对象。
- **决定：** 测试通过已确认的 `SemanticCompiler.compile`、CLI 的 compile/run/explain 行为、Benchmark runner 与 Run Manifest 产物接口；公式使用独立手算 literal，不 mock 自有编译器内部模块。
- **决定者：** 用户已确认相关架构与 Goal；执行者按 `$tdd` 固化。
- **影响：** 测试可跨内部重构，重点覆盖用户可观察链路；内部不变量通过编译结果验证。
- **回滚条件：** 公开 seam 经 Goal 变更被用户调整。

## D-002：M1 只做环境与低成本可行性探针

- **时间：** 2026-08-06T17:29:02+08:00
- **背景证据：** H-01 至 H-04 当前仅 E0/E1，尚不允许高成本实验。
- **选项：** 直接实现全链 / 先安装并运行最小探针。
- **决定：** C001 只建立锁定环境并探测目标操作的 CPU/MPS availability、correctness、同步计时噪声和可归因内存接口。
- **决定者：** Codex，遵循 `$goal-execution`。
- **影响：** 在进入 M2 前排除版本与硬件不可行路径。
- **回滚条件：** 环境改动限于仓库本地 `.venv` 和锁文件，可删除后重建；不修改系统 Python。

## D-003：Benchmark sample 保留原始窗口并使用组内 median

- **时间：** 2026-08-06T17:47:53+08:00
- **背景证据：** C001–C004 依次暴露短窗口噪声、MPS 预热漂移和 CPU 调度尖峰；直接单窗口 IQR 不稳定。
- **选项：** 删除异常点 / 放宽 3% 门禁 / 单线程 CPU / 扩大窗口并稳健聚合。
- **决定：** 保留全部原始 timed window；足量 warmup；每个 window 覆盖足够工作量；每个统计 sample 取 5 个 window 的 median；门禁在 sample 间计算 IQR/median。Run Manifest 记录 PyTorch 线程数。
- **决定者：** Codex，遵循用户确认的 3% 噪声门禁与“不隐藏偏差”原则。
- **影响：** 当前探针 CPU 1.611%、MPS 0.314%；异常 window 可诊断但不会由单点主导 run-level median。
- **回滚条件：** 真实 Case 证明组内 median 引入系统性偏差，届时必须用保留的原始数据对比并经 Goal 变更，而不是静默换口径。

## D-004：Semantic Compiler 只消费逻辑策略投影

- **时间：** 2026-08-06T18:24:08+08:00
- **背景证据：** CPU/MPS AnalysisPlan 只在 placement 不同；SemanticIR 禁止包含设备、时延和 schedule。
- **选项：** 把完整 DeploymentIntent 纳入 fingerprint / 完全忽略 DeploymentIntent / 只投影逻辑策略。
- **决定：** `semantic_deployment_plan()` 丢弃 placement，只保留可能改变逻辑 partition/communication/state 的版本化 strategy；未注册策略显式失败。Execution 阶段仍消费完整 placement。
- **决定者：** 既有 ADR-0006/0009/0029 与 C008 实测。
- **影响：** CPU/MPS semantic JSON byte-identical；同一逻辑工作可跨硬件复用 CostIR。
- **回滚条件：** 某 deployment 字段被证明会改变数学/逻辑语义时，必须把它提升为明确 strategy effect，而不是偷偷读取 placement。

## D-005：内部草稿连线后一次冻结 SemanticIR

- **时间：** 2026-08-06T18:24:08+08:00
- **背景证据：** Typed Value 的 consumer 只有遍历完跨 Region 数据流后才完整，但外部 IR 必须不可变。
- **选项：** 对外暴露可变 IR / 每步复制整图 / 编译器内部 draft 后冻结。
- **决定：** SemanticCompiler 内部使用私有 Value draft 累积 consumer，所有 pass 完成并验证后一次性生成 frozen dataclass；request/result 与插件边界始终不可变。
- **决定者：** Codex，遵循 ADR-0018/0027。
- **影响：** consumer 闭包可验证，调用方和未来插件无法原地修改现有 IR。
- **回滚条件：** 无；如需增量编译，在内部引入持久化数据结构，不放宽公开不可变契约。

## D-006：CostIR 同时保留逻辑 bytes 与物化 bytes

- **时间：** 2026-08-06T18:36:36+08:00
- **背景证据：** View/Transpose 有逻辑 result Shape，但在当前语义下只 alias storage；参数占用、读取量、累计 activation 与峰值 live-set 也不是同一口径。
- **选项：** 只报一个 bytes / 只报物化 bytes / 多口径显式分列。
- **决定：** 每个 Cost op 分列 logical read/write、materialized read/write、parameter/buffer/activation read、explicit activation 与 alias result；Program 再给唯一 parameter/buffer/artifact bytes。
- **决定者：** Codex，依据 AC-05 与用户的可解释/深入追溯要求。
- **影响：** UI 可展示逻辑规模且不虚构 alias 分配；M4 peak memory 必须另做生命周期分析。
- **回滚条件：** 不删除既有口径；需要更接近真实 traffic 时由 Backend 新增 cache/transfer estimate，不覆写 CostIR。

## D-007：Attention MatMul 显式约束输出布局

- **时间：** 2026-08-06T18:52:33+08:00
- **背景证据：** C011 证明 heads-major context transpose 后不可零物化 flatten；直接 einsum 的输出也非 contiguous；MPS 非连续 `out=` 在小 Shape 正确但目标 `S=512` 静默错误。
- **选项：** 隐藏 `.contiguous()` / 扩展 Copy 操作 / 由 MatMul Backend 满足声明的输出布局。
- **决定：** Semantic MatMul 声明 `output_layout: sequence_major_contiguous`；reference Backend 用 query-major alias view + broadcast batched MatMul 直接生成目标布局，保留 V transpose 与最终 View 的零物化语义。
- **决定者：** Codex，依据 Goal 冻结操作集合、H-04 反证条件与 CPU/MPS 目标 Shape 实测。
- **影响：** 不扩目标操作集、不隐藏物化；Model/Semantic/CostIR 最终为 59 modules、52 ops、73 values，CPU/MPS E2E max abs `7.1526e-07`。
- **回滚条件：** 后端无法满足输出布局时必须显式引入 Materialize/Copy 并触发 Goal 范围升级，不能在 runner 内偷拷贝。
