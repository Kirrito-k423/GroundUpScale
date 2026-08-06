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
