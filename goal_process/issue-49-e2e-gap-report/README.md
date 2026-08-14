# Issue #49：E2E 联合差异与 reconciliation 报告

当前中文迭代报告 Run Bundle：

`evidence/runs/issue49-20260814T0730Z-e2e-gap-report-v12`

历史 authority v6 保持字节不变，并由 v12 的 `supersedes` 锁定：

`evidence/runs/issue49-20260814T0345Z-e2e-gap-report-v6`

报告严格消费 #48 的 Schedule Achievable Frontier 与 #47 的同边界 observed
decomposition。两项来源都先通过公共 verifier，再以 Run ID、bundle kind、manifest
SHA-256 和仓库内相对路径锁定。机器 JSON、完整组件 CSV 与中文 HTML 是同一份已验证输入的三个投影。

本次真实 evidence boundary 不变：#48 的 52 个 semantic leaves 及 6 个 mandatory
schedule effects 尚未全部 qualified，#47 也缺同完整 identity 的 profiling overhead
holdout 和完整 device timeline。因此 authority 仍是 `structured-unknown`。

面向迭代的 Report Values 不再留空。预测侧从冻结 #30 Cost IR 的 52 个叶子、同 cohort
实测 P80 compute/HBM rate、50% 保守资源效率、15 µs dispatch floor 和 serialized-unfused
计划生成 D 级预测；实测 E2E 使用 #47 的 B 级 1.921530 ms 基线，组件则按预测权重缩放为
D 级“实测侧降级估计”。HTML 以简体中文展示 E2E、模块汇总、两侧独立 TOP10、E2E
贡献、联合差异、证据等级、不确定区间和平账；同一集合同时输出 JSON 与包含全部 52
组件的 CSV。这些降级值只允许形成优化假设，不得用于验收、Frontier promotion 或校准。

公共 API 以合成 contract test 覆盖数值可用路径：两侧独立 Top 10 及自身 E2E 10%
强制项、exact Stable Path union、单侧 unavailable、interval-union/overlap reconciliation、
combined uncertainty 与 versioned materiality policy gate。

本票没有运行 NPU；没有生成任何新的 timing evidence。

一键重新生成（默认自动生成带 UTC 时间的唯一 run_id）：

```bash
uv run python \
  goal_process/issue-49-e2e-gap-report/build_gap_report_bundle.py
```

回放：

```bash
uv run groundupscale verify-run \
  goal_process/issue-49-e2e-gap-report/evidence/runs/issue49-20260814T0730Z-e2e-gap-report-v12 \
  --json
```
