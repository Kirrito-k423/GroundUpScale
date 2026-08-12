# Ticket #33：M4–Ascend 同 Shape 跨硬件诊断报告

本票新增 `groundupscale.cross_hardware`，只消费两侧已经派生的
`diagnostic-result/v1alpha1` 或可验证 Run Bundle，不创建 Anchor、Surface 或
计时证据，也不把一侧 cohort 的证据复制到另一侧。

## 使用

```bash
uv run groundupscale compare-cross-hardware <m4-result-or-bundle> <ascend-result-or-bundle> --json
```

报告 schema 为 `groundupscale.dev/cross-hardware-diagnostic-report/v1alpha1`，包括：

- exact Shape、dtype、layout、execution mode 与语义操作匹配结果；
- 每侧 Hardware Cohort、四轴诊断质量、Frontier/Surface/Anchor/策略/派生引用；
- 以 `Frontier / Observation` 定义的 Frontier Efficiency、两侧综合不确定性和相对效率；
- 明确声明绝对时延不是公平效率指标；
- 缺少 capability manifest、fingerprint/cohort、preflight、correctness、主计时、Completion Boundary、Active Anchor、Surface 或四轴结果时的 `unknown`/`insufficient_evidence`。

相同 cohort、Shape 或执行域不匹配会 fail closed。报告的 `evidence_index` 保留输入
Run Bundle/结果路径，便于回放到原始证据；optional counter 缺失不会被补成零。
