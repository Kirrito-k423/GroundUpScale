# Issue #47：同边界的 Ascend observed decomposition

权威 Run Bundle：

`evidence/runs/issue47-ascend-observed-decomposition-20260813-v1`

该 Bundle 通过公共 `schedule-effect-frontier` writer 与 `verify-run` seam 发布，锁定
同一个 `two-layer-prefill` Case、`[1, 512, 512]` Shape、eager candidate、
`ascend-npu-23b93a89d5fecc79` Hardware Cohort 和 NPU Event Completion Boundary。

Baseline Timing Lane 原样保留 #30 的 20 个样本、20 次 boundary 外 warmup、NPU
Event timer、同步、52 叶正确性和无 CPU fallback 证据；中位数仍是
`1,921,530 ns`。Diagnostic Profiling Lane 独立保留 #30 的单次
`50,840,320 ns` host diagnostic trace，未覆盖 baseline 真值。

本轮没有运行 NPU。#32 的消融 Case/Shape identity 与本 pair 不兼容，不能判定本 pair
的 profiling overhead；设备算子 timeline 导出也不完整。因此 observed leaf decomposition
诚实保持 `unavailable`，没有缩放 host trace 或伪造 device duration。最小下一测量是
在相同完整 identity 下重新执行配对 baseline/diagnostic holdout，并取得可用的
Ascend device timeline；该 measurement/profiling session 必须在公共全机 NPU 锁内完成。

本地回放不需要 `torch_npu`：

```bash
uv run groundupscale verify-run \
  goal_process/issue-47-observed-decomposition/evidence/runs/issue47-ascend-observed-decomposition-20260813-v1 \
  --json
```
