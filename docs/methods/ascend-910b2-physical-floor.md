# Ascend 910B2 MatMul Resource Physical Floor

Ascend 后端只为落入 Profile 有效域且与已观测 Shape 一致的 contiguous FP32
MatMul Cost IR 区域生成
`ImplementationCandidate`。每个 Candidate 保留 minimum mathematical FLOPs、
compulsory bytes、Profile P80/P95、有效域、质量、不确定性、cohort 和原始 evidence
digest，并计算：

```text
T_compute = minimum_work_flops / compute.fp32.P80
T_memory  = compulsory_bytes / memory.hbm.P80
T_floor   = max(T_compute, T_memory)
```

这个 `max` 只属于单个 MatMul Candidate，并显式假设计算与内存可完全重叠。它不包含
dispatch、tiling、同步、contention、框架开销或不支持的 Cost IR 区域，因此
`full_duration_ns` 必须为 `null`。非 MatMul 区域列入 `unsupported_regions`，完整
program 保持 partial/unknown。环境不合格、Profile 被 quarantined、布局或 Shape
域外时仍保留 Capability evidence，但不会产生非空 floor。

结果层互不覆盖：Hardware Spec 中不可比的厂商理论值保持 unknown；真实多 Shape
Hardware Capability Profile 单独保存；Resource Physical Floor 是局部下界；Issue #29
未资格化 Operator Frontier；Issue #28 的真实 Observation 仍由原 Measurement Run
Bundle 提供。Observation/Floor 只表示优化 headroom，不计算 prediction error。

公共入口：

```bash
uv run groundupscale compile specs/plans/ascend-npu-prefill.yaml \
  --repository-root . --output /tmp/ascend-compile --json

uv run groundupscale compare-measurement \
  specs/plans/ascend-npu-prefill.yaml \
  goal_process/issue-28-npu-measurement-adapter/evidence/runs/ascend-910b2-exact-shape-512-20260810-v1 \
  --repository-root . --artifact-store .groundupscale \
  --run-id ascend-matmul-floor-comparison --json
```
