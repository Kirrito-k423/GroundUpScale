# Issue #49：E2E 联合差异与 reconciliation 报告

权威 Run Bundle：

`evidence/runs/issue49-20260814T0345Z-e2e-gap-report-v6`

报告严格消费 #48 的 Schedule Achievable Frontier 与 #47 的同边界 observed
decomposition。两项来源都先通过公共 verifier，再以 Run ID、bundle kind、manifest
SHA-256 和仓库内相对路径锁定。机器 JSON 与 HTML 是同一份已验证输入的两个投影。

本次真实 evidence boundary 是：#48 的 52 个 semantic leaves 及 6 个 mandatory
schedule effects 尚未全部 qualified，因此 selected feasible schedule 为 unknown；#47
缺同完整 identity 的 profiling overhead holdout，并且 profiler device timeline export
不完整。因此两侧各自的 Top 10、联合 gap table、E2E gap/ratio、combined uncertainty、
Frontier efficiency、relative prediction error 和 deep diagnosis 均保持 unavailable，绝不把
lower bound 或缺失字段当作 point prediction/zero。

公共 API 还以合成 contract test 覆盖数值可用路径：两侧独立 Top 10 及自身 E2E 10%
强制项、exact Stable Path union、单侧 unavailable、interval-union/overlap reconciliation、
combined uncertainty 与 versioned materiality policy gate。

本票没有运行 NPU；没有生成任何新的 timing evidence。

回放：

```bash
uv run groundupscale verify-run \
  goal_process/issue-49-e2e-gap-report/evidence/runs/issue49-20260814T0345Z-e2e-gap-report-v6 \
  --json
```
