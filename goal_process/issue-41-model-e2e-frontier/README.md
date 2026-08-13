# Issue 41：两层 Transformer 模型级 E2E Frontier 公共契约

本票在现有 `verify-run` 与 `explain` 公共入口上增加确定性模型级
Schedule Achievable Frontier 契约。它不声明真实 Ascend 性能，也不修改 issue #30
冻结的 NPU Observation；synthetic evidence 仅用于验证覆盖、结构化 unknown、四轴隔离、
同源报告和不可变重放。

## Evidence

- `issue-41-full-demo-contract-v1`：52 个语义叶子，`layer_0` 与 `layer_1`
  各 26 个唯一 Stable Path；所有 mandatory operation requirement 与 schedule effect
  齐全，因此产生 numeric Schedule Achievable Frontier 与 relative prediction error。
- `issue-41-missing-rmsnorm-phase-v1`：删除一个 RMSNorm
  `transcendental.rsqrt.fp32` candidate；同一入口保持 Resource Physical Floor 与
  Observation，Operator/Schedule Frontier 和 relative prediction error 为 structured
  unknown，并命名缺失 operation class 与所需证据。

重新生成（目标 Run ID 已存在时 writer 会拒绝覆盖）：

```sh
uv run python scripts/write-issue41-synthetic-evidence.py
```

离线重放：

```sh
uv run groundupscale verify-run \
  goal_process/issue-41-model-e2e-frontier/evidence/runs/issue-41-full-demo-contract-v1 \
  --json

uv run groundupscale explain \
  goal_process/issue-41-model-e2e-frontier/evidence/runs/issue-41-full-demo-contract-v1 \
  --json
```

`verify-run` 不只检查 artifact SHA-256。它从锁定的
`model-e2e-frontier-input` 重新派生机器结果和人类报告，因此删除 mandatory section、
破坏 Stable Path、隐藏 unknown 或只重算本地 artifact hash 仍会 fail closed。
