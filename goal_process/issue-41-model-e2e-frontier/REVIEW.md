# Issue 41 Code Review

- Fixed point: `5a0958e75c2c9323d2494136b3b26e1d4ded2b67`
- Reviewed implementation commit: `d5b2a070e7ee796f6fb69eff53eb7e76a3fd0410`
- Diff: `git diff 5a0958e75c2c9323d2494136b3b26e1d4ded2b67...HEAD`
- Spec: GitHub Issue #41（正文完整读取；无评论）

## Standards

初审共 5 项：

1. `must-fix`：synthetic evidence 未强制不可 promotion，却可生成 numeric Frontier。
2. `must-fix`：schedule 仅求和，未消费显式 dependencies 与 Resource Claims，也未保存
   critical-path/shared-resource bound。
3. `must-fix`：known Observation 可为 `0` 并触发除零；`NaN/Inf` 也未 fail closed。
4. `should-fix`：`html-report` role 指向 `.txt` / `text/plain`，manifest 不自描述。
5. `should-fix`（judgement call，Duplicated Code）：测试与 evidence 生成脚本的 fixture
   builder 重复。

修复与复审结论：

- 1 `RESOLVED`：输入现在强制 `promotion_eligible is False`，并输出
  `authority=synthetic-contract-only`；定向 red→green 测试覆盖绕过。
- 2 `RESOLVED`：输入锁定完整 dependency chain 与 candidate Resource Claims，复用
  `BoundEvent + compose_schedule_bound`，输出 serialized、critical-path、
  shared-resource、ideal-DAG、selected 五个可重算 bound。
- 3 `RESOLVED`：Observation unknown 不比较；known `0/NaN/Inf` 全部拒绝。
- 4 `RESOLVED`：人类报告改为真实 `reports/report.html`、`text/html`，role、路径、
  media type 和 manifest 一致。
- 5 `OPEN / non-blocking judgement`：重复 builder 只服务测试与可重复生成脚本，当前保留，
  以避免把测试 fixture 依赖到生产包；后续 schema 演进时应同步修改两处。

Standards 最终结论：0 must-fix open；1 个非阻塞 judgement smell。

## Spec

初审共 2 项 must-fix：

1. Observation 为 unknown 时，结果仍可标 `complete`，违反 mandatory unknown fail
   closed。
2. 未锁定 mandatory operation/effect 集，删除整个 elementwise requirement 或 schedule
   effect 条目可绕过 unknown；同时缺 elementwise candidate 的明确测试。

修复与复审结论：

- 1 `RESOLVED`：只有四轴全 known 才标 `complete`；Schedule 轴仍独立保留 numeric，
  comparison 与 relative prediction error 为 unknown。
- 2 `RESOLVED`：`mandatory_operation_classes` 与 `mandatory_effect_ids` 必须逐项精确
  对齐实际 section，删项直接 fail closed；新增 elementwise candidate 缺失测试，验证
  structured unknown、operation class 与 required evidence。

Spec 最终结论：0 must-fix open；Issue #41 五项 acceptance criteria 均有公共 seam 测试。

## Final verification

- `uv run python -m compileall -q src tests scripts`
- focused public seam + Run Bundle + schedule + reference tests
- complete/unknown committed Run Bundle `verify-run --json`
- `git diff --check`
- 未运行 NPU：本票只产生 deterministic synthetic contract evidence，且不可 promotion。
