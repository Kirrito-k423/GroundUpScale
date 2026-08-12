# Ticket #25：M4 与 Ascend 真实跨硬件回放

本目录只实现 GitHub ticket #25。它冻结 Apple M4 CPU 的真实 256/512 Shape 证据，并与 ticket #32 已冻结的 Ascend 910B2 Diagnostic Bundle 一起，经 `compare-cross-hardware.py` 生成同一 512³ MatMul 语义的跨硬件报告。本票不创建 CI promotion，也不实现 #26。

## 权威输入

- M4：`evidence/runs/issue25-m4-cpu-diagnostic-v1`
- Ascend：`../issue-32-ascend-diagnostic-bundle/evidence/runs/issue32-ascend-910b2-diagnostic-v1`

M4 cohort 为 `hvc-46ac04a4db5c57adaff46f8b4eb99bd0251a5dbfd15492dc5ba766aff21aa327`。本票在该 cohort 下重新采集了 256³ q-proj 的三个 search 与三个 holdout Run Bundle；每个 session 都有独立进程、完整原始计时、收敛 warmup、候选实现身份和 float64 oracle 正确性。它们与既有的合格 512³ ACTIVE Anchor 构成一个版本化 1D Surface，因此 512 为 `exact_anchor`，384 为同一 cell 内的 `interpolated` 查询。cohort 变化时诊断与比较均 fail closed，不复用旧 Surface。

M4 的 Resource Physical Floor 为 `153,527.65853810357 ns`，Operator Frontier 为 `154,364.57142857142 ns`，Observation 为 `154,532.75 ns`。这些值分别保留其物理下界、实现前沿和实测观察语义。Ascend 侧回放 #32 的真实 `insufficient_evidence`、`integration_overhead` 和受控负对照 `confirmed_bug`；负对照只证明直接、可复现的正确性证据能支持 bug verdict，并不宣称平台实现存在缺陷。

报告只比较两侧独立定义的 Frontier Efficiency 与证据质量。绝对延迟明确标记为 `not-a-fair-efficiency-metric`；CPU 未请求的硬件计数器不会被 NPU 的可选计数器替代。

## 复现

```bash
python -m groundupscale.cli verify-run goal_process/issue-25-cross-hardware-replay/evidence/runs/issue25-m4-cpu-diagnostic-v1 --json
python -m groundupscale.cli diagnose goal_process/issue-25-cross-hardware-replay/evidence/runs/issue25-m4-cpu-diagnostic-v1
python scripts/compare-cross-hardware.py \
  goal_process/issue-25-cross-hardware-replay/evidence/runs/issue25-m4-cpu-diagnostic-v1 \
  goal_process/issue-32-ascend-diagnostic-bundle/evidence/runs/issue32-ascend-910b2-diagnostic-v1 \
  --json
```

`build_m4_diagnostic_bundle.py` 只做确定性派生并拒绝覆盖固定 Run ID。采集规格位于 `collection/`；源 Run Bundle 位于 `evidence/modern-source-runs/runs/`。Builder 在读取每个源 Bundle 前先验证 manifest 与制品摘要，再把实际用到的 manifest、benchmark、environment、512 profile 和观察/比较制品复制进权威 Bundle，使每个报告结论都能沿 Evidence Index 回到原始 Run、政策、Anchor/Surface、schedule/ledger 与派生 ID。
