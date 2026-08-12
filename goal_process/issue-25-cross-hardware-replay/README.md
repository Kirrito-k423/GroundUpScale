# Ticket #25：M4 与 Ascend 真实跨硬件回放

本目录只实现 GitHub ticket #25。它冻结 Apple M4 CPU 与 Ascend 910B2 的真实 `M=256/384/512, N=K=512` Shape 证据，经 `compare-cross-hardware.py` 生成同一 512³ MatMul 语义的跨硬件报告。本票不创建 CI promotion，也不实现 #26。

## 权威输入

- M4：`evidence/runs/issue25-m4-cpu-diagnostic-v1`
- Ascend：`evidence/runs/issue25-ascend-910b2-diagnostic-v1`

M4 cohort 为 `hvc-46ac04a4db5c57adaff46f8b4eb99bd0251a5dbfd15492dc5ba766aff21aa327`。本票在该 cohort 下重新采集了固定 `N=K=512`、仅改变 `M` 的三个 256 search、三个 256 holdout，以及三个独立 384 boundary-confirmation Run Bundle；每个被采用的 session 都有独立进程、完整原始计时、收敛 warmup、候选实现身份和 float64 oracle 正确性。它们与既有合格的 512 ACTIVE Anchor 构成 ADR 0036 的 latency-primary 1D Surface：512 为 `exact_anchor`，384 为同一 Shape Regime 内的 `interpolated` 查询；Effective Rate 仅由 `2MNK / latency` 派生，插值不确定性由三个 confirmation residual 的 RMS 校准。cohort 变化时诊断与比较均 fail closed，不复用旧 Surface。

Ascend 侧在 A2-AK-225 的 `Ascend910B2/npu:0` 上新增 15 个独立进程：256 与 512 各三个 search、三个 holdout，再以三个 384 session 做 boundary confirmation；每个 session 保存 100 个 device-event 样本，每个样本含 100 次内循环，正确性均通过，采集后设备健康为 `OK`。该证据取代 #32 被 ADR 0036 否决的方阵 Effective-Rate Surface，但复用 #32 已冻结并摘要验证的 capability discovery、cohort fingerprint、preflight、timing plan、Observation、schedule/ledger、probe/ablation 与 Verdict 证据。#32 Bundle 本身保持不可变。

M4 的 Resource Physical Floor 为 `153,527.65853810357 ns`，Operator Frontier 为 `154,364.57142857142 ns`，Observation 为 `154,532.75 ns`。Ascend 的新固定 N/K Operator Frontier 为 `15,487.000346183777 ns`，Observation 仍为 `44,507.629499999995 ns`。这些值分别保留其物理下界、实现前沿和实测观察语义。Ascend 侧回放 #32 的真实 `insufficient_evidence`、`integration_overhead` 和受控负对照 `confirmed_bug`；负对照只证明直接、可复现的正确性证据能支持 bug verdict，并不宣称平台实现存在缺陷。

报告通过 portable semantic identity `transformer/layer-0/attention/q-proj` 验证两侧是同一语义算子，只比较两侧独立定义的 Frontier Efficiency 与证据质量。绝对延迟明确标记为 `not-a-fair-efficiency-metric`；CPU 未请求的硬件计数器以无样本的 `not_requested` diagnostic lane 保存，不会被 Baseline 样本或 NPU 的可选计数器替代。

## 复现

```bash
python -m groundupscale.cli verify-run goal_process/issue-25-cross-hardware-replay/evidence/runs/issue25-m4-cpu-diagnostic-v1 --json
python -m groundupscale.cli diagnose goal_process/issue-25-cross-hardware-replay/evidence/runs/issue25-m4-cpu-diagnostic-v1
python -m groundupscale.cli verify-run goal_process/issue-25-cross-hardware-replay/evidence/runs/issue25-ascend-910b2-diagnostic-v1 --json
python -m groundupscale.cli diagnose goal_process/issue-25-cross-hardware-replay/evidence/runs/issue25-ascend-910b2-diagnostic-v1
python scripts/compare-cross-hardware.py \
  goal_process/issue-25-cross-hardware-replay/evidence/runs/issue25-m4-cpu-diagnostic-v1 \
  goal_process/issue-25-cross-hardware-replay/evidence/runs/issue25-ascend-910b2-diagnostic-v1 \
  --json
```

`build_m4_diagnostic_bundle.py` 与 `build_ascend_diagnostic_bundle.py` 只做确定性派生并拒绝覆盖固定 Run ID。M4 采集规格位于 `collection/`；256 fit/holdout 源 Bundle 位于 `evidence/adr0036-source-runs/runs/`，384 boundary-confirmation Bundle 位于 `evidence/adr0036-confirmation-runs/runs/`。Ascend 原始 session 位于 `evidence/ascend-sessions/`，远端执行记录位于 `evidence/ascend-remote-execution.json`。Builder 读取源证据前验证 manifest、session identity 与 SHA-256，再把实际使用的证据复制进权威 Bundle，使每个报告结论都能沿 Evidence Index 回到原始 Run、策略、Anchor/Surface、schedule/ledger、probe/ablation 与派生 ID。
