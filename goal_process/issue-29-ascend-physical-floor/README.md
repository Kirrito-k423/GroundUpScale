# Issue #29：Ascend NPU MatMul Physical Floor 与 Observation Comparison

本目录固化同一个真实 `512 x 512 @ 512 x 512` FP32 MatMul 从 authored
Specs、Cost IR、Ascend Hardware Backend 到 Comparison Run Bundle 的证据。

## 范围

- 目标硬件：A2-AK-225 的单张 Ascend 910B2 V1，逻辑设备 `npu:0`。
- Stable Path：
  `model/two-layer-transformer/transformer/layer_0/attention/q_proj`。
- 本票生成 Resource Physical Floor，并与 Issue #28 的 Baseline Timing
  Observation 并列展示。
- 本票不执行 Operator Frontier qualification，也不实现 #25 的跨硬件诊断；
  Operator Frontier 和完整实现 duration 均保持 `unknown`。

## 版本化输入

- `specs/hardware/ascend-910b2.yaml`：静态硬件事实；公开材料无法唯一绑定的
  单卡 FP32/HBM 理论峰值保持 `unknown`。
- `specs/fabrics/local-ascend-910b2.yaml`：单卡实例拓扑。
- `specs/deployment-intents/ascend-npu.yaml`：部署选择。
- `specs/microbenchmarks/ascend-910b2-npu.yaml`：多 Shape 资源探针。
- `specs/hardware-capabilities/ascend-910b2-npu-local.yaml`：实测 Profile。
- `specs/plans/ascend-npu-prefill.yaml`：显式引用上述输入的 Analysis Plan。

这些文件分别表达硬件种类、实例拓扑、部署选择和校准证据，不把真实 cohort
身份写回可复用 Hardware Spec。

## 真实 Capability Profile

2026-08-10 在 Hardware Cohort `ascend-npu-23b93a89d5fecc79` 上采集：

| Resource | eligible Shapes | P80 | P95 | quality |
|---|---:|---:|---:|---|
| `compute.fp32` | 11 / 12 | 19.176 TFLOP/s | 36.875 TFLOP/s | exploratory |
| `memory.hbm` | 12 / 12 | 1.408 TB/s | 1.446 TB/s | exploratory |

每个 Shape 保留 20 个 NPU Event 原始窗口、median、Q1、Q3、IQR、正确性、
计时器和实现身份。Profile 保留原始 observation/cohort SHA-256、有效域、P80/P95
统计定义和质量状态。设备无法查询固定 power policy，因此证据没有被提升为
`qualified`，而是明确记录 `power-policy-unobserved`。

第一次 v1 采集混合了较小 copy 工作集，出现最高约 5.0 TB/s 的缓存/工作集效应；
它保留为调查证据，但没有被 Profile 使用。v2 将 copy 工作集收紧为 256–448 MiB，
Profile 只引用 v2 原始观测。

远端采集命令：

```bash
export PYTHONPATH=/home/t00906153/GroundUpScale-issue29-20260810/src
export ASCEND_RT_VISIBLE_DEVICES=0

/home/miniconda3/envs/lmz_pt27py311/bin/python -m groundupscale.cli \
  benchmark-hardware specs/microbenchmarks/ascend-910b2-npu.yaml \
  --repository-root . \
  --observation-output goal_process/issue-29-ascend-physical-floor/evidence/ascend-910b2-resource-observation-20260810-v2.json \
  --cohort-output goal_process/issue-29-ascend-physical-floor/evidence/ascend-910b2-hardware-cohort-20260810-v2.json \
  --profile-output specs/hardware-capabilities/ascend-910b2-npu-local.yaml \
  --profile-name ascend-910b2-npu-local --json
```

## Physical Floor 与 Observation

Q projection 的 Cost IR 最小需求：

```text
minimum_work_flops = 268,435,456
compulsory_bytes   = 3,145,728
compute floor      = 13,998.515 ns
memory floor       = 2,234.107 ns
Physical Floor     = max(compute, memory) = 13,998.515 ns
Observation median = 82,810 ns
Observation/Floor  = 5.916x
```

`5.916x` 是优化 headroom，不是 prediction error。Comparison 中
`relative_prediction_error=null`，完整实现 `full_duration_ns=null`。同一 Analysis
Plan 的 34 个非 MatMul 区域和 10 个超出观测 Shape/contiguous layout 有效域的
MatMul 区域（共 44 个）以 `partial-unknown` 保存；只有 8 个与 Profile 域一致的
MatMul 产生 Candidate。

权威回放入口是 v2 Comparison Run Bundle：

```bash
uv run groundupscale verify-run \
  goal_process/issue-29-ascend-physical-floor/evidence/runs/ascend-910b2-matmul-floor-comparison-20260810-v2 --json

uv run groundupscale explain \
  goal_process/issue-29-ascend-physical-floor/evidence/runs/ascend-910b2-matmul-floor-comparison-20260810-v2 --json
```

Bundle 内含锁定 Specs、Cost IR、Hardware Backend 结果、源 Measurement Manifest、
正确性、Candidate identity、Completion Boundary、原始计时、Comparison、
Explanation Graph 和 HTML 报告。
验证器会拒绝缺失/重复角色、digest 变化、cohort/Stable Path 不一致、Observation
口径变化、复制证据与源 Measurement Manifest 不一致，以及把 Physical Floor 或
Operator Frontier 冒充完整 duration 的改写。
