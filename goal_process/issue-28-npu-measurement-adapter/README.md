# Issue #28: Ascend NPU Measurement Adapter

This directory freezes the first immutable exact-Shape Run Bundle produced
through GroundUpScale's public `measure` entry point and the portable
five-operation `MeasurementAdapter` seam.

## Scope

- Public device selection: `--device ascend-npu`.
- Adapter operations, in order: `discover_capabilities`,
  `fingerprint_cohort`, `preflight`, `build_timing_plan`, and `collect`.
- Real-device case: `torch.matmul`, float32, row-major contiguous,
  `512 x 512 @ 512 x 512`, fixed seed `20260810`.
- Timing contract: 20 synchronized warmup iterations, 100 preserved raw NPU
  event samples, and an end-event plus device-synchronize Completion Boundary.
  The predeclared quality gate requires `IQR / median <= 10%` and timer
  resolution / median `<= 1%`; failures are retained and quarantined, never
  hidden by deleting samples or defining a slower cohort.
- Correctness contract: CPU float64 MatMul oracle, exact output Shape, finite
  values, `atol=0.001`, and `rtol=0.001`.

Issue #25 cross-hardware replay, Surface construction, Anchor promotion, and
diagnostic Verdict work are intentionally outside this ticket.

## Real run

The frozen bundle was collected on 2026-08-10 from A2-AK-225 using logical
device `npu:0` (`ASCEND_RT_VISIBLE_DEVICES=0`) and the compatible Python 3.11
environment at `/home/miniconda3/envs/lmz_pt27py311/bin/python`.
That trusted hardware environment records torch `2.7.1+cpu`, torch_npu
`2.7.1`, CANN `8.5.0`, driver `25.3.rc1`, and firmware `7.8.0.2.212`.
Preflight enforces the versioned `ascend-npu-runtime-v1` contract: Python
`3.11`, torch `2.7`, torch_npu `2.7`, and CANN `8.5` by major-minor. This is
the trusted-hardware contract defined beside the Adapter; the repository's
`pyproject.toml` torch pin governs the separate portable compiler/CI
environment under ADR-0028 and does not eagerly import `torch_npu`.

```bash
export PYTHONPATH=/home/t00906153/GroundUpScale-issue28-20260810/src
export ASCEND_RT_VISIBLE_DEVICES=0

/home/miniconda3/envs/lmz_pt27py311/bin/python \
  -m groundupscale.cli measure \
  --device ascend-npu \
  --m 512 --n 512 --k 512 \
  --dtype float32 \
  --layout row-major-contiguous \
  --seed 20260810 \
  --warmup 20 \
  --repetitions 100 \
  --artifact-store /home/t00906153/issue28-final-v2 \
  --run-id ascend-910b2-exact-shape-512-20260810-v1 \
  --json
```

The completed Run Manifest identifies device `ascend-npu` and Hardware Cohort
`ascend-npu-23b93a89d5fecc79`. The cohort freezes the physical/logical device
mapping, chip and VDie identity, PCIe/NUMA/HCCS topology, HBM capacity and
clock, live power, health/process guard, and sourced software versions. The
CPU correctness oracle passed. The 100 raw timing samples have a median of
82,810 ns, `IQR / median = 9.39%`, and no samples were excluded, so the
observation quality gate passed.

## Evidence

The immutable Run Bundle is at:

`evidence/runs/ascend-910b2-exact-shape-512-20260810-v1/`

Its manifest declares 16 unique artifact roles covering the Benchmark Case,
capability discovery, Hardware Cohort, preflight, timing plan, collection,
environment, candidate identity, input corpus, execution contract,
Instrumentation Profile, correctness, raw timing, memory, Completion Boundary,
and operation evidence. Every artifact records a Schema version, SHA-256
digest, and content-addressed producer lineage. The manifest records the
observation as valid and explicitly leaves frontier promotion unevaluated,
which is outside Issue #28.
The Instrumentation Profile also freezes synchronization, allowlisted metadata,
and accepted-overhead rules; diagnostic profiling remains disabled in the
Baseline Timing lane.

Verify it from the repository root:

```bash
python -m groundupscale.cli verify-run \
  goal_process/issue-28-npu-measurement-adapter/evidence/runs/ascend-910b2-exact-shape-512-20260810-v1 \
  --json
```

The verifier fails closed for content tampering, missing or duplicate required
roles, artifact/manifest Schema disagreement, device or Hardware Cohort
disagreement, recomputed cohort or producer-lineage disagreement, collection
component inconsistency, and operation evidence references that cannot be
replayed.

On a Mac without `torch_npu`, importing GroundUpScale and running compiler or
Adapter contract tests remains supported. A real `measure --device ascend-npu`
attempt publishes an immutable `blocked` Run Bundle with reason
`torch-npu-unavailable` instead of failing during import.
