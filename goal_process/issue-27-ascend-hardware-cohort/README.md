# Issue #27 — real Ascend Hardware Cohort evidence

This directory contains the ticket-scoped, real-device puncture for
[GitHub Issue #27](https://github.com/Kirrito-k423/GroundUpScale/issues/27).
It deliberately does **not** implement the public NPU Measurement Adapter or a
Run Bundle; those belong to Issue #28.

## Frozen case

- target: machine `A2-AK-225`, physical NPU 0 / chip 0, host Chip Logic ID 0,
  exposed as process-local `npu:0` with `ASCEND_RT_VISIBLE_DEVICES=0`;
- operation: one eager `torch.matmul`, FP32, contiguous
  `[512, 512] @ [512, 512]`;
- corpus: CPU `torch.randn` with seed `20260810`;
- oracle: CPU float64 MatMul, `atol=1e-3`, `rtol=1e-3`, exact shape and finite
  output required;
- timing: twenty synchronized warmups followed by one hundred raw samples in each of
  three independent Python processes;
- Completion Boundary: a start/end `torch.npu.Event` pair on the default stream,
  end-event synchronization and explicit device synchronization before the
  elapsed time is read;
- timer resolution: the CANN event API's documented floating-point millisecond
  result is converted exactly to integer nanoseconds; the frozen sessions each
  show a `20 ns` empirical output step, derived from raw samples rather than
  assumed from the API unit;
- baseline: profiler disabled. Optional profiling and counters retain explicit
  `not_requested`/`unsupported` states.
- repeatability: all raw samples are retained; a session is quarantined when
  `IQR / median > 10%`, and a three-session aggregate is rejected when any
  session median differs from the median of session medians by more than `5%`.
  This ticket does not perform Frontier qualification.

The collector fails with exit code 2 and writes a structured `blocked` record
when torch_npu import, visibility, health, process isolation, correctness,
device placement, command collection, or cohort matching fails. CPU fallback is
never accepted. It also refuses duplicate process sessions, recomputes identity
digests instead of trusting self-reported values, proves the physical-to-logical
device mapping, and never overwrites an existing evidence path.
The committed mapping row is parsed from `npu-smi info -m`; its Chip Logic ID
is checked against `ASCEND_RT_VISIBLE_DEVICES`, then cross-checked with the
physical card, singleton visibility, current process-local device, and runtime
device name. A missing or contradictory mapping fails closed.

## Reproduction

The remote workspace is `/home/t00906153/groundupscale-issue27`. Copy
`collect_ascend_matmul.py` there, source the installed CANN environment, select
one physical card, and run three separate processes:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export ISSUE27_PYTHON=/home/miniconda3/envs/atkpy39/bin/python
$ISSUE27_PYTHON collect_ascend_matmul.py collect --session-id issue27-s52 \
  --output sessions/issue27-s52.json
$ISSUE27_PYTHON collect_ascend_matmul.py collect --session-id issue27-s53 \
  --output sessions/issue27-s53.json
$ISSUE27_PYTHON collect_ascend_matmul.py collect --session-id issue27-s54 \
  --output sessions/issue27-s54.json
$ISSUE27_PYTHON collect_ascend_matmul.py merge \
  --output ascend-910b2-matmul-cohort-20260810-v5.json \
  sessions/issue27-s52.json sessions/issue27-s53.json \
  sessions/issue27-s54.json
```

The committed aggregate references every raw session by SHA-256. Raw command
snapshots retain device, software, topology, health, memory, power, and timing
sources without recording credentials.

## Frozen result

The accepted evidence was collected on 2026-08-10 from an Ascend 910B2 V1 at
physical NPU 0 / chip 0 and PCIe `0000:C1:00.0`. The selected stack was
openEuler 22.03 LTS, kernel `5.10.0-60.18.0.50.oe2203.aarch64`, driver
`25.3.rc1`, firmware `7.8.0.2.212`, CANN `8.5.0`, Python `3.9.25`, PyTorch
`2.8.0+cpu`, and torch_npu `2.8.0.post2`. The full VDie/device identity, topology,
memory, health, power, and source snapshots remain in the JSON artifacts.

All three independent sessions produced cohort digest
`53352d739a411f3b722ad05984bd4d1366fb4949d1b615f8143aecdd7e313e22`.
Each session passed the CPU float64 oracle on `npu:0` with no fallback and kept
100 raw device-event samples. Their medians were `86,390 ns`, `82,080 ns`, and
`83,420 ns`; the maximum deviation from the median of session medians was
`3.56%`. Across all 300 samples, the device-event median was `84,550 ns`
(`11,880 ns` minimum, `5,205,300 ns` maximum). No sample was excluded; the
predeclared robust IQR and cross-session gates, rather than tail deletion,
determine acceptance. These are
Observation values for this exact evidence contract, not a promoted Frontier.

The manifest reports HBM frequency and device-wide power as measured. The card
does not support querying its work mode, so power-policy evidence and AI Core
frequency/counters are explicitly `unsupported`; the observation is therefore
not Frontier-eligible. Operator profiling is `not_requested` to keep the
Baseline Timing Lane minimally intrusive. No unavailable field is filled with
zero.
