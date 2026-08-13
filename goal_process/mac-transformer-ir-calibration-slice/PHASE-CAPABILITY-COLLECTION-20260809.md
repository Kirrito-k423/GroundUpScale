# Apple M4 CPU compound-phase capability collection

Status: exploratory evidence collected; trusted promotion blocked by environment.

## Public seams exercised

- `groundupscale benchmark-hardware <suite>` collected the production
  `apple-m4-cpu-resource-envelope@0.2.0` suite through the public CLI.
- `groundupscale compile <plan>` consumed the resulting versioned profile through
  an exploratory Analysis Plan. Because the profile records
  `environment.eligible=false`, authoritative compound phase durations remained
  structured `unknown` with `phase-capabilities-incomplete`; the same exact
  phase rates are separately available as an `exploratory` provisional estimate
  for planning and report visibility.

## Collected evidence

- Raw observation:
  `evidence/apple-m4-cpu-phase-microbenchmark-observation-exploratory-20260809-v1.json`
- Raw SHA-256:
  `35ea7997db47831a8b911f2e670c8936364f820b5c77594579511f2324006dcf`
- Exploratory profile:
  `evidence/apple-m4-cpu-phase-capability-profile-exploratory-20260809-v1.yaml`
- Probe count: 19.
- Resource envelope count: 16, including all 14 exact Softmax/RMSNorm phase
  capability classes.
- Every new phase probe retained at least 12 stable Shapes. Most row and tensor
  probes retained all 15 configured Shapes; `elementwise_exp` and
  `memory_broadcast` retained 14/15 and still exceeded the 10-Shape gate.

This evidence is useful for validating the collection contract, exposing
measurement stability, and producing a visibly downgraded planning estimate.
It is not an accepted authoritative prediction input and cannot enter
Frontier, calibration, prediction error, or a diagnostic Verdict.

The exploratory P80/P95 values are retained for audit, not promotion:

| Exact resource | P80 | P95 |
|---|---:|---:|
| `compute.elementwise.divide.fp32` | 17.613 GFLOP/s | 18.091 GFLOP/s |
| `compute.elementwise.multiply.fp32` | 14.704 GFLOP/s | 16.042 GFLOP/s |
| `compute.elementwise.square.fp32` | 17.343 GFLOP/s | 18.099 GFLOP/s |
| `compute.elementwise.subtract.fp32` | 18.136 GFLOP/s | 18.248 GFLOP/s |
| `compute.reduction.max.fp32` | 40.261 GFLOP/s | 48.367 GFLOP/s |
| `compute.reduction.sum.fp32` | 43.434 GFLOP/s | 56.293 GFLOP/s |
| `compute.scalar.add.fp32` | 3.106 GFLOP/s | 3.447 GFLOP/s |
| `compute.scalar.divide.fp32` | 2.836 GFLOP/s | 3.121 GFLOP/s |
| `compute.transcendental.exp.fp32` | 5.885 GFLOP/s | 6.090 GFLOP/s |
| `compute.transcendental.rsqrt.fp32` | 3.710 GFLOP/s | 3.883 GFLOP/s |
| `memory.broadcast-read-write.fp32` | 144.686 GB/s | 147.342 GB/s |
| `memory.elementwise-read-write.fp32` | 140.874 GB/s | 146.405 GB/s |
| `memory.row-reduction.fp32` | 184.606 GB/s | 232.478 GB/s |
| `memory.row-scalar-read-write.fp32` | 58.920 GB/s | 63.824 GB/s |

The memory-pattern rates count logical traffic at the configured working sets
and include the probe's minimal operation. They are not DRAM physical-bandwidth
claims and must not replace `memory.shared` or a Surface at another Shape.

## Why promotion was rejected

Repeated strict preflight attempts failed both fixed v2 gates:

- normalized one-minute load had to be at most `0.25`;
- normalized competing CPU had to be at most `0.10`.

The final guarded attempt observed load `0.436767578125` and competing CPU
`0.3411`, with `load-above-policy` and
`total-competing-cpu-above-policy`. The attempt did not start a benchmark and
did not overwrite `specs/hardware-capabilities/apple-m4-cpu-local.yaml`.

The exploratory profile records the same fail-closed boundary. Manually changing
its eligibility bit would forge evidence and is prohibited. `SpecRepository`
rederives the complete profile from the digest-verified raw observation and
rejects any edited environment, cohort, target, suite, resource, rate, probe, or
aggregation field.

## Next trusted attempt

Run only after strict preflight passes:

```sh
uv run groundupscale benchmark-hardware \
  specs/microbenchmarks/apple-m4-cpu.yaml \
  --repository-root . \
  --observation-output \
    goal_process/mac-transformer-ir-calibration-slice/evidence/apple-m4-cpu-phase-microbenchmark-observation-v3.json \
  --profile-output specs/hardware-capabilities/apple-m4-cpu-phase-local.yaml \
  --profile-name apple-m4-cpu-phase-local \
  --profile-version 0.2.0 \
  --require-valid-environment \
  --json
```

Only after that command succeeds may the canonical Analysis Plan atomically
switch its profile reference to `apple-m4-cpu-phase-local.yaml@0.2.0`, recompile,
and produce a new prediction-versus-observation Run Bundle. The legacy
`apple-m4-cpu-local.yaml@0.1.0` remains replayable until then.
