# PROTOTYPE — Schedule Frontier time ledger

> Throwaway evidence for “原型：从 Operator Frontier 到 Schedule Frontier 的分层差距诊断”.
> This directory must not be promoted into the production implementation.

## Question

Can a small, explicit DAG for the repository's frozen two-layer Transformer keep
Resource Physical Floor, Operator Achievable Frontier, Schedule Achievable
Frontier, and an observed trace separate while supporting a lossless drill-down
from E2E to critical-path stage, module, and operator? In particular, can it
locate integration overhead when every observed operator remains within its
Operator Frontier uncertainty band?

## Falsifiable results, fixed before implementation

The hypothesis is rejected if any condition below occurs in either the baseline
or counterfactual scenario:

1. `ledger_conserved` is false: an E2E duration cannot be reconstructed from
   mutually exclusive operator-execution, explicit schedule-wait, and residual
   integration buckets, or a leaf is counted in two parent scopes.
2. `critical_path_valid` is false: candidate-local resource components do not
   use the declared local `max`/`sum` rule before dependency-path accumulation,
   or the reported Schedule Frontier differs from the longest legal path.
3. `explicit_semantics_enforced` is false: fusion, concurrency, communication
   masking, or resource contention changes time without an explicit input edge,
   candidate policy, or resource claim.
4. `integration_overhead_detected` is false: all operator observations are
   inside their uncertainty bands, E2E is slower than the Schedule Frontier,
   but the verdict is not `integration_overhead`.
5. The counterfactual changes any Operator Frontier value, or its E2E delta is
   not exactly explained by the removed explicit schedule/integration bucket.

Passing means only that this ledger shape can represent the ticket's selected
cases. It does not validate every Transformer, scheduler, or hardware cohort.

## One-command run

```sh
uv run groundupscale-prototype-schedule-frontier --run-all \
  --output /tmp/groundupscale-issue7-raw-result.json
```

Interactive mode uses the same in-memory state and renders the full current
ledger after every action:

```sh
uv run groundupscale-prototype-schedule-frontier
```

Keys switch between the observed baseline, a faster counterfactual schedule,
and a deliberately invalid implicit-overlap counterexample.

## Inputs and provenance

- `input.json` is a deliberately small aggregation of the frozen two-layer M4
  CPU case, not a new benchmark dataset.
- Repository source facts are `specs/plans/mac-cpu-prefill.yaml`,
  `tests/test_cost_totals.py`, `docs/methods/apple-m4-cpu-backend.md`, and
  `docs/methods/prediction-observation-comparison.md` at the captured git commit.
- The fixture retains the published E2E observation (`92.814479 ms`), layer
  observation (`45.059563 ms`), Q-projection observation (`0.154288 ms`), M4
  resource rates, stable paths, explicit dependencies, resource claims, and
  uncertainty bands. Remaining aggregate leaf observations are prototype input
  chosen so that their sum and residual are inspectable; they are marked
  `synthetic_for_prototype` and must not be mistaken for calibration evidence.
- The run output records the fixture SHA-256, git revision, Python/platform
  identity, every raw input row, all derived rows, invariants, Top-10 lists, and
  the rejected implicit-overlap counterexample.

## Exit check

The ticket can pass only when the machine-readable raw result reports all five
falsifiable checks as true, prints a reconstructable time ledger and critical
path, includes at least one valid counterfactual schedule, and preserves the
invalid implicit-overlap attempt as a counterexample rather than silently
accepting it.

## Recorded observation

The captured run is [`raw-result.json`](raw-result.json). On macOS 15.7.4 arm64
with Python 3.11.15, all five pre-registered checks passed:

- Resource Physical Floor: `5.553976 ms`
- Operator Achievable Frontier: `51.632000 ms`
- Schedule Achievable Frontier: `53.232000 ms`
- observed E2E trace: `92.814479 ms`
- exclusive ledger: `51.100746 ms` operator execution + `22.000000 ms`
  schedule wait + `19.713733 ms` integration = `92.814479 ms`
- verdict: `integration_overhead`; all operator observations remained within
  their prototype Frontier uncertainty bands
- explicit batched-dispatch counterfactual: exactly `12.000000 ms` faster with
  unchanged Operator Frontier and unchanged operator observation total
- implicit fusion, concurrency, communication masking, and unclaimed resource
  contention were all rejected with a specific reason

The observed answer is therefore “yes, for this deliberately bounded case.” A
production contract should retain four separate time axes, typed dependency and
resource edges, exclusive trace buckets, independent predicted/observed Top 10,
and `unknown` whenever an overlap or attribution lacks explicit evidence.
