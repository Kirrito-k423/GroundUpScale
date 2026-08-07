# Diagnose cross-hardware performance gaps with evidence-qualified frontiers

Status: Accepted

Decision source: [Wayfinder map #1](https://github.com/Kirrito-k423/GroundUpScale/issues/1),
its closed research/prototype tickets #2–#7, and the consolidated
[specification #8](https://github.com/Kirrito-k423/GroundUpScale/issues/8).

## Context

A theoretical peak, an algorithm-independent hardware floor, the best measured
operator implementation, a feasible end-to-end schedule, and one runtime
observation answer different questions. Collapsing them into one number makes
the resulting comparison impossible to interpret: a physical lower bound is
mistaken for a point prediction, a slower observation lowers historical
expectations, profiling overhead looks like a regression, or a sparse Shape
sweep is interpolated across an unvalidated kernel regime.

Cross-hardware diagnosis adds another constraint. Hardware backends cannot
promise identical counters, timers, or runtime behavior, but they must preserve
the same evidence semantics. Missing evidence must remain missing rather than
being encoded as zero, and a Frontier established on one hardware cohort cannot
be transferred to another cohort by assumption.

## Decision

GroundUpScale diagnoses performance gaps with four non-overwriting time axes:

1. **Resource Physical Floor** is derived from algorithm-independent minimum
   resource demands and validated resource rates. It may be unattainable and is
   not a current implementation prediction.
2. **Operator Achievable Frontier** is supported by correct, stable,
   reproducible, independently held-out implementation evidence in one complete
   execution domain and Hardware Validity Cohort.
3. **Schedule Achievable Frontier** composes Operator Frontiers only through
   explicit Semantic/Execution dependencies, resource claims, transformations,
   and validated schedule candidates.
4. **Observation** records what one declared timing or diagnostic lane measured
   under an explicit Completion Boundary and Instrumentation Profile.

No layer may overwrite or silently calibrate another. A distance from a
Physical Floor is optimization headroom, not prediction error.

### Frontier qualification and Shape domain

An observation has an observation-validity state and a separate Frontier-role
state. Only `QUALIFIED + ACTIVE` evidence may become an authoritative Anchor.
Qualification requires complete identity, correctness, a closed timing
boundary, warmup, independent-session repeatability, environment eligibility,
exact-Shape best-of-correct candidate comparison, and independent holdout.

Capability Surface queries are partial functions. The authoritative baseline is
a validated-domain-filtered, candidate-family-specific, local simplicial
piecewise-linear interpolation of effective rate. Exact Anchors are knots of the
same query path. A query outside a retained cell, across an unvalidated
alignment, working-set, dispatch, or candidate-support regime, or without
calibrated uncertainty returns structured `unknown`; it never falls back to a
global P80, a bounding box, nearest neighbor, or silent extrapolation.

Anchor, interpolation/model, and instrumentation uncertainty remain separate.
Their combination policy, target coverage, calibration evidence, and version
are explicit. The propagation term `u_anchor² = lambda^T Sigma lambda` is only
the Anchor component and is not a complete uncertainty model.

### Cross-hardware evidence contract

Each Hardware Adapter preserves a stable device/partition/topology/software
identity, the complete execution domain, correctness, raw timing, timer source
and resolution, warmup/repetitions, Completion Boundary, Instrumentation
Profile, environment preflight, and Measurement Capability Manifest.

Baseline Timing and Diagnostic Profiling are paired but separate lanes. Only
Baseline Timing is Frontier-eligible by default. Diagnostic timing may qualify
only after an independent overhead ablation fits the declared Error Budget and
the instrumentation mode is part of the validity domain.

Optional counters use explicit field states such as `measured`, `derived`,
`declared`, `unsupported`, `permission_denied`, `not_requested`,
`not_applicable`, `collection_failed`, or `unknown`. Missing required identity,
correctness, Completion Boundary, or the primary timer prevents Anchor
qualification. Missing optional evidence makes only the dependent attribution
unknown.

Unvalidated changes to stable hardware, software, numeric/execution, timing, or
communication identity split the cohort. Transient health, contention,
throttling, dispersion, timer, device, or collection failures are quarantined
and retried; they do not define a slower cohort.

### Schedule and trace semantics

Candidate-local resource times use `max` only for explicitly declared overlap.
Across candidates, the Schedule Frontier follows explicit Semantic data edges
and Execution order/resource edges. Fusion, concurrency, communication masking,
contention, and dispatch effects without typed evidence are `unknown`.

Trace attribution uses mutually exclusive leaves plus an explicit unattributed
residual. Parent spans are drill-down indexes and are not added again. A valid
counterfactual preserves leaf identity and proves that its recovered time equals
the explicitly removed ledger entries. It never rewrites the Operator Frontier.

### Trigger and Verdict semantics

Predicted and observed Top 10 lists are selected independently and then joined.
Deep diagnosis triggers only when the absolute gap exceeds combined uncertainty
and the item is in either Top 10 or exceeds one tenth of the E2E observation.
An anomalous Shape then receives an exact-Shape disambiguation probe with
semantic, Shape, dtype, layout/alignment, threads, cohort, candidate set,
correctness, and timing boundary locked.

The result vocabulary is `frontier_shift`, `implementation_headroom`,
`integration_overhead`, `suspected_regression`, `insufficient_evidence`, and
`confirmed_bug`. Every result records satisfied, failed, and unevaluated gates
and direct evidence references. `frontier_shift` requires C2/C3 candidate
coverage, independent holdout, at least three independent sessions, stable
neighboring Anchors, one cohort, and local Shape disambiguation. A pure latency
gap cannot confirm a bug; `confirmed_bug` requires direct reproducible
correctness or contract-violation evidence.

The automatic gate for `suspected_regression`, multi-Verdict precedence, and
other unresolved policies are intentionally not invented. Until decided, the
engine fails closed with `insufficient_evidence` while preserving the evidence.

### Evidence and replay

Every conclusion references a reproducible Diagnostic Evidence Bundle containing
resolved configuration/IR, hardware and cohort identity, execution domain,
candidate coverage, correctness, environment, paired lane identities, raw
timing and exclusions, timer/completion metadata, Surface/Anchor versions and
uncertainty, schedule/trace/alignment/ledger, ablations, Verdict gates, and
input/evidence digests. Evidence changes produce new immutable versions; old
queries and state transitions remain replayable.

The normative field semantics, configurable policies, unresolved unknowns, and
conformance matrix are defined by the
[cross-hardware performance-gap diagnosis specification](../methods/cross-hardware-performance-gap-diagnosis.md).

## Consequences

- A valid diagnostic result may contain several `unknown` fields. This is a
  successful, honest result rather than a partial failure to hide.
- Cross-hardware reports compare independently established cohort results and
  explicitly defined efficiency metrics; they do not transfer a Frontier or
  require identical counters.
- Surface construction, evidence qualification, schedule composition, and
  Verdict evaluation require versioned policies and more stored evidence than a
  single benchmark table.
- Existing Run Bundle, Stable Path, Explanation Graph, Alignment Map, Metric
  Derivation, and Error Attribution concepts are the preferred implementation
  seam, but their final schema layout is not decided by this ADR.
- The synthetic Issue #7 value `51.632 ms` is an aggregate of 24 prototype
  operator nodes. It is not a single MatMul measurement or a calibrated M4
  Operator Frontier. Synthetic ledger fixtures test conservation only and may
  not become production calibration.

## Rejected alternatives

- Treat a theoretical peak, global P80, single implementation, or one
  observation as a universal achievable Frontier.
- Interpolate across a full convex hull, per-axis bounding box, nearest neighbor,
  or mixed candidate-family point set without validated support.
- Let diagnostic profiling replace minimally intrusive baseline timing.
- Infer fusion, concurrency, communication overlap, resource contention, or a
  bug from an unexplained latency gap.
- Reuse another hardware cohort's Surface or encode unsupported counters as
  zero.
