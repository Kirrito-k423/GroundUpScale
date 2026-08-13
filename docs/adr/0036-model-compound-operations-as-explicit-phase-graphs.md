# Model compound operations as explicit phase graphs

Status: Accepted

Decision source: the prediction-versus-observation investigation of Apple M4
CPU Softmax and RMSNorm, where one semantic leaf contained multiple dependent
computations and a whole-operation roofline hid their serialization.

## Context

A semantic leaf is not necessarily one physically atomic computation. Stable
Softmax requires a maximum reduction before normalization inputs are known,
then exponentiation and a sum reduction before the final division. The current
RMSNorm reference implementation squares its input, reduces it, forms a mean,
adds epsilon, computes a reciprocal square root, and applies two scales.

Aggregating all FLOPs and input/output bytes and then applying one
`max(compute_time, memory_time)` incorrectly permits work in dependent phases to
overlap. It also hides implementation-added intermediate materialization and
may apply a matrix-multiply capability to reductions or transcendental work.
Conversely, expanding every internal phase into a public Semantic IR operation
would confuse model semantics with one implementation's kernel structure.

Some implementations can pipeline independent chunks across phases, fuse
intermediate materialization, or overlap different resources. Those are real
optimizations, but they are not consequences of a tensor Shape or an operation
name. Treating them as implicit makes the predicted duration impossible to
audit or replay.

## Decision

GroundUpScale represents every supported Compound Operation with an explicit,
hardware-independent Operator Phase Graph in Cost IR. Each Operator Phase has
a stable phase identity, operation class, Resource Demands, input/output roles,
predecessor phase identities, assumptions, and provenance. The phase metrics
sum to the operation's minimum mathematical work, while implementation-added
materializations remain the responsibility of an Implementation Candidate.

A Hardware Backend maps the phase graph to a candidate-specific internal event
graph. Each phase receives a Local Hardware Floor from capability classes that
match its operation and resource semantics. Missing reduction,
transcendental, memory-pattern, or other required capability produces a
structured `unknown`; the backend must not silently substitute an unrelated
matrix or bulk-copy rate and present it as an implementation prediction.

An exact operation-class probe may time one complete phase invocation, including
its data movement. In that case a separately measured memory-pattern term is an
independent lower-bound constraint on the same invocation, not another serial
step. The candidate composes those two constraints as
`T_local = max(T_exact_operation_probe, T_memory_pattern_floor)` and retains both
capability-profile references. Summing the two would count one invocation twice.
This rule does not authorize overlap between different phases.

Across phases, dependency composition is the critical path. When every phase
depends on the previous phase and the candidate has no Chunk Pipeline Contract,
the selected phase schedule is serialized and its duration is:

`T_compound_serial = sum_phase(T_local_phase)`

A Chunk Pipeline Contract is required before work from different chunks may
overlap. It records the chunk partition, per-chunk and cross-chunk dependencies,
resource claims, startup, steady-state initiation interval, drain, evidence,
and validity domain. The resulting latency preserves startup, steady-state, and
drain separately. Row independence, vectorization, fusion, or a runtime kernel
name alone is not pipeline evidence.

An explicit fused candidate may eliminate or retain phase materialization and
may provide a different internal schedule. It must preserve the mathematical
phase dependencies and explain every changed Resource Demand. A fused runtime
entry point does not collapse the phase graph by default.

The initial required phase graphs are:

- Softmax: `max_reduce -> subtract -> exp -> sum_reduce -> normalize`.
- RMSNorm: `square -> reduce_sum -> mean_scale -> epsilon_add -> rsqrt ->
  input_scale -> weight_scale`.

Cost IR, the selected Implementation Candidate, Run Bundle machine output, and
the human report preserve the same phase identities, dependency edges, local
times, composition policy, missing capabilities, and total. Reports do not
recompute the phase schedule.

When all exact phase capability classes are present but their measurement
environment is ineligible, the backend may also expose a numeric
`provisional_estimate_ns` for planning. This is a separate, non-overwriting
result with `provisional_evidence_tier=exploratory` and explicit reason codes;
the authoritative selected phase duration remains `unknown`. Prediction and
Observation carry independent evidence tiers and reasons in both Machine output
and the human report. A provisional value may feed an explicitly labelled
exploratory Top 10 or gap ratio, but it must never enter an Operator Frontier,
calibration, `relative_prediction_error`, diagnosis Verdict, or a "largest
discrepancy" diagnosis.

## Consequences

- Complex operators remain compact semantic leaves while their performance
  model becomes inspectable to the concrete computation phase.
- Serialized dependencies can no longer disappear behind an aggregate
  roofline, so a lower bound may increase even when total FLOPs are unchanged.
- Credible pipeline predictions require more implementation and measurement
  evidence, including startup and drain behavior.
- Existing aggregate-only Cost IR consumers must treat an absent phase graph as
  an atomic legacy operation; new compound-operation lowerings must provide the
  graph.
- Physical Floor, phase-schedule reference, implementation prediction,
  Operator Frontier, and Observation remain non-overwriting result layers.
- An exploratory profile still yields a visible planning number instead of a
  blank report, while its evidence watermark remains machine-readable and
  human-visible at every comparison boundary.

## Rejected alternatives

- Treat every Semantic IR leaf as one atomic duration region.
- Expand implementation phases into public Semantic IR child operations.
- Apply one whole-operation `max(sum(compute), sum(memory))` across dependent
  phases.
- Infer chunking, fusion, vectorization, row pipelines, or phase overlap from an
  operation or runtime-kernel name.
- Substitute matrix-multiply or bulk-copy throughput for a missing reduction or
  transcendental capability without labeling the dependent result `unknown`.
