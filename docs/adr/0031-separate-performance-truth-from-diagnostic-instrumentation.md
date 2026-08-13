# Separate performance truth from diagnostic instrumentation

Each measurable case supports a minimally instrumented benchmark run, a
structured diagnostic trace run, and optional targeted deep-probe runs. Hooks
emit scope correlation and tensor metadata while framework and runtime
collectors capture operator, device, memory, and scheduling events; per-module
device synchronization and unstructured printing are not accepted as E2E
performance truth. Raw evidence remains immutable, and an Alignment Map relates
normalized observations to compiled identities with explicit confidence and
unattributed buckets.

## Decision

Every hardware adapter exposes two paired evidence lanes:

- The **Baseline Timing Lane** collects only the synchronization and metadata
  required for a trustworthy elapsed-time distribution. It is the only lane
  whose timing may qualify as Frontier Evidence by default.
- The **Diagnostic Profiling Lane** collects PMU, CUPTI, profiler, trace,
  replay, cache, memory, communication, and scheduling evidence for the same
  Case, Shape, implementation, and Hardware Validity Cohort. A stable pair
  identity links it to the baseline. Its timing is diagnostic unless an
  independent ablation proves that the exact Instrumentation Profile fits the
  declared overhead and uncertainty budget.

This split is required because profiling can serialize work, replay kernels,
sample rather than count every event, require extra passes, conflict with other
collectors, or reduce communication bandwidth. Turning a profiler on therefore
changes evidence conditions and cannot silently redefine achievable
performance.

## Cross-hardware adapter contract

The portable contract standardizes evidence semantics, not identical counters.
An adapter provides five operations:

1. `discover_capabilities()` records timers, counters, permissions,
   unsupported features, conflicts, resolution, scope, attribution, and
   intrusion in a Measurement Capability Manifest.
2. `fingerprint_cohort()` records the complete Hardware Validity Cohort.
3. `preflight()` records transient power, thermal, health, frequency, and
   contention state and returns eligible or quarantine.
4. `build_timing_plan(case)` maps the requested Completion Boundary to CPU
   thread or rank completion, CUDA events and synchronization, or Ascend stream
   and event synchronization.
5. `collect()` emits immutable raw samples and status-rich observation fields
   with Diagnostic Evidence Bundle references.

For every backend, a Frontier-eligible baseline requires:

- stable device identity, partition and topology;
- OS, kernel, driver, firmware, runtime, framework, compiler, operator-library,
  and communication-library versions as applicable;
- dtype, numeric mode, layout, threads or affinity, NUMA, context, stream,
  graph or eager mode, concurrency, and power or clock policy;
- correctness, warmup, repetitions, raw samples, monotonic host elapsed time,
  timer source and resolution, and an explicit Completion Boundary;
- instrumentation mode, adapter and protocol versions, plus a Measurement
  Capability Manifest.

The backend matrix uses `R` for required Frontier evidence, `O/U` for an
optional field whose adapter must report either a value or a precise
unavailable status, and `N/A` for a concept that does not apply:

| Evidence capability | CPU | CUDA GPU | Ascend NPU |
| --- | --- | --- | --- |
| Stable device, partition, topology, software stack, and execution-domain identity | R | R | R |
| Power and performance policy status | R | R | R |
| Monotonic host elapsed time and timer resolution | R; asynchronous thread pools must join | R; captures dispatch and synchronized E2E | R; captures dispatch and synchronized E2E |
| Native device elapsed time | N/A for an ordinary synchronous CPU kernel | R; events plus declared stream and synchronization scope | R; same-stream events plus stream synchronization |
| Correctness, warmup, repetitions, and raw samples | R | R | R |
| Measurement Capability Manifest and instrumentation mode | R | R | R |
| Kernel or task timeline and correlation IDs | O/U: signposts, perf, or trace | O/U: CUPTI Activity | O/U: msprof or torch_npu profiler |
| Compute, cache, and memory-traffic counters | O/U: PMU-specific and permission-sensitive | O/U: CUPTI or Nsight, potentially replayed | O/U: product-, level-, and option-sensitive AI Core metrics |
| Physical link topology and counters | O/U: NIC, RDMA, or uncore and often system-scoped | O/U: NVML, NVLink, PCIe, or CUPTI | O/U: HCCS, PCIe, RoCE, or system-interconnection profiling |
| Communication semantics and completion, when communication exists | R; local rank durations aggregated without cross-node timestamp subtraction | R; collective identity and stream Completion Boundary | R; HCCL identity and stream Completion Boundary |

Native device elapsed time is additionally required for asynchronous CUDA and
Ascend operator timing. Communication cases require semantic bytes, operation,
ranks, topology, backend or algorithm identity, and a completion duration
measured in each rank's local timer domain before aggregation; absolute clocks
from different nodes are not subtracted.

Compute-pipeline, cache, DRAM or HBM traffic, link, utilization, power,
frequency, and detailed timeline counters are optional diagnostic evidence.
Their absence does not invalidate an otherwise eligible baseline, but any
diagnosis that depends on them is `unknown` or `insufficient_evidence`.

## Availability and degradation semantics

Every declared field carries an Observation Field Status. `unsupported`,
`permission_denied`, `not_requested`, and `collection_failed` remain distinct
in raw evidence even if a report groups them as `unknown`. Missing evidence is
never encoded as zero.

A Proxy Metric is permitted only when its derivation and attribution are
explicit. Semantic bytes divided by completion time may be reported as
effective algorithm bandwidth, not physical link or DRAM traffic. Modeled work
divided by time may be reported as effective throughput, not executed hardware
operations. Device-wide or system-wide utilization and traffic cannot be
attributed to one Case without additional isolation or correlation evidence.

Missing required identity, correctness, Completion Boundary, or primary timing
makes the run ineligible for Frontier Anchor promotion. Missing optional
counters preserves baseline eligibility but narrows the claims that can be
made. An adapter must never fill a missing field with measurements from another
hardware cohort.

## Cohort changes and transient failures

Device, partition, topology, software-stack, numeric-mode, execution-mode,
power-policy, timer, synchronization, adapter, or measurement-protocol changes
split the Hardware Validity Cohort unless an explicit A/B equivalence study
qualifies the change. Communication rank count, topology, backend, algorithm,
and routing policy are also validity-domain inputs.

Thermal or power throttling, health warnings, competing workloads, high sample
dispersion, timer failures, device errors, dropped records, and profiler
conflicts quarantine and retry the run. They do not justify creating a new,
slower cohort to preserve bad evidence.

## Relationship to Shape interpolation

A Capability Surface consumes only qualified Baseline Timing Lane evidence.
It remains exact at Frontier Anchors and continuous inside each connected
Validated Shape Regime. Alignment, working-set, candidate-support, kernel, or
algorithm seams without independent continuity evidence are outside the
authoritative function domain and return `unknown`; `unknown` is absence of an
authorized value, not a discontinuous capability value. A continuous
provisional baseline may guide a Shape Disambiguation Probe but cannot enter a
diagnosis verdict.
