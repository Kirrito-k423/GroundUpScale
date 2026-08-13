---
name: report-prediction-observation
description: Generate, update, review, or implement explainable predicted-versus-observed performance reports for latency, throughput, utilization, memory, or other resources. Use whenever Codex handles prediction versus measurement/actual values, E2E gaps, benchmark comparisons, performance regressions, calibration summaries, CI performance results, dashboards, Web reports, or hotspot investigations. Require independent predicted and observed Top 10 plus every item at or above 10% of its own E2E, machine-readable decomposition, visible evidence gaps, hierarchical drilldown, reconciliation, and report-generator test gates.
---

# Report Prediction Observation

Produce a reconciled comparison that lets a reader move from E2E deviation to the
modules and operations responsible for it. Never stop at one aggregate ratio when
finer aligned evidence exists.

## Executable enforcement

When the repository contains a report generator, dashboard builder, or CI comparison
pipeline, this skill is not satisfied by an agent-written summary alone. The default
generation path must:

1. construct the independent predicted and observed decompositions;
2. enforce the Top-10 union 10%-of-E2E visibility rule in code;
3. serialize the decomposition as a machine-readable artifact;
4. render it in the human-facing report; and
5. fail a test or completion gate when either side is silently absent.

Insufficient evidence may disable one side only when the generated report names the
exact evidence boundary and required instrumentation. It must never silently omit the
section. Treat every code path that calls a predicted-versus-observed comparison
builder as a trigger site for this enforcement.

## Establish comparable evidence

1. Record workload, Shape, dtype, hardware cohort, placement, software version,
   environment validity, warmup, sample count, statistic, and synchronization.
2. Name the modeled quantity exactly: vendor-theory floor, empirical hardware
   floor, implementation prediction, scheduled prediction, or calibrated prediction.
3. Treat distance from a hardware floor as optimization headroom, not prediction
   error. Calculate prediction error only for an implementation or scheduled
   prediction with a comparable observation.
4. Align nodes by stable identity and report alignment coverage. Do not silently
   compare nodes that merely share display names.
5. Distinguish a repeated benchmark statistic from a single diagnostic trace.
   Never present one trace span as the median of the E2E benchmark.

## Build non-overlapping time decompositions

Create separate predicted and observed decompositions before joining them.

- Preserve hierarchy and show inclusive parent scopes for navigation.
- Rank mutually exclusive siblings, leaf spans, or exclusive parent time for time
  composition. Do not sum an inclusive parent with its descendants.
- Derive exclusive time from interval unions when timestamped children exist.
- For overlapping execution, use critical-path contribution, interval union, and
  explicit overlap; do not force concurrent durations into a serial sum.
- For device-asynchronous traces, do not treat host enqueue spans as device time.
- Include `other` or `unattributed` so the decomposition reconciles to its parent.

For one Implementation Candidate, first derive a local floor such as:

```text
t_i = max(compute_i / compute_rate,
          materialized_bytes_i / memory_bandwidth,
          communication_i / communication_bandwidth)
```

Take `max` only across resources whose overlap is explicitly allowed inside that
candidate. Then follow the declared schedule:

```text
serialized-unfused = sum_i(t_i)
ideal-DAG           = max(longest dependency path of t_i,
                          max_r(sum_i(resource_time_i[r])))
```

For a serialized schedule, the local candidate floors are the additive predicted
decomposition. For an overlapping schedule, attribute time using critical-path
contribution, resource load, and explicit overlap rather than a serial sum. Use a
whole-scope boundary-byte roofline only when an explicit fused Implementation
Candidate supplies that boundary and its internal materialization semantics. Always
show the selected schedule alongside the bound; preserve the serialized and ideal-DAG
references separately when both are available.

## Apply the mandatory visibility rule

Apply this rule independently to predicted and observed time:

```text
mandatory = every mutually exclusive item whose time >= 10% of that side's E2E time
selected  = mandatory union the ten largest items by time
```

- If fewer than ten comparable items exist, report all of them.
- Build both a predicted Top 10 and an observed Top 10. Join the union of their
  identities for the comparison table; do not let one side's ranking hide the
  other side's hotspots.
- Report `time`, `% of own E2E`, rank, and stable path on both sides.
- For the largest mismatched selected scope, repeat the rule on its children and
  continue until reaching leaves or an explicit evidence boundary.
- Continue to show any item above the 10%-of-E2E threshold even when this produces
  more than ten rows.

## Report required sections

Lead with the largest actionable discrepancy, then include:

1. **Comparison contract** — both E2E values, units, statistic, run identities,
   modeled-value class, and whether a relative error is semantically valid.
2. **Predicted time composition** — mandatory items plus predicted Top 10.
3. **Observed time composition** — mandatory items plus observed Top 10.
4. **Joined gap ranking** — stable path, operation, predicted time, observed time,
   absolute gap, ratio, shares of each E2E, and evidence quality.
5. **Hierarchical drilldown** — inclusive scope context followed by non-overlapping
   child composition for the largest gap.
6. **Reconciliation** — selected sum, all-attributed sum, other/unattributed time,
   overlap if any, and coverage percentage.
7. **Conclusion** — distinguish capability-model error, implementation/algorithm
   inefficiency, layout/materialization, scheduling, instrumentation, and noise.

Use milliseconds for readable tables while retaining original-unit values or links
to raw evidence. State the denominator beside every percentage.

## Fail visibly when evidence is insufficient

Do not invent a decomposition from aggregate E2E values. If prediction or
observation lacks aligned child evidence:

- report the available side's mandatory/Top-10 breakdown;
- identify the exact missing spans, counters, identities, or repeated samples;
- mark the unavailable cells as `not measured` or `not modeled`;
- recommend the smallest instrumentation or lowering change needed to obtain them.

## Completion gate

Do not finalize the report unless all applicable checks pass:

- [ ] Every item at or above 10% of predicted or observed E2E is visible.
- [ ] Predicted Top 10 and observed Top 10 are both present, or all available items
      are shown when fewer than ten exist.
- [ ] Inclusive parents are not double-counted with descendants.
- [ ] The largest discrepancy is drilled down to leaves or a named evidence boundary.
- [ ] Reported sums reconcile through `other`, `unattributed`, or overlap.
- [ ] Hardware-floor headroom is not mislabeled as prediction error.
- [ ] Single-trace timing is not mislabeled as a repeated-sample statistic.
