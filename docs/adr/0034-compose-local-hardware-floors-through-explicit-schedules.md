# Compose local hardware floors through explicit schedules

ADR 0033 defines an algorithm-independent floor for one eligible resource-overlap
region. This decision makes that region explicit: a Hardware Backend may use
`max(compute, memory, communication)` only inside one Implementation Candidate
whose overlap contract permits it. An unfused operation is one candidate; a fused
module or scope is one candidate only when the backend actually provides that
fused implementation. Alias-only operations such as views and transposes contribute
zero physical memory traffic unless their selected implementation materializes data.
The compiler must not deduplicate intermediate traffic merely because operations
share a parent scope.

Across candidates, GroundUpScale composes local floors through explicit Cost IR or
Execution IR dependencies and Resource Claims. A serialized-unfused reference is
`sum_i local_floor_i`. A dependency-only optimistic bound is
`max(longest_dependency_path, max_r(sum_i resource_time_i[r]))`. A chosen feasible
schedule remains the responsibility of the Execution Planner and Scheduler, which
may add contention, launch, synchronization, queueing, and capacity effects. Results
must preserve the local candidates, serialized bound, dependency critical path,
shared-resource bound, ideal DAG bound, selected schedule assumption, and selected
bound as distinct explainable values.

This prevents an impossible whole-layer roofline from hiding serialization while
retaining a useful ideal-overlap reference. It also makes future fusion, pipeline,
compute/communication overlap, and asynchronous transfer support additive: plugins
provide different candidates or schedules instead of changing the meaning of the
hardware floor. Existing scope-level results that used
`max(sum(compute), unique_boundary_memory)` are reinterpreted only as an optimistic
resource bound when justified; they are not the latency floor of a serialized
multi-operation execution.
