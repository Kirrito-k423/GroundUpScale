# Separate physical execution constraints from scheduling

`ExecutionPlanner` emits an unscheduled `ExecutionIR` containing selected
implementations, concrete resources and routes, event dependencies, duration
models, resource claims, and lifetime constraints. Extensible
`SchedulerPlugin`s produce `ScheduleResult`s with timestamps, allocations,
waits, contention, overlap, bubbles, and capacity timelines; bounds and
alternative scheduling policies can therefore reuse the same physical plan,
and predicted schedules remain distinct from real `ObservationTrace`s.
