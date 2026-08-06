# Model physical events with typed multi-resource claims

Physical events declare typed `ResourceClaim`s over capacity, throughput, slot,
or exclusive resources, including work or amount, allocation bounds, sharing,
lifetime, affinity, and provenance. Backends provide multi-resource performance
models and schedulers resolve simultaneous competition; utilization, peak
capacity, overlap, and causally classified bubbles are derived from the
`ScheduleResult`, never supplied as opaque event-level percentages.
