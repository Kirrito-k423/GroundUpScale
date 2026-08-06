# Separate shape and driver analysis conditions

GroundUpScale represents analysis conditions as an `AnalysisCase` composed of
a `ShapeProfile`, `DriverProfile`, and `ObservationWindow`, separate from
`WorkloadIR`. Fixed shapes and fixed iterations are the primary path for
operator, model, and parallel-strategy optimization; shape distributions,
arrival processes, and queueing behavior are optional extensions for dynamic
batching, tail latency, and serving-capacity analysis.
