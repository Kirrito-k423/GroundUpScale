# Separate hardware implementation candidates from global planning

A `HardwareBackend` proposes device-specific `ImplementationCandidate`s for
supported `CostIR` regions, including applicability, event templates, resource
claims, duration models, uncertainty, and evidence. The `ExecutionPlanner`
selects among candidates and performs concrete placement, collective selection,
routing, fusion, static capacity validation, and physical constraint generation;
no local Backend independently emits the complete `ExecutionIR`.
