# Keep Cost IR symbolic and hardware-independent

`CostLowerer` converts the complete `SemanticIR` into symbolic
`ResourceDemand`s for compute, memory, communication, storage, host work, and
lifetimes while retaining formulas, assumptions, bounds, dependencies, and
provenance. `CostIR` does not contain device-specific duration, utilization, or
physical resource assignment; those appear only after implementation selection
against hardware capabilities and the concrete Fabric Graph.
