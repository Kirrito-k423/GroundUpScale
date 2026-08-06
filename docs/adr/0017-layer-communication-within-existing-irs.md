# Layer communication semantics within the existing IRs

GroundUpScale represents communication as hardware-independent
`LogicalCommunication` in `SemanticIR`, symbolic `CommunicationDemand` in
`CostIR`, selected algorithms and routes in `CommunicationPlan`, and concrete
resource-claiming `PhysicalCommunicationEvent`s in `ExecutionIR`. Logical
payload, per-rank algorithm volume, and physical per-link traffic remain
separate, so strategy plugins express collective intent while planners and
schedulers resolve topology, staging, contention, and overlap without a
separate top-level Communication IR.
