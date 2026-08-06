---
status: superseded by ADR-0029
---

# Separate three IRs and use scoped deployment intent

GroundUpScale separates logical model structure (`ModelIR`), cross-model and
cross-stage workflow (`WorkloadIR`), and resolved physical execution
(`ExecutionIR`). Hardware is described by `FabricGraph`, while
`DeploymentIntent` binds strategy and placement constraints to selected scopes;
this avoids the ambiguity of a unified graph or global configuration and makes
nested models, cyclic workloads, heterogeneous placement, and cross-hardware
data movement explicit.
