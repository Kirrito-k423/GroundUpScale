# Use a staged multi-IR compilation pipeline

GroundUpScale replaces the early three-IR framing in ADR-0001 with a staged
pipeline: Model Spec and Workload Spec independently produce Model IR and
Workload IR; Semantic Compiler combines them with analysis and logical strategy
effects into Semantic IR; Cost Lowerer produces hardware-independent Cost IR;
Hardware Backends and Execution Planner select physical implementations into
Execution IR; Scheduler Plugins then produce Schedule Results. Keeping each
representation responsible for one class of facts makes every transition
independently extensible, verifiable, and explainable.
