# Separate logical workload, analysis conditions, and deployment

GroundUpScale keeps `WorkloadSpec`, `AnalysisCase`, and `DeploymentIntent`
independent: the Workload defines the stable logical process, the Analysis Case
selects input shapes, driving behavior, and observation boundaries, and the
Deployment Intent supplies scoped execution strategies, placement constraints,
and service policies. One Workload can therefore be reused across many Cases
and candidate deployments without copying or mutating its logical graph.
