# Classify workload composition by semantics

`WorkloadIR` distinguishes structured control nodes (`Sequence`, `Parallel`,
`Branch`, `Loop`, and `Map`), temporal `Pipeline`s, long-running `Service`
containers, `ArtifactEdge` and `Stream` data edges, and action leaves. A
`ModelCall` is an expandable action leaf that references a nested `ModelIR` or
named entry point; model-internal repetition stays in `ModelIR`, while
pipeline-parallel deployment stays in `DeploymentIntent`, preventing control,
timing, data-channel, and deployment semantics from being conflated.
