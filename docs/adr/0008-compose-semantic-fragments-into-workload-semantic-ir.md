# Compose model entrypoint fragments into a workload-wide Semantic IR

Each `ModelCall` entrypoint lowers independently into a reusable
`SemanticFragment`, but the `SemanticCompiler` composes those fragments with
the enclosing `WorkloadIR`, `AnalysisCase`, non-model actions, and applicable
strategy effects. The resulting hardware-independent `SemanticIR` covers the
complete selected workload so that cross-model data flow, control composition,
state lifetimes, and global dependencies remain visible.
