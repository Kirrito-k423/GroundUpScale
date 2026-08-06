# Separate model structure from versioned state

`ModelIR` is an immutable logical structure, while weights, optimizer state,
KV cache, activations, and checkpoints are typed and versioned
`StateArtifact`s. `ModelCall`s consume and produce logical artifacts, and only
`ExecutionIR` materializes their physical replicas and shards; this lets
serving stages share a model definition without conflating their runtime state
and makes RL weight updates, conversions, and transfers explicit.
