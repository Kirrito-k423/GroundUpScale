# Separate logical state, replicas, and physical allocations

GroundUpScale distinguishes versioned logical `StateArtifact`s, placed or
sharded `ArtifactReplica`s, and concrete `MemoryAllocation`s. Execution plans
explicitly represent allocation, materialization, access, aliasing, migration,
eviction, release, and recomputation so FSDP temporaries, activation offload,
checkpointing, KV-cache growth, workspaces, allocator reserve, and fragmentation
produce schedule-dependent capacity timelines without double-counting views or
logical state.
