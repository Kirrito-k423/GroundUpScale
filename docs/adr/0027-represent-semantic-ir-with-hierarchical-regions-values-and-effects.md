# Represent Semantic IR with hierarchical regions, typed values, and state effects

Semantic IR preserves workload and model nesting through hierarchical Semantic
Regions instead of flattening the program into one DAG. Dataflow inside and
across regions uses Typed Values, while all logical state interaction is
declared through State Effects. Semantic Compiler recursively expands composite
entrypoints, invokes registered lowerers for primitive semantics, and applies
immutable Transform Proposals in deterministic named phases, retaining
provenance and rejecting conflicts. This representation supports nested models,
structured loops, services, and deployment strategy effects without admitting
hardware latency or physical placement into Semantic IR.
