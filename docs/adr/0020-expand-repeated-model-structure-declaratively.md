# Expand repeated model structure declaratively

Model Spec expresses repeated and heterogeneous module structure through
Schema-defined `repeat`, `template`, and `overrides` forms. Module Builder
expands these forms into explicit Model IR nodes with stable indexed paths and
provenance; a strictly more-specific selector may refine a broader selector,
while incompatible incomparable matches are rejected as ambiguous. This keeps
large models concise without confusing model-layer construction with Workload
IR `Map`, runtime loops, or hidden Python builders.
