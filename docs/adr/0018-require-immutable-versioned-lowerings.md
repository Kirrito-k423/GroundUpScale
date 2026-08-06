# Require immutable, versioned, and reproducible Lowerings

Every user-extensible transformation receives an immutable typed IR and a fully
pinned `CompilationContext`, then returns a new typed IR together with mandatory
provenance, diagnostics, and validation results. Plugins declare schemas,
versions, applicability, effects, and compatibility; hidden ambient inputs and
in-place mutation are forbidden, and declarative Rule Packs follow the same
contract so compilation is cacheable, reproducible, and attributable.
