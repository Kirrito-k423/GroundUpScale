# Require provenance for every transformation

Every formal transformation and plugin must emit immutable
`DerivationRecord`s into a mandatory append-only `ProvenanceGraph`, linking
stable user-facing paths and per-compilation node identities across specs, IRs,
candidate selection, execution, and report metrics. Records include versioned
rules, configurations, formulas, assumptions, bounds, evidence, alternatives,
rejection reasons, warnings, and validation results; transformations without
sufficient provenance cannot produce a formal compilation result.
