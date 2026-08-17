---
status: superseded by ADR-0039
---

# Separate authority from tiered iteration values

GroundUpScale preserves fail-closed authoritative results, but every prediction–observation iteration report also publishes non-empty numeric Report Values for E2E and component attribution. Missing authoritative evidence remains `structured-unknown`; a separate versioned derivation selects the best available value, labels Evidence Grade A–D and Generation Stage, supplies uncertainty, method, provenance and permitted use, and never promotes a degraded estimate into Frontier or calibration evidence.

## Decision

- Grade A is authoritative evidence usable for acceptance and calibration; B is reproducible direct measurement usable for comparison; C is an explicit proxy derivation usable for prioritization; D is a model-degraded estimate usable only to form the next hypothesis.
- Prediction fallback is: complete Schedule Achievable Frontier, current implementation Duration Model, partial schedule plus explicit proxy estimates, then Resource Demand plus conservative efficiency and scheduling assumptions. A Resource Physical Floor alone is never relabeled as a point prediction.
- Observation-side attribution fallback is: directly attributable baseline evidence, reproducible but incompletely qualified measurement, overhead-adjusted diagnostic proportions scaled to baseline E2E, then model weights scaled to the measured E2E. The last case is named Observation-side Degraded Estimate rather than Observation.
- Predicted and observation-side Top 10 lists are selected independently and joined by exact Stable Path. Exclusive E2E Contributions plus an explicit framework, scheduling, or unattributed residual reconcile each side to 100%; overlap and inclusive parents remain separate navigation views.
- Every displayed value carries an uncertainty interval. Grade B has at least ±15%, C at least ±30%, and D at least `[0.5×, 2.0×]` unless stronger evidence justifies a wider interval.
- A/A comparisons may be called prediction or acceptance error. Any comparison containing B–D is an Exploratory Gap and cannot trigger automatic calibration, promotion, or a conclusive verdict.
- The default human report is Simplified Chinese and is projected with JSON and CSV from the same machine Report Value set. Existing Run Bundles remain immutable; new report semantics require a new Schema, policy version, and Run Bundle identity.

## Consequences

The user always receives numbers that support another iteration, while authority and evidence boundaries remain honest. Verifiers must replay both the authoritative result and the separate Report Value derivations, grades, uncertainty, Top 10 selection, reconciliation, and all presentation formats. This ADR narrows ADR 0031 and ADR 0035 only at the iteration-report projection: their no-silent-fallback, no-zero-fill, Frontier-promotion, cohort, and diagnostic-verdict rules remain unchanged.

## Rejected alternatives

- Leave report cells empty whenever authority is unknown; this blocks prioritization and iteration.
- Fill authoritative fields with heuristics; this destroys the distinction between evidence and estimates.
- Show unlabeled point estimates or a single confidence percentage; this hides derivation stage, uncertainty and permitted use.
