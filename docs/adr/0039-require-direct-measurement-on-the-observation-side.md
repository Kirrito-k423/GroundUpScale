---
status: accepted
---

# Require direct measurement on the observation side

Prediction reports may use versioned proxy or degraded models to keep the prediction side numerically actionable, but every value presented on the measured side must derive exclusively from directly recorded samples, device events, or a replayable device timeline for the stated identity and Completion Boundary. When only E2E timing is measured, the measured decomposition contains one `Measured E2E Residual` equal to 100% of that E2E; predicted weights, operator anchors from a different execution context, and diagnostic proxies must not be scaled into measured component rows. This supersedes ADR 0038's observation-side C/D fallback because a complete-looking comparison is less valuable than preserving the semantic boundary between prediction and measurement.

## Consequences

Prediction components may retain Evidence Grades A–D. Measured values are limited to Grades A–B, and their uncertainty must be computed from recorded samples rather than a grade-based synthetic interval. A component enters a measured TOP10 or a row-level prediction-versus-measurement comparison only when its Stable Path has direct, non-overlapping timing evidence; otherwise the report shows it as a prediction and measurement priority while the measured E2E remains reconciled through the explicit residual.
