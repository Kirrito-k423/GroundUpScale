# Model operator Shape response in latency space

Status: Accepted

GroundUpScale refines the Operator Achievable Frontier from ADR 0035 by treating latency, rather than Effective Rate or Model FLOPs Utilization, as the primary response across a validated Shape Regime. A Duration Model may decompose latency into Setup Latency, declared work divided by an asymptotic rate, and an explicit regime-local residual; Effective Rate is derived as declared work divided by latency, and Model FLOPs Utilization is derived only when a comparable evidence-backed theoretical peak exists. This choice captures the observed small-Shape ramp and large-Shape steady behavior without adding a fifth performance-truth layer or allowing an Observation to overwrite the Frontier.

Capability Surfaces partition Shape space into evidence-qualified Ramp Regimes, Steady Regimes, and explicit Shape Regime Boundaries. Alignment, working-set, candidate-family, kernel-dispatch, or other execution changes that invalidate one response must split the Surface; interpolation across an unvalidated boundary returns `unknown`. Transition locations such as MatMul `M≈512` or TND FlashAttention sequence length `≈4K` are hypotheses to discover and validate for a complete execution domain, not universal constants to hard-code.

This ADR supersedes only ADR 0035's choice of local piecewise-linear interpolation of Effective Rate. ADR 0035's four non-overwriting axes, qualification gates, partial-function behavior, uncertainty separation, cohort isolation, and no-extrapolation rules remain in force.

## Considered options

- Directly interpolate Effective Rate or Model FLOPs Utilization. Rejected because fixed Setup Latency makes both strongly nonlinear in the ramp and because Model FLOPs Utilization may be undefined when the comparable theoretical peak is unknown.
- Fit one global black-box curve across all Shapes. Rejected because kernel dispatch, alignment, and working-set changes create evidence-significant Shape Regime Boundaries that must remain explainable and fail closed.
- Add a fifth ramp-specific performance layer. Rejected because ramp and steady behavior describe the Shape response of the Operator Achievable Frontier rather than a distinct source of performance truth.

## Consequences

- Qualification and query results preserve latency, Effective Rate, uncertainty components, response identity, Shape Regime identity, and direct evidence references.
- MatMul varies `M` independently while preserving declared `N` and `K`; square-Shape evidence cannot silently stand in for an `M` sweep.
- TND FlashAttention records enough sequence-distribution and execution-domain information to distinguish packed workloads that share total tokens but have different attention work.
- Fit, holdout, and boundary-confirmation evidence remain independent, and every fitted parameter or retained cell is versioned and replayable.
