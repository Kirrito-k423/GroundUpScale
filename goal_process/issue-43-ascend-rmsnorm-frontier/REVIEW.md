# Issue 43 Code Review

- Fixed point: `5a0958e75c2c9323d2494136b3b26e1d4ded2b67`
- Reviewed commit: `9a2330c9bb33a8d78ee8ab68320e71c588458326`
- Spec: GitHub issue #43, including all comments (none at review time)
- Diff: `git diff 5a0958e75c2c9323d2494136b3b26e1d4ded2b67...HEAD`

## Standards

Reviewer found three hard violations:

- Source evidence was self-described rather than recursively replayed, violating
  the Run Manifest and lineage rules in `CONTEXT.md` and
  `docs/reference/workspace-and-run-bundle.md`.
- The matching-capability path labelled a duration as `max(compute, memory)`
  without preserving and recomputing both constraints, contrary to ADR 0037.
- The new Run Manifests omitted a Compilation Fingerprint.

The reviewer also flagged a judgement-call **Divergent Change** smell in the
large bundle-kind block added to `run_bundle.py`, plus a digest-helper duplicate.

## Spec

Reviewer found four must-fix gaps:

- matching-capability evidence did not actually compute the local maximum;
- source Run Bundles were not recursively verified;
- the verifier summed a hard-coded seven-item list instead of replaying the DAG
  critical path, cycle and output identities;
- no authoritative NPU source/Frontier Bundle had been produced.

No scope creep was found. The seven phase identities matched ADR 0037.

## Resolution

- Added immutable `operator-phase-measurement` Run Bundles and required separate
  `search` and `independent-holdout` source identities for each mandatory phase.
- Both exact-operation and matching-capability evidence now retain compute/exact
  and memory-pattern constraints and recompute the phase-local maximum.
- The public verifier recursively verifies each source Run Manifest, cohort,
  phase, lane, candidate and digest. It replays the dependency DAG, rejects
  cycles or output mismatch, and recomputes topological order, serialized
  reference, critical path and RSS uncertainty.
- Added Compilation Fingerprints and persisted lock owner, start, finish,
  Hardware Cohort and visibility metadata.
- Ran one approved, whole-host-locked bounded NPU session. Remote focused tests
  passed, but `mean_scale/search` exceeded the pre-registered 10% IQR/median
  limit. Per protocol, collection stopped without a retry. Two search Runs were
  retained (`square`, `reduce_sum`); the verified Frontier is structured
  `unknown` and names all seven missing search/holdout qualification boundaries.
- The Divergent Change extraction remains a non-blocking follow-up: the public
  verifier is still the required acceptance seam and existing bundle kinds stay
  regression-covered.

Original-reviewer re-review: pending.
