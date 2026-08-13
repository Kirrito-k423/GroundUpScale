# Issue 43 Code Review

- Fixed point: `5a0958e75c2c9323d2494136b3b26e1d4ded2b67`
- Reviewed commits: `9a2330c9bb33a8d78ee8ab68320e71c588458326`,
  `e5641465a512e596f9634f8267b43ccdf0d53400`, plus the pending review-fix commit
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
  and separately sampled memory-pattern constraints, two distinct
  capability-profile references to bundle-local, resource-identified profile
  sections. The verifier recomputes each median summary from its raw samples,
  validates resource semantics, then recomputes the phase-local maximum.
  The collector maps elementwise, row-reduction, row-scalar and broadcast
  resources to distinct probes; notably row-scalar uses exactly one value per
  row, matching the graph's 2,048-byte read/write demand for 512 rows.
- The public verifier recursively verifies each source Run Manifest, cohort,
  phase, lane, candidate, observation and digest. It binds the holdout
  observation byte-for-byte to qualification evidence, both source records to
  the selected candidate, and both evidence references to their source Runs.
  A fully connected, rehashed derived-evidence attack now fails while the
  immutable source observation remains unchanged. The verifier replays the
  dependency DAG, rejects cycles or output mismatch, and recomputes topological
  order, serialized reference, critical path and RSS uncertainty.
- Structured unknown is also replayed rather than trusted: the verifier derives
  every missing phase from graph identity and absent search/holdout lanes, then
  requires its schedule to contain no candidate, constraint, duration,
  uncertainty or evidence reference. A fully rehashed forged missing set fails.
- Added one explicit Cost Compilation Fingerprint across source manifests,
  observations, phase graph, qualification and Frontier manifest, and persisted
  lock owner, start, finish,
  Hardware Cohort and visibility metadata.
- Ran one approved, whole-host-locked bounded NPU session. Remote focused tests
  passed, but `mean_scale/search` exceeded the pre-registered 10% IQR/median
  limit. Per protocol, collection stopped without a retry. Two search Runs were
  retained (`square`, `reduce_sum`) as original session records. Because those
  pre-fix observations did not contain replayable, semantic memory profile
  sections, they are deliberately excluded from the hardened qualification.
  Immutable v1 and v2 are preserved; the v3 verified Frontier explicitly
  supersedes v2 by run ID, relative path and Manifest digest. It has zero
  qualified source Runs, is structured `unknown`,
  and names all seven missing search/holdout qualification boundaries.
- The Divergent Change extraction remains a non-blocking follow-up: the public
  verifier is still the required acceptance seam and existing bundle kinds stay
  regression-covered.

The original reviewers iterated on the source-observation binding,
structured-unknown replay, capability-profile replay, Compilation Fingerprint,
and row-scalar probe Shape gaps above. Final Spec and Standards re-reviews both
PASS with no remaining must-fix. The Divergent Change extraction remains the
sole non-blocking judgement call.
