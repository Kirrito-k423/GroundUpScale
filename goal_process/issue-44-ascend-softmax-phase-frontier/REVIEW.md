# Issue 44 Code Review

Fixed point: `5a0958e75c2c9323d2494136b3b26e1d4ded2b67`

Spec source: GitHub issue #44 and its comments (no comments).

## Standards

Initial result: FAIL — three must-fix findings and one judgement-call smell.

- The phase graph omitted ADR-0037 Resource Demands, input/output roles,
  assumptions, and provenance; exact probes did not use the real Softmax
  intermediate operand domains.
- The public verifier trusted graph-local duration, uncertainty, lineage, and
  Stable Path fields instead of rebuilding them from recursive source bundles
  and the frozen #30 Semantic IR.
- Ascend host-lock metadata was outside the authoritative Frontier lineage.
- Duplicated phase maps were noted as a should-fix judgement call.

Resolution: future phase documents carry the ADR-required fields and future
probes generate each operand from the preceding mathematical Softmax phase; the verifier
replays phase/lane/candidate/domain/process/digest/duration/uncertainty and #30
Semantic IR identities; the superseding Frontier binds validated lock metadata.
Shared verifier phase specs reduce the most dangerous duplicated validation
table. Full consolidation of CLI/adapter constants remains a non-blocking
refactor because those layers intentionally expose different contracts.

## Spec

Initial result: FAIL — three must-fix and two should-fix findings.

- Source digest, uncertainty, composition, #30 paths, and holdout independence
  were not independently replayed.
- `inner_iterations` normalization did not retain aggregate event samples.
- Lock owner/time/visibility/cohort metadata was not bound to the Frontier.
- Rejected 1850Z/1900Z authority decisions and operand-domain contracts needed
  clearer audit boundaries.

Resolution: the public verifier now independently rebuilds the complete source
seam. Future measurement bundles preserve aggregate samples plus divisor,
rounding, and unit for normalization replay. The previously collected 1910Z
source bundles remain immutable and therefore retain two explicit legacy
boundaries: aggregate event samples were not recorded, and exp/sum/normalize
used synthetic operands rather than real chain intermediates. No evidence was
fabricated or retroactively changed.
`issue44-20260813T1945Z-softmax-frontier-unknown-v2` is a new immutable
superseding structured-unknown Frontier produced offline from the same ten
verified 1910Z sources. Numeric Frontier and uncertainty are unavailable; its
minimal boundary is real-chain operand evidence for exp, sum_reduce, and
normalize. It binds the real lock session metadata. `authority-selection.json`
records the minimal rejected-session reason codes for 1850Z and 1900Z. The old
1910Z qualified Frontier is preserved as immutable, non-authoritative evidence.

## Verification

- Targeted affected suite after final remediation: `45 passed in 20.22s`.
- Earlier broader affected suite: `85 passed in 31.68s`.
- Issue-specific suite includes structured-unknown replay coverage.
- `python -m compileall`: passed.
- `git diff --check`: passed.
- Superseding Frontier public verification: passed with zero failures.
- No NPU command was run during review remediation.

Final original-reviewer re-review: Standards PASS and Spec PASS; no remaining
must-fix findings. The final missing-phase replay attack test and verifier fix
are committed in `d4e88c9c26510227e2e9ca9b2f521846c0eff1f4`.
