# Issue #48: Schedule Achievable Frontier

The authority bundle composes the two-layer Transformer's 52 indexed semantic
leaves from the immutable evidence boundary delivered by issues #30 and #42–#46.
It deliberately publishes a structured unknown rather than converting partial,
cross-cohort, or unpublished evidence into numeric duration.

## Authority

`issue48-20260814T0002Z-schedule-frontier-unknown-v2` recursively verifies the
repository-contained authority bundles from #30, #42 v4, #43 v3, and #44
unknown v2 before recording their manifest SHA-256 identities. Issues #45 and
#46 expose useful contracts and inventories but no repository-contained
immutable Run Bundle that covers every indexed leaf, so they remain exact
evidence boundaries.

The result preserves:

- all 52 unique Stable Paths (26 per repeated layer);
- the separately known #30 E2E Observation of `1.921530 ms`;
- explicit serialized-unfused, ideal-DAG, and selected-feasible references;
- six rejected implicit optimizations and six mandatory schedule effects;
- exact per-path and per-effect missing-evidence boundaries.

It supersedes immutable v1 by its exact run id and manifest SHA-256; v1 remains
unchanged and replayable as the historical pre-review contract.

No NPU action was run for #48. Existing locked measurements are consumed only
through immutable source identities. Relative prediction error is unknown
because the selected feasible schedule is unknown.

Replay:

```sh
uv run groundupscale verify-run \
  goal_process/issue-48-schedule-achievable-frontier/evidence/runs/issue48-20260814T0002Z-schedule-frontier-unknown-v2 \
  --json
uv run groundupscale explain \
  goal_process/issue-48-schedule-achievable-frontier/evidence/runs/issue48-20260814T0002Z-schedule-frontier-unknown-v2 \
  --json
```
