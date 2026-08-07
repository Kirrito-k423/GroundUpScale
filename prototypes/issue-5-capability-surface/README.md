# PROTOTYPE ONLY — Capability Surface query and evidence replay

This throwaway prototype answers one question from
[原型：连续 Capability Surface 查询与证据回放](https://github.com/Kirrito-k423/GroundUpScale/issues/5):
can a retained local simplicial surface return exact knots, continuous interior
interpolation, explicit uncertainty and stable provenance while refusing
unvalidated regimes, bad cells and ineligible evidence?

It is deliberately scenario-bound. It is not a production implementation,
must not be imported by `src/groundupscale`, and must not be copied or promoted
into the implementation tickets. Only the decision and counterexamples recorded
on the issue may survive this branch.

## Falsification contract

The hypothesis is rejected if any of these observations fails:

1. Shapes 128 and 512 are exact knots through the same query path that gives
   shape 201 exactly 1.3140625 TFLOP/s with reproducible weights.
2. The two-dimensional point `(256, 320, K=256)` is interpolated at exactly
   1.55 TFLOP/s, while `(400, 400, K=256)` is `unknown` outside the retained
   triangle.
3. A non-aligned query cannot borrow aligned-only anchors.
4. A sparse cell contradicted by a local cliff confirmation is rejected instead
   of returning an authoritative smoothed value.
5. The checked-in M4 observation remains `unknown` because its environment is
   ineligible, even when its selected case is locally eligible.
6. Two evaluations of identical inputs produce the same decision digest and all
   required provenance fields.

## One-command replay

Run from the repository root:

```bash
python3.11 prototypes/issue-5-capability-surface/run.py --batch
```

The command prints every scenario's complete state, writes the raw inputs,
environment, query results, counterexamples and assertions to
`results/raw-results.json`, and exits non-zero if a pre-registered observation
fails. Running without `--batch` opens a tiny in-memory terminal driver.
