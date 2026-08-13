# Issue 44 Ascend Softmax Phase Frontier

This slice qualifies the fixed Transformer demo Softmax domain as the explicit
ADR 0037 chain `max_reduce -> subtract -> exp -> sum_reduce -> normalize`.
Every phase uses an exact operation-class probe. The selected composition is a
serialized critical-path sum; no fusion, chunk pipeline, or cross-phase overlap
is inferred.

Real evidence must be collected on the isolated remote checkout with the host
lock covering the complete search/holdout session:

```bash
GROUNDUPSCALE_ISSUE=44 ASCEND_RT_VISIBLE_DEVICES=0 ISSUE44_SESSION_ID=<unique> \
  /home/t00906153/.groundupscale/bin/with-ascend-lock \
  /home/t00906153/GroundUpScale-issue-44/goal_process/issue-44-ascend-softmax-phase-frontier/collect_phase_evidence.sh
```

The session directory records the lock owner at both boundaries, start/end
timestamps, device visibility, and the Hardware Cohort. Run IDs contain issue
44 and the caller-provided unique session ID. A lock timeout (exit 75) is a
temporary resource condition and must be retried without replacing required
holdout evidence with an estimate.
