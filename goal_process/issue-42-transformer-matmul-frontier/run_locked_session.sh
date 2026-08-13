#!/usr/bin/env bash
set -euo pipefail

repository=/home/t00906153/GroundUpScale-issue-42
session_id="${GROUNDUPSCALE_SESSION_ID:?set a unique issue42 session id}"
artifact_store="$repository/goal_process/issue-42-transformer-matmul-frontier/evidence/$session_id/artifact-store"
evidence_root="$repository/goal_process/issue-42-transformer-matmul-frontier/evidence/$session_id"
tmp_root="$repository/goal_process/issue-42-transformer-matmul-frontier/tmp/$session_id"

test "${ASCEND_RT_VISIBLE_DEVICES:-}" = 0
test "${GROUNDUPSCALE_ISSUE:-}" = 42
test -f /home/t00906153/.groundupscale/locks/ascend-910b2-host.owner

cd "$repository"
export PYTHONPATH="$repository/src"
exec /home/miniconda3/envs/lmz_pt27py311/bin/python3.11 \
  goal_process/issue-42-transformer-matmul-frontier/run_locked_session.py \
  --repository "$repository" \
  --transformer-run \
    goal_process/issue-30-ascend-transformer-demo/evidence/runs/ascend-910b2-transformer-demo-20260811-v1 \
  --artifact-store "$artifact_store" \
  --evidence-root "$evidence_root" \
  --tmp-root "$tmp_root" \
  --session-id "$session_id"
