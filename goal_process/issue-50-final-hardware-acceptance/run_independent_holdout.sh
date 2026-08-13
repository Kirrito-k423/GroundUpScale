#!/usr/bin/env bash
set -euo pipefail

repo=/home/t00906153/GroundUpScale-issue-50
run_id=issue50-20260813T175228Z-independent-e2e-holdout-v1
artifact_store="$repo/goal_process/issue-50-final-hardware-acceptance/evidence/holdout"
session_dir="$repo/goal_process/issue-50-final-hardware-acceptance/evidence/sessions/$run_id"
run_dir="$artifact_store/runs/$run_id"
wrapper=/home/t00906153/.groundupscale/bin/with-ascend-lock
python=/home/miniconda3/envs/lmz_pt27py311/bin/python

test "${GROUNDUPSCALE_ISSUE:-}" = 50
test "${ASCEND_RT_VISIBLE_DEVICES:-}" = 0
test -x "$wrapper"
test ! -e "$run_dir"
mkdir -p "$session_dir"

started_at=$(date -Iseconds)
owner_start=$(cat /home/t00906153/.groundupscale/locks/ascend-910b2-host.owner)
wrapper_sha256=$(sha256sum "$wrapper" | awk '{print $1}')
export GROUNDUPSCALE_LOCK_STARTED_AT="$started_at"
export GROUNDUPSCALE_LOCK_OWNER="$owner_start"
export GROUNDUPSCALE_LOCK_WRAPPER_SHA256="$wrapper_sha256"
printf '%s\n' "$started_at" > "$session_dir/started-at.txt"
printf '%s\n' "$owner_start" > "$session_dir/lock-owner-start.txt"
printf '%s\n' "$ASCEND_RT_VISIBLE_DEVICES" > "$session_dir/device-visibility.txt"

cd "$repo"
PYTHONPATH="$repo/src" "$python" -m groundupscale.cli run \
  specs/plans/ascend-npu-transformer-demo.yaml \
  --repository-root . \
  --artifact-store "$artifact_store" \
  --run-id "$run_id" \
  --samples 20 \
  --warmup 20 \
  --windows-per-sample 5 \
  --target-window-ms 100 \
  --json > "$session_dir/run-summary.json"

ended_at=$(date -Iseconds)
printf '%s\n' "$ended_at" > "$session_dir/ended-at.txt"
printf '%s\n' "$owner_start" > "$session_dir/lock-owner-end.txt"

PYTHONPATH="$repo/src" "$python" -m groundupscale.cli verify-run "$run_dir" --json \
  > "$session_dir/verify-run.json"
test "$(PYTHONPATH="$repo/src" "$python" -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["passed"]).lower())' "$session_dir/verify-run.json")" = true
