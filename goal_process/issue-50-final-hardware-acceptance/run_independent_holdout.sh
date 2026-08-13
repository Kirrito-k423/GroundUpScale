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

PYTHONPATH="$repo/src" "$python" - "$run_dir" "$started_at" "$ended_at" "$owner_start" "$wrapper_sha256" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
started_at, ended_at, owner, wrapper_sha256 = sys.argv[2:]
manifest_path = run / "run.manifest.json"
manifest = json.loads(manifest_path.read_text())
metadata = {
    "schema": "groundupscale.dev/ascend-host-lock-session/v1alpha1",
    "issue": 50,
    "run_id": manifest["run_id"],
    "lock_path": "/home/t00906153/.groundupscale/locks/ascend-910b2-host.lock",
    "wrapper_path": "/home/t00906153/.groundupscale/bin/with-ascend-lock",
    "wrapper_sha256": wrapper_sha256,
    "owner": owner,
    "measurement_started_at": started_at,
    "measurement_ended_at": ended_at,
    "hardware_cohort": manifest["hardware_cohort"],
    "ascend_rt_visible_devices": "0",
    "logical_device": "npu:0",
    "whole_host_exclusive": True,
}
payload = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
relative = "observation/ascend-host-lock-session.json"
target = run / relative
target.write_bytes(payload)
manifest["artifacts"].append({
    "role": "ascend-host-lock-session",
    "path": relative,
    "media_type": "application/json",
    "schema": metadata["schema"],
    "sha256": sha256(payload).hexdigest(),
    "produced_by": "groundupscale@0.1.0",
    "inputs": [],
})
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

PYTHONPATH="$repo/src" "$python" -m groundupscale.cli verify-run "$run_dir" --json \
  > "$session_dir/verify-run.json"
test "$(PYTHONPATH="$repo/src" "$python" -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["passed"]).lower())' "$session_dir/verify-run.json")" = true
