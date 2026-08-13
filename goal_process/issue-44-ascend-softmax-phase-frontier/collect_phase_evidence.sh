#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/t00906153/GroundUpScale-issue-44
artifact_store="$repo_root/goal_process/issue-44-ascend-softmax-phase-frontier/evidence"
session_id="${ISSUE44_SESSION_ID:?set ISSUE44_SESSION_ID to a unique value}"
metadata_root="$artifact_store/sessions/issue44-$session_id"
python_bin="${ISSUE44_PYTHON:-$repo_root/.venv/bin/python}"
mkdir -p "$metadata_root"

owner_file=/home/t00906153/.groundupscale/locks/ascend-910b2-host.owner
started_at="$(date -Iseconds)"
cp "$owner_file" "$metadata_root/lock-owner-start.txt"
printf '%s\n' "$started_at" > "$metadata_root/started-at.txt"
printf '%s\n' "${ASCEND_RT_VISIBLE_DEVICES:?ASCEND_RT_VISIBLE_DEVICES is required}" \
  > "$metadata_root/device-visibility.txt"

phases=(max_reduce subtract exp sum_reduce normalize)
candidates=(torch.amax torch.sub torch.exp torch.sum torch.div)
for lane in search holdout; do
  for index in "${!phases[@]}"; do
    phase="${phases[$index]}"
    candidate="${candidates[$index]}"
    run_id="issue44-${session_id}-${phase}-${lane}"
    PYTHONPATH="$repo_root/src" "$python_bin" -m groundupscale.cli measure \
      --device ascend-npu \
      --operation SoftmaxPhase \
      --phase "$phase" \
      --shape 1,8,512,512 \
      --axis -1 \
      --dtype float32 \
      --layout contiguous \
      --candidate "$candidate" \
      --seed 20260813 \
      --warmup 20 \
      --repetitions 100 \
      --inner-iterations 100 \
      --artifact-store "$artifact_store" \
      --run-id "$run_id" \
      --json > "$metadata_root/$run_id-summary.json"
  done
done

cp "$owner_file" "$metadata_root/lock-owner-end.txt"
date -Iseconds > "$metadata_root/ended-at.txt"
cp "$artifact_store/runs/issue44-${session_id}-max_reduce-search/adapter/cohort.json" \
  "$metadata_root/hardware-cohort.json"

"$python_bin" - "$metadata_root" "$artifact_store" "$session_id" <<'PY'
import json
import sys
from pathlib import Path

metadata_root, artifact_store, session_id = map(Path, sys.argv[1:])
owner_start = (metadata_root / "lock-owner-start.txt").read_text().strip()
owner_end = (metadata_root / "lock-owner-end.txt").read_text().strip()
cohort = json.loads((metadata_root / "hardware-cohort.json").read_text())
document = {
    "schema": "groundupscale.dev/ascend-host-lock-session/v1alpha1",
    "issue": 44,
    "lock_path": "/home/t00906153/.groundupscale/locks/ascend-910b2-host.lock",
    "owner_start": owner_start,
    "owner_end": owner_end,
    "started_at": (metadata_root / "started-at.txt").read_text().strip(),
    "ended_at": (metadata_root / "ended-at.txt").read_text().strip(),
    "device_visibility": (metadata_root / "device-visibility.txt").read_text().strip(),
    "hardware_cohort": cohort["cohort_id"],
    "wrapper_sha256": "22d43618f1c616b2ff70570944c7447cd851aac98bfedb111b7912fc36b94787",
}
(metadata_root / "ascend-host-lock-session.json").write_text(
    json.dumps(document, indent=2, sort_keys=True) + "\n"
)
PY
