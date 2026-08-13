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
