#!/usr/bin/env bash
set -euo pipefail

lane=${1:?usage: collect_bounded_m_sweep.sh <main|main-replication|holdout|validation|supplemental>}
workspace=${GROUNDUPSCALE_ISSUE36_WORKSPACE:-/home/t00906153/GroundUpScale-issue36-20260812}
python_bin=${GROUNDUPSCALE_NPU_PYTHON:-/home/miniconda3/envs/lmz_pt27py311/bin/python}
artifact_store=${workspace}/goal_process/issue-36-ascend-matmul-m-sweep/evidence
warmup=100
repetitions=100
inner_iterations=100

export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=${workspace}/src

measure() {
  local evidence_lane=$1
  local m=$2
  local session=$3
  local run_id=issue36-${evidence_lane}-m${m}-torch-matmul-${session}
  local run_bundle=${artifact_store}/runs/${run_id}
  if [[ -d "${run_bundle}" ]]; then
    "${python_bin}" -m groundupscale.cli verify-run "${run_bundle}" --json
    return
  fi
  "${python_bin}" -m groundupscale.cli measure \
    --device ascend-npu \
    --m "${m}" --n 512 --k 512 \
    --dtype float32 \
    --layout row-major-contiguous \
    --candidate torch.matmul \
    --seed 20260812 \
    --warmup "${warmup}" \
    --repetitions "${repetitions}" \
    --inner-iterations "${inner_iterations}" \
    --artifact-store "${artifact_store}" \
    --run-id "${run_id}" \
    --json
}

case "${lane}" in
  main|main-replication)
    sessions=(01 02 03)
    if [[ "${lane}" == "main-replication" ]]; then
      sessions=(02 03)
    fi
    for m in 1 2 4 8 16 32 64 96 127 128 129 192 255 256 257 384 511 512 513 768 1024 1536 2048 4096; do
      for session in "${sessions[@]}"; do
        measure main "${m}" "${session}"
      done
    done
    ;;
  holdout)
    for m in 1 2 4 8 16 32 64 96 127 128 129 192 255 256 257 384 511 512 513 768 1024 1536 2048 4096; do
      for session in 01 02 03; do
        measure holdout "${m}" "${session}"
      done
    done
    ;;
  validation)
    for m in 48 160 320 448 640 896 1280 3072; do
      for session in 01 02 03; do
        measure validation "${m}" "${session}"
      done
    done
    ;;
  supplemental)
    echo "supplemental points are intentionally empty until the main sweep justifies the one allowed round" >&2
    exit 2
    ;;
  *)
    echo "unknown lane: ${lane}" >&2
    exit 2
    ;;
esac
