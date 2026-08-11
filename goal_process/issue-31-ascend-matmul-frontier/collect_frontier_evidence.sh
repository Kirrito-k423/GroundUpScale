#!/usr/bin/env bash
set -euo pipefail

phase=${1:?usage: collect_frontier_evidence.sh <search-v4|holdout-v4|confirmation-v4> [candidate]}
candidate=${2:-}
workspace=${GROUNDUPSCALE_ISSUE31_WORKSPACE:-/home/t00906153/GroundUpScale-issue31-20260811}
python_bin=${GROUNDUPSCALE_NPU_PYTHON:-/home/miniconda3/envs/lmz_pt27py311/bin/python}
artifact_store=${workspace}/evidence
warmup=${ISSUE31_WARMUP:-100}
repetitions=${ISSUE31_REPETITIONS:-100}
inner_iterations=${ISSUE31_INNER_ITERATIONS:-100}

export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=${workspace}/src

measure() {
  local lane=$1
  local size=$2
  local implementation=$3
  local session=$4
  local slug=${implementation//./-}
  local run_id=issue31-${lane}-s${size}-${slug}-${session}
  "${python_bin}" -m groundupscale.cli measure \
    --device ascend-npu \
    --m "${size}" --n "${size}" --k "${size}" \
    --dtype float32 \
    --layout row-major-contiguous \
    --candidate "${implementation}" \
    --seed 20260811 \
    --warmup "${warmup}" \
    --repetitions "${repetitions}" \
    --inner-iterations "${inner_iterations}" \
    --artifact-store "${artifact_store}" \
    --run-id "${run_id}" \
    --json
}

case "${phase}" in
  search|search-v2|search-v3|search-v4)
    lane=${phase}
    for size in 256 512; do
      for implementation in torch.matmul torch.matmul.k-split-2; do
        for session in 01 02 03; do
          measure "${lane}" "${size}" "${implementation}" "${session}"
        done
      done
    done
    ;;
  retry-search-v2)
    for size in 256 512; do
      for implementation in torch.matmul torch.matmul.k-split-2; do
        for session in 04 05 06; do
          measure search-v2 "${size}" "${implementation}" "${session}"
        done
      done
    done
    ;;
  retry-direct-512)
    for session in 07 08 09; do
      measure search-v2 512 torch.matmul "${session}"
    done
    ;;
  holdout)
    if [[ -z "${candidate}" ]]; then
      echo "holdout requires the exact best-of-correct candidate" >&2
      exit 2
    fi
    for size in 256 512; do
      for session in 01 02 03; do
        measure holdout "${size}" "${candidate}" "${session}"
      done
    done
    ;;
  holdout-v3)
    if [[ -z "${candidate}" ]]; then
      echo "holdout-v3 requires the exact best-of-correct candidate" >&2
      exit 2
    fi
    for size in 256 512; do
      for session in 01 02 03; do
        measure holdout-v3 "${size}" "${candidate}" "${session}"
      done
    done
    ;;
  holdout-v4)
    if [[ -z "${candidate}" ]]; then
      echo "holdout-v4 requires the exact best-of-correct candidate" >&2
      exit 2
    fi
    for size in 256 512; do
      for session in 01 02 03; do
        measure holdout-v4 "${size}" "${candidate}" "${session}"
      done
    done
    ;;
  confirmation)
    if [[ -z "${candidate}" ]]; then
      echo "confirmation requires the selected candidate" >&2
      exit 2
    fi
    for session in 01 02 03; do
      measure confirmation 384 "${candidate}" "${session}"
    done
    ;;
  confirmation-v3)
    if [[ -z "${candidate}" ]]; then
      echo "confirmation-v3 requires the selected candidate" >&2
      exit 2
    fi
    for session in 01 02 03; do
      measure confirmation-v3 384 "${candidate}" "${session}"
    done
    ;;
  confirmation-v4)
    if [[ -z "${candidate}" ]]; then
      echo "confirmation-v4 requires the selected candidate" >&2
      exit 2
    fi
    for session in 01 02 03; do
      measure confirmation-v4 384 "${candidate}" "${session}"
    done
    ;;
  *)
    echo "unknown phase: ${phase}" >&2
    exit 2
    ;;
esac
