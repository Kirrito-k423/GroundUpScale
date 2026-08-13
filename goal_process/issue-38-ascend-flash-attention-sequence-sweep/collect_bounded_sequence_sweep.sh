#!/usr/bin/env bash
set -euo pipefail

lane=${1:?usage: collect_bounded_sequence_sweep.sh <main|holdout|validation|supplemental>}
workspace=${GROUNDUPSCALE_ISSUE38_WORKSPACE:-/home/t00906153/GroundUpScale-issue38-20260813}
python_bin=${GROUNDUPSCALE_NPU_PYTHON:-/home/miniconda3/envs/lmz_pt27py311/bin/python}
artifact_store=${workspace}/goal_process/issue-38-ascend-flash-attention-sequence-sweep/evidence
supplemental_trigger=${artifact_store}/supplemental-trigger-reviewed.yaml

export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=${workspace}/src

measure() {
  local evidence_lane=$1
  local sequence_length=$2
  local session=$3
  local run_id=issue38-${evidence_lane}-s${sequence_length}-npu-fusion-attention-${session}
  local run_bundle=${artifact_store}/runs/${run_id}
  if [[ -d "${run_bundle}" ]]; then
    "${python_bin}" -m groundupscale.cli verify-run "${run_bundle}" --json
    return
  fi
  "${python_bin}" -m groundupscale.cli measure \
    --device ascend-npu \
    --operation FlashAttentionForward \
    --sequence-count 1 \
    --sequence-length "${sequence_length}" \
    --head-count 8 --head-dimension 64 \
    --dtype float16 --layout TND \
    --candidate torch_npu.npu_fusion_attention \
    --seed 20260813 \
    --warmup 20 \
    --repetitions 100 \
    --inner-iterations 1 \
    --artifact-store "${artifact_store}" \
    --run-id "${run_id}" \
    --json
}

collect_three_sessions() {
  local evidence_lane=$1
  shift
  for sequence_length in "$@"; do
    for session in 01 02 03; do
      measure "${evidence_lane}" "${sequence_length}" "${session}"
    done
  done
}

case "${lane}" in
  main)
    collect_three_sessions main 1 2 4 8 16 32 64 96 127 128 129 192 255 256 257 384 511 512 513 768 1023 1024 1025 1536 2047 2048 2049 3072 4095 4096 4097 6144 8192
    ;;
  holdout)
    collect_three_sessions holdout 1 2 4 8 16 32 64 96 127 128 129 192 255 256 257 384 511 512 513 768 1023 1024 1025 1536 2047 2048 2049 3072 4095 4096 4097 6144 8192
    ;;
  validation)
    collect_three_sessions validation 48 160 320 448 640 896 1280 1792 2560 3584 5120 7168
    ;;
  supplemental)
    if [[ ! -f "${supplemental_trigger}" ]]; then
      echo "supplemental round requires a reviewed trigger from the main sweep" >&2
      exit 2
    fi
    mapfile -t supplemental_lengths < <(
      awk '/^[[:space:]]*-[[:space:]]*[0-9]+[[:space:]]*$/ {print $2}' "${supplemental_trigger}"
    )
    if [[ ${#supplemental_lengths[@]} -eq 0 ]]; then
      echo "reviewed supplemental trigger contains no sequence lengths" >&2
      exit 2
    fi
    collect_three_sessions supplemental "${supplemental_lengths[@]}"
    ;;
  *)
    echo "unknown lane: ${lane}" >&2
    exit 2
    ;;
esac
