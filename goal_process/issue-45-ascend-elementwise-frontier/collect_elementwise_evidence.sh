#!/usr/bin/env bash
set -euo pipefail

workspace=${GROUNDUPSCALE_ISSUE45_WORKSPACE:-/home/t00906153/GroundUpScale-issue-45}
python_bin=${GROUNDUPSCALE_NPU_PYTHON:-/home/miniconda3/envs/lmz_pt27py311/bin/python}
artifact_store=${workspace}/goal_process/issue-45-ascend-elementwise-frontier/evidence
session_id=${GROUNDUPSCALE_ISSUE45_SESSION_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
metadata_dir=${artifact_store}/session-metadata/${session_id}
owner_file=${GROUNDUPSCALE_NPU_LOCK_OWNER_FILE:-/home/t00906153/.groundupscale/locks/ascend-910b2-host.owner}

export ASCEND_RT_VISIBLE_DEVICES=0
export GROUNDUPSCALE_ISSUE=45
export GROUNDUPSCALE_HARDWARE_COHORT=ascend-npu-23b93a89d5fecc79
export GROUNDUPSCALE_NPU_LOCK_FD=${GROUNDUPSCALE_NPU_LOCK_FD:-9}
export GROUNDUPSCALE_NPU_LOCK_PATH=${GROUNDUPSCALE_NPU_LOCK_PATH:-/home/t00906153/.groundupscale/locks/ascend-910b2-host.lock}
export PYTHONPATH=${workspace}/src

mkdir -p "${metadata_dir}"
date -Iseconds > "${metadata_dir}/started-at.txt"
printf '%s\n' "${GROUNDUPSCALE_HARDWARE_COHORT}" > "${metadata_dir}/hardware-cohort.txt"
printf '%s\n' "${ASCEND_RT_VISIBLE_DEVICES}" > "${metadata_dir}/device-visibility.txt"
cp "${owner_file}" "${metadata_dir}/lock-owner-start.txt"
trap 'date -Iseconds > "${metadata_dir}/finished-at.txt"; if [[ -f "${owner_file}" ]]; then cp "${owner_file}" "${metadata_dir}/lock-owner-finish.txt"; fi' EXIT

measure() {
  local lane=$1
  local domain_id=$2
  local operation=$3
  local shape=$4
  local operand_kind=$5
  local candidate=$6
  local session=$7
  local run_id=issue45-${domain_id}-${lane}-${session_id}-${session}
  "${python_bin}" -m groundupscale.cli measure \
    --device ascend-npu \
    --operation "${operation}" \
    --elementwise-shape "${shape}" \
    --operand-kind "${operand_kind}" \
    --dtype float32 --layout contiguous \
    --candidate "${candidate}" \
    --seed 20260813 \
    --warmup 20 --repetitions 100 --inner-iterations 100 \
    --artifact-store "${artifact_store}" \
    --run-id "${run_id}" --json
}

for lane in search holdout; do
  for session in 01 02 03; do
    measure "${lane}" add-broadcast-mask Add 1,8,512,512 tensor-broadcast torch.add "${session}"
    measure "${lane}" add-residual Add 1,512,512 tensor-tensor torch.add "${session}"
    measure "${lane}" mul-attention-scale Mul 1,8,512,512 tensor-scalar torch.mul "${session}"
    measure "${lane}" mul-mlp-gate Mul 1,512,2048 tensor-tensor torch.mul "${session}"
    measure "${lane}" silu-mlp-gate SiLU 1,512,2048 tensor torch.nn.functional.silu "${session}"
  done
done
