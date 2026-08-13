#!/usr/bin/env bash
set -euo pipefail

cd /home/t00906153/GroundUpScale-issue-43
export PYTHONPATH=src
export ASCEND_RT_VISIBLE_DEVICES=0
issue43_python=/home/t00906153/super/work/fast-debug-verl/.conda-envs/verl-qwen3vl-npu/bin/python3.11
session_log=goal_process/issue-43-ascend-rmsnorm-frontier/issue43-session.log
exec > >(tee -a "$session_log") 2>&1

"$issue43_python" -c 'import torch, torch_npu, pytest, pydantic, yaml; print(torch.__version__, torch_npu.__version__, pytest.__version__, pydantic.__version__)'
"$issue43_python" -m pytest -q tests/test_ascend_rmsnorm_operator_frontier.py -x
"$issue43_python" goal_process/issue-43-ascend-rmsnorm-frontier/collect_issue43.py \
  --artifact-store /home/t00906153/GroundUpScale-issue-43/goal_process/issue-43-ascend-rmsnorm-frontier/evidence \
  --run-tag 20260813Tissue43npu01
