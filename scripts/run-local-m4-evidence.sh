#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This trusted evidence lane requires an Apple Silicon Mac." >&2
  exit 2
fi

run_tag="${1:-manual-$(date -u +%Y%m%dT%H%M%SZ)}"
artifact_store="${GROUNDUPSCALE_ARTIFACT_STORE:-.groundupscale}"

uv sync --locked --group dev
uv run python -c 'import torch; assert torch.backends.mps.is_available(), "MPS unavailable"'
uv run pytest -q

uv run groundupscale run specs/plans/mac-cpu-prefill.yaml \
  --repository-root . \
  --artifact-store "$artifact_store" \
  --run-id "${run_tag}-cpu" \
  --target-window-ms 100 \
  --windows-per-sample 9 \
  --json

PYTORCH_ENABLE_MPS_FALLBACK=0 uv run groundupscale run \
  specs/plans/mac-mps-prefill.yaml \
  --repository-root . \
  --artifact-store "$artifact_store" \
  --run-id "${run_tag}-mps" \
  --target-window-ms 100 \
  --windows-per-sample 9 \
  --json

uv run groundupscale verify-run "$artifact_store/runs/${run_tag}-cpu" --json
uv run groundupscale verify-run "$artifact_store/runs/${run_tag}-mps" --json
