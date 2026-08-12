#!/usr/bin/env bash
set -euo pipefail

device="cpu"
run_tag="manual-$(date -u +%Y%m%dT%H%M%SZ)"
artifact_store="${GROUNDUPSCALE_ARTIFACT_STORE:-.groundupscale}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      device="$2"
      shift 2
      ;;
    --tag)
      run_tag="$2"
      shift 2
      ;;
    --artifact-store)
      artifact_store="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 64
      ;;
  esac
done

if [[ ! "$run_tag" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "unsafe --tag: $run_tag" >&2
  exit 64
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
report_root="$artifact_store/trusted-hardware-ci"
report_dir="$report_root/$run_tag"
report_path="$report_dir/trusted-hardware-ci-report.json"
run_bundle="$artifact_store/runs/${run_tag}-cpu"
if [[ -e "$report_dir" || -e "$run_bundle" ]]; then
  echo "Trusted hardware run tag already exists: $run_tag" >&2
  exit 64
fi
mkdir -p "$report_dir"

write_ci_report() {
  python3 "$script_dir/write-trusted-hardware-ci-report.py" \
    --output "$report_path" \
    --run-tag "$run_tag" \
    --device "$device" \
    --previous-qualified-root "$report_root" \
    "$@"
}

if [[ "$device" != "cpu" ]]; then
  write_ci_report \
    --status hardware_unavailable \
    --reason-code unsupported-device
  echo "This local slice currently supports only --device cpu." >&2
  exit 2
fi
system_name="$(uname -s)"
machine_architecture="$(uname -m)"
if [[ "$system_name" != "Darwin" || "$machine_architecture" != "arm64" ]]; then
  availability_evidence="$report_dir/availability-failure.log"
  printf 'system=%s\narchitecture=%s\n' \
    "$system_name" "$machine_architecture" >"$availability_evidence"
  write_ci_report \
    --status hardware_unavailable \
    --reason-code unsupported-platform \
    --failure-evidence "$availability_evidence"
  echo "This trusted evidence lane requires an Apple Silicon Mac." >&2
  exit 2
fi

environment_evidence="$report_dir/environment-preflight.log"
if ! uv run groundupscale preflight --json \
  >"$environment_evidence" 2>&1; then
  write_ci_report \
    --status quarantined \
    --reason-code environment-preflight-failed \
    --failure-evidence "$environment_evidence"
  echo "Trusted hardware environment quarantined: $report_path" >&2
  exit 1
fi
cat "$environment_evidence"

collection_evidence="$report_dir/collection.log"
set +e
uv run groundupscale run specs/plans/mac-cpu-prefill.yaml \
  --repository-root . \
  --artifact-store "$artifact_store" \
  --run-id "${run_tag}-cpu" \
  --target-window-ms 100 \
  --windows-per-sample 9 \
  --require-valid-environment \
  --json >"$collection_evidence" 2>&1
collection_status=$?
set -e
if [[ $collection_status -ne 0 ]]; then
  collection_reason="collection-failed"
  if [[ $collection_status -eq 2 ]] && grep -q '"reason_codes"' "$collection_evidence"; then
    collection_reason="environment-drift-during-collection"
  fi
  write_ci_report \
    --status quarantined \
    --reason-code "$collection_reason" \
    --failure-evidence "$collection_evidence" \
    --reason-codes-from-json "$collection_evidence"
  echo "Trusted hardware evidence quarantined: $report_path" >&2
  exit 1
fi
cat "$collection_evidence"

verification_evidence="$report_dir/verification.log"
if ! uv run groundupscale verify-run "$run_bundle" --json \
  >"$verification_evidence" 2>&1; then
  write_ci_report \
    --status quarantined \
    --reason-code verification-failed \
    --run-bundle "$run_bundle" \
    --failure-evidence "$verification_evidence"
  echo "Trusted hardware evidence quarantined: $report_path" >&2
  exit 1
fi
if ! grep -Eq '"passed"[[:space:]]*:[[:space:]]*true' "$verification_evidence"; then
  write_ci_report \
    --status quarantined \
    --run-bundle "$run_bundle" \
    --reason-code verification-failed \
    --failure-evidence "$verification_evidence"
  echo "Trusted hardware evidence quarantined: $report_path" >&2
  exit 1
fi
cat "$verification_evidence"

noise_evidence="$report_dir/noise-policy-check.json"
set +e
python3 "$script_dir/check-local-m4-noise.py" \
  "$run_bundle" \
  --output "$noise_evidence"
noise_status=$?
set -e
if [[ $noise_status -ne 0 ]]; then
  noise_reason="measurement-noise-above-policy"
  if [[ $noise_status -ne 1 ]]; then
    noise_reason="measurement-evidence-invalid"
  fi
  write_ci_report \
    --status quarantined \
    --reason-code "$noise_reason" \
    --run-bundle "$run_bundle" \
    --failure-evidence "$noise_evidence"
  echo "Trusted hardware evidence quarantined: $report_path" >&2
  exit 1
fi

write_ci_report \
  --status evidence_collected \
  --run-bundle "$run_bundle"

echo "Trusted hardware CI report: $report_path"
