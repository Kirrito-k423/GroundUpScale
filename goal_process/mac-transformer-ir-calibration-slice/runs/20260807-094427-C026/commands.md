# C026 脱敏命令记录

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run groundupscale run specs/plans/mac-mps-prefill.yaml --repository-root . --run-id <new-fit-or-holdout-id> --target-window-ms 100 --windows-per-sample 9 --require-valid-environment --json
uv run groundupscale verify-run .groundupscale/runs/<run-id> --json
uv run groundupscale fit-calibration --run-bundle <fit-01> --run-bundle <fit-02> --run-bundle <fit-03> --output .groundupscale/calibration/20260807-controlled-mps-candidate.yaml --json
uv run groundupscale validate-calibration .groundupscale/calibration/20260807-controlled-mps-candidate.yaml --run-bundle <holdouts> --output .groundupscale/calibration/20260807-controlled-mps-validation.json --json
uv run groundupscale promote-calibration .groundupscale/calibration/20260807-controlled-mps-candidate.yaml .groundupscale/calibration/20260807-controlled-mps-validation.json --output evidence/calibrations/apple-m4-mps-fixed-prefill.yaml --json
```
