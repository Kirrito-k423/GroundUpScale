# C018 脱敏命令记录

```sh
uv run groundupscale run specs/plans/mac-mps-prefill.yaml --repository-root . --run-id <fit-or-holdout-id> --target-window-ms 100 --windows-per-sample 9 --json
uv run groundupscale fit-calibration --run-bundle <fit-01> --run-bundle <fit-02> --run-bundle <fit-03> --output .groundupscale/calibration/mps-candidate.yaml --json
uv run groundupscale validate-calibration .groundupscale/calibration/mps-candidate.yaml --run-bundle <holdout-01..05> --output .groundupscale/calibration/mps-validation.json --json
uv run groundupscale promote-calibration .groundupscale/calibration/mps-candidate.yaml .groundupscale/calibration/mps-validation.json --output evidence/calibrations/apple-m4-mps-fixed-prefill.yaml --json
```
