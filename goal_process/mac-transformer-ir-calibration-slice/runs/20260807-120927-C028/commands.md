# C028 脱敏命令

```sh
uv run pytest -q

uv run groundupscale benchmark-hardware \
  specs/microbenchmarks/apple-m4-cpu.yaml \
  --repository-root . \
  --observation-output goal_process/mac-transformer-ir-calibration-slice/evidence/apple-m4-cpu-microbenchmark-observation-v2.json \
  --profile-output specs/hardware-capabilities/apple-m4-cpu-local.yaml \
  --profile-name apple-m4-cpu-local \
  --preflight-sample-interval-seconds 0.2 \
  --preflight-process-samples 3 --json

uv run groundupscale run specs/plans/mac-cpu-prefill.yaml \
  --repository-root . --artifact-store .groundupscale \
  --run-id m4-cpu-envelope-20260807-v2 \
  --samples 20 --warmup 20 --windows-per-sample 5 \
  --target-window-ms 20 --collect-environment \
  --preflight-sample-interval-seconds 0.2 \
  --preflight-process-samples 3 --json

uv run groundupscale verify-run \
  .groundupscale/runs/m4-cpu-envelope-20260807-v2 --json

uv run groundupscale explain \
  .groundupscale/runs/m4-cpu-envelope-20260807-v2 --json
```
