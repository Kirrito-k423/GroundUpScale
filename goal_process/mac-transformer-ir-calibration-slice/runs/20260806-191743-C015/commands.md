# C015 脱敏命令记录

```sh
uv run pytest -q
uv run groundupscale run specs/plans/mac-cpu-prefill.yaml --repository-root . --run-id 20260806-m4-cpu-c015 --target-window-ms 100 --json
uv run groundupscale run specs/plans/mac-mps-prefill.yaml --repository-root . --run-id 20260806-m4-mps-c015 --target-window-ms 100 --json
```
