# C012 脱敏命令记录

```sh
uv run pytest -q
uv run groundupscale run specs/plans/mac-cpu-prefill.yaml --repository-root . --run-id 20260806-m4-cpu-c012 --json
uv run groundupscale run specs/plans/mac-mps-prefill.yaml --repository-root . --run-id 20260806-m4-mps-c012 --json
uv run groundupscale verify-run .groundupscale/runs/<run-id> --json
uv run groundupscale explain .groundupscale/runs/<run-id> --json
```
