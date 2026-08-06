# C010 脱敏命令记录

```sh
uv run pytest tests/test_cost_totals.py tests/test_compile_cli.py -q
uv run pytest -q
uv run groundupscale compile specs/plans/mac-cpu-prefill.yaml --repository-root . --output .groundupscale/compilations/m3-final --json
```
