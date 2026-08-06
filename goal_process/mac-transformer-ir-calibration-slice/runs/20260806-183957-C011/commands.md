# C011 脱敏命令记录

```sh
uv run pytest tests/test_semantic_compiler.py tests/test_cost_totals.py tests/test_compile_cli.py -q  # 预期 layout RED
uv run pytest -q  # layout 修正后 GREEN
uv run pytest tests/test_reference_runner.py -q  # 预期 runner RED；实现后 CPU/MPS GREEN
uv run python <inline-layout-diagnostic>  # CPU/MPS einsum stride 与 MatMul out= 目标布局穿刺
```
