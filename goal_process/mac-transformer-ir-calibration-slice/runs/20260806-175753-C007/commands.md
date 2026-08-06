# C007 脱敏命令记录

```sh
uv run pytest tests/test_structural_ir.py -q  # 预期 RED：Builder/IR 尚不存在
uv run pytest tests/test_structural_ir.py -q  # 实现后 GREEN
uv run pytest -q
```
