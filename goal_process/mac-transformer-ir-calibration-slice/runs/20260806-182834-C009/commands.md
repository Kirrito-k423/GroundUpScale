# C009 脱敏命令记录

```sh
uv run pytest tests/test_cost_lowerer.py -q  # 预期 RED：CostLowerer 尚不存在
uv run pytest tests/test_cost_lowerer.py -q  # 实现后 GREEN
```
