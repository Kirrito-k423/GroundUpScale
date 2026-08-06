# C006 脱敏命令记录

```sh
uv lock --python 3.11
uv sync --python 3.11 --group dev
uv run pytest tests/test_spec_repository.py -q  # 预期 RED：公开 seam 尚不存在
uv run pytest tests/test_spec_repository.py -q  # 实现后 GREEN
```
