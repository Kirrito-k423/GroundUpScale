# C008 脱敏命令记录

```sh
uv run pytest tests/test_semantic_compiler.py -q  # 预期 RED：SemanticCompiler 尚不存在
uv run pytest tests/test_semantic_compiler.py -q  # 实现后 GREEN
uv run pytest tests/test_compile_cli.py -q
uv run pytest -q
```
