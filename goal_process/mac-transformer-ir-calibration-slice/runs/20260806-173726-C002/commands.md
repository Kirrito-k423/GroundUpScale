# C002 脱敏命令记录

```sh
uv run pytest tests/test_probe_cli.py -q  # 预期 RED：inner_iterations 尚未实现
uv run pytest tests/test_probe_cli.py -q  # 实现后 GREEN
uv run groundupscale probe --device cpu --device mps --require-mps --warmup 5 --repeats 20 --inner-iterations 50 --matrix-size 512 --json
```
