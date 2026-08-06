# C001 脱敏命令记录

## 已执行基线命令

```sh
git status --short --branch
git rev-parse HEAD
sw_vers
system_profiler SPHardwareDataType SPDisplaysDataType  # 只保留型号/核心/内存/Metal，删除序列号与 UUID
python3 --version
python3.11 --version
uv --version
python3 -c 'import torch'  # 预期基线失败：未安装
```

## 待执行探针命令

```sh
uv lock --python 3.11
uv sync --python 3.11 --group dev
uv run pytest tests/test_probe_cli.py -q  # 预期 RED：公开 CLI 尚未实现
uv run pytest tests/test_probe_cli.py -q  # 实现后 GREEN
uv run groundupscale probe --device cpu --device mps --require-mps --warmup 5 --repeats 20 --matrix-size 512 --json
```

不得记录 token、密码或私有环境变量；命令输出的必要摘要写入本轮证据文件。
