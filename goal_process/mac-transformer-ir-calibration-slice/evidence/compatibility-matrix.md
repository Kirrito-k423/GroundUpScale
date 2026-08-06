# 版本兼容性矩阵

| 组件 | 当前版本 | 要求/接口 | 证据来源 | 候选版本 | 风险 | 结果 |
|---|---|---|---|---|---|---|
| Python | 3.11.15 | PyTorch 官方建议 Python 3.9–3.12；存在 cp311 macOS arm64 wheel | 本机 C001；[PyTorch Start Locally](https://docs.pytorch.org/get-started/locally/)；[PyPI torch 2.13.0](https://pypi.org/project/torch/2.13.0/) | 3.11.15 | 低 | 通过 |
| PyTorch | 2.13.0 | CPU、MPS、目标操作、`torch.mps.synchronize` 与 allocator 接口 | [PyTorch MPS notes](https://docs.pytorch.org/docs/stable/notes/mps.html)；C001 runtime 探针 | 2.13.0 | 低 | 通过 |
| macOS/Metal | 15.7.4 / Metal 3 | MPS backend available | 本机 `system_profiler`；C001 `is_built/is_available=true` | 不切换 | 低 | 通过 |
| uv | 0.11.14 | Python 3.11 本地环境与 lock | 本机版本 | 0.11.14 | 低 | 可用 |
| 测试框架 | pytest 9.0.2 | 公开 seam 的 red/green 测试 | `uv.lock`；C001 RED/GREEN | 9.0.2 | 低 | 通过 |
| NumPy | 2.4.6 | PyTorch Python runtime 的数组互操作依赖 | C001 启动告警；`uv.lock`；补充后快速 probe 无告警 | 2.4.6 | 低 | 通过 |

## C001 选择结论

- 选择 Python 3.11.15 + PyTorch 2.13.0；前者处于 PyTorch 官方建议范围，后者存在 CPython 3.11 的 macOS 14+ arm64 wheel。
- MPS 的 `is_available()`、同步与 allocator 接口仍须由当前机器 runtime 探针确认，wheel 存在不等于操作集可用。
- 若安装或 runtime 失败，只允许在补充官方兼容证据后尝试第二个候选；不得循环猜版本。
