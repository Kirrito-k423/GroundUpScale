# C011 真实两层 Transformer 正确性摘要

## 结论

固定 Shape 的两层 causal Transformer reference 已在 CPU 与 MPS 上执行。相同种子、输入和权重下通过声明容差，所有目标设备审计与零物化别名审计通过，未启用 MPS fallback。

## 直接证据

| 指标 | 结果 |
|---|---:|
| CPU/MPS allclose (`atol=1e-4`, `rtol=1e-3`) | PASS |
| 最大绝对误差 | `7.152557373046875e-07` |
| 最大相对误差 | `3.2234183890977874e-06` |
| Semantic leaf | 52 |
| View/Transpose storage alias | 16/16 PASS |
| 参数 bytes | 33,562,624 |
| buffer bytes | 2,097,152 |
| MPS leaf/parameter/buffer device | 全部 `mps:0` |
| MPS fallback | false |
| CPU 输出 SHA-256 | `e3f953ab9f749d6b8f551d2bd3e93a109b54f61c648689c5fc42dad731833b23` |
| MPS 输出 SHA-256 | `62a9ec2c513397a1fae4fd6d36bbc8a6fdd724dbae20423a21b4770ae71fc5b7` |

## 布局反证

- `einsum("bhqk,bkhd->bqhd")` 的 CPU/MPS 结果都不是 contiguous，不能直接零拷贝 flatten。
- MPS 对非连续 `out=` 的 MatMul 在小 Shape 正确，但冻结 `S=512` 时最大绝对误差为 `152.42796`，已拒绝该实现。
- 最终 query-major batched MatMul 在 CPU/MPS 上都直接产出 contiguous `[B,S,NH,D]`，对标准 heads-major 参考最大绝对误差 `9.1553e-05`，最终 E2E 误差如上。

## 复现

```sh
uv run pytest tests/test_reference_runner.py -q
uv run pytest -q
```

结果：专项 `3 passed`，全量 `24 passed`。
