# C001 CPU/MPS 环境探针摘要

## 冻结环境

- Python：CPython 3.11.15
- PyTorch：2.13.0
- 平台：Darwin arm64，PyTorch MPS `built=true`、`available=true`
- 随机种子：20260806
- Tensor：512 × 512，FP32
- 测量：warmup 5，20 个同步样本，每个样本执行一次完整操作组
- 数值容差：`atol=1e-5`、`rtol=1e-4`

## 正确性

| 操作 | CPU | MPS | MPS 最大绝对误差 | MPS 最大相对误差 |
|---|---:|---:|---:|---:|
| MatMul | PASS | PASS | 0 | 0 |
| Add | PASS | PASS | 0 | 0 |
| RMSNorm | PASS | PASS | 9.536743e-07 | 2.380825e-07 |
| Softmax | PASS | PASS | 1.490116e-08 | 5.391923e-07 |
| SiLU | PASS | PASS | 4.768372e-07 | 2.347775e-07 |
| Mul | PASS | PASS | 0 | 0 |
| View | PASS | PASS | 0 | 0 |
| Transpose | PASS | PASS | 0 | 0 |

## 时延噪声

| 设备 | median | IQR | IQR / median | 3% 门禁 |
|---|---:|---:|---:|---:|
| CPU | 507,854 ns | 41,959 ns | 8.262% | FAIL |
| MPS | 1,564,833 ns | 82,416 ns | 5.267% | FAIL |

结论：能力与正确性通过，但约 0.5–1.6 ms 的单操作组窗口过短，不能作为 5% 验收的测量协议。C002 将把多个操作组放进一个同步计时窗口，再按组数归一化；这不是同配置重跑。

## 内存观测

| 设备/口径 | before | after | delta | 归因边界 |
|---|---:|---:|---:|---|
| CPU process RSS | 203,833,344 | 235,421,696 | 31,588,352 | 进程级，仅诊断 |
| MPS current allocated | 0 | 8,388,608 | 8,388,608 | framework-attributed |
| MPS driver allocated | 475,136 | 17,268,736 | 16,793,600 | Driver 单列，不当作逻辑 Tensor |

MPS 的 8 MiB 正好对应本探针仍存活的两份输入/权重与六份物化输出；View/Transpose 与输入共享存储、没有另计物化量。这是 allocator 接口可用于后续 live-set 对齐的直接证据，但尚不是 AC-08 的完整证明。
