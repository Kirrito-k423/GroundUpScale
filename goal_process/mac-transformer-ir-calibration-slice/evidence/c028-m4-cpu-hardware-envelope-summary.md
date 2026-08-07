# C028：M4 CPU 多 Shape 能力包络与两层样例对照

- **日期：** 2026-08-07
- **目标设备：** Apple M4 CPU，4P+6E，16 GiB unified memory
- **Suite：** `specs/microbenchmarks/apple-m4-cpu.yaml`
- **有效原始观测：** `apple-m4-cpu-microbenchmark-observation-v2.json`
- **能力配置：** `specs/hardware-capabilities/apple-m4-cpu-local.yaml`
- **样例 Run Bundle：** `.groundupscale/runs/m4-cpu-envelope-20260807-v2`

## 能力测量结果

每个探针使用 12 个 Shape；同一 Shape 先在声明的线程数中选择最高稳定速率，
再跨 Shape 计算 P80/P95。多个探针映射到同一资源时，资源包络取探针 P80/P95
的最大值，但保留全部探针及 Shape 证据。

| 资源/探针 | P80 | P95 | 解释 |
|---|---:|---:|---|
| `compute.fp32` / matrix cube | 1.74845 TFLOP/s | 1.87490 TFLOP/s | 当前跨算子 FP32 经验硬件上界；用于算法无关地板 |
| scalar FP32 FMA | 13.8273 GFLOP/s | 14.0048 GFLOP/s | ARM64 原生 8 路独立 scalar FMADD |
| vector FP32 FMA | 27.6877 GFLOP/s | 28.3610 GFLOP/s | PyTorch elementwise vector 探针 |
| `memory.shared` / copy | 126.833 GB/s | 133.844 GB/s | read+write 聚合流量；P80 成为资源包络 |
| shared-memory triad | 95.7772 GB/s | 96.2024 GB/s | 3 reads + 1 write 聚合流量 |

Apple 官方 SoC 共享峰值为 `120 GB/s`。copy P80 比该数高 `5.69%`，处于预设
的 10% 语义核对预算内；P95 高 `11.54%`，因此只作为乐观经验边界，不用于
当前稳健地板。Apple 没有发布可比较的 CPU FP32 FLOP/s，所以不得伪造计算峰值
对照。

本轮 preflight 为 `eligible=false`，原因是 `load-above-policy` 和
`total-competing-cpu-above-policy`。用户已明确接受本轮 macOS 调度干扰用于功能
穿刺，因此该 Profile 可供 exploratory prediction 使用，但不能晋升为 trusted CI
基线。

## 算法无关地板公式

```text
T_compute = minimum mathematical FLOPs / compute.fp32 P80
T_memory  = unique scope-boundary compulsory bytes / memory.shared P80
T_floor   = max(T_compute, T_memory)
```

这里的 `compulsory bytes` 只统计 Scope 外部输入、唯一状态和外部可见输出，
不再使用当前实现的全部 `materialized bytes`。实现引入的中间物化、转换、重计算、
workspace、dispatch 和调度开销保留在更高一层，不会污染硬件能力定义。

## 两层 Transformer 预测—实测对照

| Case | 硬件地板 | 实测 median | 实测/地板 | 限制资源 | IQR/median |
|---|---:|---:|---:|---|---:|
| `matmul-q-proj` | 0.153528 ms | 0.154288 ms | 1.005× | compute.fp32 | 0.35% |
| `rmsnorm-input` | 0.016551 ms | 0.062993 ms | 3.806× | memory.shared | 1.60% |
| `softmax-attention` | 0.132278 ms | 0.699458 ms | 5.288× | memory.shared | 1.88% |
| `transformer-layer` | 2.776988 ms | 45.059563 ms | 16.226× | compute.fp32 | 3.42% |
| `two-layer-prefill` | 5.553976 ms | 92.814479 ms | 16.711× | compute.fp32 | 2.41% |

Q projection 与 matrix 能力包络非常接近，说明该 Shape 的矩阵实现已接近本轮
探针可达值。模块/E2E 的较大距离属于实现和调度优化空间，不是硬件地板的预测
误差。两层 E2E 的 `IQR/median=2.41%` 达到单 Case 噪声要求，但第一层 module 为
`3.42%`，且运行前环境门禁不合格，因此整次 Run 仍只能作为 exploratory evidence。

框架 Tensor-storage 峰值预测为 `54,534,144 B`，实测为 `69,214,208 B`，预测
少 `14,680,064 B`，绝对相对差 `21.21%`。该指标仍指向 live-set/框架存储归因
层，与本轮耗时硬件地板的能力包络彼此独立。

## 验证

- 原始观测 SHA-256 被 Capability Profile 锁定，AnalysisPlan 加载时会重新核验。
- Run Bundle 共 17 个 artifact，`verify-run` 全部 digest 通过。
- 当前全量测试：`58 passed`。
