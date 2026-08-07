# Apple M4 CPU 可公开验证的建模能力

> 研究问题：针对仓库当前 `10-core CPU + 8-core GPU + 16GB unified memory` 的 Apple M4 配置，哪些 CPU 性能参数可以由第一方公开资料直接写入硬件规格，哪些只能符号化表达，哪些必须由实测校准？

- 资料访问日期：2026-08-07
- 来源约束：Apple 官方产品规格、Apple 开发者文档和 Arm 官方架构文档；未采用媒体、跑分网站或逆向工程结果。
- 结论先行：Apple 官方公开了核心组成和 SoC 统一内存带宽，但没有公开 M4 CPU 的频率、每周期浮点吞吐、CPU 专属可持续内存带宽或 CPU 峰值 FLOP/s。因此，当前能够负责任实现的是“公开峰值带宽的乐观下界 + 未知 CPU 峰值的显式缺口 + 后续实测效率曲线”，不能把第三方频率和逆向得到的流水线宽度伪装成官方理论值。

## 与仓库配置对应的具体机器

Apple 的 [MacBook Air (13-inch, M4, 2025) 技术规格](https://support.apple.com/en-us/122209) 同时列出：

- 10 核 CPU，其中 4 个性能核、6 个能效核；
- 可选 8 核或 10 核 GPU；
- 16GB 统一内存为基础配置；
- 120GB/s 内存带宽。

这与仓库当前 `apple-m4.yaml` 的 `10-core CPU + 8-core GPU + 16GiB` 组合相符。这里的 120GB/s 是 Apple 为 M4 产品公布的**统一内存系统带宽**，不是 Apple 单独承诺给 CPU 的可持续 DRAM 带宽，也不是某个 kernel 必然可达到的带宽。

## 证据分级表

| 建模字段 | 可写入的值或状态 | 证据等级与来源 | 建模边界 |
|---|---:|---|---|
| CPU 核心总数 | 10 | Apple 官方事实：[MacBook Air 技术规格](https://support.apple.com/en-us/122209) | 当前具体配置，不应泛化到所有 M4 产品 SKU |
| 性能核数量 | 4 | Apple 官方事实：[MacBook Air 技术规格](https://support.apple.com/en-us/122209) | 性能核和能效核必须分池建模 |
| 能效核数量 | 6 | Apple 官方事实：[MacBook Air 技术规格](https://support.apple.com/en-us/122209) | 不能假设与性能核同频、同吞吐 |
| 统一内存容量 | 16GiB（当前机器配置） | 仓库/机器配置；[Apple 官方规格](https://support.apple.com/en-us/122209)说明该产品有 16GB 配置 | 容量用于资源可行性，不用于推导时延 |
| SoC 统一内存峰值带宽 | 120GB/s，即 `120_000_000_000 B/s` | Apple 官方事实：[MacBook Air 技术规格](https://support.apple.com/en-us/122209) | 只适合作为全 SoC 共享链路的乐观上界；不是 CPU 实效带宽 |
| CPU ISA/ABI | Apple 平台 ARM64/AArch64 | Apple 官方事实：[Apple ARM64 ABI](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms) | 说明可使用标准 AArch64 能力；不说明微架构吞吐 |
| CPU 向量能力 | 公开 ABI 出现 NEON 向量类型；Arm AArch64 SIMD/FP 寄存器为 128 bit | Apple + Arm 官方事实：[Apple ARM64 ABI](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms)、[Arm A64 指南](https://developer.arm.com/documentation/102374/0103/Registers-in-AArch64---general-purpose-registers) | 能证明向量表达能力，不能证明每周期能发射几条向量指令 |
| 单条 FP32 向量 FMLA 的算法 FLOPs | 4 lanes × 2 FLOPs = 8 FLOPs | 由 [Arm Coding for Neon](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Neon%20Programmers%20Guide/102159_0104_01_CodingForNeon.pdf?revision=a3235e10-ded9-4ac8-a415-badcafdd56a6) 的指令语义可推导 | 只是一条指令完成的算法操作数，不是 FLOP/cycle |
| 单条 FP64 向量 FMLA 的算法 FLOPs | 2 lanes × 2 FLOPs = 4 FLOPs | 由 [Arm A64 指南](https://developer.arm.com/documentation/102374/0103/Registers-in-AArch64---general-purpose-registers) 的 128-bit 向量宽度和 FMLA 语义可推导 | 同上 |
| CPU ML/矩阵能力 | 性能核和能效核均有“下一代 ML accelerators” | Apple 官方定性事实：[M4 发布资料](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/) | Apple 未公开这些单元的 ISA、数据类型、tile 大小或峰值吞吐，不能填数值峰值 |
| CPU 主频 | 未找到 Apple 官方公开值 | 对上述 Apple 官方规格、芯片资料和开发者资料的检索结论 | 不把第三方测频作为 `theoretical` 来源 |
| 每核每周期 FMA/向量指令吞吐 | 未找到 Apple 官方公开值 | 对上述 Apple 官方规格、芯片资料和开发者资料的检索结论 | 没有该值就无法得到 CPU FLOP/cycle |
| CPU 理论峰值 FP32/FP64 FLOP/s | 未找到 Apple 官方值，且不能由现有官方参数唯一推导 | Apple 未披露必要的频率与发射吞吐；Arm ISA 不规定 Apple 微架构吞吐 | 必须保持 `unknown`，或使用明确标记为实测的 capability profile |
| CPU 可持续内存带宽 | 未找到 Apple 官方公开值 | Apple 只公开 [SoC 统一内存带宽](https://support.apple.com/en-us/122209) | 通过本机 benchmark 按访问模式、线程数和工作集大小测量 |

## CPU、SIMD 与矩阵能力的公开事实

### 核心组成

Apple 的 [M4 发布资料](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/) 说明 M4 CPU 最多包含 4 个性能核和 6 个能效核；性能核具有更宽的 decode 和 execution engine，能效核具有更深的 execution engine，两类核心都具有下一代 ML accelerators。Apple 只给出了相对 M2 的应用性能比较，没有给出 M4 的 CPU 时钟、浮点执行端口数、每周期指令数或按 dtype 划分的峰值吞吐。

因此硬件后端应把 CPU 表达为两个异构计算池，而不是一个“10 核同质 CPU”数字：

```text
M4 CPU
├── performance-core pool: 4 cores
└── efficiency-core pool:  6 cores
```

### 标准向量能力

Apple 的 [Writing ARM64 code for Apple platforms](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms) 明确 Apple 平台使用 ARM64 ABI，并专门规定了 NEON vector types 的 ABI 名称。Arm 的 [A64 Instruction Set Architecture Guide](https://developer.arm.com/documentation/102374/0103/Registers-in-AArch64---general-purpose-registers) 说明 AArch64 有 32 个 128-bit SIMD/FP 寄存器；Arm 的 [Coding for Neon](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Neon%20Programmers%20Guide/102159_0104_01_CodingForNeon.pdf?revision=a3235e10-ded9-4ac8-a415-badcafdd56a6) 展示了 `FMLA Vd.4S` 对四个 FP32 lanes 执行乘加。

若 GroundUpScale 沿用“乘法 + 加法 = 2 FLOPs”的计数口径，则可以安全推导：

```text
一条 128-bit FP32 vector FMLA = 4 lanes × 2 = 8 FLOPs
一条 128-bit FP64 vector FMLA = 2 lanes × 2 = 4 FLOPs
```

但是这仍缺少两个决定 FLOP/s 的实现参数：每类核心的持续频率，以及每周期可完成的向量 FMLA 数量。对 FP32，完整峰值只能保留为符号表达式：

```text
peak_fp32_flops_per_second
= 4 × performance_core_frequency_hz × performance_core_vector_fmla_per_cycle × 8
 + 6 × efficiency_core_frequency_hz  × efficiency_core_vector_fmla_per_cycle  × 8
```

Apple 没有公开上述四个变量，所以不能负责任地把这个式子化成一个数值。

### ML accelerators 与“矩阵指令”

Apple 公开确认 M4 的两类 CPU 核均带 ML accelerators，但没有公开其指令编码、矩阵 tile、支持 dtype、每周期操作数或峰值吞吐。Apple 的 [Accelerate 文档](https://developer.apple.com/documentation/accelerate) 则提供受支持的公开编程面：Accelerate 在 CPU 上利用向量处理能力，并让 BLAS、LAPACK、BNNS 等库在运行时选择适合处理器的指令。

Apple 还提供受额外协议约束的 [Apple Silicon CPU Optimization Guide](https://developer.apple.com/documentation/apple-silicon/cpu-optimization-guide) 入口，其公开 landing page 提到 Advanced SIMD/FP/SME 等主题；但这个入口不能证明某个具体 M4 SKU 一定支持 SME，也没有给出可用于 M4 的公开吞吐数值。因此 SME 必须由运行时 feature probe 确认，不能按 `model: M4` 静态硬编码为可用。

因此当前只能建模以下事实：

- CPU 后端可以选择 `Accelerate/BLAS/BNNS` 这一公开库实现路径；
- CPU 内存在 ML 加速能力，但其数值性能必须由库级或算子级实测获得；
- 不能仅凭 Apple 的“ML accelerators”描述，把网络文章常称的 Apple Matrix Coprocessor/AMX 吞吐写成官方峰值；
- Neural Engine 的 38 TOPS 是独立 Neural Engine 指标，不能作为 CPU FLOP/s。Apple 的 M4 发布资料明确把 CPU、GPU 和 Neural Engine 分列。

## 内存带宽如何进入模型

官方 120GB/s 可作为统一内存 fabric 的峰值上界：

```text
soc_unified_memory_peak_bandwidth = 120_000_000_000 B/s
```

对必须访问统一内存的 `bytes`，只能生成理想带宽下界：

```text
memory_time_optimistic_lower_bound = bytes / 120_000_000_000
```

该结果必须标注为 `soc-shared-peak-lower-bound`，原因包括：

- 120GB/s 由 CPU、GPU、Neural Engine、显示与其他 SoC 客户端共享；
- CostIR 的 logical/materialized bytes 不是已经验证的 DRAM traffic；缓存命中、写分配和融合都会改变实际流量；
- 线程数、访问模式、工作集大小、页状态、功耗和温度都会影响 CPU 的可持续带宽；
- Apple 没有给出 CPU 独享或 CPU 可持续带宽。

因此不得把 `bytes / 120GB/s` 单独当作预测时延。它可以与计算下界组合成 Roofline 风格的**乐观下界**：

```text
latency_lower_bound = launch_or_dispatch_overhead
                    + max(compute_lower_bound, memory_lower_bound)
```

但在 CPU 峰值计算能力未知时，`compute_lower_bound` 仍是 unknown，整个数值时延也不能宣称为完整的理论预测。

## 对 HardwareBackend 的落地建议

公开资料足以先建立一个不编造数据的 M4 CPU backend capability：

```yaml
cpu:
  architecture: arm64
  core_pools:
    - kind: performance
      count: 4
      frequency_hz: null
      vector_bits: 128
      sustained_vector_fmla_per_cycle: null
    - kind: efficiency
      count: 6
      frequency_hz: null
      vector_bits: 128
      sustained_vector_fmla_per_cycle: null
  matrix_acceleration:
    present: true
    public_numeric_capability: null
  memory:
    capacity_bytes: 17179869184
    soc_shared_peak_bandwidth_bytes_per_second: 120000000000
    cpu_sustained_bandwidth_bytes_per_second: null
```

预测状态应分开报告：

| 结果 | 当前可给出的状态 |
|---|---|
| FLOPs、逻辑/物化 bytes、容量占用 | `analytic` |
| 统一内存带宽乐观下界 | `analytic-lower-bound-from-public-peak` |
| CPU 计算时间 | `unavailable-missing-public-throughput`，直到有受控实测 profile |
| 完整算子/E2E 时延 | `calibrated` 或 `measured`，不能标成官方理论值 |

后续本地穿刺测试应校准的是公开规格没有提供的参数，例如：

- 按性能核/能效核亲和与线程数划分的有效 FP32/FP64 throughput；
- 按工作集和访问模式划分的可持续读、写、拷贝带宽；
- Accelerate SGEMM/DGEMM 对 shape 的有效吞吐曲线；
- PyTorch CPU kernel 的固定调度开销和有效效率。

这些结果应进入带 provenance 的 `CalibrationProfile`，不能回写成 `HardwareSpec.theoretical_peak`。

### 字段归属与 provenance 规则

实现时建议把事实来源固定为四层，禁止“有一个数就填进 HardwareSpec”：

| 层 | 允许承载的 M4 CPU 字段 | 必备 provenance | 禁止事项 |
|---|---|---|---|
| `HardwareSpec` | `performance_cores=4`、`efficiency_cores=6`、`unified_memory_peak_bandwidth_Bps=120e9`、`bandwidth_scope=soc_shared`、`isa=arm64`、`neon_vector_bits=128` | `source_kind=vendor_official`、Apple/Arm URL、访问日期、适用产品/SKU | 不得填第三方频率、逆向流水线宽度、实测 GFLOP/s |
| `HardwareBackend` 的解析规则 | 单条 FP32 FMLA 为 8 FLOPs、`T_mem_lower=bytes/120e9`、P/E 分池的符号峰值公式 | `source_kind=isa_derived`、规则版本、Arm ISA URL、公式假设 | 不得把单指令 FLOPs 当成每周期或每秒吞吐 |
| `RuntimeCapabilityProbe` | 当前机器实际报告的可用指令特性、OS/库版本、线程/核心可见性；未来若探测 SME 也放在此层 | `source_kind=runtime_observed`、探测命令/API、时间、机器 identity、原始输出摘要 | 不得把一次机器探测泛化为所有 M4 SKU 的静态事实 |
| `CalibrationProfile` | 微基准得到的有效 GFLOP/s、CPU 可持续带宽、kernel/dispatch overhead、shape/线程数相关效率 | `source_kind=measured`、Run Bundle IDs、代码与环境 fingerprint、样本数、统计量和误差 | 不得回写或重命名成 `official_peak` / `theoretical_peak` |
| 可选 `ExternalEstimateProfile` | 若产品未来确实需要采用公开论文、第三方研究或逆向资料中的频率/吞吐估值 | `source_kind=third_party_estimate`、原始来源、方法、假设、置信区间、适用版本 | 默认禁用；不得混入官方 `HardwareSpec`，不得作为无提示的真值 |

对 Apple 未公开的计算字段，静态规格应显式保持 unknown，而不是省略后让调用方猜测：

```yaml
theoretical_compute:
  fp32_flops_per_second:
    value: null
    status: unknown
    reason: vendor_does_not_publish_frequency_or_fma_issue_rate
  fp64_flops_per_second:
    value: null
    status: unknown
    reason: vendor_does_not_publish_frequency_or_fma_issue_rate
  cpu_matrix_operations_per_second:
    value: null
    status: unknown
    reason: vendor_does_not_publish_cpu_ml_accelerator_throughput
```

Backend 在遇到 `unknown` 时应产生可解释的部分结果，而不是填 0、无穷或猜测值：

```text
analytic_work             = available
memory_optimistic_bound   = available (official SoC-shared peak)
compute_theoretical_bound = unavailable (missing public throughput)
empirical_prediction      = available only when a compatible CalibrationProfile is selected
```

若同时存在官方静态值和实测 profile，输出应并列保留两者：官方值负责物理乐观边界，实测值负责当前软件栈预测；实测不能覆盖或修改官方原值。

## 不能声称的结论

截至访问日期，在上述第一方资料中**未找到**以下 Apple 官方公开值：

- M4 性能核或能效核的基准/最高/持续频率；
- 两类核心每周期可完成的 FP32、FP16、BF16 或 FP64 FLOPs；
- M4 CPU 的理论峰值 FLOP/s；
- CPU ML accelerators 的公开矩阵指令规格与峰值；
- CPU 独享、CPU 可持续或 cache 层级带宽。

所以“Mac M4 CPU 理论值应该是 public”只对核心数和 SoC 统一内存带宽成立；对 CPU 峰值 FLOP/s 并不成立。若产品需要一个可运行的时延估计，应明确采用“公开结构上界 + 本机微基准校准”的混合模型，并在每个结果上展示来源和置信状态。

## 来源清单

所有来源访问日期均为 2026-08-07。

1. Apple Support, [MacBook Air (13-inch, M4, 2025) - Tech Specs](https://support.apple.com/en-us/122209)：10 核 CPU（4P+6E）、8/10 核 GPU、16GB 统一内存、120GB/s。
2. Apple Newsroom, [Apple introduces M4 chip](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/)：M4 CPU 核心组成、两类核心的定性微架构改进、两类核心均有下一代 ML accelerators；CPU/GPU/Neural Engine 的边界。
3. Apple Developer Documentation, [Writing ARM64 code for Apple platforms](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms)：Apple ARM64 ABI 及 NEON vector types。
4. Apple Developer Documentation, [Accelerate](https://developer.apple.com/documentation/accelerate)：CPU 向量处理、BLAS/LAPACK/BNNS，以及按运行处理器选择适当实现的公开软件接口。
5. Arm Developer, [A64 Instruction Set Architecture Guide — Registers in AArch64](https://developer.arm.com/documentation/102374/0103/Registers-in-AArch64---general-purpose-registers)：32 个 128-bit SIMD/FP 寄存器和向量视图。
6. Arm Developer, [Coding for Neon, Issue 04](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Neon%20Programmers%20Guide/102159_0104_01_CodingForNeon.pdf?revision=a3235e10-ded9-4ac8-a415-badcafdd56a6)：128-bit Neon 寄存器、四个 FP32 lanes 和向量 FMLA 示例。
7. Apple Developer Documentation, [Apple Silicon CPU Optimization Guide](https://developer.apple.com/documentation/apple-silicon/cpu-optimization-guide)：公开入口列出的优化主题；正文访问受额外协议约束，故不能据其入口断言 M4 的 SME 支持或吞吐。
