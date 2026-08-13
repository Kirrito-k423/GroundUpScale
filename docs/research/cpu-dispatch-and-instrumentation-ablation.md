# CPU MatMul 固定开销与维测开销消融实验

一句话结论：当前 M4 CPU 正式 Benchmark 不包含 forward hooks 或逐模块打印；
CPU MatMul 计时区间也没有 H2D。实验没有发现能解释 E2E 差距的显著正固定下发
截距，完整诊断 Trace 本身会增加约 5%～8% 的暖态 E2E 开销，但它是独立执行，
不会污染正式 Benchmark 中位数。

## 假说和判定条件

1. 如果存在类似 NPU H2D/下发的显著固定时间 `a>0`，在只让 MatMul 的 M 维和
   FLOPs 扩大 1/2/4/8/16/32 倍时，`T=a+bF` 应拟合出稳定正截距，且小时延增长
   应小于 FLOPs 增长。
2. 如果维测污染正式 E2E，正式 invocation 在计时前仍应有 active hooks，或关闭
   hooks 后 E2E 应出现可重复的大幅下降。

运行命令：

```sh
uv run python scripts/run-cpu-ablation.py \
  --repository-root . \
  --output goal_process/mac-transformer-ir-calibration-slice/evidence/cpu-dispatch-hook-ablation-20260807-v2.json \
  --samples 31 --warmup 10 --target-window-ms 30
```

环境为 macOS 15.7.4 arm64、PyTorch 2.13.0、4 个 intra-op threads 和 10 个
interop threads。每个实验点 31 个样本；Shape 和 instrumentation mode 交错随机
执行，避免固定顺序把温度、DVFS 或后台调度偏差归给某个配置。

## 实验一：MatMul Shape 缩放

固定 `K=N=512`，只把输入 `[1,M,512]` 的 `M` 从 512 放大到 16384，因此
`FLOPs=2*M*K*N` 与倍率严格线性。输入和权重在计时前分配并驻留 CPU；计时区间
只包含 `torch.matmul` 和输出分配，没有 `.to()`、copy 或 H2D。

| FLOPs 倍率 | M | 中位数 | 耗时/1× | 有效能力 |
|---:|---:|---:|---:|---:|
| 1× | 512 | 0.1562 ms | 1.000× | 1.718 TFLOP/s |
| 2× | 1,024 | 0.4097 ms | 2.622× | 1.311 TFLOP/s |
| 4× | 2,048 | 0.8145 ms | 5.214× | 1.318 TFLOP/s |
| 8× | 4,096 | 1.6590 ms | 10.619× | 1.294 TFLOP/s |
| 16× | 8,192 | 2.6051 ms | 16.675× | 1.649 TFLOP/s |
| 32× | 16,384 | 7.3676 ms | 47.160× | 1.166 TFLOP/s |

全 Shape 仿射拟合：

```text
T = intercept + FLOPs / rate
intercept = -194.3 us, bootstrap 95% CI [-200.8, -177.2] us
rate      = 1.193 TFLOP/s, bootstrap 95% CI [1.187, 1.210] TFLOP/s
R²        = 0.9773
```

负截距不是物理负开销，而是单一常速模型失配：有效能力随 M、缓存域、BLAS kernel、
线程调度和系统状态在 `1.166～1.718 TFLOP/s` 间变化。更关键的是，正固定截距会让
耗时倍率小于 FLOPs 倍率；实测除 1× 外均相反。因此数据不支持“显著固定下发时间”
是这些 Shape 慢的主因。

补充的 `[1,0,512] @ [512,512]` 零工作调用中位数为 `0.495 us`，约为 1×
MatMul 的 `0.317%`。它只近似 Python/ATen dispatcher、shape check 和空输出路径，
不能覆盖真实 BLAS 线程调度的全部固定成本，但足以说明不存在可解释毫秒级差距的
CPU H2D 项。CPU 在本实验中直接访问 CPU resident tensors；缓存和 DRAM 流量属于
memory resource demand，不应命名为 H2D。

## 实验二：forward-hook 维测消融

实验直接复用 `BenchmarkRunner._invocations()` 生成的正式两层 E2E invocation，
以变量切换四种模式；hook 的注册和移除均在计时区间外：

- `off`：正式 Benchmark 路径；
- `empty-hooks`：118 个 pre/post handles，回调不做工作；
- `timing-hooks`：回调读取 `perf_counter_ns` 并保存 duration；
- `full-trace-hooks`：进一步采集 tensor metadata、RSS 和结构化事件。

正式输入捕获完成后 active hook 数为 0，代码中也没有逐模块 `print`。完整
TraceRunner 是 Benchmark 之后的另一轮执行。

| 模式 | 第一次中位倍率及 95% CI | 第二次中位倍率及 95% CI | 结论 |
|---|---:|---:|---|
| off | 1.000× | 1.000× | 正式基线 |
| empty-hooks | 1.003× `[0.987,1.024]` | 1.013× `[1.002,1.031]` | 约 0%～1.3% |
| timing-hooks | 1.003× `[0.985,1.025]` | 1.008× `[0.998,1.032]` | 未稳定检出材料性开销 |
| full-trace-hooks | 1.048× `[1.033,1.071]` | 1.076× `[1.065,1.098]` | 暖态约增加 5%～8% |

第二次受控实验中，`off=16.384 ms`，`full-trace-hooks=17.636 ms`，关闭完整维测
只减少 `1.251 ms`。关闭后仍是 `6.833 ms` 串行硬件地板的 `2.398×`，因此“维测
导致主要 E2E 差距”被实验否定。

当前 v4 Run 的正式 Benchmark E2E 中位数为 `16.823 ms`，单次诊断 Trace E2E
为 `19.641 ms`，高 `16.8%`。这个差值同时包含完整 hook、单次未暖透执行和普通
系统噪声，所以不能用单个 Trace/Benchmark 比值估计 hook 税；受控交错消融的
`5%～8%` 才是较可靠的暖态 instrumentation overhead。

## 对建模和报告的影响

- 52 个叶子操作来自 SemanticIR/reference model 的真实模块结构，不是 hooks
  “打印出来”的额外算子。hooks 改变每个 span 的时间，不改变算子数量。
- 正式吞吐和 E2E 校验继续只使用 minimally instrumented Benchmark；关闭维测不会
  改变该值，因为计时区间本来就是 `off`。
- Top 10 实测分解来自单次诊断 Trace，必须继续标记为 diagnostic evidence，不能
  当作各算子 Benchmark median，也不能直接拿它校准 Duration Model。
- CPU HardwareBackend 暂不增加 H2D 项。若未来要刻画极小 CPU 算子，可增加独立的
  `host_dispatch/allocator/threadpool` microbenchmark；NPU/GPU 则应在 ExecutionIR
  中把 host launch、H2D/D2H 和 device kernel 作为不同事件建模。
- 后续可给 Trace 增加可选 warmup，并把 paired `off/full-trace` instrumentation tax
  写进 Run Bundle；该税只用于解释观测偏差，不能回填为模型算子耗时。

原始证据：

- `goal_process/mac-transformer-ir-calibration-slice/evidence/cpu-dispatch-hook-ablation-20260807.json`
- `goal_process/mac-transformer-ir-calibration-slice/evidence/cpu-dispatch-hook-ablation-20260807-v2.json`
