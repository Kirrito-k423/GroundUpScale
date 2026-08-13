# [研究跨硬件 Hardware Validity Cohort、计时与观测能力矩阵](https://github.com/Kirrito-k423/GroundUpScale/issues/4)验收审计

- 验收日期：2026-08-07
- 地图：[跨硬件性能预测差距诊断规范](https://github.com/Kirrito-k423/GroundUpScale/issues/1)
- 研究票：[研究跨硬件 Hardware Validity Cohort、计时与观测能力矩阵](https://github.com/Kirrito-k423/GroundUpScale/issues/4)
- 研究结论：[「研究跨硬件 Hardware Validity Cohort、计时与观测能力矩阵」的研究结论评论](https://github.com/Kirrito-k423/GroundUpScale/issues/4#issuecomment-5215287107)
- 规范词汇：[CONTEXT.md](../../CONTEXT.md)
- 审计边界：只验收现有研究，不重新开展研究，不验收或修改生产实现，不改动 Issue/地图状态。

## 结论

**PASS，可以形成规范决策并关闭[研究跨硬件 Hardware Validity Cohort、计时与观测能力矩阵](https://github.com/Kirrito-k423/GroundUpScale/issues/4)。**

研究评论已经提供了 ticket 所要求的按后端能力分类矩阵、统一 adapter 契约、cohort
失效规则、计时完成边界以及 counter 缺失时的降级语义。Exit criteria 要求的两件事均已
明确：

1. 通过 `R / O / U / N/A` 和逐字段矩阵区分跨硬件必需、可选、不可得和不适用；
2. 通过六条降级规则明确何时为 `unknown` / `insufficient_evidence`，以及何时允许带
   `proxy + attribution` 标记的代理指标。

没有需要阻止关闭的证据缺口，也不应创建新的 decision ticket。下文列出的措辞收紧项
可以在落规范时一并处理，不改变研究结论。

## 票面与“Shape 连续插值冲突”的范围核对

GitHub 上[研究跨硬件 Hardware Validity Cohort、计时与观测能力矩阵](https://github.com/Kirrito-k423/GroundUpScale/issues/4)
的实际问题是跨硬件 `Hardware Validity Cohort`、计时和观测能力，
其 Exit criteria **没有**要求研究 Shape 插值。地图把 Shape 研究明确链接到已关闭的
[研究连续能力曲面的 Shape 域、插值与不确定性协议](https://github.com/Kirrito-k423/GroundUpScale/issues/2)，
而当前工作树的 `CONTEXT.md` 已用 `Validated Shape Regime` 与 `Capability Surface`
承接该决定。

因此，本次需要做的是一致性验收，而不是把「研究跨硬件 Hardware Validity Cohort、计时与
观测能力矩阵」改造成另一张 Shape 研究票。结果是：
两项要求没有规范冲突。

设权威定义域是多个已验证连通 regime 的并：

```text
D_authoritative = R_1 ∪ R_2 ∪ ... ∪ R_n
f_i: R_i -> capability_rate
```

- `f_i` 在各自的连通 `Validated Shape Regime` 内 exact-knot 且连续；
- 未验证的 alignment、working-set、candidate-support、kernel/algorithm seam 是定义域的洞，
  不是要求连续跨越的插值区间；
- 一个查询若需要用 seam 两侧的 Anchor 共同构造插值 cell，或者不能确定自己属于哪个
  regime，权威 Frontier 返回 `unknown`；
- seam 两侧的精确 Anchor 仍各自在自己的 exact Shape 有效；`unknown` 否定的是跨 seam
  的权威插值，不是否定已有的精确观测；
- 只有独立证据验证了 seam 连续性、共同候选支持和不确定性覆盖后，才可合并 regime 或
  将 seam 纳入一个新的有效 regime；这时它已不再是“未验证的跨 regime 插值”。

换言之，“连续”是**局部且受定义域约束的连续**，不是全 Shape 空间的全局连续。
「研究跨硬件 Hardware Validity Cohort、计时与观测能力矩阵」又增加了更外层的边界：
不同 `Hardware Validity Cohort` 维护独立 Surface，绝不
用一个 cohort 的曲面填补另一个 cohort 的缺失字段。

## Exit criteria 逐项验收

| 验收项 | 证据 | Verdict |
|---|---|---|
| 给出按后端能力分类的协议矩阵 | 研究评论覆盖 Linux/M4 CPU、CUDA GPU、Ascend NPU，并逐项列出设备身份、软件栈、执行域、功耗策略、host/device timer、trace/counter、内存、通信、正确性、维测模式和 capability manifest | **PASS** |
| 给出统一 adapter 契约 | 评论定义 status-rich metric record，并给出 `discover_capabilities`、`fingerprint_cohort`、`preflight`、`build_timing_plan`、`collect` 五步 | **PASS** |
| 明确跨硬件必需字段 | 稳定设备/拓扑身份、OS/driver/runtime、framework/compiler/operator library、执行有效域、主计时、正确性/warmup/repetition/raw samples、instrumentation mode 和 capability manifest 均为 required 或 required-status | **PASS** |
| 明确可选、不可得和不适用字段 | `O / U / N/A` 显式区分可选、经探测不可得和概念不适用；每个格子必须有状态，不能静默省略 | **PASS** |
| 缺失时何时输出 unknown | required identity、正确性、完成边界或主计时缺失时不得成为 Anchor，verdict 至少为 `insufficient_evidence`；optional counter 缺失时，只把依赖该 counter 的归因置为 `unknown` | **PASS** |
| 缺失时何时允许代理指标 | 只允许语义量或模型量除以 elapsed 得到明确标记的 effective/algorithm proxy，并必须保留 scope 与 attribution；不得冒充 DRAM traffic、executed FLOPs 或 Case 专属通信量 | **PASS** |
| cohort 失效规则 | 评论分别列出身份/软件/执行/功耗/测量/通信协议的硬分叉条件，以及只应 quarantine/retry 的瞬时条件 | **PASS** |
| 具体平台事实来自官方文档或一手实验 | 评论引用 Linux kernel、Open MPI、Apple、NVIDIA/CUDA/NVML/CUPTI/NCCL 和华为 Ascend/CANN/HCCL 官方资料；关键契约均有一手资料支持 | **PASS，带非阻塞措辞收紧** |

## 一手证据清单与核验结果

### CPU / Linux / Apple

- [Linux CPUFreq](https://docs.kernel.org/admin-guide/pm/cpufreq.html)：支持动态 P-state、
  governor/允许频率范围属于运行条件，`scaling_cur_freq` 不是跨平台可靠的瞬时硬件真值。
- [Linux perf security](https://docs.kernel.org/admin-guide/perf-security.html)：支持
  `perf_event_paranoid` / `CAP_PERFMON` 权限门禁，以及 IMC、互连、PCIe uncore 指标通常
  没有进程执行上下文直接归因。
- [Linux interface statistics](https://docs.kernel.org/networking/statistics.html)：支持标准
  interface statistics 与 driver-defined ethtool statistics 的来源边界；这些系统/端口级
  统计不能自动提升为当前 Case 的专属通信量。
- [Open MPI `MPI_Wtime`](https://docs.open-mpi.org/en/v5.0.2/man-openmpi/man3/MPI_Wtime.3.html)：
  明确 Open MPI 不保证各节点时钟同步，支持每 rank 本地 duration 再聚合的契约。
- [Apple MacBook Air M4 规格](https://support.apple.com/en-us/122209)：支持 10-core CPU、
  16GB unified memory 配置和 120GB/s SoC memory bandwidth；没有给出 CPU 峰值 FLOP/s
  或 CPU 独享持续带宽。
- [Apple Instruments CPU Counters / Processor Trace](https://developer.apple.com/videos/play/wwdc2025/308/)：
  明确 CPU Counters 依赖 workload sampling，属于诊断 profiling 路径。

### CUDA / NVIDIA

- [CUDA event management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)：
  支持 event completion、约 0.5 微秒 elapsed-time 分辨率，并明确异步 event 之间可能执行
  其他 stream 工作而显著改变计时。
- [CUDA synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)：
  支持 host/device 异步语义必须按 API、参数和 stream 明确完成边界。
- [`cudaDeviceProp`](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)：支持
  UUID、compute capability、PCI、L2、global memory、async engine 和并发能力等指纹字段。
- [CUDA version management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html)：
  `cudaDriverGetVersion` 与 `cudaRuntimeGetVersion` 是不同查询，支持分别记录 driver/runtime。
- [NVML device queries](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html) 与
  [NVLink methods](https://docs.nvidia.com/deploy/nvml-api/group__NvLink.html)：支持设备/功能相关
  查询和 `NOT_SUPPORTED` / privilege-dependent 返回，因此必须 capability-probe。
- [CUPTI overview](https://docs.nvidia.com/cupti/main/main.html)：支持 activity correlation、
  MPS/MIG/vGPU/confidential-compute 等部署限制，以及部分 metric 需要 multi-pass/replay 的事实。
- [NCCL tests performance semantics](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)：
  明确区分 algorithm bandwidth 和按 collective 调整的 bus bandwidth。

### Ascend / Huawei

- [CANN 异步/同步契约 PDF](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/81RC1alpha002/apiref/appdevgapi/CANN%E7%A4%BE%E5%8C%BA%E7%89%88%208.1.RC1.alpha002%20%E5%BA%94%E7%94%A8%E5%BC%80%E5%8F%91%E6%8E%A5%E5%8F%A3%E5%8F%82%E8%80%83%2001.pdf)
  与 [`aclrtEventElapsedTime`](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/appdevgapi/aclcppdevg_03_0090.html)：
  支持同一 Stream 记录起止 Event、同步 Stream 后计算 elapsed time 的完成边界。
- [`npu-smi` basic info](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/2520/re/npu/npusmi_007.html)、
  [board info](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/24.1.0/RC/driverdevelopmentguide/atlasdg_11_0153.html)、
  [usage info](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/2520/re/npu/npusmi_020.html)：
  支持 device/chip、Bus ID、health、power、temperature、AICore、memory、board/PCB/BOM、
  software/firmware 字段，并显示物理机/容器/用户权限的支持差异。
- [`torch_npu.profiler` API](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/apiref/apilist/ptaoplist_001215.html)
  与 [Ascend profiler data](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0033.html)：
  支持 runtime 查询 activities、profiler level、AI Core metrics，以及 operator/memory/通信
  等诊断数据；若干采集项明确带额外性能开销。
- [HCCL profiling fields](https://www.hiascend.com/document/detail/en/mindstudio/700/TITools/Profiling/atlasprofiling_16_0066.html)：
  支持 communicator/rank、LOCAL/SDMA/RDMA transport、HCCS/PCIe/RoCE link type、size、
  bandwidth 和计算通信 overlap 字段。
- [HCCL Performance Tester](https://www.hiascend.com/document/detail/en/canncommercial/800/devaids/hccltool/HCCLpertest_16_0001.html)：
  明确工具报告的是受 collective 算法影响的 algorithm bandwidth。
- [HCCL FAQ：Profiling 对带宽的影响](https://www.hiascend.com/document/detail/en/canncommercial/800/hcclug/hcclug/hcclug_000046.html)：
  直接说明开启 profile data collection 会降低 bandwidth，支持将 profiling timing 与
  baseline timing 隔离；具体影响量仍取决于平台和采集配置。
- [CANN profiling option constraints](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendgraphapi/atlasgeapi_07_0150.html)：
  明确 `task_time=l0` 比 `l1` 少采字段、开销更低，并列出产品支持差异和
  `instr_profiling_freq` 与 `task_time` / `hccl` / `aic_metrics` / `l2` 等互斥条件。

## 非阻塞措辞收紧

以下问题不影响 Exit criteria，但应避免把有方向性支持的资料写成过强的平台保证：

1. `R-status` 应定义为“字段 envelope 和 availability reason 必须存在”，不等于所有平台
   都必须产生数值。例如 M4 frequency 可以是 `unsupported/unknown`，但状态不能缺失。
2. 华为官方 HCCL FAQ 已直接支持“profile data collection 会降低 bandwidth”；但影响量
   随平台、工具版本和采集配置而变，不能预设一个通用修正比例。因此规范仍只采用：
   profiling lane 的耗时默认不得晋级 Anchor；只有独立消融证明开销落入声明的 Error
   Budget，并将 instrumentation mode 纳入有效域后，才可例外晋级。
3. NVML 的保守结论应写成“字段和错误码随 device/feature/privilege 而变，必须运行时探测”，
   不需要依赖笼统的“某类消费卡支持有限”概括。
4. `unknown` 是报告层的归因结果；原始字段仍必须保留
   `unsupported / permission_denied / not_requested / collection_failed` 等原因，不能只留 `null`。

## 可直接落入规范的决策文本

以下英文文本可放入 `CONTEXT.md` 的词汇定义，或作为 ADR 的规范条款；它不要求修改
生产实现。

### Capability Surface 与 regime seam

> A Capability Surface is a versioned partial function whose authoritative
> domain is the union of connected Validated Shape Regimes. It is exact at
> Frontier Anchor knots and continuous within every retained interpolation
> cell of one regime. Alignment, working-set, candidate-support, and
> kernel/algorithm seams are holes in the authoritative domain unless
> independent evidence validates continuity, common support, and uncertainty
> coverage across the seam. A query that requires an interpolation cell across
> an unvalidated seam, or whose regime membership cannot be established,
> returns `unknown` for the authoritative Frontier. An exact observation on
> either side remains reportable; a continuous provisional baseline may be
> used only to plan probes and never to issue a diagnosis verdict.

### Cross-hardware observation contract

> Frontier evidence MUST use a minimally instrumented baseline timing lane
> with correctness, an explicit completion boundary, raw repeatable timing,
> and a complete Hardware Validity Cohort and execution-domain fingerprint.
> Intrusive PMU, CUPTI, msprof, HCCL, trace, or replay collection belongs to a
> paired diagnostic profiling lane identified by `pair_id`; its timing MUST NOT
> constrain the Frontier unless an independent ablation shows its overhead is
> within the declared Error Budget and the instrumentation mode is part of the
> validity domain.

> Every observation field MUST carry an availability status and source. Missing
> required identity, correctness, completion, or primary timing evidence makes
> the run ineligible for Frontier admission and yields at least
> `insufficient_evidence`. Missing optional counters do not invalidate an
> otherwise qualified baseline, but every attribution that depends on them is
> `unknown`. Missing or denied observations MUST NOT be encoded as zero.
> Derived proxies are allowed only when labeled with their derivation, scope,
> and attribution; effective algorithm bandwidth is not physical link or memory
> traffic, and modeled FLOP/s is not executed hardware FLOP/s.

> Hardware identity/configuration, OS/kernel, driver/firmware/runtime,
> framework/compiler/operator libraries, numeric and execution modes,
> power/clock policy, timing protocol, and communication topology/algorithm are
> part of the validity key. A material change creates a separate Hardware
> Validity Cohort unless an explicit A/B equivalence study authorizes
> compatibility. Transient throttling, contention, health faults, unstable
> samples, and collection failures quarantine or retry a run; they MUST NOT be
> normalized by inventing a slower cohort.

## 是否需要新 decision ticket

**不需要。** 研究票自身的 Exit criteria 已满足；Shape 连续性/unknown 的语义也已由地图
既有 Shape 决策和当前 `CONTEXT.md` 的局部定义域模型解决。为同一问题再开票会产生重复
决策源。

只有当项目拒绝“每个 connected regime 内连续”的定义、并坚持要求全 Shape 空间存在一个
全局连续的权威函数时，才会出现真正未解决的规范冲突。那时最小 ticket 应只问：
“权威 Capability Surface 的拓扑定义域是否允许 regime seam 成为洞，以及点查询、区间查询
和 provisional baseline 在洞上的返回语义是什么？”；它应使用 GitHub 原生 blocker 阻塞
最终规范 ADR，而不是反向阻塞或重开本研究票。当前没有触发该条件。
