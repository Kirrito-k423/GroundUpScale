# Frontier Anchor 证据准入、候选组合与真实性能劣化判据

> 研究问题：怎样证明一个精确 Shape 的测量有资格约束
> `Validated Achievable Frontier`，而不是把错误结果、当前实现低效、测量噪声或
> corner bug 固化成硬件能力？

- 研究日期：2026-08-07
- 适用范围：CPU、GPU、NPU 的单算子与通信 benchmark；模块和 E2E 的
  `Schedule Achievable Frontier` 需要另行定义调度证据协议。
- 结论性质：来源支持的事实与本项目的规范建议分开陈述；文中的状态机、默认门槛
  和 YAML 是 GroundUpScale 的设计建议，不冒充任何厂商标准。

## 一句话结论

Frontier Anchor 不能来自“全局 P80”或一次最快测量；它必须来自精确 Shape、完整
有效域和正确性门禁下的 `best-of-correct` 候选搜索，并由独立留出运行复验。只有
多实现或可审计的候选池在同一 Shape 上共同复现局部下降，且邻域锚点与硬件 cohort
稳定时，才可判为 `frontier_shift`；否则保留为 `insufficient_evidence`。

## 一手资料支持的事实

### 正确性必须先于性能结论

- SPEC CPU 2026 的工具先验证输出，再计算性能指标；它把结果定义为在已披露条件下
  的可复现实测，而非硬件绝对能力。每个 benchmark 重复三次时取中位数，或重复
  两次时取较慢值。[SPEC CPU 2026 Run Rules](https://ftp.spec.org/cpu2026/docs/runrules.html)
- MLPerf Inference 要求每个性能结果都有对应的 accuracy validation run，并要求
  accuracy 与 performance mode 使用相同代码。这支持“正确性运行可以不在计时区间，
  但必须验证同一实现制品”的做法。
  [MLPerf Inference Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- NVIDIA CUTLASS Profiler 把 verification 与 profiling 放在同一工具中，支持固定随机
  种子、`epsilon`、近零阈值及 host/device/library verification provider。
  [CUTLASS Profiler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html)
- oneDNN 的 `benchdnn` 定位就是 primitive 的 correctness verification 与
  performance benchmarking；oneDNN 的 CI/NIGHTLY 测试默认保留正确性验证，只有显式
  `NO_CORR` 才移除。
  [benchdnn README](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/README.md)、
  [oneDNN build options](https://uxlfoundation.github.io/oneDNN/dev_guide_build_options.html)
- AMD 的 `hipblaslt-bench` 同时测量性能并验证正确性；NVIDIA `nccl-tests` 也明确同时
  检查 collective 的性能与正确性。
  [hipBLASLt clients](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/clients.html)、
  [NCCL Tests](https://github.com/NVIDIA/nccl-tests)

### 不存在跨算子、跨 dtype、跨后端通用的数值容差

PyTorch 官方数值精度说明指出：浮点运算不满足结合律，CPU/GPU、不同平台、不同版本
和不同 reduction 顺序不保证 bitwise identical；TF32、FP16/BF16 reduced-precision
reduction、AMD MI200 的 denormal 行为都会改变误差性质。因此 correctness oracle
必须属于“算子语义 + dtype/accumulation mode + Shape + 数值模式”的有效域，不能只有
一个仓库全局 `rtol`。

来源：
[PyTorch numerical accuracy](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/numerical_accuracy.md)、
[torch.testing comparison implementation](https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py)。

### 异步设备必须先定义计时边界，再选择同步方法

- CUDA kernel launch 是异步的。host wall-clock 测量必须在边界同步；测 kernel/device
  执行则应在同一 stream 放置事件并等待 stop event。多 stream 情况还要避免把无关工作
  混入事件区间。
  [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- HIP 同样提供 `hipEventRecord`、`hipEventSynchronize`、`hipEventElapsedTime` 和
  `hipDeviceSynchronize`；事件 flag 本身也可能影响 fence 与计时语义，因此 flag 是
  有效域的一部分。
  [HIP asynchronous execution](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html)、
  [HIP event management](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/event_management.html)
- Ascend PyTorch 的 `torch_npu.npu.synchronize()` 等待一个 NPU 上所有 stream 的 kernel
  完成；CANN 的 `msprof op` 提供单算子板上 profiling。两者分别适合 host-visible
  边界同步和设备执行诊断，不能混成一个未声明的“算子耗时”。
  [torch_npu synchronize source](https://gitee.com/ascend/pytorch/blob/master/torch_npu/npu/utils.py)、
  [Ascend C on-board profiling](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_0008.html)
- `torch.utils.benchmark.Timer` 会 warmup、固定线程池并在需要时同步异步 accelerator，
  其设计也明确要求 replicates 并使用中位数降低噪声影响。
  [PyTorch Timer source](https://github.com/pytorch/pytorch/blob/main/torch/utils/benchmark/utils/timer.py)

### 候选搜索必须是精确 Shape、精确环境的搜索

- cuDNN 可枚举、筛选和 auto-tune 多个 engine configuration；heuristic top-1 只是预测，
  `cudnnFindPlan` 的用途正是逐个计时选出具体 problem/device 的最好 plan。
  [cuDNN Graph API](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/graph-api.html)
- hipBLASLt tuning utility 能对一个精确 `M/N/K` 使用 `AlgoMethod: all` 搜索 kernel pool，
  并保留 raw benchmark 结果。它还明确说明 best solution index 不能跨 library release
  或 device architecture 复用。
  [hipBLASLt tuning utility](https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/how-to/how-to-use-hipblaslt-tuning-utility.html)、
  [hipBLASLt offline tuning](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-6.4.2/how-to-use-hipblaslt-offline-tuning.html)
- Ascend AOE/OPAT 在实际运行环境中生成、编译、验证多个 tiling policy，最后固化最优
  policy；官方还特别说明应在 subgraph 切分后用最终 Shape 做 operator tuning。
  [Ascend AOE overview](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/aoe/aoerc_16_0002.html)

这些工具只能证明“已枚举候选中的最好者”，不能证明所有未来算法的全局最优。因此
GroundUpScale 的 Frontier 应解释为 **best validated observed frontier**，并持续报告
候选覆盖等级，不能省略“validated observed”。

### warmup、缓存和环境状态会改变结论

- Google Benchmark 支持独立 warmup time、重复运行、随机交错和完整 JSON context；
  默认报告每次 repetition，并额外计算 mean、median、standard deviation、CV。
  [Google Benchmark User Guide](https://google.github.io/benchmark/user_guide.html)
- NVIDIA 的 GEMM 测量方法要求 warmup 与 profiling loop 分离，buffer 在计时前分配，
  避免迭代间额外工作，并按目标 cache 语义轮换 buffer；还要求检查频率、功耗、温度、
  L2 hit rate 和 launch gap。
  [CUTLASS GEMM measurement methodology](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html)
- Kalibera 与 Jones 的研究指出，iteration、进程/VM invocation、编译等多层随机性不能
  被当成同一层独立样本；实验设计应测量各层 variance，再把预算放在影响最大的层次。
  [Rigorous Benchmarking in Reasonable Time](https://kar.kent.ac.uk/33611/)

因此，单进程内做很多 iteration 不能替代独立进程/会话的复验；反之，删除所有慢样本
也不能证明实现稳定。

## 建议的证据准入协议

下面使用 `MUST`、`SHOULD`、`MAY` 表示建议中的强制、推荐和可选规则。

### 1. 固定证据身份和有效域

每条 observation MUST 绑定以下不可变 identity：

1. `semantic_op_digest`：算子数学语义、输入输出、状态副作用。
2. `exact_shape`：所有维度、batch、stride、broadcast、稀疏度和动态边界实例。
3. `numeric_contract`：输入/输出/累加 dtype，TF32、fast-math、reduced reduction、
   stochastic rounding、NaN/Inf/denormal 语义。
4. `layout_contract`：layout、stride、alignment、padding、alias。
5. `execution_contract`：线程/stream 数、并发、cache state、workspace、融合和 graph/eager
   mode。
6. `hardware_cohort_digest`：设备 ID/SKU、核心或分区配置、频率/功耗模式、NUMA/MIG、
   固件、驱动、runtime、compiler、算子库及其版本。
7. `candidate_digest`：provider、algorithm/engine/solution/tiling ID、binary hash、编译参数、
   workspace 和 dispatch mode。
8. `input_corpus_digest`：生成器、seed、值分布，以及 performance fill 与 correctness
   corpus 的区别。

任一字段缺失或隐式继承环境变量，observation MUST 留在 `QUARANTINED`。

### 2. correctness oracle 门禁

正确性检查 MUST 在精确 Shape 和同一 `candidate_digest` 上完成，并在性能计时区间外
运行。最少包含：

- 整数、逻辑、索引和协议型操作：默认 exact/bitwise oracle；若语义允许其他结果，
  必须显式列出 equivalence relation。
- 浮点逐元素与矩阵运算：高精度或独立 provider 的 reference result，加上按算子、
  dtype、reduction length/conditioning 定义的 `atol + rtol * |reference|`；禁止只按 dtype
  复制一个全局容差。
- reduction、softmax、normalization、线性代数：除逐元素误差外，还要检查语义不变量，
  如概率和、单调性、范数、residual、有限性或 condition-aware bound。
- 非确定或随机操作：固定 seed 仅在语义允许时使用；否则需要 distribution/property
  oracle 及其样本量和置信规则。
- communication：对所有 rank 检查 expected payload、root、reduction op、dtype、count
  和失败 rank；只验证 rank 0 不合格。
- 输入 corpus SHOULD 同时包含典型值、零/极值、对齐边界和多个 seed。若性能会随输入
  值改变，值分布必须进入有效域而非作为“无关测试数据”。

所有候选先过 correctness，之后才进入性能排序。任何错误但很快的候选 MUST 被拒绝，
不能参与 best-of-correct。

### 3. timing validity 门禁

每个 Benchmark Case MUST 选择且只选择一种主指标语义：

| `timing_scope` | 包含内容 | 推荐时钟 |
|---|---|---|
| `device_execution` | 已驻留输入上的 device kernel/collective | CUDA/HIP/NPU device event 或官方 profiler device time |
| `host_dispatch` | host API/launch/queue submission | monotonic host clock；不得声称为 kernel time |
| `host_visible_completion` | host 调用至结果可见，包含必要同步 | 同步前后的 monotonic host clock |
| `transfer` | 明确源/目的 memory domain 的 H2D/D2H/P2P/NUMA copy | 对应 stream event 或同步 host clock |

通用门禁：

- 分配、输入初始化、编译、autotune、正确性比较 MUST 在计时区间外，除非这些工作本来
  就是被测语义的一部分。
- CPU MUST 记录 real-time/cpu-time 选择、线程池、亲和、NUMA、核心类型、频率策略、
  后台负载和 memory working set。
- GPU/NPU 的异步工作 MUST 通过同 stream 事件或明确的前后同步闭合边界；事件、stream、
  graph mode 和 fence flag MUST 记录。
- profiler/deep trace 结果只能作为诊断证据。除非已量化并接受其开销，否则不得成为
  Frontier 性能真值。
- cache-hot、cache-cold、resident-weight、rotating-buffer 是不同 execution contract，
  不得混合样本。

### 4. warmup 与重复性门禁

固定 warmup 次数 MAY 作为下限，但不能单独证明稳态。建议每个 policy 声明：

- `min_warmup_iterations` 和 `max_warmup_iterations`；
- rolling window 的 `median_drift_limit`、`dispersion_limit`、连续通过窗口数；
- GPU/NPU 的 clock/power/temperature 稳定区间；
- JIT、lazy init、autotune 和内存池初始化必须在 warmup 完成；
- 达到上限仍不收敛时，结果是 `insufficient_evidence`，不能把最后一段强制当稳态。

重复测量 SHOULD 至少包含三个独立进程/会话；这是 GroundUpScale 的建议基线，与
SPEC 的三次运行取中位数方向一致，但并非声称“三次”对所有微基准统计充分。每个会话
内部还必须满足 Benchmark Case 声明的 `min_samples` 和 `min_timed_duration`。

统计建议：

- 会话内报告 median、MAD、p10/p90 和全部 raw samples；
- 以“会话”为独立层做 hierarchical bootstrap 95% CI，不能把同一进程内的上千次
  iteration 当成上千个独立会话；
- 候选和 Shape 以 randomized/interleaved block 执行，降低温度、DVFS 和后台负载漂移
  对固定顺序的偏置；
- `repeatability_gate` 使用预先注册的 relative CI width、session-to-session spread 和
  环境 guard 阈值。阈值属于 Benchmark Case/Error Budget，不在协议里伪造一个跨硬件
  通用百分比。

### 5. 离群点处理

协议 MUST 保存所有原始样本。默认规则是“标记，不删除”：

- 只有命中预先声明、可由外部证据验证的 invalidation reason，例如 device reset、
  thermal/power guard 越界、同步失败、correctness failure、其他 workload 侵入，样本
  才能从主统计量排除。
- “比其他样本慢”本身不是排除理由；慢尾可能是真实调度行为。
- 被排除样本及理由仍保留在 evidence bundle；排除比例超过 policy 上限时，整个会话
  失败，而不是继续清洗到稳定。
- 搜索阶段的 leading warmup 样本可不进入统计，但其原始值仍保留，并与测量样本分栏。

### 6. 候选实现组合与 best-of-correct

候选枚举 MUST 记录覆盖等级：

| 等级 | 含义 | 能否单独支持 `frontier_shift` |
|---|---|---|
| `C0_SINGLE` | 只有一个实现，或多个名字落到同一 kernel/binary | 否 |
| `C1_HEURISTIC` | 一个 provider 返回 top-N heuristic candidates | 否，除非另有历史 exact anchor |
| `C2_MULTI_FAMILY` | 至少两个 algorithmically independent family/provider，且都正确 | 可以，仍需邻域和复验 |
| `C3_ENUMERATED_POOL` | provider 对该精确问题枚举 all-reported solution pool，并记录失败/不支持项 | 可以；只代表该 pool，不代表未来算法全局最优 |

“独立”必须由 algorithm/engine/solution/tiling/binary fingerprint 判断，两个 Python wrapper
调用同一个底层 kernel 不算两个 family。

候选选择使用两阶段流程，防止 search noise 造成 winner's curse：

1. **Search**：对所有 correct + timing-valid + repeatable candidates 交错测量，按声明指标
   选择最小 median latency/最大 effective rate。
2. **Holdout confirmation**：在新的进程/会话和未参与选优的样本上只复验赢家及至少一个
   runner-up。赢家未落入预注册 confirmation budget 时，回到 `PROVISIONAL` 或
   `insufficient_evidence`。
3. Anchor 的 point estimate 和 CI 来自 holdout，不来自 search 中最幸运的一次结果。

资源 scalar/vector/cube、memory、communication microbenchmark 只产生
`Resource Physical Floor` 的 capability evidence；它们不能替代 operator candidate，
也不能直接成为 `Operator Achievable Frontier` Anchor。

## Frontier Anchor 晋级与状态机

建议不要用一个状态同时表达“测量是否合法”和“是否正约束 Frontier”。机器上保留两条
正交状态轴。

### Observation validity

```text
COLLECTED
   ↓ provenance/identity complete
QUARANTINED
   ├─ correctness/timing/repeatability fail ─→ REJECTED
   └─ all gates pass ───────────────────────→ QUALIFIED
QUALIFIED
   ├─ cohort/policy/version needs recheck ─→ STALE
   └─ later proof of wrong result/provenance ─→ REVOKED
STALE
   ├─ revalidation passes ─────────────────→ QUALIFIED
   ├─ revalidation deadline exceeded ──────→ EXPIRED
   └─ invalidating evidence ───────────────→ REVOKED
```

- `REJECTED` 表示这次 observation 不合格；修复后必须产生新的 observation ID，不能修改
  原始记录后“复活”。
- `REVOKED` 表示后来发现正确性或 provenance 根本错误；所有引用它的 surface 必须重算。
- `EXPIRED` 只是当前不可用于新结论，历史证据仍保留。

### Frontier role

```text
NONE
  ↓ best-of-correct search winner
PROVISIONAL
  ├─ independent holdout + coverage policy pass ─→ ACTIVE
  └─ confirmation fail ─────────────────────────→ NONE
ACTIVE
  ├─ faster ACTIVE anchor at same coordinate ───→ SUPERSEDED
  ├─ validity becomes STALE ────────────────────→ STALE_ROLE
  └─ validity becomes REVOKED ──────────────────→ REVOKED_ROLE
STALE_ROLE
  ├─ revalidated and still selected ────────────→ ACTIVE
  └─ validity expires/revokes ──────────────────→ EXPIRED_ROLE/REVOKED_ROLE
```

只有 `validity=QUALIFIED && frontier_role=ACTIVE` 的证据能作为 Capability Surface 的
knot。`SUPERSEDED` 的较慢 Anchor 仍可作为 supporting evidence，但不再约束上前沿。

以下事件触发 stale/revalidation：

- hardware cohort digest 改变；
- driver/runtime/compiler/operator library、firmware、kernel binary 或 numerical mode 改变；
- CI drift 超过 Error Budget；
- 定期 revalidation SLA 到期；
- 相邻 Anchor 或插值模型变化使当前点成为高影响异常。

单纯“证据很老”不应自动证明它错误；但过了项目的 revalidation deadline 而无法复跑，
它不能继续支撑新的强结论。

## 性能差时的判定矩阵

### `implementation_headroom`

同时满足：

1. 当前实现的 exact-Shape observation 正确且 timing-valid；
2. 至少一个相同 cohort/semantic/layout/numeric contract 的 correct candidate 在独立
   holdout 上显著更快；
3. 差异超过两者 uncertainty 与 instrumentation budget；
4. 当前实现不是仅在 E2E 集成后变慢。若单算子正常、集成后慢，应判
   `integration_overhead`。

### `frontier_shift`

这是“原 Capability Surface 在这个 Shape 邻域过高”的结论，而不是“当前实现慢”。
建议必须全部满足：

1. 目标点此前是插值/低证据点，而不是已有更快、仍有效的 exact ACTIVE Anchor；
2. `Shape Disambiguation Probe` 在精确 Shape 上运行，coverage 至少为
   `C2_MULTI_FAMILY` 或 `C3_ENUMERATED_POOL`；
3. 所有 eligible candidate 的 holdout 结果都显著低于旧 surface 的 uncertainty band；
4. 至少三个独立会话复现，且 clock/power/temperature/线程/stream 等 guard 合格；
5. 两侧相邻 Anchor 复验仍稳定，排除整机/软件 cohort 漂移；
6. 在目标 Shape 附近增加稠密 probe，区分对齐、workspace、cache-domain、kernel switch
   或 narrow regime；
7. 用新增 Anchor 在每个 connected `Validated Shape Regime` 内重拟合**连续**
   Capability Surface。若稠密 probe 已独立验证底层 dispatch 切换处的 seam 连续性，
   可用局部窄过渡或连续分段 seam 表达；若识别出新的 alignment、working-set、
   candidate-support 或 kernel regime 而 seam 尚未验证，则必须切分 retained cells、
   设置 rejection band，并让跨 regime 的权威查询返回 `unknown`。两种情况都不得回退到
   最近邻跳变，也不得为了全局连续而虚构跨未验证 regime 的能力值。

已有 exact ACTIVE Anchor 证明某性能曾经在该不可变 cohort 下可达，所以后来更慢的
样本不能直接把它“下调”：

- 原 candidate 仍可复现：当前实现有 headroom 或 regression；
- 原 candidate 因 library/device 版本变化不可用：这是新 cohort，建立新 surface；
- 原 Anchor 后来被证明算错或 provenance 错：`REVOKED + surface retraction`，不是自然
  `frontier_shift`；
- 原环境无法复现且没有直接失效证据：`insufficient_evidence`。

### `insufficient_evidence`

命中任一项即不得强判真实性能劣化：

- 只有当前慢实现一个 candidate，或多个 candidate 实际共享同一 kernel；
- correctness oracle/tolerance 没有按算子和 numeric mode 注册；
- warmup 未收敛、session 间不稳定、排除样本比例过高；
- GPU/NPU 异步边界不闭合，或 profiler time 与 benchmark time 混用；
- hardware cohort、cache state、线程/stream、layout 或输入分布不一致；
- 查询属于外推、跨 Shape regime 或邻域 Anchor 不足；
- 差异没有超过综合 uncertainty；
- 旧 exact Anchor 无法复验，但也没有失效证据。

## 机器可执行的证据记录建议

以下 YAML 是协议骨架，数值阈值由 versioned Benchmark Case/Error Budget 提供；示例中的
值只用于展示字段，不是跨硬件标准。

```yaml
api_version: groundupscale.dev/v1alpha1
kind: FrontierEvidence
metadata:
  evidence_id: fe-sha256:...
  created_at: 2026-08-07T00:00:00Z

identity:
  semantic_op_digest: sha256:...
  exact_shape: {m: 201, n: 512, k: 512, batch: 1}
  numeric_contract:
    input_dtype: fp32
    accumulation_dtype: fp32
    output_dtype: fp32
    math_mode: ieee
  layout_contract:
    a: {layout: row_major, alignment_bytes: 64}
    b: {layout: row_major, alignment_bytes: 64}
  execution_contract:
    timing_scope: device_execution
    cache_state: rotating_cold
    threads: 4
    streams: 1
    graph_mode: eager
  hardware_cohort_digest: sha256:...
  candidate_digest: sha256:...
  input_corpus_digest: sha256:...

correctness:
  oracle_id: matmul-fp32-high-precision-v2
  reference_provider: cpu-fp64
  corpus_seeds: [11, 29, 47]
  comparison:
    kind: allclose_and_residual
    rtol: 1.0e-5
    atol: 1.0e-6
    nan_policy: reject
    inf_policy: reject
  passed: true
  report_digest: sha256:...

timing:
  clock: cuda_event
  synchronization: same_stream_stop_event
  setup_outside_interval: [allocate, initialize, compile, autotune, verify]
  warmup:
    min_iterations: 10
    max_iterations: 1000
    convergence_policy_id: warmup-gemm-v1
    converged: true
  sampling:
    independent_sessions: 3
    min_samples_per_session: 30
    min_timed_duration_ms: 500
    execution_order: randomized_interleaved_blocks
    raw_samples_digest: sha256:...
  environment_guards:
    policy_id: m4-cpu-or-h100-gpu-guard-v1
    passed: true
  excluded_samples:
    count: 0
    reasons: []

statistics:
  estimator: median_of_session_medians
  uncertainty: hierarchical_bootstrap_95ci
  latency_ns: {point: 12345, lower: 12001, upper: 12721}
  effective_rate: {value: 1.23e12, unit: FLOP/s}
  repeatability_policy_id: operator-anchor-repeatability-v1
  passed: true

candidate_search:
  coverage: C3_ENUMERATED_POOL
  enumerator: hipblaslt_all_reported
  considered: 143
  correct: 137
  timing_valid: 135
  candidates_manifest_digest: sha256:...
  selected_candidate_digest: sha256:...
  runner_up_candidate_digest: sha256:...

holdout_confirmation:
  search_samples_reused: false
  candidate_and_runner_up_rerun: true
  policy_id: anchor-holdout-v1
  passed: true
  report_digest: sha256:...

lifecycle:
  observation_validity: QUALIFIED
  frontier_role: ACTIVE
  coverage_strength: strong
  transition_history_digest: sha256:...
  supersedes: null

provenance:
  benchmark_case_digest: sha256:...
  run_bundle_ids: [run-search-..., run-holdout-...]
  source_revision: git:...
  schema_digest: sha256:...
```

机器晋级条件可直接表达为：

```text
promote_to_active :=
  identity.complete
  AND correctness.passed
  AND timing.environment_guards.passed
  AND timing.warmup.converged
  AND statistics.passed
  AND candidate_search.coverage in benchmark_case.allowed_coverage
  AND holdout_confirmation.passed
  AND observation_validity == QUALIFIED
```

## 风险与待后续票解决的问题

1. **候选池并不完备**：`C3_ENUMERATED_POOL` 也只是某个 provider 当前暴露的 pool。报告
   必须展示 coverage，禁止写成“硬件最大能力”。
2. **容差过松会奖励错误快路径**：每个 op family 需要独立 correctness policy，并用
   adversarial/condition-aware case 验证；这是未来 schema/registry 工作。
3. **选优偏差**：候选越多，search winner 越可能是噪声赢家。独立 holdout 是最低防线，
   大候选池还可能需要 multiple-comparison-aware budget。
4. **连续曲面会掩盖窄 cliff**：检测到异常必须加密邻域 probe；连续性不能成为跨未观测
   regime 平滑掉真实 corner 的借口。
5. **功耗/温度策略难以跨硬件统一**：协议统一字段和状态，不统一阈值。CPU、GPU、NPU
   backend 各自提供 guard collector 和 policy。
6. **微基准与集成态不同**：Operator Anchor 只说明独立候选可达；E2E 慢仍可能来自
   dispatch、layout conversion、cache eviction、contention 或 schedule，必须进入
   Schedule Frontier 诊断。
7. **低成本 CI 与强证据冲突**：高频 CI 可只做 stale/drift detection；ACTIVE Anchor 的
   新增、下调或 revoke 应由低频完整校准和 holdout job 完成。

## 调查结论

- 现状是：CPU/GPU/NPU 的官方工具都提供正确性、同步、warmup、重复运行或候选调优的
  部分能力，但没有一个跨硬件标准能直接给出 GroundUpScale 的 Frontier Anchor。
- 关键约束是：正确性、精确 Shape、计时边界、hardware cohort、候选覆盖、独立复验和
  原始样本必须同时成立；缺一项就不能用慢实现证明“硬件在这个 Shape 真实变慢”。
- 我之前不知道但现在知道的是：hipBLASLt 明确禁止跨 library release/device
  architecture 复用精确 Shape 的 best solution index，Ascend AOE 也强调必须在最终
  subgraph Shape 上调 operator。这直接支持每个 Hardware Validity Cohort 独立维护
  Frontier 的决定。
- 基于以上，我的判断是：采用双状态轴、best-of-correct + independent holdout，以及
  `C2/C3 + 邻域复验` 的 frontier-shift 门槛，可以阻止单次低效测量进入能力曲面；无法
  达到这些条件时，系统应诚实返回 `insufficient_evidence`。
