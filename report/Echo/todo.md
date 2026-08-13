# Echo 需求洞察报告

## 分析元数据

| 字段 | 内容 |
|---|---|
| 来源特性地图 | [map.md](map.md) |
| 仓库 URL | https://github.com/NetX-lab/Echo |
| 固定提交 SHA | 15f0d12c417459d7a1b3b66abeaa28eca0b841a2 |
| 分析日期 | 2026-08-10 |
| 对照仓/资料 | [GroundUpScale](https://github.com/Kirrito-k423/GroundUpScale)、[SimAI](https://github.com/aliyun/SimAI)、[Echo 论文](https://arxiv.org/abs/2412.12487) |

## 筛选结论

Echo 可迁移价值集中在两个已公开局部模块：单卡 PyTorch tracer 与 Nsight→XGBoost slowdown pipeline。论文完整 CC/timeline simulator 不能迁移，因为固定开源仓没有实现。任何复用都必须先修复 identity/provenance、独立 train/holdout、CLI/CI 与敏感信息治理。

## 评估方法

采用 U（稀缺性）、E（预期收益）、B（适用广度）、C（证据强度）、M（迁移成本）、R（风险）六个 1–5 分维度：

```text
Score = 20 × [0.15U + 0.25E + 0.25B + 0.15C + 0.10(6-M) + 0.10(6-R)]
```

进入推广清单需满足 U、B、C 均不低于 3；P0 ≥ 80，P1 为 65–79，其余进入 P2/验证池。

## 推广候选清单

| 特性名称 | 稀缺性依据 | 有效性证据与收益 | 广泛适用性 | 目标仓/场景 | 前置条件 | 迁移工作量（代码量） | 风险/副作用 | 评分 | 优先级 | 证据链接 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch trace importer / adapter | GroundUpScale 从 YAML semantics 出发，缺少真实 framework trace；Echo tracer 已形成 graph/runtime JSON | 可快速接入 forward/backward/optimizer observation；公共实现证据完整度中等；需自行验证 identity 与 correctness | PyTorch local/HF/custom model；后续可作为 observation source | GroundUpScale Run Bundle/Alignment Map | 固定子模块 SHA；定义 stable identity 与 unsupported ops；去除硬编码 token/GPU/path | M；约 500–1600 LoC，8–20 文件；1–2 人×2–5 人周 | FX coverage、动态控制流与版本漂移；DeepSpeed/Megatron 不可假定支持 | U3/E4/B4/C3/M3/R3；72 | P1 | [固定 PyTorchTracer](https://github.com/NetX-lab/Echo-workload-tracer/blob/8fb57b6cdc8d5b5505ea4705e6cdf0684bb77424/tracer_core/pytorch_tracer.py#L117-L237) |
| Nsight overlap evidence pipeline | SimAI/GroundUpScale 当前没有同样直接的 NCU/NSYS→overlap ground truth→predictor 公共局部流水线 | 可帮助解释通信-计算共享资源干扰；公开链路存在，但默认 train/test 重用削弱指标可信度 | NVIDIA kernel/collective overlap；需逐架构重采 | GroundUpScale Operator Frontier/diagnostic evidence | 先重构独立 fit/holdout、hardware cohort、feature/version provenance | L；约 1000–3000 LoC，12–25 文件；2 人×4–8 人周 | Nsight 权限/开销、跨架构泛化、profiling lane 不能覆盖 baseline timing | U4/E4/B3/C3/M4/R4；65 | P1 | [固定采集流水线](https://github.com/NetX-lab/Echo-slowdown/blob/7d698a77c0318f2b706ed27d3f6221f8b1e0a349/run_all.sh) |

## 验证池

| 特性名称 | 当前缺口 | 需要的实验/调研 | 晋级条件 | 证据链接 |
|---|---|---|---|---|
| XGBoost slowdown predictor | 默认同一 merged 数据进入 train/test；公开特征比论文更窄 | 按 GPU cohort 与 kernel family 建 independent holdout，对比 no-overlap、简单比例与 XGBoost | holdout MAE/MAPE 明显优于简单 baseline，且跨 session/shape uncertainty 可校准 | [数据重用证据](https://github.com/NetX-lab/Echo-slowdown/blob/7d698a77c0318f2b706ed27d3f6221f8b1e0a349/run_all.sh#L43-L50) |
| 论文 CC Estimator/Timeline Composer | 公开仓无源码、入口、参数库与 E2E fixture | 等官方继续开源或仅依据论文重做一次最小 prototype 比较价值 | 获得可运行固定实现和许可，或有明确用户需求证明值得独立研发 | [逐步开源说明](https://github.com/NetX-lab/Echo/blob/15f0d12c417459d7a1b3b66abeaa28eca0b841a2/README.md#L54-L67) |

## 试点与验收计划

| 特性名称 | 试点目标 | 改动范围 | 对照基线 | 指标与验收阈值 | 退出条件 | 依赖/负责人 | 阶段 |
|---|---|---|---|---|---|---|---|
| GPT-2 PyTorch trace import | 将 Echo graph/runtime 作为 GroundUpScale observation，不替代自有 Semantic IR | 固定 example、trace parser、identity map、Run Bundle artifacts、alignment tests | Echo 原始 JSON + GroundUpScale fixed prefill reference | 全部 nodes 有来源与 stable identity；unsupported op 显式；同 trace import digest 稳定；不把 synthetic DDP 当真实 rank trace | 3 周仍无法稳定映射，或 trace 丢失模型层级导致诊断价值不足 | 1 名工程师；可先用固定 fixture，无 GPU | 第 1 阶段 |
| 2-GPU overlap holdout | 复验 slowdown pipeline，建立独立 fit/holdout 与 instrumentation overhead | NCU/NSYS scripts、cohort manifest、split policy、baseline lane | Echo 默认流程 + 简单 overlap ratio baseline | 独立 session holdout；无数据泄漏；正确性和 profiler overhead 完整；阈值在首次 clean baseline 后确定 | 无 NVIDIA/Nsight 条件，或 predictor 不优于简单 baseline | 1–2 名工程师；NVIDIA GPU、Nsight | 第 2 阶段 |

## 分阶段行动计划

1. **证据补齐**：确认公开 example 中的历史 token 已撤销；固定两个子模块 SHA；补最小可运行 fixture 与许可证清单。
2. **最小试点**：先做纯离线 trace importer，不依赖论文未开源的 E2E simulator。
3. **跨模型/后端验证**：有 NVIDIA 环境后再做独立 slowdown holdout；至少覆盖两个 kernel family 与两个 shape regime。
4. **规模化推广**：只有独立 holdout 证明 slowdown predictor 优于简单模型，才把它晋升为 GroundUpScale evidence backend；否则保留原始 Nsight evidence，不迁移 ML predictor。

## 风险与待验证项

- Echo 最关键的论文能力未开源，不能据论文图表设计直接承诺 adapter。
- 公共 DeepSpeed/Megatron 路径当前不闭合；复用范围必须限定 PyTorch tracer。
- 默认 slowdown 数据分割有泄漏风险，任何精度数字都需重做独立 holdout。
- CUDA/NCCL/Nsight 绑定与缺少 CI/release 会增加维护成本；敏感信息治理需先处理。
