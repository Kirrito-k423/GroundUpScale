# GroundUpScale 需求洞察报告

## 分析元数据

| 字段 | 内容 |
|---|---|
| 来源特性地图 | [map.md](map.md) |
| 仓库 URL | https://github.com/Kirrito-k423/GroundUpScale |
| 固定提交 SHA | 4a6ac7aee0500d10c32ee7463daf24c8a20a5dd2 |
| 分析日期 | 2026-08-10 |
| 对照仓/资料 | [SimAI](https://github.com/aliyun/SimAI)、[Echo](https://github.com/NetX-lab/Echo)、[SimAI NSDI'25](https://www.usenix.org/conference/nsdi25/presentation/wang-xizheng-simai)、[Echo 论文](https://arxiv.org/abs/2412.12487) |

## 筛选结论

最值得保留和推广的不是“再造一个全栈 simulator”，而是可审计 IR、四轴诊断、Run Bundle 与 evidence qualification。它们在 SimAI/Echo 公开实现中没有同等强度的结构化契约，但 GroundUpScale 也尚未证明这些机制能在真实大模型、多卡网络场景中带来足够价值。推广策略应是把它们做成可接入现有 simulator/trace 的可信建模层。

## 评估方法

采用 U（稀缺性）、E（预期收益）、B（适用广度）、C（证据强度）、M（迁移成本）、R（风险）六个 1–5 分维度：

```text
Score = 20 × [0.15U + 0.25E + 0.25B + 0.15C + 0.10(6-M) + 0.10(6-R)]
```

进入推广清单需满足 U、B、C 均不低于 3；P0 ≥ 80，P1 为 65–79，其余进入 P2/验证池。

## 推广候选清单

| 特性名称 | 稀缺性依据 | 有效性证据与收益 | 广泛适用性 | 目标仓/场景 | 前置条件 | 迁移工作量（代码量） | 风险/副作用 | 评分 | 优先级 | 证据链接 |
|---|---|---|---|---|---|---|---|---|---|---|
| 四轴性能诊断与 fail-closed evidence | SimAI/Echo 重点输出模拟 step time，未见同等强度的 Physical Floor、Operator Frontier、Schedule Frontier、Observation 不覆盖契约 | 固定基线已有 schema/测试闭环；预期减少把下界、观测与预测混为一谈的错误，但尚无外部生产量化 | 可用于 simulator、benchmark、compiler、hardware lab 与 regression diagnosis | GroundUpScale 主线；后续作为 SimAI/Echo adapter 上层 | 选定一个真实 workload 与完整 evidence identity | L；约 1500–5000 LoC，15–35 文件；另需 adapters 与硬件验证 | 规范复杂度高；若没有真实用户会变成自我证明 | U4/E4/B4/C4/M4/R3；74 | P1 | [固定诊断规范](https://github.com/Kirrito-k423/GroundUpScale/blob/4a6ac7aee0500d10c32ee7463daf24c8a20a5dd2/docs/methods/cross-hardware-performance-gap-diagnosis.md) |
| Provenance IR + Immutable Run Bundle | 两个对照仓未见等价的多 IR fingerprint、artifact role/digest 和统一 explanation contract | 固定基线可拒绝覆盖和篡改，测试证据强；收益主要是复现、审计和对比效率，未量化吞吐 | 可承载本地模型、外部 workload、trace 与 simulator outputs | SimAI workload importer、Echo trace importer、内部硬件实验 | 设计稳定的外部 artifact adapter，不复制对方执行引擎 | M；约 600–1800 LoC，10–24 文件；另需 schema migration 与 compatibility tests | adapter 语义映射可能丢失信息；bundle 存储增加 | U4/E4/B5/C4/M3/R2；84 | P0 | [Run Bundle 实现](https://github.com/Kirrito-k423/GroundUpScale/blob/4a6ac7aee0500d10c32ee7463daf24c8a20a5dd2/src/groundupscale/run_bundle.py) |

## 验证池

| 特性名称 | 当前缺口 | 需要的实验/调研 | 晋级条件 | 证据链接 |
|---|---|---|---|---|
| Exact-shape Operator Frontier | 主要存在当前未提交工作区，固定基线只有诊断内核与 fixtures | 在真实 GPU/NPU 或 M4 上完成至少 3 个算子族、跨 session independent holdout 与 uncertainty calibration | 同一 workload 的 E2E 误差归因能比简单 profile/roofline 多解释至少一个可复现根因 | [Capability Surface 测试](https://github.com/Kirrito-k423/GroundUpScale/blob/4a6ac7aee0500d10c32ee7463daf24c8a20a5dd2/tests/test_capability_surface.py) |
| 分布式策略与网络主链 | 固定基线无 collective、route、contention、TP/PP/EP/CP 实现 | 先导入 SimAI workload，接一个 external network backend，禁止从头实现 packet simulator | 单一真实模型/并行 case 能重放、解释并与基线模拟/实测对齐 | [当前 strategy 拒绝逻辑](https://github.com/Kirrito-k423/GroundUpScale/blob/4a6ac7aee0500d10c32ee7463daf24c8a20a5dd2/src/groundupscale/compiler/semantic.py#L136-L148) |

## 试点与验收计划

| 特性名称 | 试点目标 | 改动范围 | 对照基线 | 指标与验收阈值 | 退出条件 | 依赖/负责人 | 阶段 |
|---|---|---|---|---|---|---|---|
| SimAI workload → GroundUpScale evidence adapter | 不复制 SimAI 执行器，导入一个 GPT-3/LLaMA training workload 并生成可追溯 Cost/Run Bundle | workload parser、identity/provenance mapping、unsupported-field diagnostics、golden test | SimAI 固定 workload 与输出 | 100% workload op/collective 有来源；未知字段显式保留；同输入 bundle digest 稳定；误差阈值在取得真实观测前不虚构 | 需要重写 SimAI 的 network/collective 核心，或 4 周仍无法闭合一个 case | 1 名工程师；SimAI 固定 SHA、一个公开结果样本 | 第 1 阶段 |
| Echo trace → GroundUpScale diagnosis adapter | 验证 trace-driven 来源能否被四轴诊断吸收 | trace schema、stable path mapping、observation alignment、slowdown evidence import | Echo PyTorch tracer + slowdown output | 至少一个 GPT-2 forward/backward case 可重放；所有预测与观测来源可钻取；不把 Echo 论文未开源部分伪装为支持 | 公开 tracer 无法产生稳定 identity，或 adapter 比保留自有 model semantics 更复杂 | 1 名工程师；NVIDIA GPU 为可选后续依赖 | 第 2 阶段 |

## 分阶段行动计划

1. **证据补齐**：冻结当前未提交工作区，修复微基准顺序/环境敏感测试；增加 LICENSE、release/tag 和明确的“已实现 vs intended”支持矩阵。
2. **最小试点**：四周内完成一个 SimAI workload adapter，不实现新网络模拟器；输出同一 case 的 GroundUpScale Run Bundle 和解释报告。
3. **跨模型/后端验证**：选一个真实 LLM workload，在 CPU/MPS 与一个 GPU/NPU 环境建立可比较证据；只扩最少算子/策略。
4. **规模化推广**：只有当外部 adapter 用户能从 provenance/diagnosis 中得到 SimAI/Echo 单独无法给出的决策价值时，才继续扩展模型、策略和后端覆盖。

## 风险与待验证项

- 最大风险不是竞品，而是规范复杂度继续快于可运行能力；`diagnostics.py` 已很大，而模型/策略/网络覆盖仍窄。
- 当前没有公开多节点误差、仿真速度或可扩展性 benchmark，不能用方法论优势替代市场/实验验证。
- 当前工作区存在真实微基准偶发失败；需把硬件噪声测试与 deterministic CI 明确分层。
- 若四周 adapter 试点仍需要复制 SimAI/Echo 的核心引擎，应停止“全栈自研”路线，收缩为可信建模与诊断层。
