# SimAI 需求洞察报告

## 分析元数据

| 字段 | 内容 |
|---|---|
| 来源特性地图 | [map.md](map.md) |
| 仓库 URL | https://github.com/aliyun/SimAI |
| 固定提交 SHA | f5efb5a93ea9be7db25a8843f9f7ff54044f6062 |
| 分析日期 | 2026-08-10 |
| 对照仓/资料 | [GroundUpScale](https://github.com/Kirrito-k423/GroundUpScale)、[Echo](https://github.com/NetX-lab/Echo)、[SimAI NSDI'25](https://www.usenix.org/conference/nsdi25/presentation/wang-xizheng-simai) |

## 筛选结论

SimAI 最值得复用的是固定 workload 格式、Analytical 快速通信层、MockNCCL/ns-3 网络后端和真实训练论文基线；不值得复制的是碎片化 fork、缺少版本化 release/CI 的集成方式，以及未充分校准的推理容量近似。对 GroundUpScale，优先做 adapter 和外部 backend，而不是重新实现 packet simulator。

## 评估方法

采用 U（稀缺性）、E（预期收益）、B（适用广度）、C（证据强度）、M（迁移成本）、R（风险）六个 1–5 分维度：

```text
Score = 20 × [0.15U + 0.25E + 0.25B + 0.15C + 0.10(6-M) + 0.10(6-R)]
```

进入推广清单需满足 U、B、C 均不低于 3；P0 ≥ 80，P1 为 65–79，其余进入 P2/验证池。

## 推广候选清单

| 特性名称 | 稀缺性依据 | 有效性证据与收益 | 广泛适用性 | 目标仓/场景 | 前置条件 | 迁移工作量（代码量） | 风险/副作用 | 评分 | 优先级 | 证据链接 |
|---|---|---|---|---|---|---|---|---|---|---|
| SimAI workload importer（不迁移执行器） | SimAI 已形成训练/推理逐层计算与 collective workload；GroundUpScale 当前只有一个 fixed prefill YAML | 可立即扩展真实模型/并行 case；保留 GroundUpScale provenance 后能建立可审计对照；收益需用一个 case 验证 | GPT、LLaMA、DeepSeek、Qwen 与自定义 workload | GroundUpScale Model/Workload/Semantic IR | 固定 AICB/SimAI schema；定义 unsupported 字段与 identity mapping | M；约 500–1500 LoC，8–18 文件；1–2 人×2–4 人周 | workload 语义不是完整模型语义；版本漂移与丢字段风险 | U3/E5/B5/C4/M3/R2；86 | P0 | [固定 workload 示例](https://github.com/aliyun/SimAI/blob/f5efb5a93ea9be7db25a8843f9f7ff54044f6062/example/workload_analytical.txt#L1-L18) |
| Analytical / ns-3 external network backend adapter | SimAI 同时提供快速 busbw 与 packet/RDMA 两档；GroundUpScale 无 collective/network runtime | NSDI'25 训练通信偏差有 A100/H100 实证；可避免重复 4000–12000+ LoC 网络工程 | 网络/拓扑/collective 设计与训练 DSE | GroundUpScale Communication Demand / Execution adapter | 固定 SimAI binary/config，定义 completion/error/evidence contract | L；adapter 约 800–2500 LoC，12–28 文件；2 人×4–8 人周；不含重写 ns-3 | 子进程/格式耦合；SimAI 无 release/tag/CI；packet 模式慢 | U4/E5/B4/C5/M4/R3；84 | P0 | [固定 ns-3 入口](https://github.com/aliyun/SimAI/blob/f5efb5a93ea9be7db25a8843f9f7ff54044f6062/astra-sim-alibabacloud/astra-sim/network_frontend/ns3/AstraSimNetwork.cc#L260-L334) |

## 验证池

| 特性名称 | 当前缺口 | 需要的实验/调研 | 晋级条件 | 证据链接 |
|---|---|---|---|---|
| 推理 PD/显存规划复用 | KV cache request 长度硬编码为 1、TP communication 置 0，推理 E2E 精度未公开校准 | 用真实 vLLM/SGLang request trace、P/D 配置和 HBM observation 建独立 holdout | memory、TTFT、TBT、E2E 在声明域内均有误差与退化状态，且不再以经验回退冒充通过 | [已知近似代码](https://github.com/aliyun/SimAI/blob/f5efb5a93ea9be7db25a8843f9f7ff54044f6062/vidur-alibabacloud/vidur/scheduler/utils/memory_planner.py#L25-L123) |
| Physical RDMA 模式 | README 标为 Beta/internal testing，缺 CI/兼容矩阵和公开性能复验 | 固定 RDMA NIC/driver/topology 复跑流量、正确性与网络 counter 对照 | 至少两个 topology 与一个 collective family 可稳定复现，并有不影响真实网络的安全门 | [固定 physical 入口](https://github.com/aliyun/SimAI/blob/f5efb5a93ea9be7db25a8843f9f7ff54044f6062/astra-sim-alibabacloud/astra-sim/network_frontend/phynet/SimAiMain.cc#L112-L173) |

## 试点与验收计划

| 特性名称 | 试点目标 | 改动范围 | 对照基线 | 指标与验收阈值 | 退出条件 | 依赖/负责人 | 阶段 |
|---|---|---|---|---|---|---|---|
| GPT-3 13B training workload adapter | 导入固定 workload，映射 compute、collective、parallel metadata 与 provenance | parser、schema adapter、golden fixture、Run Bundle artifact | SimAI fixed workload + GroundUpScale current fixed prefill | 每条 op/collective 可回溯；未映射字段显式 unknown；round-trip 不静默丢失；同输入 digest 稳定 | 4 周仍需重写 SimAI execution engine，或无法保留 rank/group identity | 1–2 名工程师；SimAI fixed SHA | 第 1 阶段 |
| External analytical backend | 用 SimAI binary 计算 communication duration，同时保留 GroundUpScale evidence semantics | subprocess contract、config lock、timeout/error mapping、comparison | SimAI analytical CLI 原生输出 | 同 workload 输出一致；失败不回退为 0；版本/config/digest 完整；性能阈值试点前测定 | 无法固定依赖版本，或输出缺 rank/collective identity 不能审计 | 1–2 名工程师；可复现 build/container | 第 2 阶段 |

## 分阶段行动计划

1. **证据补齐**：固定 SimAI 与所有子模块 SHA，保存一个可公开重放的 workload/output fixture；不执行 README 的 `submodule update --remote`。
2. **最小试点**：只实现 workload importer；证明 GroundUpScale 能为 SimAI case 增加 provenance/explanation，而不是复制模拟器。
3. **跨模型/后端验证**：在 GPT-3 13B 与一个 DeepSeek/Qwen inference case 上分别验证 training 与 inference 边界，推理结果必须标出经验降级。
4. **规模化推广**：若 adapter 稳定，再接 Analytical；ns-3 仅在确有网络研究需求时作为外部 backend，不 vendoring 大量 fork。

## 风险与待验证项

- SimAI 广度很强但版本工程弱；无 tag/release/CI，子模块动态更新会破坏复现。
- 训练论文精度证据不能覆盖后加的 inference/PD/memory 路径。
- workload support 容易被误解为真实模型执行支持；adapter 文档必须保留该边界。
- GroundUpScale 若直接移植 ns-3/MockNCCL，会立即承担数千至上万 LoC、复杂许可证/子模块和硬件验证成本，应优先外部调用。
