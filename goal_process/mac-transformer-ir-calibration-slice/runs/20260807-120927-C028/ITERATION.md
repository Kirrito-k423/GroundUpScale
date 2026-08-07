# C028：构造并接入 M4 CPU 算法无关能力包络

- **开始/结束：** 2026-08-07T11:43:00+08:00 / 2026-08-07T12:15:46+08:00
- **阶段：** VERIFY
- **动作类型：** PROBE → EXPERIMENT → INTEGRATE
- **关联目标：** M4 CPU microbenchmark、能力测量、后端配置和样例重预测

## 预注册

- **本轮 micro-goal：** 用不少于 10 个 Shape 的多类探针产生 P80/P95 资源包络，并让两层 CPU 样例使用该包络输出算法无关硬件地板。
- **当前假设：** 跨算子探针的稳健能力包络可把“硬件可达边界”与“当前算子/算法耗时”分开。
- **已有证据：** ADR-0033、Apple M4 官方能力研究和 48-test 基线。
- **证据等级：** E1 → E3（真实重复测量、后端消费和独立样例对照均完成）。
- **唯一主要变量：** HardwareBackend 从厂商带宽单项地板切换为独立 Capability Profile 的 compute/memory P80 包络。
- **判别规则：** Suite/聚合/来源核验测试通过；Profile 被真实 AnalysisPlan 消费；样例同时输出 compute、memory、max floor 与 Observation。
- **成本与风险：** 本地 CPU 分钟级运行；后台竞争可能使结果只能标记 exploratory。
- **停止与回滚：** 不覆盖 HardwareSpec 厂商事实；Profile 不匹配或来源 SHA 失败时拒绝加载。

## 执行与结果

- 首版 scalar 探针使用 PyTorch 0-D dispatch，12 个 Shape 只有 8 个稳定，聚合器按规则拒绝。
- 将 scalar 单变量替换为可审计 ARM64 原生 FMADD 内核后，11 个 Shape 稳定；其他聚合逻辑不变。
- 第二轮完成 5 个探针、2 个资源包络；原始结果和配置见 C028 summary。
- 两层 CPU Run Bundle 完成，17 个 artifact digest 全部通过。
- 错误签名：首轮 `CapabilityAggregationError: scalar ... found 8`；已有根因和单变量修复，不再重复。

## 结论

- **信息增量：** M4 CPU 当前 exploratory P80 为 1.74845 TFLOP/s 和 126.833 GB/s；后端 E2E 地板 5.554 ms，最新版代码实测 92.814 ms。
- **验收/交付更新：** 本轮四项用户目标均已有直接产物和运行证据。
- **剩余风险：** 能力 Profile 和 E2E Run 的环境门禁未通过，不能作为 trusted CI baseline；需在安静环境重测才能晋升。
- **下一 micro-goal：** 完成文档、全量测试和 diff/敏感信息审计。
- **是否需决策：** 无。
