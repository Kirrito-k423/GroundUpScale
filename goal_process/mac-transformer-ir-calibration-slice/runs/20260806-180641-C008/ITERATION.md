# C008：递归生成层次化 SemanticIR

- **开始/结束：** 2026-08-06T18:06:41+08:00 / 2026-08-06T18:24:08+08:00
- **阶段：** M2 / SEMANTIC IR
- **动作类型：** IMPLEMENT
- **关联验收/未知量：** AC-03、AC-04、AC-09、AC-10、H-04、D-03

## 预注册

- **本轮 micro-goal：** 只通过 `SemanticCompiler.compile(request)` 将 WorkloadIR skeleton、ModelCall entrypoint、AnalysisCase shape 与逻辑策略组合为层次 Region 程序，显式生成 Typed Value、State Artifact/Effect、consumer linkage、provenance、验证结果和 fingerprint。
- **当前假设：** 递归 Region + 显式 Value/Effect 可以完整表达两层 Transformer，不需要平铺 DAG；CPU/MPS placement 是物理事实，不应改变 SemanticIR 或 fingerprint。
- **已有证据：** C007 结构 IR 与 entrypoint steps；ADR-0008、0027、0029；`docs/architecture/semantic-compilation.md`。
- **证据等级：** 架构 E1，真实结构输入 E2，无语义实现。
- **唯一主要变量：** 新增 Semantic Compiler 与语义 IR；本轮不计算 FLOPs/bytes、不选择 kernel/硬件时延。
- **预期观察：** 52 个 primitive 全部成为 concrete-shape 语义 op；Workload/Model 嵌套层次保留；每个参数/Artifact 交互为显式 effect；所有值有 producer/consumer；CPU/MPS 计划编译结果相同；错误 Shape/约束被拒绝。
- **判别规则：** 公共 seam 先 RED；层次、类型、effect、provenance、determinism、placement independence 与 CLI artifact 测试全部 GREEN 后退出 M2。
- **成本与风险：** 预计 25–40 分钟；无硬件实验；主要风险是 residual/多分支 value binding 与 region 间 consumer linkage。
- **停止与回滚：** 不把 placement、latency、device、schedule 塞进 SemanticIR；不得由调用方编排内部 pass。

## 执行

- **脱敏命令：** `commands.md`
- **配置/环境差异：** 无。
- **代码差异：** 新增 SemanticIR 不可变类型、SemanticCompiler、逻辑 deployment 投影、版本化 State Effect、编译 CLI 与五类 JSON 产物；causal mask 作为每层显式 buffer read + Add。
- **日志/指标：** 公开 seam 首次因接口不存在 RED；中间 52-op 编译 GREEN 后审计发现非 causal 缺口，测试改为 RED；补充 mask 后最终全量 15 passed。CPU/MPS semantic JSON SHA-256 均为 `a48c0ebc15f7f633c6cf7734321e5317a614b761baa2a7f321073d1638b7c0be`。

## 结果

- **观察事实：** 最终 ModelIR 61 modules（7 composite、54 primitive）；SemanticIR 10 regions、54 operations、75 values、22 state artifacts、22 versioned effects；ProvenanceGraph 246 records。8 类操作全部出现，View/Transpose 标记 `materialization=zero` 且 value 保留 alias_of。CPU/MPS 两计划 compilation fingerprint 相同：`44e2dbb4d59c86ff913f43b5044b3613b95a498cea362fb1fa64fa76903033d6`。
- **推断：** Region 能覆盖 ModelCall→Transformer→Layer→Attention/MLP 嵌套，不需要把整体压平为 DAG；Value 与 Effect 仍提供跨 Region 数据依赖。Deployment placement 被正确隔离在 SemanticIR 之外。
- **证据等级变化：** 多层语义编译 E1→E2；H-04 的“冻结操作可完整表达 causal 两层模型”子命题 E1→E2，runtime E2E 待 M4。
- **信息增量：** 建立了后续 Cost Lowerer 唯一输入，且可从每个语义 op 回溯 ModelIR/Spec derivation。

## 结论

- **验收/交付更新：** M2 完成；AC-02/AC-03 继续 IN_PROGRESS（运行与 CostIR 未完成）；D-02 完成，D-03 WIP。
- **预算变化：** 无硬件实验。
- **下一 micro-goal：** M3/C009：以独立 literal 公式测试驱动 SemanticIR→硬件无关 CostIR，先实现逐操作 FLOPs/逻辑 bytes/别名物化规则。
- **是否需决策：** 无。
