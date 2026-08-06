# C009：逐操作降低为硬件无关 CostIR

- **开始/结束：** 2026-08-06T18:28:34+08:00 / 2026-08-06T18:32:33+08:00
- **阶段：** M3 / COST RULES
- **动作类型：** IMPLEMENT
- **关联验收/未知量：** AC-03、AC-05、AC-10、D-03

## 预注册

- **本轮 micro-goal：** 通过公开 `CostLowerer.lower(request)` seam 为 8 类语义操作生成公式、FLOPs、逻辑 operand/result bytes、实际物化 read/write bytes、参数/Buffer/激活分类、依赖与 provenance。
- **当前假设：** 硬件无关 CostIR 可以描述算法需求与最低逻辑数据规模；cache、kernel、fusion、带宽和 duration 必须留给 Hardware Backend。
- **已有证据：** M2 的 54-op concrete-shape SemanticIR；ADR-0009、0013、0029。
- **证据等级：** SemanticIR E2；Cost 公式 E0。
- **唯一主要变量：** 新增 CostIR、可注册 CostRule 和逐操作 Lowerer；本轮不做硬件 duration 或全模型手算总表。
- **预期观察：** MatMul/RMSNorm/Softmax/causal Add 等 literal 算例精确一致；View/Transpose logical bytes 可见但 materialized bytes=0；依赖来自 Value producer；每个 Cost op 有公式 rule 与 derivation。
- **判别规则：** 公开 seam 先 RED；独立 hard-coded literal 测试全部 GREEN 才聚合全模型总量。
- **成本与风险：** 预计 20–30 分钟；纯 CPU 确定性测试；最大风险是把逻辑 tensor bytes、缓存流量和物化流量混为一个口径。
- **停止与回滚：** 不引入设备名称、时延或实测系数；不得从实现公式本身计算测试 expected。

## 执行

- **脱敏命令：** `commands.md`
- **配置/环境差异：** 无。
- **代码差异：** 新增 CostIR hierarchy/metrics/formula/result、CostRule Protocol/Registry、8 类默认规则、统一字节分类、依赖映射与 provenance。
- **日志/指标：** 首次因 CostLowerer 不存在 RED；实现后 3 passed。

## 结果

- **观察事实：** Q projection 268,435,456 FLOPs；QK MatMul 268,435,456；RMSNorm 1,049,600；Softmax 10,477,568；causal Add 2,097,152，均与独立 literal 一致。View logical read/write 各 1,048,576 bytes，但 materialized read/write 均为 0。
- **推断：** “逻辑 tensor 数据规模”与“本操作物化流量”双口径可避免 alias 假写入，同时仍支持 UI 展示 Shape/访问规模。CostRule 只决定 FLOPs，通用 Lowerer 统一字节口径可减少插件漂移。
- **证据等级变化：** 原子 Cost 规则 E0→E2。
- **信息增量：** 全模型聚合初值已产生，待独立总表验证。

## 结论

- **验收/交付更新：** AC-05/D-03 继续 IN_PROGRESS；逐 op 公式完成。
- **预算变化：** 无硬件实验。
- **下一 micro-goal：** C010 用独立 hard-coded 总表验证单层/两层聚合、唯一 state bytes、CPU/MPS 一致性，并把 CostIR 加入 CLI。
- **是否需决策：** 无。
