# C006：严格加载并解析完整 YAML Analysis Plan

- **开始/结束：** 2026-08-06T17:52:42+08:00 / 2026-08-06T17:57:53+08:00
- **阶段：** M2 / SPEC
- **动作类型：** IMPLEMENT
- **关联验收/未知量：** AC-01、AC-02、AC-03、D-02

## 预注册

- **本轮 micro-goal：** 通过一个公开 `SpecRepository.load_analysis_plan(path)` seam，严格校验并解析 AnalysisPlan 引用的 WorkloadSpec、AnalysisCase、DeploymentIntent、HardwareSpec、FabricGraph、BenchmarkCase，以及 Workload 中引用的 ModelSpec。
- **当前假设：** 8 类单文档 YAML + 显式版本化引用足以组成一次分析输入，不需要 Python Spec Builder 或跨文件 YAML anchor。
- **已有证据：** ADR-0019、0021、0022、0023；用户已确认 YAML 是唯一人类编写 Spec。
- **证据等级：** 架构 E1，无实现。
- **唯一主要变量：** 从无 Spec runtime 到一个只负责严格校验、引用解析、路径边界和内容摘要的 repository；本轮不生成 IR。
- **预期观察：** 最小完整 bundle 可加载；未知字段、版本不匹配、越界引用和错误 kind 被拒绝；返回所有有效文档及 SHA-256。
- **判别规则：** 公共 seam 契约测试先 RED；实现后正例与负例均 GREEN，才进入 ModelIR/WorkloadIR。
- **成本与风险：** 预计小于 15 分钟；无硬件实验；新增 PyYAML/Pydantic 锁定依赖。
- **停止与回滚：** 同一 Schema 错误最多无证据重试一次；不以 `extra=allow` 绕过未知字段。

## 执行

- **脱敏命令：** `commands.md`
- **配置/环境差异：** 新增 Pydantic 2.13.4、PyYAML 6.0.3 并锁定。
- **代码差异：** 新增 8 类冻结 Pydantic Schema、JSON Schema 导出入口、唯一键 YAML Loader、仓库边界检查、kind/version/hash 验证、AnalysisBundle 解析。
- **日志/指标：** 公开 seam 先因 `ModuleNotFoundError: groundupscale.specs` RED；实现后 C006 测试 5 passed，全量 6 passed。

## 结果

- **观察事实：** 一个 AnalysisPlan 可解析出 plan、workload、analysis、deployment、hardware、fabric、benchmark 与 workload 间接引用的 model，共 8 个内容寻址源。未知字段、重复键、版本/kind 错配和越界引用均被拒绝并带路径定位。
- **推断：** YAML-only + 显式版本引用的组合方式可执行，不需要 Python Spec Builder；IR Builder 可以只消费不可变的已验证对象。
- **证据等级变化：** Spec 组合架构 E1→E2（最小完整 bundle）。
- **信息增量：** C007 可在无需再次解析 YAML 的情况下确定性生成 ModelIR/WorkloadIR。

## 结论

- **验收/交付更新：** AC-02、D-02 进入 IN_PROGRESS；尚未有真实固定模型 YAML 与 IR。
- **预算变化：** 无硬件实验。
- **下一 micro-goal：** C007 构建真实两层 Transformer YAML bundle，并显式展开 ModelIR repeat、保持 WorkloadIR ModelCall 叶子。
- **是否需决策：** 无。
