# C007：确定性生成层次 ModelIR 与 WorkloadIR

- **开始/结束：** 2026-08-06T17:57:53+08:00 / 2026-08-06T18:06:41+08:00
- **阶段：** M2 / STRUCTURAL IR
- **动作类型：** IMPLEMENT
- **关联验收/未知量：** AC-02、AC-03、H-04、D-02、D-03

## 预注册

- **本轮 micro-goal：** 从真实固定 Shape 两层 Transformer YAML bundle 构建不可变 ModelIR 与 WorkloadIR；Model repeat 展开为两个有独立 Stable Path/Node ID、共享模板 Definition ID 的 layer，WorkloadIR 的 ModelCall 仍是可展开叶子。
- **当前假设：** Definition/Instance/Compilation 三重身份可以同时表达模板复用和具体层级；结构 IR 不需要硬件或时延事实。
- **已有证据：** C006 严格 Spec bundle；ADR-0003、0020、0026、0013。
- **证据等级：** 架构 E1，无 IR 实现。
- **唯一主要变量：** 新增纯结构 Builder 与不可变 IR；本轮不生成 SemanticIR、不做 Cost 公式。
- **预期观察：** 两次构建 canonical bytes/fingerprint 相同；layer_0/layer_1 Stable Path 与 Node ID 不同、Definition ID 对应相同模板；Workload ModelCall 不直接包含模型子节点；每个 IR 节点有 derivation。
- **判别规则：** 公开 Builder 测试先 RED；身份、重复展开、层级、Workload 叶子和 determinism 全部 GREEN 后进入 SemanticCompiler。
- **成本与风险：** 预计 15–25 分钟；无硬件实验；主要风险是重复模板身份和 entrypoint 展开混淆。
- **停止与回滚：** 不以数组下标作为唯一用户身份；不得把 ModelIR 嵌套复制进 WorkloadIR。

## 执行

- **脱敏命令：** `commands.md`
- **配置/环境差异：** 使用 C006 锁定环境。
- **代码差异：** 新增不可变公共 IR、canonical serializer、身份/derivation helper、ModelBuilder/WorkloadBuilder，以及完整 CPU/MPS YAML Analysis Plan。
- **日志/指标：** 公开 Builder 首次因模块不存在 RED；实现后 C007 3 passed，全量 9 passed。真实 ModelIR 为 59 modules（7 composite、52 primitive），WorkloadIR 为 2 nodes。

## 结果

- **观察事实：** `repeat count=2` 显式展开为 layer_0/layer_1；两实例 Definition ID 相同、Stable Path/Node ID 不同。root repeat_call 展开为两条 carry call。Workload ModelCall 只保存 model name/version/reference/entrypoint 与 artifact binding，不含模型 children。重复构建 fingerprint 与 canonical JSON 完全一致。
- **推断：** 嵌套模型由 ModelIR 管，跨模型控制由 WorkloadIR 管的边界已由可运行代码证明；三重身份足以追溯模板定义与实例。
- **证据等级变化：** 结构 IR 与 repeat 身份 E1→E2；H-04 仍待 SemanticIR/E2E。
- **信息增量：** SemanticCompiler 可按 ModelCall 解析根 entrypoint，再沿显式 composite steps 递归展开，无需猜执行顺序。

## 结论

- **验收/交付更新：** AC-02/AC-03/D-02/D-03 保持 IN_PROGRESS；ModelIR/WorkloadIR 部分完成。
- **预算变化：** 无硬件实验。
- **下一 micro-goal：** C008 实现 `SemanticCompiler.compile`，生成层次 Region/Typed Value/State Effect/provenance，并证明 CPU/MPS placement 不污染语义结果。
- **是否需决策：** 无。
