# M2 里程碑报告：YAML 到层次 SemanticIR

## 结论

M2 已完成。CPU 与 MPS AnalysisPlan 使用同一组 YAML 模型/工作负载/分析条件，只在 DeploymentIntent 的 placement 不同；两者确定性产出相同 ModelIR、WorkloadIR 与 SemanticIR。SemanticIR 保留 Workload 和模型嵌套，同时用全局 Typed Value 与版本化 State Effect 表达数据流和状态交互。

## 最终规模

| 表示 | 规模/内容 |
|---|---|
| Spec source graph | 8 个显式版本化 YAML 文档 |
| ModelIR | 61 modules：7 composite、54 primitive |
| WorkloadIR | 2 nodes：Sequence + ModelCall leaf |
| SemanticIR hierarchy | 10 regions：Analysis、Sequence、ModelCall、Transformer、2×Layer、2×Attention、2×MLP |
| Semantic operations | 54，覆盖 MatMul/Add/RMSNorm/Softmax/SiLU/Mul/View/Transpose |
| Typed Value | 75，全部 concrete shape、producer 与 consumer 闭合 |
| State Artifact / Effect | 22 / 22；18 parameter、2 causal-mask buffer、input/output Artifact；read/write 版本显式 |
| Provenance | 246 条 DerivationRecord |

## 关键边界证明

1. **模型嵌套不是 Workload DAG：** WorkloadIR 的 ModelCall 是叶子；SemanticCompiler 才递归展开其 ModelIR entrypoint。
2. **层次与数据流并存：** Region 保留 Transformer/Layer/Attention/MLP 嵌套；Typed Value 跨 Region 链接 producer/consumer。
3. **物理部署不污染语义：** CPU 与 MPS semantic compilation fingerprint 均为 `44e2dbb4d59c86ff913f43b5044b3613b95a498cea362fb1fa64fa76903033d6`，两份 `semantic-ir.json` SHA-256 均为 `a48c0ebc15f7f633c6cf7734321e5317a614b761baa2a7f321073d1638b7c0be`。
4. **别名不伪装分配：** View/Transpose operation 保留 shape/layout 转换，但结果 value 标记 alias_of，operation 标记 `materialization=zero`。
5. **causal prefill 明确：** 每层 causal mask 是 buffer read + Add，再进入 Softmax；不会在 CostIR 或 runtime 阶段凭空补语义。

## TDD 与命令

- C006 Spec Repository：5 passed。
- C007 Structural IR：3 passed。
- C008 Semantic Compiler/CLI：6 passed。
- 全量：15 passed。
- 复现：`uv run groundupscale compile specs/plans/mac-cpu-prefill.yaml --repository-root . --output <dir> --json`。

## 尚未完成

- CostIR 的 FLOPs、逻辑访问量、参数/激活 bytes 与公式来源（M3）。
- PyTorch 两层模型数值执行、trace、Run Bundle 与实测对齐（M4）。
- 5% 校准与独立留出门禁（M5）。
