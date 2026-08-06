# M2 里程碑报告：YAML 到层次 SemanticIR

## 结论

M2 已完成。CPU 与 MPS AnalysisPlan 使用同一组 YAML 模型/工作负载/分析条件，只在 DeploymentIntent 的 placement 不同；两者确定性产出相同 ModelIR、WorkloadIR 与 SemanticIR。SemanticIR 保留 Workload 和模型嵌套，同时用全局 Typed Value 与版本化 State Effect 表达数据流和状态交互。

## 最终规模

| 表示 | 规模/内容 |
|---|---|
| Spec source graph | 8 个显式版本化 YAML 文档 |
| ModelIR | 59 modules：7 composite、52 primitive |
| WorkloadIR | 2 nodes：Sequence + ModelCall leaf |
| SemanticIR hierarchy | 10 regions：Analysis、Sequence、ModelCall、Transformer、2×Layer、2×Attention、2×MLP |
| Semantic operations | 52，覆盖 MatMul/Add/RMSNorm/Softmax/SiLU/Mul/View/Transpose |
| Typed Value | 73，全部 concrete shape、producer 与 consumer 闭合 |
| State Artifact / Effect | 22 / 22；18 parameter、2 causal-mask buffer、input/output Artifact；read/write 版本显式 |
| Provenance | 240 条 DerivationRecord |

## 关键边界证明

1. **模型嵌套不是 Workload DAG：** WorkloadIR 的 ModelCall 是叶子；SemanticCompiler 才递归展开其 ModelIR entrypoint。
2. **层次与数据流并存：** Region 保留 Transformer/Layer/Attention/MLP 嵌套；Typed Value 跨 Region 链接 producer/consumer。
3. **物理部署不污染语义：** CPU 与 MPS semantic compilation fingerprint 均为 `02d0facf395b847acc2bb850e039136b27b696a476d33bd26d58369eba1b2233`，两份 `semantic-ir.json` SHA-256 均为 `f079edcf266fa421dc9ac920a3d94f2db0c1efe4c77c52da6150b446b2d1df75`。
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

## M4 布局穿刺修订

C011 的真实 PyTorch 执行证明 heads-major attention context 不能直接 flatten，且直接 einsum 也不保证 sequence-major contiguous。最终 SemanticIR 显式保留 V transpose alias，并由 context MatMul 的 `output_layout` 契约直接产生连续 `[B,S,NH,D]`；以上规模、fingerprint 与 SHA 已按经 CPU/MPS 验证的最终表示修订。
