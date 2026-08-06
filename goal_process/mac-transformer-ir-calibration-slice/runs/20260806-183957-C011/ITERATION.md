# C011：真实 PyTorch reference 与布局语义穿刺

- **开始/结束：** 2026-08-06T18:39:57+08:00 / 2026-08-06T18:52:33+08:00
- **阶段：** M4 / REFERENCE CORRECTNESS
- **动作类型：** IMPLEMENT
- **关联验收/未知量：** AC-04、AC-07、H-04、D-04

## 预注册

- **本轮 micro-goal：** 实现与 YAML/SemanticIR 一致的真实两层 causal PyTorch forward，并在 CPU/MPS 固定输入与权重上证明数值一致、输出/参数均位于目标设备且未启用 MPS fallback。
- **当前假设：** 当前 attention context 的 heads-first transpose 后 flatten 不能零物化；MatMul 后端可把 contraction 结果直接写入 sequence-major contiguous storage，并保持后续 View 真正 alias。
- **已有证据：** M2/M3 语义与公式 E2；C001 原子 MPS correctness E2；尚无组合模型。
- **证据等级：** H-04 semantic E2、runtime E1。
- **唯一主要变量：** 从 IR-only 到一个真实 reference runner；允许修正由真实布局穿刺直接反证的 context 表达，但不改变 Shape、dtype、层数、数学结果或操作集合。
- **预期观察：** IR 修正后 52 semantic ops；FLOPs/状态/显式 activation 不变；CPU/MPS output allclose；MPS input/output/parameters 全是 `mps:0`，fallback env 未开启。
- **判别规则：** 先修改 layout 期望使旧 IR 测试 RED；修正 Spec/Cost 后 GREEN；reference 公开 runner 再 RED→GREEN。数值失败按最大误差与首个模块定位，不放宽到无界容差。
- **成本与风险：** 预计 20–35 分钟；单次 E2E 小于 1 分钟；无外部费用。
- **停止与回滚：** 若 einsum/context 在 MPS 不支持或非 contiguous，保留反证并升级，不能用隐藏 `.contiguous()` 冒充零物化 View。

## 布局反证与替代方案

- 直接 `einsum("bhqk,bkhd->bqhd")` 在 CPU 与 MPS 均返回非 contiguous stride，随后 `view(B,S,H)` 会失败；该路径已作为反证保留。
- `torch.matmul(..., out=heads_major_output)` 在小 Shape 的 CPU/MPS 诊断中误差为 0，但在冻结的 `S=512` 上 MPS 最大绝对误差达到 `152.42796`；证明不可采用该实现，也证明探针必须覆盖目标 Shape。
- 最终实现将 left operand 变成 query-major alias view，并用广播 batched MatMul 直接产生 contiguous `[B,S,NH,D]`；CPU/MPS 对标准 heads-major MatMul 最大绝对误差均为 `9.1553e-05`，后续 flatten View 保持 alias。
- 因此最终语义保留 `v_transpose` 零物化视图，并把 `output_layout: sequence_major_contiguous` 作为 `context_matmul` 的后端契约；不插入隐藏 Copy/Contiguous。

## 执行

- **脱敏命令：** `commands.md`
- **配置/环境差异：** 锁定环境不变。
- **代码差异：** 新增真实 PyTorch reference/runner、forward hooks 与设备/alias 审计；修正 context MatMul 输出布局契约、V transpose 和 CostRule；新增 reference correctness 测试。
- **日志/指标：** 专项 3 passed；全量 24 passed；CPU/MPS E2E max abs `7.152557e-07`、max relative `3.223419e-06`；16/16 alias checks；MPS 52 个 leaf 与全部 state 均为 `mps:0`，fallback=false。

## 结果

- **验收：** AC-04 DONE；H-04 从 semantic-only 提升为固定 Shape 本机 E2。
- **结构：** ModelIR 59 modules；SemanticIR/CostIR 52 ops；73 values；FLOPs/物化 bytes/状态占用不变。
- **产物：** `evidence/c011-reference-correctness-summary.md`。

## 结论

PASS。冻结操作集合足以表达并执行该 reference；真实布局反证已转化为显式后端布局契约，没有隐藏 Copy/Contiguous。进入 benchmark/trace/Run Bundle。
