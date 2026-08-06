# M4 里程碑报告：CPU/MPS 实测、Trace 与可解释 Run Bundle

## 结论

M4 实现与链路已完成。CPU/MPS YAML Plan 均能通过同一 CLI 生成不可覆盖、可校验的 Run Bundle；benchmark 与 trace 分离，真实两层 forward 的 52 个语义叶子全部 exact 对齐，Explanation Graph 可由 E2E 指标下钻到 Stable Path、CostRule 公式与 runtime span。

M4 同时保留了一个不能掩盖的升级点：MPS C017 的 5 个 Case 全部满足 `IQR/median<=3%`，CPU Softmax 为 `3.380%`，因此 CPU 暂不满足 M5 校准前置条件。其他成功轮不能替换失败轮来拼接通过。

## Run Bundle 证据

| 项目 | CPU C017 | MPS C017 |
|---|---:|---:|
| completed artifacts | 15 | 15 |
| digest verification | PASS | PASS |
| trace spans / operator spans | 60 / 52 | 60 / 52 |
| exact alignment coverage | 100% | 100% |
| correctness | CPU deterministic | allclose PASS、fallback=false |
| Explanation Graph | 181 nodes / 115 edges | 181 nodes / 115 edges |

Bundle 位于本地 `.groundupscale/runs/20260806-m4-{cpu,mps}-c017/`，小型、可版本化的结论进入本报告；大体量 raw windows/trace 由 manifest SHA-256 锁定且不提交 Git。

## 正式 Case 结果

| Case | CPU median | CPU IQR/median | MPS median | MPS IQR/median |
|---|---:|---:|---:|---:|
| MatMul Q projection | 0.1544 ms | 0.197% | 0.1151 ms | 0.178% |
| RMSNorm | 0.0638 ms | 1.921% | 0.1460 ms | 0.294% |
| Attention Softmax | 0.7564 ms | **3.380% FAIL** | 0.1863 ms | 2.383% |
| Transformer layer | 45.5615 ms | 1.812% | 41.9401 ms | 0.277% |
| Two-layer E2E | 91.9872 ms | 1.131% | 83.5857 ms | 0.219% |

协议：10-call steady-state pilot；operator 约 100 ms/raw window；module/E2E inner=1；每 sample 取 9 raw windows 的 median；20 samples 完整保留。C012–C016 的失败与反证 Bundle 同样保留。

## 内存口径

- Semantic live-set 基础预测：state `35,659,776 B` + peak activation `18,874,368 B` = `54,534,144 B`。
- CPU/MPS 真实 forward 的 live unique Tensor storage peak 均为 `69,214,208 B`，峰值位于 layer 1 attention out projection。
- 基础预测误差 `21.21%`；差异来自 reference Python/runtime 的实际 Tensor 生命周期，进入 M5 受控校准。
- MPS allocator point sample、driver allocation 与 CPU RSS 保留为诊断口径，不用于替代上述 framework-attributed gate metric。

## 可解释性

Run Bundle 包含：resolved input lock、environment allowlist、Model/Workload/Semantic/Cost IR、结构预测、raw benchmark、JSONL trace、Alignment Map、memory/correctness、Error Attribution、Explanation Graph 和 standalone HTML。

CLI：

```sh
groundupscale run <analysis-plan.yaml>
groundupscale verify-run <bundle>
groundupscale explain <bundle>
```

## M5 入口

- MPS：允许进入 duration/memory calibration 与独立留出。
- CPU：3% 噪声前置门禁未满足，按 Goal 停止协议搜索并需要升级决策；实现校准框架和确定性测试仍可继续，但不能宣称 CPU AC-06 通过。
