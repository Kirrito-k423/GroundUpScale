# 中文预测—实测迭代报告规范

本规范定义两层 Transformer Demo 以及其他 Benchmark Case 的预测—实测迭代报告。它落实 [ADR 0038](../adr/0038-separate-authority-from-tiered-iteration-values.md)：权威证据可以未闭合，但面向迭代的 E2E、组件时间、占比和差异必须有数值。

## 1. 两层结果模型

报告必须同时保存两套互不覆盖的结果：

1. **Authority Result**：资格化、Frontier、Completion Boundary 和 cohort 语义；证据不足时保持 `structured-unknown`。
2. **Report Values**：面向下一轮迭代的最佳可用数值；不得为空，并携带 Evidence Grade、Generation Stage、区间、推导方法、来源与允许用途。

Report Value 不能回写 Authority Result，不能参与未授权的校准、Frontier promotion 或结论性 Verdict。真实零值可以显示为 `0`；缺失证据不得编码成零。

## 2. 证据等级与允许用途

| 等级 | 中文名称 | 最低语义 | 允许用途 | 禁止用途 |
|---|---|---|---|---|
| A | 权威证据 | 同 identity/cohort/boundary，完整资格化和独立 holdout | 验收、校准、Frontier、优化结论 | 无证据外推 |
| B | 可复现实测 | 直接测量且可复现，但资格化或归因链未完全闭合 | 对比、定位、实验排序 | 自动校准、验收结论 |
| C | 代理推导 | trace、锚点、能力曲面或其他显式 proxy | 优化优先级、下一轮 probe | 声称直接实测或确定根因 |
| D | 模型降级估计 | FLOPs、bytes、Resource Demand、阶段比例或显式分配 | 形成假设、保证报告可迭代 | 诊断 Verdict、promotion、校准 |

每个数值还必须声明 Generation Stage：`resource-model`、`implementation-prediction`、`operator-frontier`、`schedule-composition`、`baseline-measurement`、`diagnostic-attribution` 或 `independent-holdout`。

## 3. 非空数值的降级顺序

### 3.1 预测侧

1. A：完整 Schedule Achievable Frontier。
2. B：当前 Implementation Candidate 的可复现 Duration Model。
3. C：已知 schedule 与缺失组件显式代理的组合。
4. D：Resource Demand、Shape、FLOPs、bytes、保守效率和调度假设形成的点估计。

Resource Physical Floor 可以并列显示为下界，但不能单独冒充点预测。D 级必须保存效率、串行/并行、dispatch 和 materialization 假设。

### 3.2 实测侧组件

1. A：同 Completion Boundary 的直接、低侵入、资格化归因。
2. B：可复现实测，但 holdout、稳定性或 attribution closure 不完整。
3. C：Diagnostic Profiling Lane 的组件比例，经显式 overhead ablation 后缩放到基准 E2E。
4. D：预测侧组件权重缩放到真实 E2E；若预测权重也缺失，则使用 Cost IR compute、memory 与 dispatch demand 权重。

D 级必须命名为“实测侧降级估计”，不能命名为“实测组件值”。框架、同步、调度和无法映射的时间保留为独立数值 residual，不得强制塞入算子。

## 4. TOP10 与 E2E 贡献

- 预测侧和实测侧分别按 Exclusive E2E Contribution 选择 TOP10。
- 任一侧达到该侧 E2E 10% 的组件强制进入。
- 联合表取两侧 exact Stable Path 并集，并保留两侧各自排名；一侧不得隐藏另一侧热点。
- 默认叶子粒度是 layer-indexed Stable Path。报告另给模块级汇总，但模块父节点仅用于导航，不重复计入贡献。
- 每侧所有叶子、`其他` 和 `框架/调度/未归因` residual 必须回收到该侧 E2E，舍入前合计精确为 100%。
- overlap、并发、critical-path 和 inclusive parent 单独展示，不再次加入贡献之和。

联合表每行至少包含：中文组件名、Stable Path、operation class、两侧时间/占比/排名、绝对差、倍率、两侧 Evidence Grade、两侧 Generation Stage、区间、推导方法、允许用途和 evidence refs。

## 5. 不确定性与数值格式

- A 使用真实样本、qualification 与 propagation policy 的区间。
- B 使用真实区间与 ±15% 下限中更保守者。
- C 至少 ±30%。
- D 至少 `[0.5×, 2.0×]`。
- 页面默认使用 μs 或 ms、三位有效数字、占比一位小数；机器 JSON 保存原始 ns 和完整精度。

A/A 差异称“预测误差”或“验收差异”。任一侧为 B–D 时必须称“探索性差异”；仍计算绝对差、倍率和占比差，但不能自动触发校准或结论性诊断。

## 6. 中文 HTML 固定结构

1. 报告标题、Run identity、Shape、dtype、Hardware Cohort 与 Completion Boundary。
2. 预测 E2E、实测 E2E、绝对差、倍率四张摘要卡。
3. Authority Status、两侧 Evidence Grade 与 Generation Stage。
4. 两侧 Exclusive E2E Contribution 堆叠图。
5. 预测/实测 TOP10 联合表。
6. 模块级汇总与 52 个叶子的下钻入口。
7. overlap、residual 与 reconciliation。
8. 证据边界、区间与允许用途。
9. 下一轮优化优先级或补测建议。
10. 完整 provenance、Run Bundle 和机器产物链接。

页面不得把 `unknown`、`unavailable` 或空单元格作为核心指标。可以显示“权威证据未闭合”，但同一位置必须给出带等级的 Report Value。

## 7. 同源输出与不可变性

同一 Report Value 集合必须投影为：

- `reports/report.html`：简体中文人类报告；
- `comparison/e2e-gap-report.json`：机器语义和 verifier 输入；
- `comparison/e2e-components.csv`：TOP10 与完整组件表。

HTML 和 CSV 不得自行补值或重算。公共 verifier 必须从锁定 source Run Bundles 重放 authority、降级选择、等级、区间、TOP10、E2E contribution、reconciliation 和三种投影。

旧 Run Bundle 不得原地修改。新格式使用新 Schema、policy version、Run ID，并以 `supersedes` 锁定前一版本的 run identity 与 manifest digest。

## 8. 一键运行边界

一次受公共 NPU host lock 保护的公开运行应完成：

```text
NPU baseline run
→ prediction
→ observation
→ best-available component attribution
→ independent TOP10 selection
→ reconciliation
→ Chinese HTML + JSON + CSV
→ verify-run
```

证据不足不得中断报告生成；它必须降级到 C/D 级并明确下一步测量。NPU 运行仍必须遵守 Hardware Cohort、device visibility、全 session flock 和 immutable Run Bundle 约束。

## 9. 验收清单

- [ ] 页面标题、状态、字段说明和建议均为简体中文。
- [ ] E2E 与 TOP10 核心数值没有空值。
- [ ] 每行都有时间、E2E 占比、等级、阶段、区间和推导方法。
- [ ] 两侧独立 TOP10、10% 强制项和 Stable Path union 正确。
- [ ] 每侧 Exclusive E2E Contribution 精确 reconciliation 到 100%。
- [ ] 最差证据条件仍生成完整 D 级报告。
- [ ] A/B/C/D 不会相互误标或改变 Authority Status。
- [ ] HTML、JSON、CSV 来自同一 Report Value 集合。
- [ ] 旧 bundle 字节不变，新 bundle public verifier PASS。
