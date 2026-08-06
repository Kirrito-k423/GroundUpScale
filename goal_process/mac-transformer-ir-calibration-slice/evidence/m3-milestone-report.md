# M3 里程碑报告：CostIR 与独立公式验证

## 结论

M3 已完成。52 个 Semantic Operation 均降低为硬件无关 Cost Operation；每个节点带公式 Rule ID、concrete expression、逻辑/物化 bytes、状态分类、dependency 和 derivation。单层与两层聚合值通过不调用生产公式的 hard-coded literal 测试。

## 两层最终结果

| 指标 | 值 |
|---|---:|
| FLOPs / serial work | 9,710,850,048 |
| ideal-parallel critical-path FLOPs | 6,489,624,576 |
| logical read bytes | 184,557,568 |
| logical write bytes | 138,412,032 |
| materialized read bytes | 167,780,352 |
| materialized write bytes | 121,634,816 |
| parameter read / unique parameter bytes | 33,562,624 |
| buffer read / unique buffer bytes | 2,097,152 |
| activation read bytes | 148,897,792 |
| explicit activation bytes | 121,634,816 |
| alias result bytes | 16,777,216 |
| workload input/output Artifact bytes | 2,097,152 |

## 口径边界

- logical bytes 是 operand/result Tensor 大小，不是 DRAM/cache traffic。
- materialized bytes 对 View/Transpose 为 0；alias logical size 仍保留供解释。
- explicit activation bytes 是累计产生量，不是峰值；峰值必须由 M4 live-set 生命周期推导。
- parameter/buffer bytes 是唯一 State Artifact 占用；本模型每个 state 每次 forward 只读一次，因此与 read bytes 相等。
- critical-path FLOPs 假设无限并行资源，只是算法工作下界，不是时延。

## 确定性与复现

- Cost compilation fingerprint：`0e7e1de24ff0b472ffa9c66407830f21f5ab1551817e4cb002a3b9b540690a02`。
- CPU/MPS `cost-ir.json` SHA-256：`1e2f1857d3ece1319644528491aea5e8eb17eb341df37091ec03d8c6ebd4c028`。
- C009 3 passed；C010 4 passed；全量 21 passed。
- 公式、范例和串并行边界：`docs/methods/cost-model-formulas.md`。

以上数量与摘要在 C011 真实布局穿刺后修订；FLOPs、物化 bytes、状态和 critical path 不变，零物化 alias 的逻辑累计量按最终 52-op 表示更新。
