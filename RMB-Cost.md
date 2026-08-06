# RMB Cost Report

> **状态：** 运行中，estimate。当前仅建立费用监控证据；尚未结束 Goal，未计算最终费用。

## 监控范围

- Goal：`mac-transformer-ir-calibration-slice`
- Goal execution 开始：2026-08-06T17:27:14+08:00
- Codex thread：`019fcfdd-7b51-7880-aacf-ac9839a19f67`
- Session JSONL：`/Users/Zhuanz/.codex/sessions/2026/08/05/rollout-2026-08-05T11-00-26-019fcfdd-7b51-7880-aacf-ac9839a19f67.jsonl`
- 启动时 Codex effective goal meter：182,481 tokens（来自 `get_goal`，不等同于原始输入/输出 token 总数）
- 结束时间：待完成

## Token 与费用分项

| 分项 | Tokens | GPT USD | GPT RMB | DeepSeek USD | DeepSeek RMB |
|---|---:|---:|---:|---:|---:|
| 输入未命中缓存 | 待从 session JSONL 计算 | 待计算 | 待计算 | 待计算 | 待计算 |
| 输入命中缓存 | 待从 session JSONL 计算 | 待计算 | 待计算 | 待计算 | 待计算 |
| 输出 | 待从 session JSONL 计算 | 待计算 | 待计算 | 待计算 | 待计算 |
| **合计** | 待计算 | 待计算 | 待计算 | 待计算 | 待计算 |

## 价格与汇率

- 本轮尚未核验最新 GPT、DeepSeek API 价格和 USD/CNY 汇率。
- 因此本报告必须保持 `estimate`；不得用未核验默认值冒充应付费用。
- Goal 结束后使用 `$rmb-cost-report` 提供的 `build_rmb_cost_report.py` 从上述 session JSONL、起止时间和 goal meter 生成最终明细。
