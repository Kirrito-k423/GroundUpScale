# RMB Cost — GitHub Issue #36

- Generated at: `2026-08-12T10:16:48Z`
- Confidence: `estimate`
- USD/CNY: `7.2`
- Task: GitHub Issue #36 bounded real Ascend MatMul M-sweep
- Session: `/Users/Zhuanz/.codex/sessions/2026/08/12/rollout-2026-08-12T16-57-35-019ff530-f94b-7433-92b3-35dc304faf23.jsonl`

## Token Usage

| Item | Tokens | M tokens |
|---|---:|---:|
| Input total | 18,295,749 | 18.295749 |
| Cached input | 18,046,720 | 18.046720 |
| Uncached input | 249,029 | 0.249029 |
| Output | 51,145 | 0.051145 |
| Reasoning output, included in output when provider reports it that way | 18,897 | 0.018897 |

## Price Assumptions

| Model | Input USD/M | Cached input USD/M | Output USD/M | Note |
|---|---:|---:|---:|---|
| gpt-5.6-sol | 5 | 0.5 | 30 | Estimate using existing repository placeholder rates; current official price was not verified in this turn. |
| deepseek-v4-pro | 0.435 | 0.003625 | 0.87 | Counterfactual estimate only; no DeepSeek API was invoked. |

## Cost Breakdown

| Model | Component | Cache status | Tokens | M tokens | USD/M | USD | RMB |
|---|---|---|---:|---:|---:|---:|---:|
| gpt-5.6-sol | Input (cache miss) | not cached | 249,029 | 0.249029 | 5 | 1.25 | 8.97 |
| gpt-5.6-sol | Input (cache hit) | cached | 18,046,720 | 18.046720 | 0.5 | 9.02 | 64.97 |
| gpt-5.6-sol | Output | not applicable | 51,145 | 0.051145 | 30 | 1.53 | 11.05 |
| deepseek-v4-pro | Input (cache miss) | not cached | 249,029 | 0.249029 | 0.435 | 0.1083 | 0.7800 |
| deepseek-v4-pro | Input (cache hit) | cached | 18,046,720 | 18.046720 | 0.003625 | 0.0654 | 0.4710 |
| deepseek-v4-pro | Output | not applicable | 51,145 | 0.051145 | 0.87 | 0.0445 | 0.3204 |

## Cost Summary

| Model | Input cache miss RMB | Input cache hit RMB | Output RMB | Total RMB | Total USD |
|---|---:|---:|---:|---:|---:|
| gpt-5.6-sol | 8.97 | 64.97 | 11.05 | 84.98 | 11.80 |
| deepseek-v4-pro | 0.7800 | 0.4710 | 0.3204 | 1.57 | 0.2182 |

## Notes

- 本段由 `rmb-cost-report` 的 session mode 生成；价格与汇率未在本轮核验，因此只能作为 estimate。
- 远端真实主 sweep 已完成首轮 24 个声明 Shape；复采因 SSH 入口在握手前关闭而中止，未删除任何原始 Run Bundle，也未启动第二个 Hardware Cohort。
- DeepSeek 数值仅作反事实成本对照，本任务没有调用 DeepSeek API。
