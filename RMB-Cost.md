# RMB Cost

- Generated at: `2026-08-11T11:54:16Z`
- Confidence: `estimate`
- USD/CNY: `7.2`
- Task: Issue #32 Ascend NPU 四轴 Diagnostic Bundle（主会话 + 双人 code review；远端恢复待续）

## Token Usage

| Item | Tokens | M tokens |
|---|---:|---:|
| Input total | 150,743,446 | 150.743446 |
| Cached input | 147,253,248 | 147.253248 |
| Uncached input | 3,490,198 | 3.490198 |
| Output | 486,243 | 0.486243 |
| Reasoning output, included in output when provider reports it that way | 163,065 | 0.163065 |

## Price Assumptions

| Model | Input USD/M | Cached input USD/M | Output USD/M | Note |
|---|---:|---:|---:|---|
| gpt-5.5 | 5 | 0.5 | 30 | Script default; verify current OpenAI pricing before payable use. |
| deepseek-v4-pro | 0.435 | 0.003625 | 0.87 | Script default; verify current DeepSeek pricing before payable use. |

## Cost Breakdown

| Model | Component | Cache status | Tokens | M tokens | USD/M | USD | RMB |
|---|---|---|---:|---:|---:|---:|---:|
| gpt-5.5 | Input (cache miss) | not cached | 3,490,198 | 3.490198 | 5 | 17.45 | 125.65 |
| gpt-5.5 | Input (cache hit) | cached | 147,253,248 | 147.253248 | 0.5 | 73.63 | 530.11 |
| gpt-5.5 | Output | not applicable | 486,243 | 0.486243 | 30 | 14.59 | 105.03 |
| deepseek-v4-pro | Input (cache miss) | not cached | 3,490,198 | 3.490198 | 0.435 | 1.52 | 10.93 |
| deepseek-v4-pro | Input (cache hit) | cached | 147,253,248 | 147.253248 | 0.003625 | 0.5338 | 3.84 |
| deepseek-v4-pro | Output | not applicable | 486,243 | 0.486243 | 0.87 | 0.4230 | 3.05 |

## Cost Summary

| Model | Input cache miss RMB | Input cache hit RMB | Output RMB | Total RMB | Total USD |
|---|---:|---:|---:|---:|---:|
| gpt-5.5 | 125.65 | 530.11 | 105.03 | 760.79 | 105.66 |
| deepseek-v4-pro | 10.93 | 3.84 | 3.05 | 17.82 | 2.48 |

## Formula

`uncached_input = input_total - cached_input`

`uncached_input_cost = uncached_input_M * input_usd_per_M`

`cached_input_cost = cached_input_M * cached_input_usd_per_M`

`output_cost = output_M * output_usd_per_M`

`total_rmb = (uncached_input_cost + cached_input_cost + output_cost) * USD_CNY`

## Notes

- Usage window: `2026-08-11T07:29:44Z` to `2026-08-11T11:54:05Z`; work remains blocked on restoration of the remote SSH path, so this report must be regenerated after the resumed collection/review/commit steps.
- Token evidence sums the latest cumulative counters from the main session and the two isolated code-review sessions:
  - `/Users/Zhuanz/.codex/sessions/2026/08/11/rollout-2026-08-11T15-23-45-019fefb4-b793-7c23-aeb3-eaba39091ded.jsonl`
  - `/Users/Zhuanz/.codex/sessions/2026/08/11/rollout-2026-08-11T16-14-23-019fefe3-1069-7ff1-b1e4-fd8c8baf7032.jsonl`
  - `/Users/Zhuanz/.codex/sessions/2026/08/11/rollout-2026-08-11T16-14-37-019fefe3-46aa-7e90-80d6-7da3933da483.jsonl`
- Re-run with current official API prices and FX before using this for reimbursement or budget approval.
- Codex goal `tokensUsed` can differ from raw session input/output because it is an effective meter, while session logs also expose cache hits and repeated context reads.
