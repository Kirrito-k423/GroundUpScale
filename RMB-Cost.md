# RMB Cost

- Generated at: `2026-08-10T07:56:29Z`
- Confidence: `estimate`
- USD/CNY: `7.2`
- Task: GitHub Issue #28 Ascend NPU Measurement Adapter
- Session: `/Users/Zhuanz/.codex/sessions/2026/08/10/rollout-2026-08-10T14-07-27-019fea48-8024-7421-aa8e-f1c12ca2fc2e.jsonl`
- Token window events: `2026-08-10T06:12:02.764Z` -> `2026-08-10T07:56:05.229Z`

## Token Usage

| Item | Tokens | M tokens |
|---|---:|---:|
| Input total | 36,968,293 | 36.968293 |
| Cached input | 36,474,368 | 36.474368 |
| Uncached input | 493,925 | 0.493925 |
| Output | 129,503 | 0.129503 |
| Reasoning output, included in output when provider reports it that way | 32,964 | 0.032964 |

## Price Assumptions

| Model | Input USD/M | Cached input USD/M | Output USD/M | Note |
|---|---:|---:|---:|---|
| gpt-5.6-sol | 5 | 0.5 | 30 | Estimate: script default GPT rate used as a proxy; current GPT-5.6-sol pricing was not verified. |
| deepseek-v4-pro | 0.435 | 0.003625 | 0.87 | Estimate only; no DeepSeek API was invoked in this task. |

## Cost Breakdown

| Model | Component | Cache status | Tokens | M tokens | USD/M | USD | RMB |
|---|---|---|---:|---:|---:|---:|---:|
| gpt-5.6-sol | Input (cache miss) | not cached | 493,925 | 0.493925 | 5 | 2.47 | 17.78 |
| gpt-5.6-sol | Input (cache hit) | cached | 36,474,368 | 36.474368 | 0.5 | 18.24 | 131.31 |
| gpt-5.6-sol | Output | not applicable | 129,503 | 0.129503 | 30 | 3.89 | 27.97 |
| deepseek-v4-pro | Input (cache miss) | not cached | 493,925 | 0.493925 | 0.435 | 0.2149 | 1.55 |
| deepseek-v4-pro | Input (cache hit) | cached | 36,474,368 | 36.474368 | 0.003625 | 0.1322 | 0.9520 |
| deepseek-v4-pro | Output | not applicable | 129,503 | 0.129503 | 0.87 | 0.1127 | 0.8112 |

## Cost Summary

| Model | Input cache miss RMB | Input cache hit RMB | Output RMB | Total RMB | Total USD |
|---|---:|---:|---:|---:|---:|
| gpt-5.6-sol | 17.78 | 131.31 | 27.97 | 177.06 | 24.59 |
| deepseek-v4-pro | 1.55 | 0.9520 | 0.8112 | 3.31 | 0.4597 |

## Formula

`uncached_input = input_total - cached_input`

`uncached_input_cost = uncached_input_M * input_usd_per_M`

`cached_input_cost = cached_input_M * cached_input_usd_per_M`

`output_cost = output_M * output_usd_per_M`

`total_rmb = (uncached_input_cost + cached_input_cost + output_cost) * USD_CNY`

## Notes

- Actual DeepSeek API usage for this task was zero; therefore actual DeepSeek
  cost was USD `0` / RMB `0`. The DeepSeek rows above are a counterfactual
  estimate using the GPT token window, not recorded DeepSeek billing usage.
- Re-run with current official API prices and FX before using this for reimbursement or budget approval.
- Codex goal `tokensUsed` can differ from raw session input/output because it is an effective meter, while session logs also expose cache hits and repeated context reads.
