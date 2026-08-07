# RMB Cost

- Generated at: `2026-08-07T04:29:21Z`
- Confidence: `estimate`
- USD/CNY: `7.2`
- Task: M4 CPU microbenchmark and empirical hardware floor
- Codex effective goal meter: `346,049` tokens (0.346049M)
- Session: `/Users/Zhuanz/.codex/sessions/2026/08/05/rollout-2026-08-05T11-00-26-019fcfdd-7b51-7880-aacf-ac9839a19f67.jsonl`
- Token window events: `2026-08-07T03:43:28.712Z` -> `2026-08-07T04:28:59.303Z`

## Token Usage

| Item | Tokens | M tokens |
|---|---:|---:|
| Input total | 22,162,540 | 22.162540 |
| Cached input | 21,894,656 | 21.894656 |
| Uncached input | 267,884 | 0.267884 |
| Output | 79,949 | 0.079949 |
| Reasoning output, included in output when provider reports it that way | 21,233 | 0.021233 |

## Price Assumptions

| Model | Input USD/M | Cached input USD/M | Output USD/M | Note |
|---|---:|---:|---:|---|
| gpt-5.5 | 5 | 0.5 | 30 | Script default; verify current OpenAI pricing before payable use. |
| deepseek-v4-pro | 0.435 | 0.003625 | 0.87 | Script default; verify current DeepSeek pricing before payable use. |

## Cost Breakdown

| Model | Component | Cache status | Tokens | M tokens | USD/M | USD | RMB |
|---|---|---|---:|---:|---:|---:|---:|
| gpt-5.5 | Input (cache miss) | not cached | 267,884 | 0.267884 | 5 | 1.34 | 9.64 |
| gpt-5.5 | Input (cache hit) | cached | 21,894,656 | 21.894656 | 0.5 | 10.95 | 78.82 |
| gpt-5.5 | Output | not applicable | 79,949 | 0.079949 | 30 | 2.40 | 17.27 |
| deepseek-v4-pro | Input (cache miss) | not cached | 267,884 | 0.267884 | 0.435 | 0.1165 | 0.8390 |
| deepseek-v4-pro | Input (cache hit) | cached | 21,894,656 | 21.894656 | 0.003625 | 0.0794 | 0.5715 |
| deepseek-v4-pro | Output | not applicable | 79,949 | 0.079949 | 0.87 | 0.0696 | 0.5008 |

## Cost Summary

| Model | Input cache miss RMB | Input cache hit RMB | Output RMB | Total RMB | Total USD |
|---|---:|---:|---:|---:|---:|
| gpt-5.5 | 9.64 | 78.82 | 17.27 | 105.73 | 14.69 |
| deepseek-v4-pro | 0.8390 | 0.5715 | 0.5008 | 1.91 | 0.2655 |

## Formula

`uncached_input = input_total - cached_input`

`uncached_input_cost = uncached_input_M * input_usd_per_M`

`cached_input_cost = cached_input_M * cached_input_usd_per_M`

`output_cost = output_M * output_usd_per_M`

`total_rmb = (uncached_input_cost + cached_input_cost + output_cost) * USD_CNY`

## Notes

- Re-run with current official API prices and FX before using this for reimbursement or budget approval.
- Codex goal `tokensUsed` can differ from raw session input/output because it is an effective meter, while session logs also expose cache hits and repeated context reads.
