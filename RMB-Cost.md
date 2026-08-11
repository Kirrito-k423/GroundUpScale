# RMB Cost

- Generated at: `2026-08-11T02:37:37Z`
- Confidence: `estimate`
- USD/CNY: `7.2`
- Task: Issue #30 Ascend NPU 两层 Transformer Demo（主会话 + Standards/Spec 复审）

## Token Usage

| Item | Tokens | M tokens |
|---|---:|---:|
| Input total | 85,142,812 | 85.142812 |
| Cached input | 83,509,760 | 83.509760 |
| Uncached input | 1,633,052 | 1.633052 |
| Output | 291,410 | 0.291410 |
| Reasoning output, included in output when provider reports it that way | 91,394 | 0.091394 |

## Price Assumptions

| Model | Input USD/M | Cached input USD/M | Output USD/M | Note |
|---|---:|---:|---:|---|
| GPT/Codex estimate | 5 | 0.5 | 30 | Executed path; exact billable model and current official pricing were not verified in this run. |
| DeepSeek counterfactual | 0.435 | 0.003625 | 0.87 | Same-token counterfactual only; no DeepSeek API call occurred and current pricing was not verified. |

## Cost Breakdown

| Model | Component | Cache status | Tokens | M tokens | USD/M | USD | RMB |
|---|---|---|---:|---:|---:|---:|---:|
| GPT/Codex estimate | Input (cache miss) | not cached | 1,633,052 | 1.633052 | 5 | 8.17 | 58.79 |
| GPT/Codex estimate | Input (cache hit) | cached | 83,509,760 | 83.509760 | 0.5 | 41.75 | 300.64 |
| GPT/Codex estimate | Output | not applicable | 291,410 | 0.291410 | 30 | 8.74 | 62.94 |
| DeepSeek counterfactual | Input (cache miss) | not cached | 1,633,052 | 1.633052 | 0.435 | 0.7104 | 5.11 |
| DeepSeek counterfactual | Input (cache hit) | cached | 83,509,760 | 83.509760 | 0.003625 | 0.3027 | 2.18 |
| DeepSeek counterfactual | Output | not applicable | 291,410 | 0.291410 | 0.87 | 0.2535 | 1.83 |

## Cost Summary

| Model | Input cache miss RMB | Input cache hit RMB | Output RMB | Total RMB | Total USD |
|---|---:|---:|---:|---:|---:|
| GPT/Codex estimate | 58.79 | 300.64 | 62.94 | 422.37 | 58.66 |
| DeepSeek counterfactual | 5.11 | 2.18 | 1.83 | 9.12 | 1.27 |

## Formula

`uncached_input = input_total - cached_input`

`uncached_input_cost = uncached_input_M * input_usd_per_M`

`cached_input_cost = cached_input_M * cached_input_usd_per_M`

`output_cost = output_M * output_usd_per_M`

`total_rmb = (uncached_input_cost + cached_input_cost + output_cost) * USD_CNY`

## Notes

- Usage window: `2026-08-11T01:28:01Z` to `2026-08-11T02:37:20Z`.
- Token evidence sums the main-session delta after the window start and the final
  cumulative counters of the two review sessions:
  - `/Users/Zhuanz/.codex/sessions/2026/08/11/rollout-2026-08-11T09-26-45-019fee6d-df81-7772-bb57-ed50760cd8f8.jsonl`
  - `/Users/Zhuanz/.codex/sessions/2026/08/11/rollout-2026-08-11T10-16-20-019fee9b-41b5-78d2-b9da-8bcbfc9917f5.jsonl`
  - `/Users/Zhuanz/.codex/sessions/2026/08/11/rollout-2026-08-11T10-16-31-019fee9b-6c8c-7043-9023-bd31dded6a87.jsonl`
- The main-session baseline event was `2026-08-11T01:27:31.816Z` with
  50,503 input, 13,824 cached-input, 1,501 output, and 745 reasoning-output
  tokens; those counters were subtracted before adding the child sessions.
- Re-run with current official API prices and FX before using this for reimbursement or budget approval.
- Codex goal `tokensUsed` can differ from raw session input/output because it is an effective meter, while session logs also expose cache hits and repeated context reads.
