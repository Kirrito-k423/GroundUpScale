# RMB Cost

- Generated at: `2026-08-11T03:27:47Z`
- Confidence: `estimate`
- USD/CNY: `7.2`
- Task: Issue #30 Ascend NPU 两层 Transformer Demo（主会话 + 两轮 Standards/Spec 复审）

## Token Usage

| Item | Tokens | M tokens |
|---|---:|---:|
| Input total | 186,354,752 | 186.354752 |
| Cached input | 182,987,776 | 182.987776 |
| Uncached input | 3,366,976 | 3.366976 |
| Output | 601,880 | 0.601880 |
| Reasoning output, included in output when provider reports it that way | 189,939 | 0.189939 |

## Price Assumptions

| Model | Input USD/M | Cached input USD/M | Output USD/M | Note |
|---|---:|---:|---:|---|
| GPT/Codex estimate | 5 | 0.5 | 30 | Executed path; exact billable model and current official pricing were not verified in this run. |
| DeepSeek counterfactual | 0.435 | 0.003625 | 0.87 | Same-token counterfactual only; no DeepSeek API call occurred and current pricing was not verified. |

## Cost Breakdown

| Model | Component | Cache status | Tokens | M tokens | USD/M | USD | RMB |
|---|---|---|---:|---:|---:|---:|---:|
| GPT/Codex estimate | Input (cache miss) | not cached | 3,366,976 | 3.366976 | 5 | 16.83 | 121.21 |
| GPT/Codex estimate | Input (cache hit) | cached | 182,987,776 | 182.987776 | 0.5 | 91.49 | 658.76 |
| GPT/Codex estimate | Output | not applicable | 601,880 | 0.601880 | 30 | 18.06 | 130.01 |
| DeepSeek counterfactual | Input (cache miss) | not cached | 3,366,976 | 3.366976 | 0.435 | 1.46 | 10.55 |
| DeepSeek counterfactual | Input (cache hit) | cached | 182,987,776 | 182.987776 | 0.003625 | 0.6633 | 4.78 |
| DeepSeek counterfactual | Output | not applicable | 601,880 | 0.601880 | 0.87 | 0.5236 | 3.77 |

## Cost Summary

| Model | Input cache miss RMB | Input cache hit RMB | Output RMB | Total RMB | Total USD |
|---|---:|---:|---:|---:|---:|
| GPT/Codex estimate | 121.21 | 658.76 | 130.01 | 909.97 | 126.39 |
| DeepSeek counterfactual | 10.55 | 4.78 | 3.77 | 19.09 | 2.65 |

## Formula

`uncached_input = input_total - cached_input`

`uncached_input_cost = uncached_input_M * input_usd_per_M`

`cached_input_cost = cached_input_M * cached_input_usd_per_M`

`output_cost = output_M * output_usd_per_M`

`total_rmb = (uncached_input_cost + cached_input_cost + output_cost) * USD_CNY`

## Notes

- Usage window: `2026-08-11T01:28:01Z` to `2026-08-11T03:26:38Z`.
- Token evidence sums the main-session delta after the window start and the final
  cumulative counters of four isolated review sessions:
  - `/Users/Zhuanz/.codex/sessions/2026/08/11/rollout-2026-08-11T09-26-45-019fee6d-df81-7772-bb57-ed50760cd8f8.jsonl`
  - `/Users/Zhuanz/.codex/sessions/2026/08/11/rollout-2026-08-11T10-16-20-019fee9b-41b5-78d2-b9da-8bcbfc9917f5.jsonl`
  - `/Users/Zhuanz/.codex/sessions/2026/08/11/rollout-2026-08-11T10-16-31-019fee9b-6c8c-7043-9023-bd31dded6a87.jsonl`
  - `/Users/Zhuanz/.codex/sessions/2026/08/11/rollout-2026-08-11T11-22-21-019feed7-b26a-7c00-9a32-945c2a04563f.jsonl`
  - `/Users/Zhuanz/.codex/sessions/2026/08/11/rollout-2026-08-11T11-22-34-019feed7-e754-7b21-92cd-f89f784a2233.jsonl`
- The main-session baseline event was `2026-08-11T01:27:31.816Z` with
  50,503 input, 13,824 cached-input, 1,501 output, and 745 reasoning-output
  tokens; those counters were subtracted before adding the child sessions.
- Re-run with current official API prices and FX before using this for reimbursement or budget approval.
- Codex goal `tokensUsed` can differ from raw session input/output because it is an effective meter, while session logs also expose cache hits and repeated context reads.
