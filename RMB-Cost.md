# RMB Cost

- Generated at: `2026-08-11T06:58:44Z`
- Confidence: `estimate`
- USD/CNY: `7.2`
- Task: Issue #31 Ascend NPU MatMul Operator Frontier（主会话 + 双人 code-review/复审）

## Token Usage

| Item | Tokens | M tokens |
|---|---:|---:|
| Input total | 151,153,935 | 151.153935 |
| Cached input | 148,823,040 | 148.823040 |
| Uncached input | 2,330,895 | 2.330895 |
| Output | 370,740 | 0.370740 |
| Reasoning output, included in output when provider reports it that way | 118,182 | 0.118182 |

## Price Assumptions

| Model | Input USD/M | Cached input USD/M | Output USD/M | Note |
|---|---:|---:|---:|---|
| GPT/Codex estimate | 5 | 0.5 | 30 | Executed path; exact billable model and current official pricing were not verified in this run. |
| DeepSeek counterfactual | 0.435 | 0.003625 | 0.87 | Same-token counterfactual only; no DeepSeek API call occurred and current pricing was not verified. |

## Cost Breakdown

| Model | Component | Cache status | Tokens | M tokens | USD/M | USD | RMB |
|---|---|---|---:|---:|---:|---:|---:|
| GPT/Codex estimate | Input (cache miss) | not cached | 2,330,895 | 2.330895 | 5 | 11.65 | 83.91 |
| GPT/Codex estimate | Input (cache hit) | cached | 148,823,040 | 148.823040 | 0.5 | 74.41 | 535.76 |
| GPT/Codex estimate | Output | not applicable | 370,740 | 0.370740 | 30 | 11.12 | 80.08 |
| DeepSeek counterfactual | Input (cache miss) | not cached | 2,330,895 | 2.330895 | 0.435 | 1.01 | 7.30 |
| DeepSeek counterfactual | Input (cache hit) | cached | 148,823,040 | 148.823040 | 0.003625 | 0.5395 | 3.88 |
| DeepSeek counterfactual | Output | not applicable | 370,740 | 0.370740 | 0.87 | 0.3225 | 2.32 |

## Cost Summary

| Model | Input cache miss RMB | Input cache hit RMB | Output RMB | Total RMB | Total USD |
|---|---:|---:|---:|---:|---:|
| GPT/Codex estimate | 83.91 | 535.76 | 80.08 | 699.76 | 97.19 |
| DeepSeek counterfactual | 7.30 | 3.88 | 2.32 | 13.51 | 1.88 |

## Formula

`uncached_input = input_total - cached_input`

`uncached_input_cost = uncached_input_M * input_usd_per_M`

`cached_input_cost = cached_input_M * cached_input_usd_per_M`

`output_cost = output_M * output_usd_per_M`

`total_rmb = (uncached_input_cost + cached_input_cost + output_cost) * USD_CNY`

## Notes

- Usage window: `2026-08-11T03:35:57Z` to `2026-08-11T06:58:18Z`.
- Token evidence sums the complete cumulative counters of the main session and two isolated code-review sessions:
  - `/Users/Zhuanz/.codex/sessions/2026/08/11/rollout-2026-08-11T11-35-57-019feee4-2836-7251-9896-c039d90acf4e.jsonl`
  - `/Users/Zhuanz/.codex/sessions/2026/08/11/rollout-2026-08-11T13-50-51-019fef5f-a9ee-7cb3-9d3d-8c922d5459fd.jsonl`
  - `/Users/Zhuanz/.codex/sessions/2026/08/11/rollout-2026-08-11T13-51-00-019fef5f-cbdf-75c0-90a0-809cfd4c03f2.jsonl`
- Re-run with current official API prices and FX before using this for reimbursement or budget approval.
- Codex goal `tokensUsed` can differ from raw session input/output because it is an effective meter, while session logs also expose cache hits and repeated context reads.
