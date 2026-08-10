# RMB Cost

- Generated at: `2026-08-10T09:46:20Z`
- Confidence: `estimate`
- USD/CNY: `7.2`
- Task: Issue #29 Ascend NPU MatMul Physical Floor 与 Observation Comparison（主会话 + 两个 code-review 子会话）

## Token Usage

| Item | Tokens | M tokens |
|---|---:|---:|
| Input total | 56,956,721 | 56.956721 |
| Cached input | 55,645,952 | 55.645952 |
| Uncached input | 1,310,769 | 1.310769 |
| Output | 264,634 | 0.264634 |
| Reasoning output, included in output when provider reports it that way | 77,263 | 0.077263 |

## Price Assumptions

| Model | Input USD/M | Cached input USD/M | Output USD/M | Note |
|---|---:|---:|---:|---|
| gpt-5.5 | 5 | 0.5 | 30 | Script default; verify current OpenAI pricing before payable use. |
| deepseek-v4-pro | 0.435 | 0.003625 | 0.87 | Script default; verify current DeepSeek pricing before payable use. |

## Cost Breakdown

| Model | Component | Cache status | Tokens | M tokens | USD/M | USD | RMB |
|---|---|---|---:|---:|---:|---:|---:|
| gpt-5.5 | Input (cache miss) | not cached | 1,310,769 | 1.310769 | 5 | 6.55 | 47.19 |
| gpt-5.5 | Input (cache hit) | cached | 55,645,952 | 55.645952 | 0.5 | 27.82 | 200.33 |
| gpt-5.5 | Output | not applicable | 264,634 | 0.264634 | 30 | 7.94 | 57.16 |
| deepseek-v4-pro | Input (cache miss) | not cached | 1,310,769 | 1.310769 | 0.435 | 0.5702 | 4.11 |
| deepseek-v4-pro | Input (cache hit) | cached | 55,645,952 | 55.645952 | 0.003625 | 0.2017 | 1.45 |
| deepseek-v4-pro | Output | not applicable | 264,634 | 0.264634 | 0.87 | 0.2302 | 1.66 |

## Cost Summary

| Model | Input cache miss RMB | Input cache hit RMB | Output RMB | Total RMB | Total USD |
|---|---:|---:|---:|---:|---:|
| gpt-5.5 | 47.19 | 200.33 | 57.16 | 304.67 | 42.32 |
| deepseek-v4-pro | 4.11 | 1.45 | 1.66 | 7.22 | 1.00 |

## Formula

`uncached_input = input_total - cached_input`

`uncached_input_cost = uncached_input_M * input_usd_per_M`

`cached_input_cost = cached_input_M * cached_input_usd_per_M`

`output_cost = output_M * output_usd_per_M`

`total_rmb = (uncached_input_cost + cached_input_cost + output_cost) * USD_CNY`

## Notes

- Token evidence is the sum of the final cumulative `token_count` event in these
  three Codex session logs:
  - `/Users/Zhuanz/.codex/sessions/2026/08/10/rollout-2026-08-10T16-15-45-019feabd-f6be-7401-9fc0-c7d52c68ac88.jsonl`
  - `/Users/Zhuanz/.codex/sessions/2026/08/10/rollout-2026-08-10T17-06-58-019feaec-d86d-7ea0-8b5f-3af4424acf66.jsonl`
  - `/Users/Zhuanz/.codex/sessions/2026/08/10/rollout-2026-08-10T17-07-13-019feaed-1436-7dc3-9abf-bfe96aef536b.jsonl`
- GPT is the executed path; DeepSeek is a same-token counterfactual estimate, not
  evidence that a DeepSeek API call occurred.
- Re-run with current official API prices and FX before using this for reimbursement or budget approval.
- Codex goal `tokensUsed` can differ from raw session input/output because it is an effective meter, while session logs also expose cache hits and repeated context reads.
