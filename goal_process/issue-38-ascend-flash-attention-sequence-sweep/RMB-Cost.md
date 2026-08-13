# RMB Cost — GitHub Issue #38

- Generated at: `2026-08-13T06:24:10Z`
- Confidence: `estimate`
- USD/CNY: `7.2`
- Task: bounded real Ascend FlashAttention sequence sweep
- Session: `/Users/Zhuanz/.codex/sessions/2026/08/13/rollout-2026-08-13T10-34-24-019ff8f8-85d1-7bf2-9282-3f49621390bd.jsonl`

## Token Usage

| Item | Tokens | M tokens |
|---|---:|---:|
| Input total | 67,886,871 | 67.886871 |
| Cached input | 67,164,928 | 67.164928 |
| Uncached input | 721,943 | 0.721943 |
| Output | 104,101 | 0.104101 |
| Reasoning output, included in output when provider reports it that way | 23,099 | 0.023099 |

## Price Assumptions

| Model | Input USD/M | Cached input USD/M | Output USD/M | Note |
|---|---:|---:|---:|---|
| gpt-5.6-sol | 5 | 0.5 | 30 | Estimate using repository placeholder rates; current official price was not verified. |
| deepseek-v4-pro | 0.435 | 0.003625 | 0.87 | Counterfactual estimate only; no DeepSeek API was invoked. |

## Cost Summary

| Model | Input cache miss RMB | Input cache hit RMB | Output RMB | Total RMB | Total USD |
|---|---:|---:|---:|---:|---:|
| gpt-5.6-sol | 25.99 | 241.79 | 22.49 | 290.27 | 40.32 |
| deepseek-v4-pro | 2.26 | 1.75 | 0.6521 | 4.67 | 0.6481 |

## Formula and boundaries

`uncached_input = input_total - cached_input`

`total_rmb = (uncached_input_M * input_price + cached_input_M * cached_price + output_M * output_price) * USD_CNY`

- 本报告由 `rmb-cost-report` 的 session mode 生成；价格与汇率未在本轮核验，只能作为 estimate。
- DeepSeek 数值仅作反事实成本对照，本任务没有调用 DeepSeek API。
- 远端 NPU 由项目已有机器提供，未计入按时租赁成本。
