# RMB Cost — GitHub Issue #38

- Generated at: `2026-08-13T05:03:12Z`
- Confidence: `estimate`
- USD/CNY: `7.2`
- Task: bounded real Ascend FlashAttention sequence sweep
- Session: `/Users/Zhuanz/.codex/sessions/2026/08/13/rollout-2026-08-13T10-34-24-019ff8f8-85d1-7bf2-9282-3f49621390bd.jsonl`

## Token Usage

| Item | Tokens | M tokens |
|---|---:|---:|
| Input total | 50,645,476 | 50.645476 |
| Cached input | 50,120,960 | 50.120960 |
| Uncached input | 524,516 | 0.524516 |
| Output | 65,744 | 0.065744 |
| Reasoning output, included in output when provider reports it that way | 15,616 | 0.015616 |

## Price Assumptions

| Model | Input USD/M | Cached input USD/M | Output USD/M | Note |
|---|---:|---:|---:|---|
| gpt-5.6-sol | 5 | 0.5 | 30 | Estimate using repository placeholder rates; current official price was not verified. |
| deepseek-v4-pro | 0.435 | 0.003625 | 0.87 | Counterfactual estimate only; no DeepSeek API was invoked. |

## Cost Summary

| Model | Input cache miss RMB | Input cache hit RMB | Output RMB | Total RMB | Total USD |
|---|---:|---:|---:|---:|---:|
| gpt-5.6-sol | 18.88 | 180.44 | 14.20 | 213.52 | 29.66 |
| deepseek-v4-pro | 1.64 | 1.31 | 0.4118 | 3.36 | 0.4671 |

## Formula and boundaries

`uncached_input = input_total - cached_input`

`total_rmb = (uncached_input_M * input_price + cached_input_M * cached_price + output_M * output_price) * USD_CNY`

- 本报告由 `rmb-cost-report` 的 session mode 生成；价格与汇率未在本轮核验，只能作为 estimate。
- DeepSeek 数值仅作反事实成本对照，本任务没有调用 DeepSeek API。
- 远端 NPU 由项目已有机器提供，未计入按时租赁成本。
