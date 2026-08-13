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

---

# RMB Cost — Apple M4 Top-5 MatMul Frontier

- Generated at: `2026-08-10T03:45:47Z`
- Confidence: `estimate`
- USD/CNY: `7.2`
- Task: Apple M4 Top-5 MatMul exact-Shape Frontier integration and demo comparison
- Session: `/Users/Zhuanz/.codex/sessions/2026/08/07/rollout-2026-08-07T20-06-56-019fdc1e-8a31-7f03-a662-43d7b78ed1e5.jsonl`
- Token window events: `2026-08-10T01:32:45.286Z` -> `2026-08-10T03:45:32.887Z`

## Token Usage

| Item | Tokens | M tokens |
|---|---:|---:|
| Input total | 62,200,312 | 62.200312 |
| Cached input | 61,223,680 | 61.223680 |
| Uncached input | 976,632 | 0.976632 |
| Output | 109,316 | 0.109316 |
| Reasoning output, included in output when provider reports it that way | 26,294 | 0.026294 |

## Price Assumptions

| Model | Input USD/M | Cached input USD/M | Output USD/M | Note |
|---|---:|---:|---:|---|
| gpt-5.6-sol | 5 | 0.5 | 30 | Estimate using script placeholder rates; current official price was not verified in this turn. |
| deepseek-v4-pro | 0.435 | 0.003625 | 0.87 | Counterfactual estimate using script placeholder rates; no DeepSeek API was invoked. |

## Cost Breakdown

| Model | Component | Cache status | Tokens | M tokens | USD/M | USD | RMB |
|---|---|---|---:|---:|---:|---:|---:|
| gpt-5.6-sol | Input (cache miss) | not cached | 976,632 | 0.976632 | 5 | 4.88 | 35.16 |
| gpt-5.6-sol | Input (cache hit) | cached | 61,223,680 | 61.223680 | 0.5 | 30.61 | 220.41 |
| gpt-5.6-sol | Output | not applicable | 109,316 | 0.109316 | 30 | 3.28 | 23.61 |
| deepseek-v4-pro | Input (cache miss) | not cached | 976,632 | 0.976632 | 0.435 | 0.4248 | 3.06 |
| deepseek-v4-pro | Input (cache hit) | cached | 61,223,680 | 61.223680 | 0.003625 | 0.2219 | 1.60 |
| deepseek-v4-pro | Output | not applicable | 109,316 | 0.109316 | 0.87 | 0.0951 | 0.6848 |

## Cost Summary

| Model | Input cache miss RMB | Input cache hit RMB | Output RMB | Total RMB | Total USD |
|---|---:|---:|---:|---:|---:|
| gpt-5.6-sol | 35.16 | 220.41 | 23.61 | 279.18 | 38.77 |
| deepseek-v4-pro | 3.06 | 1.60 | 0.6848 | 5.34 | 0.7419 |

## Formula

`uncached_input = input_total - cached_input`

`uncached_input_cost = uncached_input_M * input_usd_per_M`

`cached_input_cost = cached_input_M * cached_input_usd_per_M`

`output_cost = output_M * output_usd_per_M`

`total_rmb = (uncached_input_cost + cached_input_cost + output_cost) * USD_CNY`

## Notes

- Re-run with current official API prices and FX before using this for reimbursement or budget approval.
- Codex goal `tokensUsed` can differ from raw session input/output because it is an effective meter, while session logs also expose cache hits and repeated context reads.

---

# RMB Cost — GitHub Issue #27

- Generated at: `2026-08-10T05:22:53Z`
- Confidence: `estimate`
- USD/CNY: `7.2`
- Task: GitHub Issue #27 real Ascend Hardware Cohort
- Session: `/Users/Zhuanz/.codex/sessions/2026/08/10/rollout-2026-08-10T11-51-34-019fe9cc-15e7-7681-b5a7-369343cd8caf.jsonl`
- Token window events: `None` -> `2026-08-10T05:22:38.916Z`

## Token Usage

| Item | Tokens | M tokens |
|---|---:|---:|
| Input total | 35,159,159 | 35.159159 |
| Cached input | 34,543,616 | 34.543616 |
| Uncached input | 615,543 | 0.615543 |
| Output | 122,162 | 0.122162 |
| Reasoning output, included in output when provider reports it that way | 39,597 | 0.039597 |

## Price Assumptions

| Model | Input USD/M | Cached input USD/M | Output USD/M | Note |
|---|---:|---:|---:|---|
| gpt-5.6-sol | 5 | 0.5 | 30 | Estimate using script placeholder rates; current official price was not verified in this turn. |
| deepseek-v4-pro | 0.435 | 0.003625 | 0.87 | Counterfactual estimate using script placeholder rates; no DeepSeek API was invoked. |

## Cost Breakdown

| Model | Component | Cache status | Tokens | M tokens | USD/M | USD | RMB |
|---|---|---|---:|---:|---:|---:|---:|
| gpt-5.6-sol | Input (cache miss) | not cached | 615,543 | 0.615543 | 5 | 3.08 | 22.16 |
| gpt-5.6-sol | Input (cache hit) | cached | 34,543,616 | 34.543616 | 0.5 | 17.27 | 124.36 |
| gpt-5.6-sol | Output | not applicable | 122,162 | 0.122162 | 30 | 3.66 | 26.39 |
| deepseek-v4-pro | Input (cache miss) | not cached | 615,543 | 0.615543 | 0.435 | 0.2678 | 1.93 |
| deepseek-v4-pro | Input (cache hit) | cached | 34,543,616 | 34.543616 | 0.003625 | 0.1252 | 0.9016 |
| deepseek-v4-pro | Output | not applicable | 122,162 | 0.122162 | 0.87 | 0.1063 | 0.7652 |

## Cost Summary

| Model | Input cache miss RMB | Input cache hit RMB | Output RMB | Total RMB | Total USD |
|---|---:|---:|---:|---:|---:|
| gpt-5.6-sol | 22.16 | 124.36 | 26.39 | 172.90 | 24.01 |
| deepseek-v4-pro | 1.93 | 0.9016 | 0.7652 | 3.59 | 0.4993 |

## Formula and Notes

`uncached_input = input_total - cached_input`

`total_rmb = (uncached_input_M * input_rate + cached_input_M * cached_rate + output_M * output_rate) * USD_CNY`

- Re-run with current official API prices and FX before reimbursement or budget approval.
- DeepSeek values are counterfactual only; this task did not invoke DeepSeek API.

---

# RMB Cost — GroundUpScale / SimAI / Echo 固定提交能力联评

- Generated at: `2026-08-10T09:15:37Z`
- Confidence: `estimate`
- USD/CNY: `7.2`
- Task window: `2026-08-10T08:19:02.701Z` -> `2026-08-10T09:15:37Z`
- Token evidence window: first event `2026-08-10T08:20:54.634Z`; latest included cumulative snapshots were read immediately before report generation.
- Included sessions:
  - Main: `/Users/Zhuanz/.codex/sessions/2026/08/10/rollout-2026-08-10T16-17-52-019feabf-e42d-7153-acd4-5345144c62bd.jsonl`
  - SimAI evidence: `/Users/Zhuanz/.codex/sessions/2026/08/10/rollout-2026-08-10T16-22-21-019feac4-01d3-7ec1-aad5-82e042ff9df1.jsonl`
  - Echo evidence: `/Users/Zhuanz/.codex/sessions/2026/08/10/rollout-2026-08-10T16-22-34-019feac4-31ca-7f12-a1fa-5c6b460e0200.jsonl`
  - GroundUpScale evidence: `/Users/Zhuanz/.codex/sessions/2026/08/10/rollout-2026-08-10T16-22-45-019feac4-5fc3-7ee1-a10d-370411491fc4.jsonl`

## Token Usage

| Item | Tokens | M tokens |
|---|---:|---:|
| Input total | 27,745,782 | 27.745782 |
| Cached input | 26,650,880 | 26.650880 |
| Uncached input | 1,094,902 | 1.094902 |
| Output | 125,749 | 0.125749 |
| Reasoning output, included in output when provider reports it that way | 25,705 | 0.025705 |

## Price Assumptions

| Model | Input USD/M | Cached input USD/M | Output USD/M | Note |
|---|---:|---:|---:|---|
| gpt-5.6-sol | 5 | 0.5 | 30 | Script placeholder rates; current official price was not verified in this turn. |
| deepseek-v4-pro | 0.435 | 0.003625 | 0.87 | Counterfactual script placeholder rates; no DeepSeek API was invoked. |

## Cost Breakdown

| Model | Component | Cache status | Tokens | M tokens | USD/M | USD | RMB |
|---|---|---|---:|---:|---:|---:|---:|
| gpt-5.6-sol | Input (cache miss) | not cached | 1,094,902 | 1.094902 | 5 | 5.47 | 39.42 |
| gpt-5.6-sol | Input (cache hit) | cached | 26,650,880 | 26.650880 | 0.5 | 13.33 | 95.94 |
| gpt-5.6-sol | Output | not applicable | 125,749 | 0.125749 | 30 | 3.77 | 27.16 |
| deepseek-v4-pro | Input (cache miss) | not cached | 1,094,902 | 1.094902 | 0.435 | 0.4763 | 3.43 |
| deepseek-v4-pro | Input (cache hit) | cached | 26,650,880 | 26.650880 | 0.003625 | 0.0966 | 0.6956 |
| deepseek-v4-pro | Output | not applicable | 125,749 | 0.125749 | 0.87 | 0.1094 | 0.7877 |

## Cost Summary

| Model | Input cache miss RMB | Input cache hit RMB | Output RMB | Total RMB | Total USD |
|---|---:|---:|---:|---:|---:|
| gpt-5.6-sol | 39.42 | 95.94 | 27.16 | 162.52 | 22.57 |
| deepseek-v4-pro | 3.43 | 0.6956 | 0.7877 | 4.91 | 0.6823 |

## Formula and Notes

`uncached_input = input_total - cached_input`

`total_rmb = (uncached_input_M * input_rate + cached_input_M * cached_rate + output_M * output_rate) * USD_CNY`

- 本段由 `/Users/Zhuanz/.codex/skills/rmb-cost-report/scripts/build_rmb_cost_report.py` 的 manual token mode 生成，再追加到既有报告；四个会话各取最后一个累计 `token_count` 快照后求和。
- 本轮未核验最新 API 价格和 USD/CNY 汇率，因此只能作为 estimate，不能直接用于报销或预算审批。
- DeepSeek 数值仅作反事实成本对照，本任务没有调用 DeepSeek API。

---

# RMB Cost — GitHub Issue #36

- Generated at: `2026-08-12T10:16:48Z`
- Confidence: `estimate`
- USD/CNY: `7.2`
- Task: GitHub Issue #36 bounded real Ascend MatMul M-sweep
- Session: `/Users/Zhuanz/.codex/sessions/2026/08/12/rollout-2026-08-12T16-57-35-019ff530-f94b-7433-92b3-35dc304faf23.jsonl`
- Token window events: `None` -> `2026-08-12T10:16:38.201Z`

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

## Formula and Notes

`uncached_input = input_total - cached_input`

`total_rmb = (uncached_input_M * input_rate + cached_input_M * cached_rate + output_M * output_rate) * USD_CNY`

- 本段由 `rmb-cost-report` 的 session mode 生成；价格与汇率未在本轮核验，因此只能作为 estimate。
- 远端真实主 sweep 已完成首轮 24 个声明 Shape；复采因 SSH 入口在握手前关闭而中止，未删除任何原始 Run Bundle，也未启动第二个 Hardware Cohort。
- DeepSeek 数值仅作反事实成本对照，本任务没有调用 DeepSeek API。
