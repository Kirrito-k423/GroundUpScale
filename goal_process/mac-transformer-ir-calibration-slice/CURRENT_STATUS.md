# 当前状态

- **Goal：** `mac-transformer-ir-calibration-slice`
- **更新时间：** 2026-08-07T12:15:46+08:00
- **状态：** 黄（C028 硬件能力包络穿刺完成；可信基线仍待安静环境）
- **阶段：** M5 VERIFY
- **截止时间：** 无固定期限
- **验收进度：** 7/12

## 一分钟摘要

- **目标：** 在 Apple M4 上跑通固定 Shape 两层 Transformer 的 YAML 到 Cost IR、CPU/MPS 预测—实测和 5% 留出门禁。
- **已完成：** C028 新增 scalar/vector/matrix/copy/triad 多 Shape microbenchmark、M4 CPU P80/P95 能力包络、独立 HardwareCapabilityProfile、算法无关硬件地板和两层样例预测—实测对照；AC-01/02/03/04/05/09/11 仍为 DONE。
- **当前主阻塞：** 只阻塞 trusted calibration/能力基线晋升：当前 `mediaanalysisd` 等竞争使 preflight 不合格；不阻塞用户要求的功能穿刺。
- **关键证据：** `c028-m4-cpu-hardware-envelope-summary.md`；FP32 P80 `1.74845 TFLOP/s`、memory P80 `126.833 GB/s`；E2E floor `5.554 ms` vs observed `92.814 ms`；17 artifact digests verified；58 tests。
- **已解决：** 原 PyTorch scalar dispatch 探针不稳定且概念错误，已替换为可审计 ARM64 scalar FMADD；Scope 地板改用最小 FLOPs 和唯一 compulsory bytes，不再使用当前物化流量定义硬件。
- **下一步：** 若要晋升 trusted M4 基线，在 preflight PASS 时用同一 Suite 重测并执行 10% drift gate；本轮无需用户决策。
- **需要决策：** 无。

## 交付状态

- **代码：** C020 实现 `cb77dd0`、审计 `3cb27e9` 已推送并通过 CI；BLOCKED 交接提交后再次核对远端。
- **文档：** M1–M5 reports、C001–C019、runbook、FINAL-REPORT。
- **复现：** 本地 58 passed；`benchmark-hardware` 与 `m4-cpu-envelope-20260807-v2` 命令见 C028。
- **日志与报告：** `.groundupscale/runs/m4-cpu-envelope-20260807-v2` 保留最新 Run Bundle；过程目录保留 raw microbenchmark 与 C028 摘要；`RMB-Cost.md` 持续 estimate。

## 时间与预算

- **环境：** M1 已完成，约 9 分钟（含下载与兼容收尾）。
- **调研：** 已完成仓库、硬件、Python/uv 基线核验。
- **实现：** 六个里程碑的软件部分及 C020 环境门禁落地；42 tests。
- **实验：** C001–C027 见既有记录；C028 完成两轮能力探针（首轮被稳定性门禁拒绝、第二轮成功）和最新版两层 CPU Run。
- **文档与交付：** Goal 合同与过程账本。
- **资源等待：** C020–C021 约 16 分钟；长期服务仍在运行。
- **剩余预算：** 无固定期限；本轮最多 3 个超过 10 分钟实验；同签名无新证据重跑 1 次；版本候选 2 个。
- **费用报告：** `RMB-Cost.md` 已记录起点和 session 证据，费用明细待结束时从 JSONL 计算。

## 条件化 ETA

- **路径 A：** 若 preflight PASS，同一 Suite 可直接生成 trusted candidate 并与当前 P80 做 10% drift 检查，证据等级 E2。
- **路径 B：** 若后台竞争持续，保留当前 exploratory Profile，仅用于功能和解释链路，不晋升 CI 基线，证据等级 E3（功能）/E1（可信性能）。
- **最晚决策点：** 只有用户要求 trusted 基线或继续 5% calibration 时才需要等待安静环境。
