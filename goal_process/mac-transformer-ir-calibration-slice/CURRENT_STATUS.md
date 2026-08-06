# 当前状态

- **Goal：** `mac-transformer-ir-calibration-slice`
- **更新时间：** 2026-08-06T18:52:33+08:00
- **状态：** 绿
- **阶段：** M4 IN PROGRESS
- **截止时间：** 无固定期限
- **验收进度：** 3/12

## 一分钟摘要

- **目标：** 在 Apple M4 上跑通固定 Shape 两层 Transformer 的 YAML 到 Cost IR、CPU/MPS 预测—实测和 5% 留出门禁。
- **已完成：** M1–M3；C011 真实两层 causal Transformer reference 及 CPU/MPS E2E correctness、目标设备与 storage-alias 审计。
- **当前主阻塞：** 无；Benchmark/trace/live-set/Run Bundle 尚未实现。
- **关键证据：** `evidence/c011-reference-correctness-summary.md`；E2E max abs `7.152557e-07`；24 tests；AC-03/04/05 DONE。
- **已解决：** 目标 Shape 上 einsum 非连续与 MPS 非连续 `out=` 静默错误；使用显式 MatMul output-layout 契约，无隐藏 Copy。
- **下一步：** 实现强制 Benchmark Case、结构化 trace、live-set、Alignment Map 与不可变 Run Bundle。
- **需要决策：** 无。

## 交付状态

- **代码：** 已推送 M1–M3；C011 reference/布局修正待 checkpoint commit。
- **文档：** M1–M3 reports、C001–C011、架构 ADR 与真实 YAML bundle。
- **复现：** `uv run pytest -q` 为 24 passed；`groundupscale compile` 产出修订后 52-op CostIR。
- **日志与报告：** `runs/` 下 C001–C011；M1–M3 reports；`RMB-Cost.md` 持续监控。

## 时间与预算

- **环境：** M1 已完成，约 9 分钟（含下载与兼容收尾）。
- **调研：** 已完成仓库、硬件、Python/uv 基线核验。
- **实现：** M1 probe + M2 编译 + M3 CostIR + M4 reference；24 tests。
- **实验：** C001–C005 本机探针；C006–C010 确定性编译/公式；C011 CPU/MPS reference correctness 与布局反证。
- **文档与交付：** Goal 合同与过程账本。
- **资源等待：** 0。
- **剩余预算：** 无固定期限；本轮最多 3 个超过 10 分钟实验；同签名无新证据重跑 1 次；版本候选 2 个。
- **费用报告：** `RMB-Cost.md` 已记录起点和 session 证据，费用明细待结束时从 JSONL 计算。

## 条件化 ETA

- **路径 A：** 已成立；M1–M3 与 M4 correctness 为 E2，继续 M4 measurement→M6。
- **路径 B：** 未触发。
- **最晚决策点：** M4 首轮真实两层模型误差矩阵后，按 Goal 决定是否进入校准或升级假设。
