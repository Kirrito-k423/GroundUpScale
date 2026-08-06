# 当前状态

- **Goal：** `mac-transformer-ir-calibration-slice`
- **更新时间：** 2026-08-06T18:36:36+08:00
- **状态：** 绿
- **阶段：** M3 COMPLETE → M4
- **截止时间：** 无固定期限
- **验收进度：** 2/12

## 一分钟摘要

- **目标：** 在 Apple M4 上跑通固定 Shape 两层 Transformer 的 YAML 到 Cost IR、CPU/MPS 预测—实测和 5% 留出门禁。
- **已完成：** M1–M3；四层 IR、CostRule/公式/依赖、逐 op 到 E2E bytes/FLOPs、串行/理想并行工作界与 compile CLI 已闭合。
- **当前主阻塞：** 无；真实 PyTorch 两层 reference、E2E CPU/MPS correctness、Benchmark/trace/live-set/Run Bundle 尚未实现。
- **关键证据：** `evidence/m3-milestone-report.md`；CPU/MPS CostIR SHA 相同；21 tests；AC-03/AC-05 DONE。
- **已解决：** 算法公式口径、逻辑/物化/状态/activation bytes 分离、alias 零物化、dependency critical path 与硬件隔离。
- **下一步：** 实现 PyTorch reference runner，先证明 CPU/MPS 对同一固定权重/输入数值一致且无 fallback，再接 benchmark/trace/live-set。
- **需要决策：** 无。

## 交付状态

- **代码：** 工作区包含 M3 CostIR/Lowerer/CLI/公式测试与文档；待 milestone commit。
- **文档：** M1/M2 milestone report、C001–C008、架构 ADR 与真实 YAML bundle。
- **复现：** `uv run pytest -q` 为 21 passed；`groundupscale compile` 额外产出 CostIR 与总量。
- **日志与报告：** `runs/` 下 C001–C010；M1–M3 reports；`RMB-Cost.md` 持续监控。

## 时间与预算

- **环境：** M1 已完成，约 9 分钟（含下载与兼容收尾）。
- **调研：** 已完成仓库、硬件、Python/uv 基线核验。
- **实现：** M1 probe + M2 编译 + M3 CostIR/公式/CLI；21 tests。
- **实验：** C001–C005 本机实验；C006–C010 确定性编译/公式验证。
- **文档与交付：** Goal 合同与过程账本。
- **资源等待：** 0。
- **剩余预算：** 无固定期限；本轮最多 3 个超过 10 分钟实验；同签名无新证据重跑 1 次；版本候选 2 个。
- **费用报告：** `RMB-Cost.md` 已记录起点和 session 证据，费用明细待结束时从 JSONL 计算。

## 条件化 ETA

- **路径 A：** 已成立；M1–M3 为 E2，进入 M4→M6。
- **路径 B：** 未触发。
- **最晚决策点：** M4 首轮真实两层模型误差矩阵后，按 Goal 决定是否进入校准或升级假设。
