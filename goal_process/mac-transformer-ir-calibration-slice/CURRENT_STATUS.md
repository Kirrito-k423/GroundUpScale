# 当前状态

- **Goal：** `mac-transformer-ir-calibration-slice`
- **更新时间：** 2026-08-06T19:26:58+08:00
- **状态：** 黄（CPU noise gate 升级）
- **阶段：** M4 COMPLETE → M5
- **截止时间：** 无固定期限
- **验收进度：** 5/12

## 一分钟摘要

- **目标：** 在 Apple M4 上跑通固定 Shape 两层 Transformer 的 YAML 到 Cost IR、CPU/MPS 预测—实测和 5% 留出门禁。
- **已完成：** M1–M4；YAML→四层 IR→真实 CPU/MPS→5 Case benchmark→60-span trace→Run Bundle→HTML/Explanation Graph 完整闭环。
- **当前主阻塞：** CPU C017 Softmax `IQR/median=3.380%`，超过已确认 3%；MPS 5/5 通过。不能合法宣称 CPU 5% 校准门禁成立。
- **关键证据：** `evidence/m4-milestone-report.md`；29 tests；AC-02/03/04/05/09 DONE。
- **已解决：** benchmark/trace observer effect 分离、60/60 exact alignment、未归因桶、跨设备 Tensor storage memory 口径、不可覆盖 Bundle 与 digest verification。
- **下一步：** 实现版本化 calibration fit/holdout 框架并完成 MPS；CPU 只保留诚实失败，等待升级决策。
- **需要决策：** 无。

## 交付状态

- **代码：** M1–M3 与 C011 已推送；M4 Benchmark/Trace/Bundle 待 checkpoint commit。
- **文档：** M1–M4 reports、C001–C017、架构 ADR 与真实 YAML bundle。
- **复现：** `uv run pytest -q` 为 29 passed；C017 两个 Bundle digest PASS。
- **日志与报告：** `runs/` 下 C001–C017；本地 `.groundupscale/runs/` 保留 raw Bundle；`RMB-Cost.md` 持续监控。

## 时间与预算

- **环境：** M1 已完成，约 9 分钟（含下载与兼容收尾）。
- **调研：** 已完成仓库、硬件、Python/uv 基线核验。
- **实现：** M1 probe + M2 编译 + M3 CostIR + M4 reference/measurement/explanation；29 tests。
- **实验：** C001–C011 环境/编译/正确性；C012–C017 真实 Case 测量、协议反证和 CPU noise 升级。
- **文档与交付：** Goal 合同与过程账本。
- **资源等待：** 0。
- **剩余预算：** 无固定期限；本轮最多 3 个超过 10 分钟实验；同签名无新证据重跑 1 次；版本候选 2 个。
- **费用报告：** `RMB-Cost.md` 已记录起点和 session 证据，费用明细待结束时从 JSONL 计算。

## 条件化 ETA

- **路径 A：** MPS 成立，进入 M5。
- **路径 B：** CPU 已触发：受控测量仍有 Case 超过 3%，不可判定 5% 校准门禁。
- **最晚决策点：** 完成不依赖 CPU 门禁的 M5/M6 实现后，需要用户决定是否接受更受控的本机运行条件或调整 CPU 统计口径；当前不自行降标。
