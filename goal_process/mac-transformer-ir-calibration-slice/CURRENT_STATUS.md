# 当前状态

- **Goal：** `mac-transformer-ir-calibration-slice`
- **更新时间：** 2026-08-06T17:47:53+08:00
- **状态：** 绿
- **阶段：** M1 COMPLETE → M2
- **截止时间：** 无固定期限
- **验收进度：** 0/12

## 一分钟摘要

- **目标：** 在 Apple M4 上跑通固定 Shape 两层 Transformer 的 YAML 到 Cost IR、CPU/MPS 预测—实测和 5% 留出门禁。
- **已完成：** M1 完成；锁定 Python/PyTorch 环境、公开 probe CLI、8 类操作 CPU/MPS 正确性、allocator 接口和稳健噪声协议均有本机证据。
- **当前主阻塞：** 无环境阻塞；M2 尚未实现 YAML Schema、层次 ModelIR/WorkloadIR 与 `SemanticCompiler.compile`。
- **关键证据：** `evidence/m1-milestone-report.md`；C001–C005；CPU 最终 IQR/median 1.611%，MPS 0.314%。
- **已解决：** PyTorch/MPS 兼容、目标原子操作支持、短窗口噪声、MPS 预热漂移、CPU 调度尖峰的稳健聚合。
- **下一步：** 以固定两层 Transformer YAML bundle 为公开输入，TDD 实现严格加载、ModelIR/WorkloadIR 与层次化 SemanticIR 编译 seam。
- **需要决策：** 无。

## 交付状态

- **代码：** 工作区包含 M1 包骨架、probe CLI 与测试；待 milestone commit。
- **文档：** Goal 与架构文档已在仓库；过程账本初始化中。
- **复现：** `uv sync --python 3.11 --group dev && uv run pytest -q` 通过；CPU/MPS probe 通过。
- **日志与报告：** `runs/` 下 C001–C005；`evidence/m1-milestone-report.md`；根目录 `RMB-Cost.md` 持续监控。

## 时间与预算

- **环境：** M1 已完成，约 9 分钟（含下载与兼容收尾）。
- **调研：** 已完成仓库、硬件、Python/uv 基线核验。
- **实现：** probe CLI、能力/数值/计时/内存探针、TDD 契约。
- **实验：** C001–C005，约 11 分钟；所有失败结果保留。
- **文档与交付：** Goal 合同与过程账本。
- **资源等待：** 0。
- **剩余预算：** 无固定期限；本轮最多 3 个超过 10 分钟实验；同签名无新证据重跑 1 次；版本候选 2 个。
- **费用报告：** `RMB-Cost.md` 已记录起点和 session 证据，费用明细待结束时从 JSONL 计算。

## 条件化 ETA

- **路径 A：** 已成立；M1 为 E2，进入 M2→M6。
- **路径 B：** 未触发。
- **最晚决策点：** M4 首轮真实两层模型误差矩阵后，按 Goal 决定是否进入校准或升级假设。
