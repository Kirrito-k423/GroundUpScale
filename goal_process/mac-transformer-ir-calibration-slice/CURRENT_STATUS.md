# 当前状态

- **Goal：** `mac-transformer-ir-calibration-slice`
- **更新时间：** 2026-08-06T18:24:08+08:00
- **状态：** 绿
- **阶段：** M2 COMPLETE → M3
- **截止时间：** 无固定期限
- **验收进度：** 0/12

## 一分钟摘要

- **目标：** 在 Apple M4 上跑通固定 Shape 两层 Transformer 的 YAML 到 Cost IR、CPU/MPS 预测—实测和 5% 留出门禁。
- **已完成：** M1/M2；YAML strict Schema、真实 causal 两层模型、ModelIR/WorkloadIR、层次 SemanticIR、Typed Value/版本化 Effect/provenance 与 compile CLI 均完成。
- **当前主阻塞：** 无；CostIR 公式和独立手算 reference 尚未实现。
- **关键证据：** `evidence/m2-milestone-report.md`；最终 CPU/MPS semantic fingerprint 相同，JSON SHA-256 相同；全量 15 passed。
- **已解决：** YAML 组合边界、repeat 三重身份、ModelCall 嵌套边界、残差/Attention/MLP value 连线、causal mask、placement 与语义隔离。
- **下一步：** TDD 实现 SemanticIR→CostIR；每个 op 输出公式、FLOPs、逻辑 read/write bytes、参数/激活分类、alias 物化规则，并与独立 literal 参考精确对齐。
- **需要决策：** 无。

## 交付状态

- **代码：** 工作区包含 M2 Spec/IR/Compiler/CLI；待 milestone commit。
- **文档：** M1/M2 milestone report、C001–C008、架构 ADR 与真实 YAML bundle。
- **复现：** `uv run pytest -q` 为 15 passed；`groundupscale compile` 产出五类可检查 JSON。
- **日志与报告：** `runs/` 下 C001–C008；M1/M2 reports；`RMB-Cost.md` 持续监控。

## 时间与预算

- **环境：** M1 已完成，约 9 分钟（含下载与兼容收尾）。
- **调研：** 已完成仓库、硬件、Python/uv 基线核验。
- **实现：** M1 probe + M2 strict Specs、三层 IR、SemanticCompiler、compile CLI；15 tests。
- **实验：** C001–C005 本机实验；C006–C008 确定性编译测试。
- **文档与交付：** Goal 合同与过程账本。
- **资源等待：** 0。
- **剩余预算：** 无固定期限；本轮最多 3 个超过 10 分钟实验；同签名无新证据重跑 1 次；版本候选 2 个。
- **费用报告：** `RMB-Cost.md` 已记录起点和 session 证据，费用明细待结束时从 JSONL 计算。

## 条件化 ETA

- **路径 A：** 已成立；M1/M2 为 E2，进入 M3→M6。
- **路径 B：** 未触发。
- **最晚决策点：** M4 首轮真实两层模型误差矩阵后，按 Goal 决定是否进入校准或升级假设。
