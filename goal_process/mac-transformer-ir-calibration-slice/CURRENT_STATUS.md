# 当前状态

- **Goal：** `mac-transformer-ir-calibration-slice`
- **更新时间：** 2026-08-06T20:29:54+08:00
- **状态：** 黄（受控环境门禁已完成，等待本机竞争负载消退）
- **阶段：** M6 COMPLETE / M5 CONTROLLED RERUN PENDING
- **截止时间：** 无固定期限
- **验收进度：** 7/12

## 一分钟摘要

- **目标：** 在 Apple M4 上跑通固定 Shape 两层 Transformer 的 YAML 到 Cost IR、CPU/MPS 预测—实测和 5% 留出门禁。
- **已完成：** M1–M6 的软件、真实穿刺、Run Bundle、calibration governance、公共 CI、运行手册和远端发布；AC-01/02/03/04/05/09/11 DONE。
- **当前主阻塞：** C020 真实 preflight 检测到 normalized load 0.363>0.25、`mediaanalysisd` 58.1% CPU>25%，按预注册 policy 在 benchmark 前拒绝。Profile 未晋升，不能宣称 AC-06/07/08/10/12 DONE。
- **关键证据：** `evidence/m5-calibration-attempt-report.md`、`evidence/c020-environment-preflight-summary.md`；42 tests；GitHub Actions Success。
- **已解决：** 从 YAML/IR/公式到 CPU/MPS 实测、下钻解释、内存同口径、fit/holdout 隔离、拒绝错误晋升和 public/trusted CI 安全边界。
- **下一步：** 保持 3%/5 次/5% 原合同；待外部 Python、媒体分析/索引负载结束且 preflight 通过后，创建全新 MPS fit/holdout cohort。
- **需要决策：** 无。

## 交付状态

- **代码：** 软件实现与最终审计文档均已推送；交付时以 `git rev-parse HEAD` 和 `git ls-remote` 的一致结果为准。
- **文档：** M1–M5 reports、C001–C019、runbook、FINAL-REPORT。
- **复现：** 本地 42 passed；上一 clean checkout 34 passed；public CI Success；canonical compile deterministic。
- **日志与报告：** `.groundupscale/runs/` 保留全部 raw Bundle；过程目录保留精简证据；`RMB-Cost.md` 持续 estimate。

## 时间与预算

- **环境：** M1 已完成，约 9 分钟（含下载与兼容收尾）。
- **调研：** 已完成仓库、硬件、Python/uv 基线核验。
- **实现：** 六个里程碑的软件部分及 C020 环境门禁落地；42 tests。
- **实验：** C001–C017 环境/编译/测量；C018 3 fit + 7 holdout；C019 clean checkout/public CI；C020 真实环境前置拒绝。
- **文档与交付：** Goal 合同与过程账本。
- **资源等待：** 0。
- **剩余预算：** 无固定期限；本轮最多 3 个超过 10 分钟实验；同签名无新证据重跑 1 次；版本候选 2 个。
- **费用报告：** `RMB-Cost.md` 已记录起点和 session 证据，费用明细待结束时从 JSONL 计算。

## 条件化 ETA

- **路径 A：** 模型误差在有效 MPS holdout 上成立（最大 3.715%），但有效样本数量未成立。
- **路径 B：** CPU 与连续 MPS 均触发测量有效性升级；C020 已将升级落为可执行门禁。
- **最晚决策点：** 原合同已确认保持不变。下一次硬件采样只在 preflight PASS 后开始。
