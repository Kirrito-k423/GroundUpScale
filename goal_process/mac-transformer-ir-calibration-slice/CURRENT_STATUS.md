# 当前状态

- **Goal：** `mac-transformer-ir-calibration-slice`
- **更新时间：** 2026-08-06T20:46:06+08:00
- **状态：** 红（第三次同一外部环境阻塞，等待用户操作或授权）
- **阶段：** BLOCKED
- **截止时间：** 无固定期限
- **验收进度：** 7/12

## 一分钟摘要

- **目标：** 在 Apple M4 上跑通固定 Shape 两层 Transformer 的 YAML 到 Cost IR、CPU/MPS 预测—实测和 5% 留出门禁。
- **已完成：** M1–M6 的软件、真实穿刺、Run Bundle、calibration governance、公共 CI、运行手册和远端发布；AC-01/02/03/04/05/09/11 DONE。
- **当前主阻塞：** C021 第三次同签名 preflight 失败：normalized load `0.408>0.25`、`mediaanalysisd=89.4%`、两个长期 board 服务均 `55.6% CPU`。未经授权不能停止用户进程，禁止继续采样。
- **关键证据：** C020、C021、`HANDOVER.md`；42 tests；GitHub Actions `31102467129` Success。
- **已解决：** 从 YAML/IR/公式到 CPU/MPS 实测、下钻解释、内存同口径、fit/holdout 隔离、拒绝错误晋升和 public/trusted CI 安全边界。
- **下一步：** 用户暂停两个 autoresearch board 服务（当前 PID 18974/18975）或明确授权临时停止；等待 `mediaanalysisd` 自然降载并以 preflight PASS 作为恢复条件。
- **需要决策：** 是否允许临时停止 PID 18974/18975；推荐允许，完成 cohort 后再由用户自行恢复服务。

## 交付状态

- **代码：** C020 实现 `cb77dd0`、审计 `3cb27e9` 已推送并通过 CI；BLOCKED 交接提交后再次核对远端。
- **文档：** M1–M5 reports、C001–C019、runbook、FINAL-REPORT。
- **复现：** 本地 42 passed；public CI `31102467129` Success 且 canonical double compile deterministic。
- **日志与报告：** `.groundupscale/runs/` 保留全部 raw Bundle；过程目录保留精简证据；`RMB-Cost.md` 持续 estimate。

## 时间与预算

- **环境：** M1 已完成，约 9 分钟（含下载与兼容收尾）。
- **调研：** 已完成仓库、硬件、Python/uv 基线核验。
- **实现：** 六个里程碑的软件部分及 C020 环境门禁落地；42 tests。
- **实验：** C001–C017 环境/编译/测量；C018 3 fit + 7 holdout；C019 clean checkout/public CI；C020 真实环境前置拒绝。
- **文档与交付：** Goal 合同与过程账本。
- **资源等待：** C020–C021 约 16 分钟；长期服务仍在运行。
- **剩余预算：** 无固定期限；本轮最多 3 个超过 10 分钟实验；同签名无新证据重跑 1 次；版本候选 2 个。
- **费用报告：** `RMB-Cost.md` 已记录起点和 session 证据，费用明细待结束时从 JSONL 计算。

## 条件化 ETA

- **路径 A：** 模型误差在有效 MPS holdout 上成立（最大 3.715%），但有效样本数量未成立。
- **路径 B：** CPU 与连续 MPS 均触发测量有效性升级；C020 已将升级落为可执行门禁。
- **最晚决策点：** 已达到并进入 BLOCKED；恢复条件是用户操作/授权后 preflight PASS。
