# C026：受控环境下的全新 MPS 校准 cohort

- **开始：** 2026-08-07T09:44:27+08:00
- **阶段：** M5 / EXPERIMENT
- **动作类型：** EXPERIMENT
- **关联验收/未知量：** AC-07、AC-08、AC-10、H-02、H-03、H-06

## 预注册

- **本轮 micro-goal：** 在每个 Run 都通过 `local-apple-silicon-v1` preflight 的条件下，采集 3 个全新 MPS fit 和至少 5 个有效独立 holdout，验证逐 Case latency 与 framework Tensor peak memory 误差均 ≤5%，并在通过后晋升 Profile。
- **当前假设：** 排除 board/媒体分析竞争后，3 个 fit 与至少 5/8 holdout 会满足 `IQR/median<=3%`；C018 有效样本的最大模型误差 3.715%，所以全新 cohort 的 5% gate 可达。
- **已有证据：** C018 模型误差证据；C020 环境治理；C025 最终 preflight PASS。
- **证据等级：** E2。
- **唯一主要变量：** 从未受控 C018 切换到 passed-preflight 的全新 MPS cohort；模型、Shape、seed、CostIR、20 samples、9 windows/sample、100 ms operator target、3%/5 次/5% 均不变。
- **拟合/验证分区：** `20260807-controlled-mps-fit-01..03` 仅拟合；`holdout-01..08` 仅验证。Run ID 不重叠，不复用 C012–C018。先固定 candidate，再采 holdout。
- **预期观察：** fit 3/3 环境/noise PASS；至少 5 个 valid holdout；全部 valid latency/memory error≤5%；fallback=false；promote 成 active profile。
- **判别规则：** Run preflight exit 2 时未创建 Bundle，可等待后重试同 ID；已创建但 noise>3% 的 fit 使本轮 fit FAIL，不替换；noisy holdout 原样 quarantine，最多新增到 8。任何有效 holdout error>5% 立即判定模型 FAIL，不用 holdout 重拟合。
- **成本与风险：** 预计 15–25 分钟本机 MPS，无付费资源。每个 Run 单独发布、hash 验证；环境变差即停在门禁前。
- **停止与回滚：** 最多 3 fit + 8 holdout；达到 5 valid 后停止新增；失败保留 Bundle/candidate/validation，不覆盖基础 CostIR。无论结果如何，后续恢复 board LaunchAgent。

## 执行

- **脱敏命令：** `commands.md`
- **日志/指标：** 两次调用均在创建 Run Bundle 前被 preflight 拒绝。第一次归一化 load 为 `0.209`，但短时 `Codex Renderer` 单进程采样峰值为 `29.0%`；等待 45 秒后的第二次，归一化 load 为 `0.262`，短时 `Code Helper (Renderer)` 单进程采样峰值为 `50.3%`。两次竞争进程名称不同，没有形成持续占用证据。

## 结果

- 未生成任何 fit 或 holdout Run Bundle，没有测量数据进入拟合或验证。
- `local-apple-silicon-v1` 把“某个竞争进程在任一 1 秒窗口的单核 CPU 峰值”当成硬门禁；在 10 核机器上，普通 UI 的一次 `50.3%` 单核活动只相当于约 `5.03%` 整机容量，却会直接拒绝实验。
- C026 的原始 3% 噪声、5 次独立留出、5% 误差合同均未被修改；本轮只暴露了进入实验前的门禁度量问题。

## 结论

`ABORTED_BEFORE_EVIDENCE`。C026 不构成失败的 MPS 标定，也不能继续靠随机等待挑选窗口。下一轮先以 TDD 将门禁升级为 v2：单进程峰值保留为诊断，硬门禁改为所有竞争进程相对整机容量的持续总占用，并把 policy ID 纳入 `hardware_cohort`；代码、测试和公开 CI 通过后再预注册新的测量 cohort。
