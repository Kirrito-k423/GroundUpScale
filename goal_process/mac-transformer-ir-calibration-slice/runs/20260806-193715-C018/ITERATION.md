# C018：MPS 候选校准与五次独立留出

- **开始/结束：** 2026-08-06T19:37:15+08:00 / 2026-08-06T19:55:58+08:00
- **阶段：** M5 / CALIBRATION AND HOLDOUT
- **动作类型：** IMPLEMENT + EXPERIMENT
- **关联验收/未知量：** AC-07、AC-08、AC-10、H-02、H-03

## 预注册

- **本轮 micro-goal：** 用 3 个 MPS fit Run 生成窄适用域候选 Profile，再用 5 个完全独立 holdout Run 验证所有 5 Case latency 与 framework Tensor peak memory 均误差不超过 5%。
- **当前假设：** 在完全锁定 cohort/Shape/Cost fingerprint/measurement protocol 下，fit-run median 能预测独立 MPS holdout；Tensor storage correction factor 能稳定修正基础 live-set 的 21.21% 偏差。
- **已有证据：** C014/C015/C017 MPS 5/5 noise PASS；Calibration 单测覆盖 fit/holdout 隔离、cohort 拒绝、noisy quarantine 与 promotion；33 tests GREEN。
- **证据等级：** 目标 E2。
- **唯一主要变量：** 从实现测试进入真实 MPS 数据集；协议冻结为 operator 100 ms、module/e2e inner=1、9 windows/sample、20 samples。
- **拟合/验证分区：** `fit-01..03` 只用于拟合；`holdout-01..05` 只用于验证，Run ID 和 manifest 不重叠。若某 holdout noise >3%，按预先规则 quarantine 并新增独立 holdout，不把它用于误差判定。
- **预期观察：** fit 3/3 noise PASS；至少 5 个 valid holdout；每个 valid run 的全部 Case latency error<=5%、memory error<=5%；晋升 active profile。
- **判别规则：** 不用 holdout 更新 profile；不删除 noisy/failed Bundle；任一有效 holdout error>5% 则 profile FAIL，不重新拟合该批 holdout。
- **成本与风险：** 预计 5–8 分钟本机 MPS，无外部费用；每个 Run 单独发布和校验，持续汇报进度。
- **停止与回滚：** 最多允许为 noise quarantine 新增 3 个 holdout；模型误差失败进入归因，不静默换公式。

## 执行

- **脱敏命令：** `commands.md`
- **配置/环境差异：** MPS C017 冻结协议；新 Bundle 包含 live Tensor storage observer。
- **日志/指标：** fit 3/3 noise PASS（最大 2.870%），profile `1f66d803cc23...`；7 个 holdout 中 3 valid、4 noisy quarantine，低于 minimum=5；3 个 valid run 全部 latency/memory PASS，最大 Case error `3.715%`、memory error `0%`。

## 结果

- fit-run memory peak 均为 `69,214,208 B`；基础 peak `54,534,144 B` 保留，candidate correction factor 约 `1.26918`。
- valid holdout-01/-03/-04 的最大逐 Case误差分别为 `2.477%`、`3.715%`、`1.370%`。
- noisy quarantine：holdout-02 Softmax `3.309%`；holdout-05 Softmax `16.148%`/E2E `3.239%`；holdout-06 MatMul `5.308%`；holdout-07 MatMul `3.206%`/Softmax `4.647%`/E2E `3.540%`。
- 预注册允许最多新增 3 个 holdout；新增 2 个后有效数仍为 3，即便最后一个名额有效也只能到 4，因此提前停止，不浪费一次测量来制造不可能的门禁。

## 结论

FAIL（insufficient valid holdouts）。Candidate 没有晋升、基础 CostIR 没有覆盖、holdout 没有参与拟合。误差模型在所有有效证据上满足 5%，但当前本机连续测量环境无法满足用户确认的“至少 5 个有效 holdout 且每个 noise<=3%”。触发 Goal 升级点。
