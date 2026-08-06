# C013：修正短算子自适应窗口估计

- **开始/结束：** 2026-08-06T19:10:54+08:00 / 2026-08-06T19:14:09+08:00
- **阶段：** M4 / MEASUREMENT STABILITY
- **动作类型：** EXPERIMENT
- **关联验收/未知量：** AC-06、AC-07、H-02

## 预注册

- **本轮 micro-goal：** 消除 C012 单次 pilot 中同步固定开销导致的 operator 窗口低估，使 CPU/MPS 全部 5 Case 达到 `IQR/median <=3%`。
- **当前假设：** 用 10 次调用共享一次前后同步来估计 steady-state 单次耗时，会让实际 timed window 接近 20 ms，CPU RMSNorm 与 MPS Softmax 噪声降到 3% 内。
- **已有证据：** C012 原始窗口：CPU RMSNorm 17.3 ms/window 仍为 4.228%；MPS operator 实际约 6–9 ms/window，Softmax 3.222%；其余 8/10 Case 已通过。
- **证据等级：** 目标 E2。
- **唯一主要变量：** 自适应 pilot 从 1 invocation 改为 10 invocations；目标窗口、5-window median、20 samples、YAML warmup、Shape、设备均不变。
- **预期观察：** operator inner iterations 更接近 `20ms / steady_latency`；CPU/MPS 10/10 Case noise PASS；Bundle digest 与 60/60 alignment 继续 PASS。
- **判别规则：** 不删除 C012；不剔除任何 raw window；若同类失败持续，不再无依据重复该方案。
- **成本与风险：** 预计 1–3 分钟，无外部费用。
- **停止与回滚：** 任一 Case 仍失败则保留 C013 并进入按 Case 的稳定性诊断，不能放宽 3%。

## 执行

- **脱敏命令：** `commands.md`
- **配置/环境差异：** 仅 pilot iterations=10。
- **日志/指标：** CPU：MatMul `3.222%` FAIL、RMSNorm `3.230%` FAIL、其余 PASS；MPS：Softmax `3.768%` FAIL、其余 PASS。MPS operator inner iterations 已从 C012 的 44–53 提升至 88–124，证明 pilot 修复有效但 20 ms 仍不足以稳定全部短算子。

## 结果

- 两个 Bundle 均 completed、digest PASS；所有原始窗口保留。
- CPU E2E `92.610 ms` / `2.428%`；MPS E2E `83.222 ms` / `0.206%`。
- 10-call pilot 使 MPS MatMul/RMSNorm 分别达到 `0.287%`/`0.332%`，但 Softmax 仍为 `3.768%`；CPU 两个短算子约 `3.22%`。

## 结论

FAIL（噪声门禁）。该方案不再同签名重跑。C014 保留 10-call pilot，把唯一变量设为目标窗口 20 ms→100 ms；这是对 C012/C013 失败 Case 的直接统计响应，不改 5-window median 或 20 samples。
