# C017：九窗口稳健 sample

- **开始/结束：** 2026-08-06T19:23:00+08:00 / 2026-08-06T19:26:58+08:00
- **阶段：** M4 / ROBUST STATISTICS AUDIT
- **动作类型：** EXPERIMENT
- **关联验收/未知量：** AC-06、AC-07、H-02

## 预注册

- **本轮 micro-goal：** 在不改变单次执行、线程 cohort、窗口工作量或门禁的前提下，使偶发系统窗口不再主导 sample 中值，并冻结正式协议。
- **当前假设：** 9-window median 相比 5-window median 更能容忍 1–2 个后台/调度扰动窗口，CPU/MPS 所有 Case 的 sample IQR 将稳定低于 3%。
- **已有证据：** C012–C015 raw windows 全保留且失败 Case 跨轮漂移；C016 10/4 threads 均可通过且无一致改善；问题不是固定 Case 或 thread cohort 缺陷。
- **证据等级：** 目标 E2。
- **唯一主要变量：** `windows_per_sample: 5 → 9`；operator 100 ms target、module/e2e inner=1、10-call pilot、YAML warmup、20 samples、median/IQR、threads=10 均不变。
- **预期观察：** CPU/MPS 10/10 noise PASS；median 与 C015 同量级；180 raw windows/case 全部保留。
- **判别规则：** 不剔除任何窗口；任一失败即认为当前机器公共后台环境无法可靠支撑 3% 门禁并升级用户决策，不再调整测量协议。
- **成本与风险：** 预计 2–4 分钟，无外部费用。
- **停止与回滚：** 失败则保留完整 Bundle 和反证，停止协议搜索；成功则冻结协议用于 M5 全部 fit/holdout。

## 执行

- **脱敏命令：** `commands.md`
- **配置/环境差异：** windows/sample=9。
- **日志/指标：** MPS 5/5 PASS，最大 Softmax `2.383%`；CPU 4/5 PASS，Softmax `3.380%` FAIL，其他最高 RMSNorm `1.921%`。两 Bundle completed、15 artifacts digest PASS、alignment 60/60。

## 结果

- CPU median：MatMul `0.1544 ms`、RMSNorm `0.0638 ms`、Softmax `0.7564 ms`、Layer `45.5615 ms`、E2E `91.9872 ms`。
- MPS median：MatMul `0.1151 ms`、RMSNorm `0.1460 ms`、Softmax `0.1863 ms`、Layer `41.9401 ms`、E2E `83.5857 ms`。
- MPS 180 raw windows/case 全量门禁通过；CPU 只有 Softmax 失败且不得用其他成功轮替换本轮结果。

## 结论

M4 链路 PASS，CPU 统计前置门禁 FAIL。Benchmark/trace/alignment/Run Bundle/Explanation 均可交付；MPS 可进入 M5，CPU 不能在当前已确认 3% 口径下合法进入校准留出门禁。按预注册停止测量协议搜索，保留升级点。
