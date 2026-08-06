# C016：CPU 异构核线程数诊断

- **开始/结束：** 2026-08-06T19:21:01+08:00 / 2026-08-06T19:23:00+08:00
- **阶段：** M4 / CPU STABILITY DIAGNOSTIC
- **动作类型：** EXPERIMENT
- **关联验收/未知量：** AC-06、H-02

## 预注册

- **本轮 micro-goal：** 判定 PyTorch 默认 10 intra-op threads 跨 4P+6E 异构核调度是否是 CPU layer/E2E 跨样本漂移来源。
- **当前假设：** 固定 4 intra-op threads 会减少跨性能等级核心迁移，使 layer/E2E 的 `IQR/median` 稳定低于 3%；吞吐变化如实记录，不以更快为判据。
- **已有证据：** `sysctl`: perflevel0=4、perflevel1=6、physicalcpu=10；C012–C015 默认 10 threads 的失败 Case 漂移，MPS 同协议稳定。
- **证据等级：** 目标 E2。
- **唯一主要变量：** `torch.set_num_threads(10)` vs `4`；同一固定输入/权重、YAML warmup、20 samples × 5 windows、layer/E2E inner=1。
- **预期观察：** 4 threads 两 Case 均噪声 PASS，且相对 10 threads 有一致稳定性改善。
- **判别规则：** 不要求 4 threads 更快；若 4 threads 仍任一失败或仅偶然改善，线程假设不成立并升级。
- **成本与风险：** 预计 1 分钟，无外部费用；只影响当前进程。
- **停止与回滚：** 诊断后进程退出自动恢复；是否把 thread cohort 写进 YAML/环境契约由证据决定。

## 执行

- **脱敏命令：** `commands.md`
- **配置/环境差异：** 同进程依次 `torch.set_num_threads(10)` 与 `4`；目标 Shape/模型/统计不变。
- **日志/指标：** 10 threads：layer `48.131 ms` / `2.369%`，E2E `96.622 ms` / `2.143%`；4 threads：layer `45.830 ms` / `2.896%`，E2E `92.450 ms` / `1.739%`。四项均 PASS，但稳定性没有一致改善。

## 结果

- 4 threads 的 E2E IQR 改善，但 layer IQR 变差；不能把一次局部改善归因于 P/E core cohort。
- 默认 10 threads 在相同定向诊断中两 Case 都通过，证明 C015 的单层失败不是确定性线程配置缺陷。

## 结论

REJECT HYPOTHESIS。保持默认 10 threads 并记录 cohort；不再尝试线程候选。升级到稳健统计审计：每 sample 的原始 window 数从最低 5 增至 9，仍取 median，减少偶发系统窗口改变 sample 的概率。
