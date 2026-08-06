# C015：按 Case mode 分层计时窗口

- **开始/结束：** 2026-08-06T19:17:43+08:00 / 2026-08-06T19:21:01+08:00
- **阶段：** M4 / MEASUREMENT STABILITY
- **动作类型：** EXPERIMENT
- **关联验收/未知量：** AC-06、AC-07、H-02

## 预注册

- **本轮 micro-goal：** 冻结一个符合不同 Case 粒度的协议，使 CPU/MPS 10/10 Case 噪声门禁通过。
- **当前假设：** operator 需要约 100 ms 批量窗口摊薄 timer/sync/调度噪声；module/e2e 单次本身已是 42–95 ms，应保持 inner=1，避免连续多线程执行放大频率与系统漂移。
- **已有证据：** C014 operators 6/6 PASS；C012/C013 CPU module/e2e inner=1 均 PASS；C014 CPU E2E inner=2 FAIL；MPS 各轮长 Case 稳定。
- **证据等级：** 目标 E2。
- **唯一主要变量：** 对 module/e2e 强制 inner=1；operator 保留 100 ms target、10-call pilot；所有统计、warmup、samples、Shape 不变。
- **预期观察：** CPU/MPS 所有 Case `IQR/median <=3%`；结果量级与同协议来源 C013/C014 一致。
- **判别规则：** 两设备都完整运行；不删窗口；如仍失败，不再改测量协议并触发 Goal 升级审计。
- **成本与风险：** 预计 1–3 分钟，无外部费用。
- **停止与回滚：** 失败即保留反证并升级；成功则把协议写入运行手册和 manifest 语义。

## 执行

- **脱敏命令：** `commands.md`
- **配置/环境差异：** operator target 100 ms；module/e2e inner=1。
- **日志/指标：** MPS 5/5 PASS，最大 Softmax `2.199%`；CPU 4/5 PASS，Transformer layer `3.238%` FAIL，E2E `2.153%` PASS。两 Bundle digest/alignment/correctness PASS。

## 结果

- 分层协议达成预期方向：CPU operators 最高 `1.921%`、E2E `2.153%`；MPS 全量稳定。
- CPU layer 仍轻微失败，而该 Case 在 C012/C013/C014 分别为 `2.413%`、`2.546%`、`2.266%`；失败点在多轮间漂移，不像 Case 实现固有噪声。
- 本机 `sysctl` 证明 CPU 为 4 performance + 6 efficiency cores，而当前 PyTorch 使用 10 intra-op threads；异构核调度是可证伪的新假设。

## 结论

PARTIAL。MPS 正式协议可冻结；CPU 需先审计 intra-op thread cohort。C016 仅比较默认 10 threads 与 4 threads 的 layer/E2E 噪声，若 4 threads 无改善则按 Goal 升级，不再调整窗口。
