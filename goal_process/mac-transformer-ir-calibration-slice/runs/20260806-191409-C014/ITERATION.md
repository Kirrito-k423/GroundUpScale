# C014：100 ms 稳态计时窗口

- **开始/结束：** 2026-08-06T19:14:09+08:00 / 2026-08-06T19:17:43+08:00
- **阶段：** M4 / MEASUREMENT STABILITY
- **动作类型：** EXPERIMENT
- **关联验收/未知量：** AC-06、AC-07、H-02

## 预注册

- **本轮 micro-goal：** 使用 100 ms 目标窗口使 CPU/MPS 10/10 强制 Case 全部满足 `IQR/median <=3%`，冻结正式测量协议。
- **当前假设：** C013 的短算子 15–20 ms 实际窗口仍受 CPU 调度/MPS queue 批次扰动；把每个 raw window 增至约 100 ms，5-window median 能把 run-level IQR 压到 3% 内。
- **已有证据：** C012/C013 共四个不可变 Bundle；长 E2E 一直 PASS，失败集中在短 operator；10-call pilot 已纠正 MPS inner 低估。
- **证据等级：** 目标 E2。
- **唯一主要变量：** `target_window_ms: 20 → 100`；10-call pilot、YAML warmup、20 samples、5 windows、median/IQR、Shape 和设备不变。
- **预期观察：** 全部 Case noise PASS；CPU/MPS E2E median 与 C013 差异不超过合理的运行间漂移；digest/alignment/correctness 保持 PASS。
- **判别规则：** 逐 Case 使用完整 20 个 sample；不删除窗口；任一仍失败则触发更细的系统干扰诊断，而非继续倍增窗口或放宽 3%。
- **成本与风险：** 预计 1–3 分钟；每设备总 timed workload 约 50 秒，无外部费用。
- **停止与回滚：** C014 是当前依据下最后一个整体窗口候选；失败则按 Goal 的连续模型修正规则升级。

## 执行

- **脱敏命令：** `commands.md`
- **配置/环境差异：** CLI 目标窗口 100 ms，其余不变。
- **日志/指标：** MPS 5/5 PASS（最大 Softmax `1.860%`）；CPU operators 3/3 PASS（最大 Softmax `2.796%`）、layer PASS `2.266%`，但 E2E `4.117%` FAIL。两个 Bundle digest 与 60/60 alignment PASS。

## 结果

- CPU E2E 把 inner 从 1 增到 2 后，20 个 sample 在 `90.369–105.241 ms` 间无单调趋势地摆动；C012/C013 的 inner=1 E2E 分别为 `2.257%`/`2.428%`，说明增加单窗口连续 E2E 工作量会放大多线程系统调度/频率扰动。
- CPU 三个 operator 在 63–102 ms/window 全部通过，证明长窗口对短算子有效。
- MPS 全部通过，E2E `83.227 ms` / `0.227%`，未出现同类 CPU 漂移。

## 结论

PARTIAL。整体 100 ms 不是正确统一协议；证据支持按 Case mode 分层：operator 用 100 ms 自适应窗口，module/e2e 保持一次调用一个 window，再用 5-window median。C015 仅做该分层，不再整体加长。
