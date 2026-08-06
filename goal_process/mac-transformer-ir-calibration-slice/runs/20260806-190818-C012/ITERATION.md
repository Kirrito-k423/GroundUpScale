# C012：CPU/MPS Benchmark、Trace 与不可变 Run Bundle

- **开始/结束：** 2026-08-06T19:08:18+08:00 / 2026-08-06T19:10:54+08:00
- **阶段：** M4 / MEASUREMENT AND EXPLANATION
- **动作类型：** IMPLEMENT + EXPERIMENT
- **关联验收/未知量：** AC-02、AC-08、AC-09、AC-10、H-02、H-03、D-04、D-06

## 预注册

- **本轮 micro-goal：** 通过 CPU/MPS YAML AnalysisPlan 各自产生一个完成态 Run Bundle，包含 5 个强制 Case 的原始计时窗口、结构化 trace、exact Stable Path Alignment、live-set、未归因桶、Explanation Graph 与 HTML。
- **当前假设：** C005 的 5-window sample median 协议可迁移到真实 Case，使每个 Case `IQR/median <= 3%`；forward hooks 能覆盖所有 52 个语义叶子且不污染 benchmark 真值。
- **已有证据：** H-02 原子窗口 E2；C011 真实 E2E correctness E2；28 个实现测试 GREEN。
- **证据等级：** 目标至少 E2（真实目标 Shape、本机、原始窗口与完整环境锁）。
- **唯一主要变量：** 从测试缩短配置切换为 YAML 正式 warmup/samples 与 20 ms 目标窗口；CPU/MPS 使用相同 runner 和不同 DeploymentIntent。
- **预期观察：** 每设备 5 cases × 20 samples × 5 raw windows；所有 Case 噪声门禁通过；60 trace spans、52 operator spans、alignment coverage 100%；Bundle digest 校验通过。
- **判别规则：** benchmark 与 trace 分开执行；MPS 仅在测量边界 synchronize；任一 Case `IQR/median >3%` 保留原始窗口并单独归因，不删除异常点；trace 时间不当作 benchmark 真值。
- **成本与风险：** 两个正式 Run 预计 1–3 分钟，无外部费用；MPS allocator 只有点采样，CPU RSS 不是 framework-attributed，AC-08 不能因本轮有数字就提前宣称通过。
- **停止与回滚：** 同签名无新证据只允许重跑一次；若噪声失败，下一轮只能改变有依据的 warmup/window 协议并保留失败 Bundle。

## 执行

- **脱敏命令：** `commands.md`
- **代码差异：** 已实现 BenchmarkRunner、TraceRunner、live-set、Explanation Graph、RunBundleWriter/verify 与 CLI run/explain。
- **日志/指标：** CPU/MPS 两个 Bundle 均完成且 15 artifacts digest PASS；两者均 60/60 exact alignment。CPU 4/5 noise PASS，RMSNorm `4.228%` FAIL；MPS 4/5 noise PASS，Softmax `3.222%` FAIL。MPS correctness PASS、fallback=false。

## 结果

- CPU E2E median `92.191 ms`，IQR/median `2.257%`；MPS E2E `83.259 ms`，`0.251%`。
- live-set 基础预测：state `35,659,776 B`，peak activation `18,874,368 B`，合计 `54,534,144 B`。
- MPS trace 点采样 framework peak `74,457,088 B`；该值含实际 allocator 行为但 trace 有 observer effect，尚不能直接通过 AC-08。
- 自适应窗口用单次调用估计 steady-state；MPS synchronization 固定开销使 operator 实际窗口只有约 6–9 ms，低于目标 20 ms，是两个失败 Case 的直接协议缺陷。

## 结论

PARTIAL。Run Bundle/trace/alignment 链路成立，但 3% 噪声门禁未全过。保留两个失败 Bundle，C013 只把自适应 pilot 从单次改为 10 次组测，继续使用 20 ms 目标窗口，不改 warmup、样本数或统计方法。
