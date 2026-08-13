# C011：判定并交付 Issue #25

- **开始/结束：** 2026-08-08T03:00:00+08:00 / 2026-08-08T03:12:00+08:00
- **阶段：** HANDOVER
- **动作类型：** READ
- **关联验收/未知量：** AC-25

## 预注册

- **本轮 micro-goal：** 核查两个真实 cohort；只有均可用且证据合格才实施 #25。
- **当前假设：** M4 本机可用，第二真实独立硬件需核验。
- **已有证据：** #22–#24 CLOSED；HEAD `a5c117d`。
- **证据等级：** E1。
- **唯一主要变量：** 第二真实硬件 cohort 的可用性与证据能力。
- **预期观察：** 真实设备身份、capability discovery、preflight/timing/correctness/completion 可复核，或明确 blocker。
- **判别规则：** 任一真实 cohort 不可用/证据不足则停止整个队列；不得以 test Adapter 替代。
- **成本与风险：** 远端可能无网或硬件不匹配；日志必须脱敏。
- **停止与回滚：** 写入前完成 blocker gate；无权威证据不进入实现。

## 执行

- **脱敏命令：** 见 `../../evidence/issue-25-hardware-blocker.md`。
- **配置/环境差异：** HEAD `a5c117d`；本机 macOS M4，远端按脱敏配置核查。
- **代码差异：** 无 #25 写入、测试、提交或关闭动作。
- **日志/指标：** M4 preflight exit 2；27 个本地 Bundle 中合格环境 0、第二 cohort 0；4/4 远端 SSH/Redfish 会话无法建立。

## 结果

- **观察事实：** #22–#24 blockers 已关闭；本机为 M4 但当前 preflight 不合格；没有第二真实 cohort 的本地 Bundle；4 台远端候选机均无法建立执行/管理会话。
- **错误签名：** `load-above-policy`、`total-competing-cpu-above-policy`、SSH exit 255（KEX 前关闭）、Redfish TLS 前关闭。
- **推断：** #25 当前无法满足“两个真实 cohort”验收条件，且禁止用测试 Adapter 替代。
- **证据等级变化：** 第二 cohort 可用性从 E1 提升为 E3 反证。
- **信息增量：** 找到首个真实 blocker，满足整个队列的停止条件。

## 结论

- **验收/交付更新：** AC-25 BLOCKED；AC-26 NOT_STARTED；Goal 按停止条件完成。
- **预算变化：** 停止执行，进入最终费用汇总。
- **下一 micro-goal：** 无；外部环境恢复后由新执行从 #25 重启。
- **是否需决策：** 本轮否。
