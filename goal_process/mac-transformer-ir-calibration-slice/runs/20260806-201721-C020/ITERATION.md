# C020：受控 Mac 环境前置门禁

- **开始/结束：** 2026-08-06T20:17:21+08:00 / 2026-08-06T20:29:54+08:00
- **阶段：** M5 / HARDWARE OBSERVATION VALIDITY
- **动作类型：** IMPLEMENT + VERIFY
- **关联验收/未知量：** AC-06、AC-07、AC-08、AC-10、AC-12

## 预注册

- **本轮 micro-goal：** 将本地 Mac 的供电、热/性能告警、系统负载和竞争进程采集为 allowlist 环境证据，并在受信任测量入口中于 benchmark 之前执行硬门禁。
- **当前假设：** C012–C018 的间歇性超噪声与未受控的后台竞争负载相关；预先等待并只在稳定环境中采样可以减少无效 Run，但不能保证结果，因此仍保留逐 Run 的 3% 噪声检验。
- **已有证据：** C017 CPU Softmax 噪声 3.380%；C018 的 7 个 MPS holdout 仅 3 个有效；本轮开始快照显示外部 Python 约 84.5%/16.5% CPU、mediaanalysisd 75.9%、mds_stores 51.5%，10 核 1 分钟 load average 2.64。
- **证据等级：** 目标 E2。
- **唯一主要变量：** 增加测量前的环境可接受性门禁；不改变模型、Shape、warmup、window、3% 噪声阈值、5% 误差阈值或 5 个有效 holdout 要求。
- **预期观察：** CLI 输出结构化、可解释的环境报告；非 Darwin/arm64、非交流供电、热状态未知/告警、归一化负载过高或单一竞争进程持续占用过高时返回不合格；Run Bundle 可记录报告，trusted lane 对不合格环境在 benchmark 前失败。
- **判别规则：** 所有必选检查均通过才 `eligible=true`；未知等同不合格；不终止用户进程；环境合格也不替代 benchmark 内部的 IQR/median 检验。
- **成本与风险：** 预计 10–15 分钟，无付费服务；进程只记录 PID、进程名和 CPU 百分比，不记录命令参数、环境变量或文件路径。
- **停止与回滚：** 若 macOS 系统信息无法稳定读取，保留 unknown 和失败原因，不用默认值伪造通过；公共 Linux CI 只测试纯判定逻辑，不执行真实硬件证据。

## 执行

- **脱敏命令：** `commands.md`
- **代码差异：** 新增 `environment.py`、`preflight` CLI、Run Bundle 硬门禁、Manifest/环境 artifact 状态、calibration evidence 前置拒绝，以及 trusted local script 接入。
- **日志/指标：** 42 tests PASS；真实 3×1 s preflight 正确识别 `normalized load=0.3632>0.25`、`mediaanalysisd=58.1%>25%`，返回 `eligible=false`；真实 `run --require-valid-environment` 返回结构化拒绝且未创建 Bundle；GitHub Actions `31102277844` Success（60 s）。

## 结果

- nominal、竞争负载、unknown、版本化阈值、CLI、结构化运行拒绝、Run Bundle 通过/拒绝和 calibration 未验证证据拒绝均有回归测试。
- 当前机器 AC power、Darwin/arm64、thermal nominal 三项通过；负载与竞争进程两项失败。
- 只记录 allowlist 字段；未杀进程，未执行正式 benchmark，未生成可混入校准的 Run Bundle。

## 结论

PASS（工具与治理目标）。假设“当前存在可观测的环境干扰”得到直接支持；“门禁能把未来有效 holdout 提升到至少 5 个”仍须在环境合格后的全新 cohort 中验证。Goal 继续保持 3%/5 次/5% 原合同，当前进入外部负载消退等待，不降标、不抽样碰运气。
