# Goal #39：执行全部 agent-ready tickets

- **状态：** READY
- **父 Issue：** #39
- **范围：** #41–#50
- **显式 base commit：** `5a0958e75c2c9323d2494136b3b26e1d4ded2b67`
- **Integration branch：** `codex/integration-39`
- **执行协议：** `$goal-execution`

## 目标

依照 GitHub 原生 blocked-by frontier，使用每票独立 Agent、branch、local worktree 与远端目录完成 #41–#50；逐票 TDD、双轴 code review、提交、集成、验证并关闭，最终形成可回放的 Ascend NPU Run Bundle 与报告。

## 验收

1. #41–#50 均只在其原生 blockers CLOSED 后启动，并最终 CLOSED。
2. 每票 commit 已逐票集成到 `codex/integration-39`，相关测试与 frontier 批次全套测试通过。
3. 真实 NPU 动作仅通过 `/home/t00906153/.groundupscale/bin/with-ascend-lock` 执行，且每次 Run Bundle 保存锁、时间、Hardware Cohort 与 device visibility 元数据。
4. 每票报告 branch、SHA、测试、NPU 状态、evidence 路径与风险。
5. #50 前验证 issue #30 冻结 Run Bundle 与 `1.921530 ms` Observation 未改变。

## 权限与边界

- 允许创建分支/worktree、远端 issue 专用目录、提交、集成、推送和关闭验收通过的 issue。
- 禁止改动原工作树用户未提交内容；禁止 `git reset --hard` 与 `git checkout --`。
- 禁止把 NPU 排队编码为 GitHub blocker；禁止绕过远端 flock wrapper。
- NPU timeout 75 是资源暂不可用，应保留进度并重新排队。

## Frontier 基线（2026-08-13）

- Ready：#41、#42、#43、#44、#45、#46、#47（原生 blockers 为空）。
- #48 blocked by #41–#46。
- #49 blocked by #48、#47。
- #50 blocked by #49。

