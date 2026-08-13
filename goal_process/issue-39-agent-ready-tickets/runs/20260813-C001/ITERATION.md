# C001：建立 dependency frontier 与隔离基线

- **阶段：** BASELINE / RECON
- **动作类型：** READ / INTEGRATE

## 预注册

- **micro-goal：** 从 GitHub 原生 blocked-by 获得可审计 frontier，并建立共同 base 的隔离 worktrees。
- **预期观察：** #41–#47 ready，#48–#50 按依赖锁定；原工作树用户修改不进入新 worktrees。
- **停止与回滚：** 分支或路径碰撞即停止；不删除既有内容。

## 结果

- 原生 API：#41–#47 blocked_by=[]；#48 blocked_by=[41..46]；#49 blocked_by=[48,47]；#50 blocked_by=[49]。
- Base：`5a0958e75c2c9323d2494136b3b26e1d4ded2b67`。
- 原工作树存在 `.agents/skills` 用户修改，未复制、修改、暂存或提交。
- 已创建 integration、#41、#42、#43 worktrees。

