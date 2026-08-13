# 决策记录

## D-001：固定共同 base

- **决定：** 所有 issue worktree 从 `5a0958e75c2c9323d2494136b3b26e1d4ded2b67` 创建。
- **原因：** 满足用户隔离协议，且避开原工作树用户未提交修改。

## D-002：并发调度

- **决定：** 最多三个实现 Agent；需要双轴 review 时逐票释放足够 reviewer 槽位。
- **原因：** 同时满足用户并发上限与 code-review skill 的双 reviewer 要求。

## D-003：公共 verifier 冲突按 bundle kind 合成

- **决定：** #41–#43 集成时保留各自独立 required roles、identity 与 replay 分支，共享通用 artifact/digest/supersession 验证。
- **原因：** 三票扩展同一公共 `run_bundle.py`，任一整文件覆盖都会删除其他票的公共契约。

## D-004：structured unknown 可验收但不得升级为 numeric

- **决定：** #42、#43 在预注册 evidence boundary 未闭合时发布可递归验证的 structured unknown，并关闭对应 contract/qualification ticket。
- **原因：** issue 验收允许 honest unknown；禁止通过重复采集或伪造 numeric evidence 绕过资格化门禁。
