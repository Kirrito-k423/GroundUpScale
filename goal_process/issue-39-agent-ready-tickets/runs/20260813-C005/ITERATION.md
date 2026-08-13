# C005 — #41–#43 review 与集成

- **输入：** 三票各自从共同 base `5a0958e75c2c9323d2494136b3b26e1d4ded2b67` 形成的隔离分支。
- **门禁：** Standards/Spec 原 reviewer 均完成复审且无 remaining must-fix。
- **集成：** 按 #41、#42、#43 顺序 cherry-pick；公共 `run_bundle.py` 冲突按 bundle kind 语义合并，保留 model-E2E、MatMul 与 compound RMSNorm 三类 verifier。
- **回归：** #41 相关 35 passed；#42 联合 35 passed；#43 联合 54 passed。
- **回放：** #41 complete/unknown、#42 v4、#43 unknown-v3 权威 bundle 均 `verify-run PASS`。
- **远端：** `codex/integration-39` 已推送。
- **GitHub：** #41、#42、#43 已关闭；每次关闭后读取 native blocked-by。#48 当前仍由 OPEN 的 #44、#45、#46 阻塞。
- **边界：** #42 权威 v4 为 0/5 structured unknown；#43 为 0 source/7 graph-derived missing；未把 unknown 伪装为 numeric evidence。
