# 验收账本

| ID | 完成条件 | 状态 | 证据 |
|---|---|---|---|
| AC-41..50 | 对应 issue 验收、集成、测试、证据、关闭 | PASS | 每票双轴 review PASS；最终集成 `687 passed`；GitHub #41–#50 CLOSED（关闭 #50 后刷新） |
| AC-FRONTIER | 严格按 GitHub native blocked-by 推进 | PASS | 初始 #41–#47；#41–#46 CLOSED 后 #48；#47/#48 CLOSED 后 #49；#49 CLOSED 后 #50 |
| AC-ISOLATION | 一票一 Agent、branch、worktree、远端目录 | PASS | `codex/issue-41` … `codex/issue-50`；共同 base `5a0958e…`；独立 `GroundUpScale-issue-<n>` |
| AC-NPU | 所有权威 NPU 运行通过远端 host lock 且元数据完整 | PASS | wrapper SHA `22d43618…b94787`；固定 visibility 0；见 `evidence/npu-lock-ledger.md` |
| AC-REVIEW | 每票 Standards + Spec review 收敛 | PASS | #41–#50 原 reviewer 均最终 PASS，无剩余 must-fix |
| AC-VERIFY | authority bundles 可公共回放 | PASS | #42/#43/#44/#47/#48/#49/#50 authority 均 `verify-run` PASS；#45 30/30 source + 5/5 v2 PASS |
| AC-TEST | 逐票回归与最终完整套件 | PASS | 最终 `uv run pytest -q`: `687 passed in 129.04s` |
| AC-30 | #30 Run Bundle 与 1.921530 ms Observation 保持冻结 | PASS | tree `b7ea6484…`; manifest `4b68b99…`; benchmark `05aecc12…`; README `6cc412c1…`; `1921530.0 ns`; base diff empty |
| AC-UNKNOWN | 未闭合证据不得伪造 numeric 结论 | PASS | #50 v5 为 `structured-unknown`，公开 schedule/gap/ratio/efficiency 全 `null`，边界逐项保存 |
