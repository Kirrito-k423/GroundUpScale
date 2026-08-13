# Issue #39 final execution report

## Ticket ledger

| Issue | Branch / final child commit | NPU | Authority / outcome |
|---|---|---|---|
| #41 | `codex/issue-41` / `349f4de` | No | model E2E contract；52 leaves；committed complete/unknown synthetic bundles，仅 contract，不可 promotion |
| #42 | `codex/issue-42` / `2d78edd` | Yes | transformer-matmul v4；0/5 structured unknown：每域缺第二 eligible candidate，另四域保留 repeatability 缺口 |
| #43 | `codex/issue-43` / `50073bf` | Yes | `issue43-…rmsnorm-frontier-unknown-v3`；0 qualified sources / 7 phases missing |
| #44 | `codex/issue-44` / `621659a` | Yes | `issue44-20260813T1945Z-softmax-frontier-unknown-v2`；exp/sum/normalize 缺 real-chain operand evidence |
| #45 | `codex/issue-45` / `3b62916` | Yes | five v2 frontiers；add-residual 与 mul-attention-scale qualified；其余三域 unknown |
| #46 | `codex/issue-46` / `f4d647b` | No | alias/materialization replay seam；无真实 runtime audit/timing 时权威 unknown |
| #47 | `codex/issue-47` / `c8771a5` | No | `issue47-ascend-observed-decomposition-20260813-v1`；exact paired ablation/device timeline unavailable |
| #48 | `codex/issue-48` / `4463b51` | No | `issue48-20260814T0002Z-schedule-frontier-unknown-v2`；52 leaves、58 boundaries、三种 reference；Observation 1,921,530 ns 独立 known |
| #49 | `codex/issue-49` / `ada0452` | No | `issue49-20260814T0345Z-e2e-gap-report-v6`；0 gap rows，评分/诊断 unavailable |
| #50 | `codex/issue-50` / `c5a5a0a` | Yes | holdout v2 24/24 PASS；acceptance v5 3/3 PASS；最终 structured unknown |

所有票的 Standards 与 Spec review 均最终 PASS。集成提交保留每票原子历史；集成分支另有必要的语义兼容修复 `06f4d1a`。

## Dependency frontier

1. GitHub native blocked-by 初始返回 #41–#47 无 blocker，故第一批只启动 #41–#47。
2. #48 的 blockers 为 #41–#46；六票全部 CLOSED 后才启动 #48。
3. #49 的 blockers 为 #47、#48；两票全部 CLOSED 后才启动 #49。
4. #50 的唯一 blocker 为 #49；#49 CLOSED 后才启动 #50。
5. NPU 排队始终只由 host flock 管理，从未写入 issue dependency。

## Final verifier and test results

- Final full suite: `687 passed in 129.04s`.
- #50 holdout v2: `24/24` artifacts PASS.
- #50 acceptance v5: `3/3` artifacts PASS.
- #48 v1/v2、#49 historical v2/current v6 均保持 immutable 且可回放。
- 远端 host lock 终态：free；owner absent。

## Exact final evidence boundary

独立 holdout v2 的真实 E2E observation 为 `1,927,420 ns` median、`11,340 ns` IQR，但最终 acceptance 不发布 numeric schedule/gap/ratio/efficiency，原因是：

- #48 的 52 个 leaves 与 6 个 schedule effects 尚有 mandatory evidence gaps；
- #47 observed decomposition 缺 exact-identity paired ablation 与 usable device timeline；
- #47/#49 的 Completion Boundary 与 holdout/final contract 不同；
- holdout environment preflight 未采集；warmup convergence policy 未满足；execution contract 为 unsupported；
- #42、#43、#44、#45、#46 的 operator 级缺口如 ticket ledger 所列。

因此最终 machine/human projection 同源输出 `structured-unknown`，相关数值均为 `null`，不存在 provisional 或伪造的 unknown evidence。
