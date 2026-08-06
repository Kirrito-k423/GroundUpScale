# Goal 交接与决策包

## 当前结论

软件纵向切片、CPU/MPS correctness、IR/Cost 公式、Run Bundle、解释链、校准治理、可信本地入口和公共 CI 已完成。完整 Goal 尚未达成：CPU/MPS 的至少 5 个有效独立 holdout、active Calibration Profile 和校准证据下钻仍缺失。

C020–C021 连续三个 Goal 回合出现同一外部环境阻塞。最新 preflight：normalized one-minute load `0.407666>0.25`，`mediaanalysisd=89.4% CPU`，两个持续约 13 天的 autoresearch board 服务在采样窗口内均 `55.6% CPU`。未经授权不能停止用户进程，继续运行会违反已确认的测量合同，因此状态为 BLOCKED。

## 可立即使用的交付

- 公共仓库 `main`：实现提交 `cb77dd0`，审计提交 `3cb27e9`；42 tests。
- 公共 CI：GitHub Actions `31102467129` Success。
- 可信入口：`scripts/run-local-m4-evidence.sh`。
- 环境判定：`uv run groundupscale preflight --json`。
- 运行手册：`docs/runbooks/local-mac-calibration.md`。
- 当前审计：`FINAL-REPORT.md`、`evidence/c020-environment-preflight-summary.md`。

## 关键证据

- C018：3 个有效 MPS holdout 的最大 latency error `3.715%`，memory error `0%`，但有效数 `3<5`。
- C020：实现 fail-closed preflight，真实环境两次拒绝。
- C021：第三次同签名拒绝，normalized load `0.408`、最大竞争进程 `89.4% CPU`。
- `ACCEPTANCE.md`：7/12 完成，其余由校准证据阻塞。

## 已排除与未排除

- **已排除：** YAML/IR 编译错误、CPU/MPS 数值错误、MPS fallback、基础公式错误、fit/holdout 泄漏、公共 CI 回归。
- **未排除：** 合格环境下是否能连续得到至少 5 个噪声不超过 3% 的 holdout；这是恢复后的唯一主要实验问题。

## 复现方法

```sh
uv sync --locked --group dev
uv run pytest -q
uv run groundupscale preflight --json
```

只有 preflight exit 0 才运行：

```sh
scripts/run-local-m4-evidence.sh <new-cohort-tag>
```

不得复用 C012–C018 的未受控 Run；新的 fit 与 holdout Run ID 必须完全分离。

## 时间与费用

Goal effective meter 在 blocked 审计时为 `1,640,735 tokens / 11,892 seconds`。`RMB-Cost.md` 保持 estimate：原始 uncached input、cached input、output 分项以及最新价格/汇率未在本轮核验，不编造费用。

## 后续路径

- **路径 A（推荐）：** 临时停止两个 autoresearch board 服务，等待 `mediaanalysisd` 自然降载和 1-minute load 回落；preflight PASS 后采集全新 fit/holdout。操作后约需 20–40 分钟，证据等级 E1。
- **路径 B：** 保持服务运行，选择它们自然空闲且系统分析结束的时间恢复；耗时不确定，仍以 preflight PASS 为唯一启动条件。

## 所需决策

是否允许临时停止 C021 时的 PID 18974/18975。推荐允许；它们是长期本地 board 服务，不应由执行者未经授权停止。`mediaanalysisd` 不建议强杀，等待系统自行降载。用户恢复 Goal 后必须先重新解析 PID，不能盲用本文件中的旧 PID。
