# C020 受控 Mac 环境门禁证据

## 结论

GroundUpScale 已把供电、平台、热状态、系统负载和竞争进程变成版本化、
fail-closed 的测量前置门禁。它不会修改原校准合同，也不替代 Run 内每个 Case
的 `IQR/median <= 3%` 检验。

## Policy

`local-apple-silicon-v1` 固定要求：

| 检查 | 要求 | unknown |
|---|---|---|
| 平台 | Darwin + arm64 | 拒绝 |
| 供电 | AC power | 拒绝 |
| 热状态 | thermal/performance nominal | 拒绝 |
| 系统负载 | 1 min load / logical CPUs <= 0.25 | 拒绝 |
| 竞争进程 | coordinator/ancestor 以外单进程 <= 25% CPU，3×1 s | 拒绝 |

## 真实观察

2026-08-06T20:25:34+08:00 的真实 CLI 观察：

- PASS：Darwin/arm64、AC、thermal nominal；
- FAIL：归一化 1 分钟负载 `0.36318359375`；
- FAIL：`mediaanalysisd` 最大 `58.1% CPU`；另有两个外部 `python3.11`
  进程最大 `53.3%` 和 `35.5%`；
- 最终：`eligible=false`，reason codes 为 `load-above-policy` 和
  `competing-process-above-policy`。

因此本轮没有启动新的正式 benchmark。该结果支持“先前连续采样存在后台竞争
干扰”的判断，但尚不能证明新 policy 足以产出 5 个有效 holdout；那需要在
环境实际合格后建立完全独立的新 cohort。

## 治理闭环

- `groundupscale run --require-valid-environment` 在 benchmark 前拒绝不合格报告；
- 通过报告写入 `resolved/environment.json`，Manifest 标记
  `environment_validity: passed`；
- 普通开发 Run 标记 `not-required`；
- `fit-calibration` 和 `validate-calibration` 均拒绝没有 passed preflight 的
  Bundle，因此 C012–C018 的旧未受控 Run 不能被复用进新 cohort；
- 全量测试：`42 passed`；真实 `run --require-valid-environment` 返回
  `run-rejection/v1alpha1` 且确认目标 Run Bundle 不存在。
- 实现提交：`cb77dd0764040db7b2b316bcf2e544ae4b948c2d`；公共 Compiler CI
  `31102277844` Success（60 s）。
