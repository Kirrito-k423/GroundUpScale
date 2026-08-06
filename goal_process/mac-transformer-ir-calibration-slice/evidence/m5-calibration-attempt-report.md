# M5 校准尝试报告：模型误差通过，留出有效数量失败

## 结论

受控 calibration fit/validate/promote 框架已实现并通过测试，但真实 MPS Profile **未晋升**。3 个拟合 Run 合格；7 个独立 holdout 中只有 3 个满足预先确认的 `IQR/median<=3%`，不足最低 5 个。所有有效 holdout 的 latency 与 memory 误差均在 5% 内，说明当前失败是测量有效性，不是已观察到的模型精度失败。

## Candidate

- Profile ID：`1f66d803cc23acc3559e676f1e877aa484f8cac6f43cb016c11b315120194eb7`
- 设备/cohort：MPS / Apple M4 + 锁定 OS、PyTorch、thread、protocol。
- fit：`fit-01..03`，最大 noise `2.870%`。
- 内存：base `54,534,144 B`，observed/calibrated `69,214,208 B`；基础值未被覆盖。

## Holdout

| Run | 有效 | 最大 Case error | memory error | 隔离原因 |
|---|---:|---:|---:|---|
| holdout-01 | 是 | 2.477% | 0% | — |
| holdout-02 | 否 | 2.147% | 0% | Softmax noise 3.309% |
| holdout-03 | 是 | 3.715% | 0% | — |
| holdout-04 | 是 | 1.370% | 0% | — |
| holdout-05 | 否 | 12.979% | 0% | Softmax noise 16.148%、E2E 3.239% |
| holdout-06 | 否 | 1.639% | 0% | MatMul noise 5.308% |
| holdout-07 | 否 | 7.678% | 0% | MatMul/Softmax/E2E noise 超标 |

验证结果：`valid_holdout_runs=3 < 5`，`passed=false`。Profile 保持 candidate 且仅存在本地 `.groundupscale/calibration/`；仓库 `evidence/calibrations/` 中没有伪造的 active profile。

## Governance 证明

- 混合 device/cohort/fingerprint/Case 集显式拒绝。
- fit 与 holdout Run ID 重叠显式拒绝。
- noisy fit 显式拒绝；noisy holdout 保留并隔离。
- 只有 validation PASS 才允许 `promote_calibration`。
- Profile 精确锁定适用域，域外拒绝并回退 uncalibrated，不宣称泛化。
