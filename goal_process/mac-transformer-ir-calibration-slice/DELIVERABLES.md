# 交付账本

| ID | 交付物 | 目标位置 | 状态 | 证据/版本 |
|---|---|---|---|---|
| D-01 | Python 包、CLI、依赖与锁文件 | `src/groundupscale/`、`pyproject.toml`、锁文件 | DONE | M5：probe/compile/run/verify/explain/fit/validate/promote CLI，锁定环境，33 tests |
| D-02 | YAML Specs 与 Schema | `specs/`、`src/groundupscale/schemas/` | DONE | M2：8 类 strict Schema、CPU/MPS 完整 YAML Plan、15 tests |
| D-03 | IR、编译器、Cost Lowerer 与 provenance | `src/groundupscale/ir/`、`src/groundupscale/compiler/` | DONE | M3：四层 IR、可注册 CostRule、依赖/公式/bytes/provenance，21 tests |
| D-04 | 两层模型、CPU/MPS runner 与插桩 Adapter | `src/groundupscale/benchmark/` | DONE | M4：reference、5 Case benchmark、60-span trace、memory observer、alignment/explanation |
| D-05 | 测试与公共 CI | `tests/`、`.github/workflows/` | IN_PROGRESS | C019：34 tests 目标、Linux deterministic workflow、CI security contract；待远端绿灯 |
| D-06 | Run Bundle 与校准证据 | `.groundupscale/runs/`、`evidence/` | PARTIAL_BLOCKED | M4 bundles 完整；M5 candidate/validation failure 有证据，因 valid holdout<5 未生成 active profile |
| D-07 | 中文运行手册与最终报告 | `docs/`、`FINAL-REPORT.md` | IN_PROGRESS | C019：本地 Mac 运行/校准/安全手册已完成；待最终报告 |
| D-08 | Goal 过程证据 | `goal_process/mac-transformer-ir-calibration-slice/` | DONE_TO_CURRENT_STATE | C001–C019 全部预注册、失败、命令、里程碑和升级证据保留 |
