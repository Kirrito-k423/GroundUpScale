# 交付账本

| ID | 交付物 | 目标位置 | 状态 | 证据/版本 |
|---|---|---|---|---|
| D-01 | Python 包、CLI、依赖与锁文件 | `src/groundupscale/`、`pyproject.toml`、锁文件 | WIP | M1：包骨架、probe CLI、Python/PyTorch/NumPy 锁定；待最终 CLI |
| D-02 | YAML Specs 与 Schema | `specs/`、`src/groundupscale/schemas/` | DONE | M2：8 类 strict Schema、CPU/MPS 完整 YAML Plan、15 tests |
| D-03 | IR、编译器、Cost Lowerer 与 provenance | `src/groundupscale/ir/`、`src/groundupscale/compiler/` | WIP | M2：三层 IR + SemanticCompiler + provenance；Cost Lowerer 待 M3 |
| D-04 | 两层模型、CPU/MPS runner 与插桩 Adapter | `src/groundupscale/benchmark/` | NOT_STARTED | 待 M4 |
| D-05 | 测试与公共 CI | `tests/`、`.github/workflows/` | WIP | M2：15 tests 覆盖 strict Spec、repeat、identity、SemanticIR、CLI；CI 待 M6 |
| D-06 | Run Bundle 与校准证据 | `.groundupscale/runs/`、`evidence/` | NOT_STARTED | 待 M4/M5 |
| D-07 | 中文运行手册与最终报告 | `docs/`、`FINAL-REPORT.md` | NOT_STARTED | 待 M6 |
| D-08 | Goal 过程证据 | `goal_process/mac-transformer-ir-calibration-slice/` | WIP | M1/M2：C001–C008 与 milestone reports |
