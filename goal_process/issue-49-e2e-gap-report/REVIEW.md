# 中文预测—实测报告增量 Review

- Fixed point: `7755b003762d44d6b2dbc739a206a54017337342`
- Final Run Bundle: `issue49-20260814T0730Z-e2e-gap-report-v12`
- Standards reviewer: PASS
- Spec reviewer: PASS
- NPU: 未运行；本增量只重放冻结 #30/#47/#48 证据

## 首轮 MUST-FIX 与关闭情况

1. v1alpha2 来源重放原为可选：已改为 writer 与公共 verifier 均强制要求
   derivation、非空 source lineage 与 supersedes；删除 manifest source lineage 的复制攻击
   测试会 fail closed。
2. 跨来源 identity 未闭包：已锁定并重放 #30 execution contract、Cost IR、hardware
   backend，#47 baseline/decomposition 与 #48 schedule authority；严格核对 Shape、dtype、
   candidate、Hardware Cohort、Completion Boundary 与 #30 manifest 的双上游锚点。
3. observation component policy 可伪造：已固定并验证版本、用途与方法；Grade 与
   `permitted_use` 强绑定。
4. Authority Result 只做部分比较：已改为从 #47/#48 精确重建 predicted/observed
   side 的完整字典，覆盖 status、bound/reason、items、accounting、quality、evidence refs、
   boundaries、下一步测量和 side provenance。
5. 中文页面缺固定结构：已补运行身份、E2E 等级/阶段/用途、贡献图、模块汇总、两侧
   TOP10、联合排名/阶段、精确平账、residual/overlap、52 叶下钻、下一轮建议与产物链接。
6. 一键命令会覆盖冲突：已改为默认微秒级 UTC 唯一 Run ID，并在生成后自动执行公共
   verifier。

## 最终验证

- Focused `tests/test_gap_report.py`: 14 passed
- Affected report/verifier/frontier tests: PASS
- Full suite after主要 review fixes: 692 passed
- `compileall`: PASS
- `git diff --check`: PASS
- v12 `verify-run`: PASS, 4/4 artifacts
- v6 `verify-run`: PASS
- v6 manifest SHA-256: `d582877c0f095e5a8a918a28e2888c71ce1b28ea7746738cac8f62fbc7f1ea10`
- v6 HTML SHA-256: `36fdc7a22cfafd7065577e582a11948204fa44eb3e786e78c0f8f9f71659b2c4`

## 非阻断判断

当前公共模块包含两层 Demo 专属的 52 叶、固定 Shape 和 reason 文案；未来扩展其他
Benchmark Case 时应提取版本化 case policy。本次 v12 的身份和用途已明确锁定，因此不
影响本次验收。
