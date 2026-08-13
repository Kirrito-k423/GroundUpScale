# Issue #42 Code Review

- **Fixed point:** `5a0958e75c2c9323d2494136b3b26e1d4ded2b67`
- **Reviewed commit:** `24ccb091d4f70150e5bb72153768972ac0123dc8`
- **Diff:** `git diff 5a0958e75c2c9323d2494136b3b26e1d4ded2b67...HEAD`
- **Spec:** GitHub Issue #42（无评论）

## Standards

1. **MUST-FIX** `transformer_matmul_frontier.py:342-380`：只验源 manifest 哈希，未递归 `verify_run_bundle(source_root)`；篡改 cost-ir 后源验证失败，派生 bundle 仍通过。违 ADR0032「hashes authoritative」、ADR0035 DEB-001/003。修：重推前验证源 bundle 及 kind/status/device/cohort。
2. **MUST-FIX** `transformer_matmul_frontier.py:372,423-443`：写 bundle-relative `path`，回放却按 cwd 解析，且产物依赖 bundle 外文件。违 ADR0032 self-describing/可搬移。修：按 `root/path` 解析并嵌入或内容寻址锁定来源。
3. **MUST-FIX** `run_bundle.py:1934-1952`：未校验 manifest status/cohort 与 qualification/source；伪改为 `qualified/forged` 仍 passed。违 CONTEXT Run Manifest validity/cohort、ADR0035 ANC-001/HWC-001。修：三方强制一致。
4. **SHOULD-FIX**（Primitive Obsession/Data Clumps）`transformer_matmul_frontier.py:149-165,294-311`：domain identity 以嵌套 dict 重复传递。修：typed identity/query。其余 smell 无命中。

## Spec

1. **MUST-FIX** `transformer_matmul_frontier.py:264-315`：查询被无条件写成 unknown，兼容证据也永远 0/5，且无 known latency/rate 路径。违反“为…每一种 MatMul domain 选择证据合格…Frontier”“覆盖完整…可供 E2E schedule 消费”。修复：按完整 identity 命中 qualified Surface/exact Anchor，写 latency，并仅由 latency 派生 rate。
2. **MUST-FIX** `transformer_matmul_frontier.py:199-260`：仅接受 Surface，不支持 exact Anchor，也未测试 #36 incomplete Surface。违反“#36…不被当作 qualified Surface，但不阻止…exact Anchor”。修复：加入两类证据及 #36 回归。
3. **MUST-FIX** `transformer_matmul_frontier.py:125-196`：仅遍历 Cost IR，且硬编码 candidate/runtime；违反“从 Model Spec、Workload Spec、Execution IR 和 indexed Stable Paths 得到…完整清单”。修复：四源交叉核验，分歧即拒绝。
4. **MUST-FIX** `tests/test_transformer_matmul_frontier.py:101-148`：只有旧 q-proj 回放，无五域独立 search/holdout 或新不可变 Run Bundle。违反“新采集证据使用独立…holdout identity”。当前 q-proj 拒绝与每洞 minimum-next-measurement合格，但不足以验收。

## Summary

Standards：3 MUST-FIX、1 SHOULD-FIX；最严重为派生 verifier 可绕过 source artifact digest。Spec：4 MUST-FIX；最严重为实现永远 0/5 且缺真实 NPU search/holdout，不能完成 Issue #42。

## Resolution

- Standards 1：派生 verifier 先递归验证 source Transformer Run Bundle，且检查 kind/status/device/cohort 后才重推 inventory。
- Standards 2：全部 lineage path 统一相对当前 bundle root 解析；最终 evidence 与 source digest 均可搬移回放。
- Standards 3：Run verifier 强制 manifest、qualification、inventory/anchor 的 status/cohort 三方一致。
- Standards 4：完整 domain identity 以内容寻址 digest 固定，并在 case、candidate、execution contract 三处一致性校验；保留 dict 序列化边界以匹配 Run Bundle JSON 公共协议。
- Spec 1/2：增加 qualified exact Anchor known 路径，latency primary、rate 仅由 declared work/latency 派生；#36 incomplete Surface 显式拒绝且不遮蔽 exact Anchor。
- Spec 3：Model Spec、Workload Spec、Model IR、Semantic IR、Cost IR 五源交叉核对，任一分歧 fail closed；NPU measurement case 直接从冻结 inventory 构造完整 batch/transpose/layout contract。
- Spec 4：权威锁内 session `issue42-20260813-v1` 在固定 cohort/device 上完成 30 个独立 Run Bundles（每域 3 search + 3 holdout），全部 verifier PASS；一个域 qualified，四域按预注册 repeatability 边界发布可递归验证的 structured unknown，不追加实验轮次。

### Authoritative evidence

- `evidence/issue42-20260813-v1/session-metadata/preregistration.json`
- `evidence/issue42-20260813-v1/session-metadata/lock-owner-start.txt`
- `evidence/issue42-20260813-v1/session-metadata/session-result.json`
- `evidence/issue42-20260813-v1/artifact-store/runs/issue42-issue42-20260813-v1-transformer-matmul-frontier-final`

### Verification

- Focused review suite：`53 passed in 20.36s`
- Full suite：`561 passed in 93.77s`
- Evidence replay：本地 `37/37` Run Bundles verifier PASS（30 measurement、5 exact Anchor、2 Frontier）
- Compile：`python -m compileall -q src tests ...` PASS
- `git diff --check`：PASS

待原 Standards/Spec reviewer 定向复审。
