# Issue 45 Ascend elementwise Operator Frontiers

本目录限定 `Add`、`Mul` 与 `SiLU` 的五个 distinct execution domain。权威真实
NPU evidence 只能来自：

```text
GROUNDUPSCALE_ISSUE=45 ASCEND_RT_VISIBLE_DEVICES=0 \
/home/t00906153/.groundupscale/bin/with-ascend-lock <complete-session-script>
```

公共锁必须覆盖 correctness、search、measurement、independent holdout 与资格化前
验证组成的完整 session。每个 Run Bundle 的 `adapter/collection.json` 保存锁 owner、
Hardware Cohort、device visibility 和 collection 完成时间；session 级开始/结束快照另存
在 `evidence/session-metadata/<session-id>/`。

## 非权威尝试

2026-08-13 曾执行一次未通过公共 wrapper 的远端命令：

```text
ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=src \
/home/miniconda3/envs/lmz_pt27py311/bin/python -m pytest \
tests/test_issue45_ascend_elementwise_frontier.py -q
```

该命令以 `No module named pytest` 退出，未生成 measurement Run Bundle，且检查时
NPU 0 无运行进程、owner 文件不存在。无论是否实际初始化 NPU，这次尝试及任何可能
副产物均标记为 `invalid/non-authoritative`，不得进入资格化、Run Bundle 或验收。

首个合法锁会话 `issue45-20260813T2300Z`（18:28:07–18:37:33+08:00）完成了
30 个测量，但当时的 adapter 尚未把锁元数据内联到每个 measurement bundle，因此
该批次及其 `v1`/`v2` frontier 都不具备权威验收资格。它们不会作为下面结果的 source。

在最终会话完成后，第一次离线发布未设置
`TORCH_DEVICE_BACKEND_AUTOLOAD=0`，导入 PyTorch 时误触发 `torch_npu` backend
自动加载并因 ABI 错误退出；第二次离线发布推进至 SiLU 后因 source timing quality
quarantined 暴露 fail-closed seam 缺口。两次失败都未形成最终资格化结果；前四个残留
`v1` frontier 也不作为权威结果。

## 权威会话与结果

最终锁会话为 `issue45-20260813T1850Z`：

- 起止时间：`2026-08-13T18:46:19+08:00`–`2026-08-13T18:55:33+08:00`
- owner：`issue=45 pid=2132689 host=localhost.localdomain`
- Hardware Cohort：`ascend-npu-23b93a89d5fecc79`
- device visibility：`ASCEND_RT_VISIBLE_DEVICES=0`
- measurement：固定五域各 3 个 search + 3 个 independent holdout，30/30 verifier pass
- session metadata：远端 `evidence/session-metadata/issue45-20260813T1850Z/`
- measurement bundles：远端 `evidence/runs/issue45-*-issue45-20260813T1850Z-*`

最终纯离线 `v2` 资格化结果如下，五个 frontier Run Bundle 均通过公共 verifier：

| domain | status | latency / boundary |
| --- | --- | --- |
| add-residual | qualified | 14775.5 ns；标准不确定度 371.943880175491 ns |
| mul-attention-scale | qualified | 19646.0 ns；标准不确定度 187.11783275073847 ns |
| add-broadcast-mask | unknown | search/holdout session relative range 超过 10%；只需在相同锁定域重做独立 search/holdout |
| mul-mlp-gate | unknown | search/holdout session relative range 超过 10%；只需在相同锁定域重做独立 search/holdout |
| silu-mlp-gate | unknown | `search-01` 为 `session-dispersion-exceeds-policy`；只需替换该 invalid session |

最终 frontier 路径统一为远端：

```text
/home/t00906153/GroundUpScale-issue-45/goal_process/
  issue-45-ascend-elementwise-frontier/evidence/runs/
  issue45-<domain>-frontier-issue45-20260813T1850Z-v2
```

`unknown` 域没有数值锚点，且 exact-domain 以外的 shape/domain query 同样 fail closed；
调度器只能消费两个 `qualified` exact anchors。下一 evidence boundary 已写入各自
`frontier/qualification.json`，没有用额外采集或换卡扩大边界。
