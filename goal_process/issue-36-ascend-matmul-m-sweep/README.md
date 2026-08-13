# Issue 36：有界 Ascend MatMul M-sweep

本目录保存 ticket #36 的锁定采集计划、可恢复采集脚本和停止结论。完整源
Run Bundle corpus 已迁移到由
`evidence/datasets/issue36-ascend-operator-frontier-corpus-v1.yaml` 索引的
content-addressed GitHub Release asset；仓库保留可离线验证和诊断回放的最小 unknown
Frontier Run Bundle：
`evidence/qualifications/issue36-bounded-collection-corpus-incomplete-v1`。
实验固定
Hardware Cohort `ascend-npu-23b93a89d5fecc79`、`N=512`、`K=512`、float32、
row-major contiguous、`torch.matmul`、PyTorch eager、100 warmup、100 repetitions、
100 inner iterations、baseline timing lane、固定 seed `20260812`，不选择性删除原始
samples。

## 真实执行结论

2026-08-12 在 A2-AK-225（远端 hostname `localhost.localdomain`，root，Ascend
910B2）执行了唯一一轮主 sweep。计划中的 24 个 M Shape
`1, 2, 4, 8, 16, 32, 64, 96, 127, 128, 129, 192, 255, 256, 257, 384, 511,
512, 513, 768, 1024, 1536, 2048, 4096` 均至少完成一个可验证 Run Bundle；没有增加
未声明 Shape，也没有执行补点轮。初步 median latency 从小 Shape 的约 15–19 μs，
到 M=2048 的约 19.6 μs、M=4096 的约 30.9 μs，说明 `M≈512` 不能写成当前
cohort 的已确认边界，更不能写成全局常量。

随后尝试补齐每个晋级点所需的三个独立进程会话。进行到大 Shape 时远端 SSH 入口开始
在 key exchange 前主动关闭新连接；已完成 Run Bundle 全部保留，未重写、未选择性删除，
也没有切换到另一台机器混合 Hardware Cohort。由于三独立会话和预声明的 8 个独立
validation Shape 尚未形成完整 corpus，本票按停止条件发布结构化 `unknown`，不发布
Ramp/Steady Surface、不声称 Shape Regime Boundary、不引入额外模型复杂度。

代表性查询的稳定结果：

| 查询 | 结果 | reason code |
|---|---|---|
| Ramp `M=128` | unknown | `bounded-collection-corpus-incomplete` |
| Steady hypothesis `M=1024` | unknown | `bounded-collection-corpus-incomplete` |
| exact hypothesis `M=512` | unknown | `bounded-collection-corpus-incomplete` |
| outside-domain `M=8192` | unknown | `bounded-collection-corpus-incomplete` |

## 回放与恢复

采集脚本发现既有 run_id 时先执行 `verify-run`，验证通过后跳过，因此只能补齐缺失会话，
不会覆盖已完成证据：

```bash
cd /home/t00906153/GroundUpScale-issue36-20260812
bash goal_process/issue-36-ascend-matmul-m-sweep/collect_bounded_m_sweep.sh main-replication
bash goal_process/issue-36-ascend-matmul-m-sweep/collect_bounded_m_sweep.sh holdout
bash goal_process/issue-36-ascend-matmul-m-sweep/collect_bounded_m_sweep.sh validation
```

恢复后仍只能使用同一个 Hardware Cohort。只有完整 corpus 满足 correctness、三独立进程、
稳定性和独立 holdout，才能通过公开 `OperatorFrontierBundleWriter` seam 生成 qualified 或
rejected Run Bundle。本票已经实现并测试了 qualified、rejected、unknown 三种 qualification
的不可变写入、验证和诊断回放；完整 corpus 同时通过 Error Budget 与 Ramp/Steady 边界证据
时发布 Surface，否则发布可回放 rejected，partial corpus 则稳定发布 structured unknown。

离线验证与回放：

```bash
uv run groundupscale verify-run \
  evidence/qualifications/issue36-bounded-collection-corpus-incomplete-v1 --json
uv run groundupscale diagnose \
  evidence/qualifications/issue36-bounded-collection-corpus-incomplete-v1 --json
```
