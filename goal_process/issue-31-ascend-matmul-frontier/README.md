# Issue 31：Ascend NPU MatMul Operator Frontier

本目录保存 ticket #31 的可回放实现、真实 Ascend 910B2 测量证据，以及由这些证据资格化得到的最小 Capability Surface。权威结果是 `evidence/runs/issue31-operator-frontier-v1`；其 manifest 固定了全部 source Run Bundle 的相对路径与 SHA-256，因而候选、合同、cohort 或历史文件发生变化时，验证会失败而不是复用旧 Surface。

## 资格化结果

- Hardware Validity Cohort：`ascend-npu-23b93a89d5fecc79`
- Baseline Timing Lane：`torch.npu.Event.elapsed_time`，end-event synchronize 后再 device synchronize
- 执行合同：float32、row-major contiguous、PyTorch eager、100 warmup、100 repetitions、每个原始样本包含 100 次候选执行并取均值；不剔除任何原始样本
- 搜索候选：`torch.matmul` 与 `torch.matmul.k-split-2`，每个候选在 256 和 512 两个 exact Shape 上各有 3 个独立进程会话
- best-of-correct：`torch.matmul` 在两个 Shape 上均胜出
- 独立 holdout：每个 Anchor 3 个独立进程会话；256 的会话中位数为 16327.5、17190.5、16222.0 ns，512 为 16224.5、16437.5、16331.5 ns
- Anchor：256 和 512 均为 `QUALIFIED + ACTIVE`
- Validated Shape Regime：仅 `square MatMul s=256..512`，使用独立的 384 Shape 三会话证据确认
- 查询：512 为 `exact_anchor`，384 为 `interpolated`，640 为 `unknown/outside_validated_domain`

Surface 采用一维、单 cell、分段线性的有效速率插值，并分别保存 Anchor、插值与仪器分辨率的不确定度分量。它不外推，也不跨 dtype、layout、alignment、执行模式、候选家族或 Hardware Cohort。

## 失败历史

早期测量没有被静默丢弃：

| 批次 | 会话 | valid | quarantined | 处理 |
|---|---:|---:|---:|---|
| search v1 | 12 | 6 | 6 | 计时噪声超限，拒绝资格化 |
| search v2 | 27 | 18 | 9 | 重试仍不能形成无选择偏差的完整稳定批次，拒绝资格化 |
| holdout v2 | 6 | 2 | 4 | 独立 holdout 不完整，拒绝晋级 |
| search v3（inner=20） | 12 | 9 | 3 | `torch.matmul` 部分会话 IQR 达 11.73%–24.02%，拒绝资格化 |
| search v4（inner=100） | 12 | 12 | 0 | 完整通过，作为唯一 search source cohort |
| holdout v4 | 6 | 6 | 0 | 完整通过 |
| confirmation v4 | 3 | 3 | 0 | 完整通过 |

最终 v4 cohort 的原始 samples、summary、correctness、candidate identity、execution contract、environment 和 manifest 都保存在 `evidence/runs/`。v1–v3 的完整失败 bundles 留存在远端隔离工作区 `/home/t00906153/GroundUpScale-issue31-20260811/evidence/runs/`，本表固化其聚合结果；最终资格化只引用完整的 v4 source cohort，绝不从失败批次中挑选会话。

## 回放

```bash
uv run groundupscale verify-run \
  goal_process/issue-31-ascend-matmul-frontier/evidence/runs/issue31-operator-frontier-v1

uv run groundupscale diagnose \
  goal_process/issue-31-ascend-matmul-frontier/evidence/runs/issue31-operator-frontier-v1 \
  --json
```

重新采集时，在 A2-AK-225 的隔离工作区外调用脚本，以免工作区根目录中的旧包遮蔽 `PYTHONPATH`：

```bash
cd /home/t00906153
/home/t00906153/GroundUpScale-issue31-20260811/goal_process/issue-31-ascend-matmul-frontier/collect_frontier_evidence.sh search-v4
/home/t00906153/GroundUpScale-issue31-20260811/goal_process/issue-31-ascend-matmul-frontier/collect_frontier_evidence.sh holdout-v4 torch.matmul
/home/t00906153/GroundUpScale-issue31-20260811/goal_process/issue-31-ascend-matmul-frontier/collect_frontier_evidence.sh confirmation-v4 torch.matmul
```
