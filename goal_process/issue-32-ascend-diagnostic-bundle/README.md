# Ticket #32：真实 Ascend 910B2 四轴 Diagnostic Bundle

本目录只实现 GitHub ticket #32。它把 #29 的 Resource Physical Floor、#31 的 ACTIVE Operator Frontier、显式单节点 Schedule，以及 #30 的真实 Transformer Q 投影 Observation 对齐到一个可摘要校验的 Run Bundle；不会创建或降低下一张票所需的 Capability Surface。

权威 Bundle 位于：

`evidence/runs/issue32-ascend-910b2-diagnostic-v1`

## 真实证据

- 硬件 cohort：`ascend-npu-23b93a89d5fecc79`，设备 `Ascend910B2`，逻辑设备 `npu:0`。
- Resource Physical Floor：`13,998.514874921928 ns`，仅表示可能不可达的资源下界，不作为完整时延预测。
- Operator Achievable Frontier：复用 #31 的 512 Shape、`torch.matmul` ACTIVE holdout Anchor，`16,331.5 ns`。
- Observation：复用 #30 的真实 Q 投影 baseline，`44,507.6295 ns`。
- #32 在三个独立进程中各采集 20 个 device-event 样本；普通实现三次均通过正确性，最大绝对误差 `0.0001220703125`。
- 注入 `+0.01` 输出偏差的负对照三次均失败，每次有 `81,259` 个超容差元素；该受控反例用于证明只有可复现的直接正确性/契约证据才能得到 `confirmed_bug`，并不宣称 PyTorch 或 torch-npu 存在缺陷。

Q 投影的成对诊断把 standalone、dispatch、copy、sync 和 profiling 变体保留为原始 session 证据。exclusive ledger 对独立 session 的差分分量执行一个有记录的公共归一化，使 cohort 中位数严格守恒；原始变体样本未被改写。`profiling` 只用于测量插桩开销。远端 profiler 报告导出状态不完整，因此本票不声称获得了可用的导出算子时间线。

## 复现

在仓库根目录运行：

```bash
python -m groundupscale.cli verify-run goal_process/issue-32-ascend-diagnostic-bundle/evidence/runs/issue32-ascend-910b2-diagnostic-v1 --json
python -m groundupscale.cli diagnose goal_process/issue-32-ascend-diagnostic-bundle/evidence/runs/issue32-ascend-910b2-diagnostic-v1
```

需要从原始证据重新构建时，先在不含该固定输出目录的干净副本中运行 `build_diagnostic_bundle.py`；Builder 对固定 `run_id` fail closed，已有目录时拒绝覆盖。CLI 报告可从资格策略、ACTIVE Anchor、候选搜索和源 Run Bundle，继续钻取到 Shape Probe、Ablation、exclusive ledger、Gate 与直接缺陷证据。任一 manifest 制品缺失或摘要不匹配时，`diagnose` 会拒绝派生结果。

## 远端执行记录

采样机为 `A2-AK-225`（`192.168.9.225`），远端用户 `root`，主机名 `localhost.localdomain`，隔离工作目录为 `/home/t00906153/GroundUpScale-issue32-20260811`。命令、session/process identity、告警和最终连通性复查见 `evidence/remote-execution.json`。本票没有启动本地服务。
