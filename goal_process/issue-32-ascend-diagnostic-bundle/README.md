# Ticket #32：真实 Ascend 910B2 四轴 Diagnostic Bundle

本目录只实现 GitHub ticket #32。它把 #29 的 Resource Physical Floor、#31 的 ACTIVE Operator Frontier 与已校准 Capability Surface、显式单节点 Schedule，以及 #30 的真实 Transformer Q 投影 Observation 对齐到一个可摘要校验的 Run Bundle；不会创建、重拟合或降低下一张票所需的 Capability Surface。

权威 Bundle 位于：

`evidence/runs/issue32-ascend-910b2-diagnostic-v1`

## 真实证据

- 硬件 cohort：`ascend-npu-23b93a89d5fecc79`，设备 `Ascend910B2`，逻辑设备 `npu:0`。
- Resource Physical Floor：`13,998.514874921928 ns`，仅表示可能不可达的资源下界，不作为完整时延预测。
- Operator Achievable Frontier：复用 #31 的 512 Shape、`torch.matmul` ACTIVE holdout Anchor，`16,331.5 ns`；综合不确定区间直接由 #31 已校准 Surface 查询得到，单侧上限为 `107.2173 ns`，本票不新增未校准的重放不确定度。
- Observation：复用 #30 的真实 Q 投影 baseline，`44,507.6295 ns`。
- #32 保留三次独立 v7 Q 集成采集，并以另三个独立 v10 进程分别采集 K/V Stable Path 的 baseline 与 V 负对照；每个变体每进程包含 20 个 device-event 样本。Q/K/V 使用三组不同且逐 session 记录 SHA-256 的输入 identity，普通实现全部通过正确性。
- Diagnostic Trigger 不复用路径标签：Q 从 #30 的 `matmul-q-proj` 原始 Observation 重放，K/V 分别从 v10 的 `k_baseline`/`v_baseline` 原始 session 以“独立 session 中位数的中位数”重放；验证器会逐项核验 variant contract、Stable Path、semantic、lane、输入 identity、process identity 和样本聚合，并把每个 source artifact SHA-256 反查到远端执行记录。
- 注入 `+0.01` 输出偏差的负对照三次均失败，每次有 `81,259` 个超容差元素；该受控反例用于证明只有可复现的直接正确性/契约证据才能得到 `confirmed_bug`，并不宣称 PyTorch 或 torch-npu 存在缺陷。

Q 投影的成对诊断让 standalone、dispatch、copy、sync 和 profiling 累计变体使用同一个 2-D `torch.matmul` Frontier kernel；batch-one 输入/输出仅通过计时边界外的零拷贝 view 对齐语义。exclusive ledger 的每个非负 leaf 都由同一独立 session 的累计阶段中位数和此前阶段的单调 envelope 推导，并由验证器对已摘要校验的原始 session 重新计算；不缩放、不把派生值伪装成 raw sample。聚合后的非可加差异显式保留为非负 `unattributed` residual。`profiling` 只用于触发后的插桩开销诊断。V 负对照始终保存在 diagnostic lane，其 timing 不参与 Frontier/headroom，只以三次直接正确性失败支持受控 `confirmed_bug`。远端 profiler 报告导出状态不完整，因此本票不声称获得了可用的导出算子时间线。

## 复现

在仓库根目录运行：

```bash
python -m groundupscale.cli verify-run goal_process/issue-32-ascend-diagnostic-bundle/evidence/runs/issue32-ascend-910b2-diagnostic-v1 --json
python -m groundupscale.cli diagnose goal_process/issue-32-ascend-diagnostic-bundle/evidence/runs/issue32-ascend-910b2-diagnostic-v1
```

需要从原始证据重新构建时，先在不含该固定输出目录的干净副本中运行 `build_diagnostic_bundle.py`；Builder 对固定 `run_id` fail closed，已有目录时拒绝覆盖。Builder 会先完整验证 #29/#30/#31 源 Run Bundle，输出 Bundle 再固定其源 manifest 路径与摘要，并复制本票实际读取的源制品。CLI 报告可从资格策略、ACTIVE Anchor、候选搜索和源 Run Bundle，继续钻取到 Shape Probe、Ablation、exclusive ledger、Gate 与直接缺陷证据。任一 manifest 制品缺失或摘要不匹配时，`diagnose` 会拒绝派生结果。

## 远端执行记录

采样机为 `A2-AK-225`（`192.168.9.225`），远端用户 `root`，主机名 `localhost.localdomain`。Q 集成证据位于隔离目录 `/home/t00906153/groundupscale-issue32-v7`，K/V 语义证据位于 `/home/t00906153/groundupscale-issue32-v10`。命令、六个 session/process identity、告警和最终 NPU 健康复查见 `evidence/remote-execution.json`。本票没有启动本地服务。
