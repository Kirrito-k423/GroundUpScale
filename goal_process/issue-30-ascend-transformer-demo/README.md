# Issue #30：在 Ascend NPU 上运行两层 Transformer Demo

本目录固化 GroundUpScale 两层因果 Transformer 在真实 Ascend 910B2 上完整执行的
不可变 Run Bundle。它证明与 M4 路径相同的 Model Spec、Workload Spec、固定 Shape、
52 个 Semantic Operation 和 Stable Path 已在 `npu:0` 执行，而不是由 CPU-backed
fake runtime 代替硬件验收。

## 范围

- 目标设备：A2-AK-225 的单张 Ascend 910B2，逻辑设备 `npu:0`。
- Hardware Cohort：`ascend-npu-23b93a89d5fecc79`。
- 运行时：Python 3.11.14、torch 2.7.1、torch_npu 2.7.1。
- 输入/输出：float32、contiguous、Shape `[1, 512, 512]`。
- 执行：20 次 warmup、20 个样本、每样本 5 个窗口，主计时器为
  `torch.npu.Event.elapsed_time`。
- 正确性：CPU float32 oracle，`atol=0.001`、`rtol=0.001`，禁止 CPU fallback。

Operator Frontier qualification、跨硬件诊断和为非 MatMul 算子补齐物理下界不属于
本票。

## 真实运行

2026-08-11 以 root 设备权限在隔离目录
`/home/t00906153/GroundUpScale-issue30-20260811` 执行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=/home/t00906153/GroundUpScale-issue30-20260811/src

/home/miniconda3/envs/lmz_pt27py311/bin/python -m groundupscale.cli run \
  specs/plans/ascend-npu-transformer-demo.yaml \
  --repository-root . \
  --artifact-store goal_process/issue-30-ascend-transformer-demo/evidence \
  --run-id ascend-910b2-transformer-demo-20260811-v1 \
  --samples 20 --warmup 20 --windows-per-sample 5 \
  --target-window-ms 100 --json
```

Run Manifest 状态为 `completed`，compatibility、benchmark、trace 和 comparison 阶段
均完成。正确性检查覆盖 52 个语义叶子，全部输出设备为 `npu:0`，CPU fallback 为
`false`；最大绝对误差为 `9.5367431640625e-07`，最大相对误差为
`7.047830763440288e-07`。

| Benchmark Case | scope | median | IQR / median |
|---|---|---:|---:|
| `matmul-q-proj` | operator | 44,507.630 ns | 1.236% |
| `rmsnorm-input` | operator | 156,569.897 ns | 0.739% |
| `softmax-attention` | operator | 41,657.671 ns | 0.559% |
| `transformer-layer` | module | 1,028,740 ns | 0.475% |
| `two-layer-prefill` | e2e | 1,921,530 ns | 0.685% |

Baseline Timing 与 Diagnostic Profiling 分开保存。内存证据分别记录逻辑张量 live
set 峰值 69,214,208 B、框架 NPU allocator 峰值 93,339,648 B（本 run 前已 reset）
和进程 RSS 最大观测点 1,905,008,640 B；传输证据记录权重/缓冲区与输入 H2D、输出
D2H，共 9 条记录。

Comparison 为诚实的 `partial-base-prediction`：当前只有 MatMul 有实测物理下界，其余
算子保持 unknown，不把缺失的 floor 冒充完整 duration，也不在本票顺手实现下一阶段。

## 不可变证据与回放

权威 Run Bundle 位于：

`evidence/runs/ascend-910b2-transformer-demo-20260811-v1/`

Manifest 声明 22 个 artifact role，并为每个 artifact 保存 SHA-256 和 producer
lineage。远端运行控制台、远端独立校验和解释结果分别保存在
`evidence/remote-run-console.log`、`evidence/verify-run.json` 和
`evidence/explain.json`。Mac 无需安装 `torch_npu` 即可独立回放：

```bash
uv run groundupscale verify-run \
  goal_process/issue-30-ascend-transformer-demo/evidence/runs/ascend-910b2-transformer-demo-20260811-v1 \
  --json

uv run groundupscale explain \
  goal_process/issue-30-ascend-transformer-demo/evidence/runs/ascend-910b2-transformer-demo-20260811-v1 \
  --json
```

`verify-run` 的冻结结果为 `passed=true`、`artifact_count=22`、`failures=[]`。
