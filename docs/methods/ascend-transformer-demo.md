# Ascend NPU 两层 Transformer Demo

本入口在单张 Ascend 910B2 的 `npu:0` 上执行与 M4 路径相同的两层
Transformer Model Spec、Workload Spec、固定 Shape、52 个 Semantic Operation 和
Stable Path。它只接受 Analysis Plan 锁定的 Hardware Cohort；不会把其他机器的结果
静默并入同一 Profile。

## 运行

环境必须安装互相匹配的 PyTorch 与 `torch_npu`，并能看到目标 NPU：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/src"

python -m groundupscale.cli run \
  specs/plans/ascend-npu-transformer-demo.yaml \
  --repository-root . \
  --run-id ascend-npu-transformer-demo-$(date +%Y%m%d-%H%M%S) \
  --samples 20 --warmup 20 --windows-per-sample 5 \
  --target-window-ms 100 --json
```

Baseline Timing 使用 NPU Event 作为主计时器，warmup 在计时区外完成，并分别记录
host launch、device completion wait 和 host completion。逐模块 hook 只属于独立的
Diagnostic Profiling lane；它的时间不能进入前沿或校准证据。

## 成功证据

完成的不可变 Run Bundle 除通用编译、预测、benchmark、trace、comparison 和 HTML
报告外，还包括：

- adapter capability、hardware cohort 与 preflight；
- CPU float32 oracle 以及输入、输出、52 个叶子输出的 device/dtype/shape/stride
  审计；
- 权重、输入、输出传输记录和显式完成边界；
- 逻辑张量 live set、框架设备分配器内存与进程 RSS 三种独立口径；
- execution contract，以及可校验的 artifact SHA-256。

复验和解释不需要重新执行 NPU：

```bash
uv run groundupscale verify-run .groundupscale/runs/<run-id> --json
uv run groundupscale explain .groundupscale/runs/<run-id> --json
```

## 失败保留

缺少 `torch_npu`、cohort 不匹配、算子不支持、CPU fallback、dtype/layout 替换或
执行异常都会返回非零状态，并发布 `blocked` 或 `compatibility-failed` Run Bundle。
执行边界会把 torch_npu 对不支持 eager 算子的
[CPU fallback 警告](https://github.com/Ascend/pytorch/blob/v2.7.1/docs/zh/user_guide/troubleshooting/troubleshooting_cases/unsupported_op_called.md)
提升为 `cpu-fallback-detected` 失败；仅检查最终输出位于 NPU 不足以证明中间没有
回落。
失败 Bundle 保留已解析输入、adapter 证据、已发生的传输和结构化 reason code，仍可
由 `verify-run` 校验；不会用 CPU 结果伪装成 NPU 成功运行。
