# M1 里程碑报告：环境与本机可行性

## 结论

M1 已完成。Python 3.11.15 + PyTorch 2.13.0 在当前 Apple M4/macOS 15.7.4 上同时提供 CPU 与 MPS；MatMul、Add、RMSNorm、Softmax、SiLU、Mul、View、Transpose 均可运行，MPS 数值误差在声明容差内；同步计时和 framework-attributed MPS allocator 接口可用。

## 直接证据

- 可复现环境：`pyproject.toml`、`uv.lock`，`uv sync --python 3.11 --group dev` 成功。
- TDD：公开 `groundupscale probe` 契约先 RED 后 GREEN；最终 `pytest` 为 1 passed。
- 能力/正确性：`c001-probe-summary-20260806.md`。
- 噪声诊断：C001–C005 全部失败和通过结果均保留；最终 CPU 1.611%、MPS 0.314%。
- 内存：MPS current allocated delta 为 8,388,608 bytes，Driver allocation 单列；CPU RSS 明确标为进程级诊断。
- 环境收尾：补充并锁定 NumPy 2.4.6 后，PyTorch 启动不再产生缺少 NumPy 的告警；快速 CPU/MPS probe 通过。

## 证据等级

- H-01：E2，当前本机软件栈 CPU/MPS 可用。
- H-02：E2，当前 Shape 与稳健采样协议支持 3% 噪声门禁。
- H-03：E1，allocator 观测接口可用；真实模型 live-set 尚待 M4。
- H-04：E1，所有冻结原子操作可用；完整两层模型尚待 M2/M4。

## 未关闭风险

- M1 的操作组不是最终两层 Transformer，最终每个 Benchmark Case 必须独立确定 warmup/window 工作量。
- CPU RSS 不能冒充框架可归因峰值；AC-08 仍需 live-set 与 allocator 口径对齐。
- 当前结论只适用于锁定的软件栈与本机，不外推到其他 Mac/Shape。
