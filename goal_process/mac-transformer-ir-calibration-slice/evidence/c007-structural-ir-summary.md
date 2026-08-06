# C007 结构 IR 结果

## 真实 YAML bundle

- Model：两层 pre-norm Transformer；B=1、S=512、H=512、NH=8、D=64、I=2048、FP32。
- Workload：Sequence 中一个 `ModelCall(model=two-layer-transformer, entrypoint=prefill)`。
- Analysis：FixedShape + FixedIterations + ObservationWindow。
- Deployment：CPU 与 MPS 两份 placement intent。
- Hardware/Fabric：当前 Apple M4 CPU/GPU 与本地节点。
- Benchmark：MatMul、RMSNorm、Softmax、单层、两层 E2E 五个强制 Case。

## 编译结果

| 产物 | 节点数 | fingerprint |
|---|---:|---|
| ModelIR | 59 modules（7 composite、52 primitive） | `6d597e5e8aeef46e3df9fce79ead23a305376905706cbdb36df009b44872016d` |
| WorkloadIR | 2 nodes（Sequence + ModelCall） | `785e13ee9b612563e3afd259553b961bebd2b0f361113ee2f97e674cf64f281b` |

`layer_0` 与 `layer_1` 的 Stable Path/Node ID 不同，Definition ID 相同。ModelIR root 的 repeat_call 已确定性展开成两条带 carry 的 call；WorkloadIR ModelCall 仍是叶子，不直接拥有模型子节点。

> C008 语义审计补充：为使 `prefill` 明确具有自回归 causal 语义，每层新增一个读取 mask buffer 的 Add；最终模型为 61 modules（7 composite、54 primitive），最终 fingerprint 见 M2 milestone report。上表保留 C007 当轮的真实中间结果。
