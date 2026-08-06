# C005 稳定计时窗口结果

固定 C004 的 500 warmup、每 sample 5 个 window 和 20 个 sample，只将每个 window 的操作组数从 50 增加到 500。

| 设备 | sample median | sample IQR | IQR / median | 3% 门禁 |
|---|---:|---:|---:|---:|
| CPU | 476,684.54 ns | 7,677.71 ns | 1.611% | PASS |
| MPS | 272,522.25 ns | 856.65 ns | 0.314% | PASS |

## 最终 M1 采样结论

1. warmup 必须按实际操作组计数；当前探针使用 500 组。
2. timed window 要足够长，当前 CPU 每 window 约 0.24 秒、MPS 约 0.136 秒。
3. 一个统计 sample 由 5 个完整 window 的 median 构成；原始 window 全量保留。
4. 门禁计算 20 个 sample 的 IQR/median，不删除异常值。
5. CPU 使用 PyTorch 当前默认 intra-op=4、interop=10；后续 Run Manifest 必须显式记录线程口径。

CPU 原始数据中出现一个归一化 `1,074,540.58 ns` 的完整 window 尖峰，仍被保留；其组内 median 未被单点改变。这证明通过的是预注册稳健统计协议，而不是“挑好看的数据”。
