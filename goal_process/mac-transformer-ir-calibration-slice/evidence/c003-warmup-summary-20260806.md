# C003 足量 warmup 结果

固定 C002 的 50 组同步计时窗口，只将 warmup 从 5 个操作组提高到 500 个。

| 设备 | 归一化 median | 归一化 IQR | IQR / median | 3% 门禁 |
|---|---:|---:|---:|---:|
| CPU | 465,935.42 ns | 15,538.74 ns | 3.335% | FAIL |
| MPS | 269,820.83 ns | 1,753.95 ns | 0.650% | PASS |

MPS 从 C002 的 7.288% 降到 0.650%，且没有前段单调下降，说明 warmup 不足的诊断成立。CPU 的 20 个 window 中出现 `628,511.66 ns` 与 `598,287.50 ns` 两个调度尖峰；后续保留所有原始 window，但用多个 window 的组内 median 定义一个统计 sample，再对 sample 做 IQR 门禁。
