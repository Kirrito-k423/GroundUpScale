# C004 稳健 sample 结果

每个统计 sample 保留 5 个原始 timed window，并以组内 median 作为 sample 值；门禁计算 20 个 sample 的 IQR/median。

| 设备 | sample median | sample IQR | IQR / median | 3% 门禁 |
|---|---:|---:|---:|---:|
| CPU | 472,224.16 ns | 14,885.41 ns | 3.152% | FAIL |
| MPS | 281,513.34 ns | 699.59 ns | 0.249% | PASS |

所有 100 个原始 window 均参与预注册的组内 median，没有删点。CPU 尚差 0.152 个百分点，C005 只扩大每个 window 的工作量，以保留四线程能力并降低调度开销占比。
