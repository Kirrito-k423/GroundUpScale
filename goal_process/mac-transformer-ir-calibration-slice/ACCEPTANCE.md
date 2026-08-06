# 验收账本

| ID | 完成条件 | 状态 | 证据 | 最近更新 |
|---|---|---|---|---|
| AC-01 | 干净 checkout 可创建锁定环境并运行测试 | IN_PROGRESS | C001：`uv.lock`、本地 sync、公开 CLI 测试 1 passed；待干净 checkout/CI 复验 | 2026-08-06 |
| AC-02 | YAML 可选择 CPU/MPS 并运行固定 Shape forward/prefill | NOT_STARTED | 待 M2/M4 | 2026-08-06 |
| AC-03 | 确定性产出四层 IR 与 provenance | NOT_STARTED | 待 M2/M3 | 2026-08-06 |
| AC-04 | CPU 数值正确且 MPS 在声明容差内对齐 | IN_PROGRESS | C001：8 类原子操作通过，MPS max abs error `9.536743e-07`；待真实模型 | 2026-08-06 |
| AC-05 | FLOPs 与逻辑/参数/激活字节精确一致 | NOT_STARTED | 待 M3 | 2026-08-06 |
| AC-06 | CPU 强制 Case 留出 median latency 误差均 ≤5% | NOT_STARTED | 待 M5 | 2026-08-06 |
| AC-07 | MPS 强制 Case 留出 median latency 误差均 ≤5% 且无 fallback | NOT_STARTED | 待 M5 | 2026-08-06 |
| AC-08 | CPU/MPS 可归因峰值分配误差均 ≤5% | NOT_STARTED | 待 M5 | 2026-08-06 |
| AC-09 | E2E 偏差可下钻且保留未归因桶/置信度 | NOT_STARTED | 待 M4 | 2026-08-06 |
| AC-10 | Explanation Graph 可下钻到公式、校准和 Span | NOT_STARTED | 待 M4 | 2026-08-06 |
| AC-11 | 公共 CI 与受信任 M4 证据流程完整 | NOT_STARTED | 待 M6 | 2026-08-06 |
| AC-12 | 交付物在约定路径且远端 main 与验证提交一致 | NOT_STARTED | 待 M6 | 2026-08-06 |
