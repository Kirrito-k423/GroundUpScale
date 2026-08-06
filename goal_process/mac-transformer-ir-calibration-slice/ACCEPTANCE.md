# 验收账本

| ID | 完成条件 | 状态 | 证据 | 最近更新 |
|---|---|---|---|---|
| AC-01 | 干净 checkout 可创建锁定环境并运行测试 | IN_PROGRESS | C001：`uv.lock`、本地 sync、公开 CLI 测试 1 passed；待干净 checkout/CI 复验 | 2026-08-06 |
| AC-02 | YAML 可选择 CPU/MPS 并运行固定 Shape forward/prefill | DONE | C017：两个 AnalysisPlan 仅通过 YAML DeploymentIntent 选择 CPU/MPS；CLI 生成 completed Run Bundle | 2026-08-06 |
| AC-03 | 确定性产出四层 IR 与 provenance | DONE | M3：四层 IR、Cost Rule/公式、合并 provenance；CPU/MPS SemanticIR/CostIR byte-identical | 2026-08-06 |
| AC-04 | CPU 数值正确且 MPS 在声明容差内对齐 | DONE | C011：真实两层 causal Transformer；max abs `7.152557e-07`、max relative `3.223419e-06`；MPS 52 个叶子/参数/buffer 均为 `mps:0`，fallback=false | 2026-08-06 |
| AC-05 | FLOPs 与逻辑/参数/激活字节精确一致 | DONE | M3：逐 op + 单层 + 两层 independent literal tests，CostIR SHA/fingerprint | 2026-08-06 |
| AC-06 | CPU 强制 Case 留出 median latency 误差均 ≤5% | BLOCKED_BY_CONFIRMED_GATE | C017：Softmax noise `3.380%>3%`，未合法进入 CPU 校准；不可自行降标 | 2026-08-06 |
| AC-07 | MPS 强制 Case 留出 median latency 误差均 ≤5% 且无 fallback | BLOCKED_BY_CONFIRMED_GATE | C018：3 个 valid holdout 全部 error PASS（最大 3.715%），但 7 个中 4 个 noise quarantine，valid=3<5；Profile 未晋升 | 2026-08-06 |
| AC-08 | CPU/MPS 可归因峰值分配误差均 ≤5% | BLOCKED_BY_CONFIRMED_GATE | C018：MPS valid holdout memory error 0%，但有效数量不足；CPU 无合法 holdout。基础 54,534,144 B 与观测 69,214,208 B 均保留 | 2026-08-06 |
| AC-09 | E2E 偏差可下钻且保留未归因桶/置信度 | DONE | C017：每 Bundle 60 spans、52 operator spans、exact Stable Path coverage 100%；未归因 host 桶显式保留 | 2026-08-06 |
| AC-10 | Explanation Graph 可下钻到公式、校准和 Span | IN_PROGRESS | M4：181 nodes/115 edges，latency/throughput/memory→scope→CostRule→span；待 M5 calibration evidence 节点 | 2026-08-06 |
| AC-11 | 公共 CI 与受信任 M4 证据流程完整 | IN_PROGRESS | C019：GitHub-hosted Linux workflow + trusted local M4 script 已实现，待远端绿灯 | 2026-08-06 |
| AC-12 | 交付物在约定路径且远端 main 与验证提交一致 | NOT_STARTED | 待 M6 | 2026-08-06 |
