# 验收账本

| ID | 完成条件 | 状态 | 证据 | 最近更新 |
|---|---|---|---|---|
| AC-01 | 干净 checkout 可创建锁定环境并运行测试 | DONE | C019：公开远端 clean clone `97c3dfe`，锁定安装，34 tests；双 compile byte-identical | 2026-08-06 |
| AC-02 | YAML 可选择 CPU/MPS 并运行固定 Shape forward/prefill | DONE | C017：两个 AnalysisPlan 仅通过 YAML DeploymentIntent 选择 CPU/MPS；CLI 生成 completed Run Bundle | 2026-08-06 |
| AC-03 | 确定性产出四层 IR 与 provenance | DONE | M3：四层 IR、Cost Rule/公式、合并 provenance；CPU/MPS SemanticIR/CostIR byte-identical | 2026-08-06 |
| AC-04 | CPU 数值正确且 MPS 在声明容差内对齐 | DONE | C011：真实两层 causal Transformer；max abs `7.152557e-07`、max relative `3.223419e-06`；MPS 52 个叶子/参数/buffer 均为 `mps:0`，fallback=false | 2026-08-06 |
| AC-05 | FLOPs 与逻辑/参数/激活字节精确一致 | DONE | M3：逐 op + 单层 + 两层 independent literal tests，CostIR SHA/fingerprint | 2026-08-06 |
| AC-06 | CPU 强制 Case 留出 median latency 误差均 ≤5% | BLOCKED_BY_CONFIRMED_GATE | C017：Softmax noise `3.380%>3%`；C020 已增加受控环境硬门禁，当前真实 preflight 拒绝，待合格环境全新 cohort | 2026-08-06 |
| AC-07 | MPS 强制 Case 留出 median latency 误差均 ≤5% 且无 fallback | BLOCKED_BY_CONFIRMED_GATE | C018：3 个 valid holdout error PASS（最大 3.715%），但 valid=3<5；C020 防止未受控 Bundle 再进入校准 | 2026-08-06 |
| AC-08 | CPU/MPS 可归因峰值分配误差均 ≤5% | BLOCKED_BY_CONFIRMED_GATE | C018：MPS valid memory error 0% 但数量不足；C020 要求 passed preflight 后才能 fit/holdout | 2026-08-06 |
| AC-09 | E2E 偏差可下钻且保留未归因桶/置信度 | DONE | C017：每 Bundle 60 spans、52 operator spans、exact Stable Path coverage 100%；未归因 host 桶显式保留 | 2026-08-06 |
| AC-10 | Explanation Graph 可下钻到公式、校准和 Span | IN_PROGRESS | M4：181 nodes/115 edges，latency/throughput/memory→scope→CostRule→span；待 M5 calibration evidence 节点 | 2026-08-06 |
| AC-11 | 公共 CI 与受信任 M4 证据流程完整 | DONE | C019：GitHub Actions `31099822022` Success/56s；C020：trusted script 强制环境 preflight，public Linux lane 不运行硬件 benchmark | 2026-08-06 |
| AC-12 | 交付物在约定路径且远端 main 与验证提交一致 | IN_PROGRESS | 当前实现已推送且 clean checkout 复验；待本轮最终审计 commit，以及 Goal gate 决策后才能整体 DONE | 2026-08-06 |
