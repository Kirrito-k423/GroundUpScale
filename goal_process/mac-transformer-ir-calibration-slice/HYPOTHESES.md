# 假设账本

| ID | 可证伪假设 | 等级 | 支持证据 | 反证/替代解释 | 下一判别动作 | 状态 |
|---|---|---|---|---|---|---|
| H-01 | 锁定 PyTorch 可在 Python 3.11 上提供 CPU/MPS 执行 | E2 | C001：torch 2.13.0 安装成功，MPS built/available，8 类操作均通过 | 仅限当前 OS/架构/版本 | M4 真实 Shape 再确认 | PROVEN_LOCAL |
| H-02 | 固定条件下 CPU/MPS median 测量噪声支持 5% 门禁 | E2 | C017：MPS 5/5 Case `IQR/median<=3%`；C012–C017 全部 raw windows 保留 | CPU C017 Softmax `3.380%`，且 C012–C015 失败 Case 跨轮漂移；4P/6E thread 假设被 C016 反证 | MPS 进入 M5；CPU 按 Goal 升级，不再搜索协议 | PARTIAL_MPS_CPU_ESCALATE |
| H-03 | Framework/MPS 可归因分配粒度支持 5% 门禁 | E2 | weakref observer 在 CPU/MPS 均观测真实 live unique Tensor storage peak `69,214,208 B`；MPS allocator/driver 单列 | 基础 live-set `54,534,144 B` 低估 21.21%；allocator reservation/workspace 不在 gate metric | M5 用 fit/holdout 校准 Tensor storage peak | PROVEN_OBSERVABLE |
| H-04 | 冻结操作集合可表达并运行两层真实 Transformer | E2 | C011：52 个语义叶子真实执行；CPU/MPS E2E max abs `7.1526e-07`；16 个 View/Transpose 均经 storage alias 审计 | MPS 非连续 `out=` 在目标 Shape 静默算错，已改为 query-major batched MatMul；适用域仍限固定 Shape/版本 | M4 trace/benchmark 继续验证组合运行 | PROVEN_LOCAL |
| H-05 | 硬件无关 Cost 规则可精确计算 FLOPs 与逻辑/状态/激活字节 | E2 | M3：逐 op、单层、两层 hard-coded literal 全部精确通过 | 不代表 cache/DRAM traffic、峰值 live-set 或 duration | M4 用 runner live-set/trace 对齐，M5 duration 校准 | PROVEN_LOGICAL |
| H-06 | 供电、热状态、负载和竞争进程的前置门禁可减少不合格硬件观察 | E1→E3 | C020–C021：三个连续 Goal 回合真实 preflight 均识别同一外部负载并在 benchmark 前拒绝；42 tests 验证 fail-closed 治理 | 只证明能稳定识别当前干扰，尚未证明有效 holdout 比例会提高；门禁后仍可能有调度/设备噪声 | 用户解除外部负载后建立全新 fit/holdout cohort，并继续逐 Run 3% gate | BLOCKED_EXTERNAL_OUTCOME_PENDING |
