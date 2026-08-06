# 假设账本

| ID | 可证伪假设 | 等级 | 支持证据 | 反证/替代解释 | 下一判别动作 | 状态 |
|---|---|---|---|---|---|---|
| H-01 | 锁定 PyTorch 可在 Python 3.11 上提供 CPU/MPS 执行 | E2 | C001：torch 2.13.0 安装成功，MPS built/available，8 类操作均通过 | 仅限当前 OS/架构/版本 | M4 真实 Shape 再确认 | PROVEN_LOCAL |
| H-02 | 固定条件下 CPU/MPS median 测量噪声支持 5% 门禁 | E2 | C001–C005；最终 CPU 1.611%、MPS 0.314%，所有原始 window 保留 | 结论依赖足量 warmup、长 window、组内 median，其他 Case 需重定参数 | M4 为每个 Case 实施同协议 | PROVEN_LOCAL |
| H-03 | Framework/MPS 可归因分配粒度支持 5% 门禁 | E1 | C001 MPS current allocation delta 精确读到 8 MiB；CPU RSS 可读 | CPU RSS 是进程级；driver allocation 不等于逻辑 Tensor | M4 live-set 与 allocator 对照 | OPEN |
| H-04 | 冻结操作集合可表达并运行两层真实 Transformer | E1 | C001 的 8 类操作 CPU/MPS 可运行且数值对齐 | 尚未编译/运行真实两层结构 | M2 编译展开，M4 E2E 验证 | OPEN |
