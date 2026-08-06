# C001：锁定环境并验证 CPU/MPS 最小可行性

- **开始/结束：** 2026-08-06T17:29:02+08:00 / 2026-08-06T17:37:26+08:00
- **阶段：** BASELINE -> PROBE
- **动作类型：** PROBE
- **关联验收/未知量：** AC-01、AC-04、U-01、U-02、H-01 至 H-04

## 预注册

- **本轮 micro-goal：** 建立仓库本地 Python 3.11/PyTorch 锁定环境，并获得目标操作在 CPU/MPS 上 availability、correctness、同步计时噪声和 allocator 接口的最小证据。
- **当前假设：** H-01 至 H-04，见 `HYPOTHESES.md`。
- **已有证据：** `evidence/baseline-20260806.md`、`evidence/compatibility-matrix.md`。
- **证据等级：** H-01/H-04 E1；H-02/H-03 E0。
- **唯一主要变量：** 从“无项目依赖”切换为一个由 `uv` 锁定的 Python 3.11 项目环境；设备仅在同一探针内作为对照维度。
- **预期观察：** PyTorch 成功安装；CPU/MPS 均可执行冻结操作；输出在声明容差内；同步 median 可采集；至少能读取 CPU 进程或 MPS allocator 的可归因内存指标。
- **判别规则：** 全部通过则相关假设升至 E2；安装失败按兼容矩阵只尝试一个有依据候选；单个操作失败则记录稳定错误签名并停止该路径；`IQR/median >3%` 则 H-02 暂不成立并进入噪声诊断而非重复全量运行。
- **成本与风险：** 预计 5–15 分钟下载与探针；只占用本机 CPU/GPU；无付费费用；风险限于仓库本地 `.venv` 和下载缓存。
- **停止与回滚：** 单命令 20 分钟超时；同签名无新证据只复现一次；环境可由锁文件重建，不修改系统 Python。

## 执行

- **脱敏命令：** `commands.md`
- **配置/环境差异：** 已新增 `pyproject.toml`，选择 Python 3.11.15、PyTorch 2.13.0、pytest 9.0.2、psutil 7.2.2；`uv.lock` 已生成，`.venv` 为仓库本地且已忽略。
- **代码差异：** 已按 RED→GREEN 新增公开 `groundupscale probe` CLI、能力探针与契约测试。
- **日志/指标：** `uv lock` 解析 37 个包；`uv sync` 安装成功。首次测试按预期 RED：1 failed，错误签名为 `ModuleNotFoundError: No module named 'groundupscale'`；实现后 1 passed。真实探针摘要见 `../../evidence/c001-probe-summary-20260806.md`。

## 结果

- **观察事实：** 官方 macOS arm64 wheel 可在本机 Python 3.11 环境安装；CPU/MPS backend 均可用；8 类冻结操作全部通过；MPS 最大绝对误差为 RMSNorm 的 `9.536743e-07`；MPS framework-attributed allocation delta 为 `8,388,608` bytes。单次操作组计时噪声 CPU `8.262%`、MPS `5.267%`，均高于 3% 判别线。
- **错误签名：** RED-01：公开包尚不存在，CLI 子进程退出 1；这是预注册的实现前失败，而非环境失败。
- **推断：** H-01 已成立。目标操作 runtime 支持成立；H-04 的完整模型表达仍待 M4。MPS allocator 接口可用，但 CPU RSS 仍只能作为诊断。H-02 在当前约 0.5–1.6 ms 计时窗下不成立，下一轮必须改变测量窗口，不得原样重跑。
- **证据等级变化：** H-01 E1→E2；H-02 E0→E1（获得反证并进入方法诊断）；H-03 E0→E1；H-04 的操作支持子命题 E1→E2。
- **信息增量：** 首次获得本机 CPU/MPS 数值、计时和 allocator 直接证据；识别出短窗口噪声为 M1 剩余问题。

## 结论

- **验收/交付更新：** AC-01、AC-04、D-01、D-08 保持 IN_PROGRESS/WIP；建立了首个可复现公开命令。
- **预算变化：** 已启动费用监控；尚无高成本实验。
- **下一 micro-goal：** C002 只改变“单个计时样本内重复的操作组次数”，检验扩大窗口并归一化能否把 CPU/MPS IQR/median 降至 3% 以内。
- **是否需决策：** 当前无。
