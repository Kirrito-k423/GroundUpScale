---
goal_id: mac-transformer-ir-calibration-slice
title: 跑通 Mac CPU/MPS 两层 Transformer 性能建模与实测校准链路
status: READY
owner: Kirrito-k423
executor: Codex
created_at: 2026-08-06T00:00:00+08:00
deadline: 无固定期限
priority: P1
execution_skill: "$goal-execution"
process_dir: "goal_process/mac-transformer-ir-calibration-slice"
---

# 跑通 Mac CPU/MPS 两层 Transformer 性能建模与实测校准链路

## 一句话目标

在当前 Apple M4 Mac 上，以人类编写的 YAML 驱动一个固定 Shape 的两层小型 Transformer forward/prefill，完整产出 Model IR、Workload IR、Semantic IR 和 Cost IR，在 CPU 与 MPS 上完成可追溯的预测—实测对比，并使约定指标的独立验证误差不超过 5%。

## 背景与价值

- **背景：** GroundUpScale 已完成多层 IR、可解释性、插桩、Run Bundle 和校准治理的架构设计，但仓库尚无编译器、模型运行器或实验代码。
- **业务价值：** 用一个足够小但端到端闭合的真实穿刺，验证核心 IR seam、公式、CPU/MPS Backend、Trace 对齐和受控校准是否能够工作，为后续扩展到真实模型、训练、并行策略与异构硬件建立可信底座。
- **失败代价：** 如果先扩展模型和策略而未验证这条链路，后续误差将无法判断来自语义、公式、Backend、运行时还是测量方法，架构返工成本高。

## 范围

### 必须完成

1. 建立可复现的 Python 项目环境、依赖锁文件、命令行入口和测试框架。
2. 提供严格 Schema 校验的人类编写 YAML：Model Spec、Workload Spec、Analysis Case、Deployment Intent、Hardware Spec、Fabric Graph、Benchmark Case 和 Analysis Plan。
3. 从 YAML 确定性生成并序列化 Model IR 与 Workload IR。
4. 通过层次化 Region、Typed Value 和 State Effect 生成可解释 Semantic IR。
5. 将 MatMul、Add、RMSNorm、Softmax、SiLU、Mul 及零物化的 View/Transpose 降低为硬件无关 Cost IR，输出 FLOPs、逻辑读写量、参数/激活量、依赖和公式来源。
6. 构建固定 Shape（`B=1, S=512, H=512, heads=8, intermediate=2048, FP32`）、两层、pre-norm、包含 self-attention 与 MLP 的小型 Transformer forward/prefill 参考实现，在 Mac CPU 与 MPS 上执行。
7. 为三个核心原子操作、单层模块、两层 E2E 分别建立 Benchmark Case；Benchmark 与 trace/deep-probe 模式分离。
8. 生成不可变 Run Bundle，包含锁定输入、各层 IR、预测、原始/规范化实测、Alignment Map、误差归因、候选 Calibration Profile 和可阅读报告。
9. 对 CPU 与 MPS 分别建立窄适用域、版本化的 Calibration Profile，使用独立留出运行验证 5% 误差门禁，不静默覆盖基础公式。
10. 建立公共确定性 CI；本机真实 M4 测量只运行受信任代码，不把个人 Mac 暴露为公共 PR 的通用 self-hosted runner。
11. 将实现、测试、样例、实验证据、运行手册和最终报告提交并推送到远端 `main`。

### 明确不做

- 不实现 backward、optimizer、decode、RL、FSDP、offload、TP/PP/EP/CP 或多机通信。
- 不宣称固定 Shape、单台 M4 上得到的 Calibration Profile 可泛化到其他 Shape、模型、Mac、操作系统或框架版本。
- 不以 Web 应用作为本 Goal 的完成条件；必须产出结构化 Explanation Graph，HTML/CLI 至少有一种可阅读入口。
- 不为达到 5% 而修改实测值、隐藏未归因时间、用验证数据拟合，或把 CPU fallback 当作 MPS 结果。
- 不要求预测不可稳定归因的系统总内存、全机 GPU 利用率或后台进程开销达到 5%。

## 基线与环境

- **仓库与基线提交：** `https://github.com/Kirrito-k423/GroundUpScale.git`，`16bb280ee049d4dea9dc2f9b4b295a9e27853650`。
- **工作分支：** `main`，当前与 `origin/main` 同步且工作区在起草前干净。
- **本地环境：** MacBook Air，Apple M4，10-core CPU，8-core GPU，16 GB unified memory，Metal 3，macOS 15.7.4。
- **软件栈：** 系统 `python3` 为 3.14.3；另有 Python 3.11.15 和 `uv 0.11.14`；当前没有安装 PyTorch。推荐使用仓库本地、锁定的 Python 3.11 环境。
- **数据与模型：** 无外部数据集；使用确定性随机种子生成固定 Shape 输入和权重；冻结配置为 `B=1, S=512, H=512, heads=8, intermediate=2048, layers=2, FP32`，模型包含 pre-norm self-attention 与 MLP。
- **已有成功基线：** 无实现、无运行命令、无性能或精度基线；现有证据只有架构文档与 ADR。

## 事实、假设与未知量

### 已知事实

- F-01：仓库当前只有设计、术语、ADR 和安装的技能，没有 `src/`、`tests/`、项目依赖或 CI；证据：基线提交文件清单。
- F-02：目标机器为 16 GB unified-memory Apple M4，CPU 与 GPU 共享物理内存；证据：`system_profiler`。
- F-03：当前系统 Python 没有 PyTorch，但 Python 3.11 与 `uv` 可用；证据：本地命令探针。
- F-04：已确认 Semantic IR 使用层次化 Region、Typed Value、显式 State Effect 和不可变 Lowering；证据：ADR-0027。
- F-05：已确认 Benchmark、trace、deep-probe 分离，产物进入不可变 Run Bundle；证据：ADR-0031、ADR-0032。

### 待验证假设

- H-01：锁定的 PyTorch 版本能在本机 Python 3.11 环境中同时提供 CPU 与 MPS 执行；当前证据等级：E1；反证条件：安装失败、MPS 不可用或目标操作不受支持。
- H-02：在固定 Shape、固定软件栈和受控测量协议下，CPU 与 MPS 的基准噪声足够低，使 median latency 的 5% 校准门禁具有统计意义；当前证据等级：E0；反证条件：重复运行的 IQR/median 持续超过门禁允许噪声。
- H-03：对框架可归因的分配量，预测峰值能在不拟合验证数据的情况下达到 5%；当前证据等级：E0；反证条件：Allocator 或 MPS Driver 粒度使可观测值无法稳定对应逻辑分配。
- H-04：MatMul、Add、RMSNorm、Softmax、SiLU、Mul 及 View/Transpose 足以表达已冻结的两层 Transformer forward/prefill 并在 CPU/MPS 上得到可对齐实现；当前证据等级：E1；反证条件：参考实现或 Backend 需要未声明的物化语义或 MPS 不支持目标操作。

### 关键未知量

- U-01：本机 MPS 对冻结操作、dtype 和 Shape 的实际支持与噪声水平；不阻止 M1 开始，由 availability、correctness 和重复测量探针判定。
- U-02：Framework allocator 与 MPS Driver 暴露的可归因内存粒度能否支持 5% 门禁；不阻止 M1 开始，由独立内存探针判定，失败时按升级规则处理而不静默改口径。

## 验收标准

| ID | 可观察的完成条件 | 必需证据 | 验收人 |
|---|---|---|---|
| AC-01 | 一条记录在运行手册中的命令可从干净 checkout 创建锁定环境并运行测试 | `pyproject.toml`、锁文件、CI 日志、本地命令日志 | 用户 |
| AC-02 | 人类只修改 YAML 即可选择 CPU/MPS 并运行固定 Shape forward/prefill | 完整 Spec、Schema 测试、CLI 日志、Run Manifest | 用户 |
| AC-03 | 每次运行确定性产出 Model IR、Workload IR、Semantic IR、Cost IR，且节点身份、Shape、dtype、公式和 provenance 可追溯 | Run Bundle、golden tests、determinism tests | 用户 |
| AC-04 | 核心操作与模型的数值输出通过 CPU 参考正确性验证，MPS 在声明容差内与 CPU 对齐 | correctness 测试及原始误差报告 | 用户 |
| AC-05 | FLOPs、逻辑 Tensor 字节数、参数字节数和显式激活字节数与独立参考计算完全一致 | 公式单测、手工算例、Cost IR 与参考表 | 用户 |
| AC-06 | CPU 上所有强制 Benchmark Case 的预测 median latency 相对独立留出实测 median 误差均不超过 5% | 未参与拟合的留出 Trace、预测、逐 Case 误差表 | 用户 |
| AC-07 | MPS 上所有强制 Benchmark Case 的预测 synchronized median latency 相对独立留出实测 median 误差均不超过 5%，且无 CPU fallback | MPS 环境证据、fallback 检查、留出 Trace、逐 Case 误差表 | 用户 |
| AC-08 | CPU 与 MPS 的框架可归因峰值分配量按最终确认口径误差不超过 5%；不可归因系统内存单列且不伪装成已解释内存 | 内存观测、live-set、归因边界、逐 Case 误差表 | 用户 |
| AC-09 | 两层 E2E 偏差可通过结构化 trace 下钻到 module/operator/runtime 层，并保留未归因桶和 Alignment confidence | benchmark/trace 成对 Run Bundle、Alignment Map、Error Attribution | 用户 |
| AC-10 | Explanation Graph 能从 E2E latency/throughput/peak memory 下钻到 Stable Path、Cost Formula、校准证据和实测 Span | CLI 或 HTML 报告、解释完整性测试 | 用户 |
| AC-11 | 公共 CI 覆盖 Schema、IR、公式、数值正确性和确定性；真实 M4 证据通过受信任本地流程产生 | GitHub Actions 绿灯、本地受信任运行手册和证据 | 用户 |
| AC-12 | 所有交付物位于约定路径、远端 `main` 包含最终提交且工作区干净 | commit SHA、`git ls-remote`、最终审计清单 | 用户 |

> **已确认的 5% 口径：** 对每个强制 Case，使用至少 5 次未参与拟合的独立留出运行；先要求留出实测 `IQR / median <= 3%`，再计算 `abs(predicted_median - observed_median) / observed_median <= 5%`。吞吐由同一观察窗的完成量与时间推导。内存只门禁 framework-attributed allocation delta，不门禁全系统 RSS 或不可归因 Driver/后台开销。FLOPs、逻辑读写字节、参数和显式激活字节必须与独立参考精确一致，trace 子 Span 仅用于归因而不是 E2E 真值。

## 交付物

| ID | 交付物 | 目标位置 | 完成条件 |
|---|---|---|---|
| D-01 | Python 包、CLI、依赖与锁文件 | `src/groundupscale/`、`pyproject.toml`、锁文件 | 干净环境可安装运行 |
| D-02 | YAML Specs 与 Schema | `specs/`、`src/groundupscale/schemas/` | 正反例验证通过 |
| D-03 | IR、编译器、Cost Lowerer 与 provenance | `src/groundupscale/ir/`、`src/groundupscale/compiler/` | AC-03、AC-05 通过 |
| D-04 | 两层模型、CPU/MPS runner 与插桩 Adapter | `src/groundupscale/benchmark/` | AC-04、AC-06 至 AC-09 通过 |
| D-05 | 自动化测试与公共 CI | `tests/`、`.github/workflows/` | AC-01、AC-11 通过 |
| D-06 | 不可变运行证据与候选/晋升校准 | `.groundupscale/runs/`，必要摘要进入 `evidence/` | Manifest、hash、留出证据齐全 |
| D-07 | 中文运行手册、方法与最终误差报告 | `docs/`、`goal_process/mac-transformer-ir-calibration-slice/FINAL-REPORT.md` | 新用户可复现，逐 AC 列证据 |
| D-08 | Goal 过程证据 | `goal_process/mac-transformer-ir-calibration-slice/` | 命令、日志、决策、里程碑状态可追溯 |

## 里程碑

| ID | 里程碑 | 退出条件 | 目标时间 |
|---|---|---|---|
| M1 | Goal 与环境可行性 | Goal READY；锁定环境；CPU/MPS availability 和最小算子探针达到 E2 | 开始执行后的首个里程碑 |
| M2 | YAML 到 Semantic IR | Specs、Model/Workload/Semantic IR、身份/provenance 和 golden tests 通过 | M1 后 |
| M3 | Cost IR 与参考公式 | 核心算子和两层模型的 FLOPs/字节/内存公式通过独立参考 | M2 后 |
| M4 | CPU/MPS 实测链路 | 数值正确；benchmark/trace/deep-probe；Run Bundle 与 Alignment Map 完整 | M3 后 |
| M5 | 5% 校准门禁 | 训练/留出分离；CPU/MPS 强制 Case 全部通过最终 5% 口径 | M4 后 |
| M6 | CI、文档与发布 | 公共 CI 绿灯，最终报告完成，远端 `main` 与验证提交一致 | M5 后 |

## 预算

- **总墙钟时间：** 无固定截止；条件化 ETA 见下文。
- **人/代理执行时间：** 持续执行至所有 AC 完成或触发升级，不以单轮上下文为停止条件。
- **算力：** 当前本地 Apple M4 Mac；不默认使用远程服务器或付费云资源。
- **费用上限：** 不产生新的付费服务费用。
- **高成本实验上限：** 单轮最多 3 个超过 10 分钟的实验；超出前升级。
- **无新证据的同签名重跑上限：** 1。
- **无新依据的版本候选上限：** 2。
- **单次命令最长运行：** 默认 20 分钟；需更长时先说明原因和预期证据。
- **费用报告：** 本 Goal 预计总执行超过 20 分钟，执行阶段启用 `$rmb-cost-report`，更新仓库根目录 `RMB-Cost.md`；缺少价格或汇率核验时标记 estimate。

## 权限边界

### 已授权

- 只读调查仓库、本机硬件和公开官方文档。
- 在 `GroundUpScale` 仓库中创建和修改本 Goal 所需代码、测试、YAML Spec、CI、文档、过程证据和实验产物。
- 使用仓库本地 Python 3.11 + `uv` 创建环境、下载并锁定依赖，不修改系统级 Python。
- 在本机运行 CPU/MPS benchmark、trace、deep-probe 和 profiler，并按费用监控规则生成 `RMB-Cost.md`。
- 验证通过后 commit 并直接 push 到公共仓库远端 `main`，禁止强推和历史重写。

### 需审批

- 使用远程服务器、付费云资源、GitHub 付费 macOS runner 或产生任何新增付费服务费用。
- 改变固定 Shape、dtype、真实 Transformer 范围、5% 验收口径或 CPU/MPS 双后端范围。

### 禁止

- 把个人 Mac 注册为可执行公共 PR 代码的通用 self-hosted runner。
- 修改系统级 Python、全局安装依赖、暴露凭证或采集不受控环境变量。
- 删除用户数据、强制推送、重写远端历史或绕过失败门禁。
- 未经确认降低 5% 标准、删除强制 Case 或把验证数据并入拟合数据。

## 执行与证据规则

- 使用 `$goal-execution`。
- 每轮只设一个 micro-goal，默认只改变一个主要变量。
- 高成本实验必须达到 E2 证据等级，或获得用户明确豁免。
- 所有关键日志、配置、命令和结论归档到 `goal_process/mac-transformer-ir-calibration-slice/`。
- 不得将终端、聊天上下文、`.groundupscale/` 或 `/tmp` 作为唯一证据载体；必要摘要和 manifest 必须进入过程目录或 `evidence/`。
- 5% 校准使用显式拟合分区与独立留出分区；失败时保留基础预测、候选 Profile 和残差，禁止静默改公式。

## 汇报

- **节奏：** 每个里程碑完成时；单次持续执行超过 60 分钟时至少给一次简短进度更新。
- **对象：** 用户。
- **必答问题：** 已完成什么、当前证据等级、卡在哪里、解决过什么、时间花在哪里、5% 是否仍可达、条件化 ETA、已有交付、需要什么决策。

## 停止与升级

- **DONE：** AC-01 至 AC-12 均有当前状态的直接证据，所有必需交付物已到位并推送远端 `main`。
- **BLOCKED：** 三个连续 Goal 回合都因同一外部依赖、权限或物理环境缺口无法产生新证据，并已穷尽安全替代方案。
- **ESCALATE：** MPS 目标操作不受支持；受控重复测量噪声使 5% 不可判定；连续 3 个有依据的模型/校准修正仍不能达到 5%；必须引入目标外操作、付费/远程资源或改变验收口径。
- **ROLLBACK：** 新依赖或实现破坏已验证里程碑且无法在两次有依据修复内恢复时，使用正常 `git revert` 回到最近绿色提交；不使用破坏性 reset。
- **不得自行缩小：** 范围、CPU/MPS 双后端、IR 链或 5% 验收变化必须由用户确认。

## 条件化 ETA

- **路径 A：** 若 PyTorch CPU/MPS 可用且测量噪声满足已确认的 5% 统计口径，预计按 M1–M6 顺序推进；当前仅 E0/E1，未运行可行性探针前不给出虚假日期。
- **路径 B：** 若 MPS 支持、Allocator 可归因性或噪声假设不成立，先交付完整链路、反证和误差归因，再在最晚决策点由用户决定是否扩操作/Shape、改测量口径或接受平台限制；不得自行降标。
- **最晚决策点：** M1 完成 CPU/MPS availability 与噪声探针后，必须冻结模型结构、固定 Shape、dtype 和 5% 统计口径；M4 首轮误差矩阵后决定是否继续校准或升级模型假设。

## 待确认

- 无。执行中若出现范围、验收、预算或权限变更，必须按 Goal 变更流程重新确认。

## 确认记录

- 2026-08-06：用户给出初始目标；Goal 保持 DRAFT，等待 Q-01 至 Q-03。
- 2026-08-06：用户确认 Q-01 至 Q-03 的全部推荐值；冻结 5% 统计口径、真实 Transformer 操作集合、固定 Shape、dtype、权限与预算，Goal 状态改为 READY。
