# 跨硬件性能差距诊断规范

状态：规范性；对应 ADR 0035。

决策来源：[Wayfinder 地图 #1](https://github.com/Kirrito-k423/GroundUpScale/issues/1)、
已关闭的研究/原型子票 #2–#7，以及汇总规范
[#8](https://github.com/Kirrito-k423/GroundUpScale/issues/8)。只有这些来源中的最终决议构成
本规范的 MUST。其他 ADR、研究文档和原型原始结果用于统一词汇、实现建议和验收 fixture，
不能把未决阈值提升成全局事实。

## 1. 范围与规范语言

本规范定义 GroundUpScale 如何从可复现证据生成跨硬件性能差距诊断。它覆盖：

- Resource Physical Floor、Operator Achievable Frontier、Schedule Achievable
  Frontier 与 Observation 的独立语义；
- Frontier Evidence、Anchor 生命周期、Capability Surface 与 Shape 有效域；
- Hardware Validity Cohort、Baseline Timing Lane、Diagnostic Profiling Lane
  与 Completion Boundary；
- 显式 schedule、排他 trace ledger、Diagnostic Trigger、Shape Probe、Verdict；
- Diagnostic Evidence Bundle、回放、配置策略和 conformance tests。

`MUST/MUST NOT` 是规范约束，`SHOULD` 是有充分理由才可偏离的建议，`MAY` 是可选能力。
系统不得通过实现默认值替代本规范列出的权威 unknown。按照 [ADR 0038](../adr/0038-separate-authority-from-tiered-iteration-values.md)，迭代报告可以在独立 Report Value 字段中提供带 Evidence Grade、Generation Stage、区间、方法和用途限制的数值，但不得回写或覆盖权威字段。

## 2. 规范结果模型

一次诊断 MUST 同时保存四条互不覆盖的时间轴：

| 层 | 回答的问题 | 资格 | 禁止解释 |
| --- | --- | --- | --- |
| Resource Physical Floor | 在算法无关的最小资源需求与已验证能力下，不可能快于什么边界？ | 资源探针、最小 demand、cohort 与有效域完整 | 当前实现预测、prediction error 基线 |
| Operator Achievable Frontier | 同一完整执行域和 cohort 中，正确且可复现的实现已经达到什么水平？ | `QUALIFIED + ACTIVE` Anchors 与有效 Surface | 单次最快样本、跨 cohort 推导 |
| Schedule Achievable Frontier | 显式候选、依赖、资源与变换已经达到什么 E2E 边界？ | 可回放的 schedule candidate 与 Operator Frontiers | whole-scope roofline、隐式并发/融合 |
| Observation | 声明的 lane 与 Completion Boundary 实际测到什么？ | 原始样本、timer、环境和 instrumentation 身份 | 自动覆盖或下调 Frontier |

规范公式只在同一有效域、同一 cohort 和显式语义允许的集合内成立：

```text
F_resource(o, x) = max_r(minimum_demand_r(o, x) / validated_rate_r(cohort))

F_operator(o, x) = min_c(holdout_latency(c, x)),
  c in Eligible(o, x, cohort)

F_schedule(x) = min_s(longest_path(
  G_semantic union G_execution(s), local_frontier_times)),
  s in EligibleSchedules

gap_i = abs(Observation_i - selected_boundary_i)

Trigger_i = gap_i > combined_uncertainty_i and
  (predicted_top10_i or observed_top10_i or
   gap_i > E2E_observation / 10)
```

资源能力、候选、路径或查询点不满足公式的域条件时，对应结果 MUST 为结构化
`unknown(reason_code)`，不能把缺失量当作 0、无穷或一个猜测值。非空降级估计只能存在于独立、可重放的 Report Value 层。

## 3. 已确认决策（规范性需求）

### 3.1 四轴与比较语义

| ID | 需求 |
| --- | --- |
| BND-001 | 系统 MUST 同时保存四条时间轴，任何一层不得覆盖、重命名或反向校准另一层。 |
| BND-002 | Resource Physical Floor MUST 标注“可能不可达到”；它与 Observation 的距离 MUST NOT 被报告为 prediction error。 |
| BND-003 | Operator/Schedule Frontier MUST 有正确、稳定、可复现、可回放的已达到证据。 |
| BND-004 | 跨硬件 MUST 各自在 Hardware Validity Cohort 内建面；MUST NOT 从一个 cohort 直接推导另一个 cohort 的 Frontier。 |

### 3.2 Frontier Evidence 与 Anchor 生命周期

| ID | 需求 |
| --- | --- |
| ANC-001 | Observation validity 与 Frontier role MUST 是正交状态轴；只有 `QUALIFIED + ACTIVE` 是权威 knot。 |
| ANC-002 | Qualification MUST 包含完整身份、正确性、闭合 Completion Boundary、warmup、独立 session repeatability、环境 guard、exact-Shape best-of-correct 与独立 holdout。 |
| ANC-003 | Frontier Evidence MUST 保存候选 manifest/coverage、input/implementation digest、正确性、原始 timing/exclusions、环境/cohort、统计、holdout 和状态转换理由。 |
| ANC-004 | 后续较慢 Observation MUST NOT 降低已有 ACTIVE Anchor。cohort 变化创建独立 Surface；错误 correctness/provenance 可 revoke/retract；仅无法复现时为 insufficient evidence。 |
| ANC-005 | `frontier_shift` MUST 同时满足 C2/C3、独立 holdout、至少 3 个独立 sessions、同 cohort、稳定邻域、局部密集 Shape 消歧和同 regime 重拟合。 |

### 3.3 Capability Surface 与 Shape 有效域

| ID | 需求 |
| --- | --- |
| SUR-001 | Capability Surface MUST 是绑定完整 hard-domain key、candidate/algorithm family、cohort 与版本的偏函数。 |
| SUR-002 | 初始权威基线 MUST 先过滤 validated domain，再在保留的低维局部 simplicial cell 内对 effective rate 做 piecewise-linear interpolation；latency 由 work/rate 派生。 |
| SUR-003 | Exact Anchor MUST 是同一查询路径的 knot；同一保留 cell 内至少 C0 连续。完整凸包、逐轴包围盒、nearest neighbor 和全局 P80 MUST NOT 充当有效域。 |
| SUR-004 | 候选家族 MUST 分面保存。只有共同稳定支持域或独立验证接缝允许 envelope；支持消失或 seam 未验证时 MUST 分域、拒绝或 unknown。 |
| SUR-005 | Anchor、interpolation/model 与 instrumentation uncertainty MUST 分项保存；组合 policy、目标覆盖、校准数据和版本 MUST 随 Surface 记录。 |
| SUR-006 | 缺少足够 uncertainty calibration 时 MUST 返回 `unknown(insufficient_uncertainty_evidence)`；provisional baseline MAY 选 probe，但 MUST NOT 进入 Frontier、Trigger 或 Verdict。 |
| SUR-007 | 未验证 alignment、working set、numeric/layout、dispatch 或 candidate support 变化 MUST 为 unknown，不能跨 regime 连续。 |

`u_anchor² = lambda^T Sigma lambda` 只允许作为 Anchor covariance propagation；它不代表总
uncertainty。Rate interval 映射到 latency 时上下界方向反转，不能先假设 latency 误差对称。

### 3.4 Hardware Validity Cohort 与测量 lane

| ID | 需求 |
| --- | --- |
| HWC-001 | Frontier-eligible evidence MUST 包含稳定 device/partition/topology/software 身份、完整 execution domain、正确性、Completion Boundary、原始 timing、warmup/repetition、timer source/resolution、Instrumentation Profile 与 Measurement Capability Manifest。 |
| HWC-002 | Baseline Timing Lane 是默认 Frontier timing；Diagnostic Profiling MUST 与其配对但独立保存。只有独立 overhead ablation 落入 Error Budget 且 instrumentation mode 属于 validity domain 时，diagnostic timing 才可晋级。 |
| HWC-003 | 异步设备 operator timing MUST 覆盖 device event/stream completion；CPU 异步线程池 MUST join；分布式 duration MUST 在各 rank 本地时钟域测量后按声明 reducer 聚合，不得跨节点相减绝对时钟。 |
| HWC-004 | Optional Diagnostics MUST capability-probe，并使用 `measured/derived/declared/unsupported/permission_denied/not_requested/not_applicable/collection_failed/unknown` 状态；缺失 MUST NOT 填零。 |
| HWC-005 | 缺失身份、正确性、Completion Boundary 或主 timer MUST 阻止 Anchor；缺失 optional field 只使依赖它的 attribution unknown。 |
| HWC-006 | Proxy MUST 声明 derivation、scope 与 attribution。semantic bytes/time 只能称 effective algorithm bandwidth；modeled work/time 不能称 executed operations。 |
| HWC-007 | 稳定硬件/软件/numeric/timing/communication 身份的未验证变化 MUST 切分 cohort；瞬时 health/contention/throttling/dispersion/timer/device/collection failure MUST quarantine/retry。 |

### 3.5 Schedule Frontier 与 trace 守恒

| ID | 需求 |
| --- | --- |
| SCH-001 | Candidate 内只有显式声明的 compute/memory/communication overlap 才可 local `max`；跨 candidate 只沿显式 Semantic/Execution order/resource edges 组合。 |
| SCH-002 | 未声明 fusion、concurrency、communication masking、contention 或 dispatch 效果 MUST 为 unknown/rejection。 |
| SCH-003 | Schedule Frontier MUST 版本化保存 candidates、dependency/resource path、transformations、overlap claims、uncertainty 与 evidence；不得改写 Operator Frontier。 |
| SCH-004 | Trace additive ledger MUST 使用互斥 leaves 加显式 residual；module/E2E parent 只作索引。每个 leaf identity 在变换前后 MUST 守恒或有直接新增/删除理由。 |
| SCH-005 | Counterfactual MUST 有显式语义且 ledger 差值守恒；只回收声明的 entries。 |

### 3.6 Diagnostic Trigger、Shape Probe 与 Verdict

| ID | 需求 |
| --- | --- |
| DIA-001 | Predicted Top 10 与 observed Top 10 MUST 独立选择，诊断候选取并集。 |
| DIA-002 | 只有 absolute gap 大于 combined uncertainty，且命中任一 Top 10 或大于 E2E Observation 的 1/10，才触发深度诊断。 |
| DIA-003 | 异常 Shape MUST 触发 exact-Shape probe，并锁定 semantic、Shape、dtype、layout/stride/alignment、threads、environment/cohort、candidate set、correctness 与 Completion Boundary。 |
| DIA-004 | Verdict MUST 支持六个规范枚举，并保存 satisfied/failed/not-evaluated gates 与直接 Evidence Bundle 引用。 |
| DIA-005 | `frontier_shift` 只能通过 ANC-005；C0/C1、缺 holdout/邻域/cohort 稳定性时 MUST NOT 给出该结论。 |
| DIA-006 | `implementation_headroom` MUST 有同 cohort、同 exact Shape、正确且可复现的更快 eligible candidate；Verdict 本身不自动改 Surface。 |
| DIA-007 | `integration_overhead` MUST 有 standalone operator 接近 Frontier，并由显式 copy/dispatch/sync/wait/其他消融或守恒 ledger 解释 wrapper/E2E excess；不得下调 Operator Frontier。 |
| DIA-008 | 任一必需 gate 未通过、Surface unknown、cohort 不匹配或消融不闭合时 MUST 能输出 `insufficient_evidence`。 |
| DIA-009 | 纯 latency gap、单慢样本或 proxy 异常 MUST NOT 产生 `confirmed_bug`；它必须附直接且可复现的 correctness failure、contract violation 或已定位 defect evidence。 |
| DIA-010 | 在 `suspected_regression` gate 未决期间，系统 MUST NOT 发明默认规则；相关场景保存证据并 fail closed。 |

### 3.7 Diagnostic Evidence Bundle 与回放

| ID | 需求 |
| --- | --- |
| DEB-001 | 每个结论 MUST 绑定可复现、可验证的 Diagnostic Evidence Bundle。 |
| DEB-002 | Bundle 最小信息 MUST 覆盖 resolved config/IR、hardware/cohort、execution domain、candidates/coverage、correctness、environment、paired lanes、raw timing/exclusions、timer/completion、Surface/Anchor/uncertainty、schedule/trace/alignment/ledger、ablations、Verdict gates 与 digests。 |
| DEB-003 | Anchor、Surface、Schedule Frontier 与 conclusion MUST 使用 immutable version 或 input/evidence digest；证据变化生成新版本，旧查询和状态转换继续可回放。 |
| DEB-004 | Derived metric MUST 保存 derivation 和输入引用；无法归因的时间 MUST 显式进入 residual。 |

最终 schema 名称、字段布局、artifact 拆分和迁移机制仍是 unknown；实现可以选择布局，但必须满足
上述信息内容和外部行为。

### 3.8 Governance 与 fail-closed 边界

| ID | 需求 |
| --- | --- |
| GOV-001 | 每个可配置 policy MUST 带稳定 `policy_id/version`、适用 scope、变更理由与 revalidation 要求；缺少已批准 policy 时 MUST fail closed。 |
| GOV-002 | 本规范列出的 Authority unknown MUST 保持未决；实现 MUST NOT 以隐藏默认值、heuristic 或 Report Value 代替权威决策。迭代报告 MAY 使用版本化降级策略生成独立的 B/C/D 级数值，但 MUST 保存区间、推导、允许用途且不得 promotion。 |
| GOV-003 | #5–#7 prototype artifacts MUST 保持 decision-only/throwaway；源码、机器数值和实验阈值 MUST NOT 复制、演化或 promotion 为生产实现/校准。 |

## 4. 可配置策略

下表表示“差异必须外置并版本化”已经决定，但数值不是跨硬件常量。每个 policy MUST 带
`policy_id/version`、scope、变更理由和 revalidation 要求；缺少已批准 policy 时 fail closed。

| Policy | 可配置内容 | 已固定的下限/禁令 |
| --- | --- | --- |
| POL-CORRECTNESS | per-op/dtype oracle、atol/rtol、determinism、reference implementation | correctness 不可绕过 |
| POL-TIMING | warmup convergence、samples/duration、session interval、repeatability、outlier、confidence | `frontier_shift` 至少 3 independent sessions |
| POL-COVERAGE | 普通 Anchor 的 candidate coverage | `frontier_shift` 必须 C2/C3 |
| POL-SURFACE | coordinate transform、dimension、cell quality/span、rejection band、confirmation、coverage、uncertainty combination | 域外/未校准不得猜测 |
| POL-COHORT | allowlist、等价变化、quarantine/retry、Error Budget、profiling overhead、revalidation | 必需身份变化不得静默复用 |
| POL-SCHEDULE | distributed reducer、resource kinds、compatibility、overlap claim types、schedule pool | 只消费显式语义 |
| POL-ARTIFACT | retention、remote URI、sampling、review role | 不可覆盖历史证据 |

原型中的 `0.15` rejection、`5%/10%/35%` 场景阈值、M4/synthetic 数值，以及当前
profile 的 P80/P95 或固定 Shape 数，只属于各自实验或 profile，MUST NOT 成为跨硬件默认值。

## 5. 实现建议（非规范）

- 以现有 immutable Run Bundle 作为最高价值黑盒接缝：冻结 evidence roles 与 policy
  versions，读取 Surface Query、四轴 comparison、Verdict gates、Explanation/derivation
  和 manifest verification。
- 将 Surface evaluator、Evidence qualifier、Cohort matcher、Schedule composer、Trigger
  evaluator 与 Verdict evaluator 实现为只消费显式数据的纯决策内核。
- Hardware Adapter 可采用 discover capabilities、fingerprint、preflight、build timing
  plan、collect 五步；这不要求各平台暴露相同 counters。
- 复用 Stable Path、Metric Derivation、Explanation Graph、Alignment Map、Error Attribution
  和 Run Manifest，不建立平行 provenance 系统。
- 展示层只投影机器结果，不自行重算或隐藏 Authority unknown；所有非空 Report Values 必须先进入同一机器结果，再投影到 HTML/JSON/CSV。

## 6. 未解决、不能猜测的 unknown

- 生产 cell 尺度/质量、rejection band、confirmation 数量、高维覆盖与稀疏算法；
- 普通 Anchor 的最低 candidate coverage、session 数、holdout 比例与多重比较修正；
- 总 uncertainty 的生产组合公式及 conformal/GP 等替代模型接受标准；
- 各 op/numeric semantics 的生产正确性阈值和 oracle；
- `suspected_regression` 自动 gate、历史窗口、最小幅度、版本匹配及多 Verdict 优先级；
- Schedule Frontier 的生产 promotion 门槛和跨设备资源竞争模型；
- 各平台 optional counters 的直接 attribution 能力；
- workload-specific distributed reducer 与 asynchronous pipeline Completion Boundary；
- Diagnostic Evidence Bundle 最终 schema/迁移/artifact 布局；
- 第二真实硬件 cohort 的生产验收结果；
- CI 频率、retention、签名、审批和真实硬件门禁 policy 的具体值。

这些 Authority unknown 可以由后续 decision ticket 解决。实现不得在代码中放置一个未获批准的全局默认值；报告降级策略必须按 ADR 0038 版本化、可重放并与 Authority Result 分离。

## 7. Fixture 与生产证据边界

- #5–#7 原型均为 decision-only/throwaway；生产实现不得复制或演化其源码。
- #5 的 128/201/512 和二维数值用于查询契约测试，不是生产能力值。
- #6 的约 6.02%、40% 和 9.097 us 用于 Verdict fixture，不是通用阈值。
- #7 的 `5.553976/51.632/53.232/92.814479 ms` 只验证四轴字段、显式 schedule 和
  ledger 守恒。其中 `51.632 ms` 是两个层、24 个 synthetic candidate nodes 的聚合
  critical path，不是单一 MatMul，也不是 calibrated M4 Operator Frontier。
- 十几毫秒的 functional demo、#7 frozen sample 与 diagnostic trace 属于不同 run/lane；
  未完成同 cohort、same contract、Baseline Timing 和 preflight 对齐前不得互相替代。

## 8. Conformance 测试

最高价值接缝是：给定冻结、digest-verifiable 的 Diagnostic Evidence Bundle 与显式 policy
versions，执行诊断后读取机器结果与同源报告。测试只断言外部行为，不绑定内部类层次。

| Test ID | 外部行为 | 覆盖需求 |
| --- | --- | --- |
| CT-001 | 四轴同时存在；慢 Observation 不改前三层；Physical Floor 无 prediction error | BND-001..004 |
| CT-002 | 1D exact/continuous query；域外与 regime conflict unknown | SUR-001..003, SUR-007 |
| CT-003 | 2D barycentric query；false bounding box、hole、long edge、degenerate cell 拒绝 | SUR-002..004 |
| CT-004 | covariance Anchor term 可复算；缺 calibration/combination policy 为 unknown；旧版本可回放 | SUR-005..006, DEB-003 |
| CT-005 | provisional/QUALIFIED/ACTIVE/revoke/retract 与较慢样本不降 Anchor | ANC-001..004 |
| CT-006 | C1/缺邻域不得 shift；完整 C2/C3+holdout+3 sessions 才允许 shift | ANC-005, DIA-005 |
| CT-007 | 缺 required field 阻止 promotion；缺 optional counter 只使 attribution unknown；不填零 | HWC-001, HWC-004..007 |
| CT-008 | Baseline/Diagnostic lane 分离；无 overhead ablation 时 profiling timing 不晋级 | HWC-002 |
| CT-009 | device/thread/rank Completion Boundary 正确；跨时钟绝对 timestamp 被拒绝 | HWC-003 |
| CT-010 | candidate local overlap 与显式 paths 可复算；循环、未知 predecessor、隐式效果拒绝 | SCH-001..003 |
| CT-011 | exclusive leaves+residual 恢复 E2E；parent 不加和；counterfactual 守恒 | SCH-004..005, DEB-004 |
| CT-012 | predicted/observed Top 10 独立；uncertainty 与 materiality 双门触发 | DIA-001..003 |
| CT-013 | #6 correct faster alternative 得到 headroom；257³ C1 反例 insufficient | DIA-004, DIA-006, DIA-008 |
| CT-014 | standalone near Frontier 且显式 ablation 回收 excess，得到 integration overhead 且不降 Frontier | DIA-007 |
| CT-015 | 慢样本不能确认 bug；直接可复现 correctness/contract failure 才 confirmed bug | DIA-009 |
| CT-016 | suspected regression policy 缺失时不自动 emit，保留 evidence 并 fail closed | DIA-010 |
| CT-017 | 相同 Bundle+policy 结果确定；篡改 digest 失败；新 run/version 不覆盖旧结果 | DEB-001..004 |
| CT-018 | 机器结果与 HTML/JSON/CSV 同源；四轴、Authority unknown、Report Values、Evidence Grade、Top 10、gates、residual、cohort、lane 可下钻 | BND-001, DIA-004, DEB-002..004 |
| CT-019 | 缺 policy/version 或输入标记 prototype-only 时拒绝 promotion；Authority unknown 不被默认值覆盖，降级 Report Value 不改变 promotion 状态 | GOV-001..003 |

### MUST 到执行门的完整映射

| Requirement family | Schema constraint | Behavior tests |
| --- | --- | --- |
| BND-* | 四个独立 result roles、kind/status/uncertainty/evidence refs | CT-001, CT-018 |
| ANC-* | validity/role state、coverage、holdout、transition history、digests | CT-005, CT-006 |
| SUR-* | domain key、family/cohort/version、cell/anchors/weights、uncertainty、reason code | CT-002..004 |
| HWC-* | identity、Completion Boundary、lane/profile、capability field status、preflight/cohort | CT-007..009 |
| SCH-* | candidates、typed edges/claims、transformations、exclusive ledger/residual | CT-010, CT-011 |
| DIA-* | Top 10 membership、Trigger inputs、probe contract、Verdict gate states/evidence refs | CT-012..016 |
| DEB-* | manifest roles、schema/version、digests、lineage、derivations、immutable identities | CT-017, CT-018 |
| GOV-* | policy identity/scope/revalidation、unknown reason、prototype-only/promotion eligibility | CT-019 |

因此每个规范 MUST 都有 schema constraint、黑盒 behavior test，或两者同时作为执行门；没有仅靠
报告文案才能发现的规范约束。
