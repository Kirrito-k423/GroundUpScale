# GroundUpScale

GroundUpScale models the performance and resource behavior of heterogeneous AI
workloads while preserving the reasoning behind every estimate.

## Language

**Performance Model**:
An explainable model that predicts time and resource behavior by composing logical workload structure, deployment intent, hardware capabilities, and measured evidence.
_Avoid_: Simulator, profiler, black-box predictor

**Model IR**:
The deployment-independent logical structure of one model, including its nested components, tensors, shapes, and data dependencies.
_Avoid_: Workload graph, execution graph

**Model Spec**:
A human-authored YAML document describing model modules, hierarchy, explicit configuration, named entrypoints, assertions, and references while omitting safely derivable structure.
_Avoid_: Model IR, Python builder

**State Artifact**:
A typed and versioned logical state consumed or produced by workload stages, such as model weights, optimizer state, KV cache, activations, or checkpoints.
_Avoid_: Model IR, physical replica

**Weight Version**:
A State Artifact containing one identifiable version of parameter values that conform to a Model IR.
_Avoid_: Model definition, deployed model

**Artifact Replica**:
A physical copy or shard of a State Artifact placed on a concrete resource. Multiple replicas may realize the same logical artifact version.
_Avoid_: State Artifact, weight version

**Memory Allocation**:
A concrete buffer reserved in one Capacity Resource to materialize one or more Artifact Replicas or implementation workspaces, including requested, aligned, and allocator-reserved capacity.
_Avoid_: State Artifact, Artifact Replica

**Alias Group**:
A set of logical tensors or Artifact Replicas that share one Memory Allocation and therefore must not be counted as independent physical capacity.
_Avoid_: Replicated state

**Memory Residency**:
The interval and memory domain in which an Artifact Replica is materially available, bounded by allocation, materialization, migration, eviction, recomputation, and release events.
_Avoid_: Logical Artifact lifetime

**Workload IR**:
The hierarchical composition of control structures, temporal composition, long-running services, action nodes, and typed data edges across models and lifecycle stages.
_Avoid_: Scenario parameters, model graph

**Workload Spec**:
A human-authored YAML description of the stable logical process, including model references, lifecycle stages, control composition, Artifact flow, state evolution, and service boundaries.
_Avoid_: Analysis Case, deployment configuration

**Spec Document**:
A versioned, Schema-validated YAML document representing one supported user input kind, such as Model Spec, Workload Spec, Analysis Case, Deployment Intent, Hardware Spec, or Fabric Graph.
_Avoid_: JSON authoring contract, Python builder, exported IR

**Spec Schema**:
The versioned machine-readable contract used to parse, validate, document, migrate, and power structured editing of one YAML Spec Document kind.
_Avoid_: Separate JSON data model

**Spec Envelope**:
The common identity and version boundary of every Spec Document, consisting of its API version, kind, metadata, and typed specification body.
_Avoid_: Per-kind top-level convention

**Spec Reference**:
An explicit reference from one Spec Document to another using a resolvable location and, when reproducibility is required, a pinned version or content digest.
_Avoid_: YAML anchor, implicit global lookup, copied configuration

**Analysis Plan**:
The human-authored YAML entry point that assembles references to one Workload Spec, Analysis Case, Deployment Intent, Fabric Graph, and optional Calibration Profile for a reproducible analysis.
_Avoid_: Analysis Case, prediction report, merged mega-spec

**Run Bundle**:
An immutable, self-describing collection of locked inputs and generated artifacts from one analysis or measurement attempt, indexed by a Run Manifest.
_Avoid_: Mutable output directory, evidence dataset

**Run Manifest**:
The authoritative index of a Run Bundle's artifact roles, locations, Schema versions, content digests, producer lineage, Hardware Cohort, and validity state.
_Avoid_: Directory convention, latest symlink

**Plugin Spec Type**:
A registered, namespaced, versioned type through which a Spec Document selects a plugin capability and supplies Schema-validated configuration.
_Avoid_: Arbitrary extension map, untyped plugin options

**Module Builder**:
The deterministic transformation that validates a Model Spec, expands its declarative construction forms, and emits an explicit Model IR with stable paths and provenance.
_Avoid_: Python model builder, Workload Builder, Entrypoint Lowerer

**Model Repeat**:
A Model Spec construction form that instantiates a module template a fixed number of times and assigns each generated module a stable indexed path.
_Avoid_: Workload Map, runtime loop, copied layer list

**Structural Override**:
A Model Spec refinement that changes selected generated modules by index, range, or other Schema-defined selector, with strictly more-specific refinement and ambiguity rejection.
_Avoid_: File-order override, hidden Python mutation

**Composite Module**:
A model module whose complete structure is declared in YAML by composing registered modules, parameter bindings, Model Repeats, and Structural Overrides.
_Avoid_: Primitive Module, Python model builder

**Primitive Module**:
A registered model module that introduces semantic behavior which cannot be expressed solely by composing existing module kinds and therefore requires plugin-provided Lowering.
_Avoid_: Composite Module, opaque YAML callback

**Spec Expression**:
A pure, strongly typed expression over declared parameters, shapes, constants, and controlled expansion indices that can remain symbolic through compilation.
_Avoid_: Python expression, template engine, environment lookup

**Workload Builder**:
The extensible transformation that validates and completes a Workload Spec into a Workload IR with stable paths, resolved references, typed ports, and checked Artifact flow.
_Avoid_: Semantic Compiler, deployment planner

**Structured Control Node**:
A workload composite that selects, orders, runs concurrently, or repeats child flows. Its canonical forms are Sequence, Parallel, Branch, Loop, and Map.
_Avoid_: Pipeline, Service, Stream

**Map**:
A Structured Control Node that applies one child flow to each member of a collection with a configurable concurrency limit.
_Avoid_: Repeated model layer, Pipeline

**Pipeline**:
A temporal composition whose stages process a succession of inputs with overlap between different inputs.
_Avoid_: Sequence, pipeline-parallel deployment strategy

**Service**:
A long-running workload container that hosts a child flow and exposes streaming ingress and egress.
_Avoid_: Action node, Model Call

**Artifact Edge**:
A data edge that transfers one bounded typed Artifact between workload nodes.
_Avoid_: Stream, control edge

**Stream**:
A continuous data edge carrying a sequence of Artifacts with stable delivery, ordering, and backpressure semantics.
_Avoid_: Structured Control Node, Service

**Action Node**:
A workload leaf that performs one logical action without containing child workload nodes. Canonical actions include ModelCall, Compute, Transfer, Convert, Checkpoint, and Publish.
_Avoid_: Service, Pipeline

**Model Call**:
A Workload IR Action Node that references a complete Model IR or one named entry point and consumes or produces typed Artifacts. It is a workload leaf but expands through the referenced Model IR during compilation.
_Avoid_: Model IR, deployed replica

**Entrypoint Lowerer**:
An extensible transformation that expands one named Model IR entrypoint into a reusable, hardware-independent Semantic Fragment for a Model Call context.
_Avoid_: Module Builder, Hardware Backend

**Semantic Fragment**:
A reusable hardware-independent expansion of one Model Call entrypoint into mathematical and logical operations with typed inputs, outputs, and provenance.
_Avoid_: Complete Semantic IR, Execution IR

**Semantic Compiler**:
The transformation that composes a Workload IR, referenced Semantic Fragments, an Analysis Case, and applicable semantic strategy effects into one complete Semantic IR.
_Avoid_: Entrypoint Lowerer, Execution Planner

**Semantic IR**:
The hardware-independent semantic representation of a complete selected workload, including workload composition, expanded model operations, non-model actions, Artifact and state lifetimes, and cross-stage dependencies.
_Avoid_: Semantic Fragment, Execution IR

**Semantic Region**:
A hierarchical container in Semantic IR that preserves one structured control, workload, model, entrypoint, or module scope while allowing typed dataflow within it.
_Avoid_: Flat DAG, physical execution stage

**Typed Value**:
A hardware-independent Semantic IR value with a declared element type, shape, logical layout, producer, consumers, and optional symbolic constraints.
_Avoid_: Physical buffer, untyped edge

**State Effect**:
An explicit Semantic IR declaration that an operation reads, writes, creates, versions, aliases, materializes, migrates, evicts, or releases logical state.
_Avoid_: Hidden mutation, physical allocation

**Cost Lowerer**:
An extensible transformation that replaces Semantic IR operations with symbolic, hardware-independent resource-demand formulas while preserving dependencies, bounds, assumptions, and provenance.
_Avoid_: Hardware Backend, runtime estimator

**Cost IR**:
The hardware-independent representation of symbolic compute, memory, communication, storage, and host-work demands for a complete selected workload.
_Avoid_: Hardware latency model, Execution IR

**Resource Demand**:
A unit-checked symbolic quantity or formula describing required work, data movement, temporary capacity, or lifetime before a physical implementation is selected.
_Avoid_: Predicted duration, measured utilization

**Cost Formula**:
An explainable expression tree that derives a Resource Demand from named inputs, assumptions, and source operations and may include lower and upper bounds.
_Avoid_: Opaque fitted value

**Logical Communication**:
A hardware-independent semantic operation describing required data exchange, participants, ordering, and synchronization without selecting an algorithm or route.
_Avoid_: Communication Plan, physical link event

**Communication Demand**:
A Cost IR description of logical payload, participant and per-rank volume formulas, layout transition, synchronization constraints, and bounds before algorithm selection.
_Avoid_: Physical link traffic

**Communication Plan**:
The selected collective or transfer algorithm, chunking, channels, concrete participants, Fabric routes, staging, and conversion steps for one Communication Demand.
_Avoid_: Logical Communication, scheduled event

**Physical Communication Event**:
An Execution IR event for a concrete pack, conversion, copy, reduce, send, receive, or synchronization step with physical Resource Claims.
_Avoid_: Communication Demand, Communication Plan

**Hardware Backend**:
An extensible provider that maps supported Cost IR regions to one or more device-specific Implementation Candidates without choosing the global execution plan.
_Avoid_: Execution Planner, Fabric Graph

**Implementation Candidate**:
A device-specific realization of a Cost IR region, including applicability constraints, event templates, fusion boundaries, workspace, resource claims, duration model, uncertainty, and evidence provenance.
_Avoid_: Selected execution event, global schedule

**Resource Claim**:
A typed declaration of capacity, work, slots, or exclusivity required by one physical event, including sharing, allocation bounds, lifetime, affinity, and provenance.
_Avoid_: Scalar utilization, measured occupancy

**Capacity Resource**:
A resource whose simultaneously live claims consume finite capacity, such as HBM, DRAM, storage, or queue space.
_Avoid_: Throughput Resource

**Throughput Resource**:
A shared resource that completes declared work at an allocated rate, such as a compute engine, memory interface, or network link.
_Avoid_: Capacity Resource, measured utilization

**Slot Resource**:
A resource that admits a finite number of concurrent claimants, such as CPU cores, DMA engines, workers, or launch slots.
_Avoid_: Capacity Resource

**Bubble**:
An interval of unused eligible resource capacity classified by its causal dependency, queueing, imbalance, communication, capacity, or scheduling constraint.
_Avoid_: Unexplained idle percentage

**Execution Planner**:
The global compiler stage that selects Implementation Candidates, resolves concrete placement and routes, validates static capacity, and emits unscheduled physical events and constraints for an Execution Horizon.
_Avoid_: Hardware Backend, local cost rule

**Duration Model**:
An explainable hardware-specific prediction of an Implementation Candidate's duration and uncertainty under stated shape, layout, resource, and software conditions.
_Avoid_: Hardware-independent Cost Formula, raw observation

**Resource Physical Floor**:
An algorithm-independent lower bound on duration derived from minimum Resource Demands and evidence-backed physical resource capacities; it may be unattainable by any current implementation.
_Avoid_: Operator prediction, theoretical peak, runtime observation

**Operator Achievable Frontier**:
An evidence-qualified partial mapping from a complete execution domain and Shape to the best correct Implementation Candidate duration and derived rate within one Hardware Cohort.
_Avoid_: Resource Physical Floor, global calibration average, single observation

**Schedule Achievable Frontier**:
An evidence-qualified duration obtained by composing Operator Achievable Frontiers through explicit dependencies, Resource Claims, transformations, and validated schedule choices.
_Avoid_: Sum of operator minima, runtime observation

**Capability Surface**:
A local representation of an Operator Achievable Frontier over validated Shape coordinates, partitioned into Shape Regimes and supported by exact-Shape Frontier Anchors.
_Avoid_: Global interpolation, bounding box, nearest-neighbor estimate

**Frontier Anchor**:
A correct, stable, independently held-out exact-Shape observation whose validity and Frontier role are qualified for one complete execution domain and Hardware Cohort.
_Avoid_: Arbitrary benchmark point, unqualified observation

**Shape Regime**:
A connected Shape domain within which one Implementation Candidate family, execution behavior, and Duration Model remain evidence-qualified.
_Avoid_: Whole Shape sweep, assumed global range

**Shape Regime Boundary**:
An explicit separation between Shape Regimes across which candidate selection or duration behavior may change and prediction remains unknown until independently validated.
_Avoid_: Smoothed discontinuity, implicit extrapolation

**Setup Latency**:
The portion of an Implementation Candidate's duration that does not scale with declared work within one Shape Regime.
_Avoid_: Whole small-Shape duration, scheduling delay

**Ramp Regime**:
A Shape Regime in which Setup Latency remains material, so Effective Rate rises substantially as declared work increases.
_Avoid_: Slow hardware, unqualified small-Shape range

**Steady Regime**:
A Shape Regime in which Setup Latency is no longer material and Effective Rate remains near a qualified asymptotic level as declared work increases.
_Avoid_: Theoretical peak, universal large-Shape range

**Effective Rate**:
Declared work divided by modeled or measured duration under one complete execution domain.
_Avoid_: Model FLOPs Utilization, theoretical peak, hardware capacity

**Model FLOPs Utilization**:
Effective Rate divided by a comparable, evidence-backed theoretical FLOP peak; it is unknown when that peak is unknown or semantically incomparable.
_Avoid_: Effective Rate, empirical-envelope utilization, generic utilization

**Fabric Graph**:
The concrete compute, memory, storage, switch, and interconnect instances available to one analysis together with their topology, capacity, pools, and Hardware Spec references.
_Avoid_: Flat hardware profile, Hardware Spec, Calibration Profile

**Hardware Spec**:
A reusable description of one hardware kind's static capabilities, supported data types and operation classes, theoretical limits, memory hierarchy, ports, and capacity constraints.
_Avoid_: Fabric instance, measured performance profile

**Calibration Profile**:
A versioned evidence-backed set of fitted parameters, uncertainty, validity conditions, and software-environment metadata used by Duration Models for a hardware and implementation context.
_Avoid_: Hardware Spec, raw observation

**Candidate Calibration Profile**:
A versioned Calibration Profile proposal produced by a Calibration Run that has not yet passed its independent Error Budgets and promotion policy.
_Avoid_: Active Calibration Profile, silent model update

**Hardware Cohort**:
A compatibility class of hardware, operating system, runtime, backend, and measurement conditions within which observations and calibration parameters may be compared.
_Avoid_: Hardware marketing name, mixed-device baseline

**Benchmark Case**:
A reproducible measurement target that binds one analysis input, semantic scope, implementation mode, correctness oracle, warmup policy, repetition policy, and expected evidence fields.
_Avoid_: Ad-hoc timing script, Calibration Profile

**Observation Trace**:
Immutable raw or minimally normalized measurements from one real execution, including environment identity, timestamps, shapes, configuration, events, metrics, and collection provenance.
_Avoid_: Calibrated prediction, Hardware Spec

**Instrumentation Profile**:
A declared observation policy selecting benchmark, trace, or targeted deep-probe collection together with its synchronization, metadata, and accepted-overhead rules.
_Avoid_: Ad-hoc print statements, Benchmark Case

**Alignment Map**:
A versioned mapping from measured spans and events to Stable Paths, Semantic IR nodes, Execution IR events, or explicitly unattributed buckets, including match method and confidence.
_Avoid_: Assumed name equality, modified Observation Trace

**Observation Dataset**:
A versioned, quality-checked collection of Observation Traces partitioned for calibration, validation, or regression evaluation.
_Avoid_: Mutable metrics table, Calibration Profile

**Calibration Run**:
A reproducible process that fits a declared Duration Model on one dataset partition and evaluates it against independent evidence and Error Budgets.
_Avoid_: Online silent learning, manual parameter overwrite

**Error Budget**:
A versioned acceptance threshold for prediction error, uncertainty coverage, or drift over a stated metric and validity domain.
_Avoid_: Unrecorded tolerance

**Base Prediction**:
The prediction produced from Cost Formulas and uncalibrated Hardware Backend models before a Calibration Profile is applied.
_Avoid_: Theoretical hardware peak, Calibrated Prediction

**Calibrated Prediction**:
A Base Prediction adjusted by one identified Calibration Profile while preserving the original value, delta, uncertainty, validity domain, and evidence links.
_Avoid_: Overwritten Base Prediction

**Explanation Graph**:
A queryable derived graph connecting predicted or measured metrics to logical scopes, formulas, implementation choices, schedule causes, uncertainty, calibration evidence, and provenance.
_Avoid_: Visualization, another compilation IR

**Metric Derivation**:
The explainable derivation of one metric with its aggregation semantics, contributing scopes or events, assumptions, bounds, uncertainty, and source records.
_Avoid_: Scalar result, arbitrary percentage breakdown

**Error Attribution**:
An evidence-backed classification and decomposition of prediction-versus-observation discrepancy across semantic, cost, backend, scheduling, observation, and environment causes.
_Avoid_: Uniformly distributed residual, calibration update

**Deployment Intent**:
User-supplied constraints that associate selected workload or model scopes with execution strategies, placement requirements, and deployable service policies.
_Avoid_: Global strategy, global hardware configuration

**Strategy Configuration**:
A typed set of user-supplied parameters that selects and configures one Strategy Plugin within a Scope Binding.
_Avoid_: Compiler implementation, untyped option map

**Strategy Plugin**:
An extension that validates a Strategy Configuration and contributes explainable transformations across one or more named compilation phases.
_Avoid_: Boolean feature flag, hard-coded compiler branch

**Compatibility Rule**:
A machine-checkable constraint declared by a Strategy Plugin over applicable scopes, other strategies, artifacts, or hardware capabilities.
_Avoid_: Undocumented limitation

**Generated Effect**:
An explainable structural, lifecycle, placement, communication, transfer, or scheduling change attributed to a Strategy Plugin.
_Avoid_: Hidden mutation

**Provenance Graph**:
The mandatory append-only graph that connects entities across specifications, IRs, implementation selection, execution, and report metrics through immutable Derivation Records.
_Avoid_: Free-form debug log, duplicated node metadata

**Derivation Record**:
An immutable account of one transformation containing input and output identities, plugin and rule versions, applied bindings, formulas or transformations, assumptions, bounds, evidence, considered candidates, rejection reasons, warnings, and validation results.
_Avoid_: Unversioned explanation string

**Transform Proposal**:
An immutable plugin result describing requested additions, replacements, or refinements to a typed IR together with preconditions, conflicts, diagnostics, and Derivation Records for compiler-controlled application.
_Avoid_: In-place plugin mutation

**Compilation Fingerprint**:
A deterministic content identity derived from all effective specifications, IR inputs, Schemas, plugins, rules, strategies, and declared context that can affect one compilation result.
_Avoid_: Git commit alone, random run identifier

**Stable Path**:
A user-facing hierarchical identity that remains meaningful across compatible expansions and allows scopes and explanations to refer to the same logical entity.
_Avoid_: Per-run node identifier

**Definition ID**:
The versioned identity of a reusable module, template, or other declared definition independent of any place where it is instantiated.
_Avoid_: Stable Path, Node ID

**Node ID**:
An immutable identity for one concrete entity in one compilation, linked through Derivation Records to predecessor and successor identities.
_Avoid_: Stable Path

**Lowering Plugin**:
A versioned extension that transforms an immutable typed IR and pinned Compilation Context into a new typed IR, Derivation Records, diagnostics, and validation results.
_Avoid_: In-place mutation hook

**Compilation Context**:
The complete pinned set of schemas, plugin and rule versions, strategy bindings, analysis conditions, hardware inputs, calibration evidence, and declared environmental inputs available to one compilation.
_Avoid_: Ambient process environment, hidden global state

**Rule Pack**:
A versioned declarative collection of typed matching, formula, or rewrite rules that follows the same validation and provenance contract as code-based Lowering Plugins.
_Avoid_: Unversioned configuration snippet

**Lowering Result**:
The new immutable output IR together with mandatory Derivation Records, diagnostics, and validation results produced by one Lowering Plugin invocation.
_Avoid_: Mutated input IR

**Scope Binding**:
The association of a selected workload scope and optional model scope with a partial strategy configuration and placement constraints.
_Avoid_: Global binding

**Binding Resolution**:
The deterministic composition of all Scope Bindings applicable to one target, using inheritance and strictly more-specific refinement while preserving the source of every resolved value.
_Avoid_: Last-wins merge, file-order precedence

**Binding Ambiguity**:
A conflict in which overlapping Scope Bindings are not ordered by containment and assign incompatible values or constraints to the same target field.
_Avoid_: Implicit override

**Resolved Deployment Plan**:
The complete strategy and placement assignment for every selected workload and model scope, including provenance, produced before physical execution events are generated.
_Avoid_: Deployment Intent, Execution IR

**Execution IR**:
The unscheduled physical computation, communication, transfer, synchronization, resource claims, dependencies, and lifetime constraints selected for one Execution Horizon.
_Avoid_: Model IR, Workload IR

**Scheduler Plugin**:
An extensible solver that turns one Execution IR into a concrete predicted Schedule Result under declared queueing, priority, overlap, and contention semantics.
_Avoid_: Execution Planner, Hardware Backend

**Schedule Result**:
A predicted assignment of event start and end times, resource allocations, waits, contention delays, overlaps, bubbles, and capacity timelines for one Execution IR.
_Avoid_: Execution IR, Observation Trace

**Schedule Bound**:
An explainable optimistic, capacity-constrained, or serialized reference bound computed from an Execution IR without being presented as the chosen feasible schedule.
_Avoid_: Schedule Result, measured duration

**Execution Horizon**:
The complete finite span of a repeated or cyclic workload expanded and scheduled into an Execution IR, including required warmup and drain work.
_Avoid_: Observation Window, whole-program assumption

**Observation Window**:
The portion of an Execution Horizon whose events contribute to reported metrics, excluding warmup and drain work when configured.
_Avoid_: Execution Horizon, Driver Profile

**Analysis Case**:
The input and driving conditions applied to one selected logical workload or entrypoint, combining a Shape Profile, Driver Profile, and Observation Window.
_Avoid_: Workload IR, deployment configuration

**Shape Profile**:
The input shapes and data types processed by an Analysis Case. Its canonical forms are Fixed Shape, Shape Sweep, Shape Distribution, and Shape Trace.
_Avoid_: Arrival process, execution count

**Fixed Shape**:
One concrete assignment of all model-relevant input dimensions and data types.
_Avoid_: Shape distribution, dynamic batching

**Shape Sweep**:
A collection of Fixed Shapes evaluated independently to build performance curves or search configurations.
_Avoid_: Shape distribution

**Shape Distribution**:
A probability distribution over model-relevant input dimensions and data types.
_Avoid_: Fixed Shape, Shape Trace, arrival distribution

**Shape Trace**:
A recorded sequence of concrete model-relevant input shapes and data types used for deterministic replay.
_Avoid_: Shape Distribution, arrival trace

**Driver Profile**:
The rule that determines when and how often an Analysis Case submits work. Its canonical forms are Fixed Iterations, Closed Loop, and Open Loop.
_Avoid_: Shape Profile, Workload IR

**Fixed Iterations**:
A Driver Profile that executes a fixed number of steps or iterations, optionally separating warmup from measured work.
_Avoid_: Closed Loop, Open Loop

**Closed Loop**:
A Driver Profile that maintains fixed concurrency by submitting new work when prior work completes.
_Avoid_: Fixed Iterations, Open Loop

**Open Loop**:
A Driver Profile that submits work independently of completion according to a rate, arrival distribution, or timestamp trace.
_Avoid_: Fixed Iterations, Closed Loop
