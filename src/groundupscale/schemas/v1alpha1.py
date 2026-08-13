"""Strict v1alpha1 YAML document models.

Pydantic models are the executable schema source. ``document_schema`` exposes
their JSON Schema representation for Web forms and CI schema export.
"""

from __future__ import annotations

from math import isclose, isfinite, prod
from statistics import median, quantiles, stdev
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from groundupscale.ir.common import content_fingerprint


API_VERSION = "groundupscale.dev/v1alpha1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Metadata(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    version: str = Field(min_length=1)
    description: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class SpecReference(StrictModel):
    path: str = Field(min_length=1)
    version: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


ShapeDimension: TypeAlias = int | str


class TensorSpec(StrictModel):
    dtype: Literal["float32", "bfloat16", "float16", "int64", "bool"]
    shape: tuple[ShapeDimension, ...] = Field(min_length=1)
    layout: str = Field(default="contiguous", min_length=1)


class PortSpec(StrictModel):
    name: str = Field(min_length=1)
    tensor: TensorSpec


class StateSpec(StrictModel):
    name: str = Field(min_length=1)
    role: Literal["parameter", "buffer", "cache"] = "parameter"
    tensor: TensorSpec
    trainable: bool = True


class SymbolSpec(StrictModel):
    type: Literal["integer"]
    minimum: int | None = None
    maximum: int | None = None


class CallStepSpec(StrictModel):
    kind: Literal["call"]
    id: str = Field(min_length=1)
    target: str = Field(min_length=1)
    entrypoint: str = "forward"
    inputs: dict[str, str]
    outputs: dict[str, str]


class RepeatCallStepSpec(StrictModel):
    kind: Literal["repeat_call"]
    id: str = Field(min_length=1)
    group: str = Field(min_length=1)
    entrypoint: str = "forward"
    input_port: str
    output_port: str
    initial: str
    result: str


EntrypointStepSpec: TypeAlias = Annotated[
    CallStepSpec | RepeatCallStepSpec, Field(discriminator="kind")
]


class EntrypointSpec(StrictModel):
    name: str = Field(min_length=1)
    inputs: tuple[PortSpec, ...]
    outputs: tuple[PortSpec, ...]
    steps: tuple[EntrypointStepSpec, ...]


AttributeValue: TypeAlias = str | int | float | bool


class PrimitiveModuleSpec(StrictModel):
    id: str = Field(min_length=1)
    kind: Literal["primitive"]
    operation: Literal[
        "MatMul", "Add", "RMSNorm", "Softmax", "SiLU", "Mul", "View", "Transpose"
    ]
    inputs: tuple[PortSpec, ...]
    outputs: tuple[PortSpec, ...]
    state: tuple[StateSpec, ...] = ()
    attributes: dict[str, AttributeValue] = Field(default_factory=dict)


class ModuleRepeatSpec(StrictModel):
    id: str = Field(min_length=1)
    kind: Literal["repeat"]
    count: PositiveInt
    id_template: str = Field(min_length=1)
    template: PrimitiveModuleSpec | CompositeModuleSpec


class CompositeModuleSpec(StrictModel):
    id: str = Field(min_length=1)
    kind: Literal["composite"]
    children: tuple[PrimitiveModuleSpec | CompositeModuleSpec | ModuleRepeatSpec, ...]
    entrypoints: tuple[EntrypointSpec, ...]


ModuleRepeatSpec.model_rebuild()
CompositeModuleSpec.model_rebuild()


class ModelSpecBody(StrictModel):
    symbols: dict[str, SymbolSpec] = Field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    root: CompositeModuleSpec


class ModelSpecDocument(StrictModel):
    apiVersion: Literal[API_VERSION]
    kind: Literal["ModelSpec"]
    metadata: Metadata
    spec: ModelSpecBody


class ArtifactSpec(StrictModel):
    name: str = Field(min_length=1)
    tensor: TensorSpec
    role: Literal["input", "output", "intermediate", "state"] = "intermediate"


class ModelCallNodeSpec(StrictModel):
    id: str = Field(min_length=1)
    kind: Literal["model_call"]
    model: SpecReference
    entrypoint: str = Field(min_length=1)
    inputs: dict[str, str]
    outputs: dict[str, str]


class SequenceNodeSpec(StrictModel):
    id: str = Field(min_length=1)
    kind: Literal["sequence"]
    children: tuple[ModelCallNodeSpec | SequenceNodeSpec, ...]


SequenceNodeSpec.model_rebuild()
WorkloadNodeSpec: TypeAlias = Annotated[
    ModelCallNodeSpec | SequenceNodeSpec, Field(discriminator="kind")
]


class WorkloadSpecBody(StrictModel):
    artifacts: tuple[ArtifactSpec, ...]
    root: WorkloadNodeSpec


class WorkloadSpecDocument(StrictModel):
    apiVersion: Literal[API_VERSION]
    kind: Literal["WorkloadSpec"]
    metadata: Metadata
    spec: WorkloadSpecBody


class FixedShapeProfile(StrictModel):
    kind: Literal["fixed"]
    bindings: dict[str, PositiveInt]
    dtype: Literal["float32", "bfloat16", "float16"]


class FixedIterationsDriver(StrictModel):
    kind: Literal["fixed_iterations"]
    warmup_iterations: int = Field(ge=0)
    measured_iterations: PositiveInt


class IterationObservationWindow(StrictModel):
    kind: Literal["iterations"]
    value: PositiveInt


class AnalysisCaseBody(StrictModel):
    shape: FixedShapeProfile
    driver: FixedIterationsDriver
    observation_window: IterationObservationWindow


class AnalysisCaseDocument(StrictModel):
    apiVersion: Literal[API_VERSION]
    kind: Literal["AnalysisCase"]
    metadata: Metadata
    spec: AnalysisCaseBody


class StrategyConfiguration(StrictModel):
    type: str = Field(pattern=r"^[a-z0-9.-]+/[a-zA-Z0-9._-]+$")
    version: str = Field(min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)


class DeploymentBinding(StrictModel):
    scope: str = Field(min_length=1)
    placement: str = Field(min_length=1)
    strategies: tuple[StrategyConfiguration, ...] = ()


class DeploymentIntentBody(StrictModel):
    bindings: tuple[DeploymentBinding, ...]


class DeploymentIntentDocument(StrictModel):
    apiVersion: Literal[API_VERSION]
    kind: Literal["DeploymentIntent"]
    metadata: Metadata
    spec: DeploymentIntentBody


class CapabilityEvidence(StrictModel):
    source_kind: Literal["vendor_official", "isa_derived"]
    title: str = Field(min_length=1)
    url: str = Field(min_length=1, pattern=r"^https://")
    accessed_on: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class CpuCorePool(StrictModel):
    kind: Literal["performance", "efficiency"]
    count: PositiveInt


class CpuVectorCapability(StrictModel):
    isa: Literal["neon"]
    register_bits: PositiveInt
    fp32_fma_flops_per_instruction: PositiveInt
    fp64_fma_flops_per_instruction: PositiveInt
    evidence: tuple[CapabilityEvidence, ...] = Field(min_length=1)


class TheoreticalRate(StrictModel):
    value: PositiveFloat | None = None
    status: Literal["vendor_published", "unknown"]
    reason: str | None = None
    evidence: tuple[CapabilityEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_value_status(self) -> TheoreticalRate:
        if self.status == "unknown" and self.value is not None:
            raise ValueError("unknown theoretical rate must not contain a value")
        if self.status == "vendor_published" and self.value is None:
            raise ValueError("vendor-published theoretical rate requires a value")
        if self.status == "unknown" and not self.reason:
            raise ValueError("unknown theoretical rate requires a reason")
        return self


class CpuTheoreticalCompute(StrictModel):
    fp32_flops_per_second: TheoreticalRate
    fp64_flops_per_second: TheoreticalRate
    matrix_operations_per_second: TheoreticalRate


class UnifiedMemoryCapability(StrictModel):
    peak_bandwidth_bytes_per_second: PositiveFloat
    scope: Literal["soc_shared", "device_dedicated"]
    evidence: tuple[CapabilityEvidence, ...] = Field(min_length=1)


class NpuTheoreticalCompute(StrictModel):
    fp32_flops_per_second: TheoreticalRate


class NpuDedicatedMemoryCapability(StrictModel):
    capacity_bytes: PositiveInt
    capacity_basis: str = Field(min_length=1)
    capacity_evidence: tuple[CapabilityEvidence, ...] = Field(min_length=1)
    peak_bandwidth_bytes_per_second: TheoreticalRate
    scope: Literal["device_dedicated"]


class NpuHardwareCapabilities(StrictModel):
    architecture: Literal["ascend"]
    supported_operations: tuple[
        Literal[
            "MatMul",
            "Add",
            "RMSNorm",
            "Softmax",
            "SiLU",
            "Mul",
            "View",
            "Transpose",
        ],
        ...,
    ] = Field(min_length=1)
    supported_dtypes: tuple[Literal["float32"], ...] = Field(min_length=1)
    theoretical_compute: NpuTheoreticalCompute
    dedicated_memory: NpuDedicatedMemoryCapability


class CpuHardwareCapabilities(StrictModel):
    architecture: Literal["arm64"]
    core_pools: tuple[CpuCorePool, ...] = Field(min_length=1)
    core_topology_evidence: tuple[CapabilityEvidence, ...] = Field(min_length=1)
    vector: CpuVectorCapability
    theoretical_compute: CpuTheoreticalCompute
    unified_memory: UnifiedMemoryCapability


class HardwareDevice(StrictModel):
    id: str = Field(min_length=1)
    kind: Literal["cpu", "gpu", "npu"]
    vendor: str = Field(min_length=1)
    model: str = Field(min_length=1)
    compute_units: PositiveInt | None = None
    memory_bytes: PositiveInt
    capabilities: CpuHardwareCapabilities | NpuHardwareCapabilities | None = None

    @model_validator(mode="after")
    def validate_capabilities(self) -> HardwareDevice:
        if self.kind in {"cpu", "gpu"} and self.compute_units is None:
            raise ValueError("CPU and GPU devices require compute_units")
        if self.capabilities is None:
            return self
        if isinstance(self.capabilities, CpuHardwareCapabilities) and self.kind != "cpu":
            raise ValueError("CpuHardwareCapabilities require kind=cpu")
        if isinstance(self.capabilities, NpuHardwareCapabilities) and self.kind != "npu":
            raise ValueError("NpuHardwareCapabilities require kind=npu")
        if isinstance(self.capabilities, CpuHardwareCapabilities):
            declared_cores = sum(pool.count for pool in self.capabilities.core_pools)
        else:
            declared_cores = None
        if declared_cores is not None and declared_cores != self.compute_units:
            raise ValueError(
                "CPU core-pool count must equal the device compute_units"
            )
        if (
            isinstance(self.capabilities, NpuHardwareCapabilities)
            and self.capabilities.dedicated_memory.capacity_bytes != self.memory_bytes
        ):
            raise ValueError("NPU dedicated-memory capacity must equal memory_bytes")
        return self


class HardwareSpecBody(StrictModel):
    devices: tuple[HardwareDevice, ...] = Field(min_length=1)


class HardwareSpecDocument(StrictModel):
    apiVersion: Literal[API_VERSION]
    kind: Literal["HardwareSpec"]
    metadata: Metadata
    spec: HardwareSpecBody


class FabricNode(StrictModel):
    id: str = Field(min_length=1)
    hardware: str = Field(min_length=1)
    device: str = Field(min_length=1)


class FabricLink(StrictModel):
    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    kind: Literal["unified_memory", "pcie", "network"]
    bandwidth_bytes_per_second: PositiveFloat
    latency_seconds: float = Field(ge=0)


class FabricGraphBody(StrictModel):
    nodes: tuple[FabricNode, ...] = Field(min_length=1)
    links: tuple[FabricLink, ...]


class FabricGraphDocument(StrictModel):
    apiVersion: Literal[API_VERSION]
    kind: Literal["FabricGraph"]
    metadata: Metadata
    spec: FabricGraphBody


class BenchmarkDefinition(StrictModel):
    id: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    mode: Literal["operator", "module", "e2e"]
    warmup_iterations: int = Field(ge=0)
    samples: PositiveInt


class BenchmarkCaseBody(StrictModel):
    cases: tuple[BenchmarkDefinition, ...] = Field(min_length=1)


class BenchmarkCaseDocument(StrictModel):
    apiVersion: Literal[API_VERSION]
    kind: Literal["BenchmarkCase"]
    metadata: Metadata
    spec: BenchmarkCaseBody


class HardwareBenchmarkTarget(StrictModel):
    hardware: str = Field(min_length=1)
    device: str = Field(min_length=1)


class HardwareProbeSpec(StrictModel):
    id: str = Field(min_length=1)
    kind: Literal[
        "scalar_fma",
        "vector_fma",
        "matrix_multiply",
        "memory_copy",
        "memory_triad",
        "reduction_max",
        "reduction_sum",
        "elementwise_subtract",
        "elementwise_divide",
        "elementwise_exp",
        "elementwise_square",
        "scalar_divide",
        "scalar_add",
        "scalar_rsqrt",
        "elementwise_multiply",
        "memory_row_reduction",
        "memory_broadcast",
        "memory_elementwise",
        "memory_row_scalar",
    ]
    resource: str = Field(min_length=1)
    dtype: Literal["float32"]
    shapes: tuple[tuple[PositiveInt, ...], ...] = Field(min_length=10)
    alignment_boundaries: tuple[PositiveInt, ...] = Field(min_length=1)
    thread_counts: tuple[PositiveInt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_shapes(self) -> HardwareProbeSpec:
        row_probe_kinds = {
            "reduction_max",
            "reduction_sum",
            "elementwise_subtract",
            "elementwise_divide",
            "memory_row_reduction",
            "memory_broadcast",
        }
        expected_rank = (
            3
            if self.kind == "matrix_multiply"
            else 2
            if self.kind in row_probe_kinds
            else 1
        )
        invalid = [shape for shape in self.shapes if len(shape) != expected_rank]
        if invalid:
            raise ValueError(
                f"{self.kind} requires rank-{expected_rank} benchmark Shapes"
            )
        if len(set(self.shapes)) != len(self.shapes):
            raise ValueError("hardware probe Shapes must be distinct")
        shapes = set(self.shapes)
        for boundary in self.alignment_boundaries:
            values = (boundary - 1, boundary, boundary + 1)
            boundary_covered = (
                {((value,) * expected_rank) for value in values} <= shapes
                if expected_rank != 2
                else all(
                    any(shape[-1] == value for shape in shapes)
                    for value in values
                )
            )
            if not boundary_covered:
                raise ValueError(
                    "hardware probe must include alignment boundary triplet "
                    f"{boundary - 1}/{boundary}/{boundary + 1}"
                )
        return self


class HardwareBenchmarkSuiteBody(StrictModel):
    target: HardwareBenchmarkTarget
    warmup_iterations: int = Field(ge=0)
    samples: int = Field(ge=4)
    target_window_ms: PositiveFloat
    maximum_inner_iterations: PositiveInt
    probes: tuple[HardwareProbeSpec, ...] = Field(min_length=1)


class HardwareBenchmarkSuiteDocument(StrictModel):
    apiVersion: Literal[API_VERSION]
    kind: Literal["HardwareBenchmarkSuite"]
    metadata: Metadata
    spec: HardwareBenchmarkSuiteBody


class HardwareCapabilitySourceSuite(StrictModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class HardwareCapabilitySource(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_schema: Literal[
        "groundupscale.dev/hardware-microbenchmark-observation/v1alpha1"
    ] = Field(alias="schema", serialization_alias="schema")
    suite: HardwareCapabilitySourceSuite


class ShapeBestRate(StrictModel):
    shape: tuple[PositiveInt, ...] = Field(min_length=1)
    rate: PositiveFloat
    threads: PositiveInt


class ProbeCapabilityEnvelope(StrictModel):
    probe_id: str = Field(min_length=1)
    distinct_shape_count: int = Field(ge=10)
    shape_p80_rate: PositiveFloat
    shape_p95_rate: PositiveFloat
    shape_best_rates: tuple[ShapeBestRate, ...] = Field(min_length=10)


class HardwareResourceEnvelope(StrictModel):
    resource: str = Field(min_length=1)
    unit: Literal["FLOP/s", "B/s"]
    robust_achievable_rate: PositiveFloat
    optimistic_rate: PositiveFloat
    selected_robust_probe: str = Field(min_length=1)
    selected_optimistic_probe: str = Field(min_length=1)
    aggregation: Literal["max(probe_shape_p80)"]
    probe_envelopes: tuple[ProbeCapabilityEnvelope, ...] = Field(min_length=1)


class HardwareCohortEvidence(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_name: Literal["groundupscale.dev/hardware-cohort/v1alpha1"] = Field(
        alias="schema", serialization_alias="schema"
    )


class HardwareCapabilityValidityDomain(StrictModel):
    operation_classes: tuple[str, ...] = Field(min_length=1)
    dtype: Literal["float32"]
    layout: str = Field(min_length=1)
    logical_device: str = Field(min_length=1)
    execution_mode: str = Field(min_length=1)
    shape_support: Literal["observed-stratified-shapes-only"]


class HardwareCapabilityUncertainty(StrictModel):
    method: Literal["per-shape-median-cross-shape-quantiles"]
    robust_quantile: float = Field(gt=0, lt=1)
    optimistic_quantile: float = Field(gt=0, lt=1)
    maximum_iqr_over_median: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_quantile_order(self) -> HardwareCapabilityUncertainty:
        if self.optimistic_quantile <= self.robust_quantile:
            raise ValueError("optimistic quantile must exceed robust quantile")
        return self


class HardwareCapabilityQuality(StrictModel):
    status: Literal["qualified", "exploratory", "quarantined"]
    reason_codes: tuple[str, ...]
    eligible_shape_count_by_resource: dict[str, PositiveInt]


class HardwareCapabilityProfileBody(StrictModel):
    target: HardwareBenchmarkTarget
    hardware_cohort: str = Field(min_length=1)
    cohort_evidence: HardwareCohortEvidence | None = None
    environment: dict[str, Any]
    validity_domain: HardwareCapabilityValidityDomain | None = None
    uncertainty: HardwareCapabilityUncertainty | None = None
    quality: HardwareCapabilityQuality | None = None
    source: HardwareCapabilitySource
    resources: tuple[HardwareResourceEnvelope, ...] = Field(min_length=1)


class HardwareCapabilityProfileDocument(StrictModel):
    apiVersion: Literal[API_VERSION]
    kind: Literal["HardwareCapabilityProfile"]
    metadata: Metadata
    spec: HardwareCapabilityProfileBody


class OperatorFrontierSource(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_schema: Literal[
        "groundupscale.dev/operator-frontier-observation/v1alpha1"
    ] = Field(alias="schema", serialization_alias="schema")


class OperatorFrontierSessionEvidence(StrictModel):
    run_id: str = Field(min_length=1)
    process_id: PositiveInt
    run_bundle: str = Field(min_length=1)
    run_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    correctness_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hardware_cohort: str = Field(min_length=1)
    candidate_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_corpus_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_contract_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    correctness_oracle_policy_id: Literal["matmul-fp32-float64-oracle-v1"]
    timing_scope: Literal["host_visible_completion"]
    completion_boundary: Literal["synchronous-cpu-call-return"]
    timer_source: Literal["time.perf_counter_ns"]
    timer_resolution_ns: PositiveFloat
    instrumentation_profile: Literal["benchmark"]
    warmup_iterations: int = Field(ge=500)
    warmup_window_samples_ns: tuple[PositiveFloat, ...] = Field(min_length=7)
    warmup_median_drift: float = Field(ge=0, le=0.05)
    timed_duration_ns: PositiveFloat
    median_ns: PositiveFloat
    iqr_over_median: float = Field(ge=0, le=0.03)
    sample_count: PositiveInt
    samples_ns: tuple[PositiveFloat, ...] = Field(min_length=4)

    @model_validator(mode="after")
    def validate_raw_timing_derivation(self) -> OperatorFrontierSessionEvidence:
        values = [float(value) for value in self.samples_ns]
        rederived_median = float(median(values))
        quartiles = quantiles(values, n=4, method="inclusive")
        rederived_iqr_ratio = float(
            (quartiles[2] - quartiles[0]) / rederived_median
        )
        if self.sample_count != len(values):
            raise ValueError("frontier session sample_count must match raw samples")
        if not isclose(float(self.median_ns), rederived_median):
            raise ValueError("frontier session median must be rederived from raw samples")
        if not isclose(float(self.iqr_over_median), rederived_iqr_ratio):
            raise ValueError("frontier session IQR must be rederived from raw samples")
        return self


class OperatorFrontierSessionSets(StrictModel):
    search: tuple[OperatorFrontierSessionEvidence, ...] = Field(min_length=3)
    holdout: tuple[OperatorFrontierSessionEvidence, ...] = Field(min_length=3)


class OperatorFrontierQualificationPolicy(StrictModel):
    policy_id: Literal["exact-shape-operator-frontier-qualification"]
    version: Literal["2.0.0"]
    minimum_search_sessions: Literal[3]
    minimum_holdout_sessions: Literal[3]
    maximum_session_iqr_over_median: Literal[0.03]
    maximum_session_median_relative_range: Literal[0.05]
    maximum_search_holdout_relative_gap: Literal[0.05]
    minimum_warmup_iterations: Literal[500]
    maximum_warmup_median_drift: Literal[0.05]
    minimum_samples: Literal[20]
    minimum_windows_per_sample: Literal[5]
    minimum_timed_duration_ns: Literal[100000000]
    estimator: Literal["median(independent_holdout_session_medians)"]
    uncertainty: Literal[
        "sample-standard-deviation(independent_holdout_session_medians)"
    ]


class OperatorFrontierCandidateCoverage(StrictModel):
    level: Literal["C0_SINGLE"]
    scope: Literal["declared-runtime-candidate-family"]
    evaluated_candidate_families: tuple[str, ...] = Field(min_length=1)
    selected_candidate_family: str = Field(min_length=1)
    evaluated_candidate_digests: tuple[str, ...] = Field(min_length=1)
    selected_candidate_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    limitation: Literal[
        "does-not-establish-global-optimum-or-support-frontier-shift"
    ]


class ExactTensorExecutionContract(StrictModel):
    shape: tuple[PositiveInt, ...] = Field(min_length=1)
    stride: tuple[int, ...] = Field(min_length=1)
    dtype: Literal["float32", "bfloat16", "float16", "int64", "bool"]
    layout: Literal["row-major-contiguous", "strided"]
    minimum_alignment_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_stride(self) -> ExactTensorExecutionContract:
        if len(self.stride) != len(self.shape):
            raise ValueError("execution-contract stride rank must match tensor rank")
        if any(value < 0 for value in self.stride):
            raise ValueError("execution-contract stride values must be non-negative")
        if self.layout == "row-major-contiguous":
            expected: list[int] = []
            running = 1
            for dimension in reversed(self.shape):
                expected.append(running)
                running *= dimension
            if self.stride != tuple(reversed(expected)):
                raise ValueError(
                    "execution-contract contiguous stride must match tensor shape"
                )
        return self


class ExactOperatorExecutionContract(StrictModel):
    execution_schema: Literal[
        "groundupscale.dev/operator-execution-contract/v1alpha1"
    ] = Field(alias="schema", serialization_alias="schema")
    status: Literal["resolved"]
    operand_contracts: tuple[ExactTensorExecutionContract, ...] = Field(
        min_length=1
    )
    result_contract: ExactTensorExecutionContract
    execution_mode: Literal["eager"]
    cache_state: Literal["warm-reused-inputs-and-weights"]
    working_set_bytes: PositiveInt
    concurrency: Literal["single-operator-no-overlap"]
    execution_contract_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_digest_and_working_set(self) -> ExactOperatorExecutionContract:
        dtype_bytes = {
            "float32": 4,
            "bfloat16": 2,
            "float16": 2,
            "int64": 8,
            "bool": 1,
        }
        derived_working_set = sum(
            prod(tensor.shape) * dtype_bytes[tensor.dtype]
            for tensor in (*self.operand_contracts, self.result_contract)
        )
        if self.working_set_bytes != derived_working_set:
            raise ValueError(
                "execution-contract working_set_bytes must match exact tensors"
            )
        body = self.model_dump(mode="json", by_alias=True)
        authored_digest = body.pop("execution_contract_digest")
        if authored_digest != content_fingerprint(body):
            raise ValueError(
                "execution-contract digest must bind the canonical contract body"
            )
        return self


class ExactMatmulCorrectnessOracle(StrictModel):
    policy_id: Literal["matmul-fp32-float64-oracle-v1"]
    version: Literal["1.0.0"]
    provider: Literal["torch.float64.matmul"]
    atol: float
    rtol: float
    accumulation_dtype: Literal["float64"]
    invariants: tuple[Literal["shape-exact", "finite-output"], ...]

    @model_validator(mode="after")
    def validate_policy(self) -> ExactMatmulCorrectnessOracle:
        if (
            not isclose(self.atol, 1e-5)
            or not isclose(self.rtol, 1e-4)
            or self.invariants != ("shape-exact", "finite-output")
        ):
            raise ValueError("exact MatMul correctness Oracle policy mismatch")
        return self


class ExactMatmulCorrectnessEvidence(StrictModel):
    correctness_schema: Literal[
        "groundupscale.dev/operator-correctness-evidence/v1alpha1"
    ] = Field(alias="schema", serialization_alias="schema")
    status: Literal["passed"]
    candidate_family: str = Field(min_length=1)
    candidate_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_corpus_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle: ExactMatmulCorrectnessOracle
    max_absolute_error: float = Field(ge=0)
    max_relative_error: float = Field(ge=0)
    shape_matches: Literal[True]
    finite: Literal[True]
    actual_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_reported_error(self) -> ExactMatmulCorrectnessEvidence:
        if (
            not isfinite(self.max_absolute_error)
            or not isfinite(self.max_relative_error)
            or self.max_absolute_error > self.oracle.atol
        ):
            raise ValueError(
                "exact MatMul reported error must satisfy the absolute Oracle bound"
            )
        return self


class ExactShapeOperatorFrontierAnchor(StrictModel):
    anchor_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    stable_path_pattern: str = Field(min_length=1)
    semantic_operation: str = Field(min_length=1)
    operand_shapes: tuple[tuple[PositiveInt, ...], ...] = Field(min_length=1)
    result_shape: tuple[PositiveInt, ...] = Field(min_length=1)
    dtype: str = Field(min_length=1)
    layout: str = Field(min_length=1)
    operand_layouts: tuple[str, ...] | None = None
    result_layout: str | None = None
    operand_strides: tuple[tuple[int, ...], ...] = Field(min_length=1)
    result_stride: tuple[int, ...] = Field(min_length=1)
    minimum_alignment_bytes: PositiveInt
    working_set_bytes: PositiveInt
    threads: PositiveInt
    interop_threads: PositiveInt
    execution_mode: str = Field(min_length=1)
    candidate_family: str = Field(min_length=1)
    candidate_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_corpus_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_contract_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    timing_scope: Literal["host_visible_completion"]
    completion_boundary: Literal["synchronous-cpu-call-return"]
    instrumentation_profile: Literal["benchmark"]
    observation_validity: Literal["QUALIFIED"]
    frontier_role: Literal["ACTIVE"]
    latency_ns: PositiveFloat
    standard_uncertainty_ns: float = Field(ge=0)
    search_run_ids: tuple[str, ...] = Field(min_length=3)
    holdout_run_ids: tuple[str, ...] = Field(min_length=3)
    measurement_hardware_cohort: str = Field(min_length=1)
    qualification_policy: OperatorFrontierQualificationPolicy
    candidate_coverage: OperatorFrontierCandidateCoverage
    session_evidence: OperatorFrontierSessionSets

    @model_validator(mode="after")
    def validate_qualification_derivation(self) -> ExactShapeOperatorFrontierAnchor:
        if len(self.operand_strides) != len(self.operand_shapes) or any(
            len(stride) != len(shape)
            for stride, shape in zip(
                self.operand_strides, self.operand_shapes, strict=True
            )
        ):
            raise ValueError("frontier stride rank must bind every operand dimension")
        if len(self.result_stride) != len(self.result_shape):
            raise ValueError("frontier stride rank must bind every result dimension")
        if any(
            value < 0
            for stride in (*self.operand_strides, self.result_stride)
            for value in stride
        ):
            raise ValueError("frontier stride values must be non-negative")
        if self.operand_layouts is not None:
            if (
                len(self.operand_layouts) != len(self.operand_shapes)
                or any(not item for item in self.operand_layouts)
                or not self.result_layout
            ):
                raise ValueError("frontier layouts must bind every operand and result")
        elif self.layout == "mixed-explicit" or self.result_layout is not None:
            raise ValueError("mixed frontier layouts require explicit per-tensor layouts")
        resolved_operand_layouts = (
            self.operand_layouts
            if self.operand_layouts is not None
            else (self.layout,) * len(self.operand_shapes)
        )
        resolved_result_layout = self.result_layout or self.layout
        for shape, stride, layout in zip(
            self.operand_shapes,
            self.operand_strides,
            resolved_operand_layouts,
            strict=True,
        ):
            if layout == "row-major-contiguous":
                expected: list[int] = []
                running = 1
                for dimension in reversed(shape):
                    expected.append(running)
                    running *= dimension
                if stride != tuple(reversed(expected)):
                    raise ValueError(
                        "frontier contiguous stride must match operand shape"
                    )
        if resolved_result_layout == "row-major-contiguous":
            expected_result: list[int] = []
            running = 1
            for dimension in reversed(self.result_shape):
                expected_result.append(running)
                running *= dimension
            if self.result_stride != tuple(reversed(expected_result)):
                raise ValueError(
                    "frontier contiguous stride must match result shape"
                )
        search_ids = tuple(item.run_id for item in self.session_evidence.search)
        holdout_ids = tuple(item.run_id for item in self.session_evidence.holdout)
        if self.search_run_ids != search_ids or self.holdout_run_ids != holdout_ids:
            raise ValueError("frontier run IDs must match ordered session evidence")
        if set(search_ids) & set(holdout_ids):
            raise ValueError("frontier search and holdout run IDs must be disjoint")
        search_processes = {item.process_id for item in self.session_evidence.search}
        holdout_processes = {item.process_id for item in self.session_evidence.holdout}
        if (
            len(search_processes) != len(search_ids)
            or len(holdout_processes) != len(holdout_ids)
            or search_processes & holdout_processes
        ):
            raise ValueError("frontier search and holdout processes must be independent")
        holdout_values = [item.median_ns for item in self.session_evidence.holdout]
        if not isclose(float(self.latency_ns), float(median(holdout_values))):
            raise ValueError("frontier latency must be rederived from holdout sessions")
        if not isclose(
            float(self.standard_uncertainty_ns), float(stdev(holdout_values))
        ):
            raise ValueError("frontier uncertainty must be rederived from holdout sessions")
        families = self.candidate_coverage.evaluated_candidate_families
        if families != (self.candidate_family,) or (
            self.candidate_coverage.selected_candidate_family
            != self.candidate_family
        ):
            raise ValueError("frontier candidate coverage must bind the selected family")
        digests = self.candidate_coverage.evaluated_candidate_digests
        if digests != (self.candidate_digest,) or (
            self.candidate_coverage.selected_candidate_digest
            != self.candidate_digest
        ):
            raise ValueError("frontier candidate coverage must bind the selected digest")
        session_identities = {
            (
                item.hardware_cohort,
                item.candidate_digest,
                item.input_corpus_digest,
                item.execution_contract_digest,
                item.timing_scope,
                item.completion_boundary,
                item.instrumentation_profile,
            )
            for item in (*self.session_evidence.search, *self.session_evidence.holdout)
        }
        expected = {
            (
                self.measurement_hardware_cohort,
                self.candidate_digest,
                self.input_corpus_digest,
                self.execution_contract_digest,
                self.timing_scope,
                self.completion_boundary,
                self.instrumentation_profile,
            )
        }
        if session_identities != expected:
            raise ValueError("frontier sessions must bind the exact validity identity")
        return self


class OperatorFrontierProfileBody(StrictModel):
    target: HardwareBenchmarkTarget
    hardware_cohort: str = Field(min_length=1)
    source: OperatorFrontierSource
    anchors: tuple[ExactShapeOperatorFrontierAnchor, ...] = Field(min_length=1)


class OperatorFrontierProfileDocument(StrictModel):
    apiVersion: Literal[API_VERSION]
    kind: Literal["OperatorFrontierProfile"]
    metadata: Metadata
    spec: OperatorFrontierProfileBody


class OperatorFrontierQualificationFailureSource(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_schema: Literal[
        "groundupscale.dev/operator-frontier-qualification-attempts/v1alpha1"
    ] = Field(alias="schema", serialization_alias="schema")


class OperatorFrontierQualificationFailure(StrictModel):
    status: Literal["insufficient_evidence"]
    reason_codes: tuple[str, ...] = Field(min_length=1)
    source: OperatorFrontierQualificationFailureSource


class OperatorFrontierExecutionDomain(StrictModel):
    hardware_cohort: str = Field(min_length=1)
    threads: PositiveInt
    interop_threads: PositiveInt
    execution_mode: str = Field(min_length=1)
    timing_scope: Literal["host_visible_completion"]
    completion_boundary: Literal["synchronous-cpu-call-return"]
    instrumentation_profile: Literal["benchmark"]
    candidate_families: dict[str, str] = Field(min_length=1)
    candidate_digests: dict[str, str] = Field(min_length=1)
    execution_contract_digests: dict[str, str] = Field(min_length=1)
    execution_contracts: dict[str, ExactOperatorExecutionContract] = Field(
        min_length=1
    )
    input_corpus_digests: dict[str, str] = Field(min_length=1)
    qualification_failures: dict[
        str, OperatorFrontierQualificationFailure
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_exact_path_contracts(self) -> OperatorFrontierExecutionDomain:
        key_sets = {
            frozenset(self.candidate_families),
            frozenset(self.candidate_digests),
            frozenset(self.execution_contract_digests),
            frozenset(self.execution_contracts),
            frozenset(self.input_corpus_digests),
        }
        if len(key_sets) != 1:
            raise ValueError(
                "operator Frontier validity maps must bind the same exact Stable Paths"
            )
        for path, contract in self.execution_contracts.items():
            if (
                contract.execution_contract_digest
                != self.execution_contract_digests[path]
                or contract.execution_mode != self.execution_mode
            ):
                raise ValueError(
                    "operator Frontier execution contract body/digest/domain mismatch"
                )
        if set(self.qualification_failures) & set(self.execution_contracts):
            raise ValueError(
                "qualified and failed Frontier identities must use disjoint Stable Paths"
            )
        return self


class AnalysisPlanBody(StrictModel):
    workload: SpecReference
    analysis_case: SpecReference
    deployment_intent: SpecReference
    hardware: tuple[SpecReference, ...] = Field(min_length=1)
    hardware_capability_profiles: tuple[SpecReference, ...] = ()
    operator_frontier_profiles: tuple[SpecReference, ...] = ()
    operator_frontier_execution_domain: OperatorFrontierExecutionDomain | None = None
    fabric_graph: SpecReference
    benchmark_cases: tuple[SpecReference, ...] = Field(min_length=1)


class AnalysisPlanDocument(StrictModel):
    apiVersion: Literal[API_VERSION]
    kind: Literal["AnalysisPlan"]
    metadata: Metadata
    spec: AnalysisPlanBody


SpecDocument: TypeAlias = (
    ModelSpecDocument
    | WorkloadSpecDocument
    | AnalysisCaseDocument
    | DeploymentIntentDocument
    | HardwareSpecDocument
    | FabricGraphDocument
    | BenchmarkCaseDocument
    | HardwareBenchmarkSuiteDocument
    | HardwareCapabilityProfileDocument
    | OperatorFrontierProfileDocument
    | AnalysisPlanDocument
)


DOCUMENT_TYPES: dict[str, type[StrictModel]] = {
    "ModelSpec": ModelSpecDocument,
    "WorkloadSpec": WorkloadSpecDocument,
    "AnalysisCase": AnalysisCaseDocument,
    "DeploymentIntent": DeploymentIntentDocument,
    "HardwareSpec": HardwareSpecDocument,
    "FabricGraph": FabricGraphDocument,
    "BenchmarkCase": BenchmarkCaseDocument,
    "HardwareBenchmarkSuite": HardwareBenchmarkSuiteDocument,
    "HardwareCapabilityProfile": HardwareCapabilityProfileDocument,
    "OperatorFrontierProfile": OperatorFrontierProfileDocument,
    "AnalysisPlan": AnalysisPlanDocument,
}


def document_schema(kind: str) -> dict[str, Any]:
    """Return JSON Schema for a registered human-authored document kind."""
    try:
        document_type = DOCUMENT_TYPES[kind]
    except KeyError as error:
        raise ValueError(f"unknown spec kind: {kind}") from error
    return document_type.model_json_schema()
