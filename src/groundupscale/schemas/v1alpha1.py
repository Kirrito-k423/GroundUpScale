"""Strict v1alpha1 YAML document models.

Pydantic models are the executable schema source. ``document_schema`` exposes
their JSON Schema representation for Web forms and CI schema export.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)


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
    supported_operations: tuple[Literal["MatMul"], ...] = Field(min_length=1)
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
    ]
    resource: str = Field(min_length=1)
    dtype: Literal["float32"]
    shapes: tuple[tuple[PositiveInt, ...], ...] = Field(min_length=10)
    alignment_boundaries: tuple[PositiveInt, ...] = Field(min_length=1)
    thread_counts: tuple[PositiveInt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_shapes(self) -> HardwareProbeSpec:
        expected_rank = 3 if self.kind == "matrix_multiply" else 1
        invalid = [shape for shape in self.shapes if len(shape) != expected_rank]
        if invalid:
            raise ValueError(
                f"{self.kind} requires rank-{expected_rank} benchmark Shapes"
            )
        if len(set(self.shapes)) != len(self.shapes):
            raise ValueError("hardware probe Shapes must be distinct")
        shapes = set(self.shapes)
        for boundary in self.alignment_boundaries:
            required = {
                ((value,) * expected_rank)
                for value in (boundary - 1, boundary, boundary + 1)
            }
            if not required <= shapes:
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


class AnalysisPlanBody(StrictModel):
    workload: SpecReference
    analysis_case: SpecReference
    deployment_intent: SpecReference
    hardware: tuple[SpecReference, ...] = Field(min_length=1)
    hardware_capability_profiles: tuple[SpecReference, ...] = ()
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
    "AnalysisPlan": AnalysisPlanDocument,
}


def document_schema(kind: str) -> dict[str, Any]:
    """Return JSON Schema for a registered human-authored document kind."""
    try:
        document_type = DOCUMENT_TYPES[kind]
    except KeyError as error:
        raise ValueError(f"unknown spec kind: {kind}") from error
    return document_type.model_json_schema()
