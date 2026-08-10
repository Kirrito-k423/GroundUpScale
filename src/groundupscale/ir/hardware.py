"""Hardware-backend candidates and explicitly partial duration estimates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityEvidenceRef:
    source_kind: str
    title: str
    url: str
    accessed_on: str


@dataclass(frozen=True)
class RateAvailability:
    status: str
    value: float | None
    reason: str | None
    evidence: tuple[CapabilityEvidenceRef, ...]


@dataclass(frozen=True)
class DurationAvailability:
    status: str
    value_ns: float | None
    reason: str | None
    required_capability: str
    evidence: tuple[CapabilityEvidenceRef, ...]


@dataclass(frozen=True)
class CpuCapabilitySnapshot:
    architecture: str
    core_pools: tuple[tuple[str, int], ...]
    vector_isa: str
    vector_register_bits: int
    fp32_fma_flops_per_instruction: int
    fp64_fma_flops_per_instruction: int
    fp32_flops_per_second: RateAvailability
    peak_memory_bandwidth_bytes_per_second: float
    memory_bandwidth_scope: str
    evidence: tuple[CapabilityEvidenceRef, ...]


@dataclass(frozen=True)
class NpuCapabilitySnapshot:
    architecture: str
    supported_operations: tuple[str, ...]
    supported_dtypes: tuple[str, ...]
    fp32_flops_per_second: RateAvailability
    peak_memory_bandwidth_bytes_per_second: RateAvailability
    memory_bytes: int
    memory_scope: str
    evidence: tuple[CapabilityEvidenceRef, ...]


@dataclass(frozen=True)
class HardwareCapabilityValidityDomainSnapshot:
    operation_classes: tuple[str, ...]
    dtype: str
    layout: str
    logical_device: str
    execution_mode: str
    shape_support: str


@dataclass(frozen=True)
class HardwareCapabilityUncertaintySnapshot:
    method: str
    robust_quantile: float
    optimistic_quantile: float
    maximum_iqr_over_median: float


@dataclass(frozen=True)
class HardwareCapabilityEnvelopeSnapshot:
    resource: str
    unit: str
    robust_achievable_rate: float
    optimistic_rate: float
    selected_robust_probe: str
    selected_optimistic_probe: str
    profile_name: str
    profile_version: str
    hardware_cohort: str
    source_path: str
    source_sha256: str
    environment_eligible: bool
    quality_status: str = "legacy-unspecified"
    quality_reason_codes: tuple[str, ...] = ()
    validity_domain: HardwareCapabilityValidityDomainSnapshot | None = None
    uncertainty: HardwareCapabilityUncertaintySnapshot | None = None


@dataclass(frozen=True)
class CandidateDurationEstimate:
    model: str
    status: str
    compute_time_ns: float | None
    memory_optimistic_lower_bound_ns: float | None
    empirical_compute_time_ns: float | None
    empirical_memory_time_ns: float | None
    empirical_hardware_floor_ns: float | None
    limiting_resource: str | None
    full_duration_ns: float | None
    formula: str
    missing_capabilities: tuple[str, ...]
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class ImplementationCandidate:
    candidate_id: str
    cost_node_id: str
    stable_path: str
    operation: str
    implementation: str
    flops: int
    compulsory_bytes: int
    materialized_bytes: int
    duration: CandidateDurationEstimate


@dataclass(frozen=True)
class ProgramDurationBounds:
    flops: int
    compulsory_bytes: int
    materialized_bytes: int
    compute_time: DurationAvailability
    memory_optimistic_lower_bound_ns: float | None
    vendor_memory_time_floor_ns: float | None
    empirical_compute_time_ns: float | None
    empirical_memory_time_ns: float | None
    empirical_hardware_floor_ns: float | None
    limiting_resource: str | None
    full_duration_ns: float | None
    formula: str
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class ScopeDurationBounds:
    case_id: str
    scope: str
    operation_count: int
    flops: int
    compulsory_bytes: int
    empirical_compute_time_ns: float | None
    empirical_memory_time_ns: float | None
    empirical_hardware_floor_ns: float | None
    limiting_resource: str | None
    formula: str
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class UnsupportedCostRegion:
    cost_node_id: str
    stable_path: str
    operation: str
    reason: str


@dataclass(frozen=True)
class HardwareBackendPrediction:
    schema: str
    backend_id: str
    backend_version: str
    compilation_fingerprint: str
    placement: str
    hardware: str
    device: str
    status: str
    prediction_complete: bool
    capabilities: CpuCapabilitySnapshot | NpuCapabilitySnapshot
    measured_capabilities: tuple[HardwareCapabilityEnvelopeSnapshot, ...]
    candidates: tuple[ImplementationCandidate, ...]
    scope_bounds: tuple[ScopeDurationBounds, ...]
    program_bounds: ProgramDurationBounds
    unsupported_regions: tuple[UnsupportedCostRegion, ...] = ()
