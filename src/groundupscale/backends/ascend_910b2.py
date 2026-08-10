"""Ascend 910B2 MatMul Resource Physical Floor backend."""

from __future__ import annotations

from math import prod

from groundupscale.backends.apple_m4_cpu import (
    _evidence,
    _measured_capabilities,
    _rate,
    _resolve_device,
    _scope_compulsory_bytes,
    _scope_matches,
)
from groundupscale.ir import content_fingerprint
from groundupscale.ir.cost import CostOperation, CostProgram
from groundupscale.ir.hardware import (
    CandidateDurationEstimate,
    DurationAvailability,
    HardwareBackendPrediction,
    HardwareCapabilityEnvelopeSnapshot,
    HardwareCapabilityUncertaintySnapshot,
    HardwareCapabilityValidityDomainSnapshot,
    ImplementationCandidate,
    NpuCapabilitySnapshot,
    ProgramDurationBounds,
    ScopeDurationBounds,
    UnsupportedCostRegion,
)
from groundupscale.schemas.v1alpha1 import (
    HardwareCapabilityProfileDocument,
    NpuHardwareCapabilities,
)
from groundupscale.specs import AnalysisBundle


BACKEND_ID = "huawei.ascend.910b2.resource-envelope"
BACKEND_VERSION = "v1alpha1"
PREDICTION_SCHEMA = "groundupscale.dev/hardware-backend-prediction/v1alpha1"


def _profile_snapshots(
    profile: HardwareCapabilityProfileDocument,
) -> tuple[HardwareCapabilityEnvelopeSnapshot, ...]:
    spec = profile.spec
    validity = spec.validity_domain
    uncertainty = spec.uncertainty
    quality = spec.quality
    return tuple(
        HardwareCapabilityEnvelopeSnapshot(
            resource=resource.resource,
            unit=resource.unit,
            robust_achievable_rate=float(resource.robust_achievable_rate),
            optimistic_rate=float(resource.optimistic_rate),
            selected_robust_probe=resource.selected_robust_probe,
            selected_optimistic_probe=resource.selected_optimistic_probe,
            profile_name=profile.metadata.name,
            profile_version=profile.metadata.version,
            hardware_cohort=spec.hardware_cohort,
            source_path=spec.source.path,
            source_sha256=spec.source.sha256,
            environment_eligible=spec.environment.get("eligible") is True,
            quality_status=(
                quality.status if quality is not None else "legacy-unspecified"
            ),
            quality_reason_codes=(
                tuple(quality.reason_codes) if quality is not None else ()
            ),
            validity_domain=(
                HardwareCapabilityValidityDomainSnapshot(
                    operation_classes=tuple(validity.operation_classes),
                    dtype=validity.dtype,
                    layout=validity.layout,
                    logical_device=validity.logical_device,
                    execution_mode=validity.execution_mode,
                    shape_support=validity.shape_support,
                )
                if validity is not None
                else None
            ),
            uncertainty=(
                HardwareCapabilityUncertaintySnapshot(
                    method=uncertainty.method,
                    robust_quantile=uncertainty.robust_quantile,
                    optimistic_quantile=uncertainty.optimistic_quantile,
                    maximum_iqr_over_median=(
                        uncertainty.maximum_iqr_over_median
                    ),
                )
                if uncertainty is not None
                else None
            ),
        )
        for resource in spec.resources
    )


def matmul_problem_shape(operation: CostOperation) -> tuple[int, int, int] | None:
    """Return the unbatched (M, K, N) problem represented by one MatMul."""

    if (
        operation.operation != "MatMul"
        or len(operation.operand_types) != 2
        or len(operation.result_types) != 1
    ):
        return None
    left, right = operation.operand_types
    result = operation.result_types[0]
    if min(len(left.shape), len(right.shape), len(result.shape)) < 2:
        return None
    if any(
        prod(tensor.shape[:-2]) != 1
        for tensor in (left, right, result)
    ):
        return None
    m, k = left.shape[-2:]
    right_k, n = right.shape[-2:]
    result_m, result_n = result.shape[-2:]
    if k != right_k or (result_m, result_n) != (m, n):
        return None
    return m, k, n


def _profile_is_eligible(
    profile: HardwareCapabilityProfileDocument | None,
) -> bool:
    if profile is None:
        return False
    spec = profile.spec
    validity = spec.validity_domain
    quality = spec.quality
    return bool(
        spec.environment.get("eligible") is True
        and quality is not None
        and quality.status in {"qualified", "exploratory"}
        and validity is not None
        and "MatMul" in validity.operation_classes
        and validity.dtype == "float32"
        and validity.layout == "row-major-contiguous"
        and validity.logical_device == "npu:0"
        and validity.execution_mode == "pytorch-eager"
        and validity.shape_support == "observed-stratified-shapes-only"
    )


def _observed_compute_shapes(
    profile: HardwareCapabilityProfileDocument | None,
) -> frozenset[tuple[int, ...]]:
    if profile is None:
        return frozenset()
    return frozenset(
        shape.shape
        for resource in profile.spec.resources
        if resource.resource == "compute.fp32"
        for probe in resource.probe_envelopes
        for shape in probe.shape_best_rates
    )


def _operation_profile_rejection(
    operation: CostOperation,
    observed_compute_shapes: frozenset[tuple[int, ...]],
) -> str | None:
    tensors = operation.operand_types + operation.result_types
    if (
        any(tensor.dtype != "float32" for tensor in tensors)
        or any(tensor.layout != "contiguous" for tensor in tensors)
    ):
        return "outside-capability-validity-domain"
    problem_shape = matmul_problem_shape(operation)
    if problem_shape is None:
        return "outside-capability-validity-domain"
    if problem_shape not in observed_compute_shapes:
        return "outside-capability-observed-shapes"
    return None


def compile_ascend_910b2_prediction(
    bundle: AnalysisBundle, cost: CostProgram
) -> HardwareBackendPrediction | None:
    resolved = _resolve_device(bundle)
    if resolved is None:
        return None
    placement, hardware, device = resolved
    capabilities = device.capabilities
    if not (
        device.kind == "npu"
        and device.vendor.casefold() == "huawei"
        and device.model.casefold() == "ascend 910b2"
        and isinstance(capabilities, NpuHardwareCapabilities)
    ):
        return None

    profile = _measured_capabilities(bundle, hardware.metadata.name, device.id)
    measured_snapshots = _profile_snapshots(profile) if profile is not None else ()
    measured_by_resource = {
        item.resource: item for item in measured_snapshots
    }
    measured_compute = measured_by_resource.get("compute.fp32")
    measured_memory = measured_by_resource.get("memory.hbm")
    profile_eligible = _profile_is_eligible(profile)
    observed_compute_shapes = _observed_compute_shapes(profile)
    empirical_available = bool(
        profile_eligible
        and measured_compute is not None
        and measured_memory is not None
    )
    quality_status = (
        measured_compute.quality_status if measured_compute is not None else "unknown"
    )
    status = (
        f"partial-empirical-hardware-floor-{quality_status}"
        if empirical_available
        else "partial-unknown-resource-capability"
    )
    assumptions = (
        "FLOPs are CostIR minimum mathematical work, not implementation-added work.",
        "Compulsory bytes count each MatMul input and output once.",
        "P80 rates are cross-Shape robust achieved resource capacities.",
        "The HBM copy rate counts one read and one write as effective device-memory traffic.",
        "max(compute_time, memory_time) assumes compute and memory may overlap perfectly.",
        "Dispatch, tiling, synchronization, contention, and framework overhead are excluded.",
        "The complete uncalibrated implementation duration remains unknown.",
    )
    all_operations = tuple(cost.walk_operations())
    candidates: list[ImplementationCandidate] = []
    unsupported: list[UnsupportedCostRegion] = []
    for operation in all_operations:
        if operation.operation != "MatMul":
            unsupported.append(
                UnsupportedCostRegion(
                    cost_node_id=operation.node_id,
                    stable_path=operation.stable_path,
                    operation=operation.operation,
                    reason="unsupported-ascend-cost-operation",
                )
            )
            continue
        if not empirical_available:
            unsupported.append(
                UnsupportedCostRegion(
                    cost_node_id=operation.node_id,
                    stable_path=operation.stable_path,
                    operation=operation.operation,
                    reason="ineligible-hardware-capability-profile",
                )
            )
            continue
        rejection = _operation_profile_rejection(
            operation, observed_compute_shapes
        )
        if rejection is not None:
            unsupported.append(
                UnsupportedCostRegion(
                    cost_node_id=operation.node_id,
                    stable_path=operation.stable_path,
                    operation=operation.operation,
                    reason=rejection,
                )
            )
            continue
        compulsory_bytes = (
            operation.metrics.logical_read_bytes
            + operation.metrics.logical_write_bytes
        )
        materialized_bytes = (
            operation.metrics.materialized_read_bytes
            + operation.metrics.materialized_write_bytes
        )
        compute_time = (
            operation.metrics.flops
            * 1_000_000_000
            / measured_compute.robust_achievable_rate
            if measured_compute is not None
            else None
        )
        memory_time = (
            compulsory_bytes
            * 1_000_000_000
            / measured_memory.robust_achievable_rate
            if measured_memory is not None
            else None
        )
        floor = (
            max(compute_time, memory_time)
            if compute_time is not None and memory_time is not None
            else None
        )
        limiting_resource = (
            "compute.fp32"
            if floor is not None and compute_time >= memory_time
            else "memory.hbm"
            if floor is not None
            else None
        )
        candidates.append(
            ImplementationCandidate(
                candidate_id=(
                    "hardware-candidate:"
                    + content_fingerprint(
                        BACKEND_ID,
                        BACKEND_VERSION,
                        cost.compilation_fingerprint,
                        operation.node_id,
                        measured_snapshots,
                    )
                ),
                cost_node_id=operation.node_id,
                stable_path=operation.stable_path,
                operation=operation.operation,
                implementation="ascend-npu-matmul-resource-physical-floor",
                flops=operation.metrics.flops,
                compulsory_bytes=compulsory_bytes,
                materialized_bytes=materialized_bytes,
                duration=CandidateDurationEstimate(
                    model="algorithm-independent-resource-physical-floor",
                    status=(
                        f"empirical-hardware-floor-{quality_status}"
                        if floor is not None
                        else "unknown"
                    ),
                    compute_time_ns=compute_time,
                    memory_optimistic_lower_bound_ns=None,
                    empirical_compute_time_ns=compute_time,
                    empirical_memory_time_ns=memory_time,
                    empirical_hardware_floor_ns=floor,
                    limiting_resource=limiting_resource,
                    full_duration_ns=None,
                    formula=(
                        "max(minimum_flops / measured_fp32_P80, "
                        "compulsory_bytes / measured_hbm_copy_P80)"
                    ),
                    missing_capabilities=(
                        ()
                        if floor is not None
                        else ("measured_compute.fp32", "measured_memory.hbm")
                    ),
                    assumptions=assumptions,
                ),
            )
        )

    scope_bounds: list[ScopeDurationBounds] = []
    supported_cost_node_ids = {
        candidate.cost_node_id for candidate in candidates
    }
    for document in bundle.benchmark_cases:
        for case in document.spec.cases:
            selected = tuple(
                operation
                for operation in all_operations
                if _scope_matches(operation, case.scope)
            )
            if not selected:
                continue
            scope_flops = sum(operation.metrics.flops for operation in selected)
            scope_bytes = _scope_compulsory_bytes(all_operations, selected)
            if any(
                operation.node_id not in supported_cost_node_ids
                for operation in selected
            ):
                scope_bounds.append(
                    ScopeDurationBounds(
                        case_id=case.id,
                        scope=case.scope,
                        operation_count=len(selected),
                        flops=scope_flops,
                        compulsory_bytes=scope_bytes,
                        empirical_compute_time_ns=None,
                        empirical_memory_time_ns=None,
                        empirical_hardware_floor_ns=None,
                        limiting_resource=None,
                        formula=(
                            "unknown: selected CostIR scope is outside the "
                            "eligible Hardware Capability Profile domain"
                        ),
                        assumptions=assumptions,
                    )
                )
                continue
            scope_compute = (
                scope_flops * 1_000_000_000 / measured_compute.robust_achievable_rate
                if measured_compute is not None
                else None
            )
            scope_memory = (
                scope_bytes * 1_000_000_000 / measured_memory.robust_achievable_rate
                if measured_memory is not None
                else None
            )
            scope_floor = (
                max(scope_compute, scope_memory)
                if scope_compute is not None and scope_memory is not None
                else None
            )
            scope_bounds.append(
                ScopeDurationBounds(
                    case_id=case.id,
                    scope=case.scope,
                    operation_count=len(selected),
                    flops=scope_flops,
                    compulsory_bytes=scope_bytes,
                    empirical_compute_time_ns=scope_compute,
                    empirical_memory_time_ns=scope_memory,
                    empirical_hardware_floor_ns=scope_floor,
                    limiting_resource=(
                        "compute.fp32"
                        if scope_floor is not None and scope_compute >= scope_memory
                        else "memory.hbm"
                        if scope_floor is not None
                        else None
                    ),
                    formula=(
                        "max(sum(minimum_flops) / measured_fp32_P80, "
                        "unique_scope_boundary_bytes / measured_hbm_copy_P80)"
                    ),
                    assumptions=assumptions,
                )
            )

    theoretical_compute = capabilities.theoretical_compute.fp32_flops_per_second
    theoretical_memory = capabilities.dedicated_memory.peak_bandwidth_bytes_per_second
    capability_snapshot = NpuCapabilitySnapshot(
        architecture=capabilities.architecture,
        supported_operations=tuple(capabilities.supported_operations),
        supported_dtypes=tuple(capabilities.supported_dtypes),
        fp32_flops_per_second=_rate(theoretical_compute),
        peak_memory_bandwidth_bytes_per_second=_rate(theoretical_memory),
        memory_bytes=capabilities.dedicated_memory.capacity_bytes,
        memory_scope=capabilities.dedicated_memory.scope,
        evidence=_evidence(theoretical_compute.evidence)
        + _evidence(theoretical_memory.evidence)
        + _evidence(capabilities.dedicated_memory.capacity_evidence),
    )
    compute_availability = DurationAvailability(
        status="unknown",
        value_ns=None,
        reason=theoretical_compute.reason,
        required_capability="fp32_flops_per_second",
        evidence=_evidence(theoretical_compute.evidence),
    )
    total_metrics = cost.summary.metrics
    total_materialized_bytes = (
        total_metrics.materialized_read_bytes
        + total_metrics.materialized_write_bytes
    )
    total_compulsory_bytes = (
        cost.summary.parameter_bytes
        + cost.summary.buffer_bytes
        + cost.summary.workload_artifact_bytes
    )
    fingerprint = content_fingerprint(
        BACKEND_ID,
        BACKEND_VERSION,
        cost.compilation_fingerprint,
        placement,
        capability_snapshot,
        measured_snapshots,
        candidates,
        unsupported,
    )
    return HardwareBackendPrediction(
        schema=PREDICTION_SCHEMA,
        backend_id=BACKEND_ID,
        backend_version=BACKEND_VERSION,
        compilation_fingerprint=fingerprint,
        placement=placement,
        hardware=hardware.metadata.name,
        device=device.id,
        status=status,
        prediction_complete=False,
        capabilities=capability_snapshot,
        measured_capabilities=measured_snapshots,
        candidates=tuple(candidates),
        scope_bounds=tuple(scope_bounds),
        program_bounds=ProgramDurationBounds(
            flops=total_metrics.flops,
            compulsory_bytes=total_compulsory_bytes,
            materialized_bytes=total_materialized_bytes,
            compute_time=compute_availability,
            memory_optimistic_lower_bound_ns=None,
            vendor_memory_time_floor_ns=None,
            empirical_compute_time_ns=None,
            empirical_memory_time_ns=None,
            empirical_hardware_floor_ns=None,
            limiting_resource=None,
            full_duration_ns=None,
            formula="unknown: complete program contains unsupported CostIR operations",
            assumptions=assumptions,
        ),
        unsupported_regions=tuple(unsupported),
    )


__all__ = ["compile_ascend_910b2_prediction"]
