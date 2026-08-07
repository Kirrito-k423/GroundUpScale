"""Apple M4 CPU backend using public limits without invented peak compute."""

from __future__ import annotations

from math import prod

from groundupscale.ir import content_fingerprint
from groundupscale.ir.cost import CostOperation, CostProgram
from groundupscale.ir.hardware import (
    CandidateDurationEstimate,
    CapabilityEvidenceRef,
    CpuCapabilitySnapshot,
    DurationAvailability,
    HardwareBackendPrediction,
    HardwareCapabilityEnvelopeSnapshot,
    ImplementationCandidate,
    ProgramDurationBounds,
    RateAvailability,
    ScopeDurationBounds,
)
from groundupscale.schemas.v1alpha1 import CapabilityEvidence, TheoreticalRate
from groundupscale.specs import AnalysisBundle


BACKEND_ID = "apple.m4.cpu.resource-envelope"
BACKEND_VERSION = "v1alpha1"
PREDICTION_SCHEMA = "groundupscale.dev/hardware-backend-prediction/v1alpha1"


def _evidence(items: tuple[CapabilityEvidence, ...]) -> tuple[CapabilityEvidenceRef, ...]:
    return tuple(
        CapabilityEvidenceRef(
            source_kind=item.source_kind,
            title=item.title,
            url=item.url,
            accessed_on=item.accessed_on,
        )
        for item in items
    )


def _rate(rate: TheoreticalRate) -> RateAvailability:
    return RateAvailability(
        status=rate.status,
        value=float(rate.value) if rate.value is not None else None,
        reason=rate.reason,
        evidence=_evidence(rate.evidence),
    )


def _memory_time_ns(materialized_bytes: int, bandwidth: float) -> float:
    return materialized_bytes * 1_000_000_000 / bandwidth


_DTYPE_BYTES = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "int64": 8,
    "bool": 1,
}


def _tensor_bytes(dtype: str, shape: tuple[int, ...]) -> int:
    return _DTYPE_BYTES[dtype] * prod(shape)


def _scope_matches(operation: CostOperation, scope: str) -> bool:
    normalized = operation.stable_path.removeprefix("cost/")
    if scope.startswith("workload/"):
        return normalized == f"semantic/{scope}" or normalized.startswith(
            f"semantic/{scope}/"
        )
    if scope.startswith("model/"):
        parts = scope.split("/", 2)
        if len(parts) != 3:
            return False
        marker = f"/model/{parts[2]}"
        return normalized.endswith(marker) or marker + "/" in normalized
    return False


def _scope_compulsory_bytes(
    all_operations: tuple[CostOperation, ...],
    selected: tuple[CostOperation, ...],
) -> int:
    """Count unique external inputs/state and externally visible outputs."""

    producer: dict[str, str] = {}
    consumers: dict[str, set[str]] = {}
    tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
    alias_parent: dict[str, str] = {}
    for operation in all_operations:
        for value_id, tensor in zip(operation.operands, operation.operand_types):
            tensors.setdefault(value_id, (tensor.dtype, tensor.shape))
            consumers.setdefault(value_id, set()).add(operation.node_id)
        for value_id, tensor in zip(operation.results, operation.result_types):
            tensors.setdefault(value_id, (tensor.dtype, tensor.shape))
            producer[value_id] = operation.node_id
        if (
            operation.operation in {"View", "Transpose"}
            and len(operation.operands) == 1
            and len(operation.results) == 1
        ):
            alias_parent[operation.results[0]] = operation.operands[0]

    def root(value_id: str) -> str:
        visited: set[str] = set()
        current = value_id
        while current in alias_parent:
            if current in visited:
                raise ValueError(f"alias cycle in CostIR value {value_id}")
            visited.add(current)
            current = alias_parent[current]
        return current

    selected_ids = {operation.node_id for operation in selected}
    boundary_roots: set[str] = set()
    for operation in selected:
        for value_id in operation.operands:
            if producer.get(value_id) not in selected_ids:
                boundary_roots.add(root(value_id))
        for value_id in operation.results:
            value_consumers = consumers.get(value_id, set())
            if not value_consumers or not value_consumers <= selected_ids:
                boundary_roots.add(root(value_id))
    return sum(
        _tensor_bytes(tensors[value_id][0], tensors[value_id][1])
        for value_id in boundary_roots
    )


def _measured_capabilities(bundle: AnalysisBundle, hardware_name: str, device_id: str):
    matches = [
        profile
        for profile in bundle.hardware_capability_profiles
        if profile.spec.target.hardware == hardware_name
        and profile.spec.target.device == device_id
    ]
    if len(matches) > 1:
        raise ValueError(
            f"multiple HardwareCapabilityProfiles target {hardware_name}/{device_id}"
        )
    return matches[0] if matches else None


def _resolve_device(bundle: AnalysisBundle):
    placements = {
        binding.placement for binding in bundle.deployment_intent.spec.bindings
    }
    if len(placements) != 1:
        return None
    placement = next(iter(placements))
    fabric_node = next(
        (node for node in bundle.fabric_graph.spec.nodes if node.id == placement), None
    )
    if fabric_node is None:
        return None
    hardware = next(
        (
            document
            for document in bundle.hardware
            if document.metadata.name == fabric_node.hardware
        ),
        None,
    )
    if hardware is None:
        return None
    device = next(
        (item for item in hardware.spec.devices if item.id == fabric_node.device), None
    )
    if device is None:
        return None
    return placement, hardware, device


def compile_apple_m4_cpu_prediction(
    bundle: AnalysisBundle, cost: CostProgram
) -> HardwareBackendPrediction | None:
    """Select the M4 CPU backend and preserve vendor plus measured capability layers."""

    resolved = _resolve_device(bundle)
    if resolved is None:
        return None
    placement, hardware, device = resolved
    capabilities = device.capabilities
    if not (
        device.kind == "cpu"
        and device.vendor.casefold() == "apple"
        and device.model.casefold() == "m4"
        and capabilities is not None
    ):
        return None

    memory = capabilities.unified_memory
    compute_rate = capabilities.theoretical_compute.fp32_flops_per_second
    compute_availability = _rate(compute_rate)
    compute_time = DurationAvailability(
        status="available" if compute_rate.value is not None else "unknown",
        value_ns=(
            cost.summary.metrics.flops * 1_000_000_000 / compute_rate.value
            if compute_rate.value is not None
            else None
        ),
        reason=compute_rate.reason,
        required_capability="fp32_flops_per_second",
        evidence=_evidence(compute_rate.evidence),
    )
    profile = _measured_capabilities(
        bundle, hardware.metadata.name, device.id
    )
    measured_snapshots: tuple[HardwareCapabilityEnvelopeSnapshot, ...] = ()
    measured_compute_rate: float | None = None
    measured_memory_rate: float | None = None
    if profile is not None:
        environment_eligible = profile.spec.environment.get("eligible") is True
        measured_snapshots = tuple(
            HardwareCapabilityEnvelopeSnapshot(
                resource=resource.resource,
                unit=resource.unit,
                robust_achievable_rate=float(resource.robust_achievable_rate),
                optimistic_rate=float(resource.optimistic_rate),
                selected_robust_probe=resource.selected_robust_probe,
                selected_optimistic_probe=resource.selected_optimistic_probe,
                profile_name=profile.metadata.name,
                profile_version=profile.metadata.version,
                hardware_cohort=profile.spec.hardware_cohort,
                source_path=profile.spec.source.path,
                source_sha256=profile.spec.source.sha256,
                environment_eligible=environment_eligible,
            )
            for resource in profile.spec.resources
        )
        measured_by_resource = {
            item.resource: item for item in measured_snapshots
        }
        if "compute.fp32" in measured_by_resource:
            measured_compute_rate = measured_by_resource[
                "compute.fp32"
            ].robust_achievable_rate
        if "memory.shared" in measured_by_resource:
            measured_memory_rate = measured_by_resource[
                "memory.shared"
            ].robust_achievable_rate

    empirical_available = (
        measured_compute_rate is not None and measured_memory_rate is not None
    )
    status = (
        "empirical-hardware-lower-bound"
        if empirical_available
        else "partial-theoretical-lower-bound"
    )
    missing_compute = (
        ("fp32_flops_per_second",) if compute_rate.value is None else ()
    )
    assumptions = (
        "FLOPs are the CostIR minimum mathematical work, not implementation-added work.",
        "Compulsory bytes count unique scope-boundary inputs, state, and outputs.",
        "P80 rates are robust achieved resource capacities from diverse probe Shapes.",
        "max(compute_time, memory_time) assumes compute and memory may overlap perfectly.",
        "Scheduling, launch overhead, contention, conversions, and poor algorithms are excluded.",
    )
    all_operations = tuple(cost.walk_operations())
    candidates: list[ImplementationCandidate] = []
    for operation in all_operations:
        compulsory_bytes = (
            operation.metrics.logical_read_bytes
            + operation.metrics.logical_write_bytes
        )
        materialized_bytes = (
            operation.metrics.materialized_read_bytes
            + operation.metrics.materialized_write_bytes
        )
        vendor_memory_floor = _memory_time_ns(
            compulsory_bytes, memory.peak_bandwidth_bytes_per_second
        )
        empirical_compute = (
            operation.metrics.flops * 1_000_000_000 / measured_compute_rate
            if measured_compute_rate is not None
            else None
        )
        empirical_memory = (
            compulsory_bytes * 1_000_000_000 / measured_memory_rate
            if measured_memory_rate is not None
            else None
        )
        empirical_floor = (
            max(empirical_compute, empirical_memory)
            if empirical_compute is not None and empirical_memory is not None
            else None
        )
        limiting_resource = (
            "compute.fp32"
            if empirical_floor is not None and empirical_compute >= empirical_memory
            else "memory.shared"
            if empirical_floor is not None
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
                implementation="apple-cpu-algorithm-independent-hardware-floor",
                flops=operation.metrics.flops,
                compulsory_bytes=compulsory_bytes,
                materialized_bytes=materialized_bytes,
                duration=CandidateDurationEstimate(
                    model="algorithm-independent-empirical-roofline-floor",
                    status=status,
                    compute_time_ns=empirical_compute,
                    memory_optimistic_lower_bound_ns=vendor_memory_floor,
                    empirical_compute_time_ns=empirical_compute,
                    empirical_memory_time_ns=empirical_memory,
                    empirical_hardware_floor_ns=empirical_floor,
                    limiting_resource=limiting_resource,
                    full_duration_ns=None,
                    formula=(
                        "max(minimum_flops / measured_fp32_P80, "
                        "compulsory_bytes / measured_memory_P80)"
                    ),
                    missing_capabilities=(
                        ()
                        if empirical_available
                        else ("measured_compute.fp32", "measured_memory.shared")
                    ),
                    assumptions=assumptions,
                ),
            )
        )

    scope_bounds: list[ScopeDurationBounds] = []
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
            scope_compute = (
                scope_flops * 1_000_000_000 / measured_compute_rate
                if measured_compute_rate is not None
                else None
            )
            scope_memory = (
                scope_bytes * 1_000_000_000 / measured_memory_rate
                if measured_memory_rate is not None
                else None
            )
            scope_floor = (
                max(scope_compute, scope_memory)
                if scope_compute is not None and scope_memory is not None
                else None
            )
            scope_limiting = (
                "compute.fp32"
                if scope_floor is not None and scope_compute >= scope_memory
                else "memory.shared"
                if scope_floor is not None
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
                    limiting_resource=scope_limiting,
                    formula=(
                        "max(sum(minimum_flops) / measured_fp32_P80, "
                        "unique_scope_boundary_bytes / measured_memory_P80)"
                    ),
                    assumptions=assumptions,
                )
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
    empirical_program_compute = (
        total_metrics.flops * 1_000_000_000 / measured_compute_rate
        if measured_compute_rate is not None
        else None
    )
    empirical_program_memory = (
        total_compulsory_bytes * 1_000_000_000 / measured_memory_rate
        if measured_memory_rate is not None
        else None
    )
    empirical_program_floor = (
        max(empirical_program_compute, empirical_program_memory)
        if empirical_program_compute is not None
        and empirical_program_memory is not None
        else None
    )
    program_limiting_resource = (
        "compute.fp32"
        if empirical_program_floor is not None
        and empirical_program_compute >= empirical_program_memory
        else "memory.shared"
        if empirical_program_floor is not None
        else None
    )
    capability_evidence = (
        _evidence(capabilities.core_topology_evidence)
        + _evidence(capabilities.vector.evidence)
        + _evidence(memory.evidence)
    )
    capability_snapshot = CpuCapabilitySnapshot(
        architecture=capabilities.architecture,
        core_pools=tuple(
            (pool.kind, pool.count) for pool in capabilities.core_pools
        ),
        vector_isa=capabilities.vector.isa,
        vector_register_bits=capabilities.vector.register_bits,
        fp32_fma_flops_per_instruction=(
            capabilities.vector.fp32_fma_flops_per_instruction
        ),
        fp64_fma_flops_per_instruction=(
            capabilities.vector.fp64_fma_flops_per_instruction
        ),
        fp32_flops_per_second=compute_availability,
        peak_memory_bandwidth_bytes_per_second=float(
            memory.peak_bandwidth_bytes_per_second
        ),
        memory_bandwidth_scope=memory.scope,
        evidence=capability_evidence,
    )
    fingerprint = content_fingerprint(
        BACKEND_ID,
        BACKEND_VERSION,
        cost.compilation_fingerprint,
        placement,
        capability_snapshot,
        measured_snapshots,
        candidates,
        scope_bounds,
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
            compute_time=compute_time,
            memory_optimistic_lower_bound_ns=_memory_time_ns(
                total_compulsory_bytes, memory.peak_bandwidth_bytes_per_second
            ),
            vendor_memory_time_floor_ns=_memory_time_ns(
                total_compulsory_bytes, memory.peak_bandwidth_bytes_per_second
            ),
            empirical_compute_time_ns=empirical_program_compute,
            empirical_memory_time_ns=empirical_program_memory,
            empirical_hardware_floor_ns=empirical_program_floor,
            limiting_resource=program_limiting_resource,
            full_duration_ns=None,
            formula=(
                "empirical_hardware_floor_ns = max(minimum_flops / "
                "measured_fp32_P80, compulsory_bytes / measured_memory_P80) * 1e9"
            ),
            assumptions=assumptions,
        ),
    )


__all__ = ["compile_apple_m4_cpu_prediction"]
