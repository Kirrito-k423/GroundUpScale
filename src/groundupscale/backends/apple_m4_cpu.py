"""Apple M4 CPU backend using public limits without invented peak compute."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from math import prod
from collections import Counter

from groundupscale.ir import content_fingerprint
from groundupscale.ir.cost import CostOperation, CostProgram
from groundupscale.ir.hardware import (
    CandidateDurationEstimate,
    CandidatePhaseDuration,
    CandidatePhaseSchedule,
    CapabilityEvidenceRef,
    CpuCapabilitySnapshot,
    DurationAvailability,
    HardwareBackendPrediction,
    HardwareCapabilityEnvelopeSnapshot,
    ImplementationCandidate,
    ProgramDurationBounds,
    PhaseResourceComposition,
    PhaseSchedulePolicy,
    PhaseScheduleStatus,
    RateAvailability,
    ScopeDurationBounds,
)
from groundupscale.schemas.v1alpha1 import (
    CapabilityEvidence,
    ExactOperatorExecutionContract,
    ExactShapeOperatorFrontierAnchor,
    TheoreticalRate,
)
from groundupscale.scheduling import (
    BoundEvent,
    ScheduleBoundComposition,
    compose_schedule_bound,
)
from groundupscale.specs import AnalysisBundle


BACKEND_ID = "apple.m4.cpu.resource-envelope"
BACKEND_VERSION = "v1alpha2"
PREDICTION_SCHEMA = "groundupscale.dev/hardware-backend-prediction/v1alpha2"


@dataclass(frozen=True)
class ExactOperatorFrontierMatch:
    status: str
    reason_codes: tuple[str, ...]
    latency_ns: float | None = None
    standard_uncertainty_ns: float | None = None
    anchor_id: str | None = None
    candidate_family: str | None = None
    candidate_digest: str | None = None
    input_corpus_digest: str | None = None
    execution_contract_digest: str | None = None
    profile: str | None = None
    profile_version: str | None = None
    source_path: str | None = None
    source_sha256: str | None = None
    hardware_cohort: str | None = None


def _canonical_layout(layout: str) -> str:
    if layout == "contiguous":
        return "row-major-contiguous"
    if layout in {"transposed", "transposed-strided"}:
        return "strided"
    return layout


def _anchor_matches_execution_contract(
    anchor: ExactShapeOperatorFrontierAnchor,
    contract: ExactOperatorExecutionContract | None,
) -> bool:
    if contract is None:
        return False
    operands = contract.operand_contracts
    result = contract.result_contract
    operand_layouts = tuple(item.layout for item in operands)
    anchor_operand_layouts = (
        anchor.operand_layouts
        if anchor.operand_layouts is not None
        else (anchor.layout,) * len(anchor.operand_shapes)
    )
    return bool(
        anchor.execution_contract_digest == contract.execution_contract_digest
        and anchor.execution_mode == contract.execution_mode
        and anchor.operand_shapes == tuple(item.shape for item in operands)
        and anchor.result_shape == result.shape
        and anchor.operand_strides == tuple(item.stride for item in operands)
        and anchor.result_stride == result.stride
        and anchor_operand_layouts == operand_layouts
        and (
            anchor.result_layout
            if anchor.result_layout is not None
            else anchor.layout
        )
        == result.layout
        and {anchor.dtype}
        == {item.dtype for item in (*operands, result)}
        and anchor.minimum_alignment_bytes
        == min(item.minimum_alignment_bytes for item in (*operands, result))
        and anchor.working_set_bytes == contract.working_set_bytes
    )


def _exact_operator_frontier(
    bundle: AnalysisBundle,
    operation: CostOperation,
    *,
    hardware_name: str,
    device_id: str,
) -> ExactOperatorFrontierMatch:
    """Resolve one qualified ACTIVE exact-Shape Anchor without interpolation."""

    profiles = tuple(
        profile
        for profile in bundle.operator_frontier_profiles
        if profile.spec.target.hardware == hardware_name
        and profile.spec.target.device == device_id
    )
    if not profiles:
        return ExactOperatorFrontierMatch(
            "not-configured", ("operator-frontier-profile-not-configured",)
        )
    execution_domain = bundle.plan.spec.operator_frontier_execution_domain
    if execution_domain is None:
        return ExactOperatorFrontierMatch(
            "unknown", ("operator-frontier-execution-domain-not-declared",)
        )
    if not any(
        profile.spec.hardware_cohort == execution_domain.hardware_cohort
        for profile in profiles
    ):
        return ExactOperatorFrontierMatch(
            "unknown", ("operator-frontier-hardware-cohort-mismatch",)
        )
    profiles = tuple(
        profile
        for profile in profiles
        if profile.spec.hardware_cohort == execution_domain.hardware_cohort
    )

    normalized_path = operation.stable_path.removeprefix("cost/")
    operand_shapes = tuple(tensor.shape for tensor in operation.operand_types)
    result_shapes = tuple(tensor.shape for tensor in operation.result_types)
    dtypes = {
        tensor.dtype for tensor in operation.operand_types + operation.result_types
    }
    operand_layouts = tuple(
        _canonical_layout(tensor.layout) for tensor in operation.operand_types
    )
    result_layouts = tuple(
        _canonical_layout(tensor.layout) for tensor in operation.result_types
    )
    expected_family = execution_domain.candidate_families.get(normalized_path)
    expected_candidate_digest = execution_domain.candidate_digests.get(
        normalized_path
    )
    expected_execution_digest = execution_domain.execution_contract_digests.get(
        normalized_path
    )
    expected_execution_contract = execution_domain.execution_contracts.get(
        normalized_path
    )
    expected_input_digest = execution_domain.input_corpus_digests.get(
        normalized_path
    )
    matches: list[tuple[object, object]] = []
    for profile in profiles:
        for anchor in profile.spec.anchors:
            if (
                fnmatchcase(normalized_path, anchor.stable_path_pattern)
                and anchor.semantic_operation == operation.operation
                and anchor.operand_shapes == operand_shapes
                and len(result_shapes) == 1
                and anchor.result_shape == result_shapes[0]
                and dtypes == {anchor.dtype}
                and (
                    anchor.operand_layouts
                    if anchor.operand_layouts is not None
                    else (anchor.layout,) * len(operand_layouts)
                )
                == operand_layouts
                and (
                    anchor.result_layout
                    if anchor.result_layout is not None
                    else anchor.layout
                )
                == result_layouts[0]
                and anchor.threads == execution_domain.threads
                and anchor.interop_threads == execution_domain.interop_threads
                and anchor.execution_mode == execution_domain.execution_mode
                and anchor.candidate_family == expected_family
                and anchor.candidate_digest == expected_candidate_digest
                and anchor.execution_contract_digest == expected_execution_digest
                and _anchor_matches_execution_contract(
                    anchor, expected_execution_contract
                )
                and anchor.input_corpus_digest == expected_input_digest
                and anchor.timing_scope == execution_domain.timing_scope
                and anchor.completion_boundary == execution_domain.completion_boundary
                and anchor.instrumentation_profile
                == execution_domain.instrumentation_profile
                and anchor.measurement_hardware_cohort
                == execution_domain.hardware_cohort
            ):
                matches.append((profile, anchor))
    if not matches:
        failed_attempt = execution_domain.qualification_failures.get(normalized_path)
        return ExactOperatorFrontierMatch(
            "unknown",
            (
                "exact-shape-operator-frontier-not-found",
                *(failed_attempt.reason_codes if failed_attempt is not None else ()),
            ),
        )
    if len(matches) != 1:
        return ExactOperatorFrontierMatch(
            "unknown", ("ambiguous-exact-shape-operator-frontier",)
        )
    profile, anchor = matches[0]
    return ExactOperatorFrontierMatch(
        status="exact-anchor",
        latency_ns=float(anchor.latency_ns),
        standard_uncertainty_ns=float(anchor.standard_uncertainty_ns),
        anchor_id=anchor.anchor_id,
        candidate_family=anchor.candidate_family,
        candidate_digest=anchor.candidate_digest,
        input_corpus_digest=anchor.input_corpus_digest,
        execution_contract_digest=anchor.execution_contract_digest,
        profile=profile.metadata.name,
        profile_version=profile.metadata.version,
        source_path=profile.spec.source.path,
        source_sha256=profile.spec.source.sha256,
        hardware_cohort=profile.spec.hardware_cohort,
        reason_codes=(),
    )


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


def _compose_selected_bounds(
    selected: tuple[CostOperation, ...],
    candidates_by_cost_node: dict[str, ImplementationCandidate],
    *,
    resource_physical: bool = False,
    provisional: bool = False,
) -> ScheduleBoundComposition | None:
    """Compose unfused candidates using Cost IR dependencies."""

    selected_ids = {operation.node_id for operation in selected}
    events: list[BoundEvent] = []
    for operation in selected:
        candidate = candidates_by_cost_node[operation.node_id]
        duration = candidate.duration
        local_duration = (
            duration.provisional_estimate_ns
            if provisional
            else duration.resource_physical_floor_ns
            if resource_physical
            else duration.empirical_hardware_floor_ns
        )
        if local_duration is None:
            return None
        if provisional:
            resource_times = (("provisional.serialized", local_duration),)
        elif (
            duration.empirical_compute_time_ns is None
            or duration.empirical_memory_time_ns is None
        ):
            return None
        else:
            resource_times = (
                ("compute.fp32", duration.empirical_compute_time_ns),
                ("memory.shared", duration.empirical_memory_time_ns),
            )
        events.append(
            BoundEvent(
                event_id=operation.node_id,
                predecessor_ids=tuple(
                    predecessor_id
                    for predecessor_id in operation.dependency_cost_node_ids
                    if predecessor_id in selected_ids
                ),
                local_duration_ns=local_duration,
                resource_times_ns=resource_times,
            )
        )
    return compose_schedule_bound(tuple(events), schedule="serialized")


def _compose_candidate_phases(
    operation: CostOperation,
    *,
    rates_by_resource: dict[str, float],
    capability_errors: dict[str, str],
    evidence_refs_by_resource: dict[str, str],
) -> CandidatePhaseSchedule | None:
    graph = operation.phase_graph
    if graph is None:
        return None
    phases: list[CandidatePhaseDuration] = []
    events: list[BoundEvent] = []
    schedule_missing: set[str] = set()
    for phase in graph.phases:
        compute_rate = rates_by_resource.get(phase.compute_capability_resource)
        memory_rate = rates_by_resource.get(phase.memory_capability_resource)
        missing = tuple(
            capability_errors.get(resource, resource)
            for resource, rate in (
                (phase.compute_capability_resource, compute_rate),
                (phase.memory_capability_resource, memory_rate),
            )
            if rate is None
        )
        schedule_missing.update(missing)
        compute_time = (
            phase.minimum_flops * 1_000_000_000 / compute_rate
            if compute_rate is not None
            else None
        )
        logical_bytes = phase.logical_read_bytes + phase.logical_write_bytes
        memory_time = (
            logical_bytes * 1_000_000_000 / memory_rate
            if memory_rate is not None
            else None
        )
        local_floor = (
            max(compute_time, memory_time)
            if compute_time is not None and memory_time is not None
            else None
        )
        evidence_refs = tuple(
            evidence_refs_by_resource[resource]
            for resource in (
                phase.compute_capability_resource,
                phase.memory_capability_resource,
            )
            if resource in evidence_refs_by_resource
        )
        phases.append(
            CandidatePhaseDuration(
                phase_id=phase.phase_id,
                phase_name=phase.phase_name,
                operation_class=phase.operation_class,
                status=(
                    PhaseScheduleStatus.KNOWN
                    if not missing
                    else PhaseScheduleStatus.UNKNOWN
                ),
                predecessor_phase_ids=phase.predecessor_phase_ids,
                minimum_flops=phase.minimum_flops,
                logical_read_bytes=phase.logical_read_bytes,
                logical_write_bytes=phase.logical_write_bytes,
                required_compute_capability=phase.compute_capability_resource,
                required_memory_capability=phase.memory_capability_resource,
                compute_time_ns=compute_time,
                memory_time_ns=memory_time,
                resource_composition=PhaseResourceComposition.MAX,
                overlap_evidence_refs=(),
                capability_evidence_refs=(evidence_refs if not missing else ()),
                local_hardware_floor_ns=local_floor,
                limiting_resource=(
                    phase.compute_capability_resource
                    if local_floor is not None and compute_time >= memory_time
                    else phase.memory_capability_resource
                    if local_floor is not None
                    else None
                ),
                missing_capabilities=missing,
            )
        )
        if local_floor is not None:
            events.append(
                BoundEvent(
                    event_id=phase.phase_id,
                    predecessor_ids=phase.predecessor_phase_ids,
                    local_duration_ns=local_floor,
                    resource_times_ns=(
                        (phase.compute_capability_resource, compute_time),
                        (phase.memory_capability_resource, memory_time),
                    ),
                )
            )
    composition = (
        compose_schedule_bound(tuple(events), schedule="serialized")
        if not schedule_missing
        else None
    )
    return CandidatePhaseSchedule(
        status=(
            PhaseScheduleStatus.KNOWN
            if composition is not None
            else PhaseScheduleStatus.UNKNOWN
        ),
        policy=PhaseSchedulePolicy.SERIALIZED_NO_CHUNK,
        chunk_pipeline_contract_id=None,
        phases=tuple(phases),
        serialized_duration_ns=(
            composition.serialized_duration_ns if composition is not None else None
        ),
        critical_path_duration_ns=(
            composition.critical_path_duration_ns if composition is not None else None
        ),
        selected_duration_ns=(
            composition.selected_duration_ns if composition is not None else None
        ),
        missing_capabilities=tuple(sorted(schedule_missing)),
        formula="sum(phase.local_hardware_floor_ns)",
        assumptions=(
            "phase dependencies come from the Cost IR Operator Phase Graph",
            "each phase uses only its exact operation-class compute and memory capabilities",
            "each exact compute probe measures the complete phase invocation, including its data movement",
            "the memory-pattern probe is an independent lower-bound constraint, so the phase local floor is max(compute probe, memory floor) rather than their duplicate sum",
            "the two capability-profile references on a known phase make this same-invocation composition auditable",
            "no cross-phase or cross-chunk overlap is permitted without a Chunk Pipeline Contract",
            "missing phase capability evidence makes the compound duration unknown",
        ),
    )


def _validated_resource_rates(
    snapshots: tuple[HardwareCapabilityEnvelopeSnapshot, ...],
) -> tuple[dict[str, float], dict[str, str]]:
    counts = Counter(snapshot.resource for snapshot in snapshots)
    rates: dict[str, float] = {}
    errors: dict[str, str] = {}
    for snapshot in snapshots:
        resource = snapshot.resource
        if counts[resource] != 1:
            errors[resource] = f"{resource}:duplicate-envelope"
            continue
        expected_unit = (
            "FLOP/s"
            if resource.startswith("compute.")
            else "B/s"
            if resource.startswith("memory.")
            else None
        )
        if expected_unit is None:
            errors[resource] = f"{resource}:unsupported-resource-class"
            continue
        if snapshot.unit != expected_unit:
            errors[resource] = (
                f"{resource}:expected-{expected_unit}-got-{snapshot.unit}"
            )
            continue
        rates[resource] = snapshot.robust_achievable_rate
    return rates, errors


def _resource_evidence_refs(
    snapshots: tuple[HardwareCapabilityEnvelopeSnapshot, ...],
) -> dict[str, str]:
    counts = Counter(snapshot.resource for snapshot in snapshots)
    return {
        snapshot.resource: (
            f"capability-profile://{snapshot.profile_name}@"
            f"{snapshot.profile_version}/{snapshot.resource}"
            f"?source_sha256={snapshot.source_sha256}"
            f"&probe={snapshot.selected_robust_probe}"
        )
        for snapshot in snapshots
        if counts[snapshot.resource] == 1
    }


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
    provisional_reason_codes: tuple[str, ...] = ()
    if profile is not None:
        environment_eligible = profile.spec.environment.get("eligible") is True
        provisional_reason_codes = tuple(
            str(reason)
            for reason in profile.spec.environment.get("reason_codes", ())
        )
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
    physical_rates, physical_capability_errors = _validated_resource_rates(
        measured_snapshots
    )
    phase_rates, phase_capability_errors = _validated_resource_rates(
        tuple(
            snapshot
            for snapshot in measured_snapshots
            if snapshot.environment_eligible
        )
    )
    phase_evidence_refs = _resource_evidence_refs(
        tuple(
            snapshot
            for snapshot in measured_snapshots
            if snapshot.environment_eligible
        )
    )
    provisional_phase_rates, provisional_phase_errors = _validated_resource_rates(
        measured_snapshots
    )
    provisional_phase_evidence_refs = _resource_evidence_refs(measured_snapshots)
    measured_compute_rate = physical_rates.get("compute.fp32")
    measured_memory_rate = physical_rates.get("memory.shared")

    empirical_available = (
        measured_compute_rate is not None and measured_memory_rate is not None
    )
    status = (
        "empirical-hardware-lower-bound"
        if empirical_available
        else "partial-theoretical-lower-bound"
    )
    assumptions = (
        "FLOPs are the CostIR minimum mathematical work, not implementation-added work.",
        "Each unfused candidate counts only CostIR materialized reads and writes; aliases move zero physical bytes.",
        "P80 rates are robust achieved resource capacities from diverse probe Shapes.",
        "Generic max(compute_time, memory_time) is retained only as a Resource Physical Floor.",
        "A compound candidate is selected only from exact phase capabilities and explicit phase composition.",
        "Dependent phases are serial unless a Chunk Pipeline Contract says otherwise.",
        "Inside one phase, the exact operation probe already includes data movement; its duration and the independent memory-pattern floor compose by max with both profile refs retained.",
        "The ideal DAG bound is reported separately as max(dependency critical path, shared-resource load).",
        "Cross-operation fusion requires an explicit fused ImplementationCandidate and is not inferred from a scope boundary.",
        "Launch overhead, contention, conversions, and poor algorithms remain excluded.",
    )
    all_operations = tuple(cost.walk_operations())
    candidates: list[ImplementationCandidate] = []
    for operation in all_operations:
        materialized_bytes = (
            operation.metrics.materialized_read_bytes
            + operation.metrics.materialized_write_bytes
        )
        compulsory_bytes = materialized_bytes
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
        phase_schedule = _compose_candidate_phases(
            operation,
            rates_by_resource=phase_rates,
            capability_errors=phase_capability_errors,
            evidence_refs_by_resource=phase_evidence_refs,
        )
        provisional_phase_schedule = (
            _compose_candidate_phases(
                operation,
                rates_by_resource=provisional_phase_rates,
                capability_errors=provisional_phase_errors,
                evidence_refs_by_resource=provisional_phase_evidence_refs,
            )
            if phase_schedule is not None
            and phase_schedule.selected_duration_ns is None
            and profile is not None
            and profile.spec.environment.get("eligible") is not True
            else None
        )
        provisional_estimate = (
            provisional_phase_schedule.selected_duration_ns
            if provisional_phase_schedule is not None
            else empirical_floor
            if phase_schedule is None
            and profile is not None
            and profile.spec.environment.get("eligible") is not True
            else None
        )
        selected_floor = (
            phase_schedule.selected_duration_ns
            if phase_schedule is not None
            else empirical_floor
        )
        selected_status = (
            "phase-capabilities-incomplete"
            if phase_schedule is not None and selected_floor is None
            else "phase-serialized-resource-reference"
            if phase_schedule is not None
            else status
        )
        selected_limiter = (
            "serialized.phases" if phase_schedule is not None and selected_floor is not None
            else limiting_resource if phase_schedule is None
            else None
        )
        selected_missing = (
            phase_schedule.missing_capabilities
            if phase_schedule is not None
            else ()
            if empirical_available
            else tuple(
                physical_capability_errors.get(resource, f"measured_{resource}")
                for resource in ("compute.fp32", "memory.shared")
                if resource not in physical_rates
            )
        )
        operator_frontier = _exact_operator_frontier(
            bundle,
            operation,
            hardware_name=hardware.metadata.name,
            device_id=device.id,
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
                    status=selected_status,
                    compute_time_ns=empirical_compute,
                    memory_optimistic_lower_bound_ns=vendor_memory_floor,
                    empirical_compute_time_ns=empirical_compute,
                    empirical_memory_time_ns=empirical_memory,
                    resource_physical_floor_ns=empirical_floor,
                    empirical_hardware_floor_ns=selected_floor,
                    provisional_estimate_ns=provisional_estimate,
                    provisional_evidence_tier=(
                        "exploratory" if provisional_estimate is not None else None
                    ),
                    provisional_reason_codes=(
                        provisional_reason_codes
                        if provisional_estimate is not None
                        else ()
                    ),
                    limiting_resource=selected_limiter,
                    full_duration_ns=None,
                    formula=(
                        phase_schedule.formula
                        if phase_schedule is not None
                        else "max(minimum_flops / measured_fp32_P80, "
                        "materialized_bytes / measured_memory_P80)"
                    ),
                    missing_capabilities=selected_missing,
                    assumptions=assumptions,
                    operator_achievable_frontier_ns=(
                        operator_frontier.latency_ns
                        if operator_frontier.status == "exact-anchor"
                        else None
                    ),
                    operator_frontier_standard_uncertainty_ns=(
                        operator_frontier.standard_uncertainty_ns
                        if operator_frontier.status == "exact-anchor"
                        else None
                    ),
                    operator_frontier_match_status=operator_frontier.status,
                    operator_frontier_anchor_id=(
                        operator_frontier.anchor_id
                        if operator_frontier.status == "exact-anchor"
                        else None
                    ),
                    operator_frontier_candidate_family=(
                        operator_frontier.candidate_family
                        if operator_frontier.status == "exact-anchor"
                        else None
                    ),
                    operator_frontier_candidate_digest=(
                        operator_frontier.candidate_digest
                        if operator_frontier.status == "exact-anchor"
                        else None
                    ),
                    operator_frontier_input_corpus_digest=(
                        operator_frontier.input_corpus_digest
                        if operator_frontier.status == "exact-anchor"
                        else None
                    ),
                    operator_frontier_execution_contract_digest=(
                        operator_frontier.execution_contract_digest
                        if operator_frontier.status == "exact-anchor"
                        else None
                    ),
                    operator_frontier_profile=(
                        operator_frontier.profile
                        if operator_frontier.status == "exact-anchor"
                        else None
                    ),
                    operator_frontier_profile_version=(
                        operator_frontier.profile_version
                        if operator_frontier.status == "exact-anchor"
                        else None
                    ),
                    operator_frontier_source_path=(
                        operator_frontier.source_path
                        if operator_frontier.status == "exact-anchor"
                        else None
                    ),
                    operator_frontier_source_sha256=(
                        operator_frontier.source_sha256
                        if operator_frontier.status == "exact-anchor"
                        else None
                    ),
                    operator_frontier_hardware_cohort=(
                        operator_frontier.hardware_cohort
                        if operator_frontier.status == "exact-anchor"
                        else None
                    ),
                    operator_frontier_reason_codes=operator_frontier.reason_codes,
                ),
                phase_schedule=phase_schedule,
                provisional_phase_schedule=provisional_phase_schedule,
            )
        )

    candidates_by_cost_node = {
        candidate.cost_node_id: candidate for candidate in candidates
    }
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
            selected_candidates = tuple(
                candidates_by_cost_node[operation.node_id] for operation in selected
            )
            scope_compulsory_bytes = _scope_compulsory_bytes(
                all_operations, selected
            )
            scope_materialized_bytes = sum(
                candidate.materialized_bytes for candidate in selected_candidates
            )
            scope_compute = (
                sum(
                    candidate.duration.empirical_compute_time_ns
                    for candidate in selected_candidates
                    if candidate.duration.empirical_compute_time_ns is not None
                )
                if empirical_available
                else None
            )
            scope_memory = (
                sum(
                    candidate.duration.empirical_memory_time_ns
                    for candidate in selected_candidates
                    if candidate.duration.empirical_memory_time_ns is not None
                )
                if empirical_available
                else None
            )
            composition = _compose_selected_bounds(selected, candidates_by_cost_node)
            physical_composition = _compose_selected_bounds(
                selected,
                candidates_by_cost_node,
                resource_physical=True,
            )
            provisional_composition = _compose_selected_bounds(
                selected,
                candidates_by_cost_node,
                provisional=True,
            )
            exact_scope_candidate = (
                selected_candidates[0] if len(selected_candidates) == 1 else None
            )
            scope_bounds.append(
                ScopeDurationBounds(
                    case_id=case.id,
                    scope=case.scope,
                    operation_count=len(selected),
                    flops=scope_flops,
                    compulsory_bytes=scope_compulsory_bytes,
                    materialized_bytes=scope_materialized_bytes,
                    empirical_compute_time_ns=scope_compute,
                    empirical_memory_time_ns=scope_memory,
                    schedule="serialized-unfused",
                    serialized_hardware_floor_ns=(
                        composition.serialized_duration_ns
                        if composition is not None
                        else None
                    ),
                    critical_path_hardware_floor_ns=(
                        composition.critical_path_duration_ns
                        if composition is not None
                        else None
                    ),
                    resource_hardware_floor_ns=(
                        physical_composition.resource_duration_ns
                        if physical_composition is not None
                        else None
                    ),
                    resource_physical_floor_ns=(
                        physical_composition.selected_duration_ns
                        if physical_composition is not None
                        else None
                    ),
                    ideal_dag_hardware_floor_ns=(
                        physical_composition.ideal_dag_duration_ns
                        if physical_composition is not None
                        else None
                    ),
                    empirical_hardware_floor_ns=(
                        composition.selected_duration_ns
                        if composition is not None
                        else None
                    ),
                    provisional_estimate_ns=(
                        provisional_composition.selected_duration_ns
                        if provisional_composition is not None
                        else None
                    ),
                    provisional_evidence_tier=(
                        "exploratory"
                        if provisional_composition is not None
                        else None
                    ),
                    provisional_reason_codes=(
                        provisional_reason_codes
                        if provisional_composition is not None
                        else ()
                    ),
                    limiting_resource=(
                        "serialized.events" if composition is not None else None
                    ),
                    resource_limiting_resource=(
                        physical_composition.limiting_resource
                        if physical_composition is not None
                        else None
                    ),
                    formula=(
                        "sum(candidate_i.max(minimum_flops_i / measured_fp32_P80, "
                        "materialized_bytes_i / measured_memory_P80))"
                    ),
                    assumptions=assumptions,
                    operator_achievable_frontier_ns=(
                        exact_scope_candidate.duration.operator_achievable_frontier_ns
                        if exact_scope_candidate is not None
                        else None
                    ),
                    operator_frontier_standard_uncertainty_ns=(
                        exact_scope_candidate.duration.operator_frontier_standard_uncertainty_ns
                        if exact_scope_candidate is not None
                        else None
                    ),
                    operator_frontier_match_status=(
                        exact_scope_candidate.duration.operator_frontier_match_status
                        if exact_scope_candidate is not None
                        else "unknown"
                    ),
                    operator_frontier_anchor_ids=(
                        (exact_scope_candidate.duration.operator_frontier_anchor_id,)
                        if exact_scope_candidate is not None
                        and exact_scope_candidate.duration.operator_frontier_anchor_id
                        is not None
                        else ()
                    ),
                    operator_frontier_candidate_digest=(
                        exact_scope_candidate.duration.operator_frontier_candidate_digest
                        if exact_scope_candidate is not None
                        else None
                    ),
                    operator_frontier_input_corpus_digest=(
                        exact_scope_candidate.duration.operator_frontier_input_corpus_digest
                        if exact_scope_candidate is not None
                        else None
                    ),
                    operator_frontier_execution_contract_digest=(
                        exact_scope_candidate.duration.operator_frontier_execution_contract_digest
                        if exact_scope_candidate is not None
                        else None
                    ),
                    operator_frontier_hardware_cohort=(
                        exact_scope_candidate.duration.operator_frontier_hardware_cohort
                        if exact_scope_candidate is not None
                        else None
                    ),
                    operator_frontier_reason_codes=(
                        exact_scope_candidate.duration.operator_frontier_reason_codes
                        if exact_scope_candidate is not None
                        else ("multi-candidate-frontier-composition-not-defined",)
                    ),
                )
            )

    total_metrics = cost.summary.metrics
    total_materialized_bytes = (
        total_metrics.materialized_read_bytes
        + total_metrics.materialized_write_bytes
    )
    total_compulsory_bytes = _scope_compulsory_bytes(
        all_operations, all_operations
    )
    empirical_program_compute = (
        sum(
            candidate.duration.empirical_compute_time_ns
            for candidate in candidates
            if candidate.duration.empirical_compute_time_ns is not None
        )
        if empirical_available
        else None
    )
    empirical_program_memory = (
        sum(
            candidate.duration.empirical_memory_time_ns
            for candidate in candidates
            if candidate.duration.empirical_memory_time_ns is not None
        )
        if empirical_available
        else None
    )
    program_composition = _compose_selected_bounds(
        all_operations, candidates_by_cost_node
    )
    program_physical_composition = _compose_selected_bounds(
        all_operations,
        candidates_by_cost_node,
        resource_physical=True,
    )
    program_provisional_composition = _compose_selected_bounds(
        all_operations,
        candidates_by_cost_node,
        provisional=True,
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
    program_bounds = ProgramDurationBounds(
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
        schedule="serialized-unfused",
        serialized_hardware_floor_ns=(
            program_composition.serialized_duration_ns
            if program_composition is not None
            else None
        ),
        critical_path_hardware_floor_ns=(
            program_composition.critical_path_duration_ns
            if program_composition is not None
            else None
        ),
        resource_hardware_floor_ns=(
            program_physical_composition.resource_duration_ns
            if program_physical_composition is not None
            else None
        ),
        resource_physical_floor_ns=(
            program_physical_composition.selected_duration_ns
            if program_physical_composition is not None
            else None
        ),
        ideal_dag_hardware_floor_ns=(
            program_physical_composition.ideal_dag_duration_ns
            if program_physical_composition is not None
            else None
        ),
        empirical_hardware_floor_ns=(
            program_composition.selected_duration_ns
            if program_composition is not None
            else None
        ),
        provisional_estimate_ns=(
            program_provisional_composition.selected_duration_ns
            if program_provisional_composition is not None
            else None
        ),
        provisional_evidence_tier=(
            "exploratory" if program_provisional_composition is not None else None
        ),
        provisional_reason_codes=(
            provisional_reason_codes
            if program_provisional_composition is not None
            else ()
        ),
        limiting_resource=(
            "serialized.events" if program_composition is not None else None
        ),
        resource_limiting_resource=(
            program_physical_composition.limiting_resource
            if program_physical_composition is not None
            else None
        ),
        full_duration_ns=None,
        formula=(
            "selected = compose(candidate selected durations); "
            "resource_physical_floor = compose(max(minimum_flops / "
            "measured_fp32_P80, materialized_bytes / measured_memory_P80))"
        ),
        assumptions=assumptions,
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
        program_bounds,
    )
    return HardwareBackendPrediction(
        schema=PREDICTION_SCHEMA,
        backend_id=BACKEND_ID,
        backend_version=BACKEND_VERSION,
        compilation_fingerprint=fingerprint,
        placement=placement,
        hardware=hardware.metadata.name,
        device=device.id,
        status=(
            "phase-capabilities-incomplete"
            if any(
                candidate.phase_schedule is not None
                and candidate.phase_schedule.status == "unknown"
                for candidate in candidates
            )
            else status
        ),
        prediction_complete=False,
        capabilities=capability_snapshot,
        measured_capabilities=measured_snapshots,
        candidates=tuple(candidates),
        scope_bounds=tuple(scope_bounds),
        program_bounds=program_bounds,
    )


__all__ = ["compile_apple_m4_cpu_prediction"]
