"""Typed input adaptation and explicit Schedule Frontier composition."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from groundupscale.schedule_evidence import (
    ScheduleEvidenceUnknown,
    ScheduleFrontierError,
    canonical_digest,
    finite_nonnegative,
)
from groundupscale.schedule_model import (
    DependencyEdge,
    DependencyKind,
    ExecutionIRSelection,
    ImplementationCandidate,
    PhysicalExecutionEvent,
    ResourceCompositionKind,
    ResourceKind,
    ScheduleFrontierDefinition,
    ScheduleModelError,
    ScheduleModelUnknown,
    SchedulePolicy,
)
from groundupscale.scheduling import (
    BoundEvent,
    ConcurrencyGroup,
    ResourcePath,
    compose_candidate_local_floor,
    compose_schedule_bound,
)


AXIS_NAMES = (
    "resource_physical_floor",
    "operator_achievable_frontier",
    "schedule_achievable_frontier",
    "observation",
)
PROTOTYPE_CLASSIFICATION = {
    "aggregated",
    "prototype-only",
    "untrusted",
}
IMPLICIT_EFFECT_REJECTIONS = {
    "fusion": "fusion-requires-explicit-group",
    "concurrency": "concurrency-requires-explicit-execution-semantics",
    "communication-masking": (
        "communication-masking-requires-explicit-event-pair"
    ),
    "contention": "contention-requires-explicit-resource-claim",
    "dispatch": "dispatch-requires-explicit-transformation",
}
QUALIFICATION_GATES = {
    "baseline_timing_lane": (
        "qualified",
        "missing-qualified-baseline-timing-lane",
    ),
    "preflight": ("passed", "missing-passed-preflight"),
    "operator_frontier_anchors": (
        "qualified-active",
        "missing-qualified-active-operator-frontier-anchors",
    ),
}


@dataclass(frozen=True)
class _PreparedEvent:
    event: PhysicalExecutionEvent
    candidate: ImplementationCandidate
    local_duration_ns: float

    @property
    def resource_times_ns(self) -> tuple[tuple[str, float], ...]:
        return tuple(
            (claim.resource_id, claim.duration_ns)
            for claim in self.candidate.resource_claims
            if claim.duration_ns is not None
        )

    def to_document(self) -> dict[str, Any]:
        return {
            **self.candidate.to_document(),
            "event_id": self.event.event_id,
            "local_duration_ns": self.local_duration_ns,
            "resource_times_ns": [list(item) for item in self.resource_times_ns],
        }


def _raise_model(error: ScheduleModelError) -> None:
    if isinstance(error, ScheduleModelUnknown):
        raise ScheduleEvidenceUnknown(
            error.reason_code, **error.context
        ) from error
    raise ScheduleFrontierError(error.reason_code) from error


def prototype_fixture(document: Mapping[str, object]) -> dict[str, Any]:
    fixture = document.get("fixture")
    if not isinstance(fixture, dict):
        raise ScheduleFrontierError("missing-fixture-classification")
    classification = fixture.get("classification")
    if (
        not isinstance(classification, list)
        or set(classification) != PROTOTYPE_CLASSIFICATION
        or fixture.get("promotion_eligible") is not False
        or fixture.get("real_hardware_claim") is not None
    ):
        raise ScheduleFrontierError("invalid-prototype-fixture-classification")
    return deepcopy(fixture)


def untrusted_axes(
    document: Mapping[str, object],
) -> dict[str, dict[str, Any]]:
    axes = document.get("axes")
    if not isinstance(axes, dict) or tuple(axes) != AXIS_NAMES:
        raise ScheduleFrontierError("invalid-four-axis-input")
    result: dict[str, dict[str, Any]] = {}
    for name in AXIS_NAMES:
        axis = axes.get(name)
        if (
            not isinstance(axis, dict)
            or not finite_nonnegative(axis.get("fixture_duration_ns"))
        ):
            raise ScheduleFrontierError(f"invalid-{name.replace('_', '-')}")
        result[name] = {
            **deepcopy(axis),
            "status": "unknown",
            "reason_code": "prototype-only-untrusted-fixture",
        }
    return result


def _typed_candidates(value: object) -> tuple[ImplementationCandidate, ...]:
    if not isinstance(value, list) or not value:
        raise ScheduleFrontierError("invalid-implementation-candidate")
    candidates: list[ImplementationCandidate] = []
    candidate_ids: set[str] = set()
    for item in value:
        try:
            candidate = ImplementationCandidate.from_document(item)
        except ScheduleModelError as error:
            _raise_model(error)
        if candidate.candidate_id in candidate_ids:
            raise ScheduleFrontierError("invalid-implementation-candidate")
        candidate_ids.add(candidate.candidate_id)
        candidates.append(candidate)
    return tuple(candidates)


def _selected_candidates(
    candidates: tuple[ImplementationCandidate, ...],
    execution_ir: ExecutionIRSelection,
) -> tuple[tuple[PhysicalExecutionEvent, ImplementationCandidate], ...]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected_candidate_ids: set[str] = set()
    selected_event_ids: set[str] = set()
    selected: list[tuple[PhysicalExecutionEvent, ImplementationCandidate]] = []
    for event in execution_ir.physical_events:
        if event.event_id in selected_event_ids:
            raise ScheduleFrontierError("duplicate-selected-execution-event")
        if event.candidate_id in selected_candidate_ids:
            raise ScheduleFrontierError(
                "duplicate-implementation-candidate-selection"
            )
        candidate = candidate_by_id.get(event.candidate_id)
        if candidate is None or event.candidate != candidate:
            raise ScheduleFrontierError("invalid-selected-execution-event")
        selected_event_ids.add(event.event_id)
        selected_candidate_ids.add(event.candidate_id)
        selected.append((event, candidate))
    rejected_candidate_ids: set[str] = set()
    for rejection in execution_ir.rejected_candidates:
        if (
            rejection.candidate_id in rejected_candidate_ids
            or rejection.candidate_id in selected_candidate_ids
            or rejection.candidate_id not in candidate_by_id
        ):
            raise ScheduleFrontierError("invalid-rejected-candidate")
        rejected_candidate_ids.add(rejection.candidate_id)
    missing = [
        candidate.candidate_id
        for candidate in candidates
        if candidate.candidate_id
        not in selected_candidate_ids | rejected_candidate_ids
    ]
    if missing:
        raise ScheduleEvidenceUnknown(
            "execution-event-selection-incomplete", candidate_id=missing[0]
        )
    return tuple(selected)


def _prepare_events(
    selected: tuple[tuple[PhysicalExecutionEvent, ImplementationCandidate], ...],
) -> tuple[_PreparedEvent, ...]:
    prepared: list[_PreparedEvent] = []
    for event, candidate in selected:
        for claim in candidate.resource_claims:
            if claim.kind is not ResourceKind.THROUGHPUT or claim.duration_ns is None:
                raise ScheduleEvidenceUnknown(
                    "resource-kind-not-duration-bearing", claim_id=claim.claim_id
                )
        try:
            duration_ns = compose_candidate_local_floor(
                tuple(
                    claim.duration_ns
                    for claim in candidate.resource_claims
                    if claim.duration_ns is not None
                ),
                composition=candidate.resource_composition.kind.value,
            )
        except ValueError as error:
            raise ScheduleFrontierError(str(error)) from error
        prepared.append(_PreparedEvent(event, candidate, duration_ns))
    return tuple(prepared)


def _typed_edges(
    semantic_value: tuple[DependencyEdge, ...],
    execution_value: tuple[DependencyEdge, ...],
    event_ids: set[str],
) -> tuple[DependencyEdge, ...]:
    edges: list[DependencyEdge] = []
    edge_ids: set[str] = set()
    for value in [*semantic_value, *execution_value]:
        edge = value
        if (
            edge.edge_id in edge_ids
            or edge.source not in event_ids
            or edge.target not in event_ids
        ):
            raise ScheduleFrontierError("invalid-explicit-dependency-edge")
        edge_ids.add(edge.edge_id)
        edges.append(edge)
    return tuple(edges)


def _resource_paths(
    events: tuple[_PreparedEvent, ...],
    edges: tuple[DependencyEdge, ...],
) -> tuple[ResourcePath, ...]:
    event_ids_by_resource: dict[str, list[str]] = {}
    for event in events:
        for resource_id, _ in event.resource_times_ns:
            event_ids_by_resource.setdefault(resource_id, []).append(
                event.event.event_id
            )
    resource_edges: dict[str, list[DependencyEdge]] = {}
    for edge in edges:
        if edge.kind is DependencyKind.EXECUTION_RESOURCE:
            resource_edges.setdefault(edge.resource_id or "", []).append(edge)
    paths: list[ResourcePath] = []
    for resource_id, event_ids in event_ids_by_resource.items():
        event_id_set = set(event_ids)
        if len(event_ids) == 1:
            ordered_event_ids = tuple(event_ids)
        else:
            matching_edges = resource_edges.get(resource_id, [])
            successors: dict[str, str] = {}
            predecessor_ids: set[str] = set()
            for edge in matching_edges:
                if (
                    edge.source not in event_id_set
                    or edge.target not in event_id_set
                    or edge.source in successors
                    or edge.target in predecessor_ids
                ):
                    raise ScheduleEvidenceUnknown(
                        "invalid-execution-resource-path",
                        resource_id=resource_id,
                    )
                successors[edge.source] = edge.target
                predecessor_ids.add(edge.target)
            starts = event_id_set - predecessor_ids
            if len(matching_edges) != len(event_ids) - 1 or len(starts) != 1:
                raise ScheduleEvidenceUnknown(
                    "execution-resource-path-incomplete",
                    resource_id=resource_id,
                )
            ordered: list[str] = []
            current = next(iter(starts))
            while current not in ordered:
                ordered.append(current)
                if current not in successors:
                    break
                current = successors[current]
            if set(ordered) != event_id_set:
                raise ScheduleEvidenceUnknown(
                    "execution-resource-path-incomplete",
                    resource_id=resource_id,
                )
            ordered_event_ids = tuple(ordered)
        paths.append(
            ResourcePath(
                path_id=f"resource-path:{resource_id}",
                resource_id=resource_id,
                event_ids=ordered_event_ids,
                evidence_refs=tuple(
                    reference
                    for edge in resource_edges.get(resource_id, [])
                    for reference in edge.evidence_refs
                )
                or tuple(
                    reference
                    for event in events
                    if event.event.event_id in ordered_event_ids
                    for claim in event.candidate.resource_claims
                    if claim.resource_id == resource_id
                    for reference in claim.evidence_refs
                ),
            )
        )
    unknown_resources = set(resource_edges) - set(event_ids_by_resource)
    if unknown_resources:
        raise ScheduleEvidenceUnknown(
            "execution-resource-path-has-no-claims",
            resource_id=sorted(unknown_resources)[0],
        )
    return tuple(paths)


def _concurrency_groups(
    value: tuple[Any, ...], event_ids: set[str]
) -> tuple[ConcurrencyGroup, ...]:
    groups: list[ConcurrencyGroup] = []
    group_ids: set[str] = set()
    for item in value:
        group_id = item.group_id
        members = item.event_ids
        if (
            not group_id
            or group_id in group_ids
            or len(members) < 2
            or len(set(members)) != len(members)
            or not set(members) <= event_ids
        ):
            raise ScheduleFrontierError("invalid-concurrency-group")
        group_ids.add(group_id)
        groups.append(
            ConcurrencyGroup(group_id, tuple(members), item.evidence_refs)
        )
    return tuple(groups)


def schedule_frontier_replay_metadata(
    document: Mapping[str, object],
) -> dict[str, Any]:
    """Preserve immutable replay identity for available and unknown results."""

    frontier = document.get("schedule_frontier")
    if (
        not isinstance(frontier, dict)
        or not isinstance(frontier.get("frontier_id"), str)
        or not frontier["frontier_id"]
        or not isinstance(frontier.get("version"), str)
        or not frontier["version"]
    ):
        raise ScheduleFrontierError("invalid-schedule-frontier-definition")
    return {
        "frontier_identity": {
            "frontier_id": frontier["frontier_id"],
            "version": frontier["version"],
            "input_digest": canonical_digest(
                {
                    key: document[key]
                    for key in (
                        "schedule_policy",
                        "schedule_frontier",
                        "implementation_candidates",
                        "execution_ir",
                    )
                    if key in document
                }
            ),
            "evidence_digest": canonical_digest(
                {
                    key: document[key]
                    for key in (
                        "fixture",
                        "qualification",
                        "axes",
                        "ledger",
                        "transformations",
                        "requested_effects",
                    )
                    if key in document
                }
            ),
        },
        "uncertainty": deepcopy(frontier.get("uncertainty")),
        "evidence_refs": deepcopy(frontier.get("evidence_refs", [])),
    }


def compose_schedule_trace(
    document: Mapping[str, object], axes: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    try:
        policy = SchedulePolicy.from_document(document.get("schedule_policy"))
        ScheduleFrontierDefinition.from_document(document.get("schedule_frontier"))
        execution_ir = ExecutionIRSelection.from_document(
            document.get("execution_ir")
        )
    except ScheduleModelError as error:
        _raise_model(error)
    candidates = _typed_candidates(document.get("implementation_candidates"))
    selected = _selected_candidates(candidates, execution_ir)
    prepared = _prepare_events(selected)
    event_ids = {event.event.event_id for event in prepared}
    edges = _typed_edges(
        execution_ir.semantic_dependencies,
        execution_ir.execution_dependencies,
        event_ids,
    )
    predecessors = {event_id: [] for event_id in event_ids}
    for edge in edges:
        predecessors[edge.target].append(edge.source)
    bound_events = tuple(
        BoundEvent(
            event_id=event.event.event_id,
            predecessor_ids=tuple(predecessors[event.event.event_id]),
            local_duration_ns=event.local_duration_ns,
            resource_times_ns=event.resource_times_ns,
        )
        for event in prepared
    )
    resource_paths = _resource_paths(prepared, edges)
    concurrency_groups = _concurrency_groups(
        execution_ir.concurrency_groups, event_ids
    )
    try:
        bound = compose_schedule_bound(
            bound_events,
            schedule=policy.schedule.value,
            resource_paths=resource_paths,
            concurrency_groups=concurrency_groups,
        )
    except ValueError as error:
        raise ScheduleFrontierError(str(error)) from error
    operator_aggregate_ns = sum(
        event.local_duration_ns
        for event in prepared
        if event.candidate.role == "operator"
    )
    if operator_aggregate_ns != axes["operator_achievable_frontier"][
        "fixture_duration_ns"
    ]:
        raise ScheduleFrontierError("operator-aggregate-does-not-reconcile")
    if bound.selected_duration_ns != axes["schedule_achievable_frontier"][
        "fixture_duration_ns"
    ]:
        raise ScheduleFrontierError("schedule-frontier-does-not-reconcile")
    path_pairs = set(
        zip(bound.critical_path_event_ids, bound.critical_path_event_ids[1:])
    )
    edge_documents = [edge.to_document() for edge in edges]
    transformations = document.get("transformations")
    if not isinstance(transformations, list):
        raise ScheduleFrontierError("invalid-schedule-transformations")
    return {
        **schedule_frontier_replay_metadata(document),
        "policy": policy.to_document(),
        "execution_ir": execution_ir.header_document(),
        "implementation_candidates": [event.to_document() for event in prepared],
        "selected_events": [
            event.to_selection_document()
            for event in execution_ir.physical_events
        ],
        "rejected_candidates": [
            rejection.to_document()
            for rejection in execution_ir.rejected_candidates
        ],
        "explicit_edges": edge_documents,
        "resource_paths": [
            {
                "path_id": path.path_id,
                "resource_id": path.resource_id,
                "event_ids": list(path.event_ids),
                "evidence_refs": list(path.evidence_refs),
            }
            for path in resource_paths
        ],
        "concurrency_groups": [
            {
                "group_id": group.group_id,
                "event_ids": list(group.event_ids),
                "evidence_refs": list(group.evidence_refs),
            }
            for group in concurrency_groups
        ],
        "transformations": deepcopy(transformations),
        "overlap_claims": [
            {
                "candidate_id": event.candidate.candidate_id,
                "composition": event.candidate.resource_composition.to_document(),
                "evidence_refs": list(event.candidate.evidence_refs),
            }
            for event in prepared
            if event.candidate.resource_composition.kind
            is ResourceCompositionKind.EXPLICIT_OVERLAP
        ],
        "operator_aggregate_ns": operator_aggregate_ns,
        "composition": {
            "serialized_duration_ns": bound.serialized_duration_ns,
            "critical_path_duration_ns": bound.critical_path_duration_ns,
            "resource_duration_ns": bound.resource_duration_ns,
            "ideal_dag_duration_ns": bound.ideal_dag_duration_ns,
            "selected_duration_ns": bound.selected_duration_ns,
            "limiting_resource": bound.limiting_resource,
        },
        "critical_path_event_ids": list(bound.critical_path_event_ids),
        "dependency_path": [
            edge
            for edge in edge_documents
            if (edge["source"], edge["target"]) in path_pairs
        ],
    }


def selected_candidate_ids(document: Mapping[str, object]) -> set[str]:
    """Read planner-selected candidate identities without composing Schedule."""

    execution_ir = document.get("execution_ir")
    events = (
        execution_ir.get("physical_events")
        if isinstance(execution_ir, dict)
        else None
    )
    if not isinstance(events, list):
        raise ScheduleFrontierError("invalid-explicit-schedule-input")
    result: set[str] = set()
    for event in events:
        candidate = event.get("candidate") if isinstance(event, dict) else None
        candidate_id = (
            candidate.get("candidate_id") if isinstance(candidate, dict) else None
        )
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ScheduleFrontierError("invalid-selected-execution-event")
        if candidate_id in result:
            raise ScheduleFrontierError(
                "duplicate-implementation-candidate-selection"
            )
        result.add(candidate_id)
    return result


def effect_rejections(document: Mapping[str, object]) -> list[dict[str, str]]:
    requests = document.get("requested_effects", [])
    if not isinstance(requests, list):
        raise ScheduleFrontierError("invalid-schedule-effect-requests")
    rejections: list[dict[str, str]] = []
    for request in requests:
        effect_id = request.get("effect_id") if isinstance(request, dict) else None
        kind = request.get("kind") if isinstance(request, dict) else None
        if (
            not isinstance(effect_id, str)
            or not effect_id
            or kind not in IMPLICIT_EFFECT_REJECTIONS
        ):
            raise ScheduleFrontierError("invalid-schedule-effect-request")
        rejections.append(
            {
                "effect_id": effect_id,
                "kind": kind,
                "status": "rejected",
                "reason_code": IMPLICIT_EFFECT_REJECTIONS[kind],
            }
        )
    return rejections


def evidence_qualification(
    document: Mapping[str, object], axes: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    qualification = document.get("qualification")
    if not isinstance(qualification, dict):
        raise ScheduleFrontierError("missing-evidence-qualification")
    gates: dict[str, dict[str, str]] = {}
    for gate_name, (required_status, reason_code) in QUALIFICATION_GATES.items():
        status = qualification.get(gate_name)
        if status == required_status:
            gates[gate_name] = {"status": "available"}
        elif status == "not-provided":
            gates[gate_name] = {
                "status": "unknown",
                "reason_code": reason_code,
            }
        else:
            raise ScheduleFrontierError(
                f"invalid-{gate_name.replace('_', '-')}-qualification"
            )
    anchors = gates["operator_frontier_anchors"]
    if anchors["status"] == "unknown":
        axes["operator_achievable_frontier"]["reason_code"] = anchors[
            "reason_code"
        ]
        axes["schedule_achievable_frontier"][
            "reason_code"
        ] = "operator-frontier-unknown"
    baseline = gates["baseline_timing_lane"]
    preflight = gates["preflight"]
    if baseline["status"] == "unknown":
        axes["observation"]["reason_code"] = baseline["reason_code"]
    elif preflight["status"] == "unknown":
        axes["observation"]["reason_code"] = preflight["reason_code"]
    return {
        "status": (
            "available"
            if all(gate["status"] == "available" for gate in gates.values())
            else "unknown"
        ),
        "gates": gates,
    }


__all__ = [
    "AXIS_NAMES",
    "QUALIFICATION_GATES",
    "compose_schedule_trace",
    "effect_rejections",
    "evidence_qualification",
    "prototype_fixture",
    "untrusted_axes",
]
