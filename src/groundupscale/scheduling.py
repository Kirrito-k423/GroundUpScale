"""Schedule-aware composition of local hardware duration floors."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from groundupscale.schedule_evidence import finite_nonnegative
from groundupscale.schedule_model import ResourceCompositionKind, ScheduleKind


@dataclass(frozen=True)
class BoundEvent:
    """One schedulable event whose local overlap has already been resolved."""

    event_id: str
    predecessor_ids: tuple[str, ...]
    local_duration_ns: float
    resource_times_ns: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class ResourcePath:
    """One explicitly declared shared-resource load path."""

    path_id: str
    resource_id: str
    event_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConcurrencyGroup:
    """Events explicitly permitted to overlap across dependency branches."""

    group_id: str
    event_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScheduleBoundComposition:
    """Explainable lower bounds for two explicit schedule assumptions."""

    schedule: ScheduleKind
    serialized_duration_ns: float
    critical_path_duration_ns: float
    resource_duration_ns: float
    ideal_dag_duration_ns: float
    selected_duration_ns: float
    limiting_resource: str | None
    critical_path_event_ids: tuple[str, ...]


def compose_candidate_local_floor(
    resource_times_ns: tuple[float, ...],
    *,
    composition: ResourceCompositionKind,
) -> float:
    """Resolve candidate-local resources only under declared composition."""

    for index, duration_ns in enumerate(resource_times_ns):
        _validate_duration(duration_ns, f"resource_times_ns[{index}]")
    if not resource_times_ns:
        raise ValueError("candidate must declare at least one resource time")
    try:
        composition_kind = ResourceCompositionKind(composition)
    except ValueError as error:
        raise ValueError(f"unsupported resource composition: {composition}") from error
    if composition_kind is ResourceCompositionKind.IDENTITY:
        if len(resource_times_ns) != 1:
            raise ValueError("identity composition requires one resource time")
        return resource_times_ns[0]
    if composition_kind is ResourceCompositionKind.SERIAL:
        return sum(resource_times_ns)
    if composition_kind is ResourceCompositionKind.EXPLICIT_OVERLAP:
        if len(resource_times_ns) < 2:
            raise ValueError(
                "explicit-overlap composition requires multiple resource times"
            )
        return max(resource_times_ns)
    raise AssertionError("unreachable resource composition")


def compose_schedule_bound(
    events: tuple[BoundEvent, ...],
    *,
    schedule: ScheduleKind,
    resource_paths: tuple[ResourcePath, ...] = (),
    concurrency_groups: tuple[ConcurrencyGroup, ...] = (),
) -> ScheduleBoundComposition:
    """Compose local event floors without inventing cross-event overlap.

    ``serialized`` sums all local floors. ``dependency-only`` is the maximum
    of the dependency critical path and aggregate load on any named resource.
    Resource overlap inside an event is the responsibility of the producer of
    ``local_duration_ns``.
    """

    try:
        schedule_kind = ScheduleKind(schedule)
    except ValueError as error:
        raise ValueError(f"unsupported schedule: {schedule}") from error
    event_by_id = _validate_events(events)
    serialized_duration_ns = sum(event.local_duration_ns for event in events)
    critical_path_duration_ns, critical_path_event_ids = _critical_path(
        event_by_id
    )
    if schedule_kind is ScheduleKind.DEPENDENCY_ONLY:
        _validate_explicit_concurrency(event_by_id, concurrency_groups)

    if schedule_kind is ScheduleKind.SERIALIZED and not resource_paths:
        resource_paths = _serialized_resource_paths(events)
    resource_loads = _resource_path_loads(event_by_id, resource_paths)

    limiting_resource = None
    resource_duration_ns = 0.0
    if resource_loads:
        limiting_resource, resource_duration_ns = max(
            resource_loads.items(), key=lambda item: item[1]
        )

    ideal_dag_duration_ns = max(
        critical_path_duration_ns,
        resource_duration_ns,
    )
    if schedule_kind is ScheduleKind.SERIALIZED:
        selected_duration_ns = serialized_duration_ns
    elif schedule_kind is ScheduleKind.DEPENDENCY_ONLY:
        selected_duration_ns = ideal_dag_duration_ns
    else:
        raise AssertionError("unreachable schedule kind")

    return ScheduleBoundComposition(
        schedule=schedule_kind,
        serialized_duration_ns=serialized_duration_ns,
        critical_path_duration_ns=critical_path_duration_ns,
        resource_duration_ns=resource_duration_ns,
        ideal_dag_duration_ns=ideal_dag_duration_ns,
        selected_duration_ns=selected_duration_ns,
        limiting_resource=limiting_resource,
        critical_path_event_ids=critical_path_event_ids,
    )


def _serialized_resource_paths(
    events: tuple[BoundEvent, ...],
) -> tuple[ResourcePath, ...]:
    """Use the declared serialized event order as each resource path."""

    event_ids_by_resource: dict[str, list[str]] = defaultdict(list)
    for event in events:
        for resource_id, _ in event.resource_times_ns:
            event_ids_by_resource[resource_id].append(event.event_id)
    return tuple(
        ResourcePath(
            path_id=f"serialized-resource-path:{resource_id}",
            resource_id=resource_id,
            event_ids=tuple(event_ids),
            evidence_refs=("schedule://serialized-order",),
        )
        for resource_id, event_ids in event_ids_by_resource.items()
    )


def _resource_path_loads(
    event_by_id: dict[str, BoundEvent],
    resource_paths: tuple[ResourcePath, ...],
) -> dict[str, float]:
    claims: dict[tuple[str, str], float] = {}
    for event in event_by_id.values():
        for resource_id, duration_ns in event.resource_times_ns:
            key = (event.event_id, resource_id)
            if key in claims:
                raise ValueError(
                    f"event {event.event_id} repeats resource {resource_id}"
                )
            claims[key] = duration_ns

    assigned_claims: set[tuple[str, str]] = set()
    path_ids: set[str] = set()
    resource_ids: set[str] = set()
    resource_loads: dict[str, float] = defaultdict(float)
    for path in resource_paths:
        if (
            not path.path_id
            or path.path_id in path_ids
            or not path.resource_id
            or path.resource_id in resource_ids
            or not path.event_ids
            or not path.evidence_refs
            or len(set(path.event_ids)) != len(path.event_ids)
        ):
            raise ValueError("invalid explicit resource path")
        path_ids.add(path.path_id)
        resource_ids.add(path.resource_id)
        for event_id in path.event_ids:
            key = (event_id, path.resource_id)
            if event_id not in event_by_id or key not in claims:
                raise ValueError("invalid explicit resource path")
            if key in assigned_claims:
                raise ValueError("resource claim appears in multiple paths")
            assigned_claims.add(key)
            resource_loads[path.resource_id] += claims[key]
    if assigned_claims != set(claims):
        raise ValueError("every resource claim requires an explicit resource path")
    return resource_loads


def _validate_explicit_concurrency(
    event_by_id: dict[str, BoundEvent],
    groups: tuple[ConcurrencyGroup, ...],
) -> None:
    group_ids: set[str] = set()
    group_sets: list[set[str]] = []
    for group in groups:
        members = set(group.event_ids)
        if (
            not group.group_id
            or group.group_id in group_ids
            or len(members) < 2
            or len(members) != len(group.event_ids)
            or not members <= set(event_by_id)
            or not group.evidence_refs
        ):
            raise ValueError("invalid explicit concurrency group")
        group_ids.add(group.group_id)
        group_sets.append(members)

    ancestors: dict[str, set[str]] = {}

    def collect_ancestors(event_id: str) -> set[str]:
        if event_id in ancestors:
            return ancestors[event_id]
        result: set[str] = set()
        for predecessor_id in event_by_id[event_id].predecessor_ids:
            result.add(predecessor_id)
            result.update(collect_ancestors(predecessor_id))
        ancestors[event_id] = result
        return result

    for event_id in event_by_id:
        collect_ancestors(event_id)
    event_ids = tuple(event_by_id)
    for index, left in enumerate(event_ids):
        for right in event_ids[index + 1 :]:
            ordered = left in ancestors[right] or right in ancestors[left]
            declared_concurrent = any(
                {left, right} <= members for members in group_sets
            )
            if ordered and declared_concurrent:
                raise ValueError(
                    "explicit concurrency group contains ordered events"
                )
            if not ordered and not declared_concurrent:
                raise ValueError(
                    "unordered event pair requires explicit concurrency"
                )


def _validate_events(events: tuple[BoundEvent, ...]) -> dict[str, BoundEvent]:
    event_by_id: dict[str, BoundEvent] = {}
    for event in events:
        if not event.event_id:
            raise ValueError("event_id must not be empty")
        if event.event_id in event_by_id:
            raise ValueError(f"duplicate event_id: {event.event_id}")
        _validate_duration(event.local_duration_ns, f"{event.event_id}.local_duration_ns")
        for resource, duration_ns in event.resource_times_ns:
            if not resource:
                raise ValueError("resource name must not be empty")
            _validate_duration(duration_ns, f"{event.event_id}.{resource}")
        event_by_id[event.event_id] = event

    for event in events:
        for predecessor_id in event.predecessor_ids:
            if predecessor_id not in event_by_id:
                raise ValueError(
                    f"event {event.event_id} has unknown predecessor {predecessor_id}"
                )
    return event_by_id


def _validate_duration(duration_ns: float, label: str) -> None:
    if not finite_nonnegative(duration_ns):
        raise ValueError(f"{label} must be finite and non-negative")


def _critical_path(
    event_by_id: dict[str, BoundEvent],
) -> tuple[float, tuple[str, ...]]:
    indegree = {
        event_id: len(event.predecessor_ids)
        for event_id, event in event_by_id.items()
    }
    successors: dict[str, list[str]] = defaultdict(list)
    for event_id, event in event_by_id.items():
        for predecessor_id in event.predecessor_ids:
            successors[predecessor_id].append(event_id)
    ready = deque(
        event_id for event_id in event_by_id if indegree[event_id] == 0
    )
    longest_duration: dict[str, float] = {}
    longest_predecessor: dict[str, str | None] = {}
    processed = 0
    while ready:
        event_id = ready.popleft()
        event = event_by_id[event_id]
        predecessor_id = max(
            event.predecessor_ids,
            default=None,
            key=lambda item: longest_duration[item],
        )
        longest_predecessor[event_id] = predecessor_id
        longest_duration[event_id] = (
            longest_duration[predecessor_id] if predecessor_id is not None else 0.0
        ) + event.local_duration_ns
        processed += 1
        for successor_id in successors[event_id]:
            indegree[successor_id] -= 1
            if indegree[successor_id] == 0:
                ready.append(successor_id)
    if processed != len(event_by_id):
        raise ValueError("schedule dependencies contain a cycle")
    if not event_by_id:
        return 0.0, ()
    terminal_id = max(event_by_id, key=lambda item: longest_duration[item])
    reverse_path: list[str] = []
    current_id: str | None = terminal_id
    while current_id is not None:
        reverse_path.append(current_id)
        current_id = longest_predecessor[current_id]
    return longest_duration[terminal_id], tuple(reversed(reverse_path))
