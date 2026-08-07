"""Frozen domain types for explicit Schedule Frontier evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from groundupscale.schedule_evidence import finite_nonnegative


class ScheduleModelError(ValueError):
    """A schedule input violates the typed evidence contract."""

    def __init__(self, reason_code: str, **context: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.context = context


class ScheduleModelUnknown(ScheduleModelError):
    """Required schedule evidence is absent but the input is well formed."""


class ResourceKind(StrEnum):
    CAPACITY = "capacity"
    THROUGHPUT = "throughput"
    SLOT = "slot"
    EXCLUSIVE = "exclusive"


class ResourceCompositionKind(StrEnum):
    IDENTITY = "identity"
    SERIAL = "serial"
    EXPLICIT_OVERLAP = "explicit-overlap"


class DependencyKind(StrEnum):
    SEMANTIC_DATA = "semantic-data"
    EXECUTION_ORDER = "execution-order"
    EXECUTION_RESOURCE = "execution-resource"


class ScheduleKind(StrEnum):
    SERIALIZED = "serialized"
    DEPENDENCY_ONLY = "dependency-only"


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _string_pair(value: object, first: str, second: str) -> tuple[str, str]:
    if (
        not isinstance(value, dict)
        or not _nonempty_string(value.get(first))
        or not _nonempty_string(value.get(second))
    ):
        raise ScheduleModelError("invalid-resource-claim")
    return value[first], value[second]


def _evidence_refs(value: object, reason_code: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(_nonempty_string(reference) for reference in value)
    ):
        raise ScheduleModelUnknown(reason_code)
    return tuple(value)


@dataclass(frozen=True)
class ResourceClaim:
    claim_id: str
    resource_id: str
    kind: ResourceKind
    work_value: float
    work_unit: str
    duration_ns: float | None
    allocation_minimum: float
    allocation_maximum: float
    allocation_unit: str
    sharing: str
    lifetime_start: str
    lifetime_end: str
    affinity_kind: str
    affinity_value: str
    evidence_refs: tuple[str, ...]

    @classmethod
    def from_document(cls, value: object) -> ResourceClaim:
        claim_id = value.get("claim_id") if isinstance(value, dict) else None
        if not _nonempty_string(claim_id):
            raise ScheduleModelError("invalid-resource-claim")
        provenance = value.get("provenance")
        if provenance is None:
            raise ScheduleModelUnknown(
                "resource-claim-provenance-missing", claim_id=claim_id
            )
        evidence_refs = (
            provenance.get("evidence_refs")
            if isinstance(provenance, dict)
            else None
        )
        work = value.get("work_or_amount")
        bounds = value.get("allocation_bounds")
        try:
            kind = ResourceKind(value.get("kind"))
        except (TypeError, ValueError) as error:
            raise ScheduleModelError("invalid-resource-claim") from error
        if (
            not _nonempty_string(value.get("resource_id"))
            or not isinstance(work, dict)
            or not finite_nonnegative(work.get("value"))
            or not _nonempty_string(work.get("unit"))
            or not isinstance(bounds, dict)
            or not finite_nonnegative(bounds.get("minimum"))
            or not finite_nonnegative(bounds.get("maximum"))
            or float(bounds["minimum"]) > float(bounds["maximum"])
            or not _nonempty_string(bounds.get("unit"))
            or not _nonempty_string(value.get("sharing"))
            or not isinstance(evidence_refs, list)
            or not evidence_refs
            or not all(_nonempty_string(reference) for reference in evidence_refs)
        ):
            raise ScheduleModelError("invalid-resource-claim")
        lifetime_start, lifetime_end = _string_pair(
            value.get("lifetime"), "start", "end"
        )
        affinity_kind, affinity_value = _string_pair(
            value.get("affinity"), "kind", "value"
        )
        duration_ns = value.get("duration_ns")
        if duration_ns is not None and not finite_nonnegative(duration_ns):
            raise ScheduleModelError("invalid-resource-claim")
        return cls(
            claim_id=claim_id,
            resource_id=value["resource_id"],
            kind=kind,
            work_value=float(work["value"]),
            work_unit=work["unit"],
            duration_ns=float(duration_ns) if duration_ns is not None else None,
            allocation_minimum=float(bounds["minimum"]),
            allocation_maximum=float(bounds["maximum"]),
            allocation_unit=bounds["unit"],
            sharing=value["sharing"],
            lifetime_start=lifetime_start,
            lifetime_end=lifetime_end,
            affinity_kind=affinity_kind,
            affinity_value=affinity_value,
            evidence_refs=tuple(evidence_refs),
        )

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "claim_id": self.claim_id,
            "resource_id": self.resource_id,
            "kind": self.kind.value,
            "work_or_amount": {
                "value": self.work_value,
                "unit": self.work_unit,
            },
            "allocation_bounds": {
                "minimum": self.allocation_minimum,
                "maximum": self.allocation_maximum,
                "unit": self.allocation_unit,
            },
            "sharing": self.sharing,
            "lifetime": {
                "start": self.lifetime_start,
                "end": self.lifetime_end,
            },
            "affinity": {
                "kind": self.affinity_kind,
                "value": self.affinity_value,
            },
            "provenance": {"evidence_refs": list(self.evidence_refs)},
        }
        if self.duration_ns is not None:
            document["duration_ns"] = self.duration_ns
        return document


@dataclass(frozen=True)
class ResourceComposition:
    kind: ResourceCompositionKind
    claim_ids: tuple[str, ...]

    @classmethod
    def from_document(
        cls,
        value: object,
        *,
        candidate_id: str,
        claim_ids: tuple[str, ...],
    ) -> ResourceComposition:
        if value is None:
            raise ScheduleModelUnknown(
                "candidate-resource-composition-undeclared",
                candidate_id=candidate_id,
            )
        if not isinstance(value, dict):
            raise ScheduleModelError("invalid-resource-composition")
        try:
            kind = ResourceCompositionKind(value.get("kind"))
        except (TypeError, ValueError) as error:
            raise ScheduleModelError("invalid-resource-composition") from error
        value_claim_ids = value.get("claim_ids")
        if (
            not isinstance(value_claim_ids, list)
            or tuple(value_claim_ids) != claim_ids
        ):
            raise ScheduleModelError("resource-composition-claim-mismatch")
        return cls(kind=kind, claim_ids=claim_ids)

    def to_document(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "claim_ids": list(self.claim_ids)}


@dataclass(frozen=True)
class ImplementationCandidate:
    candidate_id: str
    role: str
    resource_claims: tuple[ResourceClaim, ...]
    resource_composition: ResourceComposition
    evidence_refs: tuple[str, ...]

    @classmethod
    def from_document(cls, value: object) -> ImplementationCandidate:
        candidate_id = value.get("candidate_id") if isinstance(value, dict) else None
        claims_value = value.get("resource_claims") if isinstance(value, dict) else None
        if (
            not _nonempty_string(candidate_id)
            or value.get("role") not in {"operator", "schedule-overhead"}
            or not isinstance(claims_value, list)
            or not claims_value
        ):
            raise ScheduleModelError("invalid-implementation-candidate")
        claims = tuple(ResourceClaim.from_document(item) for item in claims_value)
        claim_ids = tuple(claim.claim_id for claim in claims)
        if len(set(claim_ids)) != len(claim_ids):
            raise ScheduleModelError("invalid-resource-claim")
        composition = ResourceComposition.from_document(
            value.get("resource_composition"),
            candidate_id=candidate_id,
            claim_ids=claim_ids,
        )
        return cls(
            candidate_id=candidate_id,
            role=value["role"],
            resource_claims=claims,
            resource_composition=composition,
            evidence_refs=_evidence_refs(
                value.get("evidence_refs"), "candidate-evidence-missing"
            ),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "role": self.role,
            "resource_claims": [claim.to_document() for claim in self.resource_claims],
            "resource_composition": self.resource_composition.to_document(),
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class PhysicalExecutionEvent:
    event_id: str
    candidate: ImplementationCandidate
    duration_model_kind: str
    evidence_refs: tuple[str, ...]

    @classmethod
    def from_document(cls, value: object) -> PhysicalExecutionEvent:
        duration_model = (
            value.get("duration_model") if isinstance(value, dict) else None
        )
        if (
            not isinstance(value, dict)
            or not _nonempty_string(value.get("event_id"))
            or not isinstance(duration_model, dict)
            or duration_model.get("kind")
            != "candidate-local-resource-composition"
        ):
            raise ScheduleModelError("invalid-selected-execution-event")
        return cls(
            event_id=value["event_id"],
            candidate=ImplementationCandidate.from_document(
                value.get("candidate")
            ),
            duration_model_kind=duration_model["kind"],
            evidence_refs=_evidence_refs(
                value.get("evidence_refs"),
                "physical-event-evidence-missing",
            ),
        )

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    def to_selection_document(self) -> dict[str, str]:
        return {"event_id": self.event_id, "candidate_id": self.candidate_id}


@dataclass(frozen=True)
class RejectedCandidate:
    candidate_id: str
    reason_code: str
    evidence_refs: tuple[str, ...]

    @classmethod
    def from_document(cls, value: object) -> RejectedCandidate:
        if (
            not isinstance(value, dict)
            or not _nonempty_string(value.get("candidate_id"))
            or not _nonempty_string(value.get("reason_code"))
        ):
            raise ScheduleModelError("invalid-rejected-candidate")
        return cls(
            candidate_id=value["candidate_id"],
            reason_code=value["reason_code"],
            evidence_refs=_evidence_refs(
                value.get("evidence_refs"),
                "candidate-rejection-evidence-missing",
            ),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "reason_code": self.reason_code,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class ExecutionIRSelection:
    execution_ir_id: str
    schema: str
    state: str
    evidence_refs: tuple[str, ...]
    physical_events: tuple[PhysicalExecutionEvent, ...]
    rejected_candidates: tuple[RejectedCandidate, ...]
    semantic_dependencies: tuple[DependencyEdge, ...]
    execution_dependencies: tuple[DependencyEdge, ...]
    concurrency_groups: tuple[ConcurrencyDeclaration, ...]

    @classmethod
    def from_document(cls, value: object) -> ExecutionIRSelection:
        events = value.get("physical_events") if isinstance(value, dict) else None
        rejected = (
            value.get("rejected_candidates") if isinstance(value, dict) else None
        )
        semantic = (
            value.get("semantic_dependencies") if isinstance(value, dict) else None
        )
        execution = (
            value.get("execution_dependencies") if isinstance(value, dict) else None
        )
        concurrency = (
            value.get("concurrency_groups") if isinstance(value, dict) else None
        )
        if (
            not isinstance(value, dict)
            or not _nonempty_string(value.get("execution_ir_id"))
            or value.get("schema") != "groundupscale.dev/execution-ir/v1alpha1"
            or value.get("state") != "unscheduled"
            or not isinstance(events, list)
            or not isinstance(rejected, list)
            or not isinstance(semantic, list)
            or not isinstance(execution, list)
            or not isinstance(concurrency, list)
        ):
            raise ScheduleModelError("invalid-explicit-schedule-input")
        return cls(
            execution_ir_id=value["execution_ir_id"],
            schema=value["schema"],
            state=value["state"],
            evidence_refs=_evidence_refs(
                value.get("evidence_refs"), "execution-ir-evidence-missing"
            ),
            physical_events=tuple(
                PhysicalExecutionEvent.from_document(item) for item in events
            ),
            rejected_candidates=tuple(
                RejectedCandidate.from_document(item) for item in rejected
            ),
            semantic_dependencies=tuple(
                DependencyEdge.from_document(item) for item in semantic
            ),
            execution_dependencies=tuple(
                DependencyEdge.from_document(item) for item in execution
            ),
            concurrency_groups=tuple(
                ConcurrencyDeclaration.from_document(item) for item in concurrency
            ),
        )

    def header_document(self) -> dict[str, Any]:
        return {
            "execution_ir_id": self.execution_ir_id,
            "schema": self.schema,
            "state": self.state,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class DependencyEdge:
    edge_id: str
    source: str
    target: str
    kind: DependencyKind
    evidence_refs: tuple[str, ...]
    resource_id: str | None = None

    @classmethod
    def from_document(cls, value: object) -> DependencyEdge:
        if (
            not isinstance(value, dict)
            or not _nonempty_string(value.get("edge_id"))
            or not _nonempty_string(value.get("source"))
            or not _nonempty_string(value.get("target"))
            or value["source"] == value["target"]
        ):
            raise ScheduleModelError("invalid-explicit-dependency-edge")
        try:
            kind = DependencyKind(value.get("kind"))
        except (TypeError, ValueError) as error:
            raise ScheduleModelError("invalid-explicit-dependency-edge") from error
        resource_id = value.get("resource_id")
        if kind is DependencyKind.EXECUTION_RESOURCE:
            if not _nonempty_string(resource_id):
                raise ScheduleModelError("invalid-explicit-dependency-edge")
        elif resource_id is not None:
            raise ScheduleModelError("invalid-explicit-dependency-edge")
        return cls(
            edge_id=value["edge_id"],
            source=value["source"],
            target=value["target"],
            kind=kind,
            evidence_refs=_evidence_refs(
                value.get("evidence_refs"),
                "dependency-edge-evidence-missing",
            ),
            resource_id=resource_id,
        )

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value,
            "evidence_refs": list(self.evidence_refs),
        }
        if self.resource_id is not None:
            document["resource_id"] = self.resource_id
        return document


@dataclass(frozen=True)
class ConcurrencyDeclaration:
    group_id: str
    event_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    @classmethod
    def from_document(cls, value: object) -> ConcurrencyDeclaration:
        members = value.get("event_ids") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or not _nonempty_string(value.get("group_id"))
            or not isinstance(members, list)
            or len(members) < 2
            or len(set(members)) != len(members)
            or not all(_nonempty_string(member) for member in members)
        ):
            raise ScheduleModelError("invalid-concurrency-group")
        return cls(
            group_id=value["group_id"],
            event_ids=tuple(members),
            evidence_refs=_evidence_refs(
                value.get("evidence_refs"),
                "concurrency-group-evidence-missing",
            ),
        )


@dataclass(frozen=True)
class SchedulePolicy:
    policy_id: str
    version: str
    schedule: ScheduleKind

    @classmethod
    def from_document(cls, value: object) -> SchedulePolicy:
        if (
            not isinstance(value, dict)
            or not _nonempty_string(value.get("policy_id"))
            or not _nonempty_string(value.get("version"))
        ):
            raise ScheduleModelError("invalid-schedule-policy")
        try:
            schedule = ScheduleKind(value.get("schedule"))
        except (TypeError, ValueError) as error:
            raise ScheduleModelError("invalid-schedule-policy") from error
        return cls(value["policy_id"], value["version"], schedule)

    def to_document(self) -> dict[str, str]:
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "schedule": self.schedule.value,
        }


@dataclass(frozen=True)
class ScheduleFrontierDefinition:
    frontier_id: str
    version: str
    uncertainty_status: str
    uncertainty_reason_code: str | None
    uncertainty_evidence_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    @classmethod
    def from_document(cls, value: object) -> ScheduleFrontierDefinition:
        uncertainty = value.get("uncertainty") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or not _nonempty_string(value.get("frontier_id"))
            or not _nonempty_string(value.get("version"))
            or not isinstance(uncertainty, dict)
            or uncertainty.get("status") not in {"available", "unknown"}
        ):
            raise ScheduleModelError("invalid-schedule-frontier-definition")
        reason_code = uncertainty.get("reason_code")
        if uncertainty["status"] == "unknown" and not _nonempty_string(reason_code):
            raise ScheduleModelError("invalid-schedule-frontier-uncertainty")
        if uncertainty["status"] == "available" and reason_code is not None:
            raise ScheduleModelError("invalid-schedule-frontier-uncertainty")
        return cls(
            frontier_id=value["frontier_id"],
            version=value["version"],
            uncertainty_status=uncertainty["status"],
            uncertainty_reason_code=reason_code,
            uncertainty_evidence_refs=_evidence_refs(
                uncertainty.get("evidence_refs"),
                "schedule-frontier-uncertainty-evidence-missing",
            ),
            evidence_refs=_evidence_refs(
                value.get("evidence_refs"),
                "schedule-frontier-evidence-missing",
            ),
        )

    def uncertainty_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "status": self.uncertainty_status,
            "evidence_refs": list(self.uncertainty_evidence_refs),
        }
        if self.uncertainty_reason_code is not None:
            document["reason_code"] = self.uncertainty_reason_code
        return document


__all__ = [
    "ResourceClaim",
    "ResourceComposition",
    "ResourceCompositionKind",
    "ResourceKind",
    "DependencyEdge",
    "DependencyKind",
    "ConcurrencyDeclaration",
    "ExecutionIRSelection",
    "ImplementationCandidate",
    "ScheduleModelError",
    "ScheduleModelUnknown",
    "ScheduleFrontierDefinition",
    "ScheduleKind",
    "SchedulePolicy",
    "PhysicalExecutionEvent",
    "RejectedCandidate",
]
