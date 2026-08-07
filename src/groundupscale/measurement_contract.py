"""Typed cross-hardware measurement evidence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable


class MeasurementContractError(ValueError):
    """Recorded adapter evidence does not satisfy the portable contract."""


class ObservationFieldStatus(StrEnum):
    MEASURED = "measured"
    DERIVED = "derived"
    DECLARED = "declared"
    UNSUPPORTED = "unsupported"
    PERMISSION_DENIED = "permission_denied"
    NOT_REQUESTED = "not_requested"
    NOT_APPLICABLE = "not_applicable"
    COLLECTION_FAILED = "collection_failed"
    UNKNOWN = "unknown"

    @property
    def has_value(self) -> bool:
        return self in {
            self.MEASURED,
            self.DERIVED,
            self.DECLARED,
        }


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _finite_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ObservationField:
    name: str
    status: ObservationFieldStatus
    required_for_anchor: bool
    document: Mapping[str, object]

    @classmethod
    def from_document(cls, value: object) -> ObservationField:
        if not isinstance(value, dict):
            raise MeasurementContractError("invalid-observation-field")
        try:
            status = ObservationFieldStatus(value.get("status"))
        except (TypeError, ValueError) as error:
            raise MeasurementContractError("invalid-observation-field") from error
        name = value.get("field")
        derivation = value.get("derivation")
        derived_provenance_valid = (
            status is not ObservationFieldStatus.DERIVED
            or isinstance(derivation, dict)
            and _nonempty_string(derivation.get("method"))
            and isinstance(derivation.get("input_evidence_refs"), list)
            and bool(derivation["input_evidence_refs"])
            and all(
                _nonempty_string(reference)
                for reference in derivation["input_evidence_refs"]
            )
            and len(set(derivation["input_evidence_refs"]))
            == len(derivation["input_evidence_refs"])
        )
        if (
            not _nonempty_string(name)
            or not isinstance(value.get("required_for_anchor"), bool)
            or not all(
                _nonempty_string(value.get(key))
                for key in ("source", "scope", "attribution", "intrusion")
            )
            or status.has_value != ("value" in value)
            or not derived_provenance_valid
        ):
            raise MeasurementContractError("invalid-observation-field")
        return cls(
            name=name,
            status=status,
            required_for_anchor=value["required_for_anchor"],
            document=_freeze_json(value),
        )

    def to_document(self) -> dict[str, Any]:
        document = _thaw_json(self.document)
        if not isinstance(document, dict):
            raise MeasurementContractError("invalid-observation-field")
        return document


@dataclass(frozen=True)
class MeasurementCapabilityManifest:
    manifest_id: str
    adapter_id: str
    cohort_id: str
    fields: tuple[ObservationField, ...]
    evidence_ref: str

    @classmethod
    def from_document(
        cls,
        value: object,
        *,
        adapter_id: str,
        cohort_id: str,
    ) -> MeasurementCapabilityManifest:
        fields_value = value.get("fields") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or not _nonempty_string(value.get("manifest_id"))
            or value.get("adapter_id") != adapter_id
            or value.get("cohort_id") != cohort_id
            or not _nonempty_string(value.get("evidence_ref"))
            or not isinstance(fields_value, list)
            or not fields_value
        ):
            raise MeasurementContractError(
                "invalid-measurement-capability-manifest"
            )
        fields = tuple(ObservationField.from_document(field) for field in fields_value)
        names = [field.name for field in fields]
        if len(set(names)) != len(names):
            raise MeasurementContractError("invalid-observation-field")
        return cls(
            manifest_id=value["manifest_id"],
            adapter_id=adapter_id,
            cohort_id=cohort_id,
            fields=fields,
            evidence_ref=value["evidence_ref"],
        )

    @property
    def primary_timer_available(self) -> bool:
        timers = [field for field in self.fields if field.name == "timer.primary"]
        return (
            len(timers) == 1
            and timers[0].required_for_anchor
            and timers[0].status is ObservationFieldStatus.MEASURED
        )


COHORT_IDENTITY_DIMENSIONS = (
    "device",
    "partition",
    "topology",
    "software",
    "numeric_execution",
    "timer_protocol",
    "power_clock",
    "execution_context",
    "communication",
)


@dataclass(frozen=True)
class HardwareValidityIdentity:
    document: Mapping[str, object]

    @classmethod
    def from_document(cls, value: object) -> HardwareValidityIdentity:
        if not isinstance(value, dict):
            raise MeasurementContractError("invalid-cohort-identity")
        numeric = value.get("numeric_execution")
        timer = value.get("timer_protocol")
        power_clock = value.get("power_clock")
        execution_context = value.get("execution_context")
        communication = value.get("communication")
        communication_valid = isinstance(communication, dict) and (
            communication.get("status") == "not_applicable"
            or communication.get("status") == "applicable"
            and isinstance(communication.get("rank_count"), int)
            and not isinstance(communication.get("rank_count"), bool)
            and communication["rank_count"] > 0
            and all(
                _nonempty_string(communication.get(key))
                for key in ("topology", "backend", "algorithm", "routing")
            )
        )
        if not (
            all(
                _nonempty_string(value.get(key))
                for key in ("device", "partition", "topology", "software")
            )
            and isinstance(numeric, dict)
            and _nonempty_string(numeric.get("dtype"))
            and _nonempty_string(numeric.get("layout"))
            and isinstance(numeric.get("alignment_bytes"), int)
            and not isinstance(numeric.get("alignment_bytes"), bool)
            and numeric["alignment_bytes"] > 0
            and isinstance(numeric.get("threads"), int)
            and not isinstance(numeric.get("threads"), bool)
            and numeric["threads"] > 0
            and _nonempty_string(numeric.get("execution_mode"))
            and isinstance(timer, dict)
            and _nonempty_string(timer.get("source"))
            and _finite_number(timer.get("resolution_ns"))
            and timer["resolution_ns"] > 0
            and isinstance(timer.get("monotonic"), bool)
            and all(
                _nonempty_string(timer.get(key))
                for key in (
                    "completion_kind",
                    "adapter_id",
                    "adapter_version",
                    "protocol_id",
                    "protocol_version",
                )
            )
            and (
                _nonempty_string(timer.get("duration_reducer"))
                if timer.get("completion_kind")
                == "distributed-rank-local-duration"
                else timer.get("duration_reducer") is None
            )
            and isinstance(power_clock, dict)
            and _nonempty_string(power_clock.get("power_policy"))
            and _nonempty_string(power_clock.get("clock_policy"))
            and isinstance(execution_context, dict)
            and all(
                _nonempty_string(execution_context.get(key))
                for key in ("affinity", "numa", "context", "stream")
            )
            and isinstance(execution_context.get("concurrency"), int)
            and not isinstance(execution_context.get("concurrency"), bool)
            and execution_context["concurrency"] > 0
            and communication_valid
        ):
            raise MeasurementContractError("invalid-cohort-identity")
        frozen = _freeze_json(value)
        if not isinstance(frozen, Mapping):
            raise MeasurementContractError("invalid-cohort-identity")
        return cls(frozen)

    def changed_dimensions(
        self, reference: HardwareValidityIdentity
    ) -> list[str]:
        return [
            dimension
            for dimension in COHORT_IDENTITY_DIMENSIONS
            if self.document[dimension] != reference.document[dimension]
        ]


@dataclass(frozen=True)
class CohortPolicy:
    policy_id: str
    version: str
    maximum_retry_attempts: int

    @classmethod
    def from_document(cls, value: object) -> CohortPolicy:
        if not isinstance(value, dict) or not all(
            _nonempty_string(value.get(key))
            for key in (
                "policy_id",
                "version",
                "scope",
                "change_reason",
                "revalidation",
            )
        ):
            raise MeasurementContractError("invalid-cohort-policy")
        maximum = value.get("maximum_retry_attempts")
        if (
            not isinstance(maximum, int)
            or isinstance(maximum, bool)
            or maximum < 1
        ):
            raise MeasurementContractError("invalid-cohort-policy")
        return cls(value["policy_id"], value["version"], maximum)


@dataclass(frozen=True)
class ProfilingOverheadPolicy:
    policy_id: str
    version: str
    instrumentation_profiles: tuple[str, ...]
    validity_domain_ref: str
    maximum_overhead_ratio: float
    minimum_independent_sessions: int

    @classmethod
    def from_document(cls, value: object) -> ProfilingOverheadPolicy:
        if not isinstance(value, dict) or not all(
            _nonempty_string(value.get(key))
            for key in (
                "policy_id",
                "version",
                "scope",
                "change_reason",
                "revalidation",
                "validity_domain_ref",
            )
        ):
            raise MeasurementContractError("invalid-profiling-overhead-policy")
        profiles = value.get("instrumentation_profiles")
        maximum = value.get("maximum_overhead_ratio")
        minimum = value.get("minimum_independent_sessions")
        if (
            not isinstance(profiles, list)
            or not profiles
            or not all(_nonempty_string(profile) for profile in profiles)
            or len(set(profiles)) != len(profiles)
            or not _finite_number(maximum)
            or maximum < 0
            or not isinstance(minimum, int)
            or isinstance(minimum, bool)
            or minimum < 2
        ):
            raise MeasurementContractError("invalid-profiling-overhead-policy")
        return cls(
            value["policy_id"],
            value["version"],
            tuple(profiles),
            value["validity_domain_ref"],
            float(maximum),
            minimum,
        )


@dataclass(frozen=True)
class CompletionBoundary:
    kind: str

    @classmethod
    def from_document(cls, value: object) -> CompletionBoundary:
        if not isinstance(value, dict) or value.get("closed") is not True:
            raise MeasurementContractError("incomplete-completion-boundary")
        kind = value.get("kind")
        valid = False
        if kind in {"synchronous-cpu-call-return", "cpu-threadpool-join"}:
            valid = value.get("threadpool_joined") is True
        elif kind == "device-event-stream-completion":
            valid = (
                _nonempty_string(value.get("device_event_id"))
                and _nonempty_string(value.get("stream_id"))
                and value.get("stream_synchronized") is True
                and value.get("absolute_timestamps_subtracted") is not True
            )
        elif kind == "distributed-rank-local-duration":
            refs = value.get("rank_duration_refs")
            valid = (
                value.get("rank_local_durations") is True
                and value.get("absolute_timestamps_subtracted") is False
                and _nonempty_string(value.get("duration_reducer"))
                and isinstance(refs, list)
                and bool(refs)
                and all(_nonempty_string(reference) for reference in refs)
            )
        if not valid:
            raise MeasurementContractError("incomplete-completion-boundary")
        return cls(str(kind))


@dataclass(frozen=True)
class TimerEvidence:
    source: str
    resolution_ns: float

    @classmethod
    def from_documents(
        cls,
        timer: object,
        completion_boundary: object,
    ) -> TimerEvidence:
        boundary = CompletionBoundary.from_document(completion_boundary)
        if (
            not isinstance(timer, dict)
            or not _nonempty_string(timer.get("source"))
            or not _finite_number(timer.get("resolution_ns"))
            or timer["resolution_ns"] <= 0
            or timer.get("monotonic") is not True
        ):
            raise MeasurementContractError("invalid-primary-timer-protocol")
        if boundary.kind == "device-event-stream-completion" and not (
            timer.get("kind") == "device-event"
            and timer.get("device_event_id")
            == completion_boundary.get("device_event_id")
            and timer.get("stream_id") == completion_boundary.get("stream_id")
        ):
            raise MeasurementContractError("invalid-primary-timer-protocol")
        if boundary.kind == "distributed-rank-local-duration" and (
            timer.get("clock_domain") != "rank-local"
        ):
            raise MeasurementContractError("invalid-primary-timer-protocol")
        return cls(timer["source"], float(timer["resolution_ns"]))


@runtime_checkable
class MeasurementAdapter(Protocol):
    """Portable five-operation seam for hardware measurement adapters."""

    def discover_capabilities(self) -> Mapping[str, object]: ...

    def fingerprint_cohort(self) -> Mapping[str, object]: ...

    def preflight(self) -> Mapping[str, object]: ...

    def build_timing_plan(
        self, case: dict[str, object]
    ) -> Mapping[str, object]: ...

    def collect(
        self,
        case: dict[str, object],
        timing_plan: dict[str, object],
    ) -> Mapping[str, object]: ...


__all__ = [
    "COHORT_IDENTITY_DIMENSIONS",
    "CohortPolicy",
    "CompletionBoundary",
    "HardwareValidityIdentity",
    "MeasurementAdapter",
    "MeasurementCapabilityManifest",
    "MeasurementContractError",
    "ObservationField",
    "ObservationFieldStatus",
    "ProfilingOverheadPolicy",
    "TimerEvidence",
]
