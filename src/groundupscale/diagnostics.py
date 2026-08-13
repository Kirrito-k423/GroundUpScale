"""Evidence-qualified diagnosis for one exact-Shape Run Bundle."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from math import hypot, isclose, isfinite
from pathlib import Path
from re import fullmatch
from statistics import median
from typing import Any, cast
from unicodedata import normalize

from groundupscale.measurement_contract import (
    CohortPolicy,
    CompletionBoundary,
    HardwareValidityIdentity,
    MeasurementCapabilityManifest,
    MeasurementContractError,
    ProfilingOverheadPolicy,
    TimerEvidence,
)
from groundupscale.operator_shape_semantics import (
    UnsupportedOperatorShape,
    semantics_from_surface_query,
)
from groundupscale.run_bundle import verify_run_bundle

DIAGNOSTIC_EVIDENCE_SCHEMA = (
    "groundupscale.dev/diagnostic-evidence/v1alpha1"
)
DIAGNOSTIC_RESULT_SCHEMA = (
    "groundupscale.dev/diagnostic-result/v1alpha1"
)
PERFORMANCE_DIAGNOSIS_VERDICTS = (
    "frontier_shift",
    "implementation_headroom",
    "integration_overhead",
    "suspected_regression",
    "insufficient_evidence",
    "confirmed_bug",
)
_FRONTIER_MINIMUM_INDEPENDENT_SESSIONS = 3
_DIRECT_DEFECT_GATE_IDS = {
    "correctness_oracle_violation": "direct-correctness-violation",
    "execution_contract_violation": (
        "direct-execution-contract-violation"
    ),
}

_INPUT_KEYS = (
    "resolved_configuration",
    "resolved_ir",
    "hardware",
    "cohort_id",
    "execution_domain",
)
_TRANSIENT_COHORT_FAILURES = frozenset(
    {
        "throttling",
        "contention",
        "health",
        "dispersion",
        "timer",
        "device",
        "collection",
    }
)


class DiagnosticBundleError(ValueError):
    """The Run Bundle cannot produce a trustworthy diagnostic result."""


class DiagnosticBundleIntegrityError(DiagnosticBundleError):
    """A manifest or authored evidence digest did not verify."""


@dataclass(frozen=True)
class _SelectedSurfaceCell:
    cell: dict[str, Any]
    anchors: tuple[dict[str, Any], ...]
    weights: tuple[float, ...]
    exact_anchor: bool
    effective_rate: float
    primary_latency_ns: float | None = None


@dataclass(frozen=True)
class _RejectedSurfaceCell:
    cell: dict[str, Any]
    reason_code: str


@dataclass(frozen=True)
class _SurfaceUncertainty:
    anchor_standard_rate: float
    interpolation_standard_rate: float
    instrumentation_standard_rate: float
    combined_standard_rate: float
    rate_low: float
    rate_high: float


@dataclass(frozen=True)
class _LatencyUncertainty:
    anchor_standard_ns: float
    interpolation_standard_ns: float
    instrumentation_standard_ns: float
    combined_standard_ns: float
    latency_low_ns: float
    latency_high_ns: float


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _exact_versioned_identity(
    value: object,
    *,
    allow_not_applicable: bool = False,
) -> bool:
    """Reject cohort identities that cannot be reproduced exactly."""
    return (
        isinstance(value, dict)
        and _resolved_identity_string(value.get("name"))
        and _resolved_identity_string(value.get("version"))
        and (
            value.get("status") == "resolved"
            or (
                allow_not_applicable
                and value.get("status") == "not_applicable"
            )
        )
        and fullmatch(
            r"v?\d+(?:[._+-][0-9A-Za-z]+)*", value["version"]
        )
        is not None
    )


def _exact_version_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and not any(
            token in value.casefold()
            for token in ("unknown", "unspecified", "latest", "unversioned")
        )
        and fullmatch(r"v?\d+(?:[._+-][0-9A-Za-z]+)*", value) is not None
    )


def _surface_version_text(value: object) -> bool:
    """Accept legacy numeric versions or immutable content-hash versions."""
    return _exact_version_text(value) or (
        isinstance(value, str)
        and fullmatch(r"v-[0-9a-f]{16,64}", value) is not None
    )


def _surface_id_text(value: object) -> bool:
    """Accept legacy identifiers or canonical hierarchical surface URIs."""
    return _canonical_identifier(value) or (
        isinstance(value, str)
        and fullmatch(
            r"surface://[a-z0-9][a-z0-9._+-]*"
            r"(?:/[A-Za-z0-9][A-Za-z0-9._+-]*)+",
            value,
        )
        is not None
    )


def _raw_correctness_passed(
    records: object, tolerance: object
) -> bool | None:
    if (
        not isinstance(records, list)
        or not records
        or not isinstance(tolerance, dict)
        or not _finite_number(tolerance.get("atol"))
        or not _finite_number(tolerance.get("rtol"))
        or float(tolerance["atol"]) < 0
        or float(tolerance["rtol"]) < 0
    ):
        return None
    if not all(
        isinstance(record, dict)
        and _finite_number(record.get("expected"))
        and _finite_number(record.get("observed"))
        for record in records
    ):
        return None
    return all(
        abs(float(record["observed"]) - float(record["expected"]))
        <= float(tolerance["atol"])
        + float(tolerance["rtol"]) * abs(float(record["expected"]))
        for record in records
    )


def _unknown(reason_code: str) -> dict[str, Any]:
    return {
        "status": "unknown",
        "reason_code": reason_code,
        "evidence_refs": [],
    }


def _nonempty_string(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and value == normalize("NFC", value)
        and all(character.isprintable() for character in value)
    )


def _known_identity_string(value: object) -> bool:
    return (
        _nonempty_string(value)
        and not any(
            token in value.casefold()
            for token in ("unknown", "unspecified", "unversioned", "latest")
        )
        and all(
            segment not in {"", ".", ".."}
            for segment in value.split("/")
        )
    )


def _canonical_identifier(value: object) -> bool:
    return (
        _resolved_identity_string(value)
        and fullmatch(r"[a-z0-9][a-z0-9._+-]*", value) is not None
        and value not in {".", ".."}
    )


def _canonical_source_identity(value: object) -> bool:
    if not _known_identity_string(value) or value != value.casefold():
        return False
    segments = value.split("/")
    return len(segments) >= 2 and all(
        _canonical_identifier(segment) for segment in segments
    )


def _canonical_stable_path(value: object) -> bool:
    return _canonical_source_identity(value)


def _resolved_identity_string(value: object) -> bool:
    return (
        _known_identity_string(value)
        and value.casefold() not in {"n/a", "na", "none", "not_applicable"}
    )


def _finite_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _complete_execution_domain(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    shape = value.get("shape")
    return (
        isinstance(shape, dict)
        and bool(shape)
        and all(
            _nonempty_string(dimension)
            and isinstance(size, int)
            and not isinstance(size, bool)
            and size > 0
            for dimension, size in shape.items()
        )
        and all(
            _nonempty_string(value.get(key))
            for key in ("dtype", "layout", "execution_mode")
        )
        and isinstance(value.get("alignment_bytes"), int)
        and not isinstance(value["alignment_bytes"], bool)
        and value["alignment_bytes"] > 0
        and isinstance(value.get("threads"), int)
        and not isinstance(value["threads"], bool)
        and value["threads"] > 0
    )


def _complete_required_identity(document: dict[str, Any]) -> bool:
    configuration = document.get("resolved_configuration")
    resolved_ir = document.get("resolved_ir")
    hardware = document.get("hardware")
    return (
        isinstance(configuration, dict)
        and all(
            _nonempty_string(configuration.get(key))
            for key in ("analysis_plan", "benchmark_case")
        )
        and isinstance(resolved_ir, dict)
        and all(
            _nonempty_string(resolved_ir.get(key))
            for key in ("semantic_node", "operation")
        )
        and isinstance(hardware, dict)
        and all(
            _nonempty_string(hardware.get(key))
            for key in ("device", "partition", "topology", "software")
        )
        and _nonempty_string(document.get("cohort_id"))
    )


def _versioned_policy(
    document: dict[str, Any], policy_name: str
) -> dict[str, Any] | None:
    policies = document.get("policies")
    if not isinstance(policies, dict):
        return None
    policy = policies.get(policy_name)
    if not isinstance(policy, dict):
        return None
    if not all(
        _resolved_identity_string(policy.get(key))
        for key in (
            "policy_id",
            "version",
            "scope",
            "change_reason",
            "revalidation",
        )
    ):
        return None
    return policy


def _primary_timer_available(document: dict[str, Any]) -> bool:
    adapter = document.get("measurement_adapter")
    if not isinstance(adapter, dict):
        return False
    try:
        manifest = MeasurementCapabilityManifest.from_document(
            document.get("measurement_capability_manifest"),
            adapter_id=str(adapter.get("adapter_id", "")),
            cohort_id=str(document.get("cohort_id", "")),
        )
    except MeasurementContractError:
        return False
    return manifest.primary_timer_available


def _completion_boundary_valid(value: object) -> bool:
    try:
        CompletionBoundary.from_document(value)
    except MeasurementContractError:
        return False
    return True


def _timer_evidence_valid(timer: object, completion: object) -> bool:
    try:
        TimerEvidence.from_documents(timer, completion)
    except MeasurementContractError:
        return False
    return True


def _current_cohort_identity(document: dict[str, Any]) -> dict[str, Any]:
    hardware = document.get("hardware")
    execution_domain = document.get("execution_domain")
    baseline = document.get("baseline_timing_lane")
    adapter = document.get("measurement_adapter")
    hardware_value = hardware if isinstance(hardware, dict) else {}
    domain_value = execution_domain if isinstance(execution_domain, dict) else {}
    baseline_value = baseline if isinstance(baseline, dict) else {}
    timer = baseline_value.get("timer")
    timer_value = timer if isinstance(timer, dict) else {}
    completion = baseline_value.get("completion_boundary")
    completion_value = completion if isinstance(completion, dict) else {}
    adapter_value = adapter if isinstance(adapter, dict) else {}
    return {
        "device": hardware_value.get("device"),
        "partition": hardware_value.get("partition"),
        "topology": hardware_value.get("topology"),
        "software": hardware_value.get("software"),
        "power_clock": hardware_value.get("power_clock"),
        "numeric_execution": {
            key: domain_value.get(key)
            for key in (
                "dtype",
                "layout",
                "alignment_bytes",
                "threads",
                "execution_mode",
            )
        },
        "timer_protocol": {
            "source": timer_value.get("source"),
            "resolution_ns": timer_value.get("resolution_ns"),
            "monotonic": timer_value.get("monotonic"),
            "completion_kind": completion_value.get("kind"),
            "duration_reducer": completion_value.get("duration_reducer"),
            "adapter_id": adapter_value.get("adapter_id"),
            "adapter_version": adapter_value.get("adapter_version"),
            "protocol_id": adapter_value.get("protocol_id"),
            "protocol_version": adapter_value.get("protocol_version"),
        },
        "execution_context": {
            key: domain_value.get(key)
            for key in (
                "affinity",
                "numa",
                "context",
                "stream",
                "concurrency",
            )
        },
        "communication": document.get("communication_identity"),
    }


def _cohort_state(document: dict[str, Any]) -> dict[str, Any]:
    evidence = document.get("cohort_evidence")
    if evidence is None:
        return {
            "status": "insufficient_evidence",
            "reason_code": "invalid-cohort-evidence",
            "cohort_id": document.get("cohort_id"),
            "reference_cohort_id": None,
            "changed_dimensions": [],
            "retry": {"status": "not_authorized"},
            "evidence_refs": [],
        }
    policy_value = _versioned_policy(document, "cohort")
    reference_identity = (
        evidence.get("reference_identity")
        if isinstance(evidence, dict)
        else None
    )
    current_identity = _current_cohort_identity(document)
    observed_identity = (
        evidence.get("observed_identity")
        if isinstance(evidence, dict)
        else None
    )
    failures = (
        evidence.get("transient_failures")
        if isinstance(evidence, dict)
        else None
    )
    try:
        policy = CohortPolicy.from_document(policy_value)
        reference = HardwareValidityIdentity.from_document(reference_identity)
        current = HardwareValidityIdentity.from_document(current_identity)
        observed = HardwareValidityIdentity.from_document(observed_identity)
    except MeasurementContractError:
        policy = None
        reference = None
        current = None
        observed = None
    if (
        not isinstance(evidence, dict)
        or policy is None
        or not _nonempty_string(evidence.get("reference_cohort_id"))
        or not _nonempty_string(evidence.get("evidence_ref"))
        or reference is None
        or current is None
        or observed is None
        or observed.document != current.document
        or not isinstance(failures, list)
    ):
        return {
            "status": "insufficient_evidence",
            "reason_code": "invalid-cohort-evidence",
            "cohort_id": document.get("cohort_id"),
            "reference_cohort_id": (
                evidence.get("reference_cohort_id")
                if isinstance(evidence, dict)
                else None
            ),
            "changed_dimensions": [],
            "retry": {"status": "not_authorized"},
            "evidence_refs": [],
        }
    changed_dimensions = current.changed_dimensions(reference)
    if failures:
        if not all(
            isinstance(failure, dict)
            and failure.get("kind") in _TRANSIENT_COHORT_FAILURES
            and failure.get("retryable") is True
            and _nonempty_string(failure.get("evidence_ref"))
            for failure in failures
        ):
            return {
                "status": "insufficient_evidence",
                "reason_code": "invalid-transient-cohort-failure",
                "cohort_id": document.get("cohort_id"),
                "reference_cohort_id": evidence["reference_cohort_id"],
                "changed_dimensions": changed_dimensions,
                "retry": {"status": "not_authorized"},
                "evidence_refs": [evidence["evidence_ref"]],
            }
        attempt = evidence.get("retry_attempt")
        maximum_attempts = policy.maximum_retry_attempts
        if (
            not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or attempt < 1
            or not isinstance(maximum_attempts, int)
            or isinstance(maximum_attempts, bool)
            or maximum_attempts < attempt
        ):
            return {
                "status": "insufficient_evidence",
                "reason_code": "invalid-cohort-retry-policy",
                "cohort_id": document.get("cohort_id"),
                "reference_cohort_id": evidence["reference_cohort_id"],
                "changed_dimensions": changed_dimensions,
                "retry": {"status": "not_authorized"},
                "evidence_refs": [evidence["evidence_ref"]],
            }
        return {
            "status": "quarantined",
            "cohort_id": document.get("cohort_id"),
            "reference_cohort_id": evidence["reference_cohort_id"],
            "changed_dimensions": changed_dimensions,
            "transient_failures": [failure["kind"] for failure in failures],
            "retry": {
                "status": "required",
                "attempt": attempt,
                "maximum_attempts": maximum_attempts,
                "policy_ref": f"{policy.policy_id}/{policy.version}",
            },
            "evidence_refs": [
                evidence["evidence_ref"],
                *(failure["evidence_ref"] for failure in failures),
            ],
        }
    cohort_id = document.get("cohort_id")
    reference_cohort_id = evidence["reference_cohort_id"]
    if changed_dimensions and cohort_id != reference_cohort_id:
        status = "split"
        reason_code = None
    elif not changed_dimensions and cohort_id == reference_cohort_id:
        status = "matched"
        reason_code = None
    elif changed_dimensions:
        status = "insufficient_evidence"
        reason_code = "cohort-change-not-split"
    else:
        status = "insufficient_evidence"
        reason_code = "unsubstantiated-cohort-split"
    result = {
        "status": status,
        "cohort_id": cohort_id,
        "reference_cohort_id": reference_cohort_id,
        "changed_dimensions": changed_dimensions,
        "retry": {"status": "not_required"},
        "evidence_refs": [evidence["evidence_ref"]],
    }
    if reason_code is not None:
        result["reason_code"] = reason_code
    return result


def _load_evidence(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    verification = verify_run_bundle(root)
    if not verification["passed"]:
        raise DiagnosticBundleIntegrityError(
            "; ".join(str(failure) for failure in verification["failures"])
        )

    manifest = json.loads(
        (root / "run.manifest.json").read_text(encoding="utf-8")
    )
    artifacts = [
        artifact
        for artifact in manifest["artifacts"]
        if artifact.get("role") == "diagnostic-evidence"
    ]
    if len(artifacts) != 1:
        raise DiagnosticBundleError(
            "Run Bundle must contain exactly one diagnostic-evidence artifact"
        )
    evidence_path = (root / artifacts[0]["path"]).resolve()
    if root not in evidence_path.parents:
        raise DiagnosticBundleIntegrityError(
            f"diagnostic evidence path escapes bundle: {artifacts[0]['path']}"
        )
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    if document.get("schema") != DIAGNOSTIC_EVIDENCE_SCHEMA:
        raise DiagnosticBundleError("unsupported diagnostic evidence schema")
    return manifest, document


def _verified_document_digests(document: dict[str, Any]) -> dict[str, str]:
    missing_inputs = [key for key in _INPUT_KEYS if key not in document]
    if missing_inputs:
        missing = ", ".join(missing_inputs)
        raise DiagnosticBundleError(f"diagnostic evidence missing fields: {missing}")

    inputs = {key: document[key] for key in _INPUT_KEYS}
    evidence = {
        key: value
        for key, value in document.items()
        if key not in {*_INPUT_KEYS, "schema", "digests"}
    }
    expected = document.get("digests")
    if not isinstance(expected, dict):
        raise DiagnosticBundleIntegrityError("diagnostic evidence has no digests")
    actual = {
        "input_sha256": _canonical_digest(inputs),
        "evidence_sha256": _canonical_digest(evidence),
    }
    for name, digest in actual.items():
        if expected.get(name) != digest:
            raise DiagnosticBundleIntegrityError(
                f"diagnostic {name} mismatch"
            )
    return actual


def _resource_axis(
    document: dict[str, Any],
    verified_artifacts: dict[str, object] | None = None,
) -> dict[str, Any]:
    floor = document.get("resource_physical_floor")
    if not isinstance(floor, dict):
        return _unknown("missing-resource-physical-floor")
    if floor.get("status") != "known" or floor.get("value_ns") is None:
        return _unknown(
            str(floor.get("reason_code", "missing-resource-physical-floor"))
        )
    if not _complete_execution_domain(document.get("execution_domain")):
        return _unknown("incomplete-resource-physical-floor-evidence")
    terms = floor.get("resource_terms")
    if (
        floor.get("combination") != "max-explicit-overlap"
        or not isinstance(floor.get("policy_ref"), str)
        or not floor.get("evidence_refs")
        or not isinstance(terms, list)
        or not terms
    ):
        return _unknown("incomplete-resource-physical-floor-evidence")
    term_durations: list[float] = []
    for term in terms:
        minimum_demand = term.get("minimum_demand")
        validated_rate = term.get("validated_rate_per_second")
        if (
            not isinstance(minimum_demand, (int, float))
            or minimum_demand < 0
            or not isinstance(validated_rate, (int, float))
            or validated_rate <= 0
            or term.get("validated") is not True
            or not _nonempty_string(term.get("resource"))
            or term.get("validated_rate_resource") != term.get("resource")
            or term.get("cohort_id") != document["cohort_id"]
            or term.get("execution_domain") != document["execution_domain"]
            or not _nonempty_string(term.get("evidence_ref"))
        ):
            return _unknown("incomplete-resource-physical-floor-evidence")
        demand_unit = term.get("demand_unit")
        if (
            not _nonempty_string(demand_unit)
            or term.get("rate_unit") != f"{demand_unit}/s"
        ):
            return _unknown("resource-physical-floor-unit-mismatch")
        term_durations.append(minimum_demand / validated_rate * 1_000_000_000)
    derived_value = max(term_durations)
    if abs(float(floor["value_ns"]) - derived_value) > max(
        1e-9, derived_value * 1e-12
    ):
        return _unknown("resource-physical-floor-derivation-mismatch")
    source_refs = {term.get("source_evidence_ref") for term in terms}
    if source_refs != {None}:
        if (
            verified_artifacts is None
            or len(source_refs) != 1
            or not _artifact_uri(next(iter(source_refs)))
        ):
            return _unknown("resource-physical-floor-source-evidence-invalid")
        source = _verified_artifact_content(
            verified_artifacts,
            next(iter(source_refs)),
            role="source-physical-floor",
            schema=(
                "groundupscale.dev/"
                "physical-floor-observation-comparison/v1alpha1"
            ),
        )
        source_floor = (
            source.get("physical_floor") if isinstance(source, dict) else None
        )
        source_capabilities = (
            source_floor.get("capabilities")
            if isinstance(source_floor, dict)
            else None
        )
        capability_by_resource = (
            {
                capability.get("resource"): capability
                for capability in source_capabilities
                if isinstance(capability, dict)
            }
            if isinstance(source_capabilities, list)
            else {}
        )
        source_demands = {
            "compute.fp32": source_floor.get("minimum_work_flops")
            if isinstance(source_floor, dict)
            else None,
            "memory.hbm": source_floor.get("compulsory_bytes")
            if isinstance(source_floor, dict)
            else None,
        }
        if (
            not isinstance(source_floor, dict)
            or not _finite_number(
                source_floor.get("resource_physical_floor_ns")
            )
            or not isclose(
                float(source_floor["resource_physical_floor_ns"]),
                float(floor["value_ns"]),
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            or any(
                resource not in capability_by_resource
                or source_demands[resource] != term.get("minimum_demand")
                or capability_by_resource[resource].get(
                    "robust_achievable_rate"
                )
                != term.get("validated_rate_per_second")
                for resource, term in {
                    item["resource"]: item for item in terms
                }.items()
            )
        ):
            return _unknown("resource-physical-floor-source-evidence-mismatch")
    return {
        "status": "known",
        "value_ns": floor["value_ns"],
        "may_be_unattainable": True,
        "evidence_refs": list(floor.get("evidence_refs", [])),
    }


def _anchor_state_history_valid(anchor: object) -> bool:
    if not isinstance(anchor, dict):
        return False
    transitions = anchor.get("state_transitions")
    if not isinstance(transitions, list) or not transitions:
        return False
    states = {
        "observation_validity": "COLLECTED",
        "frontier_role": "NONE",
    }
    allowed = {
        "observation_validity": {
            ("COLLECTED", "QUARANTINED"),
            ("COLLECTED", "QUALIFIED"),
            ("QUARANTINED", "QUALIFIED"),
            ("QUARANTINED", "REJECTED"),
            ("QUALIFIED", "STALE"),
            ("QUALIFIED", "REVOKED"),
            ("STALE", "QUALIFIED"),
            ("STALE", "EXPIRED"),
            ("STALE", "REVOKED"),
        },
        "frontier_role": {
            ("NONE", "PROVISIONAL"),
            ("PROVISIONAL", "NONE"),
            ("PROVISIONAL", "ACTIVE"),
            ("ACTIVE", "SUPERSEDED"),
            ("ACTIVE", "STALE_ROLE"),
            ("ACTIVE", "REVOKED_ROLE"),
            ("STALE_ROLE", "ACTIVE"),
            ("STALE_ROLE", "EXPIRED_ROLE"),
            ("STALE_ROLE", "REVOKED_ROLE"),
        },
    }
    for sequence, transition in enumerate(transitions, start=1):
        if not isinstance(transition, dict):
            return False
        axis = transition.get("axis")
        source = transition.get("from")
        target = transition.get("to")
        evidence_refs = transition.get("evidence_refs")
        if (
            transition.get("sequence") != sequence
            or axis not in states
            or source != states[axis]
            or (source, target) not in allowed[axis]
            or not _nonempty_string(transition.get("reason_code"))
            or not isinstance(evidence_refs, list)
            or not evidence_refs
            or not all(_artifact_uri(ref) for ref in evidence_refs)
        ):
            return False
        states[axis] = target
    return states == {
        "observation_validity": anchor.get("observation_validity"),
        "frontier_role": anchor.get("frontier_role"),
    }


def _eligible_anchor(
    document: dict[str, Any], anchor: dict[str, Any]
) -> bool:
    candidate = document.get("candidate")
    correctness = document.get("correctness")
    environment = document.get("environment")
    policies = document.get("policies")
    measurement_adapter = document.get("measurement_adapter")
    if not all(
        isinstance(value, dict)
        for value in (candidate, correctness, environment, policies)
    ):
        return False
    qualification = _versioned_policy(document, "qualification")
    if qualification is None:
        return False
    qualification_version = qualification.get("version")
    minimum_sessions = qualification.get("minimum_independent_sessions")
    if (
        not isinstance(minimum_sessions, int)
        or isinstance(minimum_sessions, bool)
        or qualification_version not in {"v1", "v2"}
        or minimum_sessions
        < (
            _FRONTIER_MINIMUM_INDEPENDENT_SESSIONS
            if qualification_version == "v2"
            else 2
        )
    ):
        return False
    strict_qualification = qualification_version == "v2"
    candidate_id = candidate.get("candidate_id")
    if not all(
        _nonempty_string(value)
        for value in (
            candidate_id,
            candidate.get("family"),
            candidate.get("coverage"),
            candidate.get("implementation_digest"),
            correctness.get("oracle"),
            correctness.get("policy_ref"),
            correctness.get("evidence_ref"),
            environment.get("preflight_ref"),
            document.get("cohort_id"),
        )
    ) or not _complete_execution_domain(document.get("execution_domain")):
        return False

    holdout = anchor.get("holdout")
    completion = anchor.get("completion_boundary")
    timer = anchor.get("timer")
    warmup = anchor.get("warmup")
    best_of_correct = candidate.get("exact_shape_best_of_correct")
    session_ids = holdout.get("session_ids") if isinstance(holdout, dict) else None
    holdout_evidence_ref = (
        holdout.get("evidence_ref") if isinstance(holdout, dict) else None
    )
    raw_timing = anchor.get("raw_timing_ns")
    eligible_candidate_ids = (
        best_of_correct.get("eligible_candidate_ids")
        if isinstance(best_of_correct, dict)
        else None
    )
    search_session_ids = (
        best_of_correct.get("search_session_ids")
        if isinstance(best_of_correct, dict)
        else None
    )
    search_evidence_ref = (
        best_of_correct.get("evidence_ref")
        if isinstance(best_of_correct, dict)
        else None
    )
    search_timing = (
        _normalize_timing_sessions(
            best_of_correct.get("search_sessions")
            if isinstance(best_of_correct, dict)
            else None,
            expected_lane_id=anchor.get("baseline_lane_id"),
            expected_cohort_id=document.get("cohort_id"),
            minimum_sessions=minimum_sessions,
            require_authored_latency=False,
        )
        if strict_qualification
        else None
    )
    holdout_timing = (
        _normalize_timing_sessions(
            holdout.get("sessions") if isinstance(holdout, dict) else None,
            expected_lane_id=anchor.get("baseline_lane_id"),
            expected_cohort_id=document.get("cohort_id"),
            minimum_sessions=minimum_sessions,
            require_authored_latency=False,
        )
        if strict_qualification
        else None
    )
    independent_search_holdout = (
        isinstance(search_timing, dict)
        and isinstance(holdout_timing, dict)
        and set(search_timing["session_ids"]).isdisjoint(
            holdout_timing["session_ids"]
        )
        and set(search_timing["session_process_ids"].values()).isdisjoint(
            holdout_timing["session_process_ids"].values()
        )
    )
    legacy_qualification_evidence = (
        not strict_qualification
        and isinstance(search_session_ids, list)
        and bool(search_session_ids)
        and all(
            _nonempty_string(session_id)
            for session_id in search_session_ids
        )
        and isinstance(session_ids, list)
        and all(_nonempty_string(session_id) for session_id in session_ids)
        and len(set(session_ids)) >= minimum_sessions
        and set(search_session_ids).isdisjoint(session_ids)
    )
    strict_qualification_evidence = (
        strict_qualification
        and isinstance(raw_timing, list)
        and len(raw_timing) >= minimum_sessions
        and all(
            _finite_number(sample) and sample >= 0
            for sample in raw_timing
        )
        and isinstance(search_session_ids, list)
        and all(
            _nonempty_string(session_id)
            for session_id in search_session_ids
        )
        and isinstance(search_timing, dict)
        and set(search_session_ids) == set(search_timing["session_ids"])
        and len(search_session_ids) == len(set(search_session_ids))
        and isinstance(session_ids, list)
        and all(_nonempty_string(session_id) for session_id in session_ids)
        and isinstance(holdout_timing, dict)
        and set(session_ids) == set(holdout_timing["session_ids"])
        and len(set(session_ids)) >= minimum_sessions
        and independent_search_holdout
        and isinstance(holdout, dict)
        and isinstance(holdout.get("latency_ns"), (int, float))
        and not isinstance(holdout["latency_ns"], bool)
        and holdout["latency_ns"] >= 0
        and isclose(
            float(holdout["latency_ns"]),
            float(holdout_timing["aggregate_latency_ns"]),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        and sorted(float(sample) for sample in raw_timing)
        == sorted(holdout_timing["session_latencies_ns"].values())
    )
    if strict_qualification:
        state_history_compatible = _anchor_state_history_valid(anchor)
    else:
        state_history_compatible = (
            anchor.get("state_transitions") is None
            or _anchor_state_history_valid(anchor)
        )
    cohort_state = _cohort_state(document)
    return (
        anchor.get("observation_validity") == "QUALIFIED"
        and anchor.get("frontier_role") == "ACTIVE"
        and state_history_compatible
        and anchor.get("candidate_id") == candidate_id
        and anchor.get("cohort_id") == document["cohort_id"]
        and anchor.get("execution_domain") == document["execution_domain"]
        and correctness.get("passed") is True
        and anchor.get("correctness_passed") is True
        and environment.get("eligible") is True
        and _nonempty_string(anchor.get("anchor_id"))
        and _nonempty_string(anchor.get("baseline_lane_id"))
        and _nonempty_string(anchor.get("instrumentation_profile"))
        and _completion_boundary_valid(completion)
        and _timer_evidence_valid(timer, completion)
        and isinstance(warmup, dict)
        and warmup.get("converged") is True
        and isinstance(raw_timing, list)
        and bool(raw_timing)
        and all(
            _finite_number(sample) and sample >= 0
            for sample in raw_timing
        )
        and isinstance(best_of_correct, dict)
        and best_of_correct.get("passed") is True
        and best_of_correct.get("winner_candidate_id") == candidate_id
        and isinstance(eligible_candidate_ids, list)
        and candidate_id in eligible_candidate_ids
        and isinstance(search_session_ids, list)
        and bool(search_session_ids)
        and all(
            _nonempty_string(session_id) for session_id in search_session_ids
        )
        and _nonempty_string(search_evidence_ref)
        and isinstance(holdout, dict)
        and holdout.get("passed") is True
        and isinstance(holdout.get("latency_ns"), (int, float))
        and holdout["latency_ns"] >= 0
        and isinstance(session_ids, list)
        and all(
            isinstance(session_id, str) and session_id
            for session_id in session_ids
        )
        and (
            legacy_qualification_evidence
            or strict_qualification_evidence
        )
        and _nonempty_string(holdout_evidence_ref)
        and search_evidence_ref != holdout_evidence_ref
        and _nonempty_string(anchor.get("evidence_ref"))
        and _primary_timer_available(document)
        and isinstance(measurement_adapter, dict)
        and _validated_adapter_operations(document, measurement_adapter)
        is not None
        and cohort_state.get("status") in {"matched", "split"}
    )


def _operator_axis(document: dict[str, Any]) -> dict[str, Any]:
    recorded_anchors = document.get("frontier_anchors", [])
    if not isinstance(recorded_anchors, list):
        recorded_anchors = []
    anchors = [
        anchor
        for anchor in recorded_anchors
        if _eligible_anchor(document, anchor)
    ]
    if not anchors:
        return _unknown("no-qualified-active-exact-shape-anchor")
    selected = min(anchors, key=lambda item: item["holdout"]["latency_ns"])
    return {
        "status": "known",
        "value_ns": selected["holdout"]["latency_ns"],
        "anchor_id": selected["anchor_id"],
        "candidate_id": selected["candidate_id"],
        "evidence_refs": [selected["evidence_ref"]],
    }


def _frontier_anchor_lifecycles(
    document: dict[str, Any],
) -> list[dict[str, Any]]:
    recorded_anchors = document.get("frontier_anchors")
    if not isinstance(recorded_anchors, list):
        return []
    results: list[dict[str, Any]] = []
    qualification = _versioned_policy(document, "qualification")
    legacy_policy = (
        isinstance(qualification, dict)
        and qualification.get("version") == "v1"
    )
    for anchor in recorded_anchors:
        if not isinstance(anchor, dict) or not _nonempty_string(
            anchor.get("anchor_id")
        ):
            continue
        history_valid = _anchor_state_history_valid(anchor)
        legacy_replay = legacy_policy and anchor.get("state_transitions") is None
        qualified_active = _eligible_anchor(document, anchor)
        if history_valid:
            history_status = "replayable"
            history_reason = None
        elif legacy_replay:
            history_status = "legacy-replay"
            history_reason = "qualification-policy-v1"
        else:
            history_status = "invalid"
            history_reason = "invalid-anchor-state-history"
        results.append(
            {
                "anchor_id": anchor["anchor_id"],
                "candidate_id": anchor.get("candidate_id"),
                "cohort_id": anchor.get("cohort_id"),
                "execution_domain": anchor.get("execution_domain"),
                "state": {
                    "observation_validity": anchor.get(
                        "observation_validity"
                    ),
                    "frontier_role": anchor.get("frontier_role"),
                },
                "authoritative_surface_knot": qualified_active,
                "history_status": history_status,
                "history_reason_code": history_reason,
                "transitions": list(anchor.get("state_transitions", []))
                if isinstance(anchor.get("state_transitions"), list)
                else [],
            }
        )
    return results


def _schedule_axis(
    document: dict[str, Any], operator: dict[str, Any]
) -> dict[str, Any]:
    schedule = document.get("single_node_schedule")
    if not isinstance(schedule, dict):
        return _unknown("missing-single-node-schedule-evidence")
    if operator["status"] != "known":
        return _unknown("operator-frontier-unknown")
    if _versioned_policy(document, "schedule") is None:
        return _unknown("invalid-schedule-policy")
    if schedule.get("candidate_id") != operator["candidate_id"]:
        return _unknown("schedule-candidate-mismatch")
    if (
        not _nonempty_string(schedule.get("schedule_id"))
        or not _nonempty_string(schedule.get("version"))
        or not schedule.get("evidence_refs")
        or not all(
            _nonempty_string(reference)
            for reference in schedule.get("evidence_refs", [])
        )
    ):
        return _unknown("incomplete-single-node-schedule-evidence")
    if (
        schedule.get("dependencies") != []
        or schedule.get("transformations") != []
        or schedule.get("overlap_claims") != []
    ):
        return _unknown("unsupported-non-single-node-schedule")
    return {
        "status": "known",
        "value_ns": operator["value_ns"],
        "schedule_id": schedule["schedule_id"],
        "operator_frontier_ref": operator["anchor_id"],
        "evidence_refs": list(schedule.get("evidence_refs", [])),
    }


def _observation_axis(document: dict[str, Any]) -> dict[str, Any]:
    lane = document.get("baseline_timing_lane")
    if not isinstance(lane, dict):
        return _unknown("missing-baseline-timing-lane")
    if not _complete_execution_domain(document.get("execution_domain")):
        return _unknown("incomplete-execution-domain")
    samples = lane.get("raw_samples_ns")
    completion = lane.get("completion_boundary")
    timer = lane.get("timer")
    warmup = lane.get("warmup")
    correctness = document.get("correctness")
    environment = document.get("environment")
    if (
        _versioned_policy(document, "observation") is None
        or lane.get("observation_validity") not in {"COLLECTED", "QUALIFIED"}
        or lane.get("frontier_role") != "NONE"
        or not _nonempty_string(lane.get("lane_id"))
        or not _nonempty_string(lane.get("instrumentation_profile"))
        or not isinstance(samples, list)
        or not samples
        or not all(
            _finite_number(sample) and sample >= 0
            for sample in samples
        )
        or not _completion_boundary_valid(completion)
        or not _timer_evidence_valid(timer, completion)
        or not isinstance(warmup, dict)
        or warmup.get("converged") is not True
        or not isinstance(correctness, dict)
        or correctness.get("passed") is not True
        or not isinstance(environment, dict)
        or environment.get("eligible") is not True
        or not _nonempty_string(lane.get("evidence_ref"))
    ):
        return _unknown("invalid-baseline-timing-lane")
    return {
        "status": "known",
        "value_ns": median(samples),
        "observation_validity": lane["observation_validity"],
        "frontier_role": lane["frontier_role"],
        "lane_id": lane["lane_id"],
        "evidence_refs": [lane["evidence_ref"]],
    }


def _comparisons(axes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    observation = axes["observation"]
    floor = axes["resource_physical_floor"]
    frontier = axes["operator_achievable_frontier"]
    floor_distance = (
        observation["value_ns"] - floor["value_ns"]
        if observation["status"] == "known" and floor["status"] == "known"
        else None
    )
    frontier_distance = (
        observation["value_ns"] - frontier["value_ns"]
        if observation["status"] == "known" and frontier["status"] == "known"
        else None
    )
    return {
        "physical_floor_to_observation": {
            "distance_ns": floor_distance,
            "prediction_error_ns": None,
            "error_status": "not-evaluable-physical-floor",
        },
        "operator_frontier_to_observation": {
            "distance_ns": frontier_distance,
            "prediction_error_ns": frontier_distance,
            "error_status": (
                "evaluated"
                if frontier_distance is not None
                else "not-evaluable-unknown-axis"
            ),
        },
    }


def _remote_execution_records(
    evidence_ref: object,
    verified_artifacts: dict[str, object],
) -> dict[str, dict[str, Any]] | None:
    if not _artifact_uri(evidence_ref):
        return None
    source = _verified_artifact_content(
        verified_artifacts,
        evidence_ref,
        role="source-remote-execution",
        schema="groundupscale.dev/remote-execution/v1alpha1",
    )
    sessions = source.get("sessions") if isinstance(source, dict) else None
    if not isinstance(sessions, list) or not sessions:
        return None
    records: dict[str, dict[str, Any]] = {}
    for record in sessions:
        if (
            not isinstance(record, dict)
            or not _known_identity_string(record.get("session_id"))
            or not isinstance(record.get("process_id"), int)
            or isinstance(record["process_id"], bool)
            or record["process_id"] <= 0
            or not _known_identity_string(record.get("started_at"))
            or not isinstance(record.get("sha256"), str)
            or fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
            or record["session_id"] in records
        ):
            return None
        records[record["session_id"]] = record
    return records


def _source_session_matches_remote(
    ref: str,
    source: dict[str, Any],
    remote_records: dict[str, dict[str, Any]],
    verified_artifacts: dict[str, object],
) -> bool:
    artifact = verified_artifacts.get(ref)
    manifest = artifact.get("manifest") if isinstance(artifact, dict) else None
    record = remote_records.get(source.get("session_id"))
    return bool(
        isinstance(manifest, dict)
        and isinstance(record, dict)
        and manifest.get("sha256") == record.get("sha256")
        and source.get("process_id") == record.get("process_id")
        and source.get("process_started_at") == record.get("started_at")
    )


def _semantic_path_inputs_valid(
    source: dict[str, Any],
) -> bool:
    path_inputs = source.get("path_inputs")
    main_input = source.get("input")
    path_correctness = source.get("path_correctness")
    if not isinstance(path_inputs, dict) or set(path_inputs) != {"q", "k", "v"}:
        return False
    identities: list[tuple[str, str]] = []
    for identity in path_inputs.values():
        if (
            not isinstance(identity, dict)
            or set(identity) != {"left_sha256", "right_sha256"}
            or any(
                not isinstance(value, str)
                or fullmatch(r"[0-9a-f]{64}", value) is None
                for value in identity.values()
            )
        ):
            return False
        identities.append(
            (identity["left_sha256"], identity["right_sha256"])
        )
    expected_hashes = (
        {
            record.get("expected_sha256")
            for record in path_correctness.values()
            if isinstance(record, dict)
        }
        if isinstance(path_correctness, dict)
        and set(path_correctness) == {"q", "k", "v"}
        else set()
    )
    return bool(
        len(set(identities)) == 3
        and isinstance(main_input, dict)
        and main_input.get("left_sha256")
        == path_inputs["q"]["left_sha256"]
        and main_input.get("right_sha256")
        == path_inputs["q"]["right_sha256"]
        and len(expected_hashes) == 3
        and all(
            isinstance(value, str)
            and fullmatch(r"[0-9a-f]{64}", value) is not None
            for value in expected_hashes
        )
    )


def _trigger_observation_basis_valid(
    item: dict[str, Any],
    verified_artifacts: dict[str, object],
) -> bool:
    basis = item.get("observation_basis")
    if not isinstance(basis, dict):
        return False
    stable_path = item.get("stable_path")
    if (
        basis.get("stable_path") != stable_path
        or not _resolved_identity_string(basis.get("semantic"))
    ):
        return False

    if basis.get("kind") == "benchmark-case":
        source_ref = basis.get("source_evidence_ref")
        if not _artifact_uri(source_ref):
            return False
        source = _verified_artifact_content(
            verified_artifacts,
            source_ref,
            role="source-transformer-benchmark",
            schema="groundupscale.dev/benchmark-observation/v1alpha1",
        )
        cases = source.get("cases") if isinstance(source, dict) else None
        source_case = next(
            (
                case
                for case in cases
                if isinstance(case, dict)
                and case.get("case_id") == basis.get("source_case_id")
            ),
            None,
        ) if isinstance(cases, list) else None
        latency = (
            source_case.get("latency")
            if isinstance(source_case, dict)
            else None
        )
        authored_scope = (
            source_case.get("authored_scope")
            if isinstance(source_case, dict)
            else None
        )
        canonical_scope = (
            f"semantic/{authored_scope}"
            .replace("layer_", "layer-")
            .replace("q_proj", "q-proj")
            .replace("k_proj", "k-proj")
            .replace("v_proj", "v-proj")
            if isinstance(authored_scope, str)
            else None
        )
        return bool(
            isinstance(latency, dict)
            and _finite_number(latency.get("median_ns"))
            and latency["median_ns"] == item.get("observed_ns")
            and canonical_scope == stable_path
            and basis.get("semantic") == "batch-one Q projection MatMul"
        )

    if basis.get("kind") != "session-variant-aggregate":
        return False
    variant = basis.get("variant")
    input_refs = basis.get("input_refs")
    remote_records = _remote_execution_records(
        basis.get("execution_evidence_ref"), verified_artifacts
    )
    if (
        not _canonical_identifier(variant)
        or basis.get("lane") != "baseline"
        or basis.get("reducer")
        != "median-of-independent-session-medians"
        or not isinstance(input_refs, list)
        or len(input_refs) < 3
        or not all(_artifact_uri(ref) for ref in input_refs)
        or len(set(input_refs)) != len(input_refs)
        or remote_records is None
    ):
        return False

    session_medians: list[float] = []
    process_ids: set[int] = set()
    cohort_ids: set[str] = set()
    for ref in input_refs:
        source = _verified_artifact_content(
            verified_artifacts,
            ref,
            role="source-diagnostic-session",
            schema="groundupscale.dev/ascend-diagnostic-session/v1alpha1",
        )
        if not isinstance(source, dict):
            return False
        contract = source.get("execution_contract")
        variant_contracts = (
            contract.get("variant_contracts")
            if isinstance(contract, dict)
            else None
        )
        variant_contract = (
            variant_contracts.get(variant)
            if isinstance(variant_contracts, dict)
            else None
        )
        variants = source.get("variants")
        measurement = (
            variants.get(variant) if isinstance(variants, dict) else None
        )
        samples = (
            measurement.get("raw_samples_ns")
            if isinstance(measurement, dict)
            else None
        )
        measurement_semantic = (
            measurement.get("semantic")
            if isinstance(measurement, dict)
            else None
        )
        process_id = source.get("process_id")
        cohort_id = source.get("cohort_id")
        path_key = {"k_baseline": "k", "v_baseline": "v"}.get(variant)
        path_inputs = source.get("path_inputs")
        input_identity = (
            path_inputs.get(path_key)
            if isinstance(path_inputs, dict) and path_key is not None
            else None
        )
        if (
            not isinstance(variant_contract, dict)
            or variant_contract.get("semantic") != basis["semantic"]
            or variant_contract.get("stable_path") != stable_path
            or variant_contract.get("lane") != "baseline"
            or variant_contract.get("input_identity") != input_identity
            or not _semantic_path_inputs_valid(source)
            or measurement_semantic != f"{basis['semantic']} baseline"
            or not isinstance(samples, list)
            or not samples
            or not all(
                _finite_number(sample) and float(sample) >= 0
                for sample in samples
            )
            or not _known_identity_string(source.get("session_id"))
            or not isinstance(process_id, int)
            or isinstance(process_id, bool)
            or process_id <= 0
            or not _known_identity_string(cohort_id)
            or not _source_session_matches_remote(
                ref, source, remote_records, verified_artifacts
            )
        ):
            return False
        process_ids.add(process_id)
        cohort_ids.add(cohort_id)
        session_medians.append(float(median(samples)))
    return bool(
        len(process_ids) == len(input_refs)
        and len(cohort_ids) == 1
        and float(median(session_medians)) == item.get("observed_ns")
    )


def _diagnostic_trigger(
    document: dict[str, Any],
    verified_artifacts: dict[str, object] | None = None,
) -> dict[str, Any] | None:
    trigger_input = document.get("diagnostic_trigger_input")
    if not isinstance(trigger_input, dict):
        return None
    policy = trigger_input.get("policy")
    e2e_observation_ns = trigger_input.get("e2e_observation_ns")
    items = trigger_input.get("items")
    if (
        not isinstance(policy, dict)
        or not all(
            _resolved_identity_string(policy.get(key))
            for key in (
                "policy_id",
                "version",
                "scope",
                "change_reason",
                "revalidation",
            )
        )
        or not _exact_version_text(policy.get("version"))
        or not _finite_number(e2e_observation_ns)
        or float(e2e_observation_ns) <= 0
        or not isinstance(items, list)
    ):
        return _unknown("invalid-diagnostic-trigger-input")

    e2e_source_ref = trigger_input.get("source_evidence_ref")
    baseline_source_ref = trigger_input.get(
        "baseline_observation_evidence_ref"
    )
    source_evidence_required = trigger_input.get(
        "source_evidence_required", False
    )
    has_item_basis = any(
        isinstance(item, dict)
        and isinstance(item.get("observation_basis"), dict)
        for item in items
    )
    bundle_has_trigger_sources = bool(
        isinstance(verified_artifacts, dict)
        and any(
            isinstance(artifact, dict)
            and isinstance(artifact.get("manifest"), dict)
            and artifact["manifest"].get("role")
            in {
                "source-transformer-e2e-attribution",
                "source-remote-execution",
            }
            for artifact in verified_artifacts.values()
        )
    )
    if not isinstance(source_evidence_required, bool):
        return _unknown("invalid-diagnostic-trigger-input")
    if (
        source_evidence_required
        or has_item_basis
        or bundle_has_trigger_sources
        or e2e_source_ref is not None
        or baseline_source_ref is not None
    ):
        if (
            verified_artifacts is None
            or not _artifact_uri(e2e_source_ref)
            or (
                baseline_source_ref is not None
                and not _artifact_uri(baseline_source_ref)
            )
        ):
            return _unknown("diagnostic-trigger-source-evidence-invalid")
        e2e_source = _verified_artifact_content(
            verified_artifacts,
            e2e_source_ref,
            role="source-transformer-e2e-attribution",
            schema="groundupscale.dev/error-attribution/v1alpha1",
        )
        baseline_source = (
            _verified_artifact_content(
                verified_artifacts,
                baseline_source_ref,
                role="source-transformer-benchmark",
                schema=(
                    "groundupscale.dev/benchmark-observation/v1alpha1"
                ),
            )
            if baseline_source_ref is not None
            else None
        )
        baseline_cases = (
            baseline_source.get("cases")
            if isinstance(baseline_source, dict)
            else None
        )
        q_case = next(
            (
                case
                for case in baseline_cases
                if isinstance(case, dict)
                and case.get("case_id") == "matmul-q-proj"
            ),
            None,
        ) if isinstance(baseline_cases, list) else None
        q_latency = q_case.get("latency") if isinstance(q_case, dict) else None
        if (
            not isinstance(e2e_source, dict)
            or e2e_source.get("e2e_trace_host_ns") != e2e_observation_ns
            or (
                baseline_source_ref is not None
                and (
                    not isinstance(q_latency, dict)
                    or not _finite_number(q_latency.get("median_ns"))
                    or any(
                        not isinstance(item, dict)
                        or item.get("observed_ns") != q_latency["median_ns"]
                        for item in items
                    )
                )
            )
        ):
            return _unknown("diagnostic-trigger-source-evidence-mismatch")
        if any(
            not isinstance(item, dict)
            or not _trigger_observation_basis_valid(
                item, verified_artifacts
            )
            for item in items
        ):
            return _unknown("diagnostic-trigger-source-evidence-mismatch")

    normalized: list[dict[str, Any]] = []
    paths: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            return _unknown("invalid-diagnostic-trigger-item")
        stable_path = item.get("stable_path")
        values = (
            item.get("predicted_ns"),
            item.get("observed_ns"),
            item.get("combined_uncertainty_ns"),
        )
        if (
            not _canonical_stable_path(stable_path)
            or stable_path in paths
            or not all(_finite_number(value) for value in values)
            or any(float(value) < 0 for value in values)
        ):
            return _unknown("invalid-diagnostic-trigger-item")
        paths.add(stable_path)
        normalized.append(
            {
                "stable_path": stable_path,
                "predicted_ns": item["predicted_ns"],
                "observed_ns": item["observed_ns"],
                "combined_uncertainty_ns": item[
                    "combined_uncertainty_ns"
                ],
                **(
                    {"observation_basis": deepcopy(item["observation_basis"])}
                    if isinstance(item.get("observation_basis"), dict)
                    else {}
                ),
            }
        )

    def ranked(metric: str) -> list[dict[str, Any]]:
        return [
            {
                "stable_path": item["stable_path"],
                "value_ns": item[metric],
                "rank": rank,
            }
            for rank, item in enumerate(
                sorted(
                    normalized,
                    key=lambda item: (
                        -float(item[metric]),
                        item["stable_path"],
                    ),
                )[:10],
                start=1,
            )
        ]

    predicted_top10 = ranked("predicted_ns")
    observed_top10 = ranked("observed_ns")
    predicted_paths = {item["stable_path"] for item in predicted_top10}
    observed_paths = {item["stable_path"] for item in observed_top10}
    union_paths = predicted_paths | observed_paths
    top10_union = [
        {
            "stable_path": item["stable_path"],
            "predicted_top10": item["stable_path"] in predicted_paths,
            "observed_top10": item["stable_path"] in observed_paths,
        }
        for item in normalized
        if item["stable_path"] in union_paths
    ]
    e2e_tenth_ns = float(e2e_observation_ns) / 10.0
    evaluated = []
    for item in normalized:
        gap_ns = abs(
            float(item["observed_ns"]) - float(item["predicted_ns"])
        )
        uncertainty_exceeded = gap_ns > float(
            item["combined_uncertainty_ns"]
        )
        materiality = {
            "predicted_top10": item["stable_path"] in predicted_paths,
            "observed_top10": item["stable_path"] in observed_paths,
            "gap_exceeds_e2e_tenth": gap_ns > e2e_tenth_ns,
        }
        triggered = uncertainty_exceeded and any(materiality.values())
        reason_code = "triggered"
        if not uncertainty_exceeded:
            reason_code = "gap-within-combined-uncertainty"
        elif not any(materiality.values()):
            reason_code = "gap-not-material"
        evaluated.append(
            {
                **item,
                "absolute_gap_ns": gap_ns,
                "uncertainty_exceeded": uncertainty_exceeded,
                "materiality": materiality,
                "triggered": triggered,
                "reason_code": reason_code,
            }
        )
    return {
        "status": "evaluated",
        "policy": {
            key: policy[key]
            for key in (
                "policy_id",
                "version",
                "scope",
                "change_reason",
                "revalidation",
            )
        },
        "e2e_observation_ns": e2e_observation_ns,
        "e2e_tenth_ns": e2e_tenth_ns,
        "predicted_top10": predicted_top10,
        "observed_top10": observed_top10,
        "top10_union": top10_union,
        "evaluated": evaluated,
        "triggered": [item for item in evaluated if item["triggered"]],
    }


def _candidate_source_replay_valid(
    candidate: dict[str, Any],
    *,
    sessions: list[dict[str, Any]],
    stable_path: str,
    semantic: str,
    evidence_lane: str,
    direct_failure: dict[str, Any] | None,
    verified_artifacts: dict[str, object],
) -> bool:
    replay = candidate.get("source_replay")
    if replay is None:
        return evidence_lane == "baseline"
    if not isinstance(replay, dict):
        return False
    variant = replay.get("variant")
    refs = replay.get("input_refs")
    remote_records = _remote_execution_records(
        replay.get("execution_evidence_ref"), verified_artifacts
    )
    if (
        variant
        != ("negative_control" if evidence_lane == "diagnostic" else "k_baseline")
        or not isinstance(refs, list)
        or len(refs) < 3
        or len(refs) != len(sessions)
        or not all(_artifact_uri(ref) for ref in refs)
        or len(set(refs)) != len(refs)
        or remote_records is None
    ):
        return False
    sources: dict[str, tuple[str, dict[str, Any]]] = {}
    for ref in refs:
        source = _verified_artifact_content(
            verified_artifacts,
            ref,
            role="source-diagnostic-session",
            schema="groundupscale.dev/ascend-diagnostic-session/v1alpha1",
        )
        if (
            not isinstance(source, dict)
            or not _known_identity_string(source.get("session_id"))
            or source["session_id"] in sources
            or not _source_session_matches_remote(
                ref, source, remote_records, verified_artifacts
            )
        ):
            return False
        sources[source["session_id"]] = (ref, source)
    if {session.get("session_id") for session in sessions} != set(sources):
        return False
    for session in sessions:
        source = sources[session["session_id"]][1]
        contract = source.get("execution_contract")
        variant_contracts = (
            contract.get("variant_contracts")
            if isinstance(contract, dict)
            else None
        )
        variant_contract = (
            variant_contracts.get(variant)
            if isinstance(variant_contracts, dict)
            else None
        )
        variants = source.get("variants")
        measurement = (
            variants.get(variant) if isinstance(variants, dict) else None
        )
        source_samples = (
            measurement.get("raw_samples_ns")
            if isinstance(measurement, dict)
            else None
        )
        measurement_semantic = (
            measurement.get("semantic")
            if isinstance(measurement, dict)
            else None
        )
        path_key = "k" if variant == "k_baseline" else "v"
        path_inputs = source.get("path_inputs")
        input_identity = (
            path_inputs.get(path_key)
            if isinstance(path_inputs, dict)
            else None
        )
        if (
            not isinstance(variant_contract, dict)
            or variant_contract.get("semantic") != semantic
            or variant_contract.get("stable_path") != stable_path
            or variant_contract.get("lane") != evidence_lane
            or variant_contract.get("input_identity") != input_identity
            or not _semantic_path_inputs_valid(source)
            or measurement_semantic
            != (
                semantic
                if variant == "negative_control"
                else f"{semantic} baseline"
            )
            or not isinstance(source_samples, list)
            or not source_samples
            or not all(
                _finite_number(sample) and float(sample) >= 0
                for sample in source_samples
            )
            or source_samples != session.get("raw_samples_ns")
            or source.get("process_id") != session.get("process_id")
        ):
            return False
        if evidence_lane == "diagnostic":
            negative_control = source.get("negative_control")
            negative_correctness = (
                negative_control.get("correctness")
                if isinstance(negative_control, dict)
                else None
            )
            if (
                not isinstance(direct_failure, dict)
                or not isinstance(negative_correctness, dict)
                or negative_correctness.get("passed") is not False
                or negative_correctness.get("expected_sha256")
                != direct_failure.get("expected_sha256")
                or negative_correctness.get("observed_sha256")
                != direct_failure.get("observed_sha256")
                or negative_correctness.get("max_abs_difference")
                != direct_failure.get("max_abs_difference")
                or negative_correctness.get("mismatched_elements")
                != direct_failure.get("mismatched_elements")
                or not isinstance(
                    negative_correctness.get("expected_sha256"), str
                )
                or negative_correctness["expected_sha256"]
                == negative_correctness.get("observed_sha256")
                or not _finite_number(
                    negative_correctness.get("max_abs_difference")
                )
                or float(negative_correctness["max_abs_difference"]) <= 0
                or not isinstance(
                    negative_correctness.get("mismatched_elements"), int
                )
                or negative_correctness["mismatched_elements"] <= 0
            ):
                return False
    return True


def _locked_probe_contract_valid(
    contract: object,
    *,
    cohort_id: object,
    hardware: object,
    execution_domain: object,
) -> bool:
    if not isinstance(contract, dict):
        return False
    shape = contract.get("shape")
    strides = contract.get("strides")
    candidate_ids = contract.get("candidate_ids")
    environment = contract.get("environment")
    correctness_policy = contract.get("correctness_policy")
    cohort_identity = contract.get("cohort_identity")
    hardware_value = hardware if isinstance(hardware, dict) else {}
    domain_value = (
        execution_domain if isinstance(execution_domain, dict) else {}
    )
    numeric_execution = (
        cohort_identity.get("numeric_execution")
        if isinstance(cohort_identity, dict)
        else None
    )
    execution_context = (
        cohort_identity.get("execution_context")
        if isinstance(cohort_identity, dict)
        else None
    )
    timer_protocol = (
        cohort_identity.get("timer_protocol")
        if isinstance(cohort_identity, dict)
        else None
    )
    communication = (
        cohort_identity.get("communication")
        if isinstance(cohort_identity, dict)
        else None
    )
    return (
        all(
            _resolved_identity_string(contract.get(key))
            for key in ("semantic", "dtype", "layout", "cohort_id")
        )
        and contract.get("cohort_id") == cohort_id
        and contract.get("execution_domain") == domain_value
        and all(
            contract.get(key) == domain_value.get(key)
            for key in (
                "dtype",
                "layout",
                "alignment_bytes",
                "threads",
            )
        )
        and isinstance(environment, dict)
        and environment.get("eligible") is True
        and _nonempty_string(environment.get("evidence_ref"))
        and isinstance(correctness_policy, dict)
        and all(
            _resolved_identity_string(correctness_policy.get(key))
            for key in (
                "policy_id",
                "version",
                "scope",
                "change_reason",
                "revalidation",
                "oracle",
            )
        )
        and _exact_version_text(correctness_policy.get("version"))
        and _finite_number(correctness_policy.get("atol"))
        and _finite_number(correctness_policy.get("rtol"))
        and float(correctness_policy["atol"]) >= 0
        and float(correctness_policy["rtol"]) >= 0
        and isinstance(cohort_identity, dict)
        and all(
            cohort_identity.get(key) == hardware_value.get(key)
            and _resolved_identity_string(cohort_identity.get(key))
            for key in (
                "device",
                "partition",
                "topology",
                "software",
            )
        )
        and all(
            cohort_identity.get(key) == hardware_value.get(key)
            and _exact_versioned_identity(cohort_identity.get(key))
            for key in (
                "os",
                "kernel",
                "driver",
                "runtime",
                "framework",
                "compiler",
                "operator_library",
            )
        )
        and all(
            cohort_identity.get(key) == hardware_value.get(key)
            and _exact_versioned_identity(
                cohort_identity.get(key), allow_not_applicable=True
            )
            for key in ("firmware", "communication_library")
        )
        and isinstance(cohort_identity.get("power_clock"), dict)
        and bool(cohort_identity["power_clock"])
        and cohort_identity["power_clock"] == hardware_value.get("power_clock")
        and all(
            _resolved_identity_string(cohort_identity["power_clock"].get(key))
            for key in ("power_policy", "clock_policy")
        )
        and isinstance(numeric_execution, dict)
        and _resolved_identity_string(numeric_execution.get("dtype"))
        and _resolved_identity_string(numeric_execution.get("layout"))
        and numeric_execution.get("dtype") == contract.get("dtype")
        and numeric_execution.get("layout") == contract.get("layout")
        and numeric_execution.get("alignment_bytes")
        == contract.get("alignment_bytes")
        and numeric_execution.get("threads") == contract.get("threads")
        and _resolved_identity_string(
            numeric_execution.get("execution_mode")
        )
        and numeric_execution.get("execution_mode")
        == domain_value.get("execution_mode")
        and isinstance(execution_context, dict)
        and all(
            _resolved_identity_string(execution_context.get(key))
            for key in ("affinity", "numa", "context")
        )
        and _known_identity_string(execution_context.get("stream"))
        and isinstance(execution_context.get("concurrency"), int)
        and not isinstance(execution_context["concurrency"], bool)
        and execution_context["concurrency"] > 0
        and all(
            execution_context.get(key) == domain_value.get(key)
            for key in (
                "affinity",
                "numa",
                "context",
                "stream",
                "concurrency",
            )
        )
        and isinstance(timer_protocol, dict)
        and _resolved_identity_string(timer_protocol.get("source"))
        and isinstance(timer_protocol.get("resolution_ns"), int)
        and not isinstance(timer_protocol["resolution_ns"], bool)
        and timer_protocol["resolution_ns"] > 0
        and timer_protocol.get("completion_kind")
        == (
            contract.get("completion_boundary", {}).get("kind")
            if isinstance(contract.get("completion_boundary"), dict)
            else None
        )
        and all(
            _resolved_identity_string(timer_protocol.get(key))
            for key in (
                "adapter_id",
                "adapter_version",
                "protocol_id",
                "protocol_version",
            )
        )
        and _exact_version_text(timer_protocol.get("adapter_version"))
        and _exact_version_text(timer_protocol.get("protocol_version"))
        and isinstance(communication, dict)
        and communication.get("status") in {"available", "not_applicable"}
        and communication.get("status")
        == (
            "not_applicable"
            if cohort_identity["communication_library"]["status"]
            == "not_applicable"
            else "available"
        )
        and _nonempty_string(cohort_identity.get("evidence_ref"))
        and isinstance(shape, dict)
        and bool(shape)
        and all(
            _nonempty_string(name)
            and isinstance(dimensions, list)
            and bool(dimensions)
            and all(
                isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension > 0
                for dimension in dimensions
            )
            for name, dimensions in shape.items()
        )
        and isinstance(strides, dict)
        and set(strides) == set(shape)
        and all(
            isinstance(values, list)
            and len(values) == len(shape[name])
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for value in values
            )
            for name, values in strides.items()
        )
        and isinstance(contract.get("alignment_bytes"), int)
        and not isinstance(contract["alignment_bytes"], bool)
        and contract["alignment_bytes"] > 0
        and isinstance(contract.get("threads"), int)
        and not isinstance(contract["threads"], bool)
        and contract["threads"] > 0
        and isinstance(candidate_ids, list)
        and bool(candidate_ids)
        and all(_nonempty_string(candidate_id) for candidate_id in candidate_ids)
        and len(candidate_ids) == len(set(candidate_ids))
        and _completion_boundary_valid(contract.get("completion_boundary"))
    )


def _measurement_lanes_valid(
    value: object,
    *,
    stable_path: object,
    contract: object,
    integration_verdict_requested: bool = False,
) -> bool:
    if not isinstance(value, dict):
        return False
    if not isinstance(contract, dict):
        return False
    baseline = value.get("baseline")
    diagnostic = value.get("diagnostic")
    expected_case = {
        "stable_path": stable_path,
        "semantic": contract.get("semantic"),
    }
    return (
        isinstance(baseline, dict)
        and isinstance(diagnostic, dict)
        and all(
            _nonempty_string(baseline.get(key))
            for key in (
                "lane_id",
                "pair_id",
                "instrumentation_profile",
                "timer_source",
                "evidence_ref",
            )
        )
        and all(
            _nonempty_string(diagnostic.get(key))
            for key in (
                "lane_id",
                "pair_id",
                "paired_baseline_lane_id",
                "instrumentation_profile",
                "evidence_ref",
            )
        )
        and diagnostic["pair_id"] == baseline["pair_id"]
        and diagnostic["paired_baseline_lane_id"] == baseline["lane_id"]
        and diagnostic["lane_id"] != baseline["lane_id"]
        and (
            (
                diagnostic.get("timing_used_for_frontier") is False
                and diagnostic.get("timing_used_for_integration_verdict")
                is True
                and "timing_used_for_verdict" not in diagnostic
            )
            if integration_verdict_requested
            else diagnostic.get("timing_used_for_verdict") is False
        )
        and baseline.get("case") == expected_case
        and diagnostic.get("case") == expected_case
        and baseline.get("execution_domain")
        == contract.get("execution_domain")
        and diagnostic.get("execution_domain")
        == contract.get("execution_domain")
        and baseline.get("candidate_ids") == contract.get("candidate_ids")
        and diagnostic.get("candidate_ids") == contract.get("candidate_ids")
        and baseline.get("cohort_id") == contract.get("cohort_id")
        and diagnostic.get("cohort_id") == contract.get("cohort_id")
        and baseline.get("completion_boundary")
        == contract.get("completion_boundary")
        and diagnostic.get("completion_boundary")
        == contract.get("completion_boundary")
        and baseline.get("timer_source")
        == contract.get("cohort_identity", {})
        .get("timer_protocol", {})
        .get("source")
        and diagnostic.get("timer_source") == baseline.get("timer_source")
    )


def _artifact_refs(value: object) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            refs.update(_artifact_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(_artifact_refs(item))
    elif isinstance(value, str) and value.startswith("artifact://"):
        refs.add(value)
    return refs


def _artifact_uri(value: object) -> bool:
    if (
        not _nonempty_string(value)
        or fullmatch(r"artifact://[A-Za-z0-9][A-Za-z0-9._/-]*", value)
        is None
    ):
        return False
    return all(
        segment not in {"", ".", ".."}
        for segment in value.removeprefix("artifact://").split("/")
    )


def _probe_references_valid(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("_ref"):
                if not _artifact_uri(item):
                    return False
            elif key.endswith("_refs"):
                if not isinstance(item, list) or not all(
                    _artifact_uri(ref) for ref in item
                ):
                    return False
            elif not _probe_references_valid(item):
                return False
    elif isinstance(value, list):
        return all(_probe_references_valid(item) for item in value)
    return True


def _verified_bundle_artifacts(
    root: Path, manifest: dict[str, Any]
) -> dict[str, object]:
    by_uri: dict[str, list[dict[str, Any]]] = {}
    for artifact in manifest.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        uri = artifact.get("uri")
        if _artifact_uri(uri):
            by_uri.setdefault(uri, []).append(artifact)
    verified: dict[str, object] = {}
    for uri, artifacts in by_uri.items():
        if len(artifacts) != 1:
            continue
        artifact = artifacts[0]
        artifact_path = (root / artifact["path"]).resolve()
        if root not in artifact_path.parents:
            continue
        content: object = None
        if artifact.get("media_type") == "application/json":
            try:
                content = json.loads(artifact_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                content = None
        verified[uri] = {"manifest": artifact, "content": content}
    return verified


def _uncertainty_policy_structure_valid(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    target = value.get("target_coverage")
    calibration = value.get("calibration")
    calibration_records = (
        calibration.get("records")
        if isinstance(calibration, dict)
        else None
    )
    estimator = (
        calibration.get("estimator")
        if isinstance(calibration, dict)
        else None
    )
    calibration_target_ids = (
        target.get("required_calibration_target_ids")
        if isinstance(target, dict)
        else None
    )
    validation_target_ids = (
        target.get("required_validation_target_ids")
        if isinstance(target, dict)
        else None
    )
    return (
        all(
            _resolved_identity_string(value.get(key))
            for key in (
                "policy_id",
                "version",
                "scope",
                "change_reason",
                "revalidation",
            )
        )
        and _exact_version_text(value.get("version"))
        and value.get("combination_rule") == "root-sum-square"
        and isinstance(target, dict)
        and _canonical_stable_path(target.get("stable_path"))
        and isinstance(target.get("surface"), dict)
        and all(
            _nonempty_string(target["surface"].get(key))
            for key in ("surface_id", "version")
        )
        and isinstance(target.get("execution_domain_sha256"), str)
        and fullmatch(r"[0-9a-f]{64}", target["execution_domain_sha256"])
        is not None
        and _nonempty_string(target.get("cohort_id"))
        and target.get("coverage_method")
        == "independent-validation-absolute-residual-within-limit"
        and _finite_number(target.get("minimum_fraction"))
        and 0 < float(target["minimum_fraction"]) <= 1
        and isinstance(target.get("minimum_calibration_records"), int)
        and not isinstance(target["minimum_calibration_records"], bool)
        and target["minimum_calibration_records"] >= 3
        and isinstance(target.get("minimum_validation_records"), int)
        and not isinstance(target["minimum_validation_records"], bool)
        and target["minimum_validation_records"] >= 3
        and isinstance(calibration_target_ids, list)
        and bool(calibration_target_ids)
        and all(
            _canonical_identifier(target_id)
            for target_id in calibration_target_ids
        )
        and len(calibration_target_ids) == len(set(calibration_target_ids))
        and isinstance(validation_target_ids, list)
        and bool(validation_target_ids)
        and all(
            _canonical_identifier(target_id)
            for target_id in validation_target_ids
        )
        and len(validation_target_ids) == len(set(validation_target_ids))
        and set(calibration_target_ids).isdisjoint(validation_target_ids)
        and _artifact_uri(target.get("evidence_ref"))
        and isinstance(calibration, dict)
        and isinstance(estimator, dict)
        and all(
            _resolved_identity_string(estimator.get(key))
            for key in (
                "policy_id",
                "version",
                "scope",
                "change_reason",
                "revalidation",
            )
        )
        and _exact_version_text(estimator.get("version"))
        and estimator.get("method") == "max-absolute-residual"
        and _artifact_uri(calibration.get("evidence_ref"))
        and isinstance(calibration_records, list)
        and bool(calibration_records)
        and all(
            isinstance(record, dict)
            and _canonical_identifier(record.get("target_id"))
            and record.get("partition") in {"calibration", "validation"}
            and _known_identity_string(record.get("session_id"))
            and isinstance(record.get("process_id"), int)
            and not isinstance(record["process_id"], bool)
            and record["process_id"] > 0
            and record.get("component_id")
            in {"anchor", "interpolation", "instrumentation"}
            and _finite_number(record.get("predicted_ns"))
            and isinstance(record.get("observed_samples_ns"), list)
            and bool(record["observed_samples_ns"])
            and all(
                _finite_number(sample)
                for sample in record["observed_samples_ns"]
            )
            for record in calibration_records
        )
    )


def _verified_artifact_content(
    verified_artifacts: dict[str, object],
    ref: str,
    *,
    role: str,
    schema: str,
) -> dict[str, Any] | None:
    artifact = verified_artifacts.get(ref)
    if not isinstance(artifact, dict):
        return None
    manifest_entry = artifact.get("manifest")
    content = artifact.get("content")
    if (
        not isinstance(manifest_entry, dict)
        or manifest_entry.get("role") != role
        or manifest_entry.get("schema") != schema
        or manifest_entry.get("media_type") != "application/json"
        or not isinstance(content, dict)
    ):
        return None
    return content


def _uncertainty_policy_artifacts_valid(
    policy: dict[str, Any],
    *,
    stable_path: str,
    surface: dict[str, Any],
    contract: dict[str, Any],
    verified_artifacts: dict[str, object],
) -> dict[str, float] | None:
    target = policy["target_coverage"]
    expected_target = {
        "stable_path": stable_path,
        "surface": surface,
        "execution_domain_sha256": _canonical_digest(
            contract["execution_domain"]
        ),
        "cohort_id": contract["cohort_id"],
        "coverage_method": target["coverage_method"],
        "minimum_fraction": target["minimum_fraction"],
        "minimum_calibration_records": target[
            "minimum_calibration_records"
        ],
        "minimum_validation_records": target[
            "minimum_validation_records"
        ],
        "required_calibration_target_ids": target[
            "required_calibration_target_ids"
        ],
        "required_validation_target_ids": target[
            "required_validation_target_ids"
        ],
    }
    target_content = _verified_artifact_content(
        verified_artifacts,
        target["evidence_ref"],
        role="uncertainty-target-coverage",
        schema="groundupscale.dev/uncertainty-target-coverage/v1alpha1",
    )
    calibration = policy["calibration"]
    calibration_content = _verified_artifact_content(
        verified_artifacts,
        calibration["evidence_ref"],
        role="uncertainty-calibration",
        schema="groundupscale.dev/uncertainty-calibration/v1alpha1",
    )
    expected_calibration = {
        "schema": "groundupscale.dev/uncertainty-calibration/v1alpha1",
        "policy_id": policy["policy_id"],
        "version": policy["version"],
        "target_coverage": expected_target,
        "estimator": calibration["estimator"],
        "records": calibration["records"],
    }
    metadata_valid = (
        {
            key: item for key, item in target.items() if key != "evidence_ref"
        }
        == expected_target
        and target_content
        == {
            "schema": "groundupscale.dev/uncertainty-target-coverage/v1alpha1",
            **expected_target,
        }
        and calibration_content
        == expected_calibration
    )
    if not metadata_valid:
        return None
    records = calibration_content["records"]
    calibration_records = [
        record for record in records if record["partition"] == "calibration"
    ]
    validation_records = [
        record for record in records if record["partition"] == "validation"
    ]
    required_calibration_targets = set(
        target["required_calibration_target_ids"]
    )
    required_validation_targets = set(
        target["required_validation_target_ids"]
    )
    calibration_targets = {
        record["target_id"] for record in calibration_records
    }
    validation_targets = {
        record["target_id"] for record in validation_records
    }
    calibration_sessions = {
        record["session_id"] for record in calibration_records
    }
    validation_sessions = {
        record["session_id"] for record in validation_records
    }
    calibration_processes = {
        record["process_id"] for record in calibration_records
    }
    validation_processes = {
        record["process_id"] for record in validation_records
    }
    if (
        len(calibration_records) < target["minimum_calibration_records"]
        or len(validation_records) < target["minimum_validation_records"]
        or calibration_targets != required_calibration_targets
        or validation_targets != required_validation_targets
        or len(calibration_targets) != len(calibration_records)
        or len(validation_targets) != len(validation_records)
        or len(calibration_sessions) != len(calibration_records)
        or len(validation_sessions) != len(validation_records)
        or len(calibration_processes) != len(calibration_records)
        or len(validation_processes) != len(validation_records)
        or not calibration_sessions.isdisjoint(validation_sessions)
        or not calibration_processes.isdisjoint(validation_processes)
    ):
        return None
    limits: dict[str, float] = {}
    for record in calibration_records:
        component_id = record["component_id"]
        residual_limit = max(
            abs(float(sample) - float(record["predicted_ns"]))
            for sample in record["observed_samples_ns"]
        )
        limits[component_id] = max(
            limits.get(component_id, 0.0), residual_limit
        )
    if set(limits) != {"anchor", "interpolation", "instrumentation"}:
        return None
    validation_components = {
        record["component_id"] for record in validation_records
    }
    validation_residuals = [
        (
            record["component_id"],
            abs(float(sample) - float(record["predicted_ns"])),
        )
        for record in validation_records
        for sample in record["observed_samples_ns"]
    ]
    if (
        validation_components
        != {"anchor", "interpolation", "instrumentation"}
        or not validation_residuals
        or sum(
            residual <= limits[component_id]
            for component_id, residual in validation_residuals
        )
        / len(validation_residuals)
        < float(target["minimum_fraction"])
    ):
        return None
    return limits


def _uncertainty_records_artifacts_valid(
    records: object,
    limits: dict[str, float],
    verified_artifacts: dict[str, object],
) -> bool:
    if not isinstance(records, list):
        return False
    for record in records:
        if not isinstance(record, dict):
            return False
        component_id = record.get("component_id")
        value = record.get("standard_uncertainty_ns")
        content = _verified_artifact_content(
            verified_artifacts,
            record.get("evidence_ref"),
            role="uncertainty-component",
            schema="groundupscale.dev/uncertainty-component/v1alpha1",
        )
        if (
            component_id not in limits
            or not _finite_number(value)
            or float(value) > float(limits[component_id])
            or content
            != {
                "schema": "groundupscale.dev/uncertainty-component/v1alpha1",
                "component_id": component_id,
                "standard_uncertainty_ns": value,
            }
        ):
            return False
    return True


def _frontier_uncertainty_artifacts_valid(
    frontier: dict[str, Any],
    *,
    stable_path: str,
    contract: dict[str, Any],
    verified_artifacts: dict[str, object],
) -> bool:
    surface = frontier["surface"]
    old_policy = frontier["old_surface_uncertainty_policy"]
    neighbourhood = frontier["neighbourhood"]
    neighbourhood_policy = neighbourhood["qualification_policy"]
    old_limits = _uncertainty_policy_artifacts_valid(
        old_policy,
        stable_path=stable_path,
        surface=surface,
        contract=contract,
        verified_artifacts=verified_artifacts,
    )
    neighbourhood_limits = _uncertainty_policy_artifacts_valid(
        neighbourhood_policy,
        stable_path=stable_path,
        surface=surface,
        contract=contract,
        verified_artifacts=verified_artifacts,
    )
    return (
        old_limits is not None
        and neighbourhood_limits is not None
        and _uncertainty_records_artifacts_valid(
            frontier["old_surface_uncertainty_records"],
            old_limits,
            verified_artifacts,
        )
        and all(
            _uncertainty_records_artifacts_valid(
                record.get("uncertainty_records"),
                neighbourhood_limits,
                verified_artifacts,
            )
            for record in [
                *neighbourhood["anchor_records"],
                *neighbourhood["local_probe_records"],
                *neighbourhood["refit_records"],
            ]
            if isinstance(record, dict)
        )
    )


def _integration_operator_frontier_valid(
    value: object,
    *,
    stable_path: str,
    contract: dict[str, Any],
    verified_artifacts: dict[str, object],
) -> bool:
    if not isinstance(value, dict):
        return False
    surface = value.get("surface")
    policy = value.get("uncertainty_policy")
    records = value.get("uncertainty_records")
    evidence_ref = value.get("evidence_ref")
    combined_uncertainty_ns = value.get("combined_uncertainty_ns")
    if (
        value.get("schema")
        != "groundupscale.dev/operator-frontier-evidence/v1alpha1"
        or value.get("stable_path") != stable_path
        or value.get("observation_validity") != "QUALIFIED"
        or value.get("frontier_role") != "ACTIVE"
        or value.get("cohort_id") != contract.get("cohort_id")
        or value.get("execution_domain") != contract.get("execution_domain")
        or not isinstance(surface, dict)
        or not _surface_id_text(surface.get("surface_id"))
        or not _surface_version_text(surface.get("version"))
        or not _finite_number(value.get("latency_ns"))
        or float(value["latency_ns"]) <= 0
        or not _finite_number(combined_uncertainty_ns)
        or float(combined_uncertainty_ns) < 0
        or not _artifact_uri(evidence_ref)
        or value.get("evidence_refs") != [evidence_ref]
    ):
        return False
    frontier_content = _verified_artifact_content(
        verified_artifacts,
        evidence_ref,
        role="operator-frontier-evidence",
        schema="groundupscale.dev/operator-frontier-evidence/v1alpha1",
    )
    expected_content = {
        key: item for key, item in value.items() if key != "evidence_ref"
    }
    if frontier_content != expected_content:
        return False
    source_basis = value.get("uncertainty_basis")
    if (
        isinstance(source_basis, dict)
        and source_basis.get("kind")
        == "verified-capability-surface-query"
    ):
        qualification = _verified_artifact_content(
            verified_artifacts,
            source_basis.get("qualification_evidence_ref"),
            role="source-frontier-qualification",
            schema=(
                "groundupscale.dev/operator-frontier-qualification/v1alpha1"
            ),
        )
        query = source_basis.get("query")
        source_surface = (
            qualification.get("surface")
            if isinstance(qualification, dict)
            else None
        )
        if not isinstance(source_surface, dict) or not isinstance(query, dict):
            return False
        try:
            _, _, query_results = _capability_surface_results(
                {
                    "capability_surfaces": [source_surface],
                    "surface_queries": [query],
                    "cohort_id": contract.get("cohort_id"),
                }
            )
        except DiagnosticBundleIntegrityError:
            return False
        if len(query_results) != 1:
            return False
        result = query_results[0]
        work_rate_latency = result.get("work_rate_latency")
        uncertainty = result.get("uncertainty")
        interval = (
            uncertainty.get("latency_interval")
            if isinstance(uncertainty, dict)
            else None
        )
        execution_shape = contract.get("execution_domain", {}).get("shape", {})
        response_model = source_surface.get("response_model")
        expected_query_shape = (
            {"m": execution_shape.get("m")}
            if isinstance(response_model, dict)
            and response_model.get("primary_response") == "latency_ns"
            else {"s": execution_shape.get("m")}
        )
        if (
            result.get("status") != "exact_anchor"
            or not isinstance(work_rate_latency, dict)
            or not _finite_number(work_rate_latency.get("value_ns"))
            or not isinstance(interval, dict)
            or not _finite_number(interval.get("lower_ns"))
            or not _finite_number(interval.get("upper_ns"))
            or result.get("surface") != surface
            or query.get("shape") != expected_query_shape
        ):
            return False
        latency_ns = float(work_rate_latency["value_ns"])
        expected_uncertainty_ns = max(
            latency_ns - float(interval["lower_ns"]),
            float(interval["upper_ns"]) - latency_ns,
        )
        return (
            qualification.get("status") == "qualified"
            and qualification.get("hardware_cohort")
            == contract.get("cohort_id")
            and source_basis.get("source_policy")
            == {
                key: source_surface["uncertainty_policy"][key]
                for key in (
                    "policy_id",
                    "version",
                    "combination",
                    "target_coverage",
                )
            }
            and source_basis.get("latency_interval") == interval
            and set(source_basis)
            == {
                "kind",
                "qualification_evidence_ref",
                "query",
                "source_policy",
                "latency_interval",
                "surface_uncertainty_ns",
            }
            and isclose(
                float(source_basis.get("surface_uncertainty_ns", -1)),
                expected_uncertainty_ns,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            and isclose(
                float(value["latency_ns"]),
                latency_ns,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            and isclose(
                float(combined_uncertainty_ns),
                expected_uncertainty_ns,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
        )
    if not _uncertainty_policy_structure_valid(policy):
        return False
    limits = _uncertainty_policy_artifacts_valid(
        policy,
        stable_path=stable_path,
        surface=surface,
        contract=contract,
        verified_artifacts=verified_artifacts,
    )
    if (
        limits is None
        or not _uncertainty_records_artifacts_valid(
            records,
            limits,
            verified_artifacts,
        )
    ):
        return False
    assert isinstance(records, list)
    if not all(
        isinstance(record, dict)
        and _finite_number(record.get("standard_uncertainty_ns"))
        and float(record["standard_uncertainty_ns"]) >= 0
        for record in records
    ):
        return False
    component_values = {
        record["component_id"]: float(record["standard_uncertainty_ns"])
        for record in records
    }
    return (
        len(records) == 3
        and set(component_values)
        == {"anchor", "interpolation", "instrumentation"}
        and isclose(
            float(combined_uncertainty_ns),
            hypot(*component_values.values()),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    )


def _diagnostic_supporting_artifact_matches(
    verified_artifacts: dict[str, object],
    artifact_ref: object,
    payload: dict[str, Any],
) -> bool:
    schema = (
        "groundupscale.dev/diagnostic-supporting-evidence/v1alpha1"
    )
    content = _verified_artifact_content(
        verified_artifacts,
        artifact_ref,
        role="diagnostic-supporting-evidence",
        schema=schema,
    )
    return content == {"schema": schema, "payload": payload}


def _direct_diagnostic_artifact_matches(
    verified_artifacts: dict[str, object], value: object
) -> bool:
    return isinstance(value, dict) and (
        _diagnostic_supporting_artifact_matches(
            verified_artifacts,
            value.get("evidence_ref"),
            {
                key: item
                for key, item in value.items()
                if key != "evidence_ref"
            },
        )
    )


def _c3_enumerated_pool_artifacts_valid(
    frontier_evidence: object,
    verified_artifacts: dict[str, object],
) -> bool:
    if not isinstance(frontier_evidence, dict):
        return False
    if frontier_evidence.get("candidate_coverage") != "C3_ENUMERATED_POOL":
        return True
    if (
        frontier_evidence.get("enumerated_pool_manifest") is None
        and frontier_evidence.get("enumerated_pool_coverage_proof") is None
    ):
        return True
    return _direct_diagnostic_artifact_matches(
        verified_artifacts,
        frontier_evidence.get("enumerated_pool_manifest"),
    ) and _direct_diagnostic_artifact_matches(
        verified_artifacts,
        frontier_evidence.get("enumerated_pool_coverage_proof"),
    )


def _direct_defect_evidence(
    value: object,
    *,
    candidates: list[dict[str, Any]],
    contract: dict[str, Any],
    verified_artifacts: dict[str, object],
) -> dict[str, Any] | None:
    """Qualify direct, replayable evidence independently of performance."""
    if not isinstance(value, dict):
        return None
    input_summary = value.get("input_summary")
    identity = value.get("candidate_identity")
    environment = value.get("environment")
    failure = value.get("failure")
    repetitions = value.get("repetitions")
    supporting_refs = value.get("supporting_evidence_refs")
    targets = [
        candidate
        for candidate in candidates
        if candidate.get("candidate_id") == value.get("target_candidate_id")
        and candidate.get("role") == "target"
    ]
    if len(targets) != 1:
        return None
    target = targets[0]
    implementation = target["implementation_family"]

    input_sha256 = (
        input_summary.get("input_sha256")
        if isinstance(input_summary, dict)
        else None
    )
    target_sessions = target["session_process_ids"]
    kind = value.get("defect_kind")
    common_valid = (
        value.get("schema")
        == "groundupscale.dev/direct-defect-evidence/v1alpha1"
        and kind
        in _DIRECT_DEFECT_GATE_IDS
        and _artifact_uri(value.get("evidence_ref"))
        and _diagnostic_supporting_artifact_matches(
            verified_artifacts,
            value.get("evidence_ref"),
            {
                key: nested
                for key, nested in value.items()
                if key != "evidence_ref"
            },
        )
        and isinstance(supporting_refs, list)
        and bool(supporting_refs)
        and all(_artifact_uri(ref) for ref in supporting_refs)
        and isinstance(input_summary, dict)
        and isinstance(input_sha256, str)
        and fullmatch(r"[0-9a-f]{64}", input_sha256) is not None
        and input_summary.get("shape") == contract["shape"]
        and input_summary.get("dtype") == contract["dtype"]
        and input_summary.get("layout") == contract["layout"]
        and input_summary.get("strides") == contract["strides"]
        and input_summary.get("alignment_bytes")
        == contract["alignment_bytes"]
        and _direct_diagnostic_artifact_matches(
            verified_artifacts, input_summary
        )
        and identity
        == {
            "candidate_id": target["candidate_id"],
            "family_id": implementation["family_id"],
            "family_version": implementation["version"],
            "implementation_ref": implementation["implementation_ref"],
            "implementation_sha256": implementation[
                "implementation_sha256"
            ],
            "source_identity": implementation["source_identity"],
        }
        and environment
        == {
            "cohort_id": contract["cohort_id"],
            "cohort_identity": contract["cohort_identity"],
            "preflight": contract["environment"],
        }
        and isinstance(failure, dict)
        and _direct_diagnostic_artifact_matches(verified_artifacts, failure)
        and isinstance(repetitions, list)
        and len(repetitions) >= _FRONTIER_MINIMUM_INDEPENDENT_SESSIONS
        and len(repetitions) == len(target_sessions)
        and all(
            _direct_diagnostic_artifact_matches(
                verified_artifacts, repetition
            )
            for repetition in repetitions
        )
    )
    if not common_valid:
        return None

    expected_sha256 = failure.get("expected_sha256")
    observed_sha256 = failure.get("observed_sha256")
    correctness_failure_valid = (
        kind == "correctness_oracle_violation"
        and supporting_refs == [target["correctness"]["evidence_ref"]]
        and target["correctness"]["passed"] is False
        and failure.get("failure_kind") == "correctness_difference"
        and failure.get("oracle")
        == contract["correctness_policy"]["oracle"]
        and isinstance(expected_sha256, str)
        and fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None
        and isinstance(observed_sha256, str)
        and fullmatch(r"[0-9a-f]{64}", observed_sha256) is not None
        and expected_sha256 != observed_sha256
        and _finite_number(failure.get("max_abs_difference"))
        and float(failure["max_abs_difference"]) > 0
        and isinstance(failure.get("mismatched_elements"), int)
        and not isinstance(failure["mismatched_elements"], bool)
        and failure["mismatched_elements"] > 0
    )

    contract_field = failure.get("contract_field")
    allowed_contract_fields = {
        "threads",
        "alignment_bytes",
        "dtype",
        "layout",
        "completion_boundary.kind",
        "completion_boundary.closed",
        "completion_boundary.threadpool_joined",
        "execution_domain.execution_mode",
        "execution_domain.affinity",
        "execution_domain.numa",
        "execution_domain.context",
        "execution_domain.stream",
        "execution_domain.concurrency",
    }
    expected_contract_value: object = contract
    if isinstance(contract_field, str):
        for segment in contract_field.split("."):
            expected_contract_value = (
                expected_contract_value.get(segment)
                if isinstance(expected_contract_value, dict)
                else None
            )
    contract_failure_valid = (
        kind == "execution_contract_violation"
        and supporting_refs == [failure["evidence_ref"]]
        and failure.get("failure_kind") == "execution_contract_violation"
        and contract_field in allowed_contract_fields
        and failure.get("expected") == expected_contract_value
        and failure.get("observed") != expected_contract_value
    )
    if not correctness_failure_valid and not contract_failure_valid:
        return None
    repeated_sessions = {
        repetition.get("session_id"): repetition
        for repetition in repetitions
        if isinstance(repetition, dict)
    }
    if (
        len(repeated_sessions) != len(repetitions)
        or set(repeated_sessions) != set(target_sessions)
        or len(
            {
                repetition.get("process_id")
                for repetition in repetitions
                if isinstance(repetition, dict)
            }
        )
        < _FRONTIER_MINIMUM_INDEPENDENT_SESSIONS
        or not all(
            repetition.get("process_id")
            == target_sessions[repetition.get("session_id")]
            and repetition.get("input_sha256") == input_sha256
            and repetition.get("outcome") == "violation"
            for repetition in repetitions
            if isinstance(repetition, dict)
        )
    ):
        return None
    return {
        **value,
        "evidence_refs": list(
            dict.fromkeys(
                [
                    value["evidence_ref"],
                    *supporting_refs,
                    input_summary["evidence_ref"],
                    failure["evidence_ref"],
                    *[
                        repetition["evidence_ref"]
                        for repetition in repetitions
                    ],
                ]
            )
        ),
    }


def _integration_evidence_artifacts_valid(
    evidence: object,
    *,
    target_candidate: object,
    measurement_lanes: object,
    verified_artifacts: dict[str, object],
) -> bool:
    if not isinstance(evidence, dict) or not isinstance(
        target_candidate, dict
    ) or not isinstance(measurement_lanes, dict):
        return False

    def plural(value: object) -> bool:
        if not isinstance(value, dict):
            return False
        refs = value.get("evidence_refs")
        payload = {
            key: item for key, item in value.items() if key != "evidence_refs"
        }
        return (
            isinstance(refs, list)
            and bool(refs)
            and all(
                _diagnostic_supporting_artifact_matches(
                    verified_artifacts,
                    artifact_ref,
                    payload,
                )
                for artifact_ref in refs
            )
        )

    def measurement(value: object) -> bool:
        if not isinstance(value, dict):
            return True
        correctness = value.get("correctness")
        sessions = value.get("sessions")
        return (
            _direct_diagnostic_artifact_matches(verified_artifacts, value)
            and (
                correctness is None
                or _direct_diagnostic_artifact_matches(
                    verified_artifacts, correctness
                )
            )
            and (
                not isinstance(sessions, list)
                or all(
                    _direct_diagnostic_artifact_matches(
                        verified_artifacts, session
                    )
                    for session in sessions
                )
            )
        )

    def derived_ablation(value: object) -> bool:
        if not isinstance(value, dict):
            return False
        derivation = value.get("derivation")
        sessions = value.get("sessions")
        expected_prior = {
            "dispatch": ["frontier_adapter"],
            "copy": ["frontier_adapter", "dispatch"],
            "sync": ["frontier_adapter", "dispatch", "copy"],
            "profiling": [
                "frontier_adapter",
                "dispatch",
                "copy",
                "sync",
            ],
        }
        kind = value.get("kind")
        input_refs = (
            derivation.get("input_refs")
            if isinstance(derivation, dict)
            else None
        )
        if (
            kind not in expected_prior
            or not isinstance(derivation, dict)
            or derivation.get("formula")
            != (
                "max(0, median(cumulative_variant) - "
                "max(median(prior_cumulative_variants)))"
            )
            or derivation.get("sample_semantics")
            != "derived-paired-session-delta"
            or derivation.get("cumulative_variant") != kind
            or derivation.get("prior_cumulative_variants")
            != expected_prior[kind]
            or not isinstance(input_refs, list)
            or not isinstance(sessions, list)
            or len(input_refs) != len(sessions)
            or len(set(input_refs)) != len(input_refs)
        ):
            return False
        for session, input_ref in zip(sessions, input_refs, strict=True):
            source = _verified_artifact_content(
                verified_artifacts,
                input_ref,
                role="source-diagnostic-session",
                schema=(
                    "groundupscale.dev/ascend-diagnostic-session/v1alpha1"
                ),
            )
            variants = source.get("variants") if isinstance(source, dict) else None
            if (
                not isinstance(session, dict)
                or not isinstance(source, dict)
                or not isinstance(variants, dict)
                or source.get("session_id") != session.get("session_id")
                or source.get("process_id") != session.get("process_id")
                or source.get("cohort_id") != session.get("cohort_id")
                or "raw_samples_ns" in session
                or not isinstance(session.get("derived_samples_ns"), list)
                or not session["derived_samples_ns"]
            ):
                return False
            medians: dict[str, float] = {}
            for variant_name in [kind, *expected_prior[kind]]:
                variant = variants.get(variant_name)
                raw_samples = (
                    variant.get("raw_samples_ns")
                    if isinstance(variant, dict)
                    else None
                )
                if (
                    not isinstance(raw_samples, list)
                    or not raw_samples
                    or not all(_finite_number(sample) for sample in raw_samples)
                ):
                    return False
                medians[variant_name] = float(median(raw_samples))
            expected = max(
                0.0,
                medians[kind]
                - max(medians[name] for name in expected_prior[kind]),
            )
            derived_samples = session["derived_samples_ns"]
            if (
                len(derived_samples) != 1
                or not _finite_number(derived_samples[0])
                or not _finite_number(session.get("latency_ns"))
                or not isclose(
                    float(derived_samples[0]),
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                or not isclose(
                    float(session["latency_ns"]),
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
            ):
                return False
        return measurement(value)

    def ablation(value: object) -> bool:
        derivation = value.get("derivation") if isinstance(value, dict) else None
        if (
            isinstance(derivation, dict)
            and derivation.get("sample_semantics")
            == "derived-paired-session-delta"
        ):
            return derived_ablation(value)
        return measurement(value)

    target_sessions = target_candidate.get("sessions")
    target_correctness = target_candidate.get("correctness")
    baseline_lane = measurement_lanes.get("baseline")
    diagnostic_lane = measurement_lanes.get("diagnostic")
    wrapped = evidence.get("wrapped_e2e")
    ablations = evidence.get("ablations")
    ledger = evidence.get("exclusive_ledger")
    counterfactual = evidence.get("counterfactual")
    return (
        plural(evidence)
        and _direct_diagnostic_artifact_matches(
            verified_artifacts, target_correctness
        )
        and _direct_diagnostic_artifact_matches(
            verified_artifacts, baseline_lane
        )
        and _direct_diagnostic_artifact_matches(
            verified_artifacts, diagnostic_lane
        )
        and (
            not isinstance(target_sessions, list)
            or all(
                _direct_diagnostic_artifact_matches(
                    verified_artifacts, session
                )
                for session in target_sessions
            )
        )
        and measurement(wrapped)
        and (
            not isinstance(ablations, list)
            or all(ablation(item) for item in ablations)
        )
        and (ledger is None or plural(ledger))
        and (counterfactual is None or plural(counterfactual))
    )


def _frontier_shift_evidence_valid(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    surface = value.get("surface")
    surface_reference = value.get("old_surface_reference")
    uncertainty_records = value.get("old_surface_uncertainty_records")
    uncertainty_policy = value.get("old_surface_uncertainty_policy")
    holdout = value.get("holdout")
    neighbourhood = value.get("neighbourhood")
    evidence_refs = value.get("evidence_refs")
    return (
        isinstance(surface, dict)
        and all(
            _nonempty_string(surface.get(key))
            for key in ("surface_id", "version")
        )
        and _exact_version_text(surface.get("version"))
        and isinstance(surface_reference, dict)
        and surface_reference.get("surface_id") == surface.get("surface_id")
        and surface_reference.get("version") == surface.get("version")
        and _finite_number(surface_reference.get("predicted_ns"))
        and float(surface_reference["predicted_ns"]) > 0
        and isinstance(surface_reference.get("execution_domain"), dict)
        and _nonempty_string(surface_reference.get("cohort_id"))
        and _nonempty_string(surface_reference.get("evidence_ref"))
        and isinstance(uncertainty_records, list)
        and {
            record.get("component_id")
            for record in uncertainty_records
            if isinstance(record, dict)
        }
        == {"anchor", "interpolation", "instrumentation"}
        and len(uncertainty_records) == 3
        and all(
            isinstance(record, dict)
            and _finite_number(record.get("standard_uncertainty_ns"))
            and float(record["standard_uncertainty_ns"]) >= 0
            and _nonempty_string(record.get("evidence_ref"))
            for record in uncertainty_records
        )
        and _uncertainty_policy_structure_valid(uncertainty_policy)
        and isinstance(holdout, dict)
        and isinstance(holdout.get("selection_session_ids"), list)
        and all(
            _nonempty_string(session_id)
            for session_id in holdout["selection_session_ids"]
        )
        and len(holdout["selection_session_ids"])
        == len(set(holdout["selection_session_ids"]))
        and isinstance(holdout.get("sessions"), list)
        and isinstance(holdout.get("candidate_results"), list)
        and _nonempty_string(holdout.get("evidence_ref"))
        and isinstance(neighbourhood, dict)
        and _nonempty_string(neighbourhood.get("regime_id"))
        and isinstance(neighbourhood.get("qualification_policy"), dict)
        and _uncertainty_policy_structure_valid(
            neighbourhood["qualification_policy"]
        )
        and all(
            isinstance(neighbourhood["qualification_policy"].get(key), int)
            and not isinstance(
                neighbourhood["qualification_policy"][key], bool
            )
            and neighbourhood["qualification_policy"][key] > 0
            for key in (
                "minimum_stable_anchor_records",
                "minimum_refit_records",
                "local_shape_radius",
            )
        )
        and isinstance(neighbourhood.get("anchor_records"), list)
        and isinstance(neighbourhood.get("local_probe_records"), list)
        and isinstance(neighbourhood.get("refit_records"), list)
        and isinstance(evidence_refs, list)
        and bool(evidence_refs)
        and all(_nonempty_string(ref) for ref in evidence_refs)
    )


def _probe_counterexamples_valid(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(counterexample, dict)
        and _nonempty_string(counterexample.get("counterexample_id"))
        and isinstance(counterexample.get("reason_codes"), list)
        and bool(counterexample["reason_codes"])
        and all(
            _nonempty_string(reason)
            for reason in counterexample["reason_codes"]
        )
        and isinstance(counterexample.get("evidence_refs"), list)
        and bool(counterexample["evidence_refs"])
        and all(
            _nonempty_string(ref)
            for ref in counterexample["evidence_refs"]
        )
        for counterexample in value
    )


def _normalize_timing_sessions(
    sessions: object,
    *,
    expected_lane_id: str,
    expected_cohort_id: str,
    minimum_sessions: int,
    require_authored_latency: bool,
    allow_zero_samples: bool = False,
    samples_field: str = "raw_samples_ns",
) -> dict[str, Any] | None:
    """Validate one lane/cohort and derive medians from declared samples."""
    if (
        not isinstance(sessions, list)
        or not sessions
        or samples_field not in {"raw_samples_ns", "derived_samples_ns"}
    ):
        return None
    session_latencies_ns: dict[str, float] = {}
    session_process_ids: dict[str, int] = {}
    raw_samples_ns: dict[str, list[object]] = {}
    included_samples_ns: dict[str, list[float]] = {}
    session_exclusions: dict[str, list[dict[str, object]]] = {}
    evidence_refs: list[str] = []
    for session in sessions:
        samples = (
            session.get(samples_field)
            if isinstance(session, dict)
            else None
        )
        exclusions = (
            session.get("excluded_samples")
            if isinstance(session, dict)
            else None
        )
        if (
            not isinstance(session, dict)
            or not _nonempty_string(session.get("session_id"))
            or session["session_id"] in session_latencies_ns
            or not isinstance(session.get("process_id"), int)
            or isinstance(session["process_id"], bool)
            or session["process_id"] <= 0
            or session.get("lane_id") != expected_lane_id
            or session.get("cohort_id") != expected_cohort_id
            or not isinstance(samples, list)
            or not samples
            or not all(
                _finite_number(sample)
                and (
                    float(sample) >= 0
                    if allow_zero_samples
                    else float(sample) > 0
                )
                for sample in samples
            )
            or not isinstance(exclusions, list)
            or not _nonempty_string(session.get("evidence_ref"))
        ):
            return None
        excluded_indices: set[int] = set()
        normalized_exclusions: list[dict[str, object]] = []
        for exclusion in exclusions:
            if (
                not isinstance(exclusion, dict)
                or set(exclusion) != {"index", "reason"}
                or not isinstance(exclusion.get("index"), int)
                or isinstance(exclusion["index"], bool)
                or exclusion["index"] < 0
                or exclusion["index"] >= len(samples)
                or exclusion["index"] in excluded_indices
                or not _resolved_identity_string(exclusion.get("reason"))
            ):
                return None
            excluded_indices.add(exclusion["index"])
            normalized_exclusions.append(
                {
                    "index": exclusion["index"],
                    "reason": exclusion["reason"],
                }
            )
        included = [
            float(sample)
            for index, sample in enumerate(samples)
            if index not in excluded_indices
        ]
        if not included:
            return None
        session_latency_ns = float(median(included))
        if require_authored_latency and (
            not _finite_number(session.get("latency_ns"))
            or float(session["latency_ns"]) <= 0
            or not isclose(
                float(session["latency_ns"]),
                session_latency_ns,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
        ):
            return None
        session_id = session["session_id"]
        session_latencies_ns[session_id] = session_latency_ns
        session_process_ids[session_id] = session["process_id"]
        raw_samples_ns[session_id] = list(samples)
        included_samples_ns[session_id] = included
        session_exclusions[session_id] = normalized_exclusions
        evidence_refs.append(session["evidence_ref"])
    if (
        len(session_latencies_ns) < minimum_sessions
        or len(set(session_process_ids.values()))
        != len(session_process_ids)
        or len(set(session_process_ids.values())) < minimum_sessions
    ):
        return None
    return {
        "aggregate_latency_ns": float(median(session_latencies_ns.values())),
        "session_ids": list(session_latencies_ns),
        "session_latencies_ns": session_latencies_ns,
        "session_process_ids": session_process_ids,
        samples_field: raw_samples_ns,
        "samples_field": samples_field,
        "included_samples_ns": included_samples_ns,
        "session_exclusions": session_exclusions,
        "evidence_refs": evidence_refs,
    }


def _shape_disambiguation_probes(
    document: dict[str, Any],
    trigger: dict[str, Any] | None,
    verified_artifacts: dict[str, object],
) -> list[dict[str, Any]]:
    probe_value = document.get("shape_disambiguation_probes")
    probes = probe_value if isinstance(probe_value, list) else []
    triggered_paths = (
        {
            item["stable_path"]
            for item in trigger.get("triggered", [])
            if isinstance(item, dict)
            and _nonempty_string(item.get("stable_path"))
        }
        if isinstance(trigger, dict)
        else set()
    )
    results = []
    for probe in probes:
        if not isinstance(probe, dict):
            results.append(
                {
                    "status": "insufficient_evidence",
                    "reason_code": "invalid-shape-probe",
                    "evidence_refs": [],
                }
            )
            continue
        probe_id = probe.get("probe_id")
        stable_path = probe.get("stable_path")
        evidence_refs = probe.get("evidence_refs")
        result_prefix = {
            "probe_id": probe_id,
            "stable_path": stable_path,
            "evidence_refs": (
                list(evidence_refs)
                if isinstance(evidence_refs, list)
                and all(_nonempty_string(ref) for ref in evidence_refs)
                else []
            ),
        }
        if stable_path not in triggered_paths:
            results.append(
                {
                    **result_prefix,
                    "status": "not_evaluated",
                    "reason_code": "diagnostic-trigger-not-met",
                }
            )
            continue
        contract = probe.get("locked_contract")
        candidates = probe.get("candidates")
        measurement_lanes = probe.get("measurement_lanes")
        frontier_shift_evidence = probe.get("frontier_shift_evidence")
        counterexamples = probe.get("counterexamples", [])
        environment = (
            contract.get("environment")
            if isinstance(contract, dict)
            else None
        )
        if (
            isinstance(environment, dict)
            and environment.get("eligible") is not True
        ):
            results.append(
                {
                    **result_prefix,
                    "status": "insufficient_evidence",
                    "reason_code": "ineligible-probe-environment",
                }
            )
            continue
        if (
            not _nonempty_string(probe_id)
            or not _canonical_stable_path(stable_path)
            or not _locked_probe_contract_valid(
                contract,
                cohort_id=document.get("cohort_id"),
                hardware=document.get("hardware"),
                execution_domain=document.get("execution_domain"),
            )
            or not isinstance(candidates, list)
            or not _measurement_lanes_valid(
                measurement_lanes,
                stable_path=stable_path,
                contract=contract,
                integration_verdict_requested=(
                    "integration_overhead_evidence" in probe
                ),
            )
            or (
                frontier_shift_evidence is not None
                and not _frontier_shift_evidence_valid(
                    frontier_shift_evidence
                )
            )
            or (
                isinstance(frontier_shift_evidence, dict)
                and not _frontier_uncertainty_artifacts_valid(
                    frontier_shift_evidence,
                    stable_path=stable_path,
                    contract=contract,
                    verified_artifacts=verified_artifacts,
                )
            )
            or (
                isinstance(frontier_shift_evidence, dict)
                and not _c3_enumerated_pool_artifacts_valid(
                    frontier_shift_evidence,
                    verified_artifacts,
                )
            )
            or not _probe_counterexamples_valid(counterexamples)
            or not _probe_references_valid(probe)
        ):
            results.append(
                {
                    **result_prefix,
                    "status": "insufficient_evidence",
                    "reason_code": "invalid-shape-probe",
                }
            )
            continue
        if not _artifact_refs(probe).issubset(verified_artifacts):
            results.append(
                {
                    **result_prefix,
                    "status": "insufficient_evidence",
                    "reason_code": "unresolved-probe-evidence-ref",
                }
            )
            continue
        integration_evidence = probe.get("integration_overhead_evidence")
        integration_targets = [
            candidate
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("role") == "target"
        ]
        integration_artifacts_verified = (
            _integration_evidence_artifacts_valid(
                integration_evidence,
                target_candidate=integration_targets[0],
                measurement_lanes=measurement_lanes,
                verified_artifacts=verified_artifacts,
            )
            if isinstance(integration_evidence, dict)
            and len(integration_targets) == 1
            else None
        )
        integration_frontier_verified = (
            _integration_operator_frontier_valid(
                integration_evidence.get("operator_frontier"),
                stable_path=stable_path,
                contract=contract,
                verified_artifacts=verified_artifacts,
            )
            if isinstance(integration_evidence, dict)
            and len(integration_targets) == 1
            else None
        )
        locked_candidate_ids = contract["candidate_ids"]
        candidate_ids = [
            candidate.get("candidate_id")
            for candidate in candidates
            if isinstance(candidate, dict)
        ]
        if (
            len(candidate_ids) != len(candidates)
            or len(candidate_ids) != len(set(candidate_ids))
            or set(candidate_ids) != set(locked_candidate_ids)
        ):
            results.append(
                {
                    **result_prefix,
                    "status": "insufficient_evidence",
                    "reason_code": "candidate-set-does-not-match-lock",
                }
            )
            continue

        direct_defect_value = probe.get("direct_defect_evidence")
        direct_failure = (
            direct_defect_value.get("failure")
            if isinstance(direct_defect_value, dict)
            else None
        )
        candidate_evaluations = []
        malformed_candidate = False
        for candidate in candidates:
            correctness = candidate.get("correctness")
            implementation_family = candidate.get("implementation_family")
            family_artifact = (
                verified_artifacts.get(
                    implementation_family.get("manifest_ref")
                )
                if isinstance(implementation_family, dict)
                else None
            )
            family_manifest = (
                family_artifact.get("content")
                if isinstance(family_artifact, dict)
                else None
            )
            implementation_artifact = (
                verified_artifacts.get(
                    implementation_family.get("implementation_ref")
                )
                if isinstance(implementation_family, dict)
                else None
            )
            implementation_manifest_entry = (
                implementation_artifact.get("manifest")
                if isinstance(implementation_artifact, dict)
                else None
            )
            implementation_content = (
                implementation_artifact.get("content")
                if isinstance(implementation_artifact, dict)
                else None
            )
            sessions = candidate.get("sessions")
            evidence_lane = candidate.get("evidence_lane", "baseline")
            diagnostic_target_id = (
                direct_defect_value.get("target_candidate_id")
                if isinstance(direct_defect_value, dict)
                else None
            )
            correctness_passed = (
                _raw_correctness_passed(
                    correctness.get("records"), correctness.get("tolerance")
                )
                if isinstance(correctness, dict)
                else None
            )
            if (
                not _nonempty_string(candidate.get("role"))
                or not isinstance(candidate.get("eligible"), bool)
                or not isinstance(implementation_family, dict)
                or not _canonical_identifier(
                    implementation_family.get("family_id")
                )
                or not _exact_version_text(
                    implementation_family.get("version")
                )
                or not _nonempty_string(
                    implementation_family.get("manifest_ref")
                )
                or not _artifact_uri(
                    implementation_family.get("implementation_ref")
                )
                or not isinstance(
                    implementation_family.get("implementation_sha256"), str
                )
                or fullmatch(
                    r"[0-9a-f]{64}",
                    implementation_family["implementation_sha256"],
                )
                is None
                or not isinstance(family_artifact, dict)
                or not isinstance(family_artifact.get("manifest"), dict)
                or family_artifact["manifest"].get("role")
                != "implementation-family-manifest"
                or family_artifact["manifest"].get("schema")
                != "groundupscale.dev/implementation-family-manifest/v1alpha1"
                or family_artifact["manifest"].get("media_type")
                != "application/json"
                or not isinstance(family_manifest, dict)
                or family_manifest.get("schema")
                != "groundupscale.dev/implementation-family-manifest/v1alpha1"
                or family_manifest.get("family_id")
                != implementation_family.get("family_id")
                or family_manifest.get("version")
                != implementation_family.get("version")
                or family_manifest.get("implementation_sha256")
                != implementation_family.get("implementation_sha256")
                or family_manifest.get("implementation_ref")
                != implementation_family.get("implementation_ref")
                or not isinstance(implementation_manifest_entry, dict)
                or implementation_manifest_entry.get("role")
                != "candidate-implementation"
                or implementation_manifest_entry.get("schema")
                != "groundupscale.dev/candidate-implementation/v1alpha1"
                or implementation_manifest_entry.get("media_type")
                != "application/json"
                or implementation_manifest_entry.get("sha256")
                != implementation_family.get("implementation_sha256")
                or not isinstance(implementation_content, dict)
                or not _canonical_source_identity(
                    implementation_content.get("source_identity")
                )
                or implementation_content
                != {
                    "schema": (
                        "groundupscale.dev/"
                        "candidate-implementation/v1alpha1"
                    ),
                    "source_identity": implementation_content.get(
                        "source_identity"
                    ),
                }
                or family_manifest.get("source_identity")
                != implementation_content.get("source_identity")
                or not isinstance(correctness, dict)
                or correctness_passed is None
                or correctness.get("tolerance")
                != {
                    "atol": contract["correctness_policy"]["atol"],
                    "rtol": contract["correctness_policy"]["rtol"],
                }
                or not _nonempty_string(correctness.get("evidence_ref"))
                or not isinstance(sessions, list)
                or not sessions
                or evidence_lane not in {"baseline", "diagnostic"}
                or (
                    evidence_lane == "diagnostic"
                    and (
                        candidate.get("candidate_id")
                        != diagnostic_target_id
                        or measurement_lanes["diagnostic"].get(
                            "timing_used_for_verdict"
                        )
                        is not False
                    )
                )
            ):
                malformed_candidate = True
                break
            normalized_sessions = _normalize_timing_sessions(
                sessions,
                expected_lane_id=measurement_lanes[evidence_lane]["lane_id"],
                expected_cohort_id=contract["cohort_id"],
                minimum_sessions=1,
                require_authored_latency=True,
            )
            if (
                normalized_sessions is None
                or not _candidate_source_replay_valid(
                    candidate,
                    sessions=sessions,
                    stable_path=stable_path,
                    semantic=contract["semantic"],
                    evidence_lane=evidence_lane,
                    direct_failure=direct_failure,
                    verified_artifacts=verified_artifacts,
                )
            ):
                malformed_candidate = True
                break
            eligible_for_best = (
                candidate["eligible"] and correctness_passed
            )
            exclusion_reason = None
            if not candidate["eligible"]:
                exclusion_reason = "candidate-ineligible"
            elif not correctness_passed:
                exclusion_reason = "correctness-failed"
            candidate_evaluations.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "role": candidate["role"],
                    "evidence_lane": evidence_lane,
                    "implementation_family": {
                        **implementation_family,
                        "source_identity": implementation_content[
                            "source_identity"
                        ],
                    },
                    "correctness": {
                        "passed": correctness_passed,
                        "record_count": len(correctness["records"]),
                        "tolerance": dict(correctness["tolerance"]),
                        "evidence_ref": correctness["evidence_ref"],
                    },
                    "eligible_for_best_of_correct": eligible_for_best,
                    "excluded_evidence_roles": (
                        []
                        if eligible_for_best
                        else [
                            "best_of_correct",
                            "frontier_anchor",
                            "surface_winner",
                            "headroom_evidence",
                        ]
                    ),
                    "exclusion_reason": exclusion_reason,
                    "aggregate_latency_ns": normalized_sessions[
                        "aggregate_latency_ns"
                    ],
                    "session_ids": normalized_sessions["session_ids"],
                    "session_latencies_ns": normalized_sessions[
                        "session_latencies_ns"
                    ],
                    "session_process_ids": normalized_sessions[
                        "session_process_ids"
                    ],
                    "raw_samples_ns": normalized_sessions["raw_samples_ns"],
                    "excluded_samples": normalized_sessions[
                        "session_exclusions"
                    ],
                    "session_evidence_refs": normalized_sessions[
                        "evidence_refs"
                    ],
                    "evidence_refs": [
                        correctness["evidence_ref"],
                        *normalized_sessions["evidence_refs"],
                    ],
                }
            )
        direct_defect = _direct_defect_evidence(
            direct_defect_value,
            candidates=candidate_evaluations,
            contract=contract,
            verified_artifacts=verified_artifacts,
        )
        if direct_defect is not None:
            defective_target = next(
                candidate
                for candidate in candidate_evaluations
                if candidate["candidate_id"]
                == direct_defect["target_candidate_id"]
            )
            defective_target["eligible_for_best_of_correct"] = False
            defective_target["excluded_evidence_roles"] = [
                "best_of_correct",
                "frontier_anchor",
                "surface_winner",
                "headroom_evidence",
            ]
            if defective_target["exclusion_reason"] is None:
                defective_target["exclusion_reason"] = (
                    "execution-contract-failed"
                )
        eligible = [
            candidate
            for candidate in candidate_evaluations
            if candidate["eligible_for_best_of_correct"]
        ]
        if malformed_candidate or (not eligible and direct_defect is None):
            results.append(
                {
                    **result_prefix,
                    "status": "insufficient_evidence",
                    "reason_code": (
                        "invalid-candidate-evidence"
                        if malformed_candidate
                        else "no-correct-eligible-candidate"
                    ),
                }
            )
            continue
        winner = (
            min(
                eligible,
                key=lambda candidate: (
                    float(candidate["aggregate_latency_ns"]),
                    candidate["candidate_id"],
                ),
            )
            if eligible
            else None
        )
        complete_probe = {
            **result_prefix,
            "status": "complete",
            "locked_contract": contract,
            "evaluation_order": [
                "lock-exact-shape-contract",
                "validate-correctness",
                "select-best-of-correct",
            ],
            "candidate_evaluations": candidate_evaluations,
            "measurement_lanes": measurement_lanes,
            "counterexamples": counterexamples,
        }
        if winner is not None:
            complete_probe["best_of_correct"] = {
                "candidate_id": winner["candidate_id"],
                "aggregate_latency_ns": winner["aggregate_latency_ns"],
                "session_ids": winner["session_ids"],
            }
        if "integration_overhead_evidence" in probe:
            complete_probe["integration_overhead_evidence"] = probe[
                "integration_overhead_evidence"
            ]
            complete_probe["integration_operator_frontier_verified"] = (
                integration_frontier_verified
            )
            complete_probe["integration_evidence_artifacts_verified"] = (
                integration_artifacts_verified
            )
        if frontier_shift_evidence is not None:
            complete_probe["frontier_shift_evidence"] = (
                frontier_shift_evidence
            )
        if direct_defect is not None:
            complete_probe["direct_defect_evidence"] = direct_defect
        elif direct_defect_value is not None:
            complete_probe["direct_defect_rejection"] = {
                "reason_code": "invalid-or-nonqualifying-direct-evidence",
                "evidence_refs": sorted(_artifact_refs(direct_defect_value)),
            }
        results.append(complete_probe)
    represented_paths = {
        result["stable_path"]
        for result in results
        if _nonempty_string(result.get("stable_path"))
    }
    for stable_path in sorted(triggered_paths - represented_paths):
        results.append(
            {
                "probe_id": f"probe-request:{stable_path}",
                "stable_path": stable_path,
                "status": "requested",
                "reason_code": "exact-shape-probe-evidence-not-provided",
                "required_lock_fields": [
                    "semantic",
                    "shape",
                    "dtype",
                    "layout",
                    "strides",
                    "alignment_bytes",
                    "threads",
                    "execution_domain",
                    "cohort_id",
                    "cohort_identity",
                    "environment",
                    "correctness_policy",
                    "candidate_ids",
                    "completion_boundary",
                    "measurement_lanes",
                ],
                "evidence_refs": [],
            }
        )
    return results


def _verdict_policy(document: dict[str, Any]) -> dict[str, Any]:
    policy = document.get("verdict_policy")
    if policy is None:
        return {
            "status": "unknown",
            "reason_code": "verdict-policy-missing",
        }
    if (
        not isinstance(policy, dict)
        or not all(
            _resolved_identity_string(policy.get(key))
            for key in (
                "policy_id",
                "version",
                "scope",
                "change_reason",
                "revalidation",
            )
        )
        or not _exact_version_text(policy.get("version"))
        or not isinstance(policy.get("minimum_independent_sessions"), int)
        or isinstance(policy["minimum_independent_sessions"], bool)
        or policy["minimum_independent_sessions"] < 3
        or policy.get("suspected_regression_gate") != "undefined"
    ):
        return {
            "status": "unknown",
            "reason_code": "verdict-policy-invalid",
        }
    return {
        "status": "valid",
        **{
            key: policy[key]
            for key in (
                "policy_id",
                "version",
                "scope",
                "change_reason",
                "revalidation",
            )
        },
        "minimum_independent_sessions": policy[
            "minimum_independent_sessions"
        ],
        "suspected_regression_gate": "undefined",
    }


def _integration_measurement(
    value: object,
    *,
    expected_lane_id: str,
    expected_cohort_id: str,
    minimum_sessions: int,
    correctness_policy: dict[str, Any],
    allow_zero_samples: bool = False,
) -> dict[str, Any] | None:
    correctness = value.get("correctness") if isinstance(value, dict) else None
    correctness_passed = (
        _raw_correctness_passed(
            correctness.get("records"), correctness.get("tolerance")
        )
        if isinstance(correctness, dict)
        else None
    )
    if (
        not isinstance(value, dict)
        or not _nonempty_string(value.get("measurement_id"))
        or value.get("lane_id") != expected_lane_id
        or not _artifact_uri(value.get("evidence_ref"))
        or not isinstance(correctness, dict)
        or correctness_passed is not True
        or correctness.get("tolerance")
        != {
            "atol": correctness_policy.get("atol"),
            "rtol": correctness_policy.get("rtol"),
        }
        or not _artifact_uri(correctness.get("evidence_ref"))
    ):
        return None
    samples_field = (
        "derived_samples_ns"
        if isinstance(value, dict) and isinstance(value.get("derivation"), dict)
        else "raw_samples_ns"
    )
    normalized = _normalize_timing_sessions(
        value.get("sessions"),
        expected_lane_id=expected_lane_id,
        expected_cohort_id=expected_cohort_id,
        minimum_sessions=minimum_sessions,
        require_authored_latency=False,
        allow_zero_samples=allow_zero_samples,
        samples_field=samples_field,
    )
    if normalized is None:
        return None
    return {
        "measurement_id": value["measurement_id"],
        "lane_id": value["lane_id"],
        **normalized,
        "correctness": {
            "passed": True,
            "record_count": len(correctness["records"]),
            "tolerance": dict(correctness["tolerance"]),
            "evidence_ref": correctness["evidence_ref"],
        },
        "session_evidence_refs": list(normalized["evidence_refs"]),
        "evidence_refs": list(
            dict.fromkeys(
                [
                    value["evidence_ref"],
                    correctness["evidence_ref"],
                    *normalized["evidence_refs"],
                ]
            )
        ),
    }


def _integration_ledger(
    value: object,
    *,
    wrapped_e2e_ns: float,
    standalone_operator_ns: float,
) -> dict[str, Any] | None:
    if (
        not isinstance(value, dict)
        or not _resolved_identity_string(value.get("ledger_id"))
        or not _exact_version_text(value.get("version"))
        or value.get("leaf_semantics") != "mutually-exclusive"
        or not _finite_number(value.get("e2e_duration_ns"))
        or not isclose(
            float(value["e2e_duration_ns"]),
            wrapped_e2e_ns,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        or not isinstance(value.get("leaves"), list)
        or not value["leaves"]
        or not isinstance(value.get("parents"), list)
        or not value["parents"]
        or not isinstance(value.get("evidence_refs"), list)
        or not value["evidence_refs"]
        or not all(_artifact_uri(ref) for ref in value["evidence_refs"])
    ):
        return None
    leaves: list[dict[str, Any]] = []
    leaf_by_id: dict[str, dict[str, Any]] = {}
    for leaf in value["leaves"]:
        if (
            not isinstance(leaf, dict)
            or not _resolved_identity_string(leaf.get("leaf_id"))
            or leaf["leaf_id"] in leaf_by_id
            or leaf.get("kind")
            not in {
                "operator",
                "copy",
                "dispatch",
                "sync",
                "profiling",
                "wait",
                "other",
            }
            or not _finite_number(leaf.get("duration_ns"))
            or float(leaf["duration_ns"]) < 0
            or not isinstance(leaf.get("evidence_refs"), list)
            or not leaf["evidence_refs"]
            or not all(_artifact_uri(ref) for ref in leaf["evidence_refs"])
        ):
            return None
        normalized = {
            "leaf_id": leaf["leaf_id"],
            "kind": leaf["kind"],
            "duration_ns": float(leaf["duration_ns"]),
            "evidence_refs": list(leaf["evidence_refs"]),
        }
        leaves.append(normalized)
        leaf_by_id[leaf["leaf_id"]] = normalized
    operator_leaves = [leaf for leaf in leaves if leaf["kind"] == "operator"]
    if (
        len(operator_leaves) != 1
        or not isclose(
            operator_leaves[0]["duration_ns"],
            standalone_operator_ns,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    ):
        return None

    parents: list[dict[str, Any]] = []
    parent_by_id: dict[str, dict[str, Any]] = {}
    assigned_leaf_ids: list[str] = []
    for parent in value["parents"]:
        child_parent_ids = (
            parent.get("child_parent_ids")
            if isinstance(parent, dict)
            else None
        )
        leaf_ids = parent.get("leaf_ids") if isinstance(parent, dict) else None
        if (
            not isinstance(parent, dict)
            or not _resolved_identity_string(parent.get("span_id"))
            or parent["span_id"] in parent_by_id
            or parent.get("kind") not in {"e2e", "module"}
            or parent.get("additive") is not False
            or not isinstance(child_parent_ids, list)
            or not all(_resolved_identity_string(item) for item in child_parent_ids)
            or not isinstance(leaf_ids, list)
            or not all(_resolved_identity_string(item) for item in leaf_ids)
        ):
            return None
        normalized_parent = {
            "span_id": parent["span_id"],
            "kind": parent["kind"],
            "additive": False,
            "child_parent_ids": list(child_parent_ids),
            "leaf_ids": list(leaf_ids),
        }
        parents.append(normalized_parent)
        parent_by_id[parent["span_id"]] = normalized_parent
        assigned_leaf_ids.extend(leaf_ids)
    if (
        len([parent for parent in parents if parent["kind"] == "e2e"]) != 1
        or any(
            child_id not in parent_by_id
            for parent in parents
            for child_id in parent["child_parent_ids"]
        )
        or len(assigned_leaf_ids) != len(set(assigned_leaf_ids))
        or set(assigned_leaf_ids) != set(leaf_by_id)
    ):
        return None
    root = next(parent for parent in parents if parent["kind"] == "e2e")
    visit_state: dict[str, int] = {}
    stack: list[tuple[str, bool]] = [(root["span_id"], False)]
    while stack:
        parent_id, expanded = stack.pop()
        state = visit_state.get(parent_id, 0)
        if expanded:
            visit_state[parent_id] = 2
            continue
        if state == 1:
            return None
        if state == 2:
            continue
        visit_state[parent_id] = 1
        stack.append((parent_id, True))
        for child_id in reversed(
            parent_by_id[parent_id]["child_parent_ids"]
        ):
            child_state = visit_state.get(child_id, 0)
            if child_state == 1:
                return None
            if child_state != 2:
                stack.append((child_id, False))
    if {
        parent_id for parent_id, state in visit_state.items() if state == 2
    } != set(parent_by_id):
        return None
    residual = value.get("residual")
    if (
        not isinstance(residual, dict)
        or not _resolved_identity_string(residual.get("residual_id"))
        or residual.get("kind") != "unattributed"
        or not _finite_number(residual.get("duration_ns"))
        or float(residual["duration_ns"]) < 0
        or not isinstance(residual.get("evidence_refs"), list)
        or not residual["evidence_refs"]
        or not all(_artifact_uri(ref) for ref in residual["evidence_refs"])
    ):
        return None
    leaf_total_ns = sum(leaf["duration_ns"] for leaf in leaves)
    reconciled_total_ns = leaf_total_ns + float(residual["duration_ns"])
    if not isclose(
        reconciled_total_ns,
        wrapped_e2e_ns,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        return None
    return {
        "status": "conserved",
        "ledger_id": value["ledger_id"],
        "version": value["version"],
        "leaf_semantics": "mutually-exclusive",
        "e2e_duration_ns": wrapped_e2e_ns,
        "leaves": leaves,
        "parents": parents,
        "leaf_total_ns": leaf_total_ns,
        "residual": {
            "residual_id": residual["residual_id"],
            "kind": "unattributed",
            "duration_ns": float(residual["duration_ns"]),
            "evidence_refs": list(residual["evidence_refs"]),
        },
        "reconciled_total_ns": reconciled_total_ns,
        "leaf_identity_conservation": {
            "unique_leaf_count": len(leaves),
            "duplicate_leaf_ids": [],
            "unassigned_leaf_ids": [],
        },
        "parent_span_total_included_ns": 0,
        "evidence_refs": list(value["evidence_refs"]),
    }


def _integration_surface_action(
    evidence: object,
    *,
    reason_code: str,
    authoritative_latency_ns: float | None = None,
) -> dict[str, Any] | None:
    frontier = (
        evidence.get("operator_frontier")
        if isinstance(evidence, dict)
        else None
    )
    surface = frontier.get("surface") if isinstance(frontier, dict) else None
    latency_ns = frontier.get("latency_ns") if isinstance(frontier, dict) else None
    if (
        not isinstance(surface, dict)
        or not _surface_id_text(surface.get("surface_id"))
        or not _surface_version_text(surface.get("version"))
        or not _finite_number(latency_ns)
        or float(latency_ns) <= 0
    ):
        return None
    preserved_latency_ns = (
        authoritative_latency_ns
        if authoritative_latency_ns is not None
        and _finite_number(authoritative_latency_ns)
        and authoritative_latency_ns > 0
        else float(latency_ns)
    )
    return {
        "action": "preserve",
        "surface": dict(surface),
        "operator_achievable_frontier_ns": {
            "before": preserved_latency_ns,
            "after": preserved_latency_ns,
        },
        "reason_code": reason_code,
    }


def _unknown_integration_surface_action(
    trigger_item: dict[str, Any],
) -> dict[str, Any]:
    latency_ns = float(trigger_item["predicted_ns"])
    return {
        "action": "preserve",
        "surface": {
            "status": "unknown",
            "reason_code": "integration-surface-identity-unverified",
        },
        "operator_achievable_frontier_ns": {
            "before": latency_ns,
            "after": latency_ns,
        },
        "reason_code": "insufficient-evidence-cannot-lower-surface",
    }


def _integration_overhead_verdict(
    *,
    stable_path: str,
    run_id: str,
    probe: dict[str, Any],
    target: dict[str, Any],
    trigger_item: dict[str, Any],
) -> dict[str, Any]:
    evidence = probe.get("integration_overhead_evidence")
    evidence_refs = (
        sorted(_artifact_refs(evidence))
        if isinstance(evidence, dict)
        else []
    )
    bundle_ref = f"run-bundle://{run_id}"
    surface_action = _unknown_integration_surface_action(trigger_item)

    def fail(gate_id: str, reason_code: str) -> dict[str, Any]:
        result = _fail_closed_performance_verdict(
            stable_path=stable_path,
            run_id=run_id,
            probe_id=probe["probe_id"],
            failed_gate_id=gate_id,
            reason_code=reason_code,
            evidence_refs=evidence_refs or probe["evidence_refs"],
            satisfied=[
                {
                    "gate_id": "diagnostic-trigger-met",
                    "evidence_refs": [bundle_ref],
                }
            ],
            not_evaluated=[
                {
                    "gate_id": "frontier-shift",
                    "reason_code": "integration-ablation-prerequisites-failed",
                    "evidence_refs": evidence_refs,
                }
            ],
        )
        result["surface_action"] = surface_action
        return result

    if (
        not isinstance(evidence, dict)
        or evidence.get("schema")
        != "groundupscale.dev/integration-overhead-evidence/v1alpha1"
        or evidence.get("stable_path") != stable_path
        or evidence.get("cohort_id") != probe["locked_contract"]["cohort_id"]
        or not isinstance(evidence.get("evidence_refs"), list)
        or not evidence["evidence_refs"]
        or not all(_artifact_uri(ref) for ref in evidence["evidence_refs"])
        or probe.get("integration_evidence_artifacts_verified") is not True
    ):
        return fail(
            "integration-evidence-valid",
            "integration-overhead-evidence-invalid",
        )
    policy = evidence.get("policy")
    if (
        not isinstance(policy, dict)
        or not all(
            _resolved_identity_string(policy.get(key))
            for key in (
                "policy_id",
                "version",
                "scope",
                "change_reason",
                "revalidation",
            )
        )
        or not _exact_version_text(policy.get("version"))
        or not isinstance(policy.get("minimum_independent_sessions"), int)
        or isinstance(policy["minimum_independent_sessions"], bool)
        or policy["minimum_independent_sessions"] < 1
        or not _finite_number(policy.get("maximum_recovery_error_fraction"))
        or not 0 <= float(policy["maximum_recovery_error_fraction"]) <= 1
    ):
        return fail(
            "integration-policy-valid",
            "integration-overhead-policy-invalid",
        )
    lanes = probe["measurement_lanes"]
    paired_lanes = evidence.get("paired_lanes")
    if paired_lanes != {
        "pair_id": lanes["baseline"]["pair_id"],
        "baseline_lane_id": lanes["baseline"]["lane_id"],
        "diagnostic_lane_id": lanes["diagnostic"]["lane_id"],
    }:
        return fail(
            "paired-baseline-diagnostic-lanes",
            "integration-lane-identity-mismatch",
        )
    frontier = evidence.get("operator_frontier")
    verified_surface_action = _integration_surface_action(
        evidence,
        reason_code="insufficient-evidence-cannot-lower-surface",
        authoritative_latency_ns=float(trigger_item["predicted_ns"]),
    )
    if (
        probe.get("integration_operator_frontier_verified") is not True
        or verified_surface_action is None
        or not isinstance(frontier, dict)
        or not _finite_number(frontier.get("combined_uncertainty_ns"))
        or float(frontier["combined_uncertainty_ns"]) < 0
        or not isinstance(frontier.get("evidence_refs"), list)
        or not frontier["evidence_refs"]
        or not all(_artifact_uri(ref) for ref in frontier["evidence_refs"])
    ):
        return fail(
            "operator-frontier-evidence-valid",
            "operator-frontier-evidence-invalid",
        )
    standalone_ns = float(target["aggregate_latency_ns"])
    frontier_ns = float(frontier["latency_ns"])
    if (
        not isclose(
            frontier_ns,
            float(trigger_item["predicted_ns"]),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        or not isclose(
            float(frontier["combined_uncertainty_ns"]),
            float(trigger_item["combined_uncertainty_ns"]),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    ):
        return fail(
            "operator-frontier-trigger-boundary",
            "operator-frontier-trigger-boundary-mismatch",
        )
    surface_action = verified_surface_action
    if abs(standalone_ns - frontier_ns) > float(
        frontier["combined_uncertainty_ns"]
    ):
        return fail(
            "standalone-operator-within-frontier-uncertainty",
            "standalone-operator-outside-frontier-uncertainty",
        )
    minimum_sessions = policy["minimum_independent_sessions"]
    wrapped_value = evidence.get("wrapped_e2e")
    wrapped_lane_id = (
        wrapped_value.get("lane_id")
        if isinstance(wrapped_value, dict)
        else None
    )
    if wrapped_lane_id not in {
        lanes["baseline"]["lane_id"],
        lanes["diagnostic"]["lane_id"],
    }:
        return fail(
            "wrapped-e2e-evidence-valid",
            "wrapped-e2e-lane-invalid",
        )
    wrapped = _integration_measurement(
        wrapped_value,
        expected_lane_id=wrapped_lane_id,
        expected_cohort_id=evidence["cohort_id"],
        minimum_sessions=minimum_sessions,
        correctness_policy=probe["locked_contract"]["correctness_policy"],
    )
    if wrapped is None:
        return fail(
            "wrapped-e2e-evidence-valid",
            "wrapped-e2e-evidence-invalid",
        )
    target_sessions = target["session_latencies_ns"]
    target_processes = target["session_process_ids"]
    if (
        set(wrapped["session_latencies_ns"]) != set(target_sessions)
        or wrapped["session_process_ids"] != target_processes
    ):
        return fail(
            "paired-baseline-sessions",
            "standalone-wrapper-session-identity-mismatch",
        )
    ablation_values = evidence.get("ablations")
    if not isinstance(ablation_values, list) or not ablation_values:
        return fail(
            "integration-ablation-present",
            "integration-ablation-missing",
        )
    ablations = []
    ablation_ids: set[str] = set()
    removed_leaf_ids: list[str] = []
    for value in ablation_values:
        measurement = _integration_measurement(
            value,
            expected_lane_id=lanes["diagnostic"]["lane_id"],
            expected_cohort_id=evidence["cohort_id"],
            minimum_sessions=minimum_sessions,
            correctness_policy=probe["locked_contract"][
                "correctness_policy"
            ],
            allow_zero_samples=True,
        )
        value_removed = (
            value.get("removed_leaf_ids")
            if isinstance(value, dict)
            else None
        )
        if (
            measurement is None
            or not isinstance(value, dict)
            or not _resolved_identity_string(value.get("ablation_id"))
            or value["ablation_id"] in ablation_ids
            or value.get("kind")
            not in {
                "copy",
                "dispatch",
                "sync",
                "profiling",
                "wait",
                "other",
            }
            or not isinstance(value_removed, list)
            or not value_removed
            or len(value_removed) != len(set(value_removed))
            or not all(_resolved_identity_string(item) for item in value_removed)
            or set(measurement["session_latencies_ns"]) != set(target_sessions)
            or measurement["session_process_ids"] != target_processes
        ):
            return fail(
                "integration-ablation-evidence-valid",
                "integration-ablation-evidence-invalid",
            )
        ablation_ids.add(value["ablation_id"])
        removed_leaf_ids.extend(value_removed)
        ablations.append(
            {
                **measurement,
                "ablation_id": value["ablation_id"],
                "kind": value["kind"],
                "removed_leaf_ids": list(value_removed),
            }
        )
    if len(removed_leaf_ids) != len(set(removed_leaf_ids)):
        return fail(
            "integration-ablation-evidence-valid",
            "integration-ablation-leaf-overlap",
        )
    wrapped_ns = wrapped["aggregate_latency_ns"]
    ledger = _integration_ledger(
        evidence.get("exclusive_ledger"),
        wrapped_e2e_ns=wrapped_ns,
        standalone_operator_ns=standalone_ns,
    )
    if ledger is None:
        return fail(
            "exclusive-ledger-conserved",
            "integration-exclusive-ledger-not-conserved",
        )
    leaf_by_id = {leaf["leaf_id"]: leaf for leaf in ledger["leaves"]}
    if any(
        leaf_by_id.get(leaf_id, {}).get("kind") != ablation["kind"]
        for ablation in ablations
        for leaf_id in ablation["removed_leaf_ids"]
    ):
        return fail(
            "counterfactual-recovers-only-declared-leaves",
            "integration-counterfactual-kind-mismatch",
        )
    counterfactual = evidence.get("counterfactual")
    if (
        not isinstance(counterfactual, dict)
        or not _resolved_identity_string(counterfactual.get("counterfactual_id"))
        or counterfactual.get("kind") != "declared-component-removal"
        or counterfactual.get("removed_leaf_ids") != removed_leaf_ids
        or any(leaf_id not in leaf_by_id for leaf_id in removed_leaf_ids)
        or any(
            leaf_by_id[leaf_id]["kind"] == "operator"
            for leaf_id in removed_leaf_ids
        )
        or not _finite_number(counterfactual.get("declared_recovered_ns"))
        or not isinstance(counterfactual.get("evidence_refs"), list)
        or not counterfactual["evidence_refs"]
        or not all(
            _artifact_uri(ref) for ref in counterfactual["evidence_refs"]
        )
    ):
        return fail(
            "counterfactual-recovers-only-declared-leaves",
            "integration-counterfactual-invalid",
        )
    recovered_ns = sum(
        leaf_by_id[leaf_id]["duration_ns"] for leaf_id in removed_leaf_ids
    )
    ablation_recovered_ns = sum(
        ablation["aggregate_latency_ns"] for ablation in ablations
    )
    if (
        not isclose(
            recovered_ns,
            float(counterfactual["declared_recovered_ns"]),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        or not isclose(
            recovered_ns,
            ablation_recovered_ns,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    ):
        return fail(
            "counterfactual-recovers-only-declared-leaves",
            "integration-counterfactual-recovery-mismatch",
        )
    measured_excess_ns = wrapped_ns - standalone_ns
    if measured_excess_ns <= 0:
        return fail(
            "wrapped-e2e-excess-positive",
            "wrapped-e2e-has-no-positive-excess",
        )
    recovery_error_fraction = abs(measured_excess_ns - recovered_ns) / max(
        measured_excess_ns, recovered_ns
    )
    if recovery_error_fraction > float(
        policy["maximum_recovery_error_fraction"]
    ):
        return fail(
            "integration-ablation-error-budget",
            "integration-ablation-error-budget-exceeded",
        )
    satisfied = [
        {
            "gate_id": "diagnostic-trigger-met",
            "evidence_refs": [bundle_ref],
        },
        {
            "gate_id": "standalone-operator-within-frontier-uncertainty",
            "evidence_refs": list(frontier["evidence_refs"]),
        },
        {
            "gate_id": "paired-baseline-diagnostic-lanes",
            "evidence_refs": [
                lanes["baseline"]["evidence_ref"],
                lanes["diagnostic"]["evidence_ref"],
            ],
        },
        {
            "gate_id": "integration-ablation-error-budget",
            "evidence_refs": [
                ref for ablation in ablations for ref in ablation["evidence_refs"]
            ],
        },
        {
            "gate_id": "exclusive-ledger-conserved",
            "evidence_refs": ledger["evidence_refs"],
        },
        {
            "gate_id": "counterfactual-recovers-only-declared-leaves",
            "evidence_refs": list(counterfactual["evidence_refs"]),
        },
        {
            "gate_id": "operator-frontier-preserved",
            "evidence_refs": list(frontier["evidence_refs"]),
        },
    ]
    maximum_recovery_error_fraction = float(
        policy["maximum_recovery_error_fraction"]
    )
    counterfactual_e2e_ns = wrapped_ns - recovered_ns
    e2e_gap_fraction = measured_excess_ns / standalone_ns
    metrics = {
        "operator_frontier_ns": frontier_ns,
        "operator_frontier_combined_uncertainty_ns": float(
            frontier["combined_uncertainty_ns"]
        ),
        "standalone_operator_ns": standalone_ns,
        "wrapped_e2e_ns": wrapped_ns,
        "measured_excess_ns": measured_excess_ns,
        "recovered_ns": recovered_ns,
        "counterfactual_e2e_ns": counterfactual_e2e_ns,
        "e2e_gap_fraction": e2e_gap_fraction,
        "recovery_error_fraction": recovery_error_fraction,
        "maximum_recovery_error_fraction": maximum_recovery_error_fraction,
    }
    frontier_ref = frontier["evidence_refs"][0]
    ledger_ref = ledger["evidence_refs"][0]
    policy_ref = evidence["evidence_refs"][0]

    def inputs(*values: tuple[str, str]) -> list[dict[str, str]]:
        return [
            {"artifact_ref": artifact_ref, "field": field}
            for artifact_ref, field in values
        ]

    def timing_inputs(
        refs: list[str], *, samples_field: str = "raw_samples_ns"
    ) -> list[dict[str, str]]:
        return [
            input_ref
            for artifact_ref in refs
            for input_ref in inputs(
                (artifact_ref, f"payload.{samples_field}"),
                (artifact_ref, "payload.excluded_samples"),
            )
        ]

    standalone_inputs = timing_inputs(target["session_evidence_refs"])
    wrapped_inputs = timing_inputs(wrapped["session_evidence_refs"])
    metric_derivations = {
        "operator_frontier_ns": {
            "formula": "operator_frontier.latency_ns",
            "inputs": inputs((frontier_ref, "latency_ns")),
        },
        "operator_frontier_combined_uncertainty_ns": {
            "formula": "operator_frontier.combined_uncertainty_ns",
            "inputs": inputs(
                (
                    frontier_ref,
                    "combined_uncertainty_ns",
                )
            ),
        },
        "standalone_operator_ns": {
            "formula": "median(session median(included raw samples))",
            "inputs": standalone_inputs,
        },
        "wrapped_e2e_ns": {
            "formula": "median(session median(included raw samples))",
            "inputs": wrapped_inputs,
        },
        "measured_excess_ns": {
            "formula": "wrapped_e2e_ns - standalone_operator_ns",
            "inputs": [*wrapped_inputs, *standalone_inputs],
        },
        "recovered_ns": {
            "formula": "sum(exclusive ledger declared removed leaf durations)",
            "inputs": inputs(
                (ledger_ref, "payload.leaves"),
                (
                    counterfactual["evidence_refs"][0],
                    "payload.removed_leaf_ids",
                ),
                *(
                    (
                        ref,
                        "payload."
                        + ablation.get("samples_field", "raw_samples_ns"),
                    )
                    for ablation in ablations
                    for ref in ablation["session_evidence_refs"]
                ),
            ),
        },
        "counterfactual_e2e_ns": {
            "formula": "wrapped_e2e_ns - recovered_ns",
            "inputs": [
                *wrapped_inputs,
                *inputs((ledger_ref, "payload.leaves")),
            ],
        },
        "e2e_gap_fraction": {
            "formula": "measured_excess_ns / standalone_operator_ns",
            "inputs": [*wrapped_inputs, *standalone_inputs],
        },
        "recovery_error_fraction": {
            "formula": (
                "abs(measured_excess_ns - recovered_ns) / "
                "max(measured_excess_ns, recovered_ns)"
            ),
            "inputs": [
                *wrapped_inputs,
                *standalone_inputs,
                *inputs((ledger_ref, "payload.leaves")),
            ],
        },
        "maximum_recovery_error_fraction": {
            "formula": "policy.maximum_recovery_error_fraction",
            "inputs": inputs(
                (
                    policy_ref,
                    "payload.policy.maximum_recovery_error_fraction",
                )
            ),
        },
    }
    return {
        "stable_path": stable_path,
        "status": "decided",
        "verdict": "integration_overhead",
        "probe_id": probe["probe_id"],
        "metrics": metrics,
        "metric_derivations": metric_derivations,
        "wrapped_e2e": wrapped,
        "ablations": ablations,
        "ledger": ledger,
        "gates": {
            "satisfied": satisfied,
            "failed": [],
            "not_evaluated": [
                {
                    "gate_id": "frontier-shift",
                    "reason_code": "integration-overhead-attributed",
                    "evidence_refs": list(frontier["evidence_refs"]),
                },
                {
                    "gate_id": "suspected-regression",
                    "reason_code": "policy-undefined",
                    "evidence_refs": evidence_refs,
                },
            ],
        },
        "surface_action": _integration_surface_action(
            evidence,
            reason_code="integration-overhead-does-not-lower-frontier",
            authoritative_latency_ns=float(trigger_item["predicted_ns"]),
        ),
        "bundle_refs": [bundle_ref, *evidence_refs],
        "counterexamples": [],
    }


def _fail_closed_performance_verdict(
    *,
    stable_path: str,
    run_id: str,
    probe_id: object,
    failed_gate_id: str,
    reason_code: str,
    evidence_refs: list[str],
    satisfied: list[dict[str, Any]] | None = None,
    not_evaluated: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    bundle_ref = f"run-bundle://{run_id}"
    bundle_refs = list(dict.fromkeys([bundle_ref, *evidence_refs]))
    unevaluated = list(not_evaluated or [])
    unevaluated.append(
        {
            "gate_id": "suspected-regression",
            "reason_code": "policy-undefined",
            "evidence_refs": [bundle_ref],
        }
    )
    return {
        "stable_path": stable_path,
        "status": "decided",
        "verdict": "insufficient_evidence",
        "probe_id": probe_id,
        "metrics": {},
        "gates": {
            "satisfied": list(satisfied or []),
            "failed": [
                {
                    "gate_id": failed_gate_id,
                    "reason_code": reason_code,
                    "evidence_refs": evidence_refs,
                }
            ],
            "not_evaluated": unevaluated,
        },
        "bundle_refs": bundle_refs,
        "counterexamples": [
            {
                "counterexample_id": failed_gate_id,
                "reason_codes": [reason_code],
                "evidence_refs": bundle_refs,
            }
        ],
    }


def _with_direct_defect_gate_context(
    result: dict[str, Any],
    *,
    probe: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    rejection = probe.get("direct_defect_rejection")
    if isinstance(rejection, dict):
        result["gates"]["failed"].append(
            {
                "gate_id": "direct-defect-evidence-qualified",
                "reason_code": rejection["reason_code"],
                "evidence_refs": rejection["evidence_refs"],
            }
        )
        reason_code = "direct-defect-prerequisites-failed"
        evidence_refs = rejection["evidence_refs"]
    else:
        reason_code = "direct-defect-evidence-not-provided"
        evidence_refs = []
    result["gates"]["not_evaluated"].insert(
        max(len(result["gates"]["not_evaluated"]) - 1, 0),
        {
            "gate_id": "confirmed-bug",
            "reason_code": reason_code,
            "evidence_refs": evidence_refs,
        },
    )
    known_candidates = {
        counterexample.get("candidate_id")
        for counterexample in result["counterexamples"]
        if isinstance(counterexample, dict)
    }
    result["counterexamples"].extend(
        {
            "candidate_id": candidate["candidate_id"],
            "reason_code": candidate["exclusion_reason"],
            "evidence_refs": candidate["evidence_refs"],
        }
        for candidate in candidates
        if candidate["exclusion_reason"] is not None
        and candidate["candidate_id"] not in known_candidates
    )
    return result


def _c3_enumerated_pool_coverage_valid(
    frontier_evidence: dict[str, Any],
    locked_candidates: list[dict[str, Any]],
) -> bool:
    manifest = frontier_evidence.get("enumerated_pool_manifest")
    proof = frontier_evidence.get("enumerated_pool_coverage_proof")
    if not isinstance(manifest, dict) or not isinstance(proof, dict):
        return False
    candidate_ids = [
        candidate["candidate_id"] for candidate in locked_candidates
    ]
    implementation_refs = [
        candidate["implementation_family"]["implementation_ref"]
        for candidate in locked_candidates
    ]
    implementation_digests = [
        candidate["implementation_family"]["implementation_sha256"]
        for candidate in locked_candidates
    ]
    candidate_implementations = sorted(
        [
            {
                "candidate_id": candidate["candidate_id"],
                "implementation_ref": candidate["implementation_family"][
                    "implementation_ref"
                ],
                "implementation_digest": candidate[
                    "implementation_family"
                ]["implementation_sha256"],
            }
            for candidate in locked_candidates
        ],
        key=lambda item: item["candidate_id"],
    )
    manifest_ref = manifest.get("evidence_ref")
    proof_ref = proof.get("evidence_ref")
    versioned_fields = (
        "version",
        "scope",
        "change_reason",
        "revalidation",
    )
    return (
        bool(locked_candidates)
        and manifest.get("schema")
        == "groundupscale.dev/enumerated-candidate-pool/v1alpha1"
        and proof.get("schema")
        == (
            "groundupscale.dev/"
            "enumerated-candidate-pool-coverage/v1alpha1"
        )
        and _resolved_identity_string(manifest.get("pool_id"))
        and _resolved_identity_string(proof.get("proof_id"))
        and all(
            _resolved_identity_string(manifest.get(field))
            and _resolved_identity_string(proof.get(field))
            for field in versioned_fields
        )
        and _exact_version_text(manifest.get("version"))
        and _exact_version_text(proof.get("version"))
        and proof.get("pool_id") == manifest.get("pool_id")
        and proof.get("pool_version") == manifest.get("version")
        and proof.get("coverage_status") == "complete"
        and isinstance(manifest.get("candidate_ids"), list)
        and all(
            _resolved_identity_string(candidate_id)
            for candidate_id in manifest["candidate_ids"]
        )
        and len(manifest["candidate_ids"])
        == len(set(manifest["candidate_ids"]))
        and set(manifest["candidate_ids"]) == set(candidate_ids)
        and isinstance(manifest.get("candidate_implementations"), list)
        and manifest["candidate_implementations"]
        == candidate_implementations
        and isinstance(proof.get("enumerated_candidate_ids"), list)
        and all(
            _resolved_identity_string(candidate_id)
            for candidate_id in proof["enumerated_candidate_ids"]
        )
        and proof["enumerated_candidate_ids"] == manifest["candidate_ids"]
        and isinstance(manifest.get("implementation_refs"), list)
        and all(
            _artifact_uri(reference)
            for reference in manifest["implementation_refs"]
        )
        and len(manifest["implementation_refs"])
        == len(set(manifest["implementation_refs"]))
        and set(manifest["implementation_refs"]) == set(implementation_refs)
        and isinstance(manifest.get("implementation_digests"), list)
        and all(
            isinstance(digest, str)
            and fullmatch(r"[0-9a-f]{64}", digest) is not None
            for digest in manifest["implementation_digests"]
        )
        and len(manifest["implementation_digests"])
        == len(set(manifest["implementation_digests"]))
        and set(manifest["implementation_digests"])
        == set(implementation_digests)
        and _artifact_uri(manifest_ref)
        and _artifact_uri(proof_ref)
        and manifest_ref != proof_ref
        and manifest_ref not in implementation_refs
        and proof_ref not in implementation_refs
    )


def _derive_frontier_shift_gates(
    frontier_evidence: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    contract: dict[str, Any],
    measurement_lanes: dict[str, Any],
    trigger_item: dict[str, Any],
    minimum_sessions: int,
) -> tuple[tuple[str, bool, str], ...]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate["eligible_for_best_of_correct"]
    ]
    family_ids = {
        candidate["implementation_family"]["family_id"]
        for candidate in eligible
    }
    family_manifest_refs = {
        candidate["implementation_family"]["manifest_ref"]
        for candidate in eligible
    }
    implementation_digests = {
        candidate["implementation_family"]["implementation_sha256"]
        for candidate in eligible
    }
    implementation_refs = {
        candidate["implementation_family"]["implementation_ref"]
        for candidate in eligible
    }
    source_identities = {
        candidate["implementation_family"]["source_identity"]
        for candidate in eligible
    }
    c2_multi_family = (
        len(family_ids) >= 2
        and len(family_manifest_refs) >= 2
        and len(implementation_digests) >= 2
        and len(implementation_refs) >= 2
        and len(source_identities) >= 2
        and all(
            len(
                {
                    candidate["implementation_family"]["version"]
                    for candidate in eligible
                    if candidate["implementation_family"]["family_id"]
                    == family_id
                }
            )
            == 1
            for family_id in family_ids
        )
    )
    coverage_level = frontier_evidence.get("candidate_coverage")
    independent_candidate_coverage = (
        coverage_level == "C2_MULTI_FAMILY" and c2_multi_family
    ) or (
        coverage_level == "C3_ENUMERATED_POOL"
        and _c3_enumerated_pool_coverage_valid(frontier_evidence, candidates)
    )
    search_session_sets = [
        set(candidate["session_ids"]) for candidate in eligible
    ]
    common_search_sessions = (
        set.intersection(*search_session_sets)
        if search_session_sets
        else set()
    )
    search_process_ids: set[int] = set()
    process_identity_matches = bool(eligible)
    for session_id in common_search_sessions:
        process_ids = {
            candidate["session_process_ids"].get(session_id)
            for candidate in eligible
        }
        if len(process_ids) != 1 or None in process_ids:
            process_identity_matches = False
            continue
        search_process_ids.update(process_ids)
    minimum_search_sessions_met = (
        bool(eligible)
        and all(sessions == common_search_sessions for sessions in search_session_sets)
        and len(common_search_sessions) >= minimum_sessions
        and len(search_process_ids) >= minimum_sessions
        and process_identity_matches
    )

    required_sessions = max(
        _FRONTIER_MINIMUM_INDEPENDENT_SESSIONS, minimum_sessions
    )
    minimum_search_sessions_met = (
        minimum_search_sessions_met
        and len(common_search_sessions) >= required_sessions
        and len(search_process_ids) >= required_sessions
    )

    holdout = frontier_evidence["holdout"]
    baseline_lane_id = measurement_lanes["baseline"]["lane_id"]
    selection_session_ids = holdout["selection_session_ids"]
    holdout_sessions = holdout["sessions"]
    holdout_session_ids: list[str] = []
    holdout_process_ids: list[int] = []
    holdout_cohorts: list[str] = []
    holdout_by_id: dict[str, dict[str, Any]] = {}
    holdout_sessions_valid = True
    for session in holdout_sessions:
        if (
            not isinstance(session, dict)
            or not _nonempty_string(session.get("session_id"))
            or not isinstance(session.get("process_id"), int)
            or isinstance(session["process_id"], bool)
            or session["process_id"] <= 0
            or not _nonempty_string(session.get("lane_id"))
            or not _nonempty_string(session.get("cohort_id"))
            or not _nonempty_string(session.get("evidence_ref"))
        ):
            holdout_sessions_valid = False
            break
        holdout_session_ids.append(session["session_id"])
        holdout_process_ids.append(session["process_id"])
        holdout_cohorts.append(session["cohort_id"])
        holdout_by_id[session["session_id"]] = session
    independent_holdout = (
        holdout_sessions_valid
        and all(
            session.get("lane_id") == baseline_lane_id
            for session in holdout_sessions
        )
        and set(selection_session_ids) == common_search_sessions
        and set(selection_session_ids).isdisjoint(holdout_session_ids)
        and search_process_ids.isdisjoint(holdout_process_ids)
        and len(holdout_session_ids) >= required_sessions
        and len(set(holdout_session_ids)) == len(holdout_session_ids)
        and len(holdout_by_id) == len(holdout_session_ids)
        and len(set(holdout_process_ids)) == len(holdout_process_ids)
        and len(holdout_process_ids) >= required_sessions
    )
    same_cohort = independent_holdout and all(
        cohort_id == contract["cohort_id"] for cohort_id in holdout_cohorts
    )

    eligible_by_id = {
        candidate["candidate_id"]: candidate for candidate in eligible
    }
    eligible_ids = set(eligible_by_id)
    candidate_results = holdout["candidate_results"]
    result_ids = {
        result.get("candidate_id")
        for result in candidate_results
        if isinstance(result, dict)
    }
    holdout_session_set = set(holdout_session_ids)
    surface_reference = frontier_evidence["old_surface_reference"]
    surface_reference_valid = (
        surface_reference["execution_domain"]
        == contract["execution_domain"]
        and surface_reference["cohort_id"] == contract["cohort_id"]
        and {
            "surface_id": surface_reference["surface_id"],
            "version": surface_reference["version"],
        }
        == frontier_evidence["surface"]
        and isclose(
            float(surface_reference["predicted_ns"]),
            float(trigger_item["predicted_ns"]),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    )
    combined_surface_uncertainty_ns = hypot(
        *[
            float(record["standard_uncertainty_ns"])
            for record in frontier_evidence[
                "old_surface_uncertainty_records"
            ]
        ]
    )
    surface_reference_valid = surface_reference_valid and isclose(
        combined_surface_uncertainty_ns,
        float(trigger_item["combined_uncertainty_ns"]),
        rel_tol=1e-12,
        abs_tol=1e-9,
    )
    band_upper_ns = (
        float(surface_reference["predicted_ns"])
        + combined_surface_uncertainty_ns
    )

    def candidate_result_below_band(result: object) -> bool:
        if not isinstance(result, dict):
            return False
        candidate = eligible_by_id.get(result.get("candidate_id"))
        records = result.get("correctness_records")
        tolerance = result.get("correctness_tolerance")
        sessions = result.get("sessions")
        if (
            candidate is None
            or not isinstance(records, list)
            or not records
            or not isinstance(tolerance, dict)
            or not _finite_number(tolerance.get("atol"))
            or not _finite_number(tolerance.get("rtol"))
            or float(tolerance["atol"]) < 0
            or float(tolerance["rtol"]) < 0
            or tolerance
            != {
                "atol": contract["correctness_policy"]["atol"],
                "rtol": contract["correctness_policy"]["rtol"],
            }
            or result.get("correctness_evidence_ref")
            != candidate["correctness"]["evidence_ref"]
            or not isinstance(sessions, list)
            or len(sessions) != len(holdout_session_set)
        ):
            return False
        correctness_passed = all(
            isinstance(record, dict)
            and _finite_number(record.get("expected"))
            and _finite_number(record.get("observed"))
            and abs(float(record["observed"]) - float(record["expected"]))
            <= float(tolerance["atol"])
            + float(tolerance["rtol"]) * abs(float(record["expected"]))
            for record in records
        )
        normalized_timing = _normalize_timing_sessions(
            sessions,
            expected_lane_id=baseline_lane_id,
            expected_cohort_id=contract["cohort_id"],
            minimum_sessions=required_sessions,
            require_authored_latency=False,
        )
        records_match_holdout = (
            isinstance(normalized_timing, dict)
            and set(normalized_timing["session_ids"]) == holdout_session_set
            and all(
                normalized_timing["session_process_ids"].get(session_id)
                == holdout_by_id[session_id]["process_id"]
                and holdout_by_id[session_id]["lane_id"]
                == baseline_lane_id
                and holdout_by_id[session_id]["cohort_id"]
                == contract["cohort_id"]
                for session_id in normalized_timing["session_ids"]
            )
        )
        session_latencies = (
            list(normalized_timing["session_latencies_ns"].values())
            if isinstance(normalized_timing, dict)
            else []
        )
        return (
            correctness_passed
            and records_match_holdout
            and all(latency > band_upper_ns for latency in session_latencies)
            and float(median(session_latencies)) > band_upper_ns
        )

    all_candidates_below_band = (
        independent_holdout
        and same_cohort
        and surface_reference_valid
        and result_ids == eligible_ids
        and len(candidate_results) == len(eligible_ids)
        and all(
            candidate_result_below_band(result)
            for result in candidate_results
        )
    )

    neighbourhood = frontier_evidence["neighbourhood"]
    regime_id = neighbourhood["regime_id"]
    qualification_policy = neighbourhood["qualification_policy"]
    domain_shape = contract.get("execution_domain", {}).get("shape", {})

    def neighbourhood_record_valid(record: object) -> bool:
        if not isinstance(record, dict):
            return False
        correctness_passed = _raw_correctness_passed(
            record.get("correctness_records"),
            record.get("correctness_tolerance"),
        )
        sessions = record.get("holdout_sessions")
        uncertainty_records = record.get("uncertainty_records")
        if not isinstance(sessions, list):
            return False
        if (
            not isinstance(uncertainty_records, list)
            or len(uncertainty_records) != 3
            or {
                item.get("component_id")
                for item in uncertainty_records
                if isinstance(item, dict)
            }
            != {"anchor", "interpolation", "instrumentation"}
            or not all(
                isinstance(item, dict)
                and _finite_number(item.get("standard_uncertainty_ns"))
                and float(item["standard_uncertainty_ns"]) >= 0
                and _artifact_uri(item.get("evidence_ref"))
                for item in uncertainty_records
            )
        ):
            return False
        if record.get("correctness_tolerance") != {
            "atol": contract["correctness_policy"]["atol"],
            "rtol": contract["correctness_policy"]["rtol"],
        }:
            return False
        normalized_timing = _normalize_timing_sessions(
            sessions,
            expected_lane_id=baseline_lane_id,
            expected_cohort_id=contract["cohort_id"],
            minimum_sessions=required_sessions,
            require_authored_latency=False,
        )
        if not isinstance(normalized_timing, dict):
            return False
        session_ids = normalized_timing["session_ids"]
        process_ids = list(
            normalized_timing["session_process_ids"].values()
        )
        session_latencies = list(
            normalized_timing["session_latencies_ns"].values()
        )
        return (
            correctness_passed is True
            and record.get("observation_validity") == "QUALIFIED"
            and record.get("frontier_role") == "ACTIVE"
            and record.get("surface") == frontier_evidence["surface"]
            and record.get("cohort_id") == contract["cohort_id"]
            and record.get("regime_id") == regime_id
            and isinstance(record.get("shape"), dict)
            and set(record["shape"]) == set(domain_shape)
            and all(
                isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension > 0
                for dimension in record["shape"].values()
            )
            and _finite_number(record.get("predicted_ns"))
            and float(record["predicted_ns"]) > 0
            and _finite_number(record.get("observed_ns"))
            and float(record["observed_ns"]) > 0
            and len(session_ids) >= required_sessions
            and len(session_ids) == len(set(session_ids))
            and len(process_ids) == len(set(process_ids))
            and search_process_ids.isdisjoint(process_ids)
            and isclose(
                float(record["observed_ns"]),
                float(median(session_latencies)),
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            and _nonempty_string(record.get("evidence_ref"))
        )

    def record_stable(record: object) -> bool:
        combined_uncertainty_ns = (
            hypot(
                *[
                    float(item["standard_uncertainty_ns"])
                    for item in record["uncertainty_records"]
                ]
            )
            if isinstance(record, dict)
            and isinstance(record.get("uncertainty_records"), list)
            else -1.0
        )
        return (
            neighbourhood_record_valid(record)
            and abs(float(record["observed_ns"]) - float(record["predicted_ns"]))
            <= combined_uncertainty_ns
        )

    expected_local_shapes: set[tuple[tuple[str, int], ...]] = set()
    local_shape_radius = qualification_policy["local_shape_radius"]
    if isinstance(domain_shape, dict) and all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > local_shape_radius
        for value in domain_shape.values()
    ):
        for name in sorted(domain_shape):
            for delta in (-local_shape_radius, local_shape_radius):
                shape = dict(domain_shape)
                shape[name] += delta
                expected_local_shapes.add(tuple(sorted(shape.items())))
    anchors = neighbourhood["anchor_records"]
    anchor_shapes = {
        tuple(sorted(record["shape"].items()))
        for record in anchors
        if record_stable(record)
    }
    stable_anchors = (
        len(anchors)
        >= qualification_policy["minimum_stable_anchor_records"]
        and len(anchor_shapes) == len(anchors)
        and anchor_shapes.issubset(expected_local_shapes)
    )
    local_probes = neighbourhood["local_probe_records"]
    actual_local_shapes = {
        tuple(sorted(record["shape"].items()))
        for record in local_probes
        if record_stable(record)
    }
    dense_local_shapes = (
        bool(expected_local_shapes)
        and len(actual_local_shapes) == len(local_probes)
        and expected_local_shapes.issubset(actual_local_shapes)
    )
    refit_records = neighbourhood["refit_records"]
    refit_shapes = {
        tuple(sorted(record["shape"].items()))
        for record in refit_records
        if record_stable(record)
    }
    same_regime_refit = (
        len(refit_records) >= qualification_policy["minimum_refit_records"]
        and len(refit_shapes) == len(refit_records)
        and refit_shapes.issubset(expected_local_shapes)
    )
    neighbourhood_met = (
        stable_anchors and dense_local_shapes and same_regime_refit
    )
    return (
        (
            "frontier-shift-independent-candidate-coverage",
            independent_candidate_coverage,
            "c2-or-c3-independent-candidate-families-missing",
        ),
        (
            "frontier-shift-independent-holdout",
            independent_holdout,
            "independent-holdout-missing",
        ),
        (
            "frontier-shift-minimum-independent-sessions",
            minimum_search_sessions_met,
            "minimum-independent-sessions-not-met",
        ),
        (
            "frontier-shift-same-hardware-validity-cohort",
            same_cohort,
            "hardware-validity-cohort-unstable",
        ),
        (
            "frontier-shift-all-eligible-candidates-below-surface-band",
            all_candidates_below_band,
            "eligible-candidates-not-confirmed-below-surface-band",
        ),
        (
            "frontier-shift-stable-neighbouring-anchors",
            stable_anchors,
            "stable-neighbouring-anchors-missing",
        ),
        (
            "frontier-shift-local-shape-disambiguation",
            dense_local_shapes and same_regime_refit,
            "local-shape-disambiguation-incomplete",
        ),
        (
            "frontier-shift-validated-neighbourhood",
            neighbourhood_met,
            "neighbourhood-regime-not-validated",
        ),
    )


def _distinct_candidate_implementation(
    target: dict[str, Any], alternative: dict[str, Any]
) -> bool:
    target_implementation = target["implementation_family"]
    alternative_implementation = alternative["implementation_family"]
    return all(
        target_implementation[key] != alternative_implementation[key]
        for key in (
            "implementation_ref",
            "implementation_sha256",
            "source_identity",
        )
    )


def _reproducibly_faster_alternative(
    target: dict[str, Any],
    alternative: dict[str, Any],
    *,
    minimum_sessions: int,
    best_candidate_id: str,
) -> bool:
    target_sessions = target["session_latencies_ns"]
    alternative_sessions = alternative["session_latencies_ns"]
    target_processes = target["session_process_ids"]
    alternative_processes = alternative["session_process_ids"]
    common_session_ids = set(target_sessions) & set(alternative_sessions)
    return (
        _distinct_candidate_implementation(target, alternative)
        and set(target_sessions) == set(alternative_sessions)
        and len(common_session_ids) >= minimum_sessions
        and len(
            {target_processes[session_id] for session_id in common_session_ids}
        )
        >= minimum_sessions
        and all(
            target_processes.get(session_id)
            == alternative_processes.get(session_id)
            for session_id in common_session_ids
        )
        and all(
            float(alternative_sessions[session_id])
            < float(target_sessions[session_id])
            for session_id in common_session_ids
        )
        and best_candidate_id == alternative["candidate_id"]
    )


def _performance_diagnosis_verdicts(
    document: dict[str, Any],
    *,
    run_id: str,
    trigger: dict[str, Any] | None,
    probes: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if trigger is None:
        return None, []
    policy = _verdict_policy(document)
    triggered_paths = (
        {
            item["stable_path"]
            for item in trigger.get("triggered", [])
            if isinstance(item, dict)
            and _nonempty_string(item.get("stable_path"))
        }
        if isinstance(trigger, dict)
        else set()
    )
    trigger_items_by_path = {
        item["stable_path"]: item
        for item in trigger.get("triggered", [])
        if isinstance(item, dict)
        and _nonempty_string(item.get("stable_path"))
    }
    probe_by_path = {
        probe["stable_path"]: probe
        for probe in probes
        if _nonempty_string(probe.get("stable_path"))
    }
    verdicts = []
    for stable_path in sorted(triggered_paths):
        probe = probe_by_path.get(stable_path)
        bundle_ref = f"run-bundle://{run_id}"
        if policy["status"] != "valid":
            verdicts.append(
                _fail_closed_performance_verdict(
                    stable_path=stable_path,
                    run_id=run_id,
                    probe_id=(
                        probe.get("probe_id")
                        if isinstance(probe, dict)
                        else None
                    ),
                    failed_gate_id="verdict-policy-valid",
                    reason_code=policy["reason_code"],
                    evidence_refs=[bundle_ref],
                )
            )
            continue
        if not isinstance(probe, dict) or probe.get("status") != "complete":
            reason_code = (
                probe.get("reason_code", "exact-shape-probe-evidence-missing")
                if isinstance(probe, dict)
                else "exact-shape-probe-evidence-missing"
            )
            probe_refs = (
                probe.get("evidence_refs", [])
                if isinstance(probe, dict)
                else []
            )
            verdicts.append(
                _fail_closed_performance_verdict(
                    stable_path=stable_path,
                    run_id=run_id,
                    probe_id=(
                        probe.get("probe_id")
                        if isinstance(probe, dict)
                        else None
                    ),
                    failed_gate_id="exact-shape-probe-complete",
                    reason_code=reason_code,
                    evidence_refs=probe_refs,
                    satisfied=[
                        {
                            "gate_id": "diagnostic-trigger-met",
                            "evidence_refs": [bundle_ref],
                        }
                    ],
                    not_evaluated=[
                        {
                            "gate_id": "correctness-before-best-of-correct",
                            "reason_code": "probe-incomplete",
                            "evidence_refs": probe_refs,
                        },
                        {
                            "gate_id": "frontier-shift",
                            "reason_code": "probe-incomplete",
                            "evidence_refs": probe_refs,
                        },
                    ],
                )
            )
            continue
        candidates = probe["candidate_evaluations"]
        targets = [
            candidate
            for candidate in candidates
            if candidate["role"] == "target"
            and candidate["eligible_for_best_of_correct"]
        ]
        alternatives = [
            candidate
            for candidate in candidates
            if candidate["role"] == "alternative"
            and candidate["eligible_for_best_of_correct"]
        ]
        direct_defect = probe.get("direct_defect_evidence")
        if isinstance(direct_defect, dict):
            target = next(
                candidate
                for candidate in candidates
                if candidate["candidate_id"]
                == direct_defect["target_candidate_id"]
            )
            direct_refs = direct_defect["evidence_refs"]
            verdicts.append(
                {
                    "stable_path": stable_path,
                    "status": "decided",
                    "verdict": "confirmed_bug",
                    "probe_id": probe["probe_id"],
                    "direct_defect_evidence": direct_defect,
                    "gates": {
                        "satisfied": [
                            {
                                "gate_id": "diagnostic-trigger-met",
                                "evidence_refs": [bundle_ref],
                            },
                            {
                                "gate_id": "exact-shape-contract-locked",
                                "evidence_refs": probe["evidence_refs"],
                            },
                            {
                                "gate_id": _DIRECT_DEFECT_GATE_IDS[
                                    direct_defect["defect_kind"]
                                ],
                                "evidence_refs": direct_refs,
                            },
                            {
                                "gate_id": "defect-reproduced-independently",
                                "evidence_refs": [
                                    repetition["evidence_ref"]
                                    for repetition in direct_defect[
                                        "repetitions"
                                    ]
                                ],
                            },
                            {
                                "gate_id": (
                                    "defective-target-excluded-from-"
                                    "best-of-correct"
                                ),
                                "evidence_refs": target["evidence_refs"],
                            },
                        ],
                        "failed": [],
                        "not_evaluated": [
                            {
                                "gate_id": "implementation-headroom",
                                "reason_code": "confirmed-direct-defect",
                                "evidence_refs": direct_refs,
                            },
                            {
                                "gate_id": "integration-overhead",
                                "reason_code": "confirmed-direct-defect",
                                "evidence_refs": direct_refs,
                            },
                            {
                                "gate_id": "frontier-shift",
                                "reason_code": "confirmed-direct-defect",
                                "evidence_refs": direct_refs,
                            },
                            {
                                "gate_id": "suspected-regression",
                                "reason_code": "policy-undefined",
                                "evidence_refs": [bundle_ref],
                            },
                        ],
                    },
                    "bundle_refs": list(
                        dict.fromkeys(
                            [
                                bundle_ref,
                                *probe["evidence_refs"],
                                *direct_refs,
                            ]
                        )
                    ),
                    "counterexamples": list(
                        probe.get("counterexamples", [])
                    )
                    + [
                        {
                            "candidate_id": candidate["candidate_id"],
                            "reason_code": candidate["exclusion_reason"],
                            "evidence_refs": candidate["evidence_refs"],
                        }
                        for candidate in candidates
                        if candidate["exclusion_reason"] is not None
                    ],
                }
            )
            continue
        if len(targets) != 1:
            verdicts.append(
                _with_direct_defect_gate_context(
                    _fail_closed_performance_verdict(
                        stable_path=stable_path,
                        run_id=run_id,
                        probe_id=probe["probe_id"],
                        failed_gate_id="single-correct-eligible-target",
                        reason_code=(
                            "expected-exactly-one-correct-eligible-target"
                        ),
                        evidence_refs=probe["evidence_refs"],
                    ),
                    probe=probe,
                    candidates=candidates,
                )
            )
            continue
        target = targets[0]
        competing_alternatives = [
            alternative
            for alternative in alternatives
            if _reproducibly_faster_alternative(
                target,
                alternative,
                minimum_sessions=policy["minimum_independent_sessions"],
                best_candidate_id=probe["best_of_correct"]["candidate_id"],
            )
        ]
        frontier_shift_evidence = probe.get("frontier_shift_evidence")
        competing_frontier_shift = (
            isinstance(frontier_shift_evidence, dict)
            and all(
                passed
                for _, passed, _ in _derive_frontier_shift_gates(
                    frontier_shift_evidence,
                    candidates=candidates,
                    contract=probe["locked_contract"],
                    measurement_lanes=probe["measurement_lanes"],
                    trigger_item=trigger_items_by_path[stable_path],
                    minimum_sessions=policy[
                        "minimum_independent_sessions"
                    ],
                )
            )
        )
        integration_result = (
            _integration_overhead_verdict(
                stable_path=stable_path,
                run_id=run_id,
                probe=probe,
                target=target,
                trigger_item=trigger_items_by_path[stable_path],
            )
            if "integration_overhead_evidence" in probe
            else None
        )
        integration_satisfied = (
            isinstance(integration_result, dict)
            and integration_result.get("verdict") == "integration_overhead"
        )
        if integration_satisfied and (
            competing_alternatives or competing_frontier_shift
        ):
            trigger_item = trigger_items_by_path[stable_path]
            evidence = probe["integration_overhead_evidence"]
            surface_action = _unknown_integration_surface_action(trigger_item)
            if (
                isinstance(evidence, dict)
                and evidence.get("schema")
                == (
                    "groundupscale.dev/"
                    "integration-overhead-evidence/v1alpha1"
                )
                and evidence.get("stable_path") == stable_path
                and evidence.get("cohort_id")
                == probe["locked_contract"]["cohort_id"]
                and probe.get("integration_operator_frontier_verified")
                is True
                and probe.get("integration_evidence_artifacts_verified")
                is True
            ):
                verified_action = _integration_surface_action(
                    evidence,
                    reason_code=(
                        "insufficient-evidence-cannot-lower-surface"
                    ),
                    authoritative_latency_ns=float(
                        trigger_item["predicted_ns"]
                    ),
                )
                if verified_action is not None:
                    surface_action = verified_action
            precedence_refs = sorted(
                _artifact_refs(
                    {
                        "integration_overhead_evidence": evidence,
                        "frontier_shift_evidence": probe.get(
                            "frontier_shift_evidence"
                        ),
                        "alternatives": competing_alternatives,
                    }
                )
            )
            result = _fail_closed_performance_verdict(
                stable_path=stable_path,
                run_id=run_id,
                probe_id=probe["probe_id"],
                failed_gate_id="multi-verdict-precedence",
                reason_code="multi-verdict-precedence-undefined",
                evidence_refs=precedence_refs or probe["evidence_refs"],
                satisfied=[
                    {
                        "gate_id": "diagnostic-trigger-met",
                        "evidence_refs": [bundle_ref],
                    },
                    {
                        "gate_id": "exact-shape-probe-complete",
                        "evidence_refs": probe["evidence_refs"],
                    },
                ],
                not_evaluated=[
                    {
                        "gate_id": "integration-overhead",
                        "reason_code": "multi-verdict-precedence-undefined",
                        "evidence_refs": precedence_refs,
                    },
                    {
                        "gate_id": "frontier-shift",
                        "reason_code": "multi-verdict-precedence-undefined",
                        "evidence_refs": precedence_refs,
                    },
                ],
            )
            result["surface_action"] = surface_action
            verdicts.append(result)
            continue
        if competing_frontier_shift and competing_alternatives:
            precedence_refs = sorted(
                _artifact_refs(
                    {
                        "frontier_shift_evidence": frontier_shift_evidence,
                        "alternatives": competing_alternatives,
                    }
                )
            )
            result = _fail_closed_performance_verdict(
                stable_path=stable_path,
                run_id=run_id,
                probe_id=probe["probe_id"],
                failed_gate_id="multi-verdict-precedence",
                reason_code="multi-verdict-precedence-undefined",
                evidence_refs=precedence_refs or probe["evidence_refs"],
                satisfied=[
                    {
                        "gate_id": "diagnostic-trigger-met",
                        "evidence_refs": [bundle_ref],
                    },
                    {
                        "gate_id": "exact-shape-probe-complete",
                        "evidence_refs": probe["evidence_refs"],
                    },
                ],
                not_evaluated=[
                    {
                        "gate_id": "implementation-headroom",
                        "reason_code": "multi-verdict-precedence-undefined",
                        "evidence_refs": precedence_refs,
                    },
                    {
                        "gate_id": "frontier-shift",
                        "reason_code": "multi-verdict-precedence-undefined",
                        "evidence_refs": precedence_refs,
                    },
                ],
            )
            result["surface_action"] = {
                "action": "preserve",
                "surface": frontier_shift_evidence["surface"],
                "reason_code": "multi-verdict-precedence-undefined",
            }
            verdicts.append(result)
            continue
        if competing_frontier_shift:
            frontier_evidence_refs = list(
                dict.fromkeys(
                    [
                        *frontier_shift_evidence["evidence_refs"],
                        frontier_shift_evidence["holdout"]["evidence_ref"],
                        *sorted(_artifact_refs(frontier_shift_evidence)),
                    ]
                )
            )
            frontier_gates = _derive_frontier_shift_gates(
                frontier_shift_evidence,
                candidates=candidates,
                contract=probe["locked_contract"],
                measurement_lanes=probe["measurement_lanes"],
                trigger_item=trigger_items_by_path[stable_path],
                minimum_sessions=policy["minimum_independent_sessions"],
            )
            verdicts.append(
                {
                    "stable_path": stable_path,
                    "status": "decided",
                    "verdict": "frontier_shift",
                    "probe_id": probe["probe_id"],
                    "gates": {
                        "satisfied": [
                            {
                                "gate_id": "diagnostic-trigger-met",
                                "evidence_refs": [bundle_ref],
                            },
                            {
                                "gate_id": "exact-shape-probe-complete",
                                "evidence_refs": probe["evidence_refs"],
                            },
                            *[
                                {
                                    "gate_id": gate_id,
                                    "evidence_refs": frontier_evidence_refs,
                                }
                                for gate_id, passed, _ in frontier_gates
                                if passed
                            ],
                        ],
                        "failed": [],
                        "not_evaluated": [
                            {
                                "gate_id": "confirmed-bug",
                                "reason_code": (
                                    "direct-defect-evidence-not-provided"
                                ),
                                "evidence_refs": [],
                            },
                            {
                                "gate_id": "suspected-regression",
                                "reason_code": "policy-undefined",
                                "evidence_refs": frontier_evidence_refs,
                            },
                        ],
                    },
                    "bundle_refs": [
                        bundle_ref,
                        *probe["evidence_refs"],
                    ],
                    "counterexamples": list(
                        probe.get("counterexamples", [])
                    ),
                    "surface_action": {
                        "action": "create_version",
                        "surface": frontier_shift_evidence["surface"],
                        "reason_code": "qualified-frontier-shift",
                    },
                }
            )
            continue
        if integration_result is not None and (
            integration_satisfied or not competing_alternatives
        ):
            verdicts.append(integration_result)
            continue
        if not alternatives:
            verdicts.append(
                _with_direct_defect_gate_context(
                    _fail_closed_performance_verdict(
                        stable_path=stable_path,
                        run_id=run_id,
                        probe_id=probe["probe_id"],
                        failed_gate_id=(
                            "correct-eligible-alternative-present"
                        ),
                        reason_code="no-correct-eligible-alternative",
                        evidence_refs=probe["evidence_refs"],
                    ),
                    probe=probe,
                    candidates=candidates,
                )
            )
            continue
        alternative = min(
            alternatives,
            key=lambda candidate: (
                float(candidate["aggregate_latency_ns"]),
                candidate["candidate_id"],
            ),
        )
        target_implementation = target["implementation_family"]
        alternative_implementation = alternative["implementation_family"]
        distinct_implementation = _distinct_candidate_implementation(
            target,
            alternative,
        )
        if not distinct_implementation:
            implementation_refs = list(
                dict.fromkeys(
                    [
                        *probe["evidence_refs"],
                        target_implementation["manifest_ref"],
                        target_implementation["implementation_ref"],
                        alternative_implementation["manifest_ref"],
                        alternative_implementation["implementation_ref"],
                    ]
                )
            )
            verdicts.append(
                _fail_closed_performance_verdict(
                    stable_path=stable_path,
                    run_id=run_id,
                    probe_id=probe["probe_id"],
                    failed_gate_id=(
                        "distinct-target-alternative-implementation"
                    ),
                    reason_code=(
                        "target-alternative-implementation-identity-collides"
                    ),
                    evidence_refs=implementation_refs,
                    satisfied=[
                        {
                            "gate_id": "diagnostic-trigger-met",
                            "evidence_refs": [bundle_ref],
                        },
                        {
                            "gate_id": "exact-shape-contract-locked",
                            "evidence_refs": probe["evidence_refs"],
                        },
                        {
                            "gate_id": "correctness-before-best-of-correct",
                            "evidence_refs": [
                                *target["evidence_refs"],
                                *alternative["evidence_refs"],
                            ],
                        },
                    ],
                    not_evaluated=[
                        {
                            "gate_id": "reproducible-faster-alternative",
                            "reason_code": "implementation-identity-collides",
                            "evidence_refs": implementation_refs,
                        },
                        {
                            "gate_id": "frontier-shift",
                            "reason_code": "implementation-identity-collides",
                            "evidence_refs": implementation_refs,
                        },
                    ],
                )
            )
            continue
        target_sessions = target["session_latencies_ns"]
        alternative_sessions = alternative["session_latencies_ns"]
        target_processes = target["session_process_ids"]
        alternative_processes = alternative["session_process_ids"]
        common_session_ids = sorted(
            set(target_sessions) & set(alternative_sessions)
        )
        independent_session_count = len(common_session_ids)
        process_identity_matches = all(
            target_processes.get(session_id)
            == alternative_processes.get(session_id)
            for session_id in common_session_ids
        )
        independent_process_count = len(
            {
                target_processes[session_id]
                for session_id in common_session_ids
            }
        )
        faster_in_every_session = (
            set(target_sessions) == set(alternative_sessions)
            and all(
                float(alternative_sessions[session_id])
                < float(target_sessions[session_id])
                for session_id in common_session_ids
            )
        )
        reproducible_faster = _reproducibly_faster_alternative(
            target,
            alternative,
            minimum_sessions=policy["minimum_independent_sessions"],
            best_candidate_id=probe["best_of_correct"]["candidate_id"],
        )
        evidence_refs = list(
            dict.fromkeys(
                [
                    *probe["evidence_refs"],
                    *target["evidence_refs"],
                    *alternative["evidence_refs"],
                ]
            )
        )
        contract = probe["locked_contract"]
        lanes = probe["measurement_lanes"]
        cohort_evidence_refs = [
            contract["cohort_identity"]["evidence_ref"],
            *target["evidence_refs"],
            *alternative["evidence_refs"],
        ]
        satisfied = [
            {
                "gate_id": "diagnostic-trigger-met",
                "evidence_refs": evidence_refs,
            },
            {
                "gate_id": "exact-shape-contract-locked",
                "evidence_refs": [
                    contract["cohort_identity"]["evidence_ref"],
                    contract["environment"]["evidence_ref"],
                ],
            },
            {
                "gate_id": "correctness-before-best-of-correct",
                "evidence_refs": evidence_refs,
            },
            {
                "gate_id": "eligible-probe-environment",
                "evidence_refs": [
                    contract["environment"]["evidence_ref"]
                ],
            },
            {
                "gate_id": "paired-baseline-diagnostic-lanes",
                "evidence_refs": [
                    lanes["baseline"]["evidence_ref"],
                    lanes["diagnostic"]["evidence_ref"],
                ],
            },
            {
                "gate_id": "same-hardware-validity-cohort",
                "evidence_refs": cohort_evidence_refs,
            },
        ]
        failed = []
        direct_rejection = probe.get("direct_defect_rejection")
        if isinstance(direct_rejection, dict):
            failed.append(
                {
                    "gate_id": "direct-defect-evidence-qualified",
                    "reason_code": direct_rejection["reason_code"],
                    "evidence_refs": direct_rejection["evidence_refs"],
                }
            )
        if reproducible_faster:
            satisfied.append(
                {
                    "gate_id": "reproducible-faster-alternative",
                    "evidence_refs": evidence_refs,
                }
            )
        else:
            failed.append(
                {
                    "gate_id": "reproducible-faster-alternative",
                    "reason_code": "alternative-not-faster-in-required-sessions",
                    "evidence_refs": evidence_refs,
                }
            )
        frontier_shift_evidence = probe.get("frontier_shift_evidence")
        frontier_gate_failed = False
        if isinstance(frontier_shift_evidence, dict):
            frontier_evidence_refs = list(
                dict.fromkeys(
                    [
                        *frontier_shift_evidence["evidence_refs"],
                        frontier_shift_evidence["holdout"]["evidence_ref"],
                        *sorted(_artifact_refs(frontier_shift_evidence)),
                        *sorted(
                            {
                                candidate["implementation_family"][
                                    "manifest_ref"
                                ]
                                for candidate in candidates
                                if candidate[
                                    "eligible_for_best_of_correct"
                                ]
                            }
                        ),
                        *sorted(
                            {
                                candidate["implementation_family"][
                                    "implementation_ref"
                                ]
                                for candidate in candidates
                                if candidate[
                                    "eligible_for_best_of_correct"
                                ]
                            }
                        ),
                    ]
                )
            )
            frontier_gates = _derive_frontier_shift_gates(
                frontier_shift_evidence,
                candidates=candidates,
                contract=contract,
                measurement_lanes=lanes,
                trigger_item=trigger_items_by_path[stable_path],
                minimum_sessions=policy["minimum_independent_sessions"],
            )
            for gate_id, passed, reason_code in frontier_gates:
                gate = {
                    "gate_id": gate_id,
                    "evidence_refs": frontier_evidence_refs,
                }
                if passed:
                    satisfied.append(gate)
                else:
                    gate["reason_code"] = reason_code
                    failed.append(gate)
                    frontier_gate_failed = True
        candidate_counterexamples = [
            {
                "candidate_id": candidate["candidate_id"],
                "reason_code": candidate["exclusion_reason"],
                "evidence_refs": candidate["evidence_refs"],
            }
            for candidate in candidates
            if candidate["exclusion_reason"] is not None
        ]
        not_evaluated = [
            {
                "gate_id": "confirmed-bug",
                "reason_code": (
                    "direct-defect-prerequisites-failed"
                    if isinstance(direct_rejection, dict)
                    else "direct-defect-evidence-not-provided"
                ),
                "evidence_refs": (
                    direct_rejection["evidence_refs"]
                    if isinstance(direct_rejection, dict)
                    else []
                ),
            }
        ]
        if frontier_gate_failed:
            not_evaluated.append(
                {
                    "gate_id": "frontier-shift",
                    "reason_code": "prerequisites-failed",
                    "evidence_refs": frontier_evidence_refs,
                }
            )
        not_evaluated.append(
            {
                "gate_id": "suspected-regression",
                "reason_code": "policy-undefined",
                "evidence_refs": evidence_refs,
            }
        )
        verdict_result = {
            "stable_path": stable_path,
            "status": "decided",
            "verdict": (
                "implementation_headroom"
                if reproducible_faster
                else "insufficient_evidence"
            ),
            "probe_id": probe["probe_id"],
            "metrics": {
                "target_candidate_id": target["candidate_id"],
                "alternative_candidate_id": alternative["candidate_id"],
                "target_aggregate_latency_ns": target[
                    "aggregate_latency_ns"
                ],
                "alternative_aggregate_latency_ns": alternative[
                    "aggregate_latency_ns"
                ],
                "speedup_fraction": (
                    float(target["aggregate_latency_ns"])
                    / float(alternative["aggregate_latency_ns"])
                    - 1.0
                ),
                "faster_in_every_session": faster_in_every_session,
                "independent_session_count": independent_session_count,
                "independent_process_count": independent_process_count,
            },
            "gates": {
                "satisfied": satisfied,
                "failed": failed,
                "not_evaluated": not_evaluated,
            },
            "bundle_refs": [
                f"run-bundle://{run_id}",
                *probe["evidence_refs"],
            ],
            "counterexamples": [
                *probe.get("counterexamples", []),
                *candidate_counterexamples,
            ],
        }
        if isinstance(frontier_shift_evidence, dict):
            verdict_result["surface_action"] = {
                "action": "preserve",
                "surface": frontier_shift_evidence["surface"],
                "reason_code": "insufficient-evidence-cannot-lower-surface",
            }
        verdicts.append(verdict_result)
    return policy, verdicts


def _profiling_ablation_decision(
    document: dict[str, Any],
    baseline: dict[str, Any],
    diagnostic: dict[str, Any],
) -> tuple[bool, str | None]:
    ablation = diagnostic.get("overhead_ablation")
    if not isinstance(ablation, dict) or ablation.get("status") == (
        "not_provided"
    ):
        return False, "profiling-overhead-ablation-missing"
    policy_value = _versioned_policy(document, "profiling_overhead")
    try:
        policy = ProfilingOverheadPolicy.from_document(policy_value)
    except MeasurementContractError:
        return False, "profiling-overhead-ablation-unqualified"
    selection = ablation.get("selection")
    holdout = ablation.get("holdout")
    if (
        ablation.get("status") != "qualified"
        or ablation.get("instrumentation_profile")
        != diagnostic.get("instrumentation_profile")
        or diagnostic.get("instrumentation_profile")
        not in policy.instrumentation_profiles
        or not isinstance(selection, dict)
        or not isinstance(holdout, dict)
    ):
        return False, "profiling-overhead-ablation-unqualified"
    selection_sessions = selection.get("session_ids")
    baseline_sessions = holdout.get("baseline_session_ids")
    diagnostic_sessions = holdout.get("diagnostic_session_ids")
    baseline_samples = holdout.get("baseline_raw_samples_ns")
    diagnostic_samples = holdout.get("diagnostic_raw_samples_ns")
    session_lists = (selection_sessions, baseline_sessions, diagnostic_sessions)
    if (
        not all(
            isinstance(session_ids, list)
            and all(_nonempty_string(session_id) for session_id in session_ids)
            and len(set(session_ids)) >= policy.minimum_independent_sessions
            for session_ids in session_lists
        )
        or not set(selection_sessions).isdisjoint(baseline_sessions)
        or not set(selection_sessions).isdisjoint(diagnostic_sessions)
        or not set(baseline_sessions).isdisjoint(diagnostic_sessions)
        or holdout.get("pair_id") != baseline.get("pair_id")
        or holdout.get("baseline_lane_id") != baseline.get("lane_id")
        or holdout.get("diagnostic_lane_id") != diagnostic.get("lane_id")
        or not all(
            isinstance(samples, list)
            and samples
            and all(_finite_number(sample) and sample > 0 for sample in samples)
            for samples in (baseline_samples, diagnostic_samples)
        )
        or not all(
            _nonempty_string(value)
            for value in (
                selection.get("evidence_ref"),
                holdout.get("evidence_ref"),
                ablation.get("evidence_ref"),
            )
        )
        or len(
            {
                selection["evidence_ref"],
                holdout["evidence_ref"],
                ablation["evidence_ref"],
            }
        )
        != 3
    ):
        return False, "profiling-overhead-ablation-unqualified"
    baseline_median = float(median(baseline_samples))
    diagnostic_median = float(median(diagnostic_samples))
    observed_overhead_ratio = abs(diagnostic_median - baseline_median) / (
        baseline_median
    )
    if observed_overhead_ratio > policy.maximum_overhead_ratio:
        return False, "profiling-overhead-error-budget-exceeded"
    return True, None


def _validated_adapter_operations(
    document: dict[str, Any], adapter: dict[str, Any]
) -> list[dict[str, str]] | None:
    expected_operations = [
        "discover_capabilities",
        "fingerprint_cohort",
        "preflight",
        "build_timing_plan",
        "collect",
    ]
    operation_evidence = adapter.get("operation_evidence")
    timing_plan = document.get("timing_plan")
    baseline = document.get("baseline_timing_lane")
    diagnostic = document.get("diagnostic_profiling_lane")
    configuration = document.get("resolved_configuration")
    resolved_ir = document.get("resolved_ir")
    if (
        not isinstance(operation_evidence, list)
        or [
            item.get("operation") if isinstance(item, dict) else None
            for item in operation_evidence
        ]
        != expected_operations
        or not all(
            isinstance(item, dict)
            and _nonempty_string(item.get("evidence_ref"))
            for item in operation_evidence
        )
        or len({item["evidence_ref"] for item in operation_evidence})
        != len(operation_evidence)
        or not isinstance(timing_plan, dict)
        or not isinstance(baseline, dict)
        or not isinstance(diagnostic, dict)
        or not isinstance(configuration, dict)
        or not isinstance(resolved_ir, dict)
        or timing_plan.get("case")
        != {
            "benchmark_case": configuration.get("benchmark_case"),
            "semantic_node": resolved_ir.get("semantic_node"),
            "execution_domain": document.get("execution_domain"),
        }
        or timing_plan.get("pair_id") != baseline.get("pair_id")
        or timing_plan.get("baseline_lane_id") != baseline.get("lane_id")
        or timing_plan.get("diagnostic_lane_id") != diagnostic.get("lane_id")
        or timing_plan.get("completion_boundary")
        != baseline.get("completion_boundary")
        or not _nonempty_string(timing_plan.get("evidence_ref"))
    ):
        return None
    return [dict(item) for item in operation_evidence]


def _adapter_contract(
    document: dict[str, Any],
    capability_surfaces: list[dict[str, Any]],
    operator: dict[str, Any],
) -> dict[str, Any] | None:
    adapter = document.get("measurement_adapter")
    if not isinstance(adapter, dict):
        return None
    required_adapter_fields = (
        "adapter_id",
        "adapter_version",
        "protocol_id",
        "protocol_version",
        "evidence_ref",
    )
    if not all(
        _nonempty_string(adapter.get(field))
        for field in required_adapter_fields
    ):
        return {
            "status": "insufficient_evidence",
            "reason_codes": ["incomplete-measurement-adapter-identity"],
        }

    try:
        manifest = MeasurementCapabilityManifest.from_document(
            document.get("measurement_capability_manifest"),
            adapter_id=adapter["adapter_id"],
            cohort_id=str(document.get("cohort_id", "")),
        )
    except MeasurementContractError as error:
        return {
            "status": "insufficient_evidence",
            "reason_codes": [str(error)],
        }
    observation_fields = [field.to_document() for field in manifest.fields]
    operation_evidence = _validated_adapter_operations(document, adapter)
    if operation_evidence is None:
        return {
            "status": "insufficient_evidence",
            "reason_codes": ["invalid-adapter-operation-evidence"],
        }

    cohort_state = _cohort_state(document)
    admission_reasons: list[str] = []
    if not _complete_required_identity(document):
        admission_reasons.append("incomplete-required-identity")
    elif not isinstance(document.get("correctness"), dict) or document[
        "correctness"
    ].get("passed") is not True:
        admission_reasons.append("correctness-not-qualified")
    elif isinstance(cohort_state, dict) and cohort_state.get("status") == (
        "quarantined"
    ):
        admission_reasons.append("cohort-quarantined")
    elif isinstance(cohort_state, dict) and cohort_state.get("status") == (
        "insufficient_evidence"
    ):
        admission_reasons.append(
            str(cohort_state.get("reason_code", "invalid-cohort-evidence"))
        )
    else:
        anchors_value = document.get("frontier_anchors")
        anchors_for_admission = (
            anchors_value if isinstance(anchors_value, list) else []
        )
        completion_boundaries = [
            anchor.get("completion_boundary")
            for anchor in anchors_for_admission
            if isinstance(anchor, dict)
        ]
        anchor_timing_pairs = [
            (anchor.get("timer"), anchor.get("completion_boundary"))
            for anchor in anchors_for_admission
            if isinstance(anchor, dict)
        ]
        if not any(
            _completion_boundary_valid(completion)
            for completion in completion_boundaries
        ):
            admission_reasons.append("incomplete-completion-boundary")
        elif not any(
            _timer_evidence_valid(timer, completion)
            for timer, completion in anchor_timing_pairs
        ):
            admission_reasons.append("invalid-primary-timer-protocol")
        elif not _primary_timer_available(document):
            admission_reasons.append("missing-primary-timer")
        elif operator.get("status") != "known":
            admission_reasons.append(
                str(
                    operator.get(
                        "reason_code",
                        "no-qualified-active-exact-shape-anchor",
                    )
                )
            )
    anchor_admission = {
        "status": (
            "insufficient_evidence" if admission_reasons else "eligible"
        ),
        "reason_codes": admission_reasons,
    }

    baseline = document.get("baseline_timing_lane")
    diagnostic = document.get("diagnostic_profiling_lane")
    lanes: dict[str, Any] | None = None
    if isinstance(diagnostic, dict):
        pair_id = diagnostic.get("pair_id")
        common_lane_identity_valid = (
            isinstance(baseline, dict)
            and _nonempty_string(pair_id)
            and baseline.get("pair_id") == pair_id
            and diagnostic.get("paired_baseline_lane_id")
            == baseline.get("lane_id")
            and diagnostic.get("cohort_id") == document.get("cohort_id")
            and diagnostic.get("cohort_id") == baseline.get("cohort_id")
            and diagnostic.get("candidate_id") == baseline.get("candidate_id")
            and diagnostic.get("execution_domain")
            == baseline.get("execution_domain")
            and diagnostic.get("execution_domain")
            == document.get("execution_domain")
            and _nonempty_string(diagnostic.get("lane_id"))
            and _nonempty_string(diagnostic.get("instrumentation_profile"))
            and _nonempty_string(diagnostic.get("evidence_ref"))
        )
        diagnostic_not_requested = (
            diagnostic.get("status") == "not_requested"
            and diagnostic.get("timing_used_for_frontier") is False
            and _nonempty_string(diagnostic.get("reason_code"))
            and "raw_samples_ns" not in diagnostic
            and "observation_validity" not in diagnostic
            and "frontier_role" not in diagnostic
            and "timer" not in diagnostic
            and "completion_boundary" not in diagnostic
        )
        if common_lane_identity_valid and diagnostic_not_requested:
            lanes = {
                "pair_id": pair_id,
                "baseline_lane_id": baseline["lane_id"],
                "diagnostic_lane_id": diagnostic["lane_id"],
                "diagnostic_frontier_eligible": False,
                "reason_code": "diagnostic-lane-not-requested",
            }
        elif (
            not common_lane_identity_valid
            or not _completion_boundary_valid(
                diagnostic.get("completion_boundary")
            )
            or not _timer_evidence_valid(
                diagnostic.get("timer"),
                diagnostic.get("completion_boundary"),
            )
            or not isinstance(diagnostic.get("raw_samples_ns"), list)
            or not diagnostic["raw_samples_ns"]
            or not all(
                _finite_number(sample) and sample >= 0
                for sample in diagnostic["raw_samples_ns"]
            )
        ):
            lanes = {
                "pair_id": pair_id,
                "baseline_lane_id": (
                    baseline.get("lane_id")
                    if isinstance(baseline, dict)
                    else None
                ),
                "diagnostic_lane_id": diagnostic.get("lane_id"),
                "diagnostic_frontier_eligible": False,
                "reason_code": "invalid-paired-measurement-lanes",
            }
        else:
            (
                diagnostic_frontier_eligible,
                diagnostic_frontier_reason,
            ) = _profiling_ablation_decision(
                document,
                baseline,
                diagnostic,
            )
            lanes = {
                "pair_id": pair_id,
                "baseline_lane_id": baseline["lane_id"],
                "diagnostic_lane_id": diagnostic["lane_id"],
                "diagnostic_frontier_eligible": diagnostic_frontier_eligible,
                "reason_code": diagnostic_frontier_reason,
            }

    cohort_id = document.get("cohort_id")
    anchors_value = document.get("frontier_anchors")
    anchors = anchors_value if isinstance(anchors_value, list) else []
    anchor_ids = [
        anchor["anchor_id"]
        for anchor in anchors
        if isinstance(anchor, dict)
        and _nonempty_string(anchor.get("anchor_id"))
        and anchor.get("cohort_id") == cohort_id
    ]
    surface_refs = [
        {
            "surface_id": surface["surface_id"],
            "version": surface["version"],
            "input_digest": surface["input_digest"],
        }
        for surface in capability_surfaces
        if surface.get("cohort_id") == cohort_id
        and _nonempty_string(surface.get("surface_id"))
        and _nonempty_string(surface.get("version"))
        and _nonempty_string(surface.get("input_digest"))
    ]
    contract_status = (
        "quarantined"
        if isinstance(cohort_state, dict)
        and cohort_state.get("status") == "quarantined"
        else anchor_admission["status"]
    )
    result = {
        "status": contract_status,
        "adapter_id": adapter["adapter_id"],
        "adapter_version": adapter["adapter_version"],
        "protocol": {
            "protocol_id": adapter["protocol_id"],
            "protocol_version": adapter["protocol_version"],
        },
        "cohort_id": cohort_id,
        "anchor_ids": anchor_ids,
        "surface_refs": surface_refs,
        "observation_fields": observation_fields,
        "operation_evidence": operation_evidence,
        "anchor_admission": anchor_admission,
        "evidence_refs": [adapter["evidence_ref"], manifest.evidence_ref],
    }
    if lanes is not None:
        result["lanes"] = lanes
    if cohort_state is not None:
        result["cohort"] = cohort_state
    return result


def _surface_summary(
    surface: dict[str, Any], previous: dict[str, Any] | None
) -> dict[str, Any]:
    anchors = surface.get("anchors")
    cells = surface.get("cells")
    summary = {
        "surface_id": surface.get("surface_id"),
        "version": surface.get("version"),
        "previous_version": surface.get("previous_version"),
        "input_digest": surface.get("input_digest"),
        "cohort_id": surface.get("cohort_id"),
        "domain": surface.get("domain"),
        "domain_policy": surface.get("domain_policy"),
        "candidate_family": surface.get("candidate_family"),
        "algorithm_family": surface.get("algorithm_family"),
        "anchor_lifecycle_policy": surface.get(
            "anchor_lifecycle_policy"
        ),
        "coordinate": surface.get("coordinate"),
        "work_formula": surface.get("work_formula"),
        "response_model": surface.get("response_model"),
        "anchor_ids": [
            anchor.get("anchor_id")
            for anchor in anchors
            if isinstance(anchor, dict)
        ]
        if isinstance(anchors, list)
        else [],
        "cell_ids": [
            cell.get("cell_id")
            for cell in cells
            if isinstance(cell, dict)
        ]
        if isinstance(cells, list)
        else [],
    }
    if previous is not None:
        previous_anchors = previous.get("anchors")
        previous_anchor_ids = {
            anchor.get("anchor_id")
            for anchor in previous_anchors
            if isinstance(anchor, dict)
        } if isinstance(previous_anchors, list) else set()
        current_anchor_ids = set(summary["anchor_ids"])
        summary["transition"] = {
            "previous_version": previous.get("version"),
            "previous_input_digest": previous.get("input_digest"),
            "added_anchor_ids": sorted(current_anchor_ids - previous_anchor_ids),
            "removed_anchor_ids": sorted(previous_anchor_ids - current_anchor_ids),
        }
    if isinstance(surface.get("anchor_state_transitions"), list):
        summary["anchor_state_transitions"] = list(
            surface["anchor_state_transitions"]
        )
    return summary


def _unknown_surface_query(
    query: dict[str, Any],
    surface: dict[str, Any] | None,
    reason_code: str,
    rejected_cell: _RejectedSurfaceCell | None = None,
) -> dict[str, Any]:
    surface_ref = (
        {
            "surface_id": surface.get("surface_id"),
            "version": surface.get("version"),
            "input_digest": surface.get("input_digest"),
        }
        if isinstance(surface, dict)
        else {
            "surface_id": query.get("surface_id"),
            "version": query.get("surface_version"),
            "input_digest": None,
        }
    )
    candidate_family = (
        surface.get("candidate_family")
        if isinstance(surface, dict)
        else None
    )
    algorithm_family = (
        surface.get("algorithm_family")
        if isinstance(surface, dict)
        else None
    )
    return {
        "query_id": query.get("query_id"),
        "status": "unknown",
        "reason_code": reason_code,
        "surface": surface_ref,
        "cohort_id": (
            surface.get("cohort_id") if isinstance(surface, dict) else None
        ),
        "domain": surface.get("domain") if isinstance(surface, dict) else None,
        "domain_policy": (
            surface.get("domain_policy") if isinstance(surface, dict) else None
        ),
        "query_shape": query.get("shape"),
        "candidate_families": (
            [candidate_family] if _nonempty_string(candidate_family) else []
        ),
        "algorithm_families": (
            [algorithm_family] if _nonempty_string(algorithm_family) else []
        ),
        "selected_candidate_family": None,
        "selected_algorithm_family": None,
        "cell_id": (
            rejected_cell.cell.get("cell_id")
            if rejected_cell is not None
            else None
        ),
        "support_seam_id": (
            rejected_cell.cell.get("support_seam_id")
            if rejected_cell is not None
            else None
        ),
        "anchors": [],
        "weights": [],
        "effective_rate": None,
        "latency": None,
        "work_rate_latency": None,
        "response": None,
        "shape_regime": None,
        "uncertainty": None,
        "evidence_refs": (
            list(rejected_cell.cell.get("rejection_evidence_refs", []))
            if rejected_cell is not None
            and isinstance(
                rejected_cell.cell.get("rejection_evidence_refs"), list
            )
            else []
        ),
    }


def _surface_anchor_lifecycle_mode(
    surface: dict[str, Any],
) -> str | None:
    policy = surface.get("anchor_lifecycle_policy")
    if policy is None:
        return "legacy-read-only"
    if (
        isinstance(policy, dict)
        and policy.get("version") == "v2"
        and all(
            _resolved_identity_string(policy.get(key))
            for key in (
                "policy_id",
                "scope",
                "change_reason",
                "revalidation",
            )
        )
    ):
        return "strict-v2"
    return None


def _eligible_surface_anchor(
    surface: dict[str, Any], anchor: object, axes: tuple[str, ...]
) -> bool:
    if not isinstance(anchor, dict):
        return False
    lifecycle_mode = _surface_anchor_lifecycle_mode(surface)
    shape = anchor.get("shape")
    rate = anchor.get("effective_rate")
    response_model = surface.get("response_model")
    latency_primary = (
        isinstance(response_model, dict)
        and response_model.get("kind") == "piecewise-linear-latency"
        and response_model.get("primary_response") == "latency_ns"
    )
    latency = anchor.get("latency_ns")
    if lifecycle_mode == "strict-v2":
        history_compatible = _anchor_state_history_valid(anchor)
    else:
        history_compatible = (
            lifecycle_mode == "legacy-read-only"
            and (
                anchor.get("state_transitions") is None
                or _anchor_state_history_valid(anchor)
            )
        )
    return (
        _nonempty_string(anchor.get("anchor_id"))
        and _nonempty_string(anchor.get("anchor_version"))
        and isinstance(shape, dict)
        and set(shape) == set(axes)
        and all(_finite_number(shape[axis]) and shape[axis] > 0 for axis in axes)
        and (
            _finite_number(latency)
            and latency > 0
            and _finite_number(rate)
            and rate > 0
            if latency_primary
            else _finite_number(rate) and rate > 0
        )
        and anchor.get("rate_unit") == "FLOP/s"
        and _nonempty_string(anchor.get("candidate_id"))
        and anchor.get("candidate_family") == surface.get("candidate_family")
        and (
            not _nonempty_string(surface.get("algorithm_family"))
            or anchor.get("algorithm_family") == surface.get("algorithm_family")
        )
        and anchor.get("cohort_id") == surface.get("cohort_id")
        and anchor.get("domain") == surface.get("domain")
        and anchor.get("observation_validity") == "QUALIFIED"
        and anchor.get("frontier_role") == "ACTIVE"
        and lifecycle_mode is not None
        and history_compatible
        and _nonempty_string(anchor.get("evidence_ref"))
    )


def _surface_policy(surface: dict[str, Any]) -> dict[str, Any] | None:
    policy = surface.get("uncertainty_policy")
    if not isinstance(policy, dict):
        return None
    target_coverage = policy.get("target_coverage")
    if not all(
        _nonempty_string(policy.get(key))
        for key in (
            "policy_id",
            "version",
            "scope",
            "change_reason",
            "revalidation",
        )
    ) or policy.get("combination") != "root-sum-of-squares":
        return None
    if (
        not _finite_number(target_coverage)
        or not 0 < target_coverage <= 1
    ):
        return None
    return policy


def _surface_domain_policy(
    surface: dict[str, Any], dimensions: int
) -> dict[str, Any] | None:
    policy = surface.get("domain_policy")
    if dimensions == 1 and policy is None:
        return None
    if not isinstance(policy, dict) or not all(
        _nonempty_string(policy.get(key))
        for key in (
            "policy_id",
            "version",
            "scope",
            "change_reason",
            "revalidation",
        )
    ):
        return None
    max_edge_span = policy.get("max_edge_span")
    minimum_twice_area = policy.get("minimum_twice_area")
    barycentric_tolerance = policy.get("barycentric_tolerance")
    if (
        policy.get("cell_kind") != "2d-simplex"
        or not _finite_number(max_edge_span)
        or max_edge_span <= 0
        or not _finite_number(minimum_twice_area)
        or minimum_twice_area <= 0
        or not _finite_number(barycentric_tolerance)
        or barycentric_tolerance < 0
    ):
        return None
    return policy


def _select_surface_cell(
    surface: dict[str, Any],
    axes: tuple[str, ...],
    point: tuple[float, ...],
    domain_policy: dict[str, Any] | None,
) -> _SelectedSurfaceCell | _RejectedSurfaceCell | None:
    anchors_value = surface.get("anchors")
    if not isinstance(anchors_value, list):
        return None
    eligible_anchors = [
        anchor
        for anchor in anchors_value
        if _eligible_surface_anchor(surface, anchor, axes)
    ]
    anchor_by_id = {
        anchor["anchor_id"]: anchor for anchor in eligible_anchors
    }
    cells_value = surface.get("cells")
    cells = cells_value if isinstance(cells_value, list) else []
    candidates: list[
        tuple[
            float,
            str,
            dict[str, Any],
            tuple[dict[str, Any], ...],
            tuple[float, ...],
        ]
    ] = []
    rejections: list[_RejectedSurfaceCell] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        anchor_ids = cell.get("anchor_ids")
        if not isinstance(anchor_ids, list) or len(anchor_ids) != len(axes) + 1:
            continue
        cell_anchors = tuple(anchor_by_id.get(anchor_id) for anchor_id in anchor_ids)
        if any(anchor is None for anchor in cell_anchors):
            continue
        anchors = tuple(anchor for anchor in cell_anchors if anchor is not None)
        if not _nonempty_string(cell.get("cell_id")):
            continue
        if len(axes) == 1:
            if cell.get("status") not in {"retained", "regime_boundary"}:
                continue
            axis = axes[0]
            left = float(anchors[0]["shape"][axis])
            right = float(anchors[1]["shape"][axis])
            if left >= right or not left <= point[0] <= right:
                continue
            if cell.get("status") == "regime_boundary":
                rejection_evidence_refs = cell.get(
                    "rejection_evidence_refs"
                )
                if point[0] in {left, right} and not (
                    isinstance(rejection_evidence_refs, list)
                    and rejection_evidence_refs
                ):
                    continue
                rejections.append(
                    _RejectedSurfaceCell(cell, "shape_regime_unvalidated")
                )
                continue
            right_weight = (point[0] - left) / (right - left)
            weights = (1.0 - right_weight, right_weight)
            measure = right - left
        elif len(axes) == 2:
            if cell.get("status") not in (
                "retained",
                "hole",
                "candidate_support_boundary",
                "regime_boundary",
            ):
                continue
            x_axis, y_axis = axes
            vertices = tuple(
                (
                    float(anchor["shape"][x_axis]),
                    float(anchor["shape"][y_axis]),
                )
                for anchor in anchors
            )
            (x0, y0), (x1, y1), (x2, y2) = vertices
            denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (
                y0 - y2
            )
            minimum_twice_area = float(domain_policy["minimum_twice_area"])
            if abs(denominator) < minimum_twice_area:
                x, y = point
                if (
                    min(vertex[0] for vertex in vertices)
                    <= x
                    <= max(vertex[0] for vertex in vertices)
                    and min(vertex[1] for vertex in vertices)
                    <= y
                    <= max(vertex[1] for vertex in vertices)
                ):
                    rejections.append(
                        _RejectedSurfaceCell(cell, "degenerate_simplex")
                    )
                continue
            x, y = point
            first = (
                (y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)
            ) / denominator
            second = (
                (y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)
            ) / denominator
            weights = (first, second, 1.0 - first - second)
            barycentric_tolerance = float(
                domain_policy["barycentric_tolerance"]
            )
            if any(
                not isfinite(weight)
                or weight < -barycentric_tolerance
                or weight > 1.0 + barycentric_tolerance
                for weight in weights
            ):
                continue
            max_edge_span = max(
                hypot(
                    vertices[left][0] - vertices[right][0],
                    vertices[left][1] - vertices[right][1],
                )
                for left, right in ((0, 1), (1, 2), (2, 0))
            )
            if max_edge_span > float(domain_policy["max_edge_span"]):
                rejections.append(
                    _RejectedSurfaceCell(cell, "cell_span_exceeds_policy")
                )
                continue
            if cell.get("status") == "hole":
                rejections.append(
                    _RejectedSurfaceCell(cell, "explicit_domain_hole")
                )
                continue
            if cell.get("status") == "candidate_support_boundary":
                rejections.append(
                    _RejectedSurfaceCell(
                        cell, "candidate_domain_boundary_unvalidated"
                    )
                )
                continue
            if cell.get("status") == "regime_boundary":
                rejections.append(
                    _RejectedSurfaceCell(cell, "shape_regime_unvalidated")
                )
                continue
            measure = abs(denominator)
        else:
            continue
        candidates.append(
            (
                measure,
                str(cell["cell_id"]),
                cell,
                anchors,
                weights,
            )
        )
    if rejections:
        rejection_priority = {
            "explicit_domain_hole": 0,
            "shape_regime_unvalidated": 1,
            "candidate_domain_boundary_unvalidated": 2,
            "cell_span_exceeds_policy": 3,
            "degenerate_simplex": 4,
        }
        return min(
            rejections,
            key=lambda rejection: (
                rejection_priority[rejection.reason_code],
                str(rejection.cell["cell_id"]),
            ),
        )
    if not candidates:
        return None
    _, _, cell, anchors, weights = min(candidates, key=lambda item: item[:2])
    response_model = surface.get("response_model")
    latency_primary = (
        isinstance(response_model, dict)
        and response_model.get("kind") == "piecewise-linear-latency"
        and response_model.get("primary_response") == "latency_ns"
    )
    primary_latency_ns = (
        sum(
            weight * float(anchor["latency_ns"])
            for weight, anchor in zip(weights, anchors, strict=True)
        )
        if latency_primary
        else None
    )
    if primary_latency_ns is not None:
        point_map = {
            axis: value for axis, value in zip(axes, point, strict=True)
        }
        declared_work = _declared_work(surface, point_map)
        if declared_work is None or primary_latency_ns <= 0:
            return None
        effective_rate = declared_work / primary_latency_ns * 1_000_000_000.0
    else:
        effective_rate = sum(
            weight * float(anchor["effective_rate"])
            for weight, anchor in zip(weights, anchors, strict=True)
        )
    return _SelectedSurfaceCell(
        cell=cell,
        anchors=anchors,
        weights=weights,
        exact_anchor=any(
            point
            == tuple(float(anchor["shape"][axis]) for axis in axes)
            for anchor in anchors
        ),
        effective_rate=effective_rate,
        primary_latency_ns=primary_latency_ns,
    )


def _derive_latency_uncertainty(
    policy: dict[str, Any],
    all_anchors: list[Any],
    selected: _SelectedSurfaceCell,
) -> tuple[_LatencyUncertainty | None, str | None]:
    covariance = policy.get("anchor_covariance_ns2")
    if (
        selected.primary_latency_ns is None
        or not isinstance(covariance, list)
        or len(covariance) != len(all_anchors)
        or any(
            not isinstance(row, list) or len(row) != len(all_anchors)
            for row in covariance
        )
    ):
        return None, "insufficient_uncertainty_evidence"
    anchor_indices = [all_anchors.index(anchor) for anchor in selected.anchors]
    anchor_variance = 0.0
    for row_index, row_weight in zip(
        anchor_indices, selected.weights, strict=True
    ):
        for column_index, column_weight in zip(
            anchor_indices, selected.weights, strict=True
        ):
            covariance_value = covariance[row_index][column_index]
            if not _finite_number(covariance_value):
                return None, "insufficient_uncertainty_evidence"
            anchor_variance += row_weight * column_weight * covariance_value
    interpolation_standard = (
        0.0
        if selected.exact_anchor
        else selected.cell.get("interpolation_standard_uncertainty_ns")
    )
    instrumentation_standard = policy.get(
        "instrumentation_standard_uncertainty_ns"
    )
    if (
        not isfinite(anchor_variance)
        or anchor_variance < 0
        or not _finite_number(interpolation_standard)
        or interpolation_standard < 0
        or not _finite_number(instrumentation_standard)
        or instrumentation_standard < 0
    ):
        return None, "insufficient_uncertainty_evidence"
    anchor_standard = anchor_variance**0.5
    combined_standard = hypot(
        anchor_standard,
        float(interpolation_standard),
        float(instrumentation_standard),
    )
    latency_low = selected.primary_latency_ns - combined_standard
    latency_high = selected.primary_latency_ns + combined_standard
    if not all(isfinite(item) for item in (latency_low, latency_high)):
        return None, "invalid_nonfinite_latency_interval"
    if latency_low <= 0:
        return None, "invalid_nonpositive_latency_interval"
    return (
        _LatencyUncertainty(
            anchor_standard_ns=anchor_standard,
            interpolation_standard_ns=float(interpolation_standard),
            instrumentation_standard_ns=float(instrumentation_standard),
            combined_standard_ns=combined_standard,
            latency_low_ns=latency_low,
            latency_high_ns=latency_high,
        ),
        None,
    )


def _derive_surface_uncertainty(
    policy: dict[str, Any],
    all_anchors: list[Any],
    selected: _SelectedSurfaceCell,
) -> tuple[_SurfaceUncertainty | None, str | None]:
    covariance = policy.get("anchor_covariance")
    if (
        not isinstance(covariance, list)
        or len(covariance) != len(all_anchors)
        or any(
            not isinstance(row, list) or len(row) != len(all_anchors)
            for row in covariance
        )
    ):
        return None, "insufficient_uncertainty_evidence"
    anchor_indices = [all_anchors.index(anchor) for anchor in selected.anchors]
    anchor_variance = 0.0
    for row_index, row_weight in zip(
        anchor_indices, selected.weights, strict=True
    ):
        for column_index, column_weight in zip(
            anchor_indices, selected.weights, strict=True
        ):
            covariance_value = covariance[row_index][column_index]
            if not _finite_number(covariance_value):
                return None, "insufficient_uncertainty_evidence"
            anchor_variance += row_weight * column_weight * covariance_value
    interpolation_standard = (
        0.0
        if selected.exact_anchor
        else selected.cell.get("interpolation_standard_uncertainty_rate")
    )
    instrumentation_standard = policy.get(
        "instrumentation_standard_uncertainty_rate"
    )
    if (
        not isfinite(anchor_variance)
        or anchor_variance < 0
        or not _finite_number(interpolation_standard)
        or interpolation_standard < 0
        or not _finite_number(instrumentation_standard)
        or instrumentation_standard < 0
    ):
        return None, "insufficient_uncertainty_evidence"
    anchor_standard = anchor_variance**0.5
    combined_standard = hypot(
        anchor_standard,
        float(interpolation_standard),
        float(instrumentation_standard),
    )
    rate_low = selected.effective_rate - combined_standard
    rate_high = selected.effective_rate + combined_standard
    if (
        not isfinite(selected.effective_rate)
        or not isfinite(combined_standard)
        or not isfinite(rate_low)
        or not isfinite(rate_high)
    ):
        return None, "invalid_nonfinite_rate_interval"
    if rate_low <= 0:
        return None, "invalid_nonpositive_rate_interval"
    return (
        _SurfaceUncertainty(
            anchor_standard_rate=anchor_standard,
            interpolation_standard_rate=float(interpolation_standard),
            instrumentation_standard_rate=float(instrumentation_standard),
            combined_standard_rate=combined_standard,
            rate_low=rate_low,
            rate_high=rate_high,
        ),
        None,
    )


def _declared_work(
    surface: dict[str, Any],
    point: dict[str, float],
) -> float | None:
    work_formula = surface.get("work_formula")
    if not isinstance(work_formula, dict) or not _nonempty_string(
        work_formula.get("version")
    ) or work_formula.get("work_unit") != "FLOP":
        return None
    if work_formula.get("kind") == "square-matmul-2s3" and set(point) == {"s"}:
        declared_work = 2.0 * point["s"] ** 3
    elif (
        work_formula.get("kind") == "matmul-2mnk"
        and set(point) == {"m"}
        and _finite_number(work_formula.get("fixed_n"))
        and work_formula["fixed_n"] > 0
        and _finite_number(work_formula.get("fixed_k"))
        and work_formula["fixed_k"] > 0
    ):
        declared_work = (
            2.0
            * point["m"]
            * work_formula["fixed_n"]
            * work_formula["fixed_k"]
        )
    elif (
        work_formula.get("kind") == "matmul-2mnk"
        and set(point) == {"m", "n"}
        and _finite_number(work_formula.get("fixed_k"))
        and work_formula["fixed_k"] > 0
    ):
        declared_work = 2.0 * point["m"] * point["n"] * work_formula["fixed_k"]
    else:
        return None
    return declared_work if isfinite(declared_work) else None


def _derive_work_rate_latency(
    surface: dict[str, Any],
    point: dict[str, float],
    effective_rate: float,
    uncertainty: _SurfaceUncertainty,
) -> tuple[dict[str, Any], dict[str, float]] | None:
    declared_work = _declared_work(surface, point)
    if declared_work is None or not isfinite(effective_rate):
        return None
    latency = {
        "declared_work": declared_work,
        "work_unit": "FLOP",
        "value_ns": declared_work / effective_rate * 1_000_000_000,
    }
    latency_interval = {
        "lower_ns": declared_work / uncertainty.rate_high * 1_000_000_000,
        "upper_ns": declared_work / uncertainty.rate_low * 1_000_000_000,
    }
    return latency, latency_interval


def _latency_primary_response_model(
    surface: dict[str, Any], axes: tuple[str, ...]
) -> dict[str, Any] | None:
    response = surface.get("response_model")
    work_formula = surface.get("work_formula")
    if response is None:
        return None
    if (
        not isinstance(response, dict)
        or response.get("kind") != "piecewise-linear-latency"
        or response.get("primary_response") != "latency_ns"
        or not all(
            _nonempty_string(response.get(key))
            for key in (
                "response_identity",
                "shape_regime_identity",
                "version",
            )
        )
        or axes != ("m",)
        or response.get("fixed_dimensions")
        != {
            "n": work_formula.get("fixed_n")
            if isinstance(work_formula, dict)
            else None,
            "k": work_formula.get("fixed_k")
            if isinstance(work_formula, dict)
            else None,
        }
        or not isinstance(work_formula, dict)
        or work_formula.get("kind") != "matmul-2mnk"
    ):
        return {}
    return response


def _query_setup_plus_throughput_cell(
    query: dict[str, Any],
    surface: dict[str, Any],
    surface_domain: dict[str, Any],
    domain_policy: dict[str, Any] | None,
    policy: dict[str, Any],
    calibration_refs: list[str],
    anchors_value: list[Any],
    selected: _SelectedSurfaceCell,
    confirmation_refs: list[str],
) -> dict[str, Any] | None:
    response = selected.cell.get("response")
    if not isinstance(response, dict) or response.get("target") != "latency":
        return None
    work_formula = surface.get("work_formula")
    setup_latency_ns = response.get("setup_latency_ns")
    asymptotic_rate = response.get("asymptotic_rate")
    shape_regime = response.get("shape_regime")
    if (
        response.get("kind") != "setup-plus-throughput"
        or response.get("version") != "v1"
        or not isinstance(work_formula, dict)
        or work_formula.get("kind")
        not in {
            "matmul-2mnk-fixed-nk",
            "flash-attention-tnd-forward-qk-pv",
        }
        or work_formula.get("version") != "v1"
        or work_formula.get("work_unit") != "FLOP"
        or not _finite_number(setup_latency_ns)
        or setup_latency_ns < 0
        or not _finite_number(asymptotic_rate)
        or asymptotic_rate <= 0
        or not isinstance(shape_regime, dict)
        or not _nonempty_string(shape_regime.get("identity"))
        or shape_regime.get("classification") not in {"ramp", "steady"}
    ):
        return _unknown_surface_query(query, surface, "invalid_latency_response")
    try:
        semantics = semantics_from_surface_query(surface, query.get("shape"))
    except UnsupportedOperatorShape:
        return _unknown_surface_query(query, surface, "invalid_latency_response")
    if semantics.work_formula != work_formula:
        return _unknown_surface_query(query, surface, "invalid_latency_response")

    latency_ns = float(setup_latency_ns) + (
        semantics.declared_work / float(asymptotic_rate) * 1_000_000_000.0
    )
    exact_anchor = next(
        (
            anchor
            for anchor in selected.anchors
            if anchor.get("shape") == query.get("shape")
        ),
        None,
    )
    if exact_anchor is not None:
        qualified_latency_ns = exact_anchor.get("latency_ns")
        if not _finite_number(qualified_latency_ns) or qualified_latency_ns <= 0:
            return _unknown_surface_query(query, surface, "invalid_latency_response")
        latency_ns = float(qualified_latency_ns)

    covariance = policy.get("anchor_latency_covariance")
    if (
        not isinstance(covariance, list)
        or len(covariance) != len(anchors_value)
        or any(
            not isinstance(row, list) or len(row) != len(anchors_value)
            for row in covariance
        )
    ):
        return _unknown_surface_query(
            query, surface, "insufficient_uncertainty_evidence"
        )
    anchor_indices = [anchors_value.index(anchor) for anchor in selected.anchors]
    anchor_variance = 0.0
    for row_index, row_weight in zip(
        anchor_indices, selected.weights, strict=True
    ):
        for column_index, column_weight in zip(
            anchor_indices, selected.weights, strict=True
        ):
            covariance_value = covariance[row_index][column_index]
            if not _finite_number(covariance_value) or covariance_value < 0:
                return _unknown_surface_query(
                    query, surface, "insufficient_uncertainty_evidence"
                )
            anchor_variance += (
                row_weight * column_weight * float(covariance_value)
            )
    boundary_standard = policy.get("boundary_standard_uncertainty_latency_ns")
    components = {
        "anchor_standard_latency_ns": anchor_variance**0.5,
        "response_model_standard_latency_ns": (
            0.0
            if selected.exact_anchor
            else policy.get("response_model_standard_uncertainty_latency_ns")
        ),
        "instrumentation_standard_latency_ns": policy.get(
            "instrumentation_standard_uncertainty_latency_ns"
        ),
        "boundary_standard_latency_ns": boundary_standard,
    }
    if any(
        value is not None and (not _finite_number(value) or value < 0)
        for value in components.values()
    ):
        return _unknown_surface_query(
            query, surface, "insufficient_uncertainty_evidence"
        )
    combined = sum(
        float(value) ** 2
        for value in components.values()
        if value is not None
    ) ** 0.5
    latency = {
        "declared_work": semantics.declared_work,
        "work_unit": "FLOP",
        "value_ns": latency_ns,
    }
    evidence_refs = [
        *(anchor["evidence_ref"] for anchor in selected.anchors),
        *confirmation_refs,
        *calibration_refs,
        *cast(list[str], surface.get("evidence_refs", [])),
    ]
    utilization = _operator_utilization_results(
        surface,
        semantics,
        semantics.declared_work / (latency_ns * 1e-9),
    )
    return {
        "query_id": query.get("query_id"),
        "status": "exact_anchor" if selected.exact_anchor else "modeled",
        "reason_code": None,
        "surface": {
            "surface_id": surface["surface_id"],
            "version": surface["version"],
            "input_digest": surface["input_digest"],
        },
        "cohort_id": surface["cohort_id"],
        "domain": surface_domain,
        "domain_policy": domain_policy,
        "query_shape": query.get("shape"),
        "operator_shape_identity": semantics.shape_identity,
        "normalized_operator_shape": semantics.normalized_shape,
        "candidate_families": [surface["candidate_family"]],
        "algorithm_families": [],
        "selected_candidate_family": surface["candidate_family"],
        "selected_algorithm_family": None,
        "cell_id": selected.cell["cell_id"],
        "anchors": [
            {
                "anchor_id": anchor["anchor_id"],
                "anchor_version": anchor["anchor_version"],
                "shape": anchor["shape"],
                "latency_ns": anchor["latency_ns"],
                "evidence_ref": anchor["evidence_ref"],
            }
            for anchor in selected.anchors
        ],
        "weights": list(selected.weights),
        "latency": latency,
        "work_rate_latency": latency,
        "effective_rate": {
            "value": semantics.declared_work / (latency_ns * 1e-9),
            "unit": "FLOP/s",
        },
        **utilization,
        "work_formula": work_formula,
        "response": {
            "target": response["target"],
            "kind": response["kind"],
            "version": response["version"],
            "setup_latency_ns": setup_latency_ns,
            "asymptotic_rate": asymptotic_rate,
            "rate_unit": response["rate_unit"],
        },
        "shape_regime": shape_regime,
        "uncertainty": {
            "components": components,
            "combined_standard_latency_ns": combined,
            "target_coverage": policy.get("target_coverage"),
            "policy_ref": f"{policy['policy_id']}/{policy['version']}",
            "calibration_evidence_refs": list(calibration_refs),
            "latency_interval": {
                "lower_ns": max(0.0, latency_ns - combined),
                "upper_ns": latency_ns + combined,
            },
        },
        "evidence_refs": evidence_refs,
    }


def _qualified_rate_reference(
    value: object,
    *,
    semantics: OperatorShapeSemantics,
    surface: dict[str, Any],
) -> tuple[float, str] | None:
    if not isinstance(value, dict):
        return None
    rate = value.get("value")
    evidence_ref = value.get("evidence_ref")
    evidence_sha256 = value.get("evidence_sha256")
    evidence = value.get("evidence")
    surface_domain = surface.get("domain")
    comparable_domain = (
        {
            "hardware_cohort": surface.get("cohort_id"),
            "dtype": surface_domain.get("dtype"),
            "execution_mode": surface_domain.get("execution_mode"),
            "layout": surface_domain.get("layout"),
            "numeric_mode": surface_domain.get("numeric_mode", "default"),
        }
        if isinstance(surface_domain, dict)
        else None
    )
    if (
        value.get("status") != "qualified"
        or not _finite_number(rate)
        or float(rate) <= 0
        or value.get("unit") != "FLOP/s"
        or value.get("semantic_operation") != semantics.operation
        or value.get("work_formula_kind") != semantics.work_formula.get("kind")
        or not _nonempty_string(evidence_ref)
        or fullmatch(
            r"artifact://[A-Za-z0-9][A-Za-z0-9._/-]*"
            r"(?:#[A-Za-z0-9][A-Za-z0-9._-]*)?",
            evidence_ref,
        )
        is None
        or not isinstance(evidence_sha256, str)
        or fullmatch(r"[0-9a-f]{64}", evidence_sha256) is None
        or not isinstance(evidence, dict)
        or evidence.get("schema")
        != "groundupscale.dev/rate-reference-evidence/v1alpha1"
        or evidence.get("source_kind")
        not in {
            "vendor-specification",
            "reviewed-hardware-capability",
            "deterministic-test-fixture",
        }
        or not _nonempty_string(evidence.get("source_uri"))
        or evidence.get("semantic_operation") != value.get("semantic_operation")
        or evidence.get("work_formula_kind") != value.get("work_formula_kind")
        or evidence.get("value") != rate
        or evidence.get("unit") != value.get("unit")
        or comparable_domain is None
        or any(
            not _nonempty_string(expected)
            or value.get(field) != expected
            or evidence.get(field) != expected
            for field, expected in comparable_domain.items()
        )
        or _canonical_digest(evidence) != evidence_sha256
    ):
        return None
    return float(rate), cast(str, evidence_ref)


def _operator_utilization_results(
    surface: dict[str, Any],
    semantics: OperatorShapeSemantics,
    effective_rate: float,
) -> dict[str, dict[str, object]]:
    theoretical_value = surface.get("theoretical_peak")
    theoretical = _qualified_rate_reference(
        theoretical_value,
        semantics=semantics,
        surface=surface,
    )
    if theoretical is None:
        theoretical_reason = (
            "comparable-theoretical-peak-unavailable"
            if theoretical_value is None
            else "theoretical-peak-not-semantically-comparable"
        )
        mfu: dict[str, object] = {
            "status": "unknown",
            "reason_code": theoretical_reason,
            "value": None,
            "unit": "ratio",
            "evidence_refs": [],
        }
    else:
        peak, evidence_ref = theoretical
        mfu = {
            "status": "derived",
            "reason_code": None,
            "value": effective_rate / peak,
            "unit": "ratio",
            "evidence_refs": [evidence_ref],
        }

    empirical_value = surface.get("empirical_rate_envelope")
    empirical = _qualified_rate_reference(
        empirical_value,
        semantics=semantics,
        surface=surface,
    )
    if (
        empirical is None
        or not isinstance(empirical_value, dict)
        or empirical_value.get("label") != "empirical-achieved-rate-envelope"
    ):
        empirical_reason = (
            "empirical-rate-envelope-unavailable"
            if empirical_value is None
            else "empirical-rate-envelope-not-semantically-comparable"
        )
        envelope_utilization: dict[str, object] = {
            "status": "unknown",
            "reason_code": empirical_reason,
            "value": None,
            "unit": "ratio",
            "label": "empirical-envelope-utilization-not-mfu",
            "evidence_refs": [],
        }
    else:
        envelope, evidence_ref = empirical
        envelope_utilization = {
            "status": "derived",
            "reason_code": None,
            "value": effective_rate / envelope,
            "unit": "ratio",
            "label": "empirical-envelope-utilization-not-mfu",
            "evidence_refs": [evidence_ref],
        }
    return {
        "mfu": mfu,
        "empirical_envelope_utilization": envelope_utilization,
    }


def _query_capability_surface(
    query: dict[str, Any], surface: dict[str, Any]
) -> dict[str, Any]:
    coordinate = surface.get("coordinate")
    if not isinstance(coordinate, dict):
        return _unknown_surface_query(
            query, surface, "invalid_surface_coordinate_policy"
        )
    axes_value = (
        [coordinate.get("axis")]
        if "axis" in coordinate
        else coordinate.get("axes")
    )
    if (
        coordinate.get("transform") != "identity"
        or not _nonempty_string(coordinate.get("transform_version"))
        or not isinstance(axes_value, list)
        or len(axes_value) not in (1, 2)
        or not all(_nonempty_string(axis) for axis in axes_value)
        or len(set(axes_value)) != len(axes_value)
    ):
        return _unknown_surface_query(
            query, surface, "invalid_surface_coordinate_policy"
        )
    axes = tuple(axes_value)
    latency_response = _latency_primary_response_model(surface, axes)
    if latency_response == {}:
        return _unknown_surface_query(
            query, surface, "invalid_latency_response_model"
        )
    anchors_value = surface.get("anchors")
    surface_domain = surface.get("domain")
    if (
        isinstance(surface_domain, dict)
        and surface_domain.get("sequence_distribution") == "exact-only"
    ):
        try:
            query_semantics = semantics_from_surface_query(
                surface, query.get("shape")
            )
        except UnsupportedOperatorShape:
            return _unknown_surface_query(
                query,
                surface,
                "unsupported_sequence_distribution_interpolation",
            )
        matching_anchor = next(
            (
                anchor
                for anchor in anchors_value
                if isinstance(anchors_value, list)
                and isinstance(anchor, dict)
                and anchor.get("operator_shape_identity")
                == query_semantics.shape_identity
            ),
            None,
        )
        if not isinstance(matching_anchor, dict):
            return _unknown_surface_query(
                query,
                surface,
                "unsupported_sequence_distribution_interpolation",
            )
        policy = _surface_policy(surface)
        latency_ns = matching_anchor.get("latency_ns")
        standard_latency = matching_anchor.get(
            "standard_uncertainty_latency_ns"
        )
        if (
            policy is None
            or not _finite_number(latency_ns)
            or latency_ns <= 0
            or not _finite_number(standard_latency)
            or standard_latency < 0
        ):
            return _unknown_surface_query(
                query, surface, "insufficient_uncertainty_evidence"
            )
        effective_rate = query_semantics.declared_work / (
            float(latency_ns) * 1e-9
        )
        utilization = _operator_utilization_results(
            surface,
            query_semantics,
            effective_rate,
        )
        components = {
            "anchor_standard_latency_ns": float(standard_latency),
            "response_model_standard_latency_ns": 0.0,
            "instrumentation_standard_latency_ns": policy.get(
                "instrumentation_standard_uncertainty_latency_ns"
            ),
            "boundary_standard_latency_ns": None,
        }
        if any(
            value is not None and (not _finite_number(value) or value < 0)
            for value in components.values()
        ):
            return _unknown_surface_query(
                query, surface, "insufficient_uncertainty_evidence"
            )
        combined = sum(
            float(value) ** 2
            for value in components.values()
            if value is not None
        ) ** 0.5
        evidence_refs = [
            matching_anchor["evidence_ref"],
            *cast(list[str], policy.get("calibration_evidence_refs", [])),
            *cast(list[str], surface.get("evidence_refs", [])),
        ]
        latency = {
            "declared_work": query_semantics.declared_work,
            "work_unit": "FLOP",
            "value_ns": float(latency_ns),
        }
        return {
            "query_id": query.get("query_id"),
            "status": "exact_anchor",
            "reason_code": None,
            "surface": {
                "surface_id": surface["surface_id"],
                "version": surface["version"],
                "input_digest": surface["input_digest"],
            },
            "cohort_id": surface["cohort_id"],
            "domain": surface_domain,
            "domain_policy": None,
            "query_shape": query.get("shape"),
            "operator_shape_identity": query_semantics.shape_identity,
            "normalized_operator_shape": query_semantics.normalized_shape,
            "candidate_families": [surface["candidate_family"]],
            "algorithm_families": [],
            "selected_candidate_family": surface["candidate_family"],
            "selected_algorithm_family": None,
            "cell_id": None,
            "anchors": [
                {
                    "anchor_id": matching_anchor["anchor_id"],
                    "anchor_version": matching_anchor["anchor_version"],
                    "shape": matching_anchor["normalized_operator_shape"],
                    "latency_ns": latency_ns,
                    "evidence_ref": matching_anchor["evidence_ref"],
                }
            ],
            "weights": [1.0],
            "latency": latency,
            "work_rate_latency": latency,
            "effective_rate": {"value": effective_rate, "unit": "FLOP/s"},
            **utilization,
            "work_formula": surface["work_formula"],
            "response": {
                "target": "latency",
                "kind": "exact-sequence-distribution-anchor",
                "version": "v1",
                "setup_latency_ns": 0.0,
                "asymptotic_rate": effective_rate,
                "rate_unit": "FLOP/s",
            },
            "shape_regime": {
                "identity": "ragged-exact-anchor",
                "classification": "exact-only",
            },
            "uncertainty": {
                "components": components,
                "combined_standard_latency_ns": combined,
                "target_coverage": policy.get("target_coverage"),
                "policy_ref": f"{policy['policy_id']}/{policy['version']}",
                "calibration_evidence_refs": policy.get(
                    "calibration_evidence_refs", []
                ),
                "latency_interval": {
                    "lower_ns": max(0.0, float(latency_ns) - combined),
                    "upper_ns": float(latency_ns) + combined,
                },
            },
            "evidence_refs": evidence_refs,
        }
    if len(axes) == 2 and (
        not _nonempty_string(surface.get("candidate_family"))
        or not _nonempty_string(surface.get("algorithm_family"))
        or not isinstance(anchors_value, list)
        or any(
            not isinstance(anchor, dict)
            or anchor.get("candidate_family") != surface.get("candidate_family")
            or anchor.get("algorithm_family") != surface.get("algorithm_family")
            for anchor in anchors_value
        )
    ):
        return _unknown_surface_query(
            query, surface, "incomplete_candidate_family_facet"
        )
    shape = query.get("shape")
    work_formula = surface.get("work_formula")
    integer_shape_required = isinstance(work_formula, dict) and work_formula.get(
        "kind"
    ) in {
        "matmul-2mnk-fixed-nk",
        "flash-attention-tnd-forward-qk-pv",
    }
    if (
        not isinstance(shape, dict)
        or set(shape) != set(axes)
        or any(
            not _finite_number(shape[axis])
            or shape[axis] <= 0
            or integer_shape_required
            and (
                not isinstance(shape[axis], int)
                or isinstance(shape[axis], bool)
            )
            for axis in axes
        )
    ):
        return _unknown_surface_query(query, surface, "invalid_query_shape")
    surface_domain = surface.get("domain")
    query_domain = query.get("domain")
    if not isinstance(surface_domain, dict) or not isinstance(query_domain, dict):
        return _unknown_surface_query(query, surface, "incomplete_surface_domain")
    if surface_domain.get("alignment_validated") is not True:
        return _unknown_surface_query(
            query, surface, "alignment_regime_unvalidated"
        )
    if (
        surface_domain.get("working_set_validated") is False
        or len(axes) == 2
        and surface_domain.get("working_set_validated") is not True
    ):
        return _unknown_surface_query(
            query, surface, "working_set_regime_unvalidated"
        )
    if (
        surface_domain.get("kernel_dispatch_validated") is False
        or len(axes) == 2
        and surface_domain.get("kernel_dispatch_validated") is not True
    ):
        return _unknown_surface_query(
            query, surface, "kernel_dispatch_regime_unvalidated"
        )
    if surface_domain.get("regime_validated") is not True:
        return _unknown_surface_query(query, surface, "shape_regime_unvalidated")
    if query_domain.get("alignment_regime") != surface_domain.get(
        "alignment_regime"
    ):
        return _unknown_surface_query(
            query, surface, "alignment_regime_unvalidated"
        )
    if query_domain.get("working_set_regime") != surface_domain.get(
        "working_set_regime"
    ):
        return _unknown_surface_query(
            query, surface, "working_set_regime_unvalidated"
        )
    if query_domain.get("kernel_dispatch_regime") != surface_domain.get(
        "kernel_dispatch_regime"
    ):
        return _unknown_surface_query(
            query, surface, "kernel_dispatch_regime_unvalidated"
        )
    if (
        "fixed_n" in surface_domain
        and (
            query_domain.get("fixed_n") != surface_domain.get("fixed_n")
            or query_domain.get("fixed_k") != surface_domain.get("fixed_k")
        )
    ):
        return _unknown_surface_query(
            query, surface, "fixed_nk_domain_mismatch"
        )
    if query_domain != surface_domain:
        return _unknown_surface_query(query, surface, "shape_regime_unvalidated")

    domain_policy = _surface_domain_policy(surface, len(axes))
    if len(axes) == 2 and domain_policy is None:
        return _unknown_surface_query(
            query, surface, "invalid_surface_domain_policy"
        )

    policy = _surface_policy(surface)
    if policy is None:
        return _unknown_surface_query(
            query, surface, "missing_uncertainty_combination_policy"
        )
    calibration_refs = policy.get("calibration_evidence_refs")
    if (
        not isinstance(calibration_refs, list)
        or not calibration_refs
        or not all(_nonempty_string(reference) for reference in calibration_refs)
    ):
        return _unknown_surface_query(
            query, surface, "insufficient_uncertainty_evidence"
        )

    if not isinstance(anchors_value, list):
        return _unknown_surface_query(
            query, surface, "no_qualified_active_surface_anchor"
        )
    point = tuple(float(shape[axis]) for axis in axes)
    selected = _select_surface_cell(surface, axes, point, domain_policy)
    if selected is None:
        return _unknown_surface_query(
            query, surface, "outside_validated_domain"
        )
    if isinstance(selected, _RejectedSurfaceCell):
        return _unknown_surface_query(
            query,
            surface,
            selected.reason_code,
            selected,
        )
    confirmation_refs = selected.cell.get("confirmation_evidence_refs")
    if (
        not isinstance(confirmation_refs, list)
        or not confirmation_refs
        or not all(_nonempty_string(reference) for reference in confirmation_refs)
    ):
        return _unknown_surface_query(
            query, surface, "insufficient_uncertainty_evidence"
        )
    setup_response = _query_setup_plus_throughput_cell(
        query,
        surface,
        surface_domain,
        domain_policy,
        policy,
        cast(list[str], calibration_refs),
        anchors_value,
        selected,
        cast(list[str], confirmation_refs),
    )
    if setup_response is not None:
        return setup_response
    point_map = {
        axis: value for axis, value in zip(axes, point, strict=True)
    }
    if latency_response is not None:
        latency_uncertainty, uncertainty_reason = (
            _derive_latency_uncertainty(policy, anchors_value, selected)
        )
        if latency_uncertainty is None:
            return _unknown_surface_query(
                query,
                surface,
                uncertainty_reason or "insufficient_uncertainty_evidence",
            )
        declared_work = _declared_work(surface, point_map)
        if declared_work is None or selected.primary_latency_ns is None:
            return _unknown_surface_query(
                query, surface, "invalid_work_formula"
            )
        latency = {
            "declared_work": declared_work,
            "work_unit": "FLOP",
            "value_ns": selected.primary_latency_ns,
        }
        latency_interval = {
            "lower_ns": latency_uncertainty.latency_low_ns,
            "upper_ns": latency_uncertainty.latency_high_ns,
        }
        uncertainty_result = {
            "components": {
                "anchor_standard_ns": (
                    latency_uncertainty.anchor_standard_ns
                ),
                "interpolation_standard_ns": (
                    latency_uncertainty.interpolation_standard_ns
                ),
                "instrumentation_standard_ns": (
                    latency_uncertainty.instrumentation_standard_ns
                ),
            },
            "combined_standard_ns": (
                latency_uncertainty.combined_standard_ns
            ),
            "target_coverage": policy.get("target_coverage"),
            "policy_ref": f"{policy['policy_id']}/{policy['version']}",
            "calibration_evidence_refs": list(calibration_refs),
            "latency_interval": latency_interval,
        }
    else:
        uncertainty, uncertainty_reason = _derive_surface_uncertainty(
            policy, anchors_value, selected
        )
        if uncertainty is None:
            return _unknown_surface_query(
                query,
                surface,
                uncertainty_reason or "insufficient_uncertainty_evidence",
            )
        latency_derivation = _derive_work_rate_latency(
            surface,
            point_map,
            selected.effective_rate,
            uncertainty,
        )
        if latency_derivation is None:
            return _unknown_surface_query(
                query, surface, "invalid_work_formula"
            )
        latency, latency_interval = latency_derivation
        uncertainty_result = {
            "components": {
                "anchor_standard_rate": uncertainty.anchor_standard_rate,
                "interpolation_standard_rate": (
                    uncertainty.interpolation_standard_rate
                ),
                "instrumentation_standard_rate": (
                    uncertainty.instrumentation_standard_rate
                ),
            },
            "combined_standard_rate": uncertainty.combined_standard_rate,
            "target_coverage": policy.get("target_coverage"),
            "policy_ref": f"{policy['policy_id']}/{policy['version']}",
            "calibration_evidence_refs": list(calibration_refs),
            "rate_interval": {
                "lower": uncertainty.rate_low,
                "upper": uncertainty.rate_high,
            },
            "latency_interval": latency_interval,
        }
    evidence_refs = [
        *(anchor["evidence_ref"] for anchor in selected.anchors),
        *confirmation_refs,
        *calibration_refs,
        *(
            surface.get("evidence_refs")
            if isinstance(surface.get("evidence_refs"), list)
            else []
        ),
    ]
    return {
        "query_id": query.get("query_id"),
        "status": "exact_anchor" if selected.exact_anchor else "interpolated",
        "reason_code": None,
        "surface": {
            "surface_id": surface["surface_id"],
            "version": surface["version"],
            "input_digest": surface["input_digest"],
        },
        "cohort_id": surface["cohort_id"],
        "domain": surface_domain,
        "domain_policy": domain_policy,
        "query_shape": shape,
        "candidate_families": [surface["candidate_family"]],
        "algorithm_families": (
            [surface["algorithm_family"]]
            if _nonempty_string(surface.get("algorithm_family"))
            else []
        ),
        "selected_candidate_family": surface["candidate_family"],
        "selected_algorithm_family": surface.get("algorithm_family"),
        "cell_id": selected.cell["cell_id"],
        "anchors": [
            {
                "anchor_id": anchor["anchor_id"],
                "anchor_version": anchor["anchor_version"],
                "shape": anchor["shape"],
                "effective_rate": anchor["effective_rate"],
                **(
                    {"latency_ns": anchor["latency_ns"]}
                    if latency_response is not None
                    else {}
                ),
                "evidence_ref": anchor["evidence_ref"],
            }
            for anchor in selected.anchors
        ],
        "weights": list(selected.weights),
        "effective_rate": {
            "value": selected.effective_rate,
            "unit": "FLOP/s",
        },
        "work_rate_latency": latency,
        **(
            {
                "response": {
                    "primary_response": "latency_ns",
                    "response_identity": latency_response[
                        "response_identity"
                    ],
                    "shape_regime_identity": latency_response[
                        "shape_regime_identity"
                    ],
                    "value_ns": selected.primary_latency_ns,
                }
            }
            if latency_response is not None
            else {}
        ),
        "uncertainty": uncertainty_result,
        "evidence_refs": evidence_refs,
    }


def _candidate_support_policy(
    envelope: dict[str, Any],
) -> dict[str, Any] | None:
    policy = envelope.get("support_policy")
    if not isinstance(policy, dict) or not all(
        _nonempty_string(policy.get(key))
        for key in (
            "policy_id",
            "version",
            "scope",
            "change_reason",
            "revalidation",
        )
    ):
        return None
    if policy.get("rule") != "common-stable-support-or-validated-seam":
        return None
    validated_seams = policy.get("validated_seams")
    if not isinstance(validated_seams, list) or any(
        not isinstance(seam, dict)
        or not all(
            _nonempty_string(seam.get(key))
            for key in (
                "seam_id",
                "unsupported_candidate_family",
                "validation_version",
                "evidence_ref",
            )
        )
        for seam in validated_seams
    ):
        return None
    return policy


def _candidate_envelope_ref(envelope: dict[str, Any]) -> dict[str, Any]:
    return {
        "envelope_id": envelope["envelope_id"],
        "version": envelope["version"],
        "input_digest": envelope["input_digest"],
    }


def _candidate_facet_result(
    result: dict[str, Any], surface: dict[str, Any]
) -> dict[str, Any]:
    return {
        "surface": result["surface"],
        "candidate_family": surface["candidate_family"],
        "algorithm_family": surface["algorithm_family"],
        "status": result["status"],
        "reason_code": result["reason_code"],
        "cell_id": result["cell_id"],
        "support_seam_id": result.get("support_seam_id"),
        "anchors": result["anchors"],
        "weights": result["weights"],
        "effective_rate": result["effective_rate"],
        "uncertainty": result["uncertainty"],
        "evidence_refs": result["evidence_refs"],
    }


def _unknown_candidate_envelope_query(
    query: dict[str, Any],
    envelope: dict[str, Any] | None,
    reason_code: str,
    facets: list[dict[str, Any]],
) -> dict[str, Any]:
    result = _unknown_surface_query(query, None, reason_code)
    envelope_evidence_refs = (
        envelope.get("evidence_refs")
        if envelope is not None
        and isinstance(envelope.get("evidence_refs"), list)
        else []
    )
    evidence_refs = list(
        dict.fromkeys(
            [
                *(
                    reference
                    for facet in facets
                    for reference in facet["evidence_refs"]
                ),
                *envelope_evidence_refs,
            ]
        )
    )
    result.update(
        {
            "envelope": (
                _candidate_envelope_ref(envelope)
                if envelope is not None
                else {
                    "envelope_id": query.get("envelope_id"),
                    "version": query.get("envelope_version"),
                    "input_digest": None,
                }
            ),
            "cohort_id": (
                envelope.get("cohort_id") if envelope is not None else None
            ),
            "domain": envelope.get("domain") if envelope is not None else None,
            "domain_policy": (
                envelope.get("domain_policy")
                if envelope is not None
                else None
            ),
            "candidate_families": [
                facet["candidate_family"] for facet in facets
            ],
            "algorithm_families": [
                facet["algorithm_family"] for facet in facets
            ],
            "candidate_facets": facets,
            "support_policy": (
                envelope.get("support_policy")
                if envelope is not None
                else None
            ),
            "support_transitions": [],
            "evidence_refs": evidence_refs,
        }
    )
    return result


def _query_candidate_envelope(
    query: dict[str, Any],
    envelope: dict[str, Any],
    surface_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    policy = _candidate_support_policy(envelope)
    if policy is None:
        return _unknown_candidate_envelope_query(
            query, envelope, "invalid_candidate_support_policy", []
        )
    if query.get("domain") != envelope.get("domain"):
        return _unknown_candidate_envelope_query(
            query, envelope, "shape_regime_unvalidated", []
        )
    refs = envelope["surface_refs"]
    surfaces = [
        surface_index[(reference["surface_id"], reference["version"])]
        for reference in refs
    ]
    facet_queries = [
        _query_capability_surface(query, surface) for surface in surfaces
    ]
    facets = [
        _candidate_facet_result(result, surface)
        for result, surface in zip(facet_queries, surfaces, strict=True)
    ]
    known = [
        (result, surface)
        for result, surface in zip(facet_queries, surfaces, strict=True)
        if result["status"] != "unknown"
    ]
    unknown = [
        (result, surface)
        for result, surface in zip(facet_queries, surfaces, strict=True)
        if result["status"] == "unknown"
    ]
    support_transitions: list[dict[str, Any]] = []
    validated_seams = policy["validated_seams"]
    for result, surface in unknown:
        matching_seam = next(
            (
                seam
                for seam in validated_seams
                if seam["seam_id"] == result.get("support_seam_id")
                and seam["unsupported_candidate_family"]
                == surface["candidate_family"]
            ),
            None,
        )
        if (
            result["reason_code"]
            != "candidate_domain_boundary_unvalidated"
            or matching_seam is None
        ):
            reason_code = (
                "outside_validated_domain"
                if unknown
                and not known
                and all(
                    item[0]["reason_code"] == "outside_validated_domain"
                    for item in unknown
                )
                else "candidate_domain_boundary_unvalidated"
            )
            return _unknown_candidate_envelope_query(
                query, envelope, reason_code, facets
            )
        support_transitions.append(matching_seam)
    if not known:
        return _unknown_candidate_envelope_query(
            query, envelope, "outside_validated_domain", facets
        )
    winner_result, winner_surface = min(
        known,
        key=lambda item: (
            -float(item[0]["effective_rate"]["value"]),
            str(item[1]["candidate_family"]),
            str(item[1]["surface_id"]),
        ),
    )
    evidence_refs = [
        *winner_result["evidence_refs"],
        *(
            envelope.get("evidence_refs")
            if isinstance(envelope.get("evidence_refs"), list)
            else []
        ),
        *(transition["evidence_ref"] for transition in support_transitions),
    ]
    return {
        **winner_result,
        "envelope": _candidate_envelope_ref(envelope),
        "candidate_families": [
            surface["candidate_family"] for surface in surfaces
        ],
        "algorithm_families": [
            surface["algorithm_family"] for surface in surfaces
        ],
        "selected_candidate_family": winner_surface["candidate_family"],
        "selected_algorithm_family": winner_surface["algorithm_family"],
        "candidate_facets": facets,
        "support_policy": policy,
        "support_transitions": support_transitions,
        "evidence_refs": evidence_refs,
    }


def _validated_candidate_envelopes(
    envelopes: list[Any],
    surface_index: dict[tuple[str, str], dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    valid: list[dict[str, Any]] = []
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for envelope in envelopes:
        if not isinstance(envelope, dict):
            continue
        envelope_id = envelope.get("envelope_id")
        version = envelope.get("version")
        if not _nonempty_string(envelope_id) or not _nonempty_string(version):
            continue
        expected_digest = envelope.get("input_digest")
        actual_digest = _canonical_digest(
            {
                key: value
                for key, value in envelope.items()
                if key != "input_digest"
            }
        )
        if expected_digest != actual_digest:
            raise DiagnosticBundleIntegrityError(
                "candidate envelope input digest mismatch: "
                f"{envelope_id}@{version}"
            )
        envelope_key = (envelope_id, version)
        if envelope_key in index:
            raise DiagnosticBundleIntegrityError(
                f"duplicate candidate envelope version: {envelope_id}@{version}"
            )
        refs = envelope.get("surface_refs")
        if (
            not isinstance(refs, list)
            or len(refs) < 2
            or _candidate_support_policy(envelope) is None
        ):
            continue
        surfaces: list[dict[str, Any]] = []
        seen_refs: set[tuple[str, str]] = set()
        for reference in refs:
            if not isinstance(reference, dict):
                break
            ref_key = (reference.get("surface_id"), reference.get("version"))
            surface = surface_index.get(ref_key)
            if surface is None or ref_key in seen_refs:
                break
            seen_refs.add(ref_key)
            surfaces.append(surface)
        else:
            candidate_families = [
                surface.get("candidate_family") for surface in surfaces
            ]
            coordinate = surfaces[0].get("coordinate")
            work_formula = surfaces[0].get("work_formula")
            response_model = surfaces[0].get("response_model")
            if (
                len(set(candidate_families)) != len(candidate_families)
                or any(
                    not _nonempty_string(surface.get("algorithm_family"))
                    or surface.get("cohort_id") != envelope.get("cohort_id")
                    or surface.get("domain") != envelope.get("domain")
                    or surface.get("domain_policy")
                    != envelope.get("domain_policy")
                    or surface.get("coordinate") != coordinate
                    or surface.get("work_formula") != work_formula
                    or surface.get("response_model") != response_model
                    for surface in surfaces
                )
            ):
                continue
            valid.append(envelope)
            index[envelope_key] = envelope
    return valid, index


def _candidate_envelope_summary(
    envelope: dict[str, Any],
    surface_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    surfaces = [
        surface_index[(reference["surface_id"], reference["version"])]
        for reference in envelope["surface_refs"]
    ]
    return {
        "envelope_id": envelope["envelope_id"],
        "version": envelope["version"],
        "input_digest": envelope["input_digest"],
        "cohort_id": envelope["cohort_id"],
        "domain": envelope["domain"],
        "domain_policy": envelope["domain_policy"],
        "candidate_families": [
            surface["candidate_family"] for surface in surfaces
        ],
        "algorithm_families": [
            surface["algorithm_family"] for surface in surfaces
        ],
        "support_policy": envelope["support_policy"],
    }


def _validated_surface_lineage(
    surfaces: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    valid_surfaces: list[dict[str, Any]] = []
    surface_index: dict[tuple[str, str], dict[str, Any]] = {}
    root_counts: dict[str, int] = {}
    for surface in surfaces:
        if surface.get("previous_version") is None:
            surface_id = surface.get("surface_id")
            if _nonempty_string(surface_id):
                root_counts[surface_id] = root_counts.get(surface_id, 0) + 1
    ambiguous_roots = {
        surface_id for surface_id, count in root_counts.items() if count != 1
    }
    pending = [
        surface
        for surface in surfaces
        if surface.get("surface_id") not in ambiguous_roots
    ]
    while pending:
        unresolved: list[dict[str, Any]] = []
        progress = False
        for surface in pending:
            previous_version = surface.get("previous_version")
            previous = None
            if previous_version is not None:
                if not _nonempty_string(previous_version):
                    continue
                previous = surface_index.get(
                    (surface["surface_id"], previous_version)
                )
                if previous is None:
                    unresolved.append(surface)
                    continue
                if any(
                    previous.get(key) != surface.get(key)
                    for key in (
                        "cohort_id",
                        "domain",
                        "domain_policy",
                        "candidate_family",
                        "algorithm_family",
                        "anchor_lifecycle_policy",
                        "coordinate",
                        "work_formula",
                        "response_model",
                    )
                ):
                    continue
                previous_anchors = previous.get("anchors")
                current_anchors = surface.get("anchors")
                if not isinstance(previous_anchors, list) or not isinstance(
                    current_anchors, list
                ):
                    continue
                previous_by_id = {
                    anchor.get("anchor_id"): anchor
                    for anchor in previous_anchors
                    if isinstance(anchor, dict)
                }
                current_by_id = {
                    anchor.get("anchor_id"): anchor
                    for anchor in current_anchors
                    if isinstance(anchor, dict)
                }
                if not previous_by_id.keys() <= current_by_id.keys() or any(
                    current_by_id[anchor_id] != previous_anchor
                    for anchor_id, previous_anchor in previous_by_id.items()
                ):
                    continue
            valid_surfaces.append(surface)
            surface_index[(surface["surface_id"], surface["version"])] = surface
            progress = True
        if not progress:
            break
        pending = unresolved
    return valid_surfaces, surface_index


def _capability_surface_results(
    document: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    surfaces_value = document.get("capability_surfaces")
    queries_value = document.get("surface_queries")
    if surfaces_value is None and queries_value is None:
        return [], [], []
    surfaces = surfaces_value if isinstance(surfaces_value, list) else []
    queries = queries_value if isinstance(queries_value, list) else []
    digest_valid_surfaces: list[dict[str, Any]] = []
    digest_valid_index: dict[tuple[str, str], dict[str, Any]] = {}
    for surface in surfaces:
        if not isinstance(surface, dict):
            continue
        surface_id = surface.get("surface_id")
        version = surface.get("version")
        expected_digest = surface.get("input_digest")
        actual_digest = _canonical_digest(
            {key: value for key, value in surface.items() if key != "input_digest"}
        )
        if not _nonempty_string(surface_id) or not _nonempty_string(version):
            continue
        if expected_digest != actual_digest:
            raise DiagnosticBundleIntegrityError(
                "capability surface input digest mismatch: "
                f"{surface_id}@{version}"
            )
        surface_key = (surface_id, version)
        if surface_key in digest_valid_index:
            raise DiagnosticBundleIntegrityError(
                f"duplicate capability surface version: {surface_id}@{version}"
            )
        digest_valid_surfaces.append(surface)
        digest_valid_index[surface_key] = surface
    valid_surfaces, surface_index = _validated_surface_lineage(
        digest_valid_surfaces
    )
    updates_value = document.get("surface_updates")
    updates = updates_value if isinstance(updates_value, list) else []
    for update in updates:
        if not isinstance(update, dict):
            continue
        surface_id = update.get("surface_id")
        base_version = update.get("base_version")
        new_version = update.get("new_version")
        if not all(
            _nonempty_string(value)
            for value in (
                update.get("update_id"),
                surface_id,
                base_version,
                new_version,
            )
        ):
            continue
        base = surface_index.get((surface_id, base_version))
        new_key = (surface_id, new_version)
        if new_key in digest_valid_index:
            raise DiagnosticBundleIntegrityError(
                f"duplicate capability surface version: {surface_id}@{new_version}"
            )
        coordinate = base.get("coordinate") if isinstance(base, dict) else None
        axes_value = (
            [coordinate.get("axis")]
            if isinstance(coordinate, dict) and "axis" in coordinate
            else coordinate.get("axes")
            if isinstance(coordinate, dict)
            else None
        )
        axes = (
            tuple(axes_value)
            if isinstance(axes_value, list)
            and all(_nonempty_string(axis) for axis in axes_value)
            else ()
        )
        base_anchors = base.get("anchors") if isinstance(base, dict) else None
        cells = update.get("cells")
        uncertainty_policy = update.get("uncertainty_policy")
        update_evidence_refs = update.get("evidence_refs")
        if (
            not isinstance(base, dict)
            or not axes
            or not isinstance(base_anchors, list)
            or _surface_anchor_lifecycle_mode(base) != "strict-v2"
            or not isinstance(cells, list)
            or not isinstance(uncertainty_policy, dict)
            or not isinstance(update_evidence_refs, list)
            or not update_evidence_refs
            or not all(
                _artifact_uri(reference)
                for reference in update_evidence_refs
            )
        ):
            continue
        anchor_ids = {
            item.get("anchor_id")
            for item in base_anchors
            if isinstance(item, dict)
        }
        operation = update.get("operation", "add_anchor")
        anchor_state_transitions = list(
            base.get("anchor_state_transitions", [])
            if isinstance(base.get("anchor_state_transitions"), list)
            else []
        )
        if operation == "add_anchor":
            anchor = update.get("anchor")
            same_coordinate = [
                item
                for item in base_anchors
                if isinstance(item, dict)
                and isinstance(item.get("shape"), dict)
                and all(
                    item["shape"].get(axis) == anchor.get("shape", {}).get(axis)
                    for axis in axes
                )
            ] if isinstance(anchor, dict) else []
            if (
                not _eligible_surface_anchor(base, anchor, axes)
                or anchor["anchor_id"] in anchor_ids
                or any(
                    float(item["effective_rate"])
                    >= float(anchor["effective_rate"])
                    for item in same_coordinate
                )
            ):
                continue
            superseded_ids = {
                item["anchor_id"] for item in same_coordinate
            }
            new_anchors = [
                item
                for item in base_anchors
                if item.get("anchor_id") not in superseded_ids
            ]
            new_anchors.append(anchor)
            anchor_state_transitions.extend(
                {
                    "anchor_id": anchor["anchor_id"],
                    **transition,
                }
                for transition in anchor["state_transitions"]
            )
            for superseded in same_coordinate:
                anchor_state_transitions.append(
                    {
                        "anchor_id": superseded["anchor_id"],
                        "sequence": len(
                            superseded["state_transitions"]
                        ) + 1,
                        "axis": "frontier_role",
                        "from": "ACTIVE",
                        "to": "SUPERSEDED",
                        "reason_code": (
                            "faster-qualified-anchor-at-same-coordinate"
                        ),
                        "evidence_refs": list(update_evidence_refs),
                    }
                )
        elif operation == "retract_anchor":
            anchor_id = update.get("anchor_id")
            invalidation = update.get("invalidation")
            invalidation_refs = (
                invalidation.get("evidence_refs")
                if isinstance(invalidation, dict)
                else None
            )
            invalidation_transitions = (
                invalidation.get("state_transitions")
                if isinstance(invalidation, dict)
                else None
            )
            matching = [
                item
                for item in base_anchors
                if isinstance(item, dict)
                and item.get("anchor_id") == anchor_id
            ]
            if (
                len(matching) != 1
                or not _eligible_surface_anchor(base, matching[0], axes)
                or not isinstance(invalidation, dict)
                or invalidation.get("kind")
                not in {"correctness", "provenance"}
                or not _nonempty_string(invalidation.get("reason_code"))
                or not isinstance(invalidation_refs, list)
                or not invalidation_refs
                or not all(
                    _nonempty_string(reference)
                    for reference in invalidation_refs
                )
                or not isinstance(invalidation_transitions, list)
                or [
                    transition.get("axis")
                    for transition in invalidation_transitions
                    if isinstance(transition, dict)
                ]
                != ["observation_validity", "frontier_role"]
                or not _anchor_state_history_valid(
                    {
                        **matching[0],
                        "observation_validity": "REVOKED",
                        "frontier_role": "REVOKED_ROLE",
                        "state_transitions": [
                            *matching[0]["state_transitions"],
                            *invalidation_transitions,
                        ],
                    }
                )
                or any(
                    transition.get("reason_code")
                    != invalidation.get("reason_code")
                    or transition.get("evidence_refs")
                    != invalidation_refs
                    for transition in invalidation_transitions
                )
            ):
                continue
            new_anchors = [
                item
                for item in base_anchors
                if item.get("anchor_id") != anchor_id
            ]
            anchor_state_transitions.extend(
                {
                    "anchor_id": anchor_id,
                    **transition,
                }
                for transition in invalidation_transitions
            )
        else:
            continue
        base_evidence_refs = base.get("evidence_refs")
        new_anchors = sorted(
            new_anchors,
            key=lambda item: tuple(float(item["shape"][axis]) for axis in axes),
        )
        base_cells = base.get("cells")
        base_regime_ids = {
            cell.get("regime_id")
            for cell in base_cells
            if isinstance(cell, dict)
            and cell.get("status") == "retained"
            and _nonempty_string(cell.get("regime_id"))
        } if isinstance(base_cells, list) else set()
        new_cells = []
        for cell in cells:
            if not isinstance(cell, dict):
                new_cells.append(cell)
                continue
            new_cell = dict(cell)
            if (
                base_regime_ids
                and new_cell.get("regime_id") not in base_regime_ids
            ):
                new_cell["status"] = "regime_boundary"
                new_cell["rejection_evidence_refs"] = list(
                    update_evidence_refs
                )
            new_cells.append(new_cell)
        new_surface = {
            **base,
            "version": new_version,
            "previous_version": base_version,
            "anchors": new_anchors,
            "cells": new_cells,
            "uncertainty_policy": uncertainty_policy,
            "anchor_state_transitions": anchor_state_transitions,
            "evidence_refs": [
                *(
                    base_evidence_refs
                    if isinstance(base_evidence_refs, list)
                    else []
                ),
                *update_evidence_refs,
            ],
        }
        new_surface.pop("input_digest", None)
        new_surface["input_digest"] = _canonical_digest(new_surface)
        digest_valid_surfaces.append(new_surface)
        digest_valid_index[new_key] = new_surface
        valid_surfaces.append(new_surface)
        surface_index[new_key] = new_surface
    envelopes_value = document.get("candidate_envelopes")
    envelopes = envelopes_value if isinstance(envelopes_value, list) else []
    valid_envelopes, envelope_index = _validated_candidate_envelopes(
        envelopes, surface_index
    )
    results: list[dict[str, Any]] = []
    for query_value in queries:
        if not isinstance(query_value, dict):
            results.append(
                _unknown_surface_query({}, None, "invalid_surface_query")
            )
            continue
        if "envelope_id" in query_value:
            envelope = envelope_index.get(
                (
                    query_value.get("envelope_id"),
                    query_value.get("envelope_version"),
                )
            )
            if envelope is None:
                results.append(
                    _unknown_candidate_envelope_query(
                        query_value,
                        None,
                        "candidate_envelope_version_not_found",
                        [],
                    )
                )
            elif envelope.get("cohort_id") != document.get("cohort_id"):
                results.append(
                    _unknown_candidate_envelope_query(
                        query_value,
                        envelope,
                        "surface_cohort_mismatch",
                        [],
                    )
                )
            else:
                results.append(
                    _query_candidate_envelope(
                        query_value, envelope, surface_index
                    )
                )
            continue
        surface = surface_index.get(
            (query_value.get("surface_id"), query_value.get("surface_version"))
        )
        if surface is None:
            results.append(
                _unknown_surface_query(
                    query_value, None, "surface_version_not_found"
                )
            )
            continue
        if surface.get("cohort_id") != document.get("cohort_id"):
            results.append(
                _unknown_surface_query(
                    query_value, surface, "surface_cohort_mismatch"
                )
            )
            continue
        if surface.get("qualification_status") == "rejected":
            results.append(
                _unknown_surface_query(
                    query_value, surface, "qualification_rejected"
                )
            )
            continue
        if surface.get("qualification_status") == "unknown":
            results.append(
                _unknown_surface_query(
                    query_value,
                    surface,
                    str(
                        surface.get(
                            "qualification_reason_code",
                            "qualification_unknown",
                        )
                    ),
                )
            )
            continue
        results.append(_query_capability_surface(query_value, surface))
    summaries = []
    for surface in valid_surfaces:
        previous_version = surface.get("previous_version")
        previous = (
            surface_index.get((surface["surface_id"], previous_version))
            if _nonempty_string(previous_version)
            else None
        )
        summaries.append(_surface_summary(surface, previous))
    envelope_summaries = [
        _candidate_envelope_summary(envelope, surface_index)
        for envelope in valid_envelopes
    ]
    return summaries, envelope_summaries, results


def diagnose_run_bundle(path: str | Path) -> dict[str, Any]:
    """Derive one deterministic exact-Shape diagnosis from a Run Bundle."""

    root = Path(path).resolve()
    manifest, document = _load_evidence(root)
    digests = _verified_document_digests(document)
    verified_artifacts = _verified_bundle_artifacts(root, manifest)
    if _complete_required_identity(document):
        resource = _resource_axis(document, verified_artifacts)
        operator = _operator_axis(document)
        schedule = _schedule_axis(document, operator)
        observation = _observation_axis(document)
        axes = {
            "resource_physical_floor": resource,
            "operator_achievable_frontier": operator,
            "schedule_achievable_frontier": schedule,
            "observation": observation,
        }
    else:
        axes = {
            axis: _unknown("incomplete-diagnostic-identity")
            for axis in (
                "resource_physical_floor",
                "operator_achievable_frontier",
                "schedule_achievable_frontier",
                "observation",
            )
        }
    comparisons = _comparisons(axes)
    diagnostic_trigger = _diagnostic_trigger(document, verified_artifacts)
    shape_probes = _shape_disambiguation_probes(
        document,
        diagnostic_trigger,
        verified_artifacts,
    )
    verdict_policy, performance_verdicts = _performance_diagnosis_verdicts(
        document,
        run_id=manifest["run_id"],
        trigger=diagnostic_trigger,
        probes=shape_probes,
    )
    frontier_anchor_lifecycles = _frontier_anchor_lifecycles(document)
    (
        capability_surfaces,
        candidate_envelopes,
        surface_queries,
    ) = _capability_surface_results(document)
    adapter_contract = _adapter_contract(
        document, capability_surfaces, axes["operator_achievable_frontier"]
    )
    derivation_basis = {
        "schema": DIAGNOSTIC_RESULT_SCHEMA,
        "run_id": manifest["run_id"],
        "digests": digests,
        "axes": axes,
        "comparisons": comparisons,
        "policy_refs": document.get("policies", {}),
        "frontier_anchor_lifecycles": frontier_anchor_lifecycles,
        "capability_surfaces": capability_surfaces,
        "candidate_envelopes": candidate_envelopes,
        "capability_surface_queries": surface_queries,
        "adapter_contract": adapter_contract,
        "source_runs": list(document.get("source_runs", [])),
    }
    if diagnostic_trigger is not None:
        derivation_basis["diagnostic_trigger"] = diagnostic_trigger
    if shape_probes:
        derivation_basis["shape_disambiguation_probes"] = shape_probes
    if verdict_policy is not None:
        derivation_basis["verdict_policy"] = verdict_policy
        derivation_basis[
            "performance_diagnosis_verdicts"
        ] = performance_verdicts
    derivation = {
        "derivation_id": _canonical_digest(derivation_basis),
        "inputs": ["diagnostic-evidence"],
        "steps": [
            "verify-artifact-and-authored-digests",
            "qualify-exact-shape-active-anchor",
            "compose-explicit-single-node-schedule",
            "project-four-independent-axes",
            "query-versioned-capability-surfaces",
        ],
    }
    if diagnostic_trigger is not None:
        derivation["steps"].append("evaluate-diagnostic-trigger")
    if shape_probes:
        derivation["steps"].append("evaluate-exact-shape-probes")
    if verdict_policy is not None:
        derivation["steps"].append("evaluate-performance-verdicts")
    result = {
        "schema": DIAGNOSTIC_RESULT_SCHEMA,
        "run_id": manifest["run_id"],
        "status": (
            "complete"
            if all(axis["status"] == "known" for axis in axes.values())
            and all(
                query["status"] != "unknown" for query in surface_queries
            )
            and (
                adapter_contract is None
                or adapter_contract.get("status") == "eligible"
            )
            else "partial"
        ),
        "axes": axes,
        "comparisons": comparisons,
        "frontier_anchor_lifecycles": frontier_anchor_lifecycles,
        "capability_surfaces": capability_surfaces,
        "candidate_envelopes": candidate_envelopes,
        "capability_surface_queries": surface_queries,
        "evidence": {
            key: document[key]
            for key in (
                "resolved_configuration",
                "resolved_ir",
                "hardware",
                "cohort_id",
                "execution_domain",
                "candidate",
                "correctness",
                "environment",
                "baseline_timing_lane",
                "policies",
                "measurement_adapter",
                "measurement_capability_manifest",
                "diagnostic_profiling_lane",
                "cohort_evidence",
                "timing_plan",
                "frontier_anchors",
                "single_node_schedule",
            )
            if key in document
        },
        "digests": digests,
        "derivation": derivation,
        "source_runs": list(document.get("source_runs", [])),
    }
    if diagnostic_trigger is not None:
        result["diagnostic_trigger"] = diagnostic_trigger
    if shape_probes:
        result["shape_disambiguation_probes"] = shape_probes
    if verdict_policy is not None:
        result["verdict_policy"] = verdict_policy
        result["verdict_vocabulary"] = list(
            PERFORMANCE_DIAGNOSIS_VERDICTS
        )
        result["performance_diagnosis_verdicts"] = performance_verdicts
    if adapter_contract is not None:
        result["adapter_contract"] = adapter_contract
    return result


def render_diagnostic_report(result: dict[str, Any]) -> str:
    """Project a diagnostic result without creating a second derivation."""

    labels = {
        "resource_physical_floor": "Resource Physical Floor",
        "operator_achievable_frontier": "Operator Achievable Frontier",
        "schedule_achievable_frontier": "Schedule Achievable Frontier",
        "observation": "Observation",
    }
    lines = [
        f"run {result['run_id']}: {result['status']}",
        f"derivation: {result['derivation']['derivation_id']}",
    ]
    for key, label in labels.items():
        axis = result["axes"][key]
        if axis["status"] == "known":
            lines.append(f"{label}: {axis['value_ns'] / 1_000_000:.3f} ms")
        else:
            lines.append(
                f"{label}: unknown ({axis['reason_code']})"
            )
    for lifecycle in result.get("frontier_anchor_lifecycles", []):
        state = lifecycle.get("state", {})
        lines.append(
            f"Frontier Anchor {lifecycle['anchor_id']}: "
            f"{state.get('observation_validity', 'unknown')}/"
            f"{state.get('frontier_role', 'unknown')}; "
            "authoritative="
            f"{str(lifecycle.get('authoritative_surface_knot', False)).lower()}; "
            f"history={lifecycle.get('history_status', 'unknown')}"
        )
    for query in result.get("capability_surface_queries", []):
        envelope = query.get("envelope")
        if isinstance(envelope, dict):
            prefix = (
                f"Capability Envelope {query['query_id']} "
                f"[{envelope['envelope_id']}@{envelope['version']}]"
            )
        else:
            surface = query["surface"]
            prefix = (
                f"Capability Surface {query['query_id']} "
                f"[{surface['surface_id']}@{surface['version']}]"
            )
        if query["status"] == "unknown":
            details = []
            if query.get("cohort_id") is not None:
                details.append(f"cohort={query['cohort_id']}")
            if query.get("domain") is not None:
                details.append(
                    "domain="
                    + json.dumps(
                        query["domain"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
            domain_policy = query.get("domain_policy")
            if isinstance(domain_policy, dict):
                details.append(
                    "domain-policy="
                    f"{domain_policy['policy_id']}/{domain_policy['version']}"
                )
            if query.get("cell_id") is not None:
                details.append(f"cell={query['cell_id']}")
            if query.get("support_seam_id") is not None:
                details.append(f"seam={query['support_seam_id']}")
            facets = query.get("candidate_facets")
            if isinstance(facets, list) and facets:
                facet_texts = []
                for facet in facets:
                    facet_status = str(facet["status"])
                    if facet_status == "unknown":
                        facet_status += f"({facet['reason_code']})"
                    facet_details = []
                    if facet.get("cell_id") is not None:
                        facet_details.append(f"cell={facet['cell_id']}")
                    if facet.get("support_seam_id") is not None:
                        facet_details.append(
                            f"seam={facet['support_seam_id']}"
                        )
                    detail_suffix = (
                        "[" + ";".join(facet_details) + "]"
                        if facet_details
                        else ""
                    )
                    facet_texts.append(
                        f"{facet['candidate_family']}/"
                        f"{facet['algorithm_family']}:{facet_status}"
                        f"{detail_suffix}"
                    )
                details.append("facets=" + ",".join(facet_texts))
            evidence_refs = query.get("evidence_refs")
            if isinstance(evidence_refs, list) and evidence_refs:
                details.append("evidence=" + ",".join(evidence_refs))
            suffix = "; " + "; ".join(details) if details else ""
            lines.append(
                f"{prefix}: unknown ({query['reason_code']}){suffix}"
            )
            continue
        rate = query["effective_rate"]
        latency = query["work_rate_latency"]
        domain_policy = query.get("domain_policy")
        domain_policy_text = (
            f"; domain-policy={domain_policy['policy_id']}/{domain_policy['version']}"
            if isinstance(domain_policy, dict)
            else ""
        )
        if isinstance(envelope, dict):
            candidate_text = "; candidates=" + ",".join(
                f"{facet['candidate_family']}/{facet['algorithm_family']}"
                for facet in query["candidate_facets"]
            )
            winner_text = (
                f"; winner={query['selected_candidate_family']}/"
                f"{query['selected_algorithm_family']}"
            )
        else:
            candidate_text = (
                f"; candidate-family={query['selected_candidate_family']}"
            )
            winner_text = ""
        response = query.get("response")
        mfu = query.get("mfu")
        empirical_utilization = query.get("empirical_envelope_utilization")
        mfu_text = (
            f"MFU={float(mfu['value']):.9f}"
            if isinstance(mfu, dict) and mfu.get("status") == "derived"
            else f"MFU=unknown ({mfu.get('reason_code')})"
            if isinstance(mfu, dict)
            else "MFU=unknown (contract-unavailable)"
        )
        empirical_text = (
            "empirical-envelope-utilization="
            f"{float(empirical_utilization['value']):.9f}"
            if isinstance(empirical_utilization, dict)
            and empirical_utilization.get("status") == "derived"
            else "empirical-envelope-utilization=unknown"
        )
        shape_regime = query.get("shape_regime")
        if (
            isinstance(response, dict)
            and isinstance(shape_regime, dict)
            and response.get("kind") == "setup-plus-throughput"
        ):
            response_text = (
                f"response={response['kind']}/{response['version']}; "
                f"Setup Latency={response['setup_latency_ns']:.6f} ns; "
                f"asymptotic-rate={response['asymptotic_rate'] / 1_000_000_000_000:.9f} TFLOP/s; "
                f"Shape Regime={shape_regime['identity']}/{shape_regime['classification']}; "
            )
        elif isinstance(response, dict):
            response_text = (
                f"primary-response={response['primary_response']}; "
                f"response={response['value_ns']:.6f} ns; "
                f"response-identity={response['response_identity']}; "
                f"shape-regime={response['shape_regime_identity']}; "
            )
        else:
            response_text = ""
        lines.append(
            f"{prefix}: {query['status']}; "
            f"cohort={query['cohort_id']}; "
            f"domain={json.dumps(query['domain'], ensure_ascii=False, separators=(',', ':'), sort_keys=True)}; "
            f"{candidate_text.lstrip('; ')}"
            f"{winner_text}"
            f"{domain_policy_text}; "
            f"{response_text}"
            f"work-formula={json.dumps(query.get('work_formula'), ensure_ascii=False, separators=(',', ':'), sort_keys=True)}; "
            f"declared-work={latency['declared_work']:.6f} {latency['work_unit']}; "
            f"rate={rate['value'] / 1_000_000_000_000:.9f} TFLOP/s; "
            f"{mfu_text}; {empirical_text}; "
            f"latency={latency['value_ns']:.6f} ns; "
            f"cell={query['cell_id']}; "
            f"anchors={','.join(anchor['anchor_id'] for anchor in query['anchors'])}; "
            f"weights={json.dumps(query['weights'], separators=(',', ':'))}; "
            f"uncertainty={json.dumps(query['uncertainty']['components'], separators=(',', ':'), sort_keys=True)}; "
            f"evidence={','.join(query.get('evidence_refs', []))}"
        )
    trigger = result.get("diagnostic_trigger")
    if isinstance(trigger, dict):
        if trigger.get("status") == "evaluated":
            policy = trigger["policy"]
            lines.extend(
                [
                    "Diagnostic Trigger "
                    f"[{policy['policy_id']}/{policy['version']}]: evaluated",
                    "predicted Top 10: "
                    + ", ".join(
                        item["stable_path"]
                        for item in trigger["predicted_top10"]
                    ),
                    "observed Top 10: "
                    + ", ".join(
                        item["stable_path"]
                        for item in trigger["observed_top10"]
                    ),
                    "triggered: "
                    + (
                        ", ".join(
                            item["stable_path"]
                            for item in trigger["triggered"]
                        )
                        or "none"
                    ),
                ]
            )
            for item in trigger.get("evaluated", []):
                basis = item.get("observation_basis")
                if isinstance(basis, dict):
                    lines.append(
                        "trigger observation basis: "
                        f"{item['stable_path']}; "
                        + json.dumps(
                            basis,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    )
        else:
            lines.append(
                "Diagnostic Trigger: unknown "
                f"({trigger.get('reason_code', 'unknown')})"
            )
    for probe in result.get("shape_disambiguation_probes", []):
        lines.append(
            "Shape Disambiguation Probe "
            f"{probe.get('probe_id', 'unknown')}: "
            f"{probe.get('status', 'unknown')}"
        )
        contract = probe.get("locked_contract")
        if isinstance(contract, dict):
            lines.append(
                "locked contract: "
                f"semantic={contract['semantic']}; "
                "shape="
                + json.dumps(
                    contract["shape"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + f"; dtype={contract['dtype']}; layout={contract['layout']}; "
                "strides="
                + json.dumps(
                    contract["strides"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + f"; alignment={contract['alignment_bytes']}; "
                f"threads={contract['threads']}; "
                f"cohort={contract['cohort_id']}; "
                "candidates="
                + ",".join(contract["candidate_ids"])
                + "; completion="
                + json.dumps(
                    contract["completion_boundary"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        best = probe.get("best_of_correct")
        if isinstance(best, dict):
            lines.append(
                "correctness -> best-of-correct: "
                f"{best['candidate_id']}"
            )
    for verdict in result.get("performance_diagnosis_verdicts", []):
        lines.append(
            "Performance Diagnosis Verdict "
            f"{verdict['stable_path']}: {verdict['verdict']}"
        )
        gates = verdict["gates"]
        for state in ("satisfied", "failed", "not_evaluated"):
            values = gates[state]
            rendered = ", ".join(
                gate["gate_id"]
                + (
                    f"({gate['reason_code']})"
                    if gate.get("reason_code") is not None
                    else ""
                )
                for gate in values
            )
            lines.append(f"{state} gates: {rendered or 'none'}")
        lines.append("bundle refs: " + ", ".join(verdict["bundle_refs"]))
        counterexamples = verdict["counterexamples"]
        rendered_counterexamples = ", ".join(
            str(
                counterexample.get(
                    "candidate_id",
                    counterexample.get("counterexample_id", "unknown"),
                )
            )
            + "("
            + str(
                counterexample.get(
                    "reason_code",
                    ",".join(counterexample.get("reason_codes", [])),
                )
            )
            + ")"
            for counterexample in counterexamples
        )
        lines.append(
            "counterexamples: " + (rendered_counterexamples or "none")
        )
        direct_defect = verdict.get("direct_defect_evidence")
        if isinstance(direct_defect, dict):
            input_summary = direct_defect["input_summary"]
            candidate_identity = direct_defect["candidate_identity"]
            failure = direct_defect["failure"]
            lines.append(
                f"direct defect: {direct_defect['defect_kind']}; "
                f"candidate={candidate_identity['candidate_id']}; "
                f"input-sha256={input_summary['input_sha256']}"
            )
            lines.append(
                f"direct failure: {failure['failure_kind']}; "
                f"evidence={failure['evidence_ref']}"
            )
            lines.append(
                "direct environment: cohort="
                f"{direct_defect['environment']['cohort_id']}"
            )
            lines.append(
                "direct repetitions: "
                + ", ".join(
                    f"{repetition['session_id']}@{repetition['process_id']}="
                    f"{repetition['outcome']}"
                    for repetition in direct_defect["repetitions"]
                )
            )
        metrics = verdict.get("metrics")
        if (
            verdict.get("verdict") == "integration_overhead"
            and isinstance(metrics, dict)
        ):
            lines.append(
                "integration overhead: "
                f"standalone={metrics['standalone_operator_ns'] / 1_000_000:.6f} ms; "
                f"wrapped={metrics['wrapped_e2e_ns'] / 1_000_000:.6f} ms; "
                f"excess={metrics['measured_excess_ns'] / 1_000_000:.6f} ms; "
                f"recovered={metrics['recovered_ns'] / 1_000_000:.6f} ms; "
                f"recovery-error={metrics['recovery_error_fraction']:.2%}"
            )
            for ablation in verdict.get("ablations", []):
                lines.append(
                    f"ablation {ablation['ablation_id']}({ablation['kind']}): "
                    f"{ablation['aggregate_latency_ns'] / 1_000_000:.6f} ms; "
                    "removed=" + ",".join(ablation["removed_leaf_ids"])
                )
            ledger = verdict.get("ledger")
            if isinstance(ledger, dict):
                lines.append(
                    "exclusive ledger "
                    f"{ledger['ledger_id']}@{ledger['version']}: "
                    f"{ledger['status']}; leaves={len(ledger['leaves'])}; "
                    f"residual={ledger['residual']['duration_ns'] / 1_000_000:.6f} ms; "
                    "parents-included="
                    f"{ledger['parent_span_total_included_ns'] / 1_000_000:.6f} ms"
                )
        surface_action = verdict.get("surface_action")
        if (
            isinstance(surface_action, dict)
            and surface_action.get("action") == "preserve"
            and isinstance(surface_action.get("surface"), dict)
            and isinstance(
                surface_action.get("operator_achievable_frontier_ns"),
                dict,
            )
        ):
            surface = surface_action["surface"]
            frontier = surface_action["operator_achievable_frontier_ns"]
            surface_identity = (
                f"{surface['surface_id']}@{surface['version']}"
                if _surface_id_text(surface.get("surface_id"))
                and _surface_version_text(surface.get("version"))
                else "unknown-surface"
            )
            lines.append(
                "Operator Achievable Frontier preserved: "
                f"{surface_identity}; "
                f"before={frontier['before'] / 1_000_000:.6f} ms; "
                f"after={frontier['after'] / 1_000_000:.6f} ms"
            )
    evidence = result["evidence"]
    hardware_value = evidence.get("hardware")
    hardware = hardware_value if isinstance(hardware_value, dict) else {}
    configuration_value = evidence.get("resolved_configuration")
    configuration = (
        configuration_value if isinstance(configuration_value, dict) else {}
    )
    resolved_ir_value = evidence.get("resolved_ir")
    resolved_ir = resolved_ir_value if isinstance(resolved_ir_value, dict) else {}
    lane_value = evidence.get("baseline_timing_lane")
    lane = lane_value if isinstance(lane_value, dict) else {}
    candidate_value = evidence.get("candidate")
    candidate = candidate_value if isinstance(candidate_value, dict) else {}
    policies = evidence.get("policies")
    qualification = (
        policies.get("qualification") if isinstance(policies, dict) else None
    )
    best_of_correct = candidate.get("exact_shape_best_of_correct")
    lines.extend(
        [
            "hardware/cohort: "
            f"{hardware.get('device', 'unknown')} / "
            f"{evidence.get('cohort_id', 'unknown')}",
            "execution domain: "
            + json.dumps(
                evidence["execution_domain"],
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "resolved config/IR: "
            f"{configuration.get('analysis_plan', 'unknown')}:"
            f"{configuration.get('benchmark_case', 'unknown')} -> "
            f"{resolved_ir.get('semantic_node', 'unknown')}",
            "candidate/baseline lane: "
            f"{candidate.get('candidate_id', 'unknown')} / "
            f"{lane.get('lane_id', 'unknown')}",
            "qualification policy: "
            + (
                f"{qualification.get('policy_id', 'unknown')}/"
                f"{qualification.get('version', 'unknown')}"
                if isinstance(qualification, dict)
                else "unknown"
            ),
            "candidate search: "
            + (
                f"winner={best_of_correct.get('winner_candidate_id', 'unknown')}; "
                "eligible="
                + ",".join(best_of_correct.get("eligible_candidate_ids", []))
                + "; sessions="
                + ",".join(best_of_correct.get("search_session_ids", []))
                + f"; evidence={best_of_correct.get('evidence_ref', 'unknown')}"
                if isinstance(best_of_correct, dict)
                else "unknown"
            ),
            f"raw bundle: run-bundle://{result['run_id']}",
            f"input digest: {result['digests']['input_sha256']}",
            f"evidence digest: {result['digests']['evidence_sha256']}",
        ]
    )
    for source_run in result.get("source_runs", []):
        if isinstance(source_run, dict):
            lines.append(
                f"source run {source_run.get('run_id', 'unknown')}: "
                f"{source_run.get('role', 'unknown')}; "
                f"path={source_run.get('path', 'unknown')}; "
                "manifest-sha256="
                f"{source_run.get('manifest_sha256', 'unknown')}"
            )
    lines.append(
        "Resource Physical Floor distance is optimization headroom; "
        "not prediction error."
    )
    diagnostic_lane_value = evidence.get("diagnostic_profiling_lane")
    measurement_lanes = (
        ("baseline", lane),
        (
            "diagnostic",
            diagnostic_lane_value
            if isinstance(diagnostic_lane_value, dict)
            else {},
        ),
    )
    for lane_label, measurement_lane in measurement_lanes:
        completion_value = measurement_lane.get("completion_boundary")
        completion = (
            completion_value if isinstance(completion_value, dict) else {}
        )
        timer_value = measurement_lane.get("timer")
        timer = timer_value if isinstance(timer_value, dict) else {}
        completion_parts = [
            f"kind={completion.get('kind', 'unknown')}",
            f"timer={timer.get('source', 'unknown')}",
        ]
        if completion.get("kind") == "device-event-stream-completion":
            completion_parts.append(
                f"stream={completion.get('stream_id', 'unknown')}"
            )
            completion_parts.append(
                f"event={completion.get('device_event_id', 'unknown')}"
            )
        elif completion.get("kind") == "distributed-rank-local-duration":
            completion_parts.append(
                f"reducer={completion.get('duration_reducer', 'unknown')}"
            )
            completion_parts.append(
                f"clock-domain={timer.get('clock_domain', 'unknown')}"
            )
        lines.append(
            f"{lane_label} completion: " + "; ".join(completion_parts)
        )
    adapter_contract = result.get("adapter_contract")
    if isinstance(adapter_contract, dict):
        protocol = adapter_contract.get("protocol")
        if isinstance(protocol, dict) and _nonempty_string(
            adapter_contract.get("adapter_id")
        ):
            lines.append(
                "measurement adapter: "
                f"{adapter_contract['adapter_id']}@"
                f"{adapter_contract.get('adapter_version', 'unknown')}; "
                f"protocol={protocol.get('protocol_id', 'unknown')}@"
                f"{protocol.get('protocol_version', 'unknown')}; "
                f"status={adapter_contract.get('status', 'unknown')}"
            )
        cohort = adapter_contract.get("cohort")
        if isinstance(cohort, dict):
            changed_dimensions = cohort.get("changed_dimensions")
            changes = (
                ",".join(str(item) for item in changed_dimensions)
                if isinstance(changed_dimensions, list) and changed_dimensions
                else "none"
            )
            retry = cohort.get("retry")
            retry_status = (
                retry.get("status", "unknown")
                if isinstance(retry, dict)
                else "unknown"
            )
            lines.append(
                "hardware validity cohort: "
                f"{cohort.get('status', 'unknown')}; "
                f"current={cohort.get('cohort_id', 'unknown')}; "
                "reference="
                f"{cohort.get('reference_cohort_id', 'unknown')}; "
                f"changes={changes}; retry={retry_status}"
            )
        admission = adapter_contract.get("anchor_admission")
        if isinstance(admission, dict):
            reasons = admission.get("reason_codes")
            reason_suffix = (
                "; reasons=" + ",".join(str(item) for item in reasons)
                if isinstance(reasons, list) and reasons
                else ""
            )
            lines.append(
                "anchor admission: "
                f"{admission.get('status', 'unknown')}{reason_suffix}"
            )
        observation_fields = adapter_contract.get("observation_fields")
        if isinstance(observation_fields, list):
            field_statuses = sorted(
                f"{field.get('field', 'unknown')}="
                f"{field.get('status', 'unknown')}"
                for field in observation_fields
                if isinstance(field, dict)
            )
            lines.append(
                "observation field statuses: " + ", ".join(field_statuses)
            )
        lanes = adapter_contract.get("lanes")
        if isinstance(lanes, dict):
            lines.append(
                "measurement lanes: "
                f"pair={lanes['pair_id']}; "
                f"baseline={lanes['baseline_lane_id']}; "
                f"diagnostic={lanes['diagnostic_lane_id']}; "
                "diagnostic-frontier-eligible="
                f"{str(lanes['diagnostic_frontier_eligible']).lower()}; "
                f"reason={lanes['reason_code']}"
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "DiagnosticBundleError",
    "DiagnosticBundleIntegrityError",
    "diagnose_run_bundle",
    "render_diagnostic_report",
]
