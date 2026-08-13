"""Evidence-qualified alias and materialization decisions for layout operations."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from groundupscale.ir import content_fingerprint


SCHEMA = "groundupscale.dev/alias-materialization-evidence/v1alpha1"


def _valid_tensor_contract(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    shape = value.get("shape")
    stride = value.get("stride")
    return bool(
        isinstance(value.get("device"), str)
        and value.get("device")
        and isinstance(value.get("dtype"), str)
        and value.get("dtype")
        and isinstance(value.get("layout"), str)
        and value.get("layout")
        and isinstance(shape, (list, tuple))
        and shape
        and all(isinstance(item, int) and item >= 0 for item in shape)
        and isinstance(stride, (list, tuple))
        and len(stride) == len(shape)
        and all(isinstance(item, int) and item >= 0 for item in stride)
    )


def build_alias_materialization_evidence(
    *,
    alias_audits: Sequence[Mapping[str, object]],
    expected_operations: Sequence[Mapping[str, object]],
    selected_candidates: Mapping[str, object],
    execution_mode: str,
    hardware_cohort: str,
) -> dict[str, Any]:
    """Bind runtime storage audits to selected layout-operation candidates."""

    audits = {
        str(audit.get("stable_path")): dict(audit)
        for audit in alias_audits
        if isinstance(audit.get("stable_path"), str)
    }
    operations: list[dict[str, Any]] = []
    for expected in expected_operations:
        stable_path = str(expected["stable_path"])
        audit = audits.get(stable_path)
        selected = selected_candidates.get(stable_path)
        selected_document = dict(selected) if isinstance(selected, Mapping) else {}
        candidate_id = (
            selected_document.get("candidate_id")
            if selected_document
            else selected
        )
        candidate = {
            "candidate_id": candidate_id,
            "execution_mode": execution_mode,
            "hardware_cohort": hardware_cohort,
        }
        audited = bool(
            audit is not None
            and audit.get("operation") == expected.get("operation")
            and _valid_tensor_contract(audit.get("input_contract"))
            and _valid_tensor_contract(audit.get("output_contract"))
            and isinstance(candidate_id, str)
            and candidate_id
        )
        aliases = bool(
            audited
            and audit.get("input_storage_identity")
            == audit.get("output_storage_identity")
        )
        materializes = bool(
            audited
            and audit.get("input_storage_identity")
            != audit.get("output_storage_identity")
        )
        duration = selected_document.get("duration_ns")
        evidence_refs = selected_document.get("evidence_refs")
        materialization_qualified = bool(
            materializes
            and isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and duration >= 0
            and isinstance(evidence_refs, list)
            and evidence_refs
            and all(isinstance(reference, str) and reference for reference in evidence_refs)
        )
        event_id = (
            "physical-event:"
            + content_fingerprint(
                stable_path, candidate, audit, duration, evidence_refs
            )
            if materialization_qualified
            else None
        )
        physical_event = (
            {
                "event_id": event_id,
                "stable_path": stable_path,
                "kind": "materialization",
                "candidate_id": candidate_id,
                "duration_ns": duration,
                "resource_claims": [
                    {
                        "resource_id": "memory.interface",
                        "kind": "throughput",
                        "read_bytes": expected["logical_read_bytes"],
                        "write_bytes": expected["logical_write_bytes"],
                        "lifetime": {"start": event_id, "end": event_id},
                        "provenance": {"evidence_refs": evidence_refs},
                    }
                ],
                "provenance": {"evidence_refs": evidence_refs},
            }
            if materialization_qualified
            else None
        )
        decision = (
            "alias-preserving"
            if aliases
            else "materialization"
            if materialization_qualified
            else "unknown"
        )
        operations.append(
            {
                "stable_path": stable_path,
                "operation": expected["operation"],
                "selected_candidate": candidate,
                "alias_audit": audit,
                "decision": decision,
                "resource_demand": (
                    {
                        "status": "known",
                        "memory_read_bytes": 0,
                        "memory_write_bytes": 0,
                    }
                    if aliases
                    else {
                        "status": "known",
                        "memory_read_bytes": expected["logical_read_bytes"],
                        "memory_write_bytes": expected["logical_write_bytes"],
                    }
                    if materialization_qualified
                    else {"status": "unknown"}
                ),
                "duration": (
                    {
                        "status": "known",
                        "value_ns": 0,
                        "evidence_kind": "verified-alias-preserving-candidate",
                    }
                    if aliases
                    else {
                        "status": "known",
                        "value_ns": duration,
                        "evidence_refs": evidence_refs,
                    }
                    if materialization_qualified
                    else {"status": "unknown", "value_ns": None}
                ),
                "physical_event": physical_event,
            }
        )
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "qualified"
            if operations and all(item["decision"] != "unknown" for item in operations)
            else "unknown"
        ),
        "hardware_cohort": hardware_cohort,
        "execution_mode": execution_mode,
        "operations": operations,
        "schedule": {
            "physical_events": [
                item["physical_event"]
                for item in operations
                if item["physical_event"] is not None
            ]
        },
        "decomposition": {
            "materialization_duration_ns": sum(
                item["duration"]["value_ns"]
                for item in operations
                if item["decision"] == "materialization"
            ),
            "alias_duration_ns": 0,
            "unknown_stable_paths": [
                item["stable_path"]
                for item in operations
                if item["decision"] == "unknown"
            ],
        },
    }
    document["evidence_version_id"] = content_fingerprint(document)
    return document


def verify_alias_materialization_evidence(
    document: Mapping[str, object],
) -> dict[str, Any]:
    """Verify public alias/materialization evidence without trusting summaries."""

    failures: list[str] = []
    body = dict(document)
    evidence_version_id = body.pop("evidence_version_id", None)
    if evidence_version_id != content_fingerprint(body):
        failures.append("evidence version digest mismatch")
    cohort = body.get("hardware_cohort")
    execution_mode = body.get("execution_mode")
    operations = body.get("operations")
    if body.get("schema") != SCHEMA or not isinstance(operations, list):
        failures.append("invalid alias materialization evidence schema")
        operations = []
    expected_events: list[object] = []
    materialization_duration = 0
    unknown_paths: list[object] = []
    for operation in operations:
        if not isinstance(operation, dict):
            failures.append("invalid alias materialization operation")
            continue
        candidate = operation.get("selected_candidate")
        audit = operation.get("alias_audit")
        decision = operation.get("decision")
        demand = operation.get("resource_demand")
        duration = operation.get("duration")
        event = operation.get("physical_event")
        if (
            not isinstance(candidate, dict)
            or not isinstance(candidate.get("candidate_id"), str)
            or not candidate.get("candidate_id")
            or candidate.get("hardware_cohort") != cohort
            or candidate.get("execution_mode") != execution_mode
        ):
            failures.append("selected candidate execution domain mismatch")
        if decision == "alias-preserving":
            if (
                not isinstance(audit, dict)
                or not _valid_tensor_contract(audit.get("input_contract"))
                or not _valid_tensor_contract(audit.get("output_contract"))
                or audit.get("input_storage_identity")
                != audit.get("output_storage_identity")
                or demand
                != {
                    "status": "known",
                    "memory_read_bytes": 0,
                    "memory_write_bytes": 0,
                }
                or duration
                != {
                    "status": "known",
                    "value_ns": 0,
                    "evidence_kind": "verified-alias-preserving-candidate",
                }
                or event is not None
            ):
                failures.append("unverified alias zero")
        elif decision == "materialization":
            if (
                not isinstance(audit, dict)
                or not _valid_tensor_contract(audit.get("input_contract"))
                or not _valid_tensor_contract(audit.get("output_contract"))
                or audit.get("input_storage_identity")
                == audit.get("output_storage_identity")
                or not isinstance(demand, dict)
                or demand.get("status") != "known"
                or not isinstance(duration, dict)
                or duration.get("status") != "known"
                or not isinstance(event, dict)
                or event.get("duration_ns") != duration.get("value_ns")
                or event.get("stable_path") != operation.get("stable_path")
            ):
                failures.append("invalid materialization event")
            else:
                expected_events.append(event)
                materialization_duration += duration["value_ns"]
        elif decision == "unknown":
            unknown_paths.append(operation.get("stable_path"))
            if (
                demand != {"status": "unknown"}
                or duration != {"status": "unknown", "value_ns": None}
                or event is not None
            ):
                failures.append("invalid structured unknown")
        else:
            failures.append("invalid alias materialization decision")
    if body.get("schedule") != {"physical_events": expected_events}:
        failures.append("physical event schedule mismatch")
    if body.get("decomposition") != {
        "materialization_duration_ns": materialization_duration,
        "alias_duration_ns": 0,
        "unknown_stable_paths": unknown_paths,
    }:
        failures.append("alias materialization decomposition mismatch")
    expected_status = (
        "qualified"
        if operations and not unknown_paths
        else "unknown"
    )
    if body.get("status") != expected_status:
        failures.append("alias materialization status mismatch")
    return {"passed": not failures, "failures": failures}


__all__ = [
    "build_alias_materialization_evidence",
    "verify_alias_materialization_evidence",
]
