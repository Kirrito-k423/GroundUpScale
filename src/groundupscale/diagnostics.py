"""Evidence-qualified diagnosis for one exact-Shape Run Bundle."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from typing import Any

from groundupscale.run_bundle import verify_run_bundle


DIAGNOSTIC_EVIDENCE_SCHEMA = (
    "groundupscale.dev/diagnostic-evidence/v1alpha1"
)
DIAGNOSTIC_RESULT_SCHEMA = (
    "groundupscale.dev/diagnostic-result/v1alpha1"
)

_INPUT_KEYS = (
    "resolved_configuration",
    "resolved_ir",
    "hardware",
    "cohort_id",
    "execution_domain",
)
_EVIDENCE_KEYS = (
    "candidate",
    "correctness",
    "environment",
    "baseline_timing_lane",
    "frontier_anchors",
    "resource_physical_floor",
    "single_node_schedule",
    "policies",
)


class DiagnosticBundleError(ValueError):
    """The Run Bundle cannot produce a trustworthy diagnostic result."""


class DiagnosticBundleIntegrityError(DiagnosticBundleError):
    """A manifest or authored evidence digest did not verify."""


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _unknown(reason_code: str) -> dict[str, Any]:
    return {
        "status": "unknown",
        "reason_code": reason_code,
        "evidence_refs": [],
    }


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


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
    return policy


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
        key: document[key] for key in _EVIDENCE_KEYS if key in document
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


def _resource_axis(document: dict[str, Any]) -> dict[str, Any]:
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
    return {
        "status": "known",
        "value_ns": floor["value_ns"],
        "may_be_unattainable": True,
        "evidence_refs": list(floor.get("evidence_refs", [])),
    }


def _eligible_anchor(
    document: dict[str, Any], anchor: dict[str, Any]
) -> bool:
    candidate = document.get("candidate")
    correctness = document.get("correctness")
    environment = document.get("environment")
    policies = document.get("policies")
    if not all(
        isinstance(value, dict)
        for value in (candidate, correctness, environment, policies)
    ):
        return False
    qualification = _versioned_policy(document, "qualification")
    if qualification is None:
        return False
    minimum_sessions = qualification.get("minimum_independent_sessions")
    if not isinstance(minimum_sessions, int) or minimum_sessions < 2:
        return False
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
    return (
        anchor.get("observation_validity") == "QUALIFIED"
        and anchor.get("frontier_role") == "ACTIVE"
        and anchor.get("candidate_id") == candidate_id
        and anchor.get("cohort_id") == document["cohort_id"]
        and anchor.get("execution_domain") == document["execution_domain"]
        and correctness.get("passed") is True
        and anchor.get("correctness_passed") is True
        and environment.get("eligible") is True
        and _nonempty_string(anchor.get("anchor_id"))
        and _nonempty_string(anchor.get("baseline_lane_id"))
        and _nonempty_string(anchor.get("instrumentation_profile"))
        and isinstance(completion, dict)
        and _nonempty_string(completion.get("kind"))
        and completion.get("closed") is True
        and completion.get("threadpool_joined") is True
        and isinstance(timer, dict)
        and _nonempty_string(timer.get("source"))
        and isinstance(timer.get("resolution_ns"), (int, float))
        and timer["resolution_ns"] > 0
        and isinstance(warmup, dict)
        and warmup.get("converged") is True
        and isinstance(raw_timing, list)
        and bool(raw_timing)
        and all(
            isinstance(sample, (int, float)) and sample >= 0
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
        and all(isinstance(session_id, str) and session_id for session_id in session_ids)
        and len(set(session_ids)) >= minimum_sessions
        and set(search_session_ids).isdisjoint(session_ids)
        and _nonempty_string(holdout_evidence_ref)
        and search_evidence_ref != holdout_evidence_ref
        and _nonempty_string(anchor.get("evidence_ref"))
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
            isinstance(sample, (int, float)) and sample >= 0
            for sample in samples
        )
        or not isinstance(completion, dict)
        or not _nonempty_string(completion.get("kind"))
        or completion.get("closed") is not True
        or completion.get("threadpool_joined") is not True
        or not isinstance(timer, dict)
        or not _nonempty_string(timer.get("source"))
        or not isinstance(timer.get("resolution_ns"), (int, float))
        or timer["resolution_ns"] <= 0
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


def diagnose_run_bundle(path: str | Path) -> dict[str, Any]:
    """Derive one deterministic exact-Shape diagnosis from a Run Bundle."""

    root = Path(path).resolve()
    manifest, document = _load_evidence(root)
    digests = _verified_document_digests(document)
    if _complete_required_identity(document):
        resource = _resource_axis(document)
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
    derivation_basis = {
        "schema": DIAGNOSTIC_RESULT_SCHEMA,
        "run_id": manifest["run_id"],
        "digests": digests,
        "axes": axes,
        "comparisons": comparisons,
        "policy_refs": document.get("policies", {}),
    }
    derivation = {
        "derivation_id": _canonical_digest(derivation_basis),
        "inputs": ["diagnostic-evidence"],
        "steps": [
            "verify-artifact-and-authored-digests",
            "qualify-exact-shape-active-anchor",
            "compose-explicit-single-node-schedule",
            "project-four-independent-axes",
        ],
    }
    return {
        "schema": DIAGNOSTIC_RESULT_SCHEMA,
        "run_id": manifest["run_id"],
        "status": (
            "complete"
            if all(axis["status"] == "known" for axis in axes.values())
            else "partial"
        ),
        "axes": axes,
        "comparisons": comparisons,
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
            )
            if key in document
        },
        "digests": digests,
        "derivation": derivation,
    }


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
            f"input digest: {result['digests']['input_sha256']}",
            f"evidence digest: {result['digests']['evidence_sha256']}",
        ]
    )
    lines.append(
        "Resource Physical Floor distance is optimization headroom; "
        "not prediction error."
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "DiagnosticBundleError",
    "DiagnosticBundleIntegrityError",
    "diagnose_run_bundle",
    "render_diagnostic_report",
]
