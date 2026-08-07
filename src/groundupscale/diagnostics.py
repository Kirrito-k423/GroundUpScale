"""Evidence-qualified diagnosis for one exact-Shape Run Bundle."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import hypot, isfinite
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
    "capability_surfaces",
    "candidate_envelopes",
    "surface_updates",
    "surface_queries",
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
        "work_rate_latency": None,
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


def _eligible_surface_anchor(
    surface: dict[str, Any], anchor: object, axes: tuple[str, ...]
) -> bool:
    if not isinstance(anchor, dict):
        return False
    shape = anchor.get("shape")
    rate = anchor.get("effective_rate")
    return (
        _nonempty_string(anchor.get("anchor_id"))
        and _nonempty_string(anchor.get("anchor_version"))
        and isinstance(shape, dict)
        and set(shape) == set(axes)
        and all(_finite_number(shape[axis]) and shape[axis] > 0 for axis in axes)
        and _finite_number(rate)
        and rate > 0
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
            if cell.get("status") != "retained":
                continue
            axis = axes[0]
            left = float(anchors[0]["shape"][axis])
            right = float(anchors[1]["shape"][axis])
            if left >= right or not left <= point[0] <= right:
                continue
            right_weight = (point[0] - left) / (right - left)
            weights = (1.0 - right_weight, right_weight)
            measure = right - left
        elif len(axes) == 2:
            if cell.get("status") not in (
                "retained",
                "hole",
                "candidate_support_boundary",
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
            "candidate_domain_boundary_unvalidated": 1,
            "cell_span_exceeds_policy": 2,
            "degenerate_simplex": 3,
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


def _derive_work_rate_latency(
    surface: dict[str, Any],
    point: dict[str, float],
    effective_rate: float,
    uncertainty: _SurfaceUncertainty,
) -> tuple[dict[str, Any], dict[str, float]] | None:
    work_formula = surface.get("work_formula")
    if not isinstance(work_formula, dict) or not _nonempty_string(
        work_formula.get("version")
    ) or work_formula.get("work_unit") != "FLOP":
        return None
    if work_formula.get("kind") == "square-matmul-2s3" and set(point) == {"s"}:
        declared_work = 2.0 * point["s"] ** 3
    elif (
        work_formula.get("kind") == "matmul-2mnk"
        and set(point) == {"m", "n"}
        and _finite_number(work_formula.get("fixed_k"))
        and work_formula["fixed_k"] > 0
    ):
        declared_work = 2.0 * point["m"] * point["n"] * work_formula["fixed_k"]
    else:
        return None
    if not isfinite(declared_work) or not isfinite(effective_rate):
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
    anchors_value = surface.get("anchors")
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
    if (
        not isinstance(shape, dict)
        or set(shape) != set(axes)
        or any(not _finite_number(shape[axis]) or shape[axis] <= 0 for axis in axes)
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
        {axis: value for axis, value in zip(axes, point, strict=True)},
        selected.effective_rate,
        uncertainty,
    )
    if latency_derivation is None:
        return _unknown_surface_query(query, surface, "invalid_work_formula")
    latency, latency_interval = latency_derivation
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
        "uncertainty": {
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
        },
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
    pending = list(surfaces)
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
                        "coordinate",
                        "work_formula",
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
        anchor = update.get("anchor")
        base_anchors = base.get("anchors") if isinstance(base, dict) else None
        cells = update.get("cells")
        uncertainty_policy = update.get("uncertainty_policy")
        update_evidence_refs = update.get("evidence_refs")
        if (
            not isinstance(base, dict)
            or not axes
            or not isinstance(base_anchors, list)
            or not _eligible_surface_anchor(base, anchor, axes)
            or not isinstance(cells, list)
            or not isinstance(uncertainty_policy, dict)
            or not isinstance(update_evidence_refs, list)
            or not update_evidence_refs
            or not all(
                _nonempty_string(reference)
                for reference in update_evidence_refs
            )
        ):
            continue
        anchor_ids = {
            item.get("anchor_id")
            for item in base_anchors
            if isinstance(item, dict)
        }
        if anchor["anchor_id"] in anchor_ids:
            continue
        base_evidence_refs = base.get("evidence_refs")
        new_anchors = sorted(
            [*base_anchors, anchor],
            key=lambda item: tuple(float(item["shape"][axis]) for axis in axes),
        )
        new_surface = {
            **base,
            "version": new_version,
            "previous_version": base_version,
            "anchors": new_anchors,
            "cells": cells,
            "uncertainty_policy": uncertainty_policy,
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
    (
        capability_surfaces,
        candidate_envelopes,
        surface_queries,
    ) = _capability_surface_results(document)
    derivation_basis = {
        "schema": DIAGNOSTIC_RESULT_SCHEMA,
        "run_id": manifest["run_id"],
        "digests": digests,
        "axes": axes,
        "comparisons": comparisons,
        "policy_refs": document.get("policies", {}),
        "capability_surfaces": capability_surfaces,
        "candidate_envelopes": candidate_envelopes,
        "capability_surface_queries": surface_queries,
    }
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
    return {
        "schema": DIAGNOSTIC_RESULT_SCHEMA,
        "run_id": manifest["run_id"],
        "status": (
            "complete"
            if all(axis["status"] == "known" for axis in axes.values())
            and all(
                query["status"] != "unknown" for query in surface_queries
            )
            else "partial"
        ),
        "axes": axes,
        "comparisons": comparisons,
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
        lines.append(
            f"{prefix}: {query['status']}; "
            f"cohort={query['cohort_id']}; "
            f"domain={json.dumps(query['domain'], ensure_ascii=False, separators=(',', ':'), sort_keys=True)}; "
            f"{candidate_text.lstrip('; ')}"
            f"{winner_text}"
            f"{domain_policy_text}; "
            f"rate={rate['value'] / 1_000_000_000_000:.9f} TFLOP/s; "
            f"latency={latency['value_ns']:.6f} ns; "
            f"cell={query['cell_id']}; "
            f"anchors={','.join(anchor['anchor_id'] for anchor in query['anchors'])}; "
            f"weights={json.dumps(query['weights'], separators=(',', ':'))}; "
            f"uncertainty={json.dumps(query['uncertainty']['components'], separators=(',', ':'), sort_keys=True)}"
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
