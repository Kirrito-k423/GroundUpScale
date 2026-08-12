"""Fail-closed comparison of independently qualified cross-hardware diagnoses.

The comparator is deliberately a projection over already-derived diagnostic
results.  It does not create Anchors, query another cohort's Surface, or infer
missing counters.  A report is useful only when both sides describe the same
semantic Shape and each side carries its own complete evidence contract.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from math import hypot
from pathlib import Path
from typing import Any

from groundupscale.diagnostics import DiagnosticBundleError, diagnose_run_bundle


CROSS_HARDWARE_REPORT_SCHEMA = (
    "groundupscale.dev/cross-hardware-diagnostic-report/v1alpha1"
)
_REQUIRED_EVIDENCE = (
    "hardware",
    "cohort_id",
    "execution_domain",
    "correctness",
    "environment",
    "measurement_capability_manifest",
    "cohort_evidence",
    "timing_plan",
    "baseline_timing_lane",
    "resolved_ir",
)
_AXES = (
    "resource_physical_floor",
    "operator_achievable_frontier",
    "schedule_achievable_frontier",
    "observation",
)
_VERDICTS = {
    "frontier_shift",
    "implementation_headroom",
    "integration_overhead",
    "suspected_regression",
    "insufficient_evidence",
    "confirmed_bug",
}


def _unknown(reason_code: str) -> dict[str, Any]:
    return {"status": "unknown", "reason_code": reason_code, "evidence_refs": []}


def _string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _shape(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping) or not value:
        return None
    result: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            return None
        result[key] = item
    return result


def _refs(value: object) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"evidence_ref", "evidence_refs", "artifact_ref", "artifact_refs", "source_ref", "source_refs", "run_bundle"}:
                found.extend(_refs(item))
            elif isinstance(item, (Mapping, list, tuple)):
                found.extend(_refs(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            if _string(item):
                found.append(str(item))
            elif isinstance(item, (Mapping, list, tuple)):
                found.extend(_refs(item))
    elif _string(value):
        found.append(str(value))
    return list(dict.fromkeys(found))


def _evidence_quality(result: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    evidence = result.get("evidence")
    if not isinstance(evidence, Mapping):
        return {"status": "unknown", "reason_codes": ["missing-evidence"]}
    if result.get("schema") != "groundupscale.dev/diagnostic-result/v1alpha1":
        reasons.append("unsupported-diagnostic-result-schema")
    for key in _REQUIRED_EVIDENCE:
        if key not in evidence:
            reasons.append(f"missing-{key.replace('_', '-')}")
    resolved_ir = evidence.get("resolved_ir")
    if not isinstance(resolved_ir, Mapping) or not _string(resolved_ir.get("operation")):
        reasons.append("missing-semantic-operation")
    if result.get("status") != "complete":
        reasons.append("diagnostic-result-incomplete")
    correctness = evidence.get("correctness")
    if not isinstance(correctness, Mapping) or correctness.get("passed") is not True:
        reasons.append("correctness-not-qualified")
    environment = evidence.get("environment")
    if not isinstance(environment, Mapping) or environment.get("eligible") is not True:
        reasons.append("environment-not-eligible")
    hardware = evidence.get("hardware")
    if not isinstance(hardware, Mapping) or any(
        not _string(hardware.get(key)) for key in ("device", "partition", "topology", "software")
    ):
        reasons.append("hardware-fingerprint-incomplete")
    capability = evidence.get("measurement_capability_manifest")
    if not isinstance(capability, Mapping) or capability.get("status") not in {"complete", "completed", "qualified", "eligible"}:
        reasons.append("measurement-capability-manifest-not-qualified")
    cohort_evidence = evidence.get("cohort_evidence")
    cohort_valid = (
        isinstance(cohort_evidence, Mapping)
        and (
            cohort_evidence.get("status") in {"qualified", "matched", "complete"}
            or (
                _string(cohort_evidence.get("reference_cohort_id"))
                and isinstance(cohort_evidence.get("observed_identity"), Mapping)
                and isinstance(cohort_evidence.get("reference_identity"), Mapping)
                and cohort_evidence.get("reference_cohort_id") == evidence.get("cohort_id")
            )
        )
    )
    if not cohort_valid:
        reasons.append("cohort-evidence-not-qualified")
    lane = evidence.get("baseline_timing_lane")
    timing_plan = evidence.get("timing_plan")
    if not isinstance(timing_plan, Mapping) or not _string(timing_plan.get("evidence_ref")):
        reasons.append("timing-plan-missing")
    if not isinstance(lane, Mapping):
        reasons.append("missing-baseline-timing-lane")
    else:
        boundary = lane.get("completion_boundary")
        timer = lane.get("timer")
        if not isinstance(boundary, Mapping) or boundary.get("closed") is not True:
            reasons.append("completion-boundary-not-closed")
        if not isinstance(timer, Mapping) or not _string(timer.get("source")):
            reasons.append("primary-timer-missing")
        if not isinstance(lane.get("raw_samples_ns"), list) or not lane["raw_samples_ns"]:
            reasons.append("raw-timing-samples-missing")
        warmup = lane.get("warmup")
        if not isinstance(warmup, Mapping) or warmup.get("converged") is not True:
            reasons.append("warmup-not-qualified")
    axes = result.get("axes")
    if not isinstance(axes, Mapping):
        reasons.append("missing-four-axis-result")
    else:
        for axis in _AXES:
            if not isinstance(axes.get(axis), Mapping) or axes[axis].get("status") != "known":
                reasons.append(f"axis-{axis.replace('_', '-')}-unknown")
    anchors = evidence.get("frontier_anchors")
    if not isinstance(anchors, list):
        anchors = result.get("frontier_anchor_lifecycles")
    if not isinstance(anchors, list):
        anchors = []
    if not any(
        isinstance(anchor, Mapping) and anchor.get("frontier_role") == "ACTIVE"
        for anchor in anchors
    ):
        reasons.append("missing-active-frontier-anchor")
    queries = result.get("capability_surface_queries")

    def valid_query(query: object) -> bool:
        if not isinstance(query, Mapping) or query.get("status", "known") != "known":
            return False
        surface = query.get("surface")
        surface_id = query.get("surface_id")
        if isinstance(surface, Mapping):
            surface_id = surface.get("surface_id", surface_id)
        return _string(surface_id)

    query_valid = (
        isinstance(queries, list)
        and bool(queries)
        and all(valid_query(query) for query in queries)
    )
    if not query_valid:
        reasons.append("capability-surface-query-not-qualified")
    verdicts = result.get("performance_diagnosis_verdicts")
    if not isinstance(verdicts, list) or not verdicts:
        reasons.append("missing-performance-verdicts")
    elif any(
        not isinstance(verdict, Mapping) or verdict.get("verdict") not in _VERDICTS
        for verdict in verdicts
    ):
        reasons.append("invalid-performance-verdict-vocabulary")
    return {
        "status": "known" if not reasons else "unknown",
        "reason_codes": list(dict.fromkeys(reasons)),
    }


def _side_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    evidence = result.get("evidence") if isinstance(result.get("evidence"), Mapping) else {}
    hardware = evidence.get("hardware") if isinstance(evidence.get("hardware"), Mapping) else {}
    cohort = evidence.get("cohort_id")
    quality = _evidence_quality(result)
    axes = result.get("axes") if isinstance(result.get("axes"), Mapping) else {}
    frontier = axes.get("operator_achievable_frontier") if isinstance(axes.get("operator_achievable_frontier"), Mapping) else {}
    observation = axes.get("observation") if isinstance(axes.get("observation"), Mapping) else {}
    comparisons = result.get("comparisons")
    comparison = comparisons.get("operator_frontier_to_observation", {}) if isinstance(comparisons, Mapping) else {}
    if not isinstance(comparison, Mapping):
        comparison = {}
    frontier_value = frontier.get("value_ns")
    observation_value = observation.get("value_ns")
    efficiency: object = _unknown("frontier-or-observation-unknown")
    if quality["status"] == "known" and isinstance(frontier_value, (int, float)) and isinstance(observation_value, (int, float)) and observation_value > 0:
        efficiency = frontier_value / observation_value
    queries = result.get("capability_surface_queries")
    surface_refs = _refs(queries)
    anchor_refs = _refs(frontier) + _refs(evidence.get("frontier_anchors")) + _refs(result.get("frontier_anchor_lifecycles"))
    verdicts = result.get("performance_diagnosis_verdicts")
    if not isinstance(verdicts, list):
        verdicts = [{"status": "unknown", "reason_code": "verdicts-missing", "evidence_refs": []}]
    return {
        "device": hardware.get("device"),
        "cohort_id": cohort,
        "run_id": result.get("run_id"),
        "independent": True,
        "evidence_quality": quality,
        "metrics": {
            "frontier_efficiency": efficiency,
            "frontier_ns": frontier_value if quality["status"] == "known" else None,
            "observation_ns": observation_value if quality["status"] == "known" else None,
            "combined_uncertainty_ns": comparison.get("combined_uncertainty_ns") if quality["status"] == "known" else _unknown("uncertainty-not-qualified"),
            "distance_ns": comparison.get("distance_ns") if quality["status"] == "known" else None,
        },
        "evidence_refs": _refs(evidence),
        "input_refs": _refs(evidence.get("resolved_configuration")) + _refs(evidence.get("resolved_ir")),
        "source_run_refs": _refs(evidence.get("source_runs")),
        "anchor_refs": list(dict.fromkeys(anchor_refs)),
        "surface_refs": list(dict.fromkeys(surface_refs)),
        "policy_refs": _refs(evidence.get("policies")),
        "derivation_refs": _refs(result.get("derivation")),
        "verdicts": verdicts,
        "verdict_refs": _refs(verdicts),
        "probe_refs": _refs(result.get("shape_disambiguation_probes")),
        "ledger_refs": _refs(evidence.get("single_node_schedule")) + _refs(evidence.get("diagnostic_profiling_lane")),
    }


def compare_cross_hardware(
    m4_result: Mapping[str, Any],
    ascend_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two diagnostic results without transferring evidence across cohorts."""
    m4 = _side_payload(m4_result)
    ascend = _side_payload(ascend_result)
    m4_evidence = m4_result.get("evidence")
    ascend_evidence = ascend_result.get("evidence")
    m4_domain = m4_evidence.get("execution_domain", {}) if isinstance(m4_evidence, Mapping) else {}
    ascend_domain = ascend_evidence.get("execution_domain", {}) if isinstance(ascend_evidence, Mapping) else {}
    m4_shape = _shape(m4_domain.get("shape")) if isinstance(m4_domain, Mapping) else None
    ascend_shape = _shape(ascend_domain.get("shape")) if isinstance(ascend_domain, Mapping) else None
    if m4_shape is None or ascend_shape is None:
        shape = _unknown("exact-shape-missing")
    elif m4_shape != ascend_shape:
        shape = {"status": "unknown", "reason_code": "exact-shape-mismatch", "evidence_refs": []}
    else:
        shape = {"status": "matched", "shape": m4_shape}
    # dtype/layout are the cross-hardware semantic contract.  Execution mode,
    # thread count and alignment are hardware-specific validity dimensions and
    # must remain in each side's evidence instead of being falsely equated.
    domain_keys = ("dtype", "layout")
    domain_match = (
        all(m4_domain.get(key) == ascend_domain.get(key) for key in domain_keys)
        if isinstance(m4_domain, Mapping) and isinstance(ascend_domain, Mapping)
        else False
    )
    semantic_m4 = m4_evidence.get("resolved_ir", {}) if isinstance(m4_evidence, Mapping) else {}
    semantic_ascend = ascend_evidence.get("resolved_ir", {}) if isinstance(ascend_evidence, Mapping) else {}
    semantic_operation_m4 = semantic_m4.get("operation") if isinstance(semantic_m4, Mapping) else None
    semantic_operation_ascend = semantic_ascend.get("operation") if isinstance(semantic_ascend, Mapping) else None
    stable_path_m4 = semantic_m4.get("stable_path") if isinstance(semantic_m4, Mapping) else None
    stable_path_ascend = semantic_ascend.get("stable_path") if isinstance(semantic_ascend, Mapping) else None
    semantic_match = (
        isinstance(semantic_m4, Mapping) and isinstance(semantic_ascend, Mapping)
        and _string(semantic_operation_m4)
        and semantic_operation_m4 == semantic_operation_ascend
        and (
            not _string(stable_path_m4)
            or not _string(stable_path_ascend)
            or stable_path_m4 == stable_path_ascend
        )
    )
    if not domain_match:
        shape = {"status": "unknown", "reason_code": "execution-domain-mismatch", "evidence_refs": []}
    cohort_reasons: list[str] = []
    if not _string(m4["cohort_id"]) or not _string(ascend["cohort_id"]):
        cohort_reasons.append("hardware-cohort-missing")
    elif m4["cohort_id"] == ascend["cohort_id"]:
        ascend["independent"] = False
        ascend["reason_code"] = "hardware-cohort-reused"
        cohort_reasons.append("hardware-cohort-reused")
    if not semantic_match:
        cohort_reasons.append("semantic-operation-mismatch")
    status = "complete" if shape.get("status") == "matched" and not cohort_reasons and m4["evidence_quality"]["status"] == "known" and ascend["evidence_quality"]["status"] == "known" else "insufficient_evidence"
    return {
        "schema": CROSS_HARDWARE_REPORT_SCHEMA,
        "status": status,
        "shape_comparison": shape,
        "semantic_comparison": {"status": "matched" if semantic_match else "unknown", "operation": semantic_operation_m4},
        "cohorts": {
            "m4": {"cohort_id": m4["cohort_id"], "independent": True},
            "ascend": {"cohort_id": ascend["cohort_id"], "independent": ascend["independent"], **({"reason_code": ascend["reason_code"]} if "reason_code" in ascend else {})},
        },
        "sides": {"m4": m4, "ascend": ascend},
        "metrics": {"m4": m4["metrics"], "ascend": ascend["metrics"]},
        "evidence_index": {
            "m4": {key: m4[key] for key in ("run_id", "evidence_refs", "input_refs", "source_run_refs", "anchor_refs", "surface_refs", "policy_refs", "derivation_refs", "verdict_refs", "probe_refs", "ledger_refs")},
            "ascend": {key: ascend[key] for key in ("run_id", "evidence_refs", "input_refs", "source_run_refs", "anchor_refs", "surface_refs", "policy_refs", "derivation_refs", "verdict_refs", "probe_refs", "ledger_refs")},
        },
        "cross_hardware_comparison": _cross_hardware_metrics(m4, ascend),
        "gate": {"status": "satisfied" if status == "complete" else "failed", "reason_codes": cohort_reasons + m4["evidence_quality"]["reason_codes"] + ascend["evidence_quality"]["reason_codes"]},
    }


def _cross_hardware_metrics(m4: Mapping[str, Any], ascend: Mapping[str, Any]) -> dict[str, Any]:
    """Compare normalized efficiencies, never raw latency as a fairness metric."""
    m4_efficiency = m4["metrics"].get("frontier_efficiency")
    ascend_efficiency = ascend["metrics"].get("frontier_efficiency")
    m4_uncertainty = m4["metrics"].get("combined_uncertainty_ns")
    ascend_uncertainty = ascend["metrics"].get("combined_uncertainty_ns")
    if isinstance(m4_efficiency, (int, float)) and isinstance(ascend_efficiency, (int, float)) and m4_efficiency > 0:
        efficiency_ratio: object = ascend_efficiency / m4_efficiency
    else:
        efficiency_ratio = _unknown("frontier-efficiency-not-comparable")
    if isinstance(m4_uncertainty, (int, float)) and isinstance(ascend_uncertainty, (int, float)):
        combined_uncertainty: object = hypot(float(m4_uncertainty), float(ascend_uncertainty))
    else:
        combined_uncertainty = _unknown("combined-uncertainty-not-qualified")
    return {
        "frontier_efficiency_ratio": efficiency_ratio,
        "combined_uncertainty_ns": combined_uncertainty,
        "absolute_latency_comparison": "not-a-fair-efficiency-metric",
    }


def load_cross_hardware_input(path: str | Path) -> dict[str, Any]:
    """Load a diagnostic result JSON or derive one from a verified Run Bundle."""
    source = Path(path).resolve()
    if source.is_dir():
        result = diagnose_run_bundle(source)
        result["source_bundle"] = str(source)
        return result
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"cross-hardware input must be an object: {source}")
    value.setdefault("source_result", str(source))
    return value


def compare_cross_hardware_inputs(
    m4_input: str | Path,
    ascend_input: str | Path,
) -> dict[str, Any]:
    """Load two result/bundle inputs and attach their immutable source paths."""
    def load_or_invalid(source: str | Path) -> dict[str, Any]:
        try:
            return load_cross_hardware_input(source)
        except (DiagnosticBundleError, KeyError, OSError, TypeError, ValueError) as error:
            return {
                "schema": "groundupscale.dev/diagnostic-result/v1alpha1",
                "status": "invalid",
                "evidence": {},
                "axes": {},
                "source_result": str(Path(source).resolve()),
                "load_error": str(error),
            }

    m4 = load_or_invalid(m4_input)
    ascend = load_or_invalid(ascend_input)
    report = compare_cross_hardware(m4, ascend)
    report["evidence_index"]["m4"]["source"] = m4.get("source_bundle", m4.get("source_result"))
    report["evidence_index"]["ascend"]["source"] = ascend.get("source_bundle", ascend.get("source_result"))
    return report


def render_cross_hardware_report(report: Mapping[str, Any]) -> str:
    """Render a deterministic text projection of a cross-hardware report."""
    lines = [
        f"cross-hardware diagnostic: {report.get('status')}",
        f"exact Shape: {report.get('shape_comparison')}",
    ]
    for side in ("m4", "ascend"):
        payload = report.get("metrics", {}).get(side, {})
        efficiency = payload.get("frontier_efficiency")
        if isinstance(efficiency, Mapping):
            value = "unknown"
        else:
            value = f"{efficiency:.6f}"
        cohort = report.get("cohorts", {}).get(side, {}).get("cohort_id")
        lines.append(f"{side} cohort={cohort}; Frontier Efficiency={value}")
        quality = report.get("sides", {}).get(side, {}).get("evidence_quality", {})
        lines.append(f"{side} evidence quality={quality.get('status')}; reasons={quality.get('reason_codes', [])}")
    lines.append("evidence index: " + json.dumps(report.get("evidence_index", {}), ensure_ascii=False, sort_keys=True))
    return "\n".join(lines) + "\n"


__all__ = [
    "CROSS_HARDWARE_REPORT_SCHEMA",
    "compare_cross_hardware",
    "compare_cross_hardware_inputs",
    "load_cross_hardware_input",
    "render_cross_hardware_report",
]
