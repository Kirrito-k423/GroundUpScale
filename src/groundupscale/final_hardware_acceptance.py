"""Immutable final hardware acceptance for one independent E2E holdout."""

from __future__ import annotations

from hashlib import sha256
from html import escape
import json
from math import isfinite, sqrt
from pathlib import Path
from statistics import median
import tempfile
from typing import Any, Mapping

from groundupscale.ir import canonical_data
from groundupscale.run_bundle import RUN_ID_PATTERN, RunBundleExistsError


INPUT_SCHEMA = "groundupscale.dev/final-hardware-acceptance-input/v1alpha1"
RESULT_SCHEMA = "groundupscale.dev/final-hardware-acceptance/v1alpha1"
REPORT_SCHEMA = "groundupscale.dev/final-hardware-acceptance-html/v1alpha1"
PRODUCER = "groundupscale@0.1.0"


class FinalAcceptanceError(ValueError):
    """The evidence cannot support a final hardware acceptance verdict."""


def _bytes(value: object) -> bytes:
    return (
        json.dumps(canonical_data(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode()


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def _mapping(value: object, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalAcceptanceError(reason)
    return value


def _number(value: object, reason: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) < 0
    ):
        raise FinalAcceptanceError(reason)
    return float(value)


def _summary(samples: list[float]) -> tuple[float, float]:
    ordered = sorted(samples)
    if len(ordered) < 3:
        raise FinalAcceptanceError("holdout-insufficient-samples")

    def quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return float(median(ordered)), float(quantile(0.75) - quantile(0.25))


def _validate_identity(identity: Mapping[str, Any]) -> None:
    for field in (
        "model_spec_sha256",
        "workload_spec_sha256",
        "analysis_case_sha256",
        "deployment_intent_sha256",
        "case",
        "dtype",
        "hardware_cohort",
        "completion_boundary",
    ):
        if not isinstance(identity.get(field), str) or not identity[field]:
            raise FinalAcceptanceError(f"invalid-identity-{field}")
    if identity.get("shape") != [1, 512, 512] or identity.get("dtype") != "float32":
        raise FinalAcceptanceError("unsupported-final-acceptance-semantics")


def _validate_holdout(holdout: Mapping[str, Any], identity: Mapping[str, Any]) -> list[str]:
    if holdout.get("identity") != identity:
        raise FinalAcceptanceError("holdout-identity-mismatch")
    samples_value = holdout.get("raw_samples_ns")
    if not isinstance(samples_value, list):
        raise FinalAcceptanceError("invalid-holdout-samples")
    samples = [_number(value, "invalid-holdout-samples") for value in samples_value]
    actual_median, actual_iqr = _summary(samples)
    if (
        holdout.get("sample_count") != len(samples)
        or _number(holdout.get("median_ns"), "invalid-holdout-median") != actual_median
        or _number(holdout.get("iqr_ns"), "invalid-holdout-iqr") != actual_iqr
        or holdout.get("observation_digest")
        != sha256(json.dumps(samples_value, separators=(",", ":")).encode()).hexdigest()
    ):
        raise FinalAcceptanceError("holdout-sample-summary-mismatch")
    if holdout.get("run_id") in set(holdout.get("construction_run_ids", [])):
        raise FinalAcceptanceError("holdout-run-identity-not-independent")
    warmup = _mapping(holdout.get("warmup"), "invalid-holdout-warmup")
    if not isinstance(warmup.get("iterations"), int) or warmup["iterations"] <= 0 or warmup.get("outside_timing_boundary") is not True:
        raise FinalAcceptanceError("invalid-holdout-warmup")
    timer = _mapping(holdout.get("timer"), "invalid-holdout-timer")
    sync = _mapping(holdout.get("synchronization"), "invalid-holdout-synchronization")
    correctness = _mapping(holdout.get("correctness"), "invalid-holdout-correctness")
    environment = _mapping(holdout.get("environment"), "invalid-holdout-environment")
    lock_session = _mapping(environment.get("lock_session"), "invalid-holdout-lock-session")
    gates = _mapping(holdout.get("gates"), "invalid-holdout-gates")
    boundaries: list[str] = []
    required_gates = (
        "environment", "correctness", "no_cpu_fallback", "timing",
        "synchronization", "execution_contract",
    )
    boundaries.extend(f"holdout-gate:{gate}" for gate in required_gates if gates.get(gate) != "passed")
    if timer.get("primary") != "torch.npu.Event.elapsed_time" or timer.get("unit") != "ns":
        boundaries.append("holdout-timer")
    if sync.get("protocol") != identity["completion_boundary"] or sync.get("passed") is not True:
        boundaries.append("holdout-synchronization")
    if correctness.get("passed") is not True or correctness.get("no_cpu_fallback") is not True or correctness.get("semantic_leaf_count") != 52:
        boundaries.append("holdout-correctness-or-cpu-fallback")
    if environment.get("device") != "npu:0" or environment.get("visibility") != "0":
        boundaries.append("holdout-environment")
    if (
        lock_session.get("schema")
        != "groundupscale.dev/ascend-host-lock-session/v1alpha1"
        or lock_session.get("issue") != 50
        or lock_session.get("run_id") != holdout.get("run_id")
        or lock_session.get("hardware_cohort") != identity["hardware_cohort"]
        or lock_session.get("ascend_rt_visible_devices") != "0"
        or lock_session.get("logical_device") != "npu:0"
        or lock_session.get("whole_host_exclusive") is not True
        or lock_session.get("lock_path")
        != "/home/t00906153/.groundupscale/locks/ascend-910b2-host.lock"
        or lock_session.get("wrapper_path")
        != "/home/t00906153/.groundupscale/bin/with-ascend-lock"
        or lock_session.get("wrapper_sha256")
        != "22d43618f1c616b2ff70570944c7447cd851aac98bfedb111b7912fc36b94787"
        or not isinstance(lock_session.get("measurement_started_at"), str)
        or not isinstance(lock_session.get("measurement_ended_at"), str)
        or not isinstance(lock_session.get("owner"), str)
        or not lock_session["owner"].startswith("issue=50 ")
    ):
        boundaries.append("holdout-host-lock-session")
    return sorted(set(boundaries))


def compose_final_acceptance(document: Mapping[str, object]) -> dict[str, Any]:
    """Validate all evidence and fail closed unless every public gate is complete."""

    if document.get("schema") != INPUT_SCHEMA:
        raise FinalAcceptanceError("unsupported-final-acceptance-input")
    identity = dict(_mapping(document.get("identity"), "missing-final-identity"))
    _validate_identity(identity)
    schedule = _mapping(document.get("schedule"), "missing-final-schedule")
    holdout = _mapping(document.get("holdout"), "missing-final-holdout")
    decomposition = _mapping(document.get("decomposition"), "missing-final-decomposition")
    construction_ids = document.get("construction_run_ids")
    if not isinstance(construction_ids, list) or not all(isinstance(item, str) and item for item in construction_ids):
        raise FinalAcceptanceError("invalid-construction-run-identities")
    if holdout.get("run_id") in construction_ids:
        raise FinalAcceptanceError("holdout-run-identity-not-independent")
    sources = document.get("source_bundles")
    if not isinstance(sources, list) or len(sources) != 4:
        raise FinalAcceptanceError("final-acceptance-requires-locked-sources")
    if {item.get("source_role") for item in sources if isinstance(item, Mapping)} != {
        "schedule-frontier", "observed-decomposition", "gap-report", "independent-holdout"
    }:
        raise FinalAcceptanceError("final-acceptance-source-roles-mismatch")
    locked_ids = {
        item.get("run_id") for item in sources if isinstance(item, Mapping)
    }
    if set(construction_ids) != locked_ids - {holdout.get("run_id")}:
        raise FinalAcceptanceError("construction-run-lineage-mismatch")
    source_identities = document.get("source_identities")
    if not isinstance(source_identities, list) or len(source_identities) != len(sources):
        raise FinalAcceptanceError("source-identity-lineage-missing")
    for source_identity in source_identities:
        locked = _mapping(source_identity, "source-identity-lineage-missing")
        if locked.get("run_id") not in locked_ids or locked.get("identity") != identity:
            raise FinalAcceptanceError("source-identity-mismatch")

    holdout_boundaries = _validate_holdout(holdout, identity)
    leaves_value = schedule.get("leaves")
    surfaces_value = schedule.get("surfaces")
    edges_value = schedule.get("edges")
    if not isinstance(leaves_value, list) or not isinstance(surfaces_value, list) or not isinstance(edges_value, list):
        raise FinalAcceptanceError("invalid-schedule-evidence")
    paths = {leaf.get("stable_path") for leaf in leaves_value if isinstance(leaf, Mapping)}
    if paths != set(schedule.get("stable_paths", [])):
        raise FinalAcceptanceError("schedule-stable-path-mismatch")
    candidate_ids = {
        candidate
        for surface in surfaces_value if isinstance(surface, Mapping)
        for candidate in surface.get("candidate_ids", []) if isinstance(candidate, str)
    }
    for leaf in leaves_value:
        item = _mapping(leaf, "invalid-schedule-leaf")
        candidate = item.get("selected_candidate_id")
        if candidate is not None and candidate not in candidate_ids:
            raise FinalAcceptanceError("selected-candidate-not-in-surface")
    for edge in edges_value:
        if not isinstance(edge, list) or len(edge) != 2 or not set(edge) <= paths:
            raise FinalAcceptanceError("schedule-edge-stable-path-mismatch")
    known_durations = [
        _number(item.get("duration_ns"), "invalid-schedule-leaf-duration")
        for item in leaves_value if isinstance(item, Mapping) and item.get("duration_ns") is not None
    ]
    schedule_known = schedule.get("status") == "known"
    schedule_ns = schedule.get("selected_complete_schedule_duration_ns")
    missing_schedule = schedule.get("missing_evidence", [])
    if not isinstance(missing_schedule, list) or not all(isinstance(item, str) for item in missing_schedule):
        raise FinalAcceptanceError("invalid-schedule-evidence-boundary")
    if schedule_known and missing_schedule:
        raise FinalAcceptanceError("known-schedule-contains-missing-evidence")
    if schedule_known:
        schedule_ns = _number(schedule_ns, "invalid-schedule-duration")
        execution_ir = _mapping(schedule.get("execution_ir"), "missing-schedule-execution-ir")
        events = execution_ir.get("physical_events")
        dependencies = execution_ir.get("dependency_edges")
        claims = execution_ir.get("resource_claims")
        transformations = execution_ir.get("transformations")
        if (
            execution_ir.get("status") != "known"
            or execution_ir.get("critical_path_duration_ns") != schedule_ns
            or not isinstance(events, list) or not events
            or not isinstance(dependencies, list)
            or not isinstance(claims, list) or not claims
            or not isinstance(transformations, list)
        ):
            raise FinalAcceptanceError("invalid-selected-schedule-execution-ir")
        event_ids = {
            event.get("event_id") for event in events if isinstance(event, Mapping)
        }
        if len(event_ids) != len(events) or None in event_ids:
            raise FinalAcceptanceError("invalid-selected-schedule-execution-ir")
        if any(
            not isinstance(edge, list) or len(edge) != 2 or not set(edge) <= event_ids
            for edge in dependencies
        ):
            raise FinalAcceptanceError("invalid-selected-schedule-execution-ir")
        predecessors = {event_id: set() for event_id in event_ids}
        for predecessor, successor in dependencies:
            predecessors[successor].add(predecessor)
        remaining = set(event_ids)
        longest: dict[str, float] = {}
        while remaining:
            ready = sorted(event_id for event_id in remaining if predecessors[event_id] <= longest.keys())
            if not ready:
                raise FinalAcceptanceError("invalid-selected-schedule-execution-ir")
            for event_id in ready:
                event = next(item for item in events if item["event_id"] == event_id)
                duration = _number(event.get("duration_ns"), "invalid-selected-schedule-execution-ir")
                longest[event_id] = duration + max((longest[item] for item in predecessors[event_id]), default=0.0)
                remaining.remove(event_id)
        if max(longest.values()) != schedule_ns:
            raise FinalAcceptanceError("invalid-selected-schedule-execution-ir")
        if any(
            not isinstance(claim, Mapping)
            or claim.get("event_id") not in event_ids
            or not isinstance(claim.get("resource_id"), str)
            or claim.get("claim_kind") not in {"capacity", "throughput", "exclusive"}
            for claim in claims
        ):
            raise FinalAcceptanceError("invalid-selected-schedule-execution-ir")
        if any(
            not isinstance(item, Mapping)
            or item.get("event_id") not in event_ids
            or not isinstance(item.get("kind"), str)
            for item in transformations
        ):
            raise FinalAcceptanceError("invalid-selected-schedule-execution-ir")
        if any(
            isinstance(item, Mapping)
            and item.get("duration_ns") is not None
            and not any(
                isinstance(event, Mapping)
                and event.get("event_id") == item.get("event_id")
                and event.get("duration_ns") == item.get("duration_ns")
                for event in execution_ir["physical_events"]
            )
            for item in leaves_value
        ):
            raise FinalAcceptanceError("schedule-duration-reconciliation-mismatch")
        leaf_uncertainties = [
            _number(item.get("standard_uncertainty_ns"), "invalid-schedule-uncertainty")
            for item in leaves_value if isinstance(item, Mapping)
        ]
        expected_schedule_uncertainty = sqrt(sum(value**2 for value in leaf_uncertainties))
        if _number(schedule.get("standard_uncertainty_ns"), "invalid-schedule-uncertainty") != expected_schedule_uncertainty:
            raise FinalAcceptanceError("schedule-uncertainty-reconciliation-mismatch")
    elif any(
        isinstance(item, Mapping)
        and (item.get("duration_ns") is not None or item.get("selected_candidate_id") is not None)
        for item in leaves_value
    ):
        raise FinalAcceptanceError("unknown-schedule-contains-selected-evidence")
    reconciliation = _mapping(decomposition.get("reconciliation"), "invalid-decomposition-reconciliation")
    observed_ns = _number(holdout.get("median_ns"), "invalid-holdout-median")
    expected_observed_uncertainty = _number(holdout.get("iqr_ns"), "invalid-holdout-iqr") / 1.349
    if abs(
        _number(holdout.get("standard_uncertainty_ns"), "invalid-holdout-uncertainty")
        - expected_observed_uncertainty
    ) > 1e-9:
        raise FinalAcceptanceError("holdout-uncertainty-reconciliation-mismatch")
    decomposition_boundaries: list[str] = []
    if decomposition.get("status") != "available":
        decomposition_boundaries.append("observed-decomposition-unavailable")
    elif (
        _number(reconciliation.get("observed_e2e_ns"), "invalid-decomposition-reconciliation") != observed_ns
        or _number(reconciliation.get("accounted_e2e_ns"), "invalid-decomposition-reconciliation") != observed_ns
        or _number(reconciliation.get("residual_ns"), "invalid-decomposition-reconciliation") != 0
    ):
        raise FinalAcceptanceError("decomposition-reconciliation-mismatch")

    evidence_boundary = {
        "schedule": list(missing_schedule) if not schedule_known else [],
        "holdout": holdout_boundaries,
        "decomposition": decomposition_boundaries,
    }
    complete = not any(evidence_boundary.values())
    schedule_u = _number(schedule.get("standard_uncertainty_ns"), "invalid-schedule-uncertainty") if complete else None
    observed_u = expected_observed_uncertainty if complete else None
    gap = abs(observed_ns - schedule_ns) if complete else None
    metrics = {
        "selected_complete_schedule_achievable_frontier_ns": schedule_ns if complete else None,
        "qualified_e2e_observation_ns": observed_ns if complete else None,
        "absolute_gap_ns": gap,
        "relative_gap": gap / observed_ns if complete and observed_ns else None,
        "observation_to_frontier_ratio": observed_ns / schedule_ns if complete and schedule_ns else None,
        "combined_uncertainty_ns": sqrt(schedule_u**2 + observed_u**2) if complete else None,
        "frontier_efficiency": schedule_ns / observed_ns if complete and observed_ns else None,
    }
    return {
        "schema": RESULT_SCHEMA,
        "status": "accepted" if complete else "structured-unknown",
        "identity": identity,
        "holdout_run_id": holdout.get("run_id"),
        "construction_run_ids": list(construction_ids),
        "schedule": dict(schedule),
        "holdout": dict(holdout),
        "decomposition": dict(decomposition),
        "source_bundles": list(document.get("source_bundles", [])),
        "metrics": metrics,
        "evidence_boundary": evidence_boundary,
        "derivation": {"input_sha256": _digest(_bytes(document))},
    }


def render_final_acceptance_html(result: Mapping[str, Any]) -> str:
    payload = json.dumps(canonical_data(result), ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    metrics = result["metrics"]
    boundaries = result["evidence_boundary"]
    return f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Final Ascend hardware acceptance</title></head>
<body><h1>Final Ascend hardware acceptance</h1><p>Status: <strong>{escape(str(result['status']))}</strong></p>
<p>Selected complete Schedule Achievable Frontier: {metrics['selected_complete_schedule_achievable_frontier_ns']}; qualified E2E Observation: {metrics['qualified_e2e_observation_ns']}; absolute gap: {metrics['absolute_gap_ns']}; relative gap: {metrics['relative_gap']}; ratio: {metrics['observation_to_frontier_ratio']}; combined uncertainty: {metrics['combined_uncertainty_ns']}; Frontier efficiency: {metrics['frontier_efficiency']}.</p>
<p>Evidence boundary: {escape(json.dumps(boundaries, ensure_ascii=False, sort_keys=True))}</p>
<script id=\"groundupscale-final-acceptance\" type=\"application/json\">{payload}</script></body></html>\n"""


def write_final_acceptance_bundle(artifact_store: str | Path, *, run_id: str, document: Mapping[str, object]) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise FinalAcceptanceError("unsafe-run-id")
    result = compose_final_acceptance(document)
    root = Path(artifact_store).resolve() / "runs"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_id
    if destination.exists():
        raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=root))
    artifacts: list[dict[str, object]] = []

    def write(role: str, path: str, content: bytes, schema: str, inputs: list[str], media_type: str = "application/json") -> None:
        target = temporary / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        artifacts.append({"role": role, "path": path, "schema": schema, "media_type": media_type, "sha256": _digest(content), "produced_by": PRODUCER, "inputs": inputs})

    try:
        write("final-hardware-acceptance-input", "resolved/final-hardware-acceptance-input.json", _bytes(document), INPUT_SCHEMA, [])
        write("final-hardware-acceptance", "acceptance/final-hardware-acceptance.json", _bytes(result), RESULT_SCHEMA, ["resolved/final-hardware-acceptance-input.json"])
        write("html-report", "reports/report.html", render_final_acceptance_html(result).encode(), REPORT_SCHEMA, ["acceptance/final-hardware-acceptance.json"], "text/html")
        manifest = {
            "schema": "groundupscale.dev/run-manifest/v1alpha1", "run_id": run_id,
            "bundle_kind": "final-hardware-acceptance", "status": "completed",
            "hardware_cohort": result["identity"]["hardware_cohort"], "producer": PRODUCER,
            "artifacts": artifacts,
            **({"source_bundles": list(document["source_bundles"])} if isinstance(document.get("source_bundles"), list) else {}),
        }
        (temporary / "run.manifest.json").write_bytes(_bytes(manifest))
        temporary.rename(destination)
    except BaseException:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


__all__ = ["FinalAcceptanceError", "compose_final_acceptance", "render_final_acceptance_html", "write_final_acceptance_bundle"]
