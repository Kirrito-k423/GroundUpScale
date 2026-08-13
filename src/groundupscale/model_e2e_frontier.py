"""Public model-level Schedule Frontier Run Bundle contract."""

from __future__ import annotations

from hashlib import sha256
import json
from html import escape
from math import isfinite, sqrt
from pathlib import Path
import tempfile
from typing import Any, Mapping

from groundupscale.ir import canonical_data
from groundupscale.run_bundle import RUN_ID_PATTERN, RunBundleExistsError
from groundupscale.schedule_model import ScheduleKind
from groundupscale.scheduling import BoundEvent, compose_schedule_bound


INPUT_SCHEMA = "groundupscale.dev/model-e2e-frontier-input/v1alpha1"
RESULT_SCHEMA = "groundupscale.dev/model-e2e-frontier-result/v1alpha1"
REPORT_SCHEMA = "groundupscale.dev/model-e2e-frontier-report/v1alpha1"
MANIFEST_SCHEMA = "groundupscale.dev/run-manifest/v1alpha1"
PRODUCER = "groundupscale@0.1.0"


class ModelE2EFrontierError(ValueError):
    """The public model E2E contract is malformed or cannot be replayed."""


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            canonical_data(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _digest_document(value: object) -> str:
    return _digest_bytes(_json_bytes(value))


def _mapping(value: object, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelE2EFrontierError(reason)
    return value


def _nonempty(value: object, reason: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelE2EFrontierError(reason)
    return value


def _finite_nonnegative(value: object, reason: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(float(value))
        or value < 0
    ):
        raise ModelE2EFrontierError(reason)
    return float(value)


def _refs(value: object, reason: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ModelE2EFrontierError(reason)
    return list(value)


def _axis(value: object, name: str) -> dict[str, Any]:
    axis = _mapping(value, f"invalid-{name.replace('_', '-')}-axis")
    status = axis.get("status")
    if status not in {"known", "unknown"}:
        raise ModelE2EFrontierError(f"invalid-{name.replace('_', '-')}-axis")
    if status == "known":
        numeric_value = _finite_nonnegative(
            axis.get("value_ns"),
            f"invalid-{name.replace('_', '-')}-axis",
        )
        if name == "observation" and numeric_value == 0:
            raise ModelE2EFrontierError("invalid-observation-axis")
        return {
            "status": "known",
            "value_ns": numeric_value,
            "evidence_refs": _refs(
                axis.get("evidence_refs"),
                f"invalid-{name.replace('_', '-')}-axis",
            ),
        }
    return {
        "status": "unknown",
        "value_ns": None,
        "reason_code": _nonempty(
            axis.get("reason_code"),
            f"invalid-{name.replace('_', '-')}-axis",
        ),
    }


def _candidate(
    value: object,
    *,
    expected_operation_class: str,
    expected_stable_path: str,
) -> dict[str, Any]:
    candidate = _mapping(value, "invalid-model-candidate")
    operation_class = _nonempty(
        candidate.get("operation_class"), "invalid-model-candidate"
    )
    if operation_class != expected_operation_class:
        raise ModelE2EFrontierError("candidate-operation-class-mismatch")
    stable_path = _nonempty(
        candidate.get("stable_path"), "invalid-model-candidate"
    )
    if stable_path != expected_stable_path:
        raise ModelE2EFrontierError("candidate-stable-path-mismatch")
    claims_value = candidate.get("resource_claims")
    if not isinstance(claims_value, list) or not claims_value:
        raise ModelE2EFrontierError("invalid-model-candidate-resource-claims")
    claims: list[dict[str, Any]] = []
    for claim_value in claims_value:
        claim = _mapping(
            claim_value, "invalid-model-candidate-resource-claims"
        )
        claim_duration = _finite_nonnegative(
            claim.get("duration_ns"),
            "invalid-model-candidate-resource-claims",
        )
        claims.append(
            {
                "resource_id": _nonempty(
                    claim.get("resource_id"),
                    "invalid-model-candidate-resource-claims",
                ),
                "duration_ns": claim_duration,
                "evidence_refs": _refs(
                    claim.get("evidence_refs"),
                    "invalid-model-candidate-resource-claims",
                ),
            }
        )
    duration_ns = _finite_nonnegative(
        candidate.get("duration_ns"), "invalid-model-candidate"
    )
    if max(claim["duration_ns"] for claim in claims) != duration_ns:
        raise ModelE2EFrontierError("candidate-resource-claims-do-not-reconcile")
    return {
        "candidate_id": _nonempty(
            candidate.get("candidate_id"), "invalid-model-candidate"
        ),
        "stable_path": stable_path,
        "operation_class": operation_class,
        "duration_ns": duration_ns,
        "standard_uncertainty_ns": _finite_nonnegative(
            candidate.get("standard_uncertainty_ns"),
            "invalid-model-candidate",
        ),
        "evidence_refs": _refs(
            candidate.get("evidence_refs"), "invalid-model-candidate"
        ),
        "resource_claims": claims,
    }


def compose_model_e2e_frontier(document: Mapping[str, object]) -> dict[str, Any]:
    """Derive four non-overwriting axes from one locked model input."""

    if document.get("schema") != INPUT_SCHEMA:
        raise ModelE2EFrontierError("unsupported-model-e2e-frontier-input")
    evidence = _mapping(document.get("evidence"), "invalid-model-evidence")
    if evidence.get("classification") != "deterministic-synthetic":
        raise ModelE2EFrontierError("invalid-model-evidence-classification")
    if evidence.get("promotion_eligible") is not False:
        raise ModelE2EFrontierError(
            "synthetic-evidence-cannot-be-promotion-eligible"
        )
    evidence_refs = _refs(evidence.get("evidence_refs"), "invalid-model-evidence")
    model = _mapping(document.get("model"), "invalid-model-coverage")
    expected_count = model.get("expected_semantic_leaf_count")
    repeated_indices = model.get("repeated_layer_indices")
    leaves_value = model.get("semantic_leaves")
    if (
        not isinstance(expected_count, int)
        or expected_count <= 0
        or repeated_indices != [0, 1]
        or not isinstance(leaves_value, list)
        or len(leaves_value) != expected_count
    ):
        raise ModelE2EFrontierError("invalid-model-coverage")

    seen_paths: set[str] = set()
    seen_candidate_ids: set[str] = set()
    predicted_leaves: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    missing_operator: list[dict[str, str]] = []
    operator_duration_ns = 0.0
    candidate_uncertainties: list[float] = []
    resolved_candidates: list[dict[str, Any]] = []
    for leaf_value in leaves_value:
        leaf = _mapping(leaf_value, "invalid-semantic-leaf")
        stable_path = _nonempty(leaf.get("stable_path"), "invalid-semantic-leaf")
        operation = _nonempty(
            leaf.get("operation_class"), "invalid-semantic-leaf"
        )
        requirements = leaf.get("requirements")
        mandatory_classes = leaf.get("mandatory_operation_classes")
        if stable_path in seen_paths:
            raise ModelE2EFrontierError("duplicate-semantic-leaf-stable-path")
        if not isinstance(requirements, list) or not requirements:
            raise ModelE2EFrontierError("invalid-semantic-leaf-requirements")
        requirement_classes = [
            requirement.get("operation_class")
            if isinstance(requirement, Mapping)
            else None
            for requirement in requirements
        ]
        if (
            not isinstance(mandatory_classes, list)
            or not mandatory_classes
            or requirement_classes != mandatory_classes
            or not all(
                isinstance(item, str) and item for item in mandatory_classes
            )
        ):
            raise ModelE2EFrontierError("mandatory-operation-class-mismatch")
        seen_paths.add(stable_path)
        leaf_duration_ns = 0.0
        leaf_candidate_ids: list[str] = []
        leaf_evidence_refs: list[str] = []
        leaf_missing: list[str] = []
        for requirement_value in requirements:
            requirement = _mapping(
                requirement_value, "invalid-semantic-leaf-requirement"
            )
            operation_class = _nonempty(
                requirement.get("operation_class"),
                "invalid-semantic-leaf-requirement",
            )
            required_evidence = _nonempty(
                requirement.get("required_evidence"),
                "invalid-semantic-leaf-requirement",
            )
            candidate_value = requirement.get("candidate")
            if candidate_value is None:
                leaf_missing.append(operation_class)
                missing_item = {
                    "stable_path": stable_path,
                    "operation_class": operation_class,
                    "required_evidence": required_evidence,
                }
                missing.append(missing_item)
                missing_operator.append(missing_item)
                continue
            candidate = _candidate(
                candidate_value,
                expected_operation_class=operation_class,
                expected_stable_path=stable_path,
            )
            if candidate["candidate_id"] in seen_candidate_ids:
                raise ModelE2EFrontierError("duplicate-model-candidate-id")
            seen_candidate_ids.add(candidate["candidate_id"])
            leaf_candidate_ids.append(candidate["candidate_id"])
            leaf_evidence_refs.extend(candidate["evidence_refs"])
            leaf_duration_ns += candidate["duration_ns"]
            candidate_uncertainties.append(candidate["standard_uncertainty_ns"])
            resolved_candidates.append(candidate)
        predicted_leaves.append(
            {
                "stable_path": stable_path,
                "operation_class": operation,
                "status": "unknown" if leaf_missing else "known",
                "duration_ns": None if leaf_missing else leaf_duration_ns,
                "candidate_ids": leaf_candidate_ids,
                "evidence_refs": leaf_evidence_refs,
                "missing_operation_classes": leaf_missing,
            }
        )
        if not leaf_missing:
            operator_duration_ns += leaf_duration_ns

    for index in (0, 1):
        marker = f"/layer_{index}/"
        if sum(marker in path for path in seen_paths) != expected_count // 2:
            raise ModelE2EFrontierError("indexed-repeated-layer-coverage-mismatch")

    schedule = _mapping(document.get("schedule"), "invalid-model-schedule")
    schedule_refs = _refs(
        schedule.get("evidence_refs"), "invalid-model-schedule"
    )
    effects_value = schedule.get("mandatory_effects")
    mandatory_effect_ids = schedule.get("mandatory_effect_ids")
    if (
        schedule.get("kind") != "serialized-unfused"
        or not isinstance(effects_value, list)
        or not effects_value
    ):
        raise ModelE2EFrontierError("invalid-model-schedule")
    effect_ids = [
        effect.get("effect_id") if isinstance(effect, Mapping) else None
        for effect in effects_value
    ]
    if (
        not isinstance(mandatory_effect_ids, list)
        or not mandatory_effect_ids
        or effect_ids != mandatory_effect_ids
        or not all(
            isinstance(item, str) and item for item in mandatory_effect_ids
        )
    ):
        raise ModelE2EFrontierError("mandatory-schedule-effect-mismatch")
    effect_duration_ns = 0.0
    effect_uncertainties: list[float] = []
    schedule_effects: list[dict[str, Any]] = []
    for effect_value in effects_value:
        effect = _mapping(effect_value, "invalid-mandatory-schedule-effect")
        effect_id = _nonempty(
            effect.get("effect_id"), "invalid-mandatory-schedule-effect"
        )
        operation_class = _nonempty(
            effect.get("operation_class"), "invalid-mandatory-schedule-effect"
        )
        required_evidence = _nonempty(
            effect.get("required_evidence"), "invalid-mandatory-schedule-effect"
        )
        candidate_value = effect.get("candidate")
        if candidate_value is None:
            missing.append(
                {
                    "stable_path": f"schedule/{effect_id}",
                    "operation_class": operation_class,
                    "required_evidence": required_evidence,
                }
            )
            schedule_effects.append(
                {
                    "effect_id": effect_id,
                    "operation_class": operation_class,
                    "status": "unknown",
                    "duration_ns": None,
                }
            )
            continue
        candidate = _candidate(
            candidate_value,
            expected_operation_class=operation_class,
            expected_stable_path=f"schedule/{effect_id}",
        )
        if candidate["candidate_id"] in seen_candidate_ids:
            raise ModelE2EFrontierError("duplicate-model-candidate-id")
        seen_candidate_ids.add(candidate["candidate_id"])
        effect_duration_ns += candidate["duration_ns"]
        effect_uncertainties.append(candidate["standard_uncertainty_ns"])
        resolved_candidates.append(candidate)
        schedule_effects.append(
            {
                "effect_id": effect_id,
                "operation_class": operation_class,
                "status": "known",
                "duration_ns": candidate["duration_ns"],
                "candidate_id": candidate["candidate_id"],
                "evidence_refs": candidate["evidence_refs"],
            }
        )

    dependencies_value = schedule.get("dependencies")
    if not isinstance(dependencies_value, list):
        raise ModelE2EFrontierError("invalid-model-schedule-dependencies")
    explicit_dependencies: list[dict[str, Any]] = []
    for dependency_value in dependencies_value:
        dependency = _mapping(
            dependency_value, "invalid-model-schedule-dependencies"
        )
        explicit_dependencies.append(
            {
                "source": _nonempty(
                    dependency.get("source"),
                    "invalid-model-schedule-dependencies",
                ),
                "target": _nonempty(
                    dependency.get("target"),
                    "invalid-model-schedule-dependencies",
                ),
                "evidence_refs": _refs(
                    dependency.get("evidence_refs"),
                    "invalid-model-schedule-dependencies",
                ),
            }
        )

    axes_value = _mapping(document.get("axes"), "invalid-model-axes")
    resource_axis = _axis(
        axes_value.get("resource_physical_floor"), "resource_physical_floor"
    )
    observation_axis = _axis(axes_value.get("observation"), "observation")
    missing_classes = sorted({item["operation_class"] for item in missing})
    missing_operator_classes = sorted(
        {item["operation_class"] for item in missing_operator}
    )
    model_evidence_complete = not missing
    operator_axis = (
        {
            "status": "known",
            "value_ns": operator_duration_ns,
            "aggregation": {
                "kind": "serialized-semantic-leaf-frontiers",
                "semantic_leaf_count": expected_count,
            },
            "evidence_refs": sorted(
                {ref for leaf in predicted_leaves for ref in leaf["evidence_refs"]}
            ),
        }
        if not missing_operator
        else {
            "status": "unknown",
            "value_ns": None,
            "reason_code": "mandatory-operator-evidence-missing",
            "missing_operation_classes": missing_operator_classes,
        }
    )
    selected_duration_ns = operator_duration_ns + effect_duration_ns
    physical_events: list[dict[str, Any]] = []
    schedule_composition: dict[str, Any] | None = None
    if model_evidence_complete:
        candidate_ids = [candidate["candidate_id"] for candidate in resolved_candidates]
        expected_dependencies = list(zip(candidate_ids, candidate_ids[1:]))
        actual_dependencies = [
            (dependency["source"], dependency["target"])
            for dependency in explicit_dependencies
        ]
        if actual_dependencies != expected_dependencies:
            raise ModelE2EFrontierError("explicit-schedule-dependency-mismatch")
        predecessor_by_id = {candidate_id: [] for candidate_id in candidate_ids}
        for source, target in actual_dependencies:
            predecessor_by_id[target].append(source)
        bound_events: list[BoundEvent] = []
        for candidate in resolved_candidates:
            resource_times = tuple(
                (claim["resource_id"], claim["duration_ns"])
                for claim in candidate["resource_claims"]
            )
            bound_events.append(
                BoundEvent(
                    event_id=candidate["candidate_id"],
                    predecessor_ids=tuple(
                        predecessor_by_id[candidate["candidate_id"]]
                    ),
                    local_duration_ns=candidate["duration_ns"],
                    resource_times_ns=resource_times,
                )
            )
            physical_events.append(
                {
                    "event_id": candidate["candidate_id"],
                    "stable_path": candidate["stable_path"],
                    "operation_class": candidate["operation_class"],
                    "duration_ns": candidate["duration_ns"],
                    "resource_claims": candidate["resource_claims"],
                    "evidence_refs": candidate["evidence_refs"],
                }
            )
        bound = compose_schedule_bound(
            tuple(bound_events), schedule=ScheduleKind.SERIALIZED
        )
        if bound.selected_duration_ns != selected_duration_ns:
            raise ModelE2EFrontierError("selected-schedule-does-not-reconcile")
        schedule_composition = {
            "serialized_unfused_duration_ns": bound.serialized_duration_ns,
            "critical_path_duration_ns": bound.critical_path_duration_ns,
            "shared_resource_duration_ns": bound.resource_duration_ns,
            "ideal_dag_duration_ns": bound.ideal_dag_duration_ns,
            "selected_feasible_duration_ns": bound.selected_duration_ns,
            "limiting_resource": bound.limiting_resource,
            "critical_path_event_ids": list(bound.critical_path_event_ids),
        }
    schedule_axis = (
        {
            "status": "known",
            "value_ns": selected_duration_ns,
            "evidence_refs": schedule_refs,
        }
        if model_evidence_complete
        else {
            "status": "unknown",
            "value_ns": None,
            "reason_code": "mandatory-model-evidence-missing",
            "missing_operation_classes": missing_classes,
        }
    )
    uncertainty = _mapping(document.get("uncertainty"), "invalid-uncertainty")
    if uncertainty.get("combination") != "root-sum-square":
        raise ModelE2EFrontierError("invalid-uncertainty-combination")
    schedule_component = _finite_nonnegative(
        uncertainty.get("schedule_component_ns"), "invalid-uncertainty"
    )
    observation_component = _finite_nonnegative(
        uncertainty.get("observation_component_ns"), "invalid-uncertainty"
    )
    combined_uncertainty_ns = sqrt(
        sum(value * value for value in candidate_uncertainties)
        + sum(value * value for value in effect_uncertainties)
        + schedule_component * schedule_component
        + observation_component * observation_component
    )
    if model_evidence_complete and observation_axis["status"] == "known":
        observation_ns = observation_axis["value_ns"]
        relative_error = (selected_duration_ns - observation_ns) / observation_ns
        comparison = {
            "status": "evaluated",
            "absolute_gap_ns": selected_duration_ns - observation_ns,
            "schedule_to_observation_ratio": selected_duration_ns / observation_ns,
            "frontier_efficiency": selected_duration_ns / observation_ns,
            "relative_prediction_error": relative_error,
            "error_status": "evaluated-schedule-frontier",
            "combined_uncertainty_ns": combined_uncertainty_ns,
        }
    else:
        comparison = {
            "status": "unknown",
            "absolute_gap_ns": None,
            "schedule_to_observation_ratio": None,
            "frontier_efficiency": None,
            "relative_prediction_error": None,
            "error_status": "unknown-incomplete-schedule-frontier",
            "combined_uncertainty_ns": None,
        }

    all_axes_known = all(
        axis["status"] == "known"
        for axis in (
            resource_axis,
            operator_axis,
            schedule_axis,
            observation_axis,
        )
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete" if all_axes_known else "unknown",
        "model_id": _nonempty(model.get("model_id"), "invalid-model-coverage"),
        "hardware_cohort": _nonempty(
            evidence.get("hardware_cohort"), "invalid-model-evidence"
        ),
        "evidence": {
            "classification": evidence["classification"],
            "authority": "synthetic-contract-only",
            "source_issue": evidence.get("source_issue"),
            "promotion_eligible": evidence.get("promotion_eligible"),
            "evidence_refs": evidence_refs,
        },
        "coverage": {
            "semantic_leaf_count": expected_count,
            "repeated_layer_indices": list(repeated_indices),
            "stable_path_unique": len(seen_paths) == expected_count,
            "predicted_leaves": predicted_leaves,
        },
        "axes": {
            "resource_physical_floor": resource_axis,
            "operator_achievable_frontier": operator_axis,
            "schedule_achievable_frontier": schedule_axis,
            "observation": observation_axis,
        },
        "schedule": {
            "policy_id": _nonempty(
                schedule.get("policy_id"), "invalid-model-schedule"
            ),
            "version": _nonempty(
                schedule.get("version"), "invalid-model-schedule"
            ),
            "kind": schedule["kind"],
            "mandatory_effects": schedule_effects,
            "physical_events": physical_events,
            "explicit_dependencies": explicit_dependencies,
            "serialized_unfused_duration_ns": (
                schedule_composition["serialized_unfused_duration_ns"]
                if schedule_composition is not None
                else None
            ),
            "critical_path_duration_ns": (
                schedule_composition["critical_path_duration_ns"]
                if schedule_composition is not None
                else None
            ),
            "shared_resource_duration_ns": (
                schedule_composition["shared_resource_duration_ns"]
                if schedule_composition is not None
                else None
            ),
            "ideal_dag_duration_ns": (
                schedule_composition["ideal_dag_duration_ns"]
                if schedule_composition is not None
                else None
            ),
            "selected_feasible_duration_ns": (
                schedule_composition["selected_feasible_duration_ns"]
                if schedule_composition is not None
                else None
            ),
            "limiting_resource": (
                schedule_composition["limiting_resource"]
                if schedule_composition is not None
                else None
            ),
            "critical_path_event_ids": (
                schedule_composition["critical_path_event_ids"]
                if schedule_composition is not None
                else []
            ),
        },
        "uncertainty": {
            "policy_id": _nonempty(
                uncertainty.get("policy_id"), "invalid-uncertainty"
            ),
            "version": _nonempty(
                uncertainty.get("version"), "invalid-uncertainty"
            ),
            "combination": "root-sum-square",
            "candidate_components_ns": candidate_uncertainties,
            "schedule_effect_components_ns": effect_uncertainties,
            "schedule_component_ns": schedule_component,
            "observation_component_ns": observation_component,
            "combined_ns": (
                combined_uncertainty_ns if model_evidence_complete else None
            ),
            "evidence_refs": _refs(
                uncertainty.get("evidence_refs"), "invalid-uncertainty"
            ),
        },
        "comparison": comparison,
        "missing_evidence": missing,
        "derivation": {
            "input_sha256": _digest_document(document),
            "steps": [
                "validate-indexed-stable-path-coverage",
                "resolve-mandatory-operation-candidates",
                "resolve-mandatory-schedule-effects",
                "compose-selected-feasible-schedule",
                "compare-same-boundary-observation",
            ],
        },
    }
    result["derivation"]["result_sha256"] = _digest_document(
        {key: value for key, value in result.items() if key != "derivation"}
    )
    return result


def render_model_e2e_frontier_report(result: Mapping[str, object]) -> str:
    """Project the verified machine result without a second derivation."""

    if result.get("schema") != RESULT_SCHEMA:
        raise ModelE2EFrontierError("invalid-model-e2e-frontier-result")
    axes = _mapping(result.get("axes"), "invalid-model-e2e-frontier-result")
    labels = {
        "resource_physical_floor": "Resource Physical Floor",
        "operator_achievable_frontier": "Operator Achievable Frontier",
        "schedule_achievable_frontier": "Schedule Achievable Frontier",
        "observation": "E2E Observation",
    }
    lines = [
        f"<h1>Model E2E Frontier: {escape(str(result['status']))}</h1>",
        f"<p>Model: {escape(str(result['model_id']))}</p>",
        f"<p>Hardware Cohort: {escape(str(result['hardware_cohort']))}</p>",
        "<ul>",
    ]
    for name, label in labels.items():
        axis = _mapping(axes.get(name), "invalid-model-e2e-frontier-result")
        if axis.get("status") == "known":
            lines.append(
                f"<li>{label}: {float(axis['value_ns']) / 1_000_000:.6f} ms</li>"
            )
        else:
            lines.append(
                f"<li>{label}: unknown "
                f"({escape(str(axis.get('reason_code', 'unknown')))})</li>"
            )
    lines.append("</ul>")
    comparison = _mapping(
        result.get("comparison"), "invalid-model-e2e-frontier-result"
    )
    lines.append(
        "<p>Relative prediction error: "
        + (
            f"{float(comparison['relative_prediction_error']):.9f}"
            if comparison.get("relative_prediction_error") is not None
            else "unknown ("
            + escape(str(comparison.get("error_status", "unknown")))
            + ")"
        )
        + "</p>"
    )
    missing = result.get("missing_evidence")
    if isinstance(missing, list) and missing:
        lines.append("<h2>Missing mandatory evidence</h2><ul>")
        for item in missing:
            if isinstance(item, Mapping):
                lines.append(
                    f"<li>{escape(str(item.get('stable_path')))}: "
                    f"{escape(str(item.get('operation_class')))}; "
                    f"required={escape(str(item.get('required_evidence')))}</li>"
                )
        lines.append("</ul>")
    lines.append(
        "<p>Four axes are independent and never overwrite one another.</p>"
    )
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>GroundUpScale Model E2E Frontier</title></head><body>\n"
        + "\n".join(lines)
        + "\n</body></html>\n"
    )


def write_model_e2e_frontier_bundle(
    document: Mapping[str, object],
    artifact_store: str | Path,
    *,
    run_id: str,
) -> Path:
    """Write one immutable replayable bundle from the public report seam."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ModelE2EFrontierError("unsafe-model-e2e-run-id")
    result = compose_model_e2e_frontier(document)
    report = render_model_e2e_frontier_report(result)
    runs_root = Path(artifact_store).resolve() / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    destination = runs_root / run_id
    if destination.exists():
        raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
    artifacts: list[dict[str, Any]] = []

    def write(
        role: str,
        relative: str,
        payload: bytes,
        media_type: str,
        schema: str,
        inputs: list[str],
    ) -> None:
        path = temporary / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts.append(
            {
                "role": role,
                "path": relative,
                "media_type": media_type,
                "schema": schema,
                "sha256": _digest_bytes(payload),
                "produced_by": PRODUCER,
                "inputs": inputs,
            }
        )

    try:
        write(
            "model-e2e-frontier-input",
            "resolved/model-e2e-frontier-input.json",
            _json_bytes(document),
            "application/json",
            INPUT_SCHEMA,
            [],
        )
        write(
            "prediction-observation-comparison",
            "comparison/model-e2e-frontier.json",
            _json_bytes(result),
            "application/json",
            RESULT_SCHEMA,
            ["model-e2e-frontier-input"],
        )
        write(
            "html-report",
            "reports/report.html",
            report.encode("utf-8"),
            "text/html",
            REPORT_SCHEMA,
            ["prediction-observation-comparison"],
        )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "run_id": run_id,
            "bundle_kind": "model-e2e-frontier",
            "status": result["status"],
            "hardware_cohort": result["hardware_cohort"],
            "producer_lineage": {
                "producer": PRODUCER,
                "source": "python://groundupscale.model_e2e_frontier",
            },
            "immutability": (
                "writer refuses an existing run_id; artifact digests and "
                "semantic replay are authoritative"
            ),
            "artifacts": artifacts,
        }
        (temporary / "run.manifest.json").write_bytes(_json_bytes(manifest))
        temporary.rename(destination)
    except BaseException:
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if temporary.exists():
            temporary.rmdir()
        raise
    return destination


def load_model_e2e_frontier_report(path: str | Path) -> dict[str, Any]:
    """Load machine and human forms only after public bundle verification."""

    from groundupscale.run_bundle import verify_run_bundle

    root = Path(path).resolve()
    verification = verify_run_bundle(root)
    if verification.get("passed") is not True:
        raise ModelE2EFrontierError(
            "model E2E Run Bundle failed verification: "
            + "; ".join(verification.get("failures", []))
        )
    return {
        "machine_result": json.loads(
            (root / "comparison/model-e2e-frontier.json").read_text(
                encoding="utf-8"
            )
        ),
        "human_report": (
            root / "reports/report.html"
        ).read_text(encoding="utf-8"),
    }


__all__ = [
    "INPUT_SCHEMA",
    "ModelE2EFrontierError",
    "compose_model_e2e_frontier",
    "load_model_e2e_frontier_report",
    "render_model_e2e_frontier_report",
    "write_model_e2e_frontier_bundle",
]
