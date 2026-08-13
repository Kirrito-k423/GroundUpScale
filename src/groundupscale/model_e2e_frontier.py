"""Public model-level Schedule Frontier Run Bundle contract."""

from __future__ import annotations

from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
import tempfile
from typing import Any, Mapping

from groundupscale.ir import canonical_data
from groundupscale.run_bundle import RUN_ID_PATTERN, RunBundleExistsError


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
        return {
            "status": "known",
            "value_ns": _finite_nonnegative(
                axis.get("value_ns"),
                f"invalid-{name.replace('_', '-')}-axis",
            ),
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
    return {
        "candidate_id": _nonempty(
            candidate.get("candidate_id"), "invalid-model-candidate"
        ),
        "stable_path": stable_path,
        "operation_class": operation_class,
        "duration_ns": _finite_nonnegative(
            candidate.get("duration_ns"), "invalid-model-candidate"
        ),
        "standard_uncertainty_ns": _finite_nonnegative(
            candidate.get("standard_uncertainty_ns"),
            "invalid-model-candidate",
        ),
        "evidence_refs": _refs(
            candidate.get("evidence_refs"), "invalid-model-candidate"
        ),
    }


def compose_model_e2e_frontier(document: Mapping[str, object]) -> dict[str, Any]:
    """Derive four non-overwriting axes from one locked model input."""

    if document.get("schema") != INPUT_SCHEMA:
        raise ModelE2EFrontierError("unsupported-model-e2e-frontier-input")
    evidence = _mapping(document.get("evidence"), "invalid-model-evidence")
    if evidence.get("classification") != "deterministic-synthetic":
        raise ModelE2EFrontierError("invalid-model-evidence-classification")
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
    for leaf_value in leaves_value:
        leaf = _mapping(leaf_value, "invalid-semantic-leaf")
        stable_path = _nonempty(leaf.get("stable_path"), "invalid-semantic-leaf")
        operation = _nonempty(
            leaf.get("operation_class"), "invalid-semantic-leaf"
        )
        requirements = leaf.get("requirements")
        if stable_path in seen_paths:
            raise ModelE2EFrontierError("duplicate-semantic-leaf-stable-path")
        if not isinstance(requirements, list) or not requirements:
            raise ModelE2EFrontierError("invalid-semantic-leaf-requirements")
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
    if (
        schedule.get("kind") != "serialized-unfused"
        or not isinstance(effects_value, list)
        or not effects_value
    ):
        raise ModelE2EFrontierError("invalid-model-schedule")
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

    axes_value = _mapping(document.get("axes"), "invalid-model-axes")
    resource_axis = _axis(
        axes_value.get("resource_physical_floor"), "resource_physical_floor"
    )
    observation_axis = _axis(axes_value.get("observation"), "observation")
    missing_classes = sorted({item["operation_class"] for item in missing})
    missing_operator_classes = sorted(
        {item["operation_class"] for item in missing_operator}
    )
    complete = not missing
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
    schedule_axis = (
        {
            "status": "known",
            "value_ns": selected_duration_ns,
            "evidence_refs": schedule_refs,
        }
        if complete
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
    if complete and observation_axis["status"] == "known":
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

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete" if complete else "unknown",
        "model_id": _nonempty(model.get("model_id"), "invalid-model-coverage"),
        "hardware_cohort": _nonempty(
            evidence.get("hardware_cohort"), "invalid-model-evidence"
        ),
        "evidence": {
            "classification": evidence["classification"],
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
            "serialized_unfused_duration_ns": (
                selected_duration_ns if complete else None
            ),
            "ideal_dag_duration_ns": (
                max(
                    [
                        leaf["duration_ns"]
                        for leaf in predicted_leaves
                        if leaf["duration_ns"] is not None
                    ]
                    + [effect_duration_ns]
                )
                if complete
                else None
            ),
            "selected_feasible_duration_ns": (
                selected_duration_ns if complete else None
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
            "combined_ns": combined_uncertainty_ns if complete else None,
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
        f"Model E2E Frontier: {result['status']}",
        f"Model: {result['model_id']}",
        f"Hardware Cohort: {result['hardware_cohort']}",
    ]
    for name, label in labels.items():
        axis = _mapping(axes.get(name), "invalid-model-e2e-frontier-result")
        if axis.get("status") == "known":
            lines.append(f"{label}: {float(axis['value_ns']) / 1_000_000:.6f} ms")
        else:
            lines.append(f"{label}: unknown ({axis.get('reason_code', 'unknown')})")
    comparison = _mapping(
        result.get("comparison"), "invalid-model-e2e-frontier-result"
    )
    lines.append(
        "Relative prediction error: "
        + (
            f"{float(comparison['relative_prediction_error']):.9f}"
            if comparison.get("relative_prediction_error") is not None
            else f"unknown ({comparison.get('error_status', 'unknown')})"
        )
    )
    missing = result.get("missing_evidence")
    if isinstance(missing, list) and missing:
        lines.append("Missing mandatory evidence:")
        for item in missing:
            if isinstance(item, Mapping):
                lines.append(
                    f"- {item.get('stable_path')}: {item.get('operation_class')}; "
                    f"required={item.get('required_evidence')}"
                )
    lines.append(
        "Four axes are independent and never overwrite one another."
    )
    return "\n".join(lines) + "\n"


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
            "reports/model-e2e-frontier.txt",
            report.encode("utf-8"),
            "text/plain",
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
            root / "reports/model-e2e-frontier.txt"
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
