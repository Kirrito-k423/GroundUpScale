"""Replayable E2E prediction-observation gap and reconciliation report."""

from __future__ import annotations

import csv
from hashlib import sha256
from html import escape
import io
import json
from math import fsum, isfinite, sqrt
from pathlib import Path
import tempfile
from typing import Any, Mapping

from groundupscale.ir import canonical_data
from groundupscale.run_bundle import RUN_ID_PATTERN, RunBundleExistsError


INPUT_SCHEMA = "groundupscale.dev/e2e-gap-report-input/v1alpha1"
RESULT_SCHEMA = "groundupscale.dev/e2e-gap-report/v1alpha1"
REPORT_SCHEMA = "groundupscale.dev/e2e-gap-report-html/v1alpha1"
TIERED_INPUT_SCHEMA = "groundupscale.dev/e2e-gap-report-input/v1alpha2"
TIERED_RESULT_SCHEMA = "groundupscale.dev/e2e-gap-report/v1alpha2"
TIERED_REPORT_SCHEMA = "groundupscale.dev/e2e-gap-report-html/v1alpha2"
PRODUCER = "groundupscale@0.1.0"
TIERED_DERIVATION_SCHEMA = (
    "groundupscale.dev/e2e-gap-report-value-derivation/v1alpha1"
)


class GapReportError(ValueError):
    """The report input cannot be interpreted without inventing evidence."""


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(canonical_data(value), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def _repository_root(start: Path) -> Path:
    current = start.resolve()
    while current != current.parent and not (current / "pyproject.toml").is_file():
        current = current.parent
    if not (current / "pyproject.toml").is_file():
        raise GapReportError("report-source-repository-not-found")
    return current


def _load_locked_source_artifact(
    document: Mapping[str, object],
    contract: Mapping[str, Any],
    repository_root: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    sources = document.get("source_bundles")
    if not isinstance(sources, list):
        raise GapReportError("tiered-report-source-lineage-missing")
    run_id = contract.get("run_id")
    relative_artifact = contract.get("artifact_path")
    artifact_digest = contract.get("artifact_sha256")
    source = next(
        (
            item
            for item in sources
            if isinstance(item, Mapping) and item.get("run_id") == run_id
        ),
        None,
    )
    if (
        source is None
        or not isinstance(relative_artifact, str)
        or not relative_artifact
        or Path(relative_artifact).is_absolute()
        or not isinstance(artifact_digest, str)
        or len(artifact_digest) != 64
    ):
        raise GapReportError("tiered-report-source-contract-invalid")
    source_relative = source.get("path")
    manifest_digest = source.get("manifest_sha256")
    if (
        not isinstance(source_relative, str)
        or Path(source_relative).is_absolute()
        or not isinstance(manifest_digest, str)
    ):
        raise GapReportError("tiered-report-source-contract-invalid")
    source_root = (repository_root / source_relative).resolve()
    artifact_path = (source_root / relative_artifact).resolve()
    try:
        source_root.relative_to(repository_root)
        artifact_path.relative_to(source_root)
    except ValueError as error:
        raise GapReportError("tiered-report-source-path-escape") from error
    manifest_path = source_root / "run.manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = artifact_path.read_bytes()
        artifact = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GapReportError("tiered-report-source-not-readable") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("run_id") != run_id
        or _digest(manifest_path.read_bytes()) != manifest_digest
        or _digest(payload) != artifact_digest
        or not isinstance(artifact, dict)
        or not any(
            isinstance(entry, Mapping)
            and entry.get("path") == relative_artifact
            and entry.get("sha256") == artifact_digest
            for entry in manifest.get("artifacts", [])
        )
    ):
        raise GapReportError("tiered-report-source-digest-mismatch")
    evidence_ref = (
        f"run://{run_id}@sha256:{manifest_digest}#{relative_artifact}"
    )
    return artifact, evidence_ref, manifest


def _cost_leaves(value: object) -> list[Mapping[str, Any]]:
    leaves: list[Mapping[str, Any]] = []

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            if (
                isinstance(node.get("operation"), str)
                and isinstance(node.get("stable_path"), str)
                and isinstance(node.get("metrics"), Mapping)
            ):
                leaves.append(node)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    paths = [str(item["stable_path"]) for item in leaves]
    if len(leaves) != 52 or len(paths) != len(set(paths)):
        raise GapReportError("tiered-report-cost-leaf-inventory-mismatch")
    return sorted(leaves, key=lambda item: str(item["stable_path"]))


def _validate_authority_projection(
    document: Mapping[str, object],
    schedule_authority: Mapping[str, Any],
    decomposition: Mapping[str, Any],
    schedule_manifest: Mapping[str, Any],
    observation_manifest: Mapping[str, Any],
) -> None:
    predicted = _mapping(document.get("predicted"), "missing-predicted-side")
    observed = _mapping(document.get("observed"), "missing-observed-side")
    identity = _mapping(document.get("identity"), "missing-report-identity")
    sources = document.get("source_bundles")
    if not isinstance(sources, list):
        raise GapReportError("tiered-report-source-lineage-missing")

    def manifest_ref(manifest: Mapping[str, Any]) -> str:
        source = next(
            (
                item
                for item in sources
                if isinstance(item, Mapping)
                and item.get("run_id") == manifest.get("run_id")
            ),
            None,
        )
        if not isinstance(source, Mapping) or not isinstance(
            source.get("manifest_sha256"), str
        ):
            raise GapReportError("tiered-report-source-lineage-missing")
        return (
            f"run://{manifest['run_id']}@sha256:"
            f"{source['manifest_sha256']}"
        )

    predicted_leaves = schedule_authority.get("coverage", {}).get(
        "predicted_leaves"
    )
    if not isinstance(predicted_leaves, list):
        raise GapReportError("missing-schedule-authority-leaves")
    expected_predicted_items = [
        {
            "stable_path": leaf.get("stable_path"),
            "operation_class": leaf.get("operation_class"),
            "status": leaf.get("status"),
            "duration_ns": leaf.get("duration_ns"),
            "standard_uncertainty_ns": None,
            "evidence_quality": (
                "evidence-qualified-candidate"
                if leaf.get("status") == "known"
                else "structured-unknown"
            ),
            "evidence_refs": leaf.get("evidence_refs"),
            "accounting_interval": [index, index + 1],
            "evidence_boundaries": leaf.get("missing_operation_classes"),
        }
        for index, leaf in enumerate(predicted_leaves)
        if isinstance(leaf, Mapping)
    ]
    missing_evidence = schedule_authority.get("missing_evidence")
    expected_boundaries = (
        [item.get("required_evidence") for item in missing_evidence]
        if isinstance(missing_evidence, list)
        else []
    )
    schedule = _mapping(
        schedule_authority.get("schedule"), "missing-schedule-authority"
    )
    uncertainty = _mapping(
        schedule_authority.get("uncertainty"),
        "missing-schedule-authority-uncertainty",
    )
    expected_predicted = {
        "identity": dict(identity),
        "status": (
            "known" if schedule_authority.get("status") == "complete" else "unknown"
        ),
        "e2e_duration_ns": schedule.get("selected_feasible_duration_ns"),
        "standard_uncertainty_ns": uncertainty.get("combined_ns"),
        "bound_kind": "point-prediction",
        "items": expected_predicted_items,
        "reason_code": "incomplete-schedule-frontier",
        "evidence_boundaries": expected_boundaries,
        "required_next_measurement": (
            "qualify every mandatory leaf and schedule effect in the same Hardware Cohort"
        ),
        "evidence_refs": [manifest_ref(schedule_manifest)],
    }
    if dict(predicted) != expected_predicted:
        raise GapReportError("tiered-report-predicted-authority-replay-mismatch")
    observed_decomposition = _mapping(
        decomposition.get("observed_decomposition"),
        "missing-observed-decomposition",
    )
    expected_observed_items = [
        {
            "stable_path": leaf.get("stable_path"),
            "operation_class": leaf.get("operation_class"),
            "status": "known",
            "duration_ns": leaf.get("duration_ns"),
            "standard_uncertainty_ns": leaf.get("standard_uncertainty_ns"),
            "evidence_quality": leaf.get("evidence_quality", "direct-qualified"),
            "evidence_refs": leaf.get("evidence_refs", []),
            "accounting_interval": [index, index + 1],
        }
        for index, leaf in enumerate(observed_decomposition.get("leaves", []))
        if isinstance(leaf, Mapping)
    ]
    expected_observed = {
        "identity": dict(identity),
        "status": observed_decomposition.get("status"),
        "e2e_duration_ns": observed_decomposition.get("e2e_duration_ns"),
        "standard_uncertainty_ns": None,
        "items": expected_observed_items,
        "reason_code": observed_decomposition.get("reason_code"),
        "evidence_boundaries": observed_decomposition.get(
            "evidence_boundaries", []
        ),
        "required_next_measurement": observed_decomposition.get(
            "required_next_measurement"
        ),
        "accounting": "interval-union-or-critical-path",
        "evidence_refs": [manifest_ref(observation_manifest)],
    }
    if dict(observed) != expected_observed:
        raise GapReportError("tiered-report-observed-authority-replay-mismatch")


def derive_tiered_iteration_report(
    document: Mapping[str, object], repository_root: str | Path
) -> dict[str, Any]:
    """Replay grade-D components and grade-B E2E from locked source artifacts."""

    derivation = _mapping(
        document.get("iteration_report_derivation"),
        "missing-iteration-report-derivation",
    )
    if derivation.get("schema") != TIERED_DERIVATION_SCHEMA:
        raise GapReportError("unsupported-iteration-report-derivation")
    artifacts = _mapping(
        derivation.get("artifacts"), "missing-iteration-report-source-artifacts"
    )
    root = _repository_root(Path(repository_root))
    cost_ir, cost_ref, model_manifest = _load_locked_source_artifact(
        document,
        _mapping(artifacts.get("cost_ir"), "missing-cost-ir-source"),
        root,
    )
    backend, backend_ref, backend_manifest = _load_locked_source_artifact(
        document,
        _mapping(artifacts.get("hardware_backend"), "missing-backend-source"),
        root,
    )
    decomposition, observation_ref, observation_manifest = _load_locked_source_artifact(
        document,
        _mapping(
            artifacts.get("baseline_observation"),
            "missing-baseline-observation-source",
        ),
        root,
    )
    execution_contract, contract_ref, contract_manifest = (
        _load_locked_source_artifact(
            document,
            _mapping(
                artifacts.get("execution_contract"),
                "missing-execution-contract-source",
            ),
            root,
        )
    )
    schedule_authority, schedule_ref, schedule_manifest = (
        _load_locked_source_artifact(
            document,
            _mapping(
                artifacts.get("schedule_authority"),
                "missing-schedule-authority-source",
            ),
            root,
        )
    )
    identity = _mapping(document.get("identity"), "missing-report-identity")
    observed_identity = _mapping(
        decomposition.get("identity"), "missing-observation-identity"
    )
    expected_identity = {
        "case": observed_identity.get("benchmark_case"),
        "shape": observed_identity.get("shape"),
        "dtype": execution_contract.get("dtype"),
        "candidate_id": observed_identity.get("candidate_id"),
        "hardware_cohort": observed_identity.get("hardware_cohort"),
        "completion_boundary": observed_identity.get("completion_boundary"),
    }
    if dict(identity) != expected_identity:
        raise GapReportError("tiered-report-identity-mismatch")
    if (
        model_manifest != backend_manifest
        or model_manifest != contract_manifest
        or model_manifest.get("hardware_cohort") != identity["hardware_cohort"]
        or observation_manifest.get("hardware_cohort")
        != identity["hardware_cohort"]
        or schedule_manifest.get("hardware_cohort") != identity["hardware_cohort"]
        or schedule_authority.get("model_id") != "two-layer-transformer-prefill"
        or schedule_authority.get("hardware_cohort")
        != identity["hardware_cohort"]
        or schedule_authority.get("axes", {}).get("observation", {}).get(
            "value_ns"
        )
        != decomposition.get("baseline_e2e_observation", {}).get("median_ns")
    ):
        raise GapReportError("tiered-report-source-identity-mismatch")
    shape_contract = _mapping(
        execution_contract.get("shape"), "missing-execution-shape-contract"
    )
    if (
        execution_contract.get("dtype") != "float32"
        or shape_contract.get("dtype") != "float32"
        or shape_contract.get("bindings")
        != {"B": 1, "D": 64, "H": 512, "I": 2048, "NH": 8, "S": 512}
        or identity["shape"] != [1, 512, 512]
        or execution_contract.get("baseline_timing", {}).get(
            "completion_protocol"
        )
        != str(identity["completion_boundary"]).replace(
            "end-npu-event", "end-event", 1
        )
    ):
        raise GapReportError("tiered-report-execution-contract-mismatch")
    model_run_id = model_manifest.get("run_id")
    model_manifest_digest = next(
        (
            source.get("manifest_sha256")
            for source in document.get("source_bundles", [])
            if isinstance(source, Mapping) and source.get("run_id") == model_run_id
        ),
        None,
    )
    observed_anchor = next(
        (
            source
            for source in observation_manifest.get("source_runs", [])
            if isinstance(source, Mapping) and source.get("run_id") == model_run_id
        ),
        None,
    )
    scheduled_anchor = next(
        (
            source
            for source in schedule_authority.get("evidence", {}).get(
                "source_bundles", []
            )
            if isinstance(source, Mapping)
            and source.get("run_id") == model_run_id
        ),
        None,
    )
    if (
        not isinstance(observed_anchor, Mapping)
        or not isinstance(scheduled_anchor, Mapping)
        or observed_anchor.get("manifest_sha256") != model_manifest_digest
        or scheduled_anchor.get("manifest_sha256") != model_manifest_digest
    ):
        raise GapReportError("tiered-report-model-source-anchor-mismatch")
    _validate_authority_projection(
        document,
        schedule_authority,
        decomposition,
        schedule_manifest,
        observation_manifest,
    )
    model_policy = _mapping(
        derivation.get("prediction_model"), "missing-prediction-model"
    )
    if (
        model_policy.get("policy_id")
        != "serialized-resource-model-with-dispatch-floor-v1"
        or model_policy.get("version") != "1"
        or model_policy.get("compute_efficiency") != 0.50
        or model_policy.get("memory_efficiency") != 0.50
        or model_policy.get("dispatch_floor_ns") != 15_000.0
        or model_policy.get("schedule") != "serialized-unfused"
        or model_policy.get("purpose") != "iteration-prior-only"
    ):
        raise GapReportError("unsupported-prediction-model")
    observation_policy = _mapping(
        derivation.get("observation_component_model"),
        "missing-observation-component-model",
    )
    if observation_policy != {
        "policy_id": "scale-predicted-weights-to-observed-e2e-v1",
        "version": "1",
        "purpose": "diagnostic-attribution-only",
    }:
        raise GapReportError("unsupported-observation-component-model")
    compute_efficiency = _number(
        model_policy.get("compute_efficiency"), "invalid-compute-efficiency"
    )
    memory_efficiency = _number(
        model_policy.get("memory_efficiency"), "invalid-memory-efficiency"
    )
    dispatch_ns = _number(
        model_policy.get("dispatch_floor_ns"), "invalid-dispatch-floor"
    )
    if not 0 < compute_efficiency <= 1 or not 0 < memory_efficiency <= 1:
        raise GapReportError("invalid-resource-efficiency")
    capabilities = backend.get("measured_capabilities")
    if not isinstance(capabilities, list):
        raise GapReportError("missing-measured-capabilities")
    rates = {
        item.get("resource"): item.get("robust_achievable_rate")
        for item in capabilities
        if isinstance(item, Mapping)
    }
    compute_rate = _number(rates.get("compute.fp32"), "missing-compute-rate")
    memory_rate = _number(rates.get("memory.hbm"), "missing-memory-rate")
    leaves = _cost_leaves(cost_ir.get("root"))
    predicted_items = []
    for index, leaf in enumerate(leaves):
        metrics = _mapping(leaf.get("metrics"), "invalid-cost-leaf-metrics")
        flops = _number(metrics.get("flops"), "invalid-cost-leaf-flops")
        materialized_bytes = _number(
            metrics.get("materialized_read_bytes"), "invalid-cost-leaf-bytes"
        ) + _number(
            metrics.get("materialized_write_bytes"), "invalid-cost-leaf-bytes"
        )
        compute_ns = flops * 1_000_000_000 / compute_rate / compute_efficiency
        memory_ns = (
            materialized_bytes * 1_000_000_000 / memory_rate / memory_efficiency
        )
        duration_ns = max(compute_ns, memory_ns, dispatch_ns)
        cost_path = str(leaf["stable_path"])
        stable_path = cost_path.removeprefix("cost/")
        predicted_items.append(
            {
                "stable_path": stable_path,
                "operation_class": leaf["operation"],
                "duration_ns": duration_ns,
                "evidence_grade": "D",
                "generation_stage": "resource-model",
                "method": "serialized-resource-model-with-dispatch-floor-v1",
                "permitted_use": "hypothesis-only",
                "uncertainty_interval_ns": [duration_ns * 0.5, duration_ns * 2.0],
                "evidence_refs": [cost_ref, backend_ref, contract_ref, schedule_ref],
                "accounting_interval": [index, index + 1],
                "resource_estimate": {
                    "compute_ns": compute_ns,
                    "memory_ns": memory_ns,
                    "dispatch_floor_ns": dispatch_ns,
                    "selected_ns": duration_ns,
                },
            }
        )
    predicted_e2e = fsum(item["duration_ns"] for item in predicted_items)
    baseline = _mapping(
        decomposition.get("baseline_e2e_observation"),
        "missing-baseline-observation",
    )
    if baseline.get("status") != "valid":
        raise GapReportError("baseline-observation-not-valid")
    observed_e2e = _number(
        baseline.get("median_ns"), "invalid-baseline-observation"
    )
    source_summary = _mapping(
        baseline.get("timing_summary"), "missing-baseline-timing-summary"
    )
    if _number(source_summary.get("median_ns"), "invalid-baseline-median") != observed_e2e:
        raise GapReportError("baseline-observation-median-mismatch")
    report_policy = derivation.get("report_policy")
    if (
        not isinstance(report_policy, Mapping)
        or report_policy.get("policy_id") != "tiered-report-values-v1"
        or report_policy.get("version") != "1"
        or report_policy.get("grade_minimum_intervals")
        != {"B": [0.85, 1.15], "C": [0.70, 1.30], "D": [0.50, 2.00]}
    ):
        raise GapReportError("missing-iteration-report-policy")
    return {
        "policy": dict(report_policy),
        "predicted": {
            "e2e_duration_ns": predicted_e2e,
            "evidence_grade": "D",
            "generation_stage": "resource-model",
            "method": "serialized-resource-model-with-dispatch-floor-v1",
            "permitted_use": "hypothesis-only",
            "uncertainty_interval_ns": [predicted_e2e * 0.5, predicted_e2e * 2.0],
            "evidence_refs": [cost_ref, backend_ref, contract_ref, schedule_ref],
            "items": predicted_items,
            "residual": {"label": "框架/调度/未归因", "duration_ns": 0.0},
        },
        "observed": {
            "e2e_duration_ns": observed_e2e,
            "evidence_grade": "B",
            "generation_stage": "baseline-measurement",
            "method": "benchmark-median",
            "permitted_use": "iteration-baseline-only",
            "uncertainty_interval_ns": [observed_e2e * 0.85, observed_e2e * 1.15],
            "component_method": "scale-predicted-weights-to-observed-e2e",
            "evidence_refs": [observation_ref],
        },
    }


def _mapping(value: object, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GapReportError(reason)
    return value


def _number(value: object, reason: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(float(value))
        or float(value) < 0
    ):
        raise GapReportError(reason)
    return float(value)


def _refs(value: object) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise GapReportError("invalid-evidence-refs")
    return list(value)


def _items(side: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = side.get("items", [])
    if not isinstance(raw, list):
        raise GapReportError("invalid-side-items")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    accounting_ids: set[str] = set()
    for raw_item in raw:
        item = _mapping(raw_item, "invalid-side-item")
        path = item.get("stable_path")
        if not isinstance(path, str) or not path or path in seen:
            raise GapReportError("invalid-or-duplicate-stable-path")
        seen.add(path)
        if item.get("inclusive") is True:
            raise GapReportError("inclusive-parent-is-navigation-only")
        descendants = item.get("descendants", [])
        if not isinstance(descendants, list) or not all(
            isinstance(child, str) and child for child in descendants
        ):
            raise GapReportError("invalid-exclusive-parent-descendants")
        accounting_id = item.get("accounting_id", path)
        if not isinstance(accounting_id, str) or accounting_id in accounting_ids:
            raise GapReportError("non-mutually-exclusive-items")
        accounting_ids.add(accounting_id)
        accounting_kind = item.get("accounting_kind", "mutually-exclusive-leaf")
        if accounting_kind not in {
            "mutually-exclusive-leaf",
            "exclusive-parent",
        }:
            raise GapReportError("invalid-mutually-exclusive-accounting-kind")
        if accounting_kind == "exclusive-parent" and not descendants:
            raise GapReportError("exclusive-parent-requires-descendants")
        interval = item.get("accounting_interval")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(endpoint, (int, float)) for endpoint in interval)
            or interval[0] < 0
            or interval[1] <= interval[0]
        ):
            raise GapReportError("missing-mutual-exclusivity-proof")
        status = item.get("status", "known")
        duration = item.get("duration_ns")
        if status == "known":
            duration = _number(duration, "invalid-item-duration")
        elif status == "unknown" and duration is None:
            duration = None
        else:
            raise GapReportError("invalid-item-status")
        uncertainty = item.get("standard_uncertainty_ns")
        if uncertainty is not None:
            uncertainty = _number(uncertainty, "invalid-item-uncertainty")
        items.append(
            {
                "stable_path": path,
                "operation_class": item.get("operation_class"),
                "status": status,
                "duration_ns": duration,
                "standard_uncertainty_ns": uncertainty,
                "evidence_quality": item.get(
                    "evidence_quality",
                    "direct-qualified" if status == "known" else "structured-unknown",
                ),
                "evidence_refs": _refs(item.get("evidence_refs", [])),
                "accounting_id": accounting_id,
                "accounting_kind": accounting_kind,
                "descendants": list(descendants),
                "accounting_interval": list(interval),
                **(
                    {"evidence_boundaries": list(item["evidence_boundaries"])}
                    if isinstance(item.get("evidence_boundaries"), list)
                    else {}
                ),
            }
        )
    known_paths = {item["stable_path"] for item in items}
    for item in items:
        if (
            item["accounting_kind"] == "exclusive-parent"
            and known_paths.intersection(item["descendants"])
        ):
            raise GapReportError("exclusive-parent-and-descendant-double-count")
    intervals = sorted(
        (item["accounting_interval"], item["stable_path"]) for item in items
    )
    for (left, _), (right, _) in zip(intervals, intervals[1:]):
        if left[1] > right[0]:
            raise GapReportError("non-mutually-exclusive-items")
    return items


def _select(
    items: list[dict[str, Any]], e2e_ns: float | None, policy: Mapping[str, Any]
) -> dict[str, Any]:
    known = [item for item in items if item["duration_ns"] is not None]
    ranked = sorted(known, key=lambda item: (-item["duration_ns"], item["stable_path"]))
    enriched = []
    for rank, item in enumerate(ranked, start=1):
        enriched.append(
            {
                **item,
                "rank": rank,
                "share_of_e2e": item["duration_ns"] / e2e_ns if e2e_ns else None,
            }
        )
    top_k = policy.get("top_k")
    threshold = policy.get("mandatory_share_of_e2e")
    if not isinstance(top_k, int) or top_k <= 0 or top_k != 10:
        raise GapReportError("invalid-top-k-policy")
    threshold = _number(threshold, "invalid-share-policy")
    if threshold != 0.10:
        raise GapReportError("invalid-share-policy")
    top10 = enriched[:top_k]
    top_paths = {item["stable_path"] for item in top10}
    mandatory = [
        item
        for item in enriched
        if e2e_ns is not None and item["duration_ns"] >= e2e_ns * threshold
    ]
    mandatory_paths = {item["stable_path"] for item in mandatory}
    selected = [
        {
            **item,
            "selection_reasons": [
                reason
                for reason, applies in (
                    ("top10", item["stable_path"] in top_paths),
                    ("at-least-10%-of-e2e", item["stable_path"] in mandatory_paths),
                )
                if applies
            ],
        }
        for item in enriched
        if item["stable_path"] in top_paths | mandatory_paths
    ]
    return {
        "all_items": enriched,
        "unknown_items": [item for item in items if item["duration_ns"] is None],
        "top10": top10,
        "mandatory": mandatory,
        "selected": selected,
    }


def _side(value: object, policy: Mapping[str, Any], *, predicted: bool) -> dict[str, Any]:
    side = _mapping(value, "invalid-report-side")
    status = side.get("status")
    available = status in {"known", "available"}
    e2e = _number(side.get("e2e_duration_ns"), "invalid-side-e2e") if available else None
    items = _items(side)
    selected = _select(items, e2e, policy)
    all_attributed = sum(item["duration_ns"] for item in selected["all_items"])
    selected_ns = sum(item["duration_ns"] for item in selected["selected"])
    unattributed_value = side.get("unattributed_ns")
    overlap_value = side.get("overlap_ns")
    unattributed = (
        _number(unattributed_value, "invalid-unattributed")
        if available and unattributed_value is not None
        else None
    )
    overlap = (
        _number(overlap_value, "invalid-overlap")
        if available and overlap_value is not None
        else None
    )
    accounted = (
        all_attributed + unattributed - overlap
        if available and unattributed is not None and overlap is not None
        else None
    )
    residual = e2e - accounted if e2e is not None and accounted is not None else None
    reconciled = residual is not None and abs(residual) <= max(1e-6, e2e * 1e-12)
    reconciliation = {
        "status": "reconciled" if reconciled else "unknown",
        "e2e_ns": e2e,
        "selected_ns": selected_ns if available else None,
        "all_attributed_ns": all_attributed if available else None,
        "other_ns": max(0.0, all_attributed - selected_ns) if available else None,
        "unattributed_ns": unattributed,
        "overlap_ns": overlap,
        "accounted_e2e_ns": accounted,
        "residual_ns": 0.0 if reconciled else residual,
    }
    uncertainty = side.get("standard_uncertainty_ns")
    return {
        "status": status,
        "available": available and reconciled,
        "identity": dict(_mapping(side.get("identity"), "missing-side-identity")),
        "e2e_duration_ns": e2e,
        "standard_uncertainty_ns": (
            _number(uncertainty, "invalid-side-uncertainty")
            if uncertainty is not None
            else None
        ),
        "bound_kind": side.get("bound_kind") if predicted else None,
        "accounting": side.get("accounting"),
        "evidence_refs": _refs(side.get("evidence_refs", [])),
        **selected,
        "reconciliation": reconciliation,
        **(
            {
                "reason_code": side.get("reason_code"),
                "evidence_boundaries": list(side.get("evidence_boundaries", [])),
                "required_next_measurement": side.get("required_next_measurement"),
            }
            if not available
            else {}
        ),
    }


CLASSIFICATIONS = {
    "capability-model",
    "implementation-headroom",
    "materialization-layout",
    "scheduling-integration",
    "instrumentation",
    "noise",
}

EVIDENCE_GRADES = {"A", "B", "C", "D"}
GENERATION_STAGES = {
    "resource-model",
    "implementation-prediction",
    "operator-frontier",
    "schedule-composition",
    "baseline-measurement",
    "diagnostic-attribution",
    "independent-holdout",
}

GRADE_PERMITTED_USE = {
    "A": "acceptance-and-calibration",
    "B": "iteration-baseline-only",
    "C": "hypothesis-only",
    "D": "hypothesis-only",
}


def _uncertainty_interval(
    value: object, *, duration_ns: float, grade: str
) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
    ):
        raise GapReportError("invalid-report-value-uncertainty")
    lower = _number(value[0], "invalid-report-value-uncertainty")
    upper = _number(value[1], "invalid-report-value-uncertainty")
    if lower > duration_ns or upper < duration_ns:
        raise GapReportError("invalid-report-value-uncertainty")
    minimum = {
        "A": (1.0, 1.0),
        "B": (0.85, 1.15),
        "C": (0.70, 1.30),
        "D": (0.50, 2.00),
    }[grade]
    if lower > duration_ns * minimum[0] or upper < duration_ns * minimum[1]:
        raise GapReportError("report-value-uncertainty-too-narrow")
    return [lower, upper]


def _report_value_item(value: object) -> dict[str, Any]:
    item = _mapping(value, "invalid-report-value-item")
    path = item.get("stable_path")
    grade = item.get("evidence_grade")
    stage = item.get("generation_stage")
    method = item.get("method")
    if (
        not isinstance(path, str)
        or not path
        or grade not in EVIDENCE_GRADES
        or stage not in GENERATION_STAGES
        or not isinstance(method, str)
        or not method
    ):
        raise GapReportError("invalid-report-value-item")
    duration = _number(item.get("duration_ns"), "invalid-report-value-item")
    permitted_use = item.get("permitted_use", GRADE_PERMITTED_USE[str(grade)])
    if permitted_use != GRADE_PERMITTED_USE[str(grade)]:
        raise GapReportError("report-value-grade-permitted-use-mismatch")
    interval = _uncertainty_interval(
        item.get("uncertainty_interval_ns"), duration_ns=duration, grade=str(grade)
    )
    accounting_interval = item.get("accounting_interval")
    if (
        not isinstance(accounting_interval, list)
        or len(accounting_interval) != 2
        or not all(isinstance(endpoint, (int, float)) for endpoint in accounting_interval)
        or accounting_interval[0] < 0
        or accounting_interval[1] <= accounting_interval[0]
    ):
        raise GapReportError("invalid-report-value-accounting")
    return {
        "stable_path": path,
        "operation_class": item.get("operation_class"),
        "status": "known",
        "duration_ns": duration,
        "standard_uncertainty_ns": (interval[1] - interval[0]) / 2.0,
        "uncertainty_interval_ns": interval,
        "evidence_grade": grade,
        "generation_stage": stage,
        "method": method,
        "permitted_use": permitted_use,
        "evidence_quality": f"grade-{str(grade).lower()}",
        "evidence_refs": _refs(item.get("evidence_refs", [])),
        "accounting_interval": list(accounting_interval),
    }


def _tiered_side(
    value: object,
    policy: Mapping[str, Any],
    *,
    derived_from: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    side = _mapping(value, "invalid-tiered-report-side")
    grade = side.get("evidence_grade")
    stage = side.get("generation_stage")
    method = side.get("method")
    if (
        grade not in EVIDENCE_GRADES
        or stage not in GENERATION_STAGES
        or not isinstance(method, str)
        or not method
    ):
        raise GapReportError("invalid-tiered-report-side")
    e2e = _number(side.get("e2e_duration_ns"), "invalid-tiered-report-e2e")
    permitted_use = side.get("permitted_use", GRADE_PERMITTED_USE[str(grade)])
    if permitted_use != GRADE_PERMITTED_USE[str(grade)]:
        raise GapReportError("report-value-grade-permitted-use-mismatch")
    interval = _uncertainty_interval(
        side.get("uncertainty_interval_ns"), duration_ns=e2e, grade=str(grade)
    )
    if derived_from is None:
        raw_items = side.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise GapReportError("missing-tiered-report-items")
        items = [_report_value_item(item) for item in raw_items]
    else:
        if side.get("component_method") != "scale-predicted-weights-to-observed-e2e":
            raise GapReportError("unsupported-observation-component-method")
        source_items = derived_from["all_items"]
        source_e2e = _number(
            derived_from.get("e2e_duration_ns"), "invalid-tiered-report-e2e"
        )
        side_refs = _refs(side.get("evidence_refs", []))
        items = []
        for source in source_items:
            duration = e2e * source["duration_ns"] / source_e2e
            items.append(
                {
                    **source,
                    "duration_ns": duration,
                    "standard_uncertainty_ns": duration * 0.75,
                    "uncertainty_interval_ns": [duration * 0.5, duration * 2.0],
                    "evidence_grade": "D",
                    "generation_stage": "diagnostic-attribution",
                    "method": "scale-predicted-weights-to-observed-e2e",
                    "permitted_use": "hypothesis-only",
                    "evidence_quality": "grade-d",
                    "evidence_refs": sorted(
                        set(source.get("evidence_refs", []) + side_refs)
                    ),
                }
            )
        if items:
            observed_sum = fsum(item["duration_ns"] for item in items)
            corrected_duration = items[-1]["duration_ns"] + (e2e - observed_sum)
            items[-1]["duration_ns"] = corrected_duration
            items[-1]["standard_uncertainty_ns"] = corrected_duration * 0.75
            items[-1]["uncertainty_interval_ns"] = [
                corrected_duration * 0.5,
                corrected_duration * 2.0,
            ]
    paths = [item["stable_path"] for item in items]
    if len(paths) != len(set(paths)):
        raise GapReportError("duplicate-tiered-report-path")
    selected = _select(items, e2e, policy)
    attributed = fsum(item["duration_ns"] for item in selected["all_items"])
    residual_value = side.get("residual", {"label": "框架/调度/未归因", "duration_ns": 0.0})
    residual = _mapping(residual_value, "invalid-tiered-report-residual")
    residual_ns = _number(
        residual.get("duration_ns"), "invalid-tiered-report-residual"
    )
    reconciled = fsum((attributed, residual_ns)) == e2e
    if not reconciled:
        raise GapReportError("tiered-report-values-do-not-reconcile")
    return {
        "e2e_duration_ns": e2e,
        "evidence_grade": grade,
        "generation_stage": stage,
        "method": method,
        "permitted_use": permitted_use,
        "uncertainty_interval_ns": interval,
        "evidence_refs": _refs(side.get("evidence_refs", [])),
        **selected,
        "reconciliation": {
            "status": "reconciled",
            "e2e_ns": e2e,
            "all_components_ns": attributed,
            "residual_label": residual.get("label"),
            "residual_ns": residual_ns,
            "overlap_ns": 0.0,
            "share_total": (attributed + residual_ns) / e2e,
        },
    }


def _compose_tiered_report_values(
    document: Mapping[str, object], policy: Mapping[str, Any]
) -> dict[str, Any]:
    iteration = _mapping(document.get("iteration_report"), "missing-iteration-report")
    iteration_policy = _mapping(
        iteration.get("policy"), "missing-iteration-report-policy"
    )
    if (
        iteration_policy.get("policy_id") != "tiered-report-values-v1"
        or iteration_policy.get("version") != "1"
    ):
        raise GapReportError("unsupported-iteration-report-policy")
    predicted = _tiered_side(iteration.get("predicted"), policy)
    observed = _tiered_side(
        iteration.get("observed"), policy, derived_from=predicted
    )
    predicted_by_path = {
        item["stable_path"]: item for item in predicted["all_items"]
    }
    observed_by_path = {
        item["stable_path"]: item for item in observed["all_items"]
    }
    selected_paths = {
        item["stable_path"] for item in predicted["selected"]
    } | {item["stable_path"] for item in observed["selected"]}
    rows = []
    for path in sorted(selected_paths):
        p = predicted_by_path[path]
        o = observed_by_path[path]
        rows.append(
            {
                "stable_path": path,
                "operation_class": p.get("operation_class") or o.get("operation_class"),
                "predicted_time_ns": p["duration_ns"],
                "observed_time_ns": o["duration_ns"],
                "absolute_gap_ns": abs(o["duration_ns"] - p["duration_ns"]),
                "ratio": o["duration_ns"] / p["duration_ns"]
                if p["duration_ns"]
                else 0.0,
                "predicted_share_of_e2e": p["share_of_e2e"],
                "observed_share_of_e2e": o["share_of_e2e"],
                "predicted_rank": p["rank"],
                "observed_rank": o["rank"],
                "predicted_evidence_grade": p["evidence_grade"],
                "observed_evidence_grade": o["evidence_grade"],
                "predicted_generation_stage": p["generation_stage"],
                "observed_generation_stage": o["generation_stage"],
                "predicted_uncertainty_interval_ns": p["uncertainty_interval_ns"],
                "observed_uncertainty_interval_ns": o["uncertainty_interval_ns"],
                "predicted_method": p["method"],
                "observed_method": o["method"],
                "permitted_use": "hypothesis-only",
                "predicted_evidence_refs": p["evidence_refs"],
                "observed_evidence_refs": o["evidence_refs"],
            }
        )
    p_e2e = predicted["e2e_duration_ns"]
    o_e2e = observed["e2e_duration_ns"]
    comparison_kind = (
        "acceptance-error"
        if predicted["evidence_grade"] == observed["evidence_grade"] == "A"
        else "exploratory-gap"
    )
    metrics = {
        "comparison_kind": comparison_kind,
        "e2e_absolute_gap_ns": abs(o_e2e - p_e2e),
        "e2e_ratio": o_e2e / p_e2e if p_e2e else 0.0,
        "relative_gap": abs(o_e2e - p_e2e) / o_e2e if o_e2e else 0.0,
        "permitted_use": (
            "acceptance-and-calibration"
            if comparison_kind == "acceptance-error"
            else "hypothesis-only"
        ),
    }
    return {
        "report_values": {"predicted": predicted, "observed": observed},
        "iteration_gap_table": rows,
        "iteration_metrics": metrics,
        "iteration_status": "numeric-report-values-available",
        "iteration_provenance": {
            "source_bundles": document.get("source_bundles", []),
            "derivation": document.get("iteration_report_derivation"),
        },
    }


def _classification(
    row: Mapping[str, Any], evidence_by_path: Mapping[str, Any]
) -> dict[str, Any] | None:
    evidence = evidence_by_path.get(row["stable_path"])
    if not isinstance(evidence, Mapping):
        return None
    classification = evidence.get("classification")
    refs = evidence.get("evidence_refs")
    if classification not in CLASSIFICATIONS or not isinstance(refs, list) or not refs:
        return None
    return {
        "classification": classification,
        "classification_reason_code": evidence.get("reason_code"),
        "classification_evidence_refs": _refs(refs),
    }


def compose_gap_report(document: Mapping[str, object]) -> dict[str, Any]:
    """Compose one report without converting missing evidence into zero."""

    tiered = document.get("schema") == TIERED_INPUT_SCHEMA
    if document.get("schema") not in {INPUT_SCHEMA, TIERED_INPUT_SCHEMA}:
        raise GapReportError("unsupported-gap-report-input")
    identity = dict(_mapping(document.get("identity"), "invalid-report-identity"))
    policy = _mapping(document.get("policy"), "invalid-report-policy")
    predicted = _side(document.get("predicted"), policy, predicted=True)
    observed = _side(document.get("observed"), policy, predicted=False)
    if predicted["identity"] != identity or observed["identity"] != identity:
        raise GapReportError("side-identity-mismatch")

    predicted_by_path = {item["stable_path"]: item for item in predicted["all_items"]}
    observed_by_path = {item["stable_path"]: item for item in observed["all_items"]}
    selected_paths = {
        item["stable_path"] for item in predicted["selected"]
    } | {item["stable_path"] for item in observed["selected"]}
    rows = []
    for path in sorted(selected_paths):
        p = predicted_by_path.get(path)
        o = observed_by_path.get(path)
        p_ns = p["duration_ns"] if p else None
        o_ns = o["duration_ns"] if o else None
        gap = abs(o_ns - p_ns) if p_ns is not None and o_ns is not None else None
        p_uncertainty = p.get("standard_uncertainty_ns") if p else None
        o_uncertainty = o.get("standard_uncertainty_ns") if o else None
        combined = (
            sqrt(p_uncertainty**2 + o_uncertainty**2)
            if p_uncertainty is not None and o_uncertainty is not None
            else None
        )
        rows.append(
            {
                "stable_path": path,
                "operation_class": (o or p or {}).get("operation_class"),
                "predicted_time_ns": p_ns,
                "observed_time_ns": o_ns,
                "absolute_gap_ns": gap,
                "ratio": o_ns / p_ns if p_ns not in {None, 0} and o_ns is not None else None,
                "predicted_rank": p.get("rank") if p else None,
                "observed_rank": o.get("rank") if o else None,
                "predicted_share_of_e2e": p.get("share_of_e2e") if p else None,
                "observed_share_of_e2e": o.get("share_of_e2e") if o else None,
                "combined_uncertainty_ns": combined,
                "diagnosis_eligible": combined is not None,
                "predicted_evidence_quality": p.get("evidence_quality") if p else "unavailable",
                "observed_evidence_quality": o.get("evidence_quality") if o else "unavailable",
                "predicted_evidence_refs": p.get("evidence_refs", []) if p else [],
                "observed_evidence_refs": o.get("evidence_refs", []) if o else [],
            }
        )
    rows.sort(key=lambda row: (row["absolute_gap_ns"] is None, -(row["absolute_gap_ns"] or 0), row["stable_path"]))

    p_e2e, o_e2e = predicted["e2e_duration_ns"], observed["e2e_duration_ns"]
    p_u, o_u = predicted["standard_uncertainty_ns"], observed["standard_uncertainty_ns"]
    point_prediction = predicted.get("bound_kind") == "point-prediction"
    comparable = (
        predicted["available"]
        and observed["available"]
        and p_e2e is not None
        and o_e2e is not None
        and p_u is not None
        and o_u is not None
        and point_prediction
    )
    e2e_gap = abs(o_e2e - p_e2e) if comparable else None
    combined_e2e = sqrt(p_u**2 + o_u**2) if comparable else None
    metrics = {
        "e2e_absolute_gap_ns": e2e_gap,
        "e2e_ratio": o_e2e / p_e2e if comparable and p_e2e else None,
        "combined_uncertainty_ns": combined_e2e,
        "frontier_efficiency": p_e2e / o_e2e if comparable and o_e2e else None,
        "relative_prediction_error": e2e_gap / o_e2e if comparable and o_e2e else None,
        "applicability": "comparable-point-prediction" if comparable else "unavailable-non-point-or-missing-side",
    }

    diagnosis_policy = _mapping(policy.get("deep_diagnosis"), "invalid-diagnosis-policy")
    minimum_gap = _number(diagnosis_policy.get("minimum_absolute_gap_ns"), "invalid-diagnosis-policy")
    minimum_relative = _number(diagnosis_policy.get("minimum_relative_gap"), "invalid-diagnosis-policy")
    triggered = []
    evidence_by_path = _mapping(
        document.get("diagnostic_evidence", {}), "invalid-diagnostic-evidence"
    )
    if comparable:
        for row in rows:
            gap = row["absolute_gap_ns"]
            uncertainty = row["combined_uncertainty_ns"]
            observed_ns = row["observed_time_ns"]
            relative = gap / observed_ns if gap is not None and observed_ns else None
            if (
                gap is not None
                and uncertainty is not None
                and gap > uncertainty
                and gap > minimum_gap
                and relative is not None
                and relative > minimum_relative
            ):
                classification = _classification(row, evidence_by_path)
                if classification is not None:
                    triggered.append({**row, **classification})
    material_rows = []
    if comparable:
        for row in rows:
            gap = row["absolute_gap_ns"]
            uncertainty = row["combined_uncertainty_ns"]
            observed_ns = row["observed_time_ns"]
            relative = gap / observed_ns if gap is not None and observed_ns else None
            if (
                gap is not None and uncertainty is not None and gap > uncertainty
                and gap > minimum_gap and relative is not None
                and relative > minimum_relative
            ):
                material_rows.append(row)
    diagnosis = {
        "status": "evaluated" if comparable else "unavailable",
        "policy": dict(policy),
        "triggered": triggered,
        "reason_code": None if comparable else "comparison-not-applicable",
    }
    if material_rows:
        largest = max(material_rows, key=lambda row: row["absolute_gap_ns"])
        scopes = document.get("scopes", [])
        if not isinstance(scopes, list):
            raise GapReportError("invalid-navigation-scopes")
        navigation = next(
            (
                scope
                for scope in scopes
                if isinstance(scope, Mapping)
                and scope.get("kind") == "inclusive-navigation"
                and scope.get("children_accounting") == "non-overlapping"
                and largest["stable_path"] in scope.get("children", [])
            ),
            None,
        )
        classification = _classification(largest, evidence_by_path)
        drilldown = {
            "kind": "actionable-operation" if classification else "evidence-boundary",
            "stable_path": largest["stable_path"],
            "classification": classification.get("classification") if classification else None,
            "evidence_boundary": (
                None if classification else "diagnostic-classification-evidence-missing"
            ),
            "navigation_scope": navigation.get("stable_path") if navigation else None,
            "non_overlapping_children": list(navigation.get("children", [])) if navigation else [],
            "evidence_refs": sorted(set(largest["predicted_evidence_refs"] + largest["observed_evidence_refs"])),
        }
    elif not predicted["available"] or not observed["available"]:
        drilldown = {
            "kind": "evidence-boundary",
            "stable_path": None,
            "evidence_boundaries": {
                "predicted": predicted.get("evidence_boundaries", []),
                "observed": observed.get("evidence_boundaries", []),
            },
            "required_next_measurement": {
                "predicted": predicted.get("required_next_measurement"),
                "observed": observed.get("required_next_measurement"),
            },
        }
    else:
        drilldown = {"kind": "none", "stable_path": None, "evidence_boundaries": []}
    result = {
        "schema": TIERED_RESULT_SCHEMA if tiered else RESULT_SCHEMA,
        "status": "complete" if comparable else "structured-unknown",
        "identity": identity,
        "visibility_rule": {
            "top_k": 10,
            "mandatory_share_of_e2e": 0.10,
            "selection": "independent-per-side-then-exact-stable-path-union",
        },
        "predicted": predicted,
        "observed": observed,
        "gap_table": rows,
        "metrics": metrics,
        "diagnosis": diagnosis,
        "drilldown": drilldown,
        "derivation": {"input_sha256": _digest(_json_bytes(document))},
    }
    if tiered:
        result.update(_compose_tiered_report_values(document, policy))
    return result


def _fmt(value: object) -> str:
    return "unavailable" if value is None else f"{float(value):.3f}"


def render_gap_report_html(report: Mapping[str, Any]) -> str:
    """Project the machine report verbatim into a human-readable report."""

    if report.get("schema") == TIERED_RESULT_SCHEMA:
        return _render_tiered_gap_report_html(report)

    rows = "".join(
        "<tr>"
        f"<td><code>{escape(str(row['stable_path']))}</code></td>"
        f"<td>{_fmt(row['predicted_time_ns'])}</td><td>{_fmt(row['observed_time_ns'])}</td>"
        f"<td>{_fmt(row['absolute_gap_ns'])}</td><td>{_fmt(row['ratio'])}</td>"
        f"<td>{escape(str(row['predicted_evidence_quality']))}</td>"
        f"<td>{escape(str(row['observed_evidence_quality']))}</td>"
        "</tr>"
        for row in report["gap_table"]
    )
    boundary_value = report["drilldown"].get("evidence_boundaries", [])
    if isinstance(boundary_value, Mapping):
        boundaries = "; ".join(
            f"{side}: {', '.join(str(item) for item in values) or 'none'}"
            for side, values in boundary_value.items()
        )
    else:
        boundaries = ", ".join(str(item) for item in boundary_value) or "none"
    payload = json.dumps(canonical_data(report), ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>E2E prediction-observation gap report</title></head>
<body><h1>E2E prediction-observation gap report</h1>
<p>Status: <strong>{escape(str(report['status']))}</strong>. Evidence boundary: {escape(boundaries)}.</p>
<p>E2E absolute gap (ns): {_fmt(report['metrics']['e2e_absolute_gap_ns'])}; ratio: {_fmt(report['metrics']['e2e_ratio'])}; combined uncertainty (ns): {_fmt(report['metrics']['combined_uncertainty_ns'])}; Frontier efficiency: {_fmt(report['metrics']['frontier_efficiency'])}; relative prediction error: {_fmt(report['metrics']['relative_prediction_error'])}.</p>
<table><thead><tr><th>Stable Path</th><th>Predicted ns</th><th>Observed ns</th><th>Absolute gap ns</th><th>Ratio</th><th>Predicted evidence</th><th>Observed evidence</th></tr></thead><tbody>{rows}</tbody></table>
<script type=\"application/json\" id=\"groundupscale-gap-report\">{payload}</script></body></html>\n"""


def _display_ms(value_ns: object) -> str:
    return f"{float(value_ns) / 1_000_000:.3g} ms"


def _display_percent(value: object) -> str:
    return f"{float(value) * 100:.1f}%"


OPERATION_NAMES_ZH = {
    "MatMul": "矩阵乘",
    "RMSNorm": "均方根归一化",
    "Softmax": "Softmax 归一化",
    "Add": "逐元素加法",
    "Mul": "逐元素乘法",
    "SiLU": "SiLU 激活",
    "View": "视图变换",
    "Transpose": "转置",
}

STAGE_NAMES_ZH = {
    "qualified-measurement": "资格化实测",
    "baseline-measurement": "基线实测",
    "diagnostic-attribution": "诊断归因",
    "resource-model": "资源模型",
}

METHOD_NAMES_ZH = {
    "serialized-resource-model-with-dispatch-floor-v1": (
        "串行资源模型 + 15 µs 派发下限"
    ),
    "benchmark-median": "独立基线测量中位数",
    "scale-predicted-weights-to-observed-e2e": (
        "按预测组件权重分配到实测 E2E"
    ),
}


def _operation_name_zh(value: object) -> str:
    name = str(value) if value is not None else "未分类"
    return OPERATION_NAMES_ZH.get(name, name)


def _stage_name_zh(value: object) -> str:
    name = str(value)
    return STAGE_NAMES_ZH.get(name, name)


def _method_name_zh(value: object) -> str:
    name = str(value)
    return METHOD_NAMES_ZH.get(name, name)


def _permitted_use_zh(value: object) -> str:
    return {
        "hypothesis-only": "仅用于优化假设",
        "iteration-baseline-only": "仅用于迭代基线",
        "acceptance-and-calibration": "可用于验收与校准",
    }.get(str(value), str(value))


def _render_refs(value: object) -> str:
    refs = _refs(value)
    body = "<br>".join(f"<code>{escape(ref)}</code>" for ref in refs)
    return f"<details><summary>{len(refs)} 条</summary>{body}</details>"


def _display_interval(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2:
        raise GapReportError("invalid-report-value-uncertainty")
    return f"[{_display_ms(value[0])}, {_display_ms(value[1])}]"


def _module_label(stable_path: object) -> str:
    path = str(stable_path)
    layer = None
    if "/layer_0/" in path:
        layer = "第1层"
    elif "/layer_1/" in path:
        layer = "第2层"
    if layer is None:
        return "其他组件"
    if "/attention/" in path:
        return f"{layer} / 注意力"
    if "/mlp/" in path:
        return f"{layer} / 前馈网络"
    if "norm" in path:
        return f"{layer} / 归一化"
    return f"{layer} / 其他"


def _render_top10_rows(items: object) -> str:
    if not isinstance(items, list):
        raise GapReportError("invalid-tiered-top10")
    return "".join(
        "<tr>"
        f"<td>{int(item['rank'])}</td>"
        f"<td><code>{escape(str(item['stable_path']))}</code></td>"
        f"<td>{escape(_operation_name_zh(item.get('operation_class')))}</td>"
        f"<td>{_display_ms(item['duration_ns'])}</td>"
        f"<td>{_display_percent(item['share_of_e2e'])}</td>"
        f"<td>{_display_interval(item['uncertainty_interval_ns'])}</td>"
        f"<td>{escape(str(item['evidence_grade']))}</td>"
        f"<td>{escape(_stage_name_zh(item['generation_stage']))}</td>"
        f"<td>{escape(_method_name_zh(item['method']))}</td>"
        f"<td>{escape(_permitted_use_zh(item['permitted_use']))}</td>"
        "</tr>"
        for item in items
    )


def _render_module_rows(
    predicted: Mapping[str, Any], observed: Mapping[str, Any]
) -> str:
    totals: dict[str, dict[str, float]] = {}
    for side_name, side in (("predicted", predicted), ("observed", observed)):
        items = side.get("all_items")
        if not isinstance(items, list):
            raise GapReportError("invalid-tiered-report-items")
        for item in items:
            label = _module_label(item["stable_path"])
            totals.setdefault(label, {"predicted": 0.0, "observed": 0.0})[
                side_name
            ] += float(item["duration_ns"])
    return "".join(
        "<tr>"
        f"<td>{escape(label)}</td>"
        f"<td>{_display_ms(values['predicted'])}</td>"
        f"<td>{_display_percent(values['predicted'] / predicted['e2e_duration_ns'])}</td>"
        f"<td>{_display_ms(values['observed'])}</td>"
        f"<td>{_display_percent(values['observed'] / observed['e2e_duration_ns'])}</td>"
        "</tr>"
        for label, values in sorted(totals.items())
    )


def _module_totals(side: Mapping[str, Any]) -> list[tuple[str, float]]:
    totals: dict[str, float] = {}
    items = side.get("all_items")
    if not isinstance(items, list):
        raise GapReportError("invalid-tiered-report-items")
    for item in items:
        label = _module_label(item["stable_path"])
        totals[label] = totals.get(label, 0.0) + float(item["duration_ns"])
    return sorted(totals.items())


def _render_contribution_bar(side: Mapping[str, Any]) -> str:
    segments = "".join(
        f'<span title="{escape(label)}：{_display_percent(value / side["e2e_duration_ns"])}" '
        f'style="width:{value / side["e2e_duration_ns"] * 100:.6f}%"></span>'
        for label, value in _module_totals(side)
    )
    return f'<div class="stacked">{segments}</div>'


def _render_leaf_drilldown(
    predicted: Mapping[str, Any], observed: Mapping[str, Any]
) -> str:
    predicted_items = {
        item["stable_path"]: item for item in predicted.get("all_items", [])
    }
    observed_items = {
        item["stable_path"]: item for item in observed.get("all_items", [])
    }
    if predicted_items.keys() != observed_items.keys():
        raise GapReportError("tiered-report-path-set-mismatch")
    return "".join(
        "<tr>"
        f"<td><code>{escape(path)}</code></td>"
        f"<td>{escape(_operation_name_zh(predicted_items[path].get('operation_class')))}</td>"
        f"<td>{_display_ms(predicted_items[path]['duration_ns'])}</td>"
        f"<td>{_display_percent(predicted_items[path]['share_of_e2e'])}</td>"
        f"<td>{_display_ms(observed_items[path]['duration_ns'])}</td>"
        f"<td>{_display_percent(observed_items[path]['share_of_e2e'])}</td>"
        f"<td>{escape(str(predicted_items[path]['evidence_grade']))}</td>"
        f"<td>{escape(_stage_name_zh(predicted_items[path]['generation_stage']))}</td>"
        f"<td>{_display_interval(predicted_items[path]['uncertainty_interval_ns'])}</td>"
        f"<td>{escape(_method_name_zh(predicted_items[path]['method']))}</td>"
        f"<td>{escape(str(observed_items[path]['evidence_grade']))}</td>"
        f"<td>{escape(_stage_name_zh(observed_items[path]['generation_stage']))}</td>"
        f"<td>{_display_interval(observed_items[path]['uncertainty_interval_ns'])}</td>"
        f"<td>{escape(_method_name_zh(observed_items[path]['method']))}</td>"
        "</tr>"
        for path in sorted(predicted_items)
    )


def _render_provenance(report: Mapping[str, Any]) -> str:
    provenance = _mapping(
        report.get("iteration_provenance"), "missing-iteration-provenance"
    )
    sources = provenance.get("source_bundles")
    if not isinstance(sources, list) or not sources:
        raise GapReportError("missing-iteration-provenance")
    return "".join(
        "<li>"
        f"<code>{escape(str(source.get('run_id')))}</code> "
        f"（{escape(str(source.get('bundle_kind')))}；manifest "
        f"<code>{escape(str(source.get('manifest_sha256')))}</code>）"
        "</li>"
        for source in sources
        if isinstance(source, Mapping)
    )


def _render_tiered_gap_report_html(report: Mapping[str, Any]) -> str:
    values = _mapping(report.get("report_values"), "missing-report-values")
    predicted = _mapping(values.get("predicted"), "missing-predicted-report-values")
    observed = _mapping(values.get("observed"), "missing-observed-report-values")
    metrics = _mapping(report.get("iteration_metrics"), "missing-iteration-metrics")
    rows = report.get("iteration_gap_table")
    if not isinstance(rows, list):
        raise GapReportError("invalid-iteration-gap-table")
    gap_rows = "".join(
        "<tr>"
        f"<td><code>{escape(str(row['stable_path']))}</code></td>"
        f"<td>{escape(_operation_name_zh(row.get('operation_class')))}</td>"
        f"<td>{_display_ms(row['predicted_time_ns'])}</td>"
        f"<td>{_display_percent(row['predicted_share_of_e2e'])}</td>"
        f"<td>{_display_ms(row['observed_time_ns'])}</td>"
        f"<td>{_display_percent(row['observed_share_of_e2e'])}</td>"
        f"<td>{_display_ms(row['absolute_gap_ns'])}</td>"
        f"<td>{float(row['ratio']):.3f}×</td>"
        f"<td>{int(row['predicted_rank'])}</td>"
        f"<td>{int(row['observed_rank'])}</td>"
        f"<td>{escape(str(row['predicted_evidence_grade']))}/{escape(str(row['observed_evidence_grade']))}</td>"
        f"<td>{escape(_stage_name_zh(row['predicted_generation_stage']))}</td>"
        f"<td>{escape(_stage_name_zh(row['observed_generation_stage']))}</td>"
        f"<td>{_display_interval(row['predicted_uncertainty_interval_ns'])}</td>"
        f"<td>{_display_interval(row['observed_uncertainty_interval_ns'])}</td>"
        f"<td>{escape(_method_name_zh(row['predicted_method']))}</td>"
        f"<td>{escape(_method_name_zh(row['observed_method']))}</td>"
        f"<td>{escape(_permitted_use_zh(row['permitted_use']))}</td>"
        f"<td>{_render_refs(row['predicted_evidence_refs'])}{_render_refs(row['observed_evidence_refs'])}</td>"
        "</tr>"
        for row in rows
    )
    payload = json.dumps(
        canonical_data(report), ensure_ascii=False, sort_keys=True
    ).replace("</", "<\\/")
    authority_text = (
        "权威证据已闭合"
        if report.get("status") == "complete"
        else "权威证据未闭合；以下数值按已声明策略降级"
    )
    comparison_text = (
        "验收差异"
        if metrics.get("comparison_kind") == "acceptance-error"
        else "探索性差异"
    )
    authority_boundaries = report.get("drilldown", {}).get(
        "evidence_boundaries", {}
    )
    predicted_boundary_count = 0
    observed_boundary_count = 0
    if isinstance(authority_boundaries, Mapping):
        predicted_boundaries = authority_boundaries.get("predicted", [])
        observed_boundaries = authority_boundaries.get("observed", [])
        if isinstance(predicted_boundaries, list):
            predicted_boundary_count = len(predicted_boundaries)
        if isinstance(observed_boundaries, list):
            observed_boundary_count = len(observed_boundaries)
    predicted_interval = _display_interval(predicted["uncertainty_interval_ns"])
    observed_interval = _display_interval(observed["uncertainty_interval_ns"])
    module_rows = _render_module_rows(predicted, observed)
    identity = _mapping(report.get("identity"), "missing-report-identity")
    leaf_rows = _render_leaf_drilldown(predicted, observed)
    leaf_count = len(predicted.get("all_items", []))
    provenance_rows = _render_provenance(report)
    return f"""<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><title>两层 Transformer 预测—实测迭代报告</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1500px;margin:0 auto;padding:24px;color:#17202a}}.cards{{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:12px}}.card{{border:1px solid #d9e2ec;border-radius:10px;padding:16px;background:#f8fafc}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}th,td{{border-bottom:1px solid #d9e2ec;padding:8px;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}code{{font-size:12px;overflow-wrap:anywhere}}.notice{{padding:12px;border-left:4px solid #f39c12;background:#fff8e7}}.stacked{{height:22px;display:flex;border-radius:6px;overflow:hidden;background:#eef2f7;margin:6px 0 14px}}.stacked span:nth-child(4n+1){{background:#2563eb}}.stacked span:nth-child(4n+2){{background:#16a34a}}.stacked span:nth-child(4n+3){{background:#f59e0b}}.stacked span:nth-child(4n){{background:#9333ea}}@media(max-width:900px){{.cards{{grid-template-columns:1fr 1fr}}table{{display:block;overflow-x:auto}}}}</style></head>
<body><h1>两层 Transformer 预测—实测迭代报告</h1>
<p class=\"notice\">{authority_text}。当前差异口径：<strong>{comparison_text}</strong>。</p>
<h2>运行身份</h2><p>案例：<code>{escape(str(identity['case']))}</code>；Shape：<code>{escape(str(identity['shape']))}</code>；dtype：<code>{escape(str(identity['dtype']))}</code>；候选：<code>{escape(str(identity['candidate_id']))}</code>；Hardware Cohort：<code>{escape(str(identity['hardware_cohort']))}</code>；Completion Boundary：<code>{escape(str(identity['completion_boundary']))}</code>。</p>
<div class=\"cards\"><div class=\"card\"><strong>预测 E2E</strong><br>{_display_ms(predicted['e2e_duration_ns'])}<br>不确定区间 {predicted_interval}<br>等级 {escape(str(predicted['evidence_grade']))}<br>阶段：{escape(_stage_name_zh(predicted['generation_stage']))}<br>方法：{escape(_method_name_zh(predicted['method']))}<br>{escape(_permitted_use_zh(predicted['permitted_use']))}</div>
<div class=\"card\"><strong>实测 E2E</strong><br>{_display_ms(observed['e2e_duration_ns'])}<br>不确定区间 {observed_interval}<br>等级 {escape(str(observed['evidence_grade']))}<br>阶段：{escape(_stage_name_zh(observed['generation_stage']))}<br>方法：{escape(_method_name_zh(observed['method']))}<br>{escape(_permitted_use_zh(observed['permitted_use']))}</div>
<div class=\"card\"><strong>绝对差</strong><br>{_display_ms(metrics['e2e_absolute_gap_ns'])}</div>
<div class=\"card\"><strong>倍率</strong><br>{float(metrics['e2e_ratio']):.3f}×</div></div>
<h2>证据等级与适用范围</h2><p>A：权威验收；B：可复现实测，仅作迭代基线；C：代理推导；D：模型降级估计，仅用于提出优化假设。当前 E2E 为预测 {escape(str(predicted['evidence_grade']))} / 实测 {escape(str(observed['evidence_grade']))}；组件对比均为探索性差异，不用于验收或校准。</p>
<h2>模块汇总</h2><table><thead><tr><th>模块</th><th>预测时间</th><th>预测 E2E 贡献</th><th>实测侧归因时间</th><th>实测 E2E 贡献</th></tr></thead><tbody>{module_rows}</tbody></table>
<h3>预测侧贡献构成</h3>{_render_contribution_bar(predicted)}<h3>实测侧降级归因构成</h3>{_render_contribution_bar(observed)}
<h2>预测侧 TOP10</h2><table><thead><tr><th>排名</th><th>Stable Path</th><th>组件</th><th>时间</th><th>本侧 E2E 占比</th><th>不确定区间</th><th>等级</th><th>生成阶段</th><th>推导方法</th><th>允许用途</th></tr></thead><tbody>{_render_top10_rows(predicted['top10'])}</tbody></table>
<h2>实测侧 TOP10（降级估计）</h2><table><thead><tr><th>排名</th><th>Stable Path</th><th>组件</th><th>时间</th><th>本侧 E2E 占比</th><th>不确定区间</th><th>等级</th><th>生成阶段</th><th>推导方法</th><th>允许用途</th></tr></thead><tbody>{_render_top10_rows(observed['top10'])}</tbody></table>
<h2>预测—实测 TOP10 联合对比</h2><table><thead><tr><th>Stable Path</th><th>组件</th><th>预测时间</th><th>预测占比</th><th>实测侧降级估计</th><th>实测侧占比</th><th>绝对差</th><th>倍率</th><th>预测排名</th><th>实测排名</th><th>证据等级</th><th>预测阶段</th><th>实测阶段</th><th>预测不确定区间</th><th>实测不确定区间</th><th>预测方法</th><th>实测归因方法</th><th>允许用途</th><th>来源</th></tr></thead><tbody>{gap_rows}</tbody></table>
<h2>平账与证据边界</h2><p>预测组件 + <strong>框架/调度/未归因</strong> = 预测 E2E（{_display_percent(predicted['reconciliation']['share_total'])}）；实测侧归因组件 + <strong>框架/调度/未归因</strong> = 实测 E2E（{_display_percent(observed['reconciliation']['share_total'])}）。当前仍有预测侧 {predicted_boundary_count} 项、实测侧 {observed_boundary_count} 项权威证据待升级；本报告已按固定降级策略提供完整数值，详细边界保存在同源 JSON 中。</p>
<p>预测 residual：{_display_ms(predicted['reconciliation']['residual_ns'])}，overlap：{_display_ms(predicted['reconciliation']['overlap_ns'])}；实测侧 residual：{_display_ms(observed['reconciliation']['residual_ns'])}，overlap：{_display_ms(observed['reconciliation']['overlap_ns'])}。</p>
<h2>下一轮建议</h2><ol><li>优先验证两层前馈网络的 gate/up/down projection；它们是当前 D 级模型中贡献最大的组件。</li><li>采集同一身份的 Ascend device timeline 与 profiling overhead 独立 holdout，把实测侧组件从 D 级降级归因升级为 B/A 级直接证据。</li><li>补齐 #42–#46 指定的候选、phase 与 layout audit 后重建 #48 schedule，再把本页探索性差异升级为可验收差异。</li></ol>
<h2>{leaf_count} 个叶子下钻</h2><details><summary>展开全部 {leaf_count} 个 indexed Stable Path</summary><table><thead><tr><th>Stable Path</th><th>组件</th><th>预测时间</th><th>预测贡献</th><th>实测侧降级估计</th><th>实测贡献</th><th>预测等级</th><th>预测阶段</th><th>预测区间</th><th>预测方法</th><th>实测等级</th><th>实测阶段</th><th>实测区间</th><th>实测方法</th></tr></thead><tbody>{leaf_rows}</tbody></table></details>
<h2>来源与机器产物</h2><ul>{provenance_rows}</ul><p><a href=\"../comparison/e2e-gap-report.json\">同源机器 JSON</a> · <a href=\"../comparison/e2e-components.csv\">全部 {leaf_count} 组件 CSV</a> · <a href=\"../resolved/e2e-gap-report-input.json\">锁定输入与 Derivation Record</a></p>
<script type=\"application/json\" id=\"groundupscale-gap-report\">{payload}</script></body></html>\n"""


def render_gap_report_csv(report: Mapping[str, Any]) -> str:
    """Project tiered component rows into a stable machine-readable CSV."""

    if report.get("schema") != TIERED_RESULT_SCHEMA:
        raise GapReportError("csv-requires-tiered-report-values")
    report_values = _mapping(report.get("report_values"), "missing-report-values")
    predicted = _mapping(
        report_values.get("predicted"), "missing-predicted-report-values"
    )
    observed = _mapping(
        report_values.get("observed"), "missing-observed-report-values"
    )
    predicted_items = predicted.get("all_items")
    observed_items = observed.get("all_items")
    if not isinstance(predicted_items, list) or not isinstance(observed_items, list):
        raise GapReportError("invalid-tiered-report-items")
    predicted_by_path = {item["stable_path"]: item for item in predicted_items}
    observed_by_path = {item["stable_path"]: item for item in observed_items}
    if predicted_by_path.keys() != observed_by_path.keys():
        raise GapReportError("tiered-report-path-set-mismatch")
    selected_paths = {
        row["stable_path"] for row in report.get("iteration_gap_table", [])
    }
    fields = [
        "stable_path",
        "operation_class",
        "selected_in_top10_union",
        "predicted_time_ns",
        "predicted_share_of_e2e",
        "predicted_rank",
        "predicted_evidence_grade",
        "predicted_generation_stage",
        "predicted_uncertainty_interval_ns",
        "predicted_method",
        "observed_time_ns",
        "observed_share_of_e2e",
        "observed_rank",
        "observed_evidence_grade",
        "observed_generation_stage",
        "observed_uncertainty_interval_ns",
        "observed_method",
        "absolute_gap_ns",
        "ratio",
        "permitted_use",
        "predicted_evidence_refs",
        "observed_evidence_refs",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for path in sorted(predicted_by_path):
        predicted_item = predicted_by_path[path]
        observed_item = observed_by_path[path]
        predicted_ns = float(predicted_item["duration_ns"])
        observed_ns = float(observed_item["duration_ns"])
        record = {
            "stable_path": path,
            "operation_class": predicted_item.get("operation_class"),
            "selected_in_top10_union": path in selected_paths,
            "predicted_time_ns": predicted_ns,
            "predicted_share_of_e2e": predicted_item["share_of_e2e"],
            "predicted_rank": predicted_item["rank"],
            "predicted_evidence_grade": predicted_item["evidence_grade"],
            "predicted_generation_stage": predicted_item["generation_stage"],
            "predicted_uncertainty_interval_ns": predicted_item[
                "uncertainty_interval_ns"
            ],
            "predicted_method": predicted_item["method"],
            "observed_time_ns": observed_ns,
            "observed_share_of_e2e": observed_item["share_of_e2e"],
            "observed_rank": observed_item["rank"],
            "observed_evidence_grade": observed_item["evidence_grade"],
            "observed_generation_stage": observed_item["generation_stage"],
            "observed_uncertainty_interval_ns": observed_item[
                "uncertainty_interval_ns"
            ],
            "observed_method": observed_item["method"],
            "absolute_gap_ns": abs(observed_ns - predicted_ns),
            "ratio": observed_ns / predicted_ns if predicted_ns else 0.0,
            "permitted_use": "hypothesis-only",
            "predicted_evidence_refs": predicted_item["evidence_refs"],
            "observed_evidence_refs": observed_item["evidence_refs"],
        }
        for field in (
            "predicted_uncertainty_interval_ns",
            "observed_uncertainty_interval_ns",
            "predicted_evidence_refs",
            "observed_evidence_refs",
        ):
            record[field] = json.dumps(record[field], separators=(",", ":"))
        writer.writerow(record)
    return output.getvalue()


def write_gap_report_bundle(
    artifact_store: str | Path, *, run_id: str, document: Mapping[str, object]
) -> Path:
    """Write an immutable report bundle whose two projections share one input."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise GapReportError("unsafe-run-id")
    if document.get("schema") == TIERED_INPUT_SCHEMA:
        if (
            document.get("iteration_report_derivation") is None
            or not isinstance(document.get("source_bundles"), list)
            or not document.get("source_bundles")
            or not isinstance(document.get("supersedes"), list)
            or not document.get("supersedes")
        ):
            raise GapReportError("tiered-report-replay-contract-required")
        repository_root = _repository_root(Path.cwd())
        replayed_iteration = derive_tiered_iteration_report(
            document, repository_root
        )
        if document.get("iteration_report") != replayed_iteration:
            raise GapReportError("tiered-report-source-replay-mismatch")
    report = compose_gap_report(document)
    tiered = report.get("schema") == TIERED_RESULT_SCHEMA
    selected_rows = report["gap_table"]
    sources_value = document.get("source_bundles")
    if selected_rows:
        if (
            not isinstance(sources_value, list)
            or not sources_value
            or not all(
                isinstance(source, Mapping)
                and isinstance(source.get("path"), str)
                and isinstance(source.get("manifest_sha256"), str)
                and source.get("verification_passed") is True
                for source in sources_value
            )
        ):
            raise GapReportError("selected-rows-require-locked-source-bundles")
        source_digests = {
            source.get("run_id"): source.get("manifest_sha256")
            for source in sources_value
            if isinstance(source, Mapping)
        }
        for row in selected_rows:
            for side in ("predicted", "observed"):
                if row[f"{side}_time_ns"] is None:
                    continue
                refs = row[f"{side}_evidence_refs"]
                if not refs or not all(
                    any(
                        ref == f"run://{run_id}@sha256:{digest}"
                        or ref.startswith(
                            f"run://{run_id}@sha256:{digest}#"
                        )
                        for run_id, digest in source_digests.items()
                    )
                    for ref in refs
                ):
                    raise GapReportError("selected-row-direct-source-ref-mismatch")
    root = Path(artifact_store).resolve() / "runs"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_id
    if destination.exists():
        raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=root))
    artifacts = []

    def write(role: str, relative: str, payload: bytes, media_type: str, schema: str, inputs: list[str]) -> None:
        path = temporary / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts.append({
            "role": role, "path": relative, "media_type": media_type,
            "schema": schema, "sha256": _digest(payload), "produced_by": PRODUCER,
            "inputs": inputs,
        })

    try:
        write(
            "e2e-gap-report-input",
            "resolved/e2e-gap-report-input.json",
            _json_bytes(document),
            "application/json",
            TIERED_INPUT_SCHEMA if tiered else INPUT_SCHEMA,
            [],
        )
        write(
            "e2e-gap-report",
            "comparison/e2e-gap-report.json",
            _json_bytes(report),
            "application/json",
            TIERED_RESULT_SCHEMA if tiered else RESULT_SCHEMA,
            ["resolved/e2e-gap-report-input.json"],
        )
        if tiered:
            write(
                "e2e-components-csv",
                "comparison/e2e-components.csv",
                render_gap_report_csv(report).encode("utf-8"),
                "text/csv",
                "groundupscale.dev/e2e-components-csv/v1alpha1",
                ["comparison/e2e-gap-report.json"],
            )
        write(
            "html-report",
            "reports/report.html",
            render_gap_report_html(report).encode(),
            "text/html",
            TIERED_REPORT_SCHEMA if tiered else REPORT_SCHEMA,
            ["comparison/e2e-gap-report.json"],
        )
        manifest = {
            "schema": "groundupscale.dev/run-manifest/v1alpha1",
            "run_id": run_id,
            "bundle_kind": "e2e-gap-report",
            "status": "completed",
            "hardware_cohort": report["identity"].get("hardware_cohort"),
            "producer": PRODUCER,
            "artifacts": artifacts,
            **(
                {"source_bundles": list(document["source_bundles"])}
                if isinstance(document.get("source_bundles"), list)
                and all(
                    isinstance(source, Mapping)
                    and isinstance(source.get("path"), str)
                    and isinstance(source.get("manifest_sha256"), str)
                    for source in document["source_bundles"]
                )
                else {}
            ),
            **(
                {"supersedes": list(document["supersedes"])}
                if isinstance(document.get("supersedes"), list)
                else {}
            ),
        }
        (temporary / "run.manifest.json").write_bytes(_json_bytes(manifest))
        temporary.rename(destination)
    except BaseException:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def main() -> None:
    """Build a bundle from one locked JSON input without touching hardware."""

    import argparse

    parser = argparse.ArgumentParser(prog="python -m groundupscale.gap_report")
    parser.add_argument("input")
    parser.add_argument("--artifact-store", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    document = json.loads(Path(args.input).read_text(encoding="utf-8"))
    destination = write_gap_report_bundle(
        args.artifact_store, run_id=args.run_id, document=document
    )
    print(destination)


if __name__ == "__main__":
    main()


__all__ = [
    "GapReportError", "compose_gap_report", "derive_tiered_iteration_report",
    "render_gap_report_html", "render_gap_report_csv", "write_gap_report_bundle",
]
