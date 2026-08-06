"""Controlled fitting, independent holdout validation, and profile promotion."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

import yaml

from groundupscale.ir import content_fingerprint


class CalibrationError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CalibrationError(f"expected JSON object: {path}")
    return value


def _cost_items(region: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield region
    for item in region.get("items", []):
        if "items" in item:
            yield from _cost_items(item)
        else:
            yield item


def _load_evidence(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    manifest = _read_json(root / "run.manifest.json")
    benchmark = _read_json(root / "observation/raw/benchmark.json")
    memory = _read_json(root / "observation/memory.json")
    cost = _read_json(root / "ir/cost.ir.json")
    if manifest.get("status") != "completed":
        raise CalibrationError(f"run {root} is not completed")
    cases = {case["case_id"]: case for case in benchmark["cases"]}
    if len(cases) != len(benchmark["cases"]):
        raise CalibrationError(f"run {root} contains duplicate case IDs")
    return {
        "root": root,
        "manifest": manifest,
        "benchmark": benchmark,
        "memory": memory,
        "cost": cost,
        "cases": cases,
    }


def _cohort_key(evidence: dict[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
    manifest = evidence["manifest"]
    return (
        manifest["device"],
        manifest["cost_compilation_fingerprint"],
        manifest["hardware_cohort"],
        tuple(sorted(evidence["cases"])),
    )


def _assert_same_cohort(evidence: list[dict[str, Any]]) -> None:
    if not evidence:
        raise CalibrationError("at least one Run Bundle is required")
    expected = _cohort_key(evidence[0])
    for item in evidence[1:]:
        if _cohort_key(item) != expected:
            raise CalibrationError(
                "calibration evidence must share device, CostIR fingerprint, "
                "hardware cohort, and Benchmark Case set"
            )


def _framework_peak(memory: dict[str, Any]) -> int:
    try:
        return int(
            memory["framework_tensor_storage"]["peak_framework_tensor_bytes"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CalibrationError(
            "Run Bundle lacks framework Tensor storage peak observation"
        ) from error


def _base_cost_for_scope(cost: dict[str, Any], stable_path: str) -> dict[str, Any]:
    cost_stable_path = f"cost/{stable_path}"
    matches = [
        item
        for item in _cost_items(cost["root"])
        if item.get("stable_path") in {stable_path, cost_stable_path}
    ]
    if len(matches) != 1:
        raise CalibrationError(
            f"resolved scope {stable_path!r} maps to {len(matches)} CostIR items"
        )
    item = matches[0]
    return {
        "cost_node_id": item["node_id"],
        "stable_path": item["stable_path"],
        "runtime_stable_path": stable_path,
        "operation_or_region": item.get("operation", item.get("kind")),
        "metrics": item["metrics"],
    }


def fit_calibration(
    run_bundles: Iterable[str | Path],
    *,
    maximum_noise: float = 0.03,
) -> dict[str, Any]:
    evidence = [_load_evidence(path) for path in run_bundles]
    _assert_same_cohort(evidence)
    noisy = [
        (item["manifest"]["run_id"], case_id, case["latency"]["iqr_over_median"])
        for item in evidence
        for case_id, case in item["cases"].items()
        if case["latency"]["iqr_over_median"] > maximum_noise
    ]
    if noisy:
        raise CalibrationError(f"noisy fitting evidence is forbidden: {noisy}")

    first = evidence[0]
    case_models: list[dict[str, Any]] = []
    for case_id in sorted(first["cases"]):
        case = first["cases"][case_id]
        fit_medians = [item["cases"][case_id]["latency"]["median_ns"] for item in evidence]
        predicted = float(statistics.median(fit_medians))
        case_models.append(
            {
                "case_id": case_id,
                "stable_path": case["resolved_scope"],
                "prediction_kind": "exact-cohort-case-median",
                "predicted_median_ns": predicted,
                "fit_observed_median_ns": fit_medians,
                "base_cost": _base_cost_for_scope(
                    first["cost"], case["resolved_scope"]
                ),
            }
        )
    memory_observed = [_framework_peak(item["memory"]) for item in evidence]
    base_memory = int(
        first["cost"]["summary"]["parameter_bytes"]
        + first["cost"]["summary"]["buffer_bytes"]
    )
    # The full uncalibrated live-set is recorded in prediction/metrics.json; the
    # fit uses that value when present and never rewrites it.
    prediction = _read_json(first["root"] / "prediction/metrics.json")
    base_peak = int(prediction["live_set"]["predicted_framework_peak_bytes"])
    calibrated_peak = int(statistics.median(memory_observed))
    applicability = {
        "device": first["manifest"]["device"],
        "hardware_cohort": first["manifest"]["hardware_cohort"],
        "compilation_fingerprint": first["manifest"]["compilation_fingerprint"],
        "cost_compilation_fingerprint": first["manifest"][
            "cost_compilation_fingerprint"
        ],
        "shape": first["benchmark"].get("shape", "locked-by-compilation-fingerprint"),
        "seed_policy": "deterministic-reference-v1",
        "benchmark_case_ids": sorted(first["cases"]),
        "torch_num_threads": first["benchmark"]["torch_num_threads"],
        "instrumentation_profile": "benchmark",
    }
    fit_run_ids = [item["manifest"]["run_id"] for item in evidence]
    profile_id = content_fingerprint(
        "groundupscale.calibration/v1alpha1",
        applicability,
        case_models,
        memory_observed,
        fit_run_ids,
    )
    return {
        "apiVersion": "groundupscale.dev/v1alpha1",
        "kind": "CalibrationProfile",
        "metadata": {
            "name": f"{applicability['device']}-m4-fixed-prefill-{profile_id[:8]}",
            "version": "0.1.0-candidate",
            "profile_id": profile_id,
            "status": "candidate",
            "created_at": datetime.now(UTC).isoformat(),
        },
        "spec": {
            "applicability": applicability,
            "fit_evidence": fit_run_ids,
            "maximum_fit_noise": maximum_noise,
            "duration_models": case_models,
            "memory_model": {
                "prediction_kind": "exact-cohort-live-storage-factor",
                "base_state_bytes": base_memory,
                "base_predicted_peak_bytes": base_peak,
                "fit_observed_peak_bytes": memory_observed,
                "calibrated_peak_bytes": calibrated_peak,
                "correction_factor": calibrated_peak / base_peak,
            },
            "governance": {
                "base_cost_ir_immutable": True,
                "out_of_domain_behavior": "reject-and-fall-back-to-uncalibrated",
                "minimum_holdout_runs": 5,
                "maximum_relative_error": 0.05,
                "maximum_holdout_noise": 0.03,
            },
        },
    }


def validate_calibration(
    profile: dict[str, Any],
    run_bundles: Iterable[str | Path],
) -> dict[str, Any]:
    evidence = [_load_evidence(path) for path in run_bundles]
    _assert_same_cohort(evidence)
    applicability = profile["spec"]["applicability"]
    expected = (
        applicability["device"],
        applicability["cost_compilation_fingerprint"],
        applicability["hardware_cohort"],
        tuple(applicability["benchmark_case_ids"]),
    )
    if _cohort_key(evidence[0]) != expected:
        raise CalibrationError("holdout evidence is outside profile applicability")
    fit_ids = set(profile["spec"]["fit_evidence"])
    overlap = [item["manifest"]["run_id"] for item in evidence if item["manifest"]["run_id"] in fit_ids]
    if overlap:
        raise CalibrationError(f"fit/holdout evidence overlap is forbidden: {overlap}")

    maximum_noise = float(profile["spec"]["governance"]["maximum_holdout_noise"])
    maximum_error = float(profile["spec"]["governance"]["maximum_relative_error"])
    minimum_runs = int(profile["spec"]["governance"]["minimum_holdout_runs"])
    models = {model["case_id"]: model for model in profile["spec"]["duration_models"]}
    run_results: list[dict[str, Any]] = []
    for item in evidence:
        noise_failures = [
            {
                "case_id": case_id,
                "iqr_over_median": case["latency"]["iqr_over_median"],
            }
            for case_id, case in item["cases"].items()
            if case["latency"]["iqr_over_median"] > maximum_noise
        ]
        case_results = []
        for case_id, case in sorted(item["cases"].items()):
            predicted = float(models[case_id]["predicted_median_ns"])
            observed = float(case["latency"]["median_ns"])
            error = abs(predicted - observed) / observed
            case_results.append(
                {
                    "case_id": case_id,
                    "predicted_median_ns": predicted,
                    "observed_median_ns": observed,
                    "relative_error": error,
                    "passed": error <= maximum_error,
                }
            )
        predicted_memory = int(profile["spec"]["memory_model"]["calibrated_peak_bytes"])
        observed_memory = _framework_peak(item["memory"])
        memory_error = abs(predicted_memory - observed_memory) / observed_memory
        valid = not noise_failures
        run_results.append(
            {
                "run_id": item["manifest"]["run_id"],
                "valid_for_holdout": valid,
                "noise_failures": noise_failures,
                "case_results": case_results,
                "memory_result": {
                    "predicted_peak_bytes": predicted_memory,
                    "observed_peak_bytes": observed_memory,
                    "relative_error": memory_error,
                    "passed": memory_error <= maximum_error,
                },
                "passed": valid
                and all(result["passed"] for result in case_results)
                and memory_error <= maximum_error,
            }
        )
    valid_results = [result for result in run_results if result["valid_for_holdout"]]
    passed = len(valid_results) >= minimum_runs and all(
        result["passed"] for result in valid_results
    )
    return {
        "schema": "groundupscale.dev/calibration-validation/v1alpha1",
        "profile_id": profile["metadata"]["profile_id"],
        "fit_evidence": list(profile["spec"]["fit_evidence"]),
        "holdout_evidence": [item["manifest"]["run_id"] for item in evidence],
        "minimum_valid_holdout_runs": minimum_runs,
        "valid_holdout_runs": len(valid_results),
        "quarantined_noisy_runs": len(run_results) - len(valid_results),
        "maximum_relative_error": maximum_error,
        "maximum_holdout_noise": maximum_noise,
        "run_results": run_results,
        "passed": passed,
        "validated_at": datetime.now(UTC).isoformat(),
    }


def promote_calibration(
    profile: dict[str, Any], validation: dict[str, Any]
) -> dict[str, Any]:
    if validation.get("profile_id") != profile["metadata"]["profile_id"]:
        raise CalibrationError("validation does not belong to candidate profile")
    if not validation.get("passed"):
        raise CalibrationError("cannot promote a profile that failed holdout validation")
    promoted = deepcopy(profile)
    promoted["metadata"]["status"] = "active"
    promoted["metadata"]["version"] = "0.1.0"
    promoted["metadata"]["promoted_at"] = datetime.now(UTC).isoformat()
    promoted["spec"]["validation"] = {
        "schema": validation["schema"],
        "holdout_evidence": validation["holdout_evidence"],
        "valid_holdout_runs": validation["valid_holdout_runs"],
        "maximum_relative_error": validation["maximum_relative_error"],
        "maximum_holdout_noise": validation["maximum_holdout_noise"],
        "passed": True,
    }
    return promoted


def load_calibration_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CalibrationError(f"expected CalibrationProfile mapping: {path}")
    return value


def write_calibration_yaml(path: str | Path, profile: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(profile, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


__all__ = [
    "CalibrationError",
    "fit_calibration",
    "load_calibration_yaml",
    "promote_calibration",
    "validate_calibration",
    "write_calibration_yaml",
]
