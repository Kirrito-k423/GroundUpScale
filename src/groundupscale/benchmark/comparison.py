"""Prediction-versus-observation comparison with explicit validity semantics."""

from __future__ import annotations

from typing import Any

from groundupscale.ir import HardwareBackendPrediction, content_fingerprint


COMPARISON_SCHEMA = (
    "groundupscale.dev/prediction-observation-comparison/v1alpha1"
)


def _candidate_is_in_scope(candidate_path: str, runtime_scope: str) -> bool:
    normalized = candidate_path.removeprefix("cost/")
    return normalized == runtime_scope or normalized.startswith(runtime_scope + "/")


def build_prediction_observation_comparison(
    *,
    hardware_prediction: HardwareBackendPrediction | None,
    benchmark: dict[str, Any],
    live_set: dict[str, Any],
    tensor_storage_observation: dict[str, Any],
) -> dict[str, Any]:
    """Align base predictions and observations without overstating partial bounds."""

    bounds_by_case = (
        {bound.case_id: bound for bound in hardware_prediction.scope_bounds}
        if hardware_prediction is not None
        else {}
    )
    latency_cases: list[dict[str, Any]] = []
    for case in benchmark["cases"]:
        candidates = (
            tuple(
                candidate
                for candidate in hardware_prediction.candidates
                if _candidate_is_in_scope(
                    candidate.stable_path, case["resolved_scope"]
                )
            )
            if hardware_prediction is not None
            else ()
        )
        scope_bound = bounds_by_case.get(case["case_id"])
        hardware_floor = (
            scope_bound.empirical_hardware_floor_ns
            if scope_bound is not None
            else None
        )
        observed_median = float(case["latency"]["median_ns"])
        distance_from_floor = (
            observed_median - hardware_floor
            if hardware_floor is not None
            else None
        )
        distance_ratio = (
            observed_median / hardware_floor
            if hardware_floor is not None and hardware_floor > 0
            else None
        )
        prediction_status = (
            hardware_prediction.status
            if hardware_prediction is not None and scope_bound is not None
            else "unmapped-scope"
            if hardware_prediction is not None
            else "unavailable"
        )
        error_status = (
            "not-evaluable-hardware-floor"
            if hardware_floor is not None
            else "not-evaluable-unmapped-prediction"
            if hardware_prediction is not None
            else "not-evaluable-no-prediction"
        )
        latency_cases.append(
            {
                "case_id": case["case_id"],
                "scope": case["resolved_scope"],
                "mode": case["mode"],
                "predicted": {
                    "status": prediction_status,
                    "kind": "algorithm-independent-empirical-hardware-floor",
                    "minimum_work_flops": (
                        scope_bound.flops if scope_bound is not None else None
                    ),
                    "compulsory_bytes": (
                        scope_bound.compulsory_bytes
                        if scope_bound is not None
                        else None
                    ),
                    "empirical_compute_time_ns": (
                        scope_bound.empirical_compute_time_ns
                        if scope_bound is not None
                        else None
                    ),
                    "empirical_memory_time_ns": (
                        scope_bound.empirical_memory_time_ns
                        if scope_bound is not None
                        else None
                    ),
                    "empirical_hardware_floor_ns": hardware_floor,
                    "limiting_resource": (
                        scope_bound.limiting_resource
                        if scope_bound is not None
                        else None
                    ),
                    "full_duration_ns": None,
                    "candidate_count": len(candidates),
                    "operation_count": (
                        scope_bound.operation_count
                        if scope_bound is not None
                        else 0
                    ),
                    "aggregation": (
                        "max(sum(minimum_work)/compute_P80, "
                        "unique_scope_boundary_bytes/memory_P80)"
                    ),
                    "assumptions": (
                        list(hardware_prediction.program_bounds.assumptions)
                        if hardware_prediction is not None
                        else []
                    ),
                },
                "observed": {
                    "kind": "benchmark-median",
                    "median_ns": observed_median,
                    "q1_ns": float(case["latency"]["q1_ns"]),
                    "q3_ns": float(case["latency"]["q3_ns"]),
                    "iqr_over_median": float(
                        case["latency"]["iqr_over_median"]
                    ),
                    "throughput_per_second": float(
                        case["latency"]["throughput_per_second"]
                    ),
                },
                "comparison": {
                    "observed_minus_hardware_floor_ns": distance_from_floor,
                    "observed_to_hardware_floor_ratio": distance_ratio,
                    "relative_prediction_error": None,
                    "error_status": error_status,
                    "interpretation": (
                        "The numerical value is an algorithm-independent empirical "
                        "hardware floor, not the current implementation duration. "
                        "The distance is optimization headroom, not prediction error."
                    ),
                },
            }
        )

    predicted_peak = int(live_set["predicted_framework_peak_bytes"])
    observed_peak = int(tensor_storage_observation["peak_framework_tensor_bytes"])
    signed_memory_error = predicted_peak - observed_peak
    absolute_memory_error = abs(signed_memory_error)
    memory_comparison = {
        "predicted": {
            "framework_peak_bytes": predicted_peak,
            "kind": "semantic-live-set",
            "peak_stable_path": live_set["peak_operation_stable_path"],
            "exclusions": list(live_set["exclusions"]),
        },
        "observed": {
            "framework_peak_bytes": observed_peak,
            "kind": tensor_storage_observation["observer"],
            "peak_stable_path": tensor_storage_observation["peak_stable_path"],
            "exclusions": list(tensor_storage_observation["excludes"]),
        },
        "comparison": {
            "predicted_minus_observed_bytes": signed_memory_error,
            "absolute_error_bytes": absolute_memory_error,
            "absolute_relative_error": (
                absolute_memory_error / observed_peak if observed_peak else None
            ),
            "predicted_to_observed_ratio": (
                predicted_peak / observed_peak if observed_peak else None
            ),
            "error_status": "evaluated" if observed_peak else "undefined-zero-observation",
        },
    }
    status = (
        "empirical-hardware-floor-with-observation"
        if hardware_prediction is not None
        and hardware_prediction.status == "empirical-hardware-lower-bound"
        else "partial-base-prediction"
        if hardware_prediction is not None
        and not hardware_prediction.prediction_complete
        else "observation-only"
        if hardware_prediction is None
        else "base-prediction"
    )
    summary = {
        "aligned_latency_cases": sum(
            1
            for item in latency_cases
            if item["predicted"]["empirical_hardware_floor_ns"] is not None
        ),
        "evaluable_latency_errors": 0,
        "evaluable_memory_errors": 1 if observed_peak else 0,
    }
    base_prediction = {
        "hardware_backend_id": (
            hardware_prediction.backend_id
            if hardware_prediction is not None
            else None
        ),
        "hardware_compilation_fingerprint": (
            hardware_prediction.compilation_fingerprint
            if hardware_prediction is not None
            else None
        ),
        "prediction_complete": (
            hardware_prediction.prediction_complete
            if hardware_prediction is not None
            else False
        ),
    }
    fingerprint = content_fingerprint(
        COMPARISON_SCHEMA,
        status,
        base_prediction,
        latency_cases,
        memory_comparison,
        summary,
    )
    return {
        "schema": COMPARISON_SCHEMA,
        "comparison_fingerprint": fingerprint,
        "status": status,
        "base_prediction": base_prediction,
        "latency_cases": latency_cases,
        "memory": memory_comparison,
        "summary": summary,
    }


__all__ = ["build_prediction_observation_comparison"]
