"""Prediction-versus-observation comparison with explicit validity semantics."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any

from groundupscale.benchmark.decomposition import build_latency_decomposition
from groundupscale.benchmark.frontier_evidence import (
    ExactTimingValidity,
    embedded_digest_is_valid,
    validate_exact_matmul_correctness,
    validate_exact_timing_evidence,
)
from groundupscale.ir import (
    HardwareBackendPrediction,
    canonical_data,
    content_fingerprint,
)
from groundupscale.schemas.v1alpha1 import ExactOperatorExecutionContract


COMPARISON_SCHEMA = (
    "groundupscale.dev/prediction-observation-comparison/v1alpha2"
)
FRONTIER_COMPARISON_UNCERTAINTY_POLICY = {
    "policy_id": "exact-frontier-observation-combined-uncertainty",
    "version": "1.0.0",
    "composition": "root-sum-square",
    "maximum_session_repeatability_fraction": 0.05,
    "coverage_basis": "six-session-qualification-bound-and-current-IQR",
}


def _candidate_is_in_scope(candidate_path: str, runtime_scope: str) -> bool:
    normalized = candidate_path.removeprefix("cost/")
    return normalized == runtime_scope or normalized.startswith(runtime_scope + "/")


def _current_correctness_is_qualified(
    record: dict[str, Any] | None,
    *,
    candidate: object,
    input_corpus: object,
    execution: object,
) -> bool:
    return bool(
        validate_exact_matmul_correctness(
            record,
            candidate=candidate,
            input_corpus=input_corpus,
            execution=execution,
        )
    )


@dataclass(frozen=True)
class _CurrentFrontierObservationValidity:
    applicable: bool
    candidate_identity_valid: bool
    input_corpus_valid: bool
    execution_contract_valid: bool
    correctness_valid: bool
    timing: ExactTimingValidity
    gate_reason_codes: tuple[str, ...]


def _validate_current_frontier_observation(
    case: dict[str, Any],
    *,
    operator_record: dict[str, Any] | None,
    environment_evidence_tier: str,
    environment_reason_codes: tuple[str, ...],
) -> _CurrentFrontierObservationValidity:
    candidate = case.get("candidate_identity")
    input_corpus = case.get("input_corpus")
    execution = case.get("execution_contract")
    timing = validate_exact_timing_evidence(case)
    applicable = bool(
        case.get("mode") == "operator"
        and all(
            isinstance(value, dict) and value.get("status") == "resolved"
            for value in (candidate, input_corpus, execution)
        )
    )
    candidate_valid = applicable and embedded_digest_is_valid(
        candidate, "candidate_digest"
    )
    input_valid = applicable and embedded_digest_is_valid(
        input_corpus, "input_corpus_digest"
    )
    execution_valid = False
    if applicable:
        try:
            ExactOperatorExecutionContract.model_validate(execution)
            execution_valid = True
        except ValueError:
            pass
    correctness_valid = applicable and _current_correctness_is_qualified(
        operator_record,
        candidate=candidate,
        input_corpus=input_corpus,
        execution=execution,
    )
    reasons: list[str] = []
    if applicable:
        if not candidate_valid:
            reasons.append("operator-candidate-identity-digest-invalid")
        if not input_valid:
            reasons.append("operator-input-corpus-digest-invalid")
        if not execution_valid:
            reasons.append("operator-execution-contract-invalid")
        if environment_evidence_tier != "qualified":
            reasons.extend(environment_reason_codes)
            if not environment_reason_codes:
                reasons.append("environment-evidence-unqualified")
        if not correctness_valid:
            reasons.append("operator-correctness-evidence-unqualified")
        if not timing.qualified:
            reasons.extend(timing.reason_codes)
    return _CurrentFrontierObservationValidity(
        applicable=applicable,
        candidate_identity_valid=candidate_valid,
        input_corpus_valid=input_valid,
        execution_contract_valid=execution_valid,
        correctness_valid=correctness_valid,
        timing=timing,
        gate_reason_codes=tuple(dict.fromkeys(reasons)),
    )


def build_prediction_observation_comparison(
    *,
    hardware_prediction: HardwareBackendPrediction | None,
    benchmark: dict[str, Any],
    trace: dict[str, Any] | None,
    live_set: dict[str, Any],
    tensor_storage_observation: dict[str, Any],
    observation_evidence_tier: str = "unverified",
    observation_reason_codes: tuple[str, ...] = (),
    observation_hardware_cohort: str | None = None,
    observation_operator_cases: tuple[dict[str, Any], ...] = (),
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
        provisional_estimate = (
            scope_bound.provisional_estimate_ns
            if scope_bound is not None
            else None
        )
        operator_frontier = (
            scope_bound.operator_achievable_frontier_ns
            if scope_bound is not None
            else None
        )
        operator_frontier_uncertainty = (
            scope_bound.operator_frontier_standard_uncertainty_ns
            if scope_bound is not None
            else None
        )
        operator_records = tuple(
            item
            for item in observation_operator_cases
            if item.get("case_id") == case.get("case_id")
            and item.get("stable_path") == case.get("resolved_scope")
        )
        current_candidate = case.get("candidate_identity")
        current_input = case.get("input_corpus")
        current_execution = case.get("execution_contract")
        current_operator_record = operator_records[0] if len(operator_records) == 1 else None
        current_validity = _validate_current_frontier_observation(
            case,
            operator_record=current_operator_record,
            environment_evidence_tier=observation_evidence_tier,
            environment_reason_codes=observation_reason_codes,
        )
        timing_validity = current_validity.timing
        exact_observation_applicable = current_validity.applicable
        case_observation_reason_codes = list(current_validity.gate_reason_codes)
        case_observation_status = (
            "qualified"
            if exact_observation_applicable and not case_observation_reason_codes
            else "unqualified"
            if exact_observation_applicable
            else "not-evaluated"
        )
        case_evidence_tier = (
            case_observation_status
            if exact_observation_applicable
            else observation_evidence_tier
        )
        case_reason_codes = (
            case_observation_reason_codes
            if exact_observation_applicable
            else list(observation_reason_codes)
        )
        frontier_observation_reason_codes: list[str] = []
        if operator_frontier is not None:
            if observation_hardware_cohort != scope_bound.operator_frontier_hardware_cohort:
                frontier_observation_reason_codes.append(
                    "operator-frontier-observation-cohort-mismatch"
                )
            if (
                not current_validity.candidate_identity_valid
            ):
                frontier_observation_reason_codes.append(
                    "operator-frontier-observation-candidate-invalid"
                )
            elif (
                not isinstance(current_candidate, dict)
                or current_candidate.get("candidate_digest")
                != scope_bound.operator_frontier_candidate_digest
            ):
                frontier_observation_reason_codes.append(
                    "operator-frontier-observation-candidate-mismatch"
                )
            if (
                not current_validity.input_corpus_valid
            ):
                frontier_observation_reason_codes.append(
                    "operator-frontier-observation-input-corpus-invalid"
                )
            elif (
                not isinstance(current_input, dict)
                or current_input.get("input_corpus_digest")
                != scope_bound.operator_frontier_input_corpus_digest
            ):
                frontier_observation_reason_codes.append(
                    "operator-frontier-observation-input-corpus-mismatch"
                )
            if (
                not current_validity.execution_contract_valid
            ):
                frontier_observation_reason_codes.append(
                    "operator-frontier-observation-execution-domain-invalid"
                )
            elif (
                not isinstance(current_execution, dict)
                or current_execution.get("execution_contract_digest")
                != scope_bound.operator_frontier_execution_contract_digest
            ):
                frontier_observation_reason_codes.append(
                    "operator-frontier-observation-execution-domain-mismatch"
                )
            if not current_validity.correctness_valid:
                frontier_observation_reason_codes.append(
                    "operator-frontier-observation-correctness-unqualified"
                )
            if (
                observation_evidence_tier != "qualified"
                or not timing_validity.qualified
            ):
                frontier_observation_reason_codes.append(
                    "operator-frontier-observation-timing-unqualified"
                )
                frontier_observation_reason_codes.extend(
                    timing_validity.reason_codes
                )
        comparable_frontier = (
            operator_frontier if not frontier_observation_reason_codes else None
        )
        comparable_uncertainty = (
            operator_frontier_uncertainty
            if comparable_frontier is not None
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
        provisional_gap = (
            observed_median - provisional_estimate
            if provisional_estimate is not None
            else None
        )
        provisional_ratio = (
            observed_median / provisional_estimate
            if provisional_estimate is not None and provisional_estimate > 0
            else None
        )
        frontier_gap = (
            observed_median - comparable_frontier
            if comparable_frontier is not None
            else None
        )
        observed_to_frontier_ratio = (
            observed_median / comparable_frontier
            if comparable_frontier is not None and comparable_frontier > 0
            else None
        )
        frontier_efficiency = (
            comparable_frontier / observed_median
            if comparable_frontier is not None and observed_median > 0
            else None
        )
        current_timer = (
            case.get("timing_contract", {}).get("timer")
            if isinstance(case.get("timing_contract"), dict)
            else None
        )
        observation_iqr_half_width = abs(
            float(case["latency"]["q3_ns"]) - float(case["latency"]["q1_ns"])
        ) / 2
        timer_resolution = (
            float(current_timer["resolution_ns"])
            if comparable_frontier is not None
            and isinstance(current_timer, dict)
            and isinstance(current_timer.get("resolution_ns"), (int, float))
            else None
        )
        repeatability_budget = (
            max(abs(comparable_frontier), abs(observed_median))
            * FRONTIER_COMPARISON_UNCERTAINTY_POLICY[
                "maximum_session_repeatability_fraction"
            ]
            if comparable_frontier is not None
            else None
        )
        combined_frontier_uncertainty = (
            hypot(
                float(comparable_uncertainty),
                observation_iqr_half_width,
                float(repeatability_budget),
                float(timer_resolution),
            )
            if comparable_frontier is not None
            and comparable_uncertainty is not None
            and repeatability_budget is not None
            and timer_resolution is not None
            else None
        )
        frontier_efficiency_status = (
            "qualified"
            if comparable_frontier is not None
            else "not-evaluable-observation-domain"
            if operator_frontier is not None
            else "not-evaluable"
        )
        frontier_gap_status = (
            "within-combined-uncertainty"
            if frontier_gap is not None
            and combined_frontier_uncertainty is not None
            and abs(frontier_gap) <= combined_frontier_uncertainty
            else "faster-observation-requires-requalification"
            if frontier_gap is not None
            and combined_frontier_uncertainty is not None
            and frontier_gap < -combined_frontier_uncertainty
            else "measurable-headroom-to-frontier"
            if frontier_gap is not None
            and combined_frontier_uncertainty is not None
            and frontier_gap > combined_frontier_uncertainty
            else "not-evaluable"
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
            else "not-evaluable-phase-capabilities-incomplete"
            if scope_bound is not None
            else "not-evaluable-unmapped-prediction"
            if hardware_prediction is not None
            else "not-evaluable-no-prediction"
        )
        compound_phase_schedule = (
            canonical_data(candidates[0].phase_schedule)
            if len(candidates) == 1 and candidates[0].phase_schedule is not None
            else None
        )
        compound_provisional_phase_schedule = (
            canonical_data(candidates[0].provisional_phase_schedule)
            if len(candidates) == 1
            and candidates[0].provisional_phase_schedule is not None
            else None
        )
        latency_cases.append(
            {
                "case_id": case["case_id"],
                "scope": case["resolved_scope"],
                "mode": case["mode"],
                "predicted": {
                    "status": prediction_status,
                    "kind": (
                        "algorithm-independent-empirical-serialized-hardware-floor"
                    ),
                    "minimum_work_flops": (
                        scope_bound.flops if scope_bound is not None else None
                    ),
                    "compulsory_bytes": (
                        scope_bound.compulsory_bytes
                        if scope_bound is not None
                        else None
                    ),
                    "materialized_bytes": (
                        scope_bound.materialized_bytes
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
                    "provisional_estimate_ns": provisional_estimate,
                    "provisional_evidence_tier": (
                        scope_bound.provisional_evidence_tier
                        if scope_bound is not None
                        else None
                    ),
                    "provisional_reason_codes": (
                        list(scope_bound.provisional_reason_codes)
                        if scope_bound is not None
                        else []
                    ),
                    "operator_achievable_frontier_ns": operator_frontier,
                    "operator_frontier_standard_uncertainty_ns": (
                        operator_frontier_uncertainty
                    ),
                    "operator_frontier_match_status": (
                        scope_bound.operator_frontier_match_status
                        if scope_bound is not None
                        else "unmapped-scope"
                    ),
                    "operator_frontier_anchor_ids": (
                        list(scope_bound.operator_frontier_anchor_ids)
                        if scope_bound is not None
                        else []
                    ),
                    "operator_frontier_hardware_cohort": (
                        scope_bound.operator_frontier_hardware_cohort
                        if scope_bound is not None
                        else None
                    ),
                    "operator_frontier_candidate_digest": (
                        scope_bound.operator_frontier_candidate_digest
                        if scope_bound is not None
                        else None
                    ),
                    "operator_frontier_input_corpus_digest": (
                        scope_bound.operator_frontier_input_corpus_digest
                        if scope_bound is not None
                        else None
                    ),
                    "operator_frontier_execution_contract_digest": (
                        scope_bound.operator_frontier_execution_contract_digest
                        if scope_bound is not None
                        else None
                    ),
                    "operator_frontier_reason_codes": (
                        list(scope_bound.operator_frontier_reason_codes)
                        + frontier_observation_reason_codes
                        if scope_bound is not None
                        else ["unmapped-scope"]
                    ),
                    "schedule": (
                        scope_bound.schedule if scope_bound is not None else None
                    ),
                    "serialized_hardware_floor_ns": (
                        scope_bound.serialized_hardware_floor_ns
                        if scope_bound is not None
                        else None
                    ),
                    "critical_path_hardware_floor_ns": (
                        scope_bound.critical_path_hardware_floor_ns
                        if scope_bound is not None
                        else None
                    ),
                    "resource_hardware_floor_ns": (
                        scope_bound.resource_hardware_floor_ns
                        if scope_bound is not None
                        else None
                    ),
                    "resource_physical_floor_ns": (
                        scope_bound.resource_physical_floor_ns
                        if scope_bound is not None
                        else None
                    ),
                    "ideal_dag_hardware_floor_ns": (
                        scope_bound.ideal_dag_hardware_floor_ns
                        if scope_bound is not None
                        else None
                    ),
                    "limiting_resource": (
                        scope_bound.limiting_resource
                        if scope_bound is not None
                        else None
                    ),
                    "resource_limiting_resource": (
                        scope_bound.resource_limiting_resource
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
                    "compound_phase_schedule": compound_phase_schedule,
                    "compound_provisional_phase_schedule": (
                        compound_provisional_phase_schedule
                    ),
                    "aggregation": (
                        "selected=sum(candidate selected durations); dependent "
                        "compound phases are serial without an explicit chunk "
                        "pipeline contract"
                    ),
                    "assumptions": (
                        list(hardware_prediction.program_bounds.assumptions)
                        if hardware_prediction is not None
                        else []
                    ),
                },
                "observed": {
                    "kind": "benchmark-median",
                    "environment_evidence_tier": observation_evidence_tier,
                    "environment_reason_codes": list(observation_reason_codes),
                    "evidence_tier": case_evidence_tier,
                    "reason_codes": case_reason_codes,
                    "frontier_observation_gate": {
                        "status": case_observation_status,
                        "reason_codes": case_observation_reason_codes,
                    },
                    "hardware_cohort": observation_hardware_cohort,
                    "candidate_digest": (
                        current_candidate.get("candidate_digest")
                        if isinstance(current_candidate, dict)
                        else None
                    ),
                    "input_corpus_digest": (
                        current_input.get("input_corpus_digest")
                        if isinstance(current_input, dict)
                        else None
                    ),
                    "execution_contract_digest": (
                        current_execution.get("execution_contract_digest")
                        if isinstance(current_execution, dict)
                        else None
                    ),
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
                    "observed_minus_provisional_ns": provisional_gap,
                    "observed_to_provisional_ratio": provisional_ratio,
                    "observed_minus_operator_frontier_ns": frontier_gap,
                    "observed_to_operator_frontier_ratio": (
                        observed_to_frontier_ratio
                    ),
                    "operator_frontier_efficiency": frontier_efficiency,
                    "frontier_efficiency_status": frontier_efficiency_status,
                    "operator_frontier_gap_status": frontier_gap_status,
                    "operator_frontier_combined_uncertainty_ns": (
                        combined_frontier_uncertainty
                    ),
                    "operator_frontier_uncertainty_components_ns": {
                        "anchor_standard_uncertainty": comparable_uncertainty,
                        "current_observation_iqr_half_width": (
                            observation_iqr_half_width
                            if comparable_frontier is not None
                            else None
                        ),
                        "session_repeatability_budget": repeatability_budget,
                        "timer_resolution": timer_resolution,
                    },
                    "operator_frontier_uncertainty_policy": (
                        FRONTIER_COMPARISON_UNCERTAINTY_POLICY
                        if comparable_frontier is not None
                        else None
                    ),
                    "operator_frontier_comparison_reason_codes": (
                        frontier_observation_reason_codes
                    ),
                    "relative_prediction_error": None,
                    "error_status": error_status,
                    "interpretation": (
                        "The exact-Shape Operator Frontier and Observation share "
                        "the declared Shape, execution domain, candidate family, "
                        "and hardware cohort. Frontier Efficiency compares achieved "
                        "capability with that Frontier; it is not prediction error. "
                        f"Gap status: {frontier_gap_status}."
                        if comparable_frontier is not None
                        else "The exact-Shape Operator Frontier exists, but the "
                        "current Observation did not reproduce its cohort, candidate, "
                        "input corpus, execution, correctness, and timing validity key; "
                        "Frontier Efficiency is not evaluable."
                        if operator_frontier is not None
                        else "The selected value is unavailable because at least one "
                        "compound phase lacks exact capability evidence. The generic "
                        "resource physical floor remains a non-adoptable reference."
                        + (
                            " A separately labeled exploratory provisional estimate "
                            "is available for planning but is not authoritative."
                            if provisional_estimate is not None
                            else ""
                        )
                        if scope_bound is not None and hardware_floor is None
                        else "The numerical value is an algorithm-independent empirical "
                        "hardware floor for the declared serialized-unfused schedule, "
                        "not the current implementation duration. The distance is "
                        "optimization headroom, not prediction error."
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
        "exploratory-estimate-with-observation"
        if any(
            item["predicted"]["provisional_estimate_ns"] is not None
            for item in latency_cases
        )
        else
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
        "provisional_latency_comparisons": sum(
            1
            for item in latency_cases
            if item["predicted"]["provisional_estimate_ns"] is not None
        ),
        "qualified_frontier_comparisons": sum(
            1
            for item in latency_cases
            if item["comparison"]["frontier_efficiency_status"] == "qualified"
        ),
        "evaluable_memory_errors": 1 if observed_peak else 0,
    }
    latency_decomposition = (
        build_latency_decomposition(
            hardware_prediction,
            trace,
            frontier_observation_by_path={
                item["scope"]: {
                    "status": item["comparison"]["frontier_efficiency_status"],
                    "reason_codes": item["comparison"][
                        "operator_frontier_comparison_reason_codes"
                    ],
                }
                for item in latency_cases
                if item["predicted"]["operator_frontier_match_status"]
                == "exact-anchor"
            },
        )
        if hardware_prediction is not None and trace is not None
        else None
    )
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
        "provisional_estimate_ns": (
            hardware_prediction.program_bounds.provisional_estimate_ns
            if hardware_prediction is not None
            else None
        ),
        "provisional_evidence_tier": (
            hardware_prediction.program_bounds.provisional_evidence_tier
            if hardware_prediction is not None
            else None
        ),
    }
    fingerprint = content_fingerprint(
        COMPARISON_SCHEMA,
        status,
        base_prediction,
        latency_cases,
        latency_decomposition,
        memory_comparison,
        summary,
    )
    return {
        "schema": COMPARISON_SCHEMA,
        "comparison_fingerprint": fingerprint,
        "status": status,
        "base_prediction": base_prediction,
        "latency_cases": latency_cases,
        "latency_decomposition": latency_decomposition,
        "memory": memory_comparison,
        "summary": summary,
    }


__all__ = ["build_prediction_observation_comparison"]
