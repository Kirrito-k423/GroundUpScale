"""Pure verdict logic for the disposable issue-6 prototype."""

from __future__ import annotations

import statistics
from typing import Any


def _quartiles(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return float(q1), float(q3)


def _aggregate(sessions: list[dict[str, Any]], section: str, key: str) -> dict[str, Any]:
    measurements = [session[section]["measurements"][key] for session in sessions]
    medians = [float(item["summary"]["median_ns"]) for item in measurements]
    median_ns = float(statistics.median(medians))
    q1_ns, q3_ns = _quartiles(medians)
    return {
        "measurement_id": key,
        "session_medians_ns": medians,
        "median_of_session_medians_ns": median_ns,
        "q1_ns": q1_ns,
        "q3_ns": q3_ns,
        "iqr_ns": q3_ns - q1_ns,
        "iqr_over_median": (q3_ns - q1_ns) / median_ns,
        "all_sessions_correct": all(
            item.get("correctness", {}).get("passed") is True for item in measurements
        ),
        "all_sessions_output_contiguous": all(
            item.get("correctness", {})
            .get("output_layout", {})
            .get("contiguous")
            is True
            for item in measurements
        ),
    }


def classify_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    sessions = evidence["sessions"]
    environment_eligible = evidence["environment"]["eligible"] is True
    reference_rate = float(evidence["protocol"]["old_256_reference_rate_flops_per_s"])
    anomaly_work = int(evidence["protocol"]["anomaly_work_flops"])
    aligned_work = int(evidence["protocol"]["aligned_work_flops"])

    target = _aggregate(sessions, "anomaly", "torch-direct")
    alternative_ids = [
        key
        for key, manifest in evidence["candidate_manifest"].items()
        if manifest["role"] == "alternative"
    ]
    alternatives = [_aggregate(sessions, "anomaly", key) for key in alternative_ids]
    correct_alternatives = [item for item in alternatives if item["all_sessions_correct"]]
    best = min(correct_alternatives, key=lambda item: item["median_of_session_medians_ns"])
    all_session_faster = all(
        alternative < direct
        for alternative, direct in zip(
            best["session_medians_ns"], target["session_medians_ns"], strict=True
        )
    )
    speedup_fraction = (
        target["median_of_session_medians_ns"] / best["median_of_session_medians_ns"] - 1.0
    )
    uncertainty_floor = max(
        0.05, target["iqr_over_median"] + best["iqr_over_median"]
    )
    recovered_rate = anomaly_work * 1_000_000_000.0 / best["median_of_session_medians_ns"]
    recovery_ratio = recovered_rate / reference_rate
    headroom_checks = {
        "environment_eligible": environment_eligible,
        "target_correct_in_all_sessions": target["all_sessions_correct"],
        "correct_alternative_exists": bool(correct_alternatives),
        "best_alternative_faster_in_every_session": all_session_faster,
        "speedup_exceeds_locked_uncertainty": speedup_fraction > uncertainty_floor,
        "best_recovers_90_percent_of_old_reference": recovery_ratio >= 0.90,
    }
    headroom_verdict = (
        "implementation_headroom"
        if all(headroom_checks.values())
        else "insufficient_evidence"
    )
    headroom = {
        "scenario": "257-cube-exact-shape-candidate-search",
        "verdict": headroom_verdict,
        "target": target,
        "best_correct_alternative": best,
        "speedup_fraction": speedup_fraction,
        "locked_uncertainty_fraction": uncertainty_floor,
        "recovered_rate_flops_per_s": recovered_rate,
        "recovery_vs_old_256_reference": recovery_ratio,
        "checks": headroom_checks,
        "failed_checks": [name for name, passed in headroom_checks.items() if not passed],
    }

    legacy_target = _aggregate(
        sessions,
        "implementation_headroom",
        "legacy-einsum-contiguous",
    )
    legacy_candidate = _aggregate(
        sessions,
        "implementation_headroom",
        "batched-matmul-transpose-contiguous",
    )
    legacy_all_session_faster = all(
        candidate < target
        for candidate, target in zip(
            legacy_candidate["session_medians_ns"],
            legacy_target["session_medians_ns"],
            strict=True,
        )
    )
    legacy_speedup_fraction = (
        legacy_target["median_of_session_medians_ns"]
        / legacy_candidate["median_of_session_medians_ns"]
        - 1.0
    )
    legacy_uncertainty = max(
        0.05,
        legacy_target["iqr_over_median"]
        + legacy_candidate["iqr_over_median"],
    )
    legacy_checks = {
        "environment_eligible": environment_eligible,
        "target_correct_in_all_sessions": legacy_target["all_sessions_correct"],
        "candidate_correct_in_all_sessions": legacy_candidate[
            "all_sessions_correct"
        ],
        "target_output_contiguous_in_all_sessions": legacy_target[
            "all_sessions_output_contiguous"
        ],
        "candidate_output_contiguous_in_all_sessions": legacy_candidate[
            "all_sessions_output_contiguous"
        ],
        "candidate_faster_in_every_session": legacy_all_session_faster,
        "speedup_exceeds_locked_uncertainty": legacy_speedup_fraction
        > legacy_uncertainty,
    }
    legacy_headroom = {
        "scenario": "context-matmul-legacy-einsum-vs-batched-matmul",
        "verdict": (
            "implementation_headroom"
            if all(legacy_checks.values())
            else "insufficient_evidence"
        ),
        "target": legacy_target,
        "correct_alternative": legacy_candidate,
        "speedup_fraction": legacy_speedup_fraction,
        "locked_uncertainty_fraction": legacy_uncertainty,
        "checks": legacy_checks,
        "failed_checks": [
            name for name, passed in legacy_checks.items() if not passed
        ],
    }

    operator = _aggregate(sessions, "integration", "operator")
    copies = _aggregate(sessions, "integration", "copy-twice")
    e2e = _aggregate(sessions, "integration", "operator-plus-copy-twice")
    operator_rate = aligned_work * 1_000_000_000.0 / operator["median_of_session_medians_ns"]
    operator_reference_delta = abs(operator_rate / reference_rate - 1.0)
    e2e_gap_fraction = (
        e2e["median_of_session_medians_ns"] / operator["median_of_session_medians_ns"] - 1.0
    )
    integration_uncertainty = max(
        0.10, operator["iqr_over_median"] + e2e["iqr_over_median"]
    )
    measured_excess_ns = (
        e2e["median_of_session_medians_ns"] - operator["median_of_session_medians_ns"]
    )
    copy_ablation_ns = copies["median_of_session_medians_ns"]
    ablation_relative_error = abs(measured_excess_ns - copy_ablation_ns) / max(
        measured_excess_ns, copy_ablation_ns
    )
    integration_checks = {
        "environment_eligible": environment_eligible,
        "standalone_operator_correct": operator["all_sessions_correct"],
        "wrapped_e2e_correct": e2e["all_sessions_correct"],
        "standalone_within_10_percent_of_old_anchor": operator_reference_delta <= 0.10,
        "e2e_gap_exceeds_locked_uncertainty": e2e_gap_fraction > integration_uncertainty,
        "copy_ablation_explains_excess_within_35_percent": ablation_relative_error <= 0.35,
    }
    integration_verdict = (
        "integration_overhead"
        if all(integration_checks.values())
        else "insufficient_evidence"
    )
    integration = {
        "scenario": "256-cube-standalone-vs-materializing-wrapper",
        "verdict": integration_verdict,
        "operator": operator,
        "copy_twice_ablation": copies,
        "wrapped_e2e": e2e,
        "operator_rate_flops_per_s": operator_rate,
        "operator_reference_delta_fraction": operator_reference_delta,
        "e2e_gap_fraction": e2e_gap_fraction,
        "locked_uncertainty_fraction": integration_uncertainty,
        "measured_excess_ns": measured_excess_ns,
        "copy_ablation_ns": copy_ablation_ns,
        "ablation_relative_error": ablation_relative_error,
        "checks": integration_checks,
        "failed_checks": [name for name, passed in integration_checks.items() if not passed],
    }

    target_only = {
        "scenario": "257-cube-target-only-counterexample",
        "verdict": "insufficient_evidence",
        "candidate_coverage": "C0_TARGET_ONLY",
        "retained_observation": target,
        "reason_codes": [
            "no_correct_alternative_candidate",
            "single_slow_measurement_cannot_lower_frontier",
        ],
    }
    frontier_shift_gate = {
        "scenario": "257-cube-frontier-shift-gate",
        "verdict": "insufficient_evidence",
        "candidate_coverage": "C1_MULTIPLE_WRAPPERS_SINGLE_ACCELERATE_LIBRARY",
        "reason_codes": [
            "c2_or_c3_independent_candidate_families_missing",
            "neighbourhood_regime_not_validated",
        ] + ([] if environment_eligible else ["environment_ineligible"]),
        "prohibited_verdict": "frontier_shift",
    }

    negative_control_rejected = all(
        session["anomaly"]["measurements"]["truncated-256-negative-control"][
            "correctness"
        ]["passed"]
        is False
        for session in sessions
    )
    observed_verdicts = {
        headroom["verdict"],
        legacy_headroom["verdict"],
        integration["verdict"],
        target_only["verdict"],
        frontier_shift_gate["verdict"],
    }
    assertions = {
        "three_independent_sessions": len(sessions) == 3
        and len({session["process_id"] for session in sessions}) == 3,
        "environment_eligible": environment_eligible,
        "negative_control_rejected_in_every_session": negative_control_rejected,
        "implementation_headroom_observed": legacy_headroom["verdict"]
        == "implementation_headroom",
        "integration_overhead_observed": integration["verdict"]
        == "integration_overhead",
        "insufficient_evidence_counterexample_preserved": target_only["verdict"]
        == "insufficient_evidence"
        and frontier_shift_gate["verdict"] == "insufficient_evidence",
        "at_least_two_distinct_verdicts": len(observed_verdicts) >= 2,
    }
    return {
        "scenarios": [
            headroom,
            legacy_headroom,
            integration,
            target_only,
            frontier_shift_gate,
        ],
        "assertions": assertions,
        "exit_criteria_passed": all(assertions.values()),
    }
