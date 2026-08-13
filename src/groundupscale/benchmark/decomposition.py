"""Mandatory predicted-versus-observed latency decomposition."""

from __future__ import annotations

from math import isclose
from typing import Any

from groundupscale.ir import HardwareBackendPrediction


TOP_K = 10
MANDATORY_SHARE = 0.10
UNATTRIBUTED_PATH = "unattributed://host-runtime"


def _union_duration(intervals: list[tuple[int, int]]) -> int:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _rank_and_select(
    items: list[dict[str, Any]], e2e_ns: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(items, key=lambda item: (-item["time_ns"], item["stable_path"]))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
        item["share_of_e2e"] = item["time_ns"] / e2e_ns if e2e_ns else None

    top10 = ranked[:TOP_K]
    mandatory = [
        item for item in ranked if item["time_ns"] >= e2e_ns * MANDATORY_SHARE
    ]
    top10_paths = {item["stable_path"] for item in top10}
    mandatory_paths = {item["stable_path"] for item in mandatory}
    selected = []
    for item in ranked:
        path = item["stable_path"]
        if path not in top10_paths and path not in mandatory_paths:
            continue
        selected.append(
            {
                **item,
                "selection_reasons": [
                    reason
                    for reason, applies in (
                        ("top10", path in top10_paths),
                        ("at-least-10%-of-e2e", path in mandatory_paths),
                    )
                    if applies
                ],
            }
        )
    return ranked, top10, mandatory, selected


def _predicted_decomposition(
    prediction: HardwareBackendPrediction,
) -> dict[str, Any]:
    authored_e2e_ns = prediction.program_bounds.empirical_hardware_floor_ns
    provisional = False
    if authored_e2e_ns is None:
        authored_e2e_ns = prediction.program_bounds.provisional_estimate_ns
        provisional = authored_e2e_ns is not None
    if authored_e2e_ns is None:
        return {
            "available": False,
            "reason": "selected hardware floor is unavailable",
            "all_items": [],
            "top10": [],
            "mandatory": [],
            "selected": [],
        }

    items = []
    for candidate in prediction.candidates:
        exact_frontier = (
            candidate.duration.operator_achievable_frontier_ns
            if provisional
            and candidate.duration.operator_frontier_match_status == "exact-anchor"
            else None
        )
        fallback = (
            candidate.duration.provisional_estimate_ns
            if provisional
            else candidate.duration.empirical_hardware_floor_ns
        )
        selected_duration = exact_frontier if exact_frontier is not None else fallback
        if selected_duration is None:
            continue
        item = {
            "stable_path": candidate.stable_path.removeprefix("cost/"),
            "operation": candidate.operation,
            "time_ns": selected_duration,
            "source_id": candidate.candidate_id,
            "evidence": (
                "exact-operator-frontier"
                if exact_frontier is not None
                else
                "exploratory-provisional-candidate-estimate"
                if provisional
                else "implementation-candidate-local-hardware-floor"
            ),
        }
        if exact_frontier is not None:
            item.update(
                {
                    "frontier_anchor_id": (
                        candidate.duration.operator_frontier_anchor_id
                    ),
                    "frontier_profile": candidate.duration.operator_frontier_profile,
                    "frontier_profile_version": (
                        candidate.duration.operator_frontier_profile_version
                    ),
                    "frontier_standard_uncertainty_ns": (
                        candidate.duration.operator_frontier_standard_uncertainty_ns
                    ),
                }
            )
        items.append(item)
    exact_item_count = sum(
        item["evidence"] == "exact-operator-frontier" for item in items
    )
    e2e_ns = sum(float(item["time_ns"]) for item in items) if provisional else authored_e2e_ns
    ranked, top10, mandatory, selected = _rank_and_select(items, e2e_ns)
    all_attributed_ns = sum(item["time_ns"] for item in ranked)
    residual_ns = e2e_ns - all_attributed_ns
    if not isclose(residual_ns, 0.0, abs_tol=max(1e-6, e2e_ns * 1e-12)):
        raise ValueError(
            "serialized prediction does not reconcile to candidate local floors: "
            f"e2e={e2e_ns}, attributed={all_attributed_ns}"
        )
    selected_sum_ns = sum(item["time_ns"] for item in selected)
    return {
        "available": True,
        "kind": (
            "mixed-exact-frontier-and-provisional-estimate"
            if provisional and exact_item_count
            else
            "exploratory-provisional-estimate"
            if provisional
            else "algorithm-independent-empirical-hardware-floor"
        ),
        "statistic": (
            "modeled-exploratory-provisional-estimate"
            if provisional
            else "modeled-serialized-hardware-floor"
        ),
        "evidence_tier": (
            prediction.program_bounds.provisional_evidence_tier
            if provisional
            else "qualified"
        ),
        "reason_codes": (
            list(prediction.program_bounds.provisional_reason_codes)
            if provisional
            else []
        ),
        "schedule": prediction.program_bounds.schedule,
        "e2e_ns": e2e_ns,
        "source_program_estimate_ns": authored_e2e_ns,
        "exact_frontier_override_count": exact_item_count,
        "exact_frontier_override_delta_ns": e2e_ns - authored_e2e_ns,
        "all_items": ranked,
        "top10": top10,
        "mandatory": mandatory,
        "selected": selected,
        "reconciliation": {
            "all_attributed_ns": all_attributed_ns,
            "selected_sum_ns": selected_sum_ns,
            "other_ns": max(0.0, e2e_ns - selected_sum_ns),
            "unattributed_ns": 0.0,
            "overlap_ns": 0.0,
            "coverage": all_attributed_ns / e2e_ns if e2e_ns else None,
        },
    }


def _observed_decomposition(trace: dict[str, Any]) -> dict[str, Any]:
    e2e_events = [
        event for event in trace["events"] if event["runtime_kind"] == "e2e"
    ]
    if len(e2e_events) != 1:
        return {
            "available": False,
            "reason": f"expected one E2E trace span, found {len(e2e_events)}",
            "all_items": [],
            "top10": [],
            "mandatory": [],
            "selected": [],
        }
    operation_events = [
        event for event in trace["events"] if event["runtime_kind"] == "operation"
    ]
    unsupported_clocks = sorted(
        {
            event["clock_domain"]
            for event in operation_events
            if event["clock_domain"] != "host-synchronous"
        }
    )
    if unsupported_clocks:
        return {
            "available": False,
            "reason": (
                "operation spans are not synchronous device durations: "
                + ", ".join(unsupported_clocks)
            ),
            "all_items": [],
            "top10": [],
            "mandatory": [],
            "selected": [],
        }

    e2e_ns = float(e2e_events[0]["host_duration_ns"])
    items = [
        {
            "stable_path": event["stable_path"],
            "operation": event["operation"],
            "time_ns": float(event["host_duration_ns"]),
            "source_id": event["span_id"],
            "evidence": "single-diagnostic-trace-span",
            "host_started_ns": event["host_started_ns"],
            "host_ended_ns": event["host_ended_ns"],
        }
        for event in operation_events
    ]
    unattributed_ns = float(trace["error_attribution"]["unattributed_host_ns"])
    if unattributed_ns > 0:
        items.append(
            {
                "stable_path": UNATTRIBUTED_PATH,
                "operation": "UnattributedHostTime",
                "time_ns": unattributed_ns,
                "source_id": None,
                "evidence": trace["error_attribution"]["unattributed_reason"],
                "host_started_ns": None,
                "host_ended_ns": None,
            }
        )

    ranked, top10, mandatory, selected = _rank_and_select(items, e2e_ns)
    leaf_intervals = [
        (event["host_started_ns"], event["host_ended_ns"])
        for event in operation_events
    ]
    leaf_sum_ns = sum(event["host_duration_ns"] for event in operation_events)
    leaf_union_ns = _union_duration(leaf_intervals)
    selected_paths = {item["stable_path"] for item in selected}
    selected_intervals = [
        (event["host_started_ns"], event["host_ended_ns"])
        for event in operation_events
        if event["stable_path"] in selected_paths
    ]
    selected_unattributed_ns = (
        unattributed_ns if UNATTRIBUTED_PATH in selected_paths else 0.0
    )
    selected_accounted_ns = _union_duration(selected_intervals) + selected_unattributed_ns
    reconciled_total_ns = leaf_union_ns + unattributed_ns
    return {
        "available": True,
        "kind": "host-synchronous-operation-time",
        "statistic": "single-diagnostic-trace",
        "instrumentation_profile": trace["instrumentation_profile"],
        "e2e_ns": e2e_ns,
        "all_items": ranked,
        "top10": top10,
        "mandatory": mandatory,
        "selected": selected,
        "reconciliation": {
            "all_leaf_span_sum_ns": leaf_sum_ns,
            "attributed_interval_union_ns": leaf_union_ns,
            "reconciled_total_ns": reconciled_total_ns,
            "selected_accounted_ns": selected_accounted_ns,
            "other_ns": max(0.0, e2e_ns - selected_accounted_ns),
            "unattributed_ns": unattributed_ns,
            "overlap_ns": max(0, leaf_sum_ns - leaf_union_ns),
            "attributed_coverage": leaf_union_ns / e2e_ns if e2e_ns else None,
            "coverage": reconciled_total_ns / e2e_ns if e2e_ns else None,
        },
    }


def _join_selected(
    predicted: dict[str, Any],
    observed: dict[str, Any],
    frontier_observation_by_path: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    predicted_by_path = {
        item["stable_path"]: item for item in predicted.get("all_items", [])
    }
    observed_by_path = {
        item["stable_path"]: item for item in observed.get("all_items", [])
    }
    selected_paths = {
        item["stable_path"] for item in predicted.get("selected", [])
    } | {item["stable_path"] for item in observed.get("selected", [])}
    rows = []
    for path in selected_paths:
        predicted_item = predicted_by_path.get(path)
        observed_item = observed_by_path.get(path)
        predicted_ns = predicted_item["time_ns"] if predicted_item else None
        observed_ns = observed_item["time_ns"] if observed_item else None
        gap_ns = (
            observed_ns - predicted_ns
            if observed_ns is not None and predicted_ns is not None
            else None
        )
        frontier_observation = frontier_observation_by_path.get(path, {})
        frontier_observation_status = frontier_observation.get("status")
        exact_frontier_evidence = (
            predicted_item is not None
            and predicted_item.get("evidence") == "exact-operator-frontier"
        )
        rows.append(
            {
                "stable_path": path,
                "operation": (
                    (observed_item or predicted_item or {}).get("operation")
                ),
                "predicted_time_ns": predicted_ns,
                "predicted_share_of_e2e": (
                    predicted_item.get("share_of_e2e") if predicted_item else None
                ),
                "predicted_rank": (
                    predicted_item.get("rank") if predicted_item else None
                ),
                "observed_time_ns": observed_ns,
                "observed_share_of_e2e": (
                    observed_item.get("share_of_e2e") if observed_item else None
                ),
                "observed_rank": (
                    observed_item.get("rank") if observed_item else None
                ),
                "observed_minus_predicted_ns": gap_ns,
                "observed_to_predicted_ratio": (
                    observed_ns / predicted_ns
                    if observed_ns is not None
                    and predicted_ns is not None
                    and predicted_ns > 0
                    else None
                ),
                "predicted_evidence": (
                    predicted_item.get("evidence") if predicted_item else None
                ),
                "frontier_anchor_id": (
                    predicted_item.get("frontier_anchor_id")
                    if predicted_item
                    else None
                ),
                "frontier_observation_status": frontier_observation_status,
                "frontier_observation_reason_codes": list(
                    frontier_observation.get("reason_codes", [])
                ),
                "evidence_quality": (
                    "unattributed-evidence-boundary"
                    if path == UNATTRIBUTED_PATH
                    else "exact-operator-frontier+qualified-current-benchmark+single-diagnostic-trace"
                    if exact_frontier_evidence
                    and observed_item is not None
                    and frontier_observation_status == "qualified"
                    else "exact-operator-frontier+unqualified-current-benchmark+single-diagnostic-trace"
                    if exact_frontier_evidence
                    and observed_item is not None
                    and frontier_observation_status is not None
                    else "exact-operator-frontier+single-diagnostic-trace"
                    if exact_frontier_evidence and observed_item is not None
                    else "exact-stable-path"
                    if predicted_item is not None and observed_item is not None
                    else "not-observed-in-diagnostic-trace"
                    if predicted_item is not None
                    else "not-modeled"
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["observed_minus_predicted_ns"] is None,
            -abs(row["observed_minus_predicted_ns"] or 0),
            row["stable_path"],
        ),
    )


def build_latency_decomposition(
    hardware_prediction: HardwareBackendPrediction,
    trace: dict[str, Any],
    *,
    frontier_observation_by_path: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the report's mandatory independent Top-10 decompositions."""

    predicted = _predicted_decomposition(hardware_prediction)
    observed = _observed_decomposition(trace)
    joined = _join_selected(
        predicted,
        observed,
        frontier_observation_by_path or {},
    )
    aligned = [
        row
        for row in joined
        if row["observed_minus_predicted_ns"] is not None
        and row["stable_path"] != UNATTRIBUTED_PATH
    ]
    provisional_mode = predicted.get("kind") in {
        "exploratory-provisional-estimate",
        "mixed-exact-frontier-and-provisional-estimate",
    }
    largest = (
        {
            **max(aligned, key=lambda row: abs(row["observed_minus_predicted_ns"])),
            "evidence_boundary": "operation-leaf",
            "drilldown_status": "leaf-has-no-finer-instrumented-children",
        }
        if aligned and not provisional_mode
        else None
    )
    return {
        "schema": "groundupscale.dev/latency-decomposition/v1alpha1",
        "visibility_rule": {
            "top_k": TOP_K,
            "mandatory_share_of_e2e": MANDATORY_SHARE,
            "selection": "top-k union every item at or above the E2E share threshold",
        },
        "predicted": predicted,
        "observed": observed,
        "joined": joined,
        "comparison_role": (
            "exploratory-planning-only"
            if provisional_mode
            else "diagnostic-comparison"
        ),
        "largest_discrepancy": largest,
    }


__all__ = ["build_latency_decomposition"]
