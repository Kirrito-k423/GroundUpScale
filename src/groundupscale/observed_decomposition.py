"""Same-boundary observed decomposition for paired measurement lanes."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
import re
from statistics import median
from typing import Any, Mapping


SCHEDULE_EFFECT_INPUT_SCHEMA = (
    "groundupscale.dev/schedule-effect-input/v1alpha1"
)
OBSERVED_DECOMPOSITION_SCHEMA = (
    "groundupscale.dev/observed-decomposition/v1alpha1"
)


class ObservedDecompositionError(ValueError):
    """Paired observations cannot be decomposed at a trustworthy boundary."""


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _positive_samples(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(sample, (int, float))
            and not isinstance(sample, bool)
            and isfinite(float(sample))
            and sample > 0
            for sample in value
        )
    )


def _linear_percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def timing_summary(value: object) -> dict[str, float | int]:
    """Return replayable repeated-sample statistics for one timing lane."""

    if not _positive_samples(value):
        raise ObservedDecompositionError("invalid-baseline-raw-samples")
    samples = [float(sample) for sample in value]
    sample_median = float(median(samples))
    q1 = _linear_percentile(samples, 0.25)
    q3 = _linear_percentile(samples, 0.75)
    median_absolute_deviation = float(
        median(abs(sample - sample_median) for sample in samples)
    )
    return {
        "count": len(samples),
        "minimum_ns": min(samples),
        "q1_ns": q1,
        "median_ns": sample_median,
        "q3_ns": q3,
        "maximum_ns": max(samples),
        "iqr_ns": q3 - q1,
        "iqr_fraction_of_median": (q3 - q1) / sample_median,
        "median_absolute_deviation_ns": median_absolute_deviation,
        "mad_fraction_of_median": median_absolute_deviation / sample_median,
    }


def _identity(value: object) -> dict[str, object]:
    required = (
        "benchmark_case",
        "shape",
        "candidate_id",
        "hardware_cohort",
        "completion_boundary",
    )
    if (
        not isinstance(value, dict)
        or not all(key in value for key in required)
        or not all(_nonempty(value[key]) for key in required if key != "shape")
        or not isinstance(value["shape"], list)
        or not value["shape"]
        or not all(
            isinstance(dimension, int)
            and not isinstance(dimension, bool)
            and dimension > 0
            for dimension in value["shape"]
        )
    ):
        raise ObservedDecompositionError("invalid-complete-measurement-identity")
    return deepcopy(value)


def _source(value: object) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or not _nonempty(value.get("evidence_ref"))
        or not isinstance(value.get("artifact_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", value["artifact_sha256"]) is None
    ):
        raise ObservedDecompositionError("invalid-observation-source")
    return deepcopy(value)


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    current_start: float | None = None
    current_end: float | None = None
    for start, end in sorted(intervals):
        if current_start is None:
            current_start, current_end = start, end
        elif start <= float(current_end):
            current_end = max(float(current_end), end)
        else:
            total += float(current_end) - current_start
            current_start, current_end = start, end
    if current_start is not None:
        total += float(current_end) - current_start
    return total


def _ablation_decision(
    pair_id: str,
    identity: Mapping[str, object],
    baseline: Mapping[str, object],
    diagnostic: Mapping[str, object],
) -> tuple[bool, str | None, dict[str, object] | None]:
    value = diagnostic.get("overhead_ablation")
    if not isinstance(value, dict) or value.get("status") in {
        "not_provided",
        "unavailable",
    }:
        reason_code = (
            value.get("reason_code")
            if isinstance(value, dict) and _nonempty(value.get("reason_code"))
            else "profiling-overhead-ablation-missing"
        )
        return False, str(reason_code), None
    policy = value.get("policy")
    selection = value.get("selection")
    holdout = value.get("holdout")
    if (
        value.get("status") != "qualified"
        or value.get("instrumentation_profile")
        != diagnostic.get("instrumentation_profile")
        or not isinstance(policy, dict)
        or not _nonempty(policy.get("policy_id"))
        or not _nonempty(policy.get("version"))
        or not isinstance(policy.get("minimum_independent_sessions"), int)
        or policy["minimum_independent_sessions"] < 2
        or not isinstance(policy.get("maximum_overhead_ratio"), (int, float))
        or isinstance(policy.get("maximum_overhead_ratio"), bool)
        or not 0 <= float(policy["maximum_overhead_ratio"]) < 1
        or not isinstance(
            policy.get("maximum_iqr_fraction_of_median"), (int, float)
        )
        or isinstance(policy.get("maximum_iqr_fraction_of_median"), bool)
        or not 0 <= float(policy["maximum_iqr_fraction_of_median"]) < 1
        or not isinstance(selection, dict)
        or not isinstance(holdout, dict)
        or holdout.get("identity") != identity
        or holdout.get("pair_id") != pair_id
        or holdout.get("baseline_lane_id") != baseline.get("lane_id")
        or holdout.get("diagnostic_lane_id") != diagnostic.get("lane_id")
    ):
        return False, "profiling-overhead-ablation-unqualified", None
    session_groups = (
        selection.get("session_ids"),
        holdout.get("baseline_session_ids"),
        holdout.get("diagnostic_session_ids"),
    )
    minimum = policy["minimum_independent_sessions"]
    if (
        not all(
            isinstance(group, list)
            and len(group) >= minimum
            and len(set(group)) == len(group)
            and all(_nonempty(session_id) for session_id in group)
            for group in session_groups
        )
        or len(holdout.get("baseline_raw_samples_ns", []))
        != len(session_groups[1])
        or len(holdout.get("diagnostic_raw_samples_ns", []))
        != len(session_groups[2])
        or not set(session_groups[0]).isdisjoint(session_groups[1])
        or not set(session_groups[0]).isdisjoint(session_groups[2])
        or not set(session_groups[1]).isdisjoint(session_groups[2])
        or not _positive_samples(holdout.get("baseline_raw_samples_ns"))
        or not _positive_samples(holdout.get("diagnostic_raw_samples_ns"))
        or not all(
            _nonempty(reference)
            for reference in (
                selection.get("evidence_ref"),
                holdout.get("evidence_ref"),
                value.get("evidence_ref"),
            )
        )
    ):
        return False, "profiling-overhead-ablation-unqualified", None
    baseline_median = float(median(holdout["baseline_raw_samples_ns"]))
    diagnostic_median = float(median(holdout["diagnostic_raw_samples_ns"]))
    overhead_ratio = abs(diagnostic_median - baseline_median) / baseline_median
    baseline_summary = timing_summary(holdout["baseline_raw_samples_ns"])
    diagnostic_summary = timing_summary(holdout["diagnostic_raw_samples_ns"])
    decision = {
        "status": "qualified",
        "policy": deepcopy(policy),
        "observed_overhead_ratio": overhead_ratio,
        "baseline_timing_summary": baseline_summary,
        "diagnostic_timing_summary": diagnostic_summary,
        "selection_evidence_ref": selection["evidence_ref"],
        "holdout_evidence_ref": holdout["evidence_ref"],
        "decision_evidence_ref": value["evidence_ref"],
    }
    if (
        overhead_ratio > float(policy["maximum_overhead_ratio"])
        or float(baseline_summary["iqr_fraction_of_median"])
        > float(policy["maximum_iqr_fraction_of_median"])
        or float(diagnostic_summary["iqr_fraction_of_median"])
        > float(policy["maximum_iqr_fraction_of_median"])
    ):
        decision["status"] = "error-budget-exceeded"
        return False, "profiling-overhead-error-budget-exceeded", decision
    return True, None, decision


def _timeline_decomposition(
    timeline: object,
) -> dict[str, object]:
    if (
        not isinstance(timeline, dict)
        or not _nonempty(timeline.get("clock_domain"))
        or not isinstance(timeline.get("started_ns"), (int, float))
        or not isinstance(timeline.get("ended_ns"), (int, float))
        or isinstance(timeline.get("started_ns"), bool)
        or isinstance(timeline.get("ended_ns"), bool)
        or float(timeline["ended_ns"]) <= float(timeline["started_ns"])
        or not isinstance(timeline.get("intervals"), list)
        or not timeline["intervals"]
    ):
        raise ObservedDecompositionError("invalid-device-timeline")
    source = _source(timeline.get("source"))
    timeline_start = float(timeline["started_ns"])
    timeline_end = float(timeline["ended_ns"])
    interval_ids: set[str] = set()
    stable_leaf_paths: set[str] = set()
    leaves: list[dict[str, object]] = []
    parents: list[dict[str, object]] = []
    leaf_intervals: list[tuple[float, float]] = []
    for value in timeline["intervals"]:
        if not isinstance(value, dict):
            raise ObservedDecompositionError("invalid-device-timeline-interval")
        span_id = value.get("span_id")
        stable_path = value.get("stable_path")
        kind = value.get("kind")
        start = value.get("started_ns")
        end = value.get("ended_ns")
        if (
            not _nonempty(span_id)
            or span_id in interval_ids
            or not _nonempty(stable_path)
            or kind not in {"leaf", "exclusive-parent", "inclusive-parent"}
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or isinstance(start, bool)
            or isinstance(end, bool)
            or not timeline_start <= float(start) < float(end) <= timeline_end
            or not _nonempty(value.get("evidence_ref"))
        ):
            raise ObservedDecompositionError("invalid-device-timeline-interval")
        if kind in {"leaf", "exclusive-parent"} and stable_path in stable_leaf_paths:
            raise ObservedDecompositionError("duplicate-additive-stable-path")
        interval_ids.add(span_id)
        duration = float(end) - float(start)
        item = {
            **deepcopy(value),
            "duration_ns": duration,
            "additive": kind != "inclusive-parent",
        }
        if kind == "inclusive-parent":
            parents.append(item)
        else:
            stable_leaf_paths.add(stable_path)
            leaves.append(item)
            leaf_intervals.append((float(start), float(end)))
    if any(
        item.get("parent_span_id") is not None
        and item.get("parent_span_id") not in interval_ids
        for item in [*leaves, *parents]
    ):
        raise ObservedDecompositionError("unknown-device-timeline-parent")
    e2e_duration = timeline_end - timeline_start
    additive_sum = sum(float(item["duration_ns"]) for item in leaves)
    all_attributed = _union_duration(leaf_intervals)
    overlap = additive_sum - all_attributed
    unattributed = e2e_duration - all_attributed
    if unattributed < 0:
        raise ObservedDecompositionError("device-timeline-exceeds-boundary")
    return {
        "status": "available",
        "accounting": "device-interval-union",
        "clock_domain": timeline["clock_domain"],
        "e2e_duration_ns": e2e_duration,
        "leaves": leaves,
        "inclusive_parents": parents,
        "reconciliation": {
            "all_attributed_ns": all_attributed,
            "unattributed_ns": unattributed,
            "overlap_ns": overlap,
            "reconciled_e2e_ns": all_attributed + unattributed,
        },
        "source": source,
    }


def compose_observed_decomposition(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Compose paired lanes without allowing diagnostics to replace baseline truth."""

    if document.get("schema") != SCHEDULE_EFFECT_INPUT_SCHEMA:
        raise ObservedDecompositionError("unsupported-schedule-effect-input-schema")
    pair_id = document.get("pair_id")
    identity = _identity(document.get("identity"))
    baseline = document.get("baseline_timing_lane")
    diagnostic = document.get("diagnostic_profiling_lane")
    if (
        not _nonempty(pair_id)
        or not isinstance(baseline, dict)
        or not isinstance(diagnostic, dict)
        or baseline.get("pair_id") != pair_id
        or diagnostic.get("pair_id") != pair_id
        or diagnostic.get("paired_baseline_lane_id") != baseline.get("lane_id")
        or not _nonempty(baseline.get("lane_id"))
        or not _nonempty(diagnostic.get("lane_id"))
        or _identity(baseline.get("identity")) != identity
        or _identity(diagnostic.get("identity")) != identity
        or not _nonempty(baseline.get("instrumentation_profile"))
        or not _nonempty(diagnostic.get("instrumentation_profile"))
        or not _positive_samples(baseline.get("raw_samples_ns"))
        or baseline.get("timing_summary")
        != timing_summary(baseline.get("raw_samples_ns"))
        or not isinstance(baseline.get("normalized_window_samples_ns"), list)
        or len(baseline["normalized_window_samples_ns"])
        != len(baseline["raw_samples_ns"])
        or not isinstance(baseline.get("windows_per_sample"), int)
        or baseline["windows_per_sample"] <= 0
        or not all(
            isinstance(window, list)
            and len(window) == baseline["windows_per_sample"]
            and all(
                isinstance(sample, (int, float))
                and not isinstance(sample, bool)
                and sample > 0
                for sample in window
            )
            for window in baseline["normalized_window_samples_ns"]
        )
        or not isinstance(baseline.get("warmup"), dict)
        or baseline["warmup"].get("outside_timing_boundary") is not True
        or not isinstance(baseline.get("timer"), dict)
        or not _nonempty(baseline["timer"].get("kind"))
        or not _nonempty(baseline.get("synchronization"))
        or not isinstance(baseline.get("correctness"), dict)
        or baseline["correctness"].get("passed") is not True
        or not isinstance(diagnostic.get("instrumentation_timing"), dict)
        or not _nonempty(diagnostic["instrumentation_timing"].get("clock_domain"))
        or not _nonempty(diagnostic["instrumentation_timing"].get("source"))
        or not isinstance(
            diagnostic["instrumentation_timing"].get("elapsed_ns"),
            (int, float),
        )
        or diagnostic["instrumentation_timing"]["elapsed_ns"] <= 0
    ):
        raise ObservedDecompositionError("invalid-paired-measurement-lanes")
    baseline_source = _source(baseline.get("source"))
    qualified, reason_code, ablation = _ablation_decision(
        pair_id, identity, baseline, diagnostic
    )
    if qualified:
        decomposition = _timeline_decomposition(diagnostic.get("device_timeline"))
    else:
        device_timeline_status = diagnostic.get("device_timeline_status")
        evidence_boundaries = [reason_code]
        if isinstance(device_timeline_status, dict) and _nonempty(
            device_timeline_status.get("reason_code")
        ):
            evidence_boundaries.append(device_timeline_status["reason_code"])
        decomposition = {
            "status": "unavailable",
            "reason_code": reason_code,
            "evidence_boundaries": evidence_boundaries,
            "required_next_measurement": (
                "collect an independent exact-identity paired baseline/diagnostic "
                "holdout for the versioned overhead and uncertainty Error Budget; "
                "export a complete same-identity Ascend device timeline"
                if "profiler-device-timeline-export-incomplete"
                in evidence_boundaries
                else "collect an independent exact-identity paired baseline/diagnostic "
                "holdout for the versioned overhead and uncertainty Error Budget"
            ),
            "e2e_duration_ns": None,
            "leaves": [],
            "inclusive_parents": [],
            "reconciliation": {
                "all_attributed_ns": None,
                "unattributed_ns": None,
                "overlap_ns": None,
                "reconciled_e2e_ns": None,
            },
        }
    samples = baseline["raw_samples_ns"]
    return {
        "schema": OBSERVED_DECOMPOSITION_SCHEMA,
        "pair_id": pair_id,
        "identity": identity,
        "baseline_e2e_observation": {
            "status": "valid",
            "kind": "baseline-timing-median",
            "median_ns": median(samples),
            "raw_sample_count": len(samples),
            "timing_summary": deepcopy(baseline["timing_summary"]),
            "normalized_window_samples_ns": deepcopy(
                baseline["normalized_window_samples_ns"]
            ),
            "windows_per_sample": baseline["windows_per_sample"],
            "instrumentation_profile": baseline["instrumentation_profile"],
            "timer": deepcopy(baseline["timer"]),
            "synchronization": baseline["synchronization"],
            "warmup": deepcopy(baseline["warmup"]),
            "correctness": deepcopy(baseline["correctness"]),
            "source": baseline_source,
        },
        "diagnostic_instrumentation_timing": deepcopy(
            diagnostic["instrumentation_timing"]
        ),
        "diagnostic_source": _source(
            diagnostic.get("device_timeline", {}).get("source")
            if isinstance(diagnostic.get("device_timeline"), dict)
            else diagnostic.get("source")
        ),
        "profiling_overhead_ablation": ablation
        or {"status": "unavailable", "reason_code": reason_code},
        "observed_decomposition": decomposition,
    }


__all__ = [
    "OBSERVED_DECOMPOSITION_SCHEMA",
    "SCHEDULE_EFFECT_INPUT_SCHEMA",
    "ObservedDecompositionError",
    "compose_observed_decomposition",
    "timing_summary",
]
