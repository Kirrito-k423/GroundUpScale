"""Shared fail-closed validators for exact-Shape Frontier evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from statistics import median, quantiles
from typing import Any

from groundupscale.ir import content_fingerprint
from groundupscale.schemas.v1alpha1 import ExactMatmulCorrectnessEvidence


EXACT_FRONTIER_TIMING_POLICY = {
    "policy_id": "local-m4-exact-shape-timing-v1",
    "version": "1.0.0",
    "minimum_warmup_iterations": 500,
    "convergence_window_count": 7,
    "convergence_iterations_per_window": 20,
    "maximum_warmup_median_drift": 0.05,
    "minimum_samples": 20,
    "minimum_windows_per_sample": 5,
    "minimum_timed_duration_ns": 100_000_000,
}


@dataclass(frozen=True)
class ExactTimingValidity:
    qualified: bool
    reason_codes: tuple[str, ...]
    samples_ns: tuple[float, ...] = ()
    median_ns: float | None = None
    q1_ns: float | None = None
    q3_ns: float | None = None
    iqr_over_median: float | None = None
    timed_duration_ns: float | None = None
    timer_resolution_ns: float | None = None
    warmup_iterations: int | None = None
    warmup_window_samples_ns: tuple[float, ...] = ()
    warmup_median_drift: float | None = None


def embedded_digest_is_valid(document: object, field: str) -> bool:
    if not isinstance(document, dict):
        return False
    authored = document.get(field)
    if not isinstance(authored, str) or len(authored) != 64:
        return False
    body = {key: value for key, value in document.items() if key != field}
    return authored == content_fingerprint(body)


def validate_exact_matmul_correctness(
    record: object,
    *,
    candidate: object,
    input_corpus: object,
    execution: object,
) -> ExactMatmulCorrectnessEvidence | None:
    if (
        not isinstance(record, dict)
        or not isinstance(candidate, dict)
        or not isinstance(input_corpus, dict)
        or not isinstance(execution, dict)
        or record.get("candidate_identity") != candidate
        or record.get("input_corpus") != input_corpus
        or record.get("execution_contract") != execution
    ):
        return None
    try:
        evidence = ExactMatmulCorrectnessEvidence.model_validate(
            record.get("correctness")
        )
    except ValueError:
        return None
    if (
        evidence.candidate_family != candidate.get("family")
        or evidence.candidate_digest != candidate.get("candidate_digest")
        or evidence.input_corpus_digest
        != input_corpus.get("input_corpus_digest")
    ):
        return None
    return evidence


def validate_exact_timing_evidence(case: dict[str, Any]) -> ExactTimingValidity:
    latency = case.get("latency")
    timing = case.get("timing_contract")
    warmup = case.get("warmup_convergence")
    if not all(isinstance(value, dict) for value in (latency, timing, warmup)):
        return ExactTimingValidity(False, ("timing-contract-missing",))
    assert isinstance(latency, dict)
    assert isinstance(timing, dict)
    assert isinstance(warmup, dict)
    policy = timing.get("policy")
    timer = timing.get("timer")
    samples = latency.get("samples_ns")
    raw_windows = latency.get("window_samples_ns")
    normalized_windows = latency.get("normalized_window_samples_ns")
    inner_iterations = latency.get("inner_iterations")
    warmup_windows = warmup.get("window_samples_ns")
    sample_count = case.get("samples")
    if (
        policy != EXACT_FRONTIER_TIMING_POLICY
        or warmup.get("policy") != policy
        or timing.get("timing_scope") != "host_visible_completion"
        or timing.get("completion_boundary") != "synchronous-cpu-call-return"
        or timing.get("instrumentation_profile") != "benchmark"
        or timing.get("exclusions") != []
        or not isinstance(timer, dict)
        or timer.get("source") != "time.perf_counter_ns"
        or timer.get("monotonic") is not True
        or not isinstance(timer.get("resolution_ns"), (int, float))
        or not isfinite(float(timer["resolution_ns"]))
        or float(timer["resolution_ns"]) <= 0
        or not isinstance(sample_count, int)
        or sample_count < EXACT_FRONTIER_TIMING_POLICY["minimum_samples"]
        or not isinstance(samples, list)
        or len(samples) != sample_count
        or not isinstance(raw_windows, list)
        or len(raw_windows) != sample_count
        or not isinstance(normalized_windows, list)
        or len(normalized_windows) != sample_count
        or not isinstance(inner_iterations, int)
        or inner_iterations <= 0
        or not isinstance(warmup.get("warmup_iterations"), int)
        or warmup["warmup_iterations"]
        < EXACT_FRONTIER_TIMING_POLICY["minimum_warmup_iterations"]
        or not isinstance(warmup_windows, list)
        or len(warmup_windows)
        < EXACT_FRONTIER_TIMING_POLICY["convergence_window_count"]
        or warmup.get("converged") is not True
    ):
        return ExactTimingValidity(False, ("timing-policy-or-shape-invalid",))
    sample_values: list[float] = []
    total_timed_duration = 0.0
    minimum_windows = EXACT_FRONTIER_TIMING_POLICY["minimum_windows_per_sample"]
    authored_windows_per_sample = latency.get("windows_per_sample")
    for authored_sample, raw, normalized in zip(
        samples, raw_windows, normalized_windows, strict=True
    ):
        if (
            not isinstance(authored_sample, (int, float))
            or not isinstance(raw, list)
            or not isinstance(normalized, list)
            or len(raw) < minimum_windows
            or len(normalized) != len(raw)
            or authored_windows_per_sample != len(raw)
        ):
            return ExactTimingValidity(False, ("timing-raw-windows-invalid",))
        raw_values = [float(value) for value in raw]
        normalized_values = [float(value) for value in normalized]
        if any(
            not isfinite(value) or value <= 0
            for value in (*raw_values, *normalized_values)
        ):
            return ExactTimingValidity(False, ("timing-nonfinite",))
        if any(
            not isclose(raw_value / inner_iterations, normalized_value)
            for raw_value, normalized_value in zip(
                raw_values, normalized_values, strict=True
            )
        ):
            return ExactTimingValidity(False, ("timing-normalization-mismatch",))
        derived_sample = float(median(normalized_values))
        if not isclose(float(authored_sample), derived_sample):
            return ExactTimingValidity(False, ("timing-sample-derivation-mismatch",))
        sample_values.append(derived_sample)
        total_timed_duration += sum(raw_values)
    if total_timed_duration < EXACT_FRONTIER_TIMING_POLICY["minimum_timed_duration_ns"]:
        return ExactTimingValidity(False, ("timing-duration-insufficient",))
    sample_quartiles = quantiles(sample_values, n=4, method="inclusive")
    sample_median = float(median(sample_values))
    iqr_ratio = float((sample_quartiles[2] - sample_quartiles[0]) / sample_median)
    if (
        iqr_ratio > 0.03
        or not isclose(float(latency.get("median_ns", -1)), sample_median)
        or not isclose(float(latency.get("q1_ns", -1)), float(sample_quartiles[0]))
        or not isclose(float(latency.get("q3_ns", -1)), float(sample_quartiles[2]))
        or not isclose(float(latency.get("iqr_over_median", -1)), iqr_ratio)
    ):
        return ExactTimingValidity(False, ("timing-statistics-invalid",))
    warmup_values = [float(value) for value in warmup_windows]
    if any(not isfinite(value) or value <= 0 for value in warmup_values):
        return ExactTimingValidity(False, ("warmup-windows-invalid",))
    warmup_first = float(median(warmup_values[:3]))
    warmup_last = float(median(warmup_values[-3:]))
    warmup_drift = abs(warmup_last - warmup_first) / warmup_last
    if (
        warmup_drift > EXACT_FRONTIER_TIMING_POLICY["maximum_warmup_median_drift"]
        or not isinstance(warmup.get("median_drift"), (int, float))
        or not isclose(float(warmup["median_drift"]), warmup_drift)
    ):
        return ExactTimingValidity(False, ("warmup-convergence-invalid",))
    return ExactTimingValidity(
        qualified=True,
        reason_codes=(),
        samples_ns=tuple(sample_values),
        median_ns=sample_median,
        q1_ns=float(sample_quartiles[0]),
        q3_ns=float(sample_quartiles[2]),
        iqr_over_median=iqr_ratio,
        timed_duration_ns=total_timed_duration,
        timer_resolution_ns=float(timer["resolution_ns"]),
        warmup_iterations=int(warmup["warmup_iterations"]),
        warmup_window_samples_ns=tuple(warmup_values),
        warmup_median_drift=warmup_drift,
    )
