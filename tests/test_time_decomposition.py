from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from groundupscale.benchmark.decomposition import build_latency_decomposition
from groundupscale.benchmark.trace import TraceRunner
from groundupscale.pipeline import compile_analysis_plan


REPOSITORY_ROOT = Path(__file__).parents[1]
Q_PROJ_STABLE_PATH = (
    "semantic/workload/transformer-prefill/request/model-prefill/model/"
    "transformer/layer_0/attention/q_proj"
)


def test_latency_decomposition_enforces_independent_top10_and_visibility_rule() -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )
    assert compiled.hardware_prediction is not None
    trace = TraceRunner(
        compiled.bundle,
        compiled.semantic.semantic_ir,
    ).run()

    decomposition = build_latency_decomposition(
        compiled.hardware_prediction,
        trace,
    )

    predicted = decomposition["predicted"]
    observed = decomposition["observed"]
    assert predicted["available"] is False
    assert predicted["reason"] == "selected hardware floor is unavailable"
    assert predicted["top10"] == []
    assert observed["statistic"] == "single-diagnostic-trace"
    assert len(observed["top10"]) == 10
    for side in (observed,):
        threshold = side["e2e_ns"] * 0.1
        mandatory_paths = {item["stable_path"] for item in side["mandatory"]}
        assert {
            item["stable_path"]
            for item in side["all_items"]
            if item["time_ns"] >= threshold
        } <= mandatory_paths
        assert len(side["selected"]) >= 10

    assert observed["reconciliation"]["attributed_interval_union_ns"] <= (
        observed["e2e_ns"]
    )
    assert observed["reconciliation"]["coverage"] == pytest.approx(1.0)
    expected_joined_paths = {
        item["stable_path"] for item in predicted["selected"]
    } | {item["stable_path"] for item in observed["selected"]}
    assert {item["stable_path"] for item in decomposition["joined"]} == (
        expected_joined_paths
    )
    assert decomposition["largest_discrepancy"] is None


def test_exploratory_top10_prefers_an_exact_stable_path_frontier() -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )
    prediction = compiled.hardware_prediction
    assert prediction is not None
    exact_q_proj = next(
        candidate
        for candidate in prediction.candidates
        if candidate.stable_path.endswith("/layer_0/attention/q_proj")
    )
    assert exact_q_proj.duration.operator_frontier_match_status == "exact-anchor"
    assert exact_q_proj.duration.operator_achievable_frontier_ns is not None
    provisional_total = sum(
        candidate.duration.provisional_estimate_ns or 0.0
        for candidate in prediction.candidates
    )
    exploratory_prediction = replace(
        prediction,
        program_bounds=replace(
            prediction.program_bounds,
            provisional_estimate_ns=provisional_total,
            provisional_evidence_tier="exploratory",
            provisional_reason_codes=("phase-capabilities-incomplete",),
        ),
    )
    trace = TraceRunner(
        compiled.bundle,
        compiled.semantic.semantic_ir,
    ).run()

    decomposition = build_latency_decomposition(
        exploratory_prediction,
        trace,
        frontier_observation_by_path={
            Q_PROJ_STABLE_PATH: {
                "status": "not-evaluable-observation-domain",
                "reason_codes": ["operator-frontier-observation-timing-unqualified"],
            }
        },
    )

    q_proj = next(
        item
        for item in decomposition["predicted"]["all_items"]
        if item["stable_path"].endswith("/layer_0/attention/q_proj")
    )
    assert q_proj["time_ns"] == pytest.approx(
        exact_q_proj.duration.operator_achievable_frontier_ns
    )
    assert q_proj["evidence"] == "exact-operator-frontier"
    assert q_proj["frontier_anchor_id"] == (
        exact_q_proj.duration.operator_frontier_anchor_id
    )
    joined_q_proj = next(
        item
        for item in decomposition["joined"]
        if item["stable_path"].endswith("/layer_0/attention/q_proj")
    )
    assert joined_q_proj["predicted_evidence"] == "exact-operator-frontier"
    assert joined_q_proj["evidence_quality"] == (
        "exact-operator-frontier+unqualified-current-benchmark+"
        "single-diagnostic-trace"
    )
    assert joined_q_proj["frontier_observation_status"] == (
        "not-evaluable-observation-domain"
    )
    assert joined_q_proj["frontier_observation_reason_codes"] == [
        "operator-frontier-observation-timing-unqualified"
    ]
    assert decomposition["predicted"]["kind"] == (
        "mixed-exact-frontier-and-provisional-estimate"
    )
    assert decomposition["comparison_role"] == "exploratory-planning-only"
    assert decomposition["largest_discrepancy"] is None
