from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import shutil

import pytest

from groundupscale.gap_report import (
    GapReportError,
    compose_gap_report,
    derive_tiered_iteration_report,
    render_gap_report_html,
    write_gap_report_bundle,
)
from groundupscale.run_bundle import verify_run_bundle


def _input() -> dict[str, object]:
    predicted = [
        {
            "stable_path": f"semantic/model/leaf_{index:02d}",
            "operation_class": "MatMul",
            "status": "known",
            "duration_ns": 200 - index * 5,
            "standard_uncertainty_ns": 2,
            "evidence_refs": [f"run://prediction@sha256:{'a' * 64}#p{index}"],
            "accounting_interval": [index * 10, index * 10 + 10],
        }
        for index in range(12)
    ]
    observed = [
        {
            "stable_path": f"semantic/model/leaf_{index:02d}",
            "operation_class": "MatMul",
            "duration_ns": 100 + index * 10,
            "standard_uncertainty_ns": 3,
            "evidence_refs": [f"run://observation@sha256:{'b' * 64}#o{index}"],
            "accounting_interval": [index * 10, index * 10 + 10],
        }
        for index in range(12)
    ]
    return {
        "schema": "groundupscale.dev/e2e-gap-report-input/v1alpha1",
        "identity": {
            "case": "two-layer-prefill",
            "shape": [1, 512, 512],
            "dtype": "float32",
            "candidate_id": "eager-v1",
            "hardware_cohort": "ascend-test",
            "completion_boundary": "device-event-completion",
        },
        "policy": {
            "policy_id": "issue49-materiality-v1",
            "version": "1",
            "top_k": 10,
            "mandatory_share_of_e2e": 0.10,
            "deep_diagnosis": {
                "minimum_absolute_gap_ns": 50,
                "minimum_relative_gap": 0.20,
            },
        },
        "predicted": {
            "identity": {
                "case": "two-layer-prefill", "shape": [1, 512, 512],
                "dtype": "float32",
                "candidate_id": "eager-v1", "hardware_cohort": "ascend-test",
                "completion_boundary": "device-event-completion",
            },
            "status": "known",
            "e2e_duration_ns": 2000,
            "standard_uncertainty_ns": 10,
            "bound_kind": "point-prediction",
            "items": predicted,
            "unattributed_ns": 20,
            "overlap_ns": 90,
            "evidence_refs": [f"run://prediction@sha256:{'a' * 64}"],
        },
        "observed": {
            "identity": {
                "case": "two-layer-prefill", "shape": [1, 512, 512],
                "dtype": "float32",
                "candidate_id": "eager-v1", "hardware_cohort": "ascend-test",
                "completion_boundary": "device-event-completion",
            },
            "status": "available",
            "e2e_duration_ns": 2100,
            "standard_uncertainty_ns": 20,
            "items": observed,
            "unattributed_ns": 280,
            "overlap_ns": 40,
            "accounting": "interval-union",
            "evidence_refs": [f"run://observation@sha256:{'b' * 64}"],
        },
        "source_bundles": [
            {
                "run_id": "prediction", "bundle_kind": "fixture",
                "path": "fixtures/prediction", "manifest_sha256": "a" * 64,
                "verification_passed": True,
            },
            {
                "run_id": "observation", "bundle_kind": "fixture",
                "path": "fixtures/observation", "manifest_sha256": "b" * 64,
                "verification_passed": True,
            },
        ],
    }


def _unavailable_input() -> dict[str, object]:
    document = _input()
    document.pop("source_bundles")
    for side in ("predicted", "observed"):
        document[side] = {  # type: ignore[index]
            "identity": deepcopy(document["identity"]),
            "status": "unavailable" if side == "observed" else "unknown",
            "e2e_duration_ns": None,
            "items": [],
            "reason_code": f"{side}-missing",
            "evidence_boundaries": [f"{side}-missing"],
            "required_next_measurement": f"collect {side}",
            "evidence_refs": [],
        }
    return document


def _tiered_iteration_input() -> dict[str, object]:
    document = _unavailable_input()
    document["schema"] = "groundupscale.dev/e2e-gap-report-input/v1alpha2"
    document["source_bundles"] = [
        {
            "run_id": "synthetic-contract",
            "bundle_kind": "fixture",
            "path": "fixtures/synthetic-contract",
            "manifest_sha256": "c" * 64,
            "verification_passed": True,
        }
    ]
    predicted_items = [
        {
            "stable_path": f"semantic/model/leaf_{index:02d}",
            "operation_class": "MatMul",
            "duration_ns": float(120 - index * 5),
            "evidence_grade": "D",
            "generation_stage": "resource-model",
            "method": "cost-demand-with-conservative-efficiency",
            "uncertainty_interval_ns": [float(60 - index * 2.5), float(240 - index * 10)],
            "evidence_refs": [f"run://prediction@sha256:{'a' * 64}#p{index}"],
            "accounting_interval": [index * 10, index * 10 + 10],
        }
        for index in range(12)
    ]
    predicted_e2e = sum(item["duration_ns"] for item in predicted_items)
    document["iteration_report"] = {
        "policy": {
            "policy_id": "direct-measurement-observation-v2",
            "version": "2",
            "grade_minimum_intervals": {
                "C": [0.70, 1.30],
                "D": [0.50, 2.00],
            },
            "measured_uncertainty": "recorded-sample-statistics-only",
        },
        "predicted": {
            "e2e_duration_ns": predicted_e2e,
            "evidence_grade": "D",
            "generation_stage": "resource-model",
            "method": "cost-demand-with-conservative-efficiency",
            "uncertainty_interval_ns": [predicted_e2e * 0.5, predicted_e2e * 2.0],
            "items": predicted_items,
            "residual": {
                "label": "框架/调度/未归因",
                "duration_ns": 0.0,
            },
        },
        "observed": {
            "e2e_duration_ns": 1920.0,
            "evidence_grade": "B",
            "generation_stage": "baseline-measurement",
            "method": "benchmark-median-with-iqr",
            "uncertainty_interval_ns": [1900.0, 1940.0],
            "component_method": "direct-measurements-only",
            "items": [],
            "residual": {
                "label": "未分解实测残差",
                "duration_ns": 1920.0,
                "evidence_grade": "B",
                "generation_stage": "baseline-measurement",
                "method": "measured-e2e-minus-direct-components",
                "uncertainty_interval_ns": [1900.0, 1940.0],
                "permitted_use": "iteration-baseline-only",
                "evidence_refs": [f"run://observation@sha256:{'b' * 64}"],
            },
            "evidence_refs": [f"run://observation@sha256:{'b' * 64}"],
        },
    }
    return document


def test_compose_selects_each_side_independently_and_joins_exact_union() -> None:
    report = compose_gap_report(_input())

    assert [row["stable_path"] for row in report["predicted"]["top10"]] == [
        f"semantic/model/leaf_{index:02d}" for index in range(10)
    ]
    assert [row["stable_path"] for row in report["observed"]["top10"]] == [
        f"semantic/model/leaf_{index:02d}" for index in range(11, 1, -1)
    ]
    assert {row["stable_path"] for row in report["gap_table"]} == {
        f"semantic/model/leaf_{index:02d}" for index in range(12)
    }
    edge = next(
        row for row in report["gap_table"]
        if row["stable_path"] == "semantic/model/leaf_11"
    )
    assert edge["predicted_rank"] == 12
    assert edge["observed_rank"] == 1
    assert edge["predicted_time_ns"] == 145
    assert edge["observed_time_ns"] == 210
    assert edge["absolute_gap_ns"] == 65
    assert edge["ratio"] == pytest.approx(210 / 145)
    assert edge["predicted_evidence_refs"] == [f"run://prediction@sha256:{'a' * 64}#p11"]
    assert edge["observed_evidence_refs"] == [f"run://observation@sha256:{'b' * 64}#o11"]


def test_only_direct_measurements_enter_observed_values() -> None:
    report = compose_gap_report(_tiered_iteration_input())

    assert report["status"] == "structured-unknown"
    assert report["iteration_status"] == "prediction-and-e2e-measurement-available"
    assert report["report_values"]["predicted"]["e2e_duration_ns"] > 0
    assert report["report_values"]["observed"]["e2e_duration_ns"] == 1920.0
    assert report["report_values"]["predicted"]["evidence_grade"] == "D"
    assert report["report_values"]["observed"]["evidence_grade"] == "B"
    assert len(report["report_values"]["predicted"]["top10"]) == 10
    assert report["report_values"]["observed"]["top10"] == []
    assert report["report_values"]["observed"]["all_items"] == []
    assert report["report_values"]["observed"]["reconciliation"] == {
        "status": "reconciled",
        "e2e_ns": 1920.0,
        "all_components_ns": 0.0,
        "residual_label": "未分解实测残差",
        "residual_ns": 1920.0,
        "residual_evidence_grade": "B",
        "residual_generation_stage": "baseline-measurement",
        "residual_method": "measured-e2e-minus-direct-components",
        "residual_uncertainty_interval_ns": [1900.0, 1940.0],
        "residual_evidence_refs": [f"run://observation@sha256:{'b' * 64}"],
        "overlap_ns": 0.0,
        "share_total": 1.0,
    }
    assert report["report_values"]["predicted"]["reconciliation"]["share_total"] == 1.0
    assert report["report_values"]["observed"]["reconciliation"]["share_total"] == 1.0
    assert report["iteration_gap_table"] == []
    assert len(report["prediction_measurement_priorities"]) == 10
    assert all(
        row["minimum_next_measurement"] == "同 Stable Path 的直接 device timing"
        for row in report["prediction_measurement_priorities"]
    )
    assert report["iteration_metrics"]["comparison_kind"] == "exploratory-gap"
    assert report["iteration_metrics"]["e2e_absolute_gap_ns"] is not None

    html = render_gap_report_html(report)
    assert "<title>两层 Transformer 预测—实测迭代报告</title>" in html
    assert "探索性差异" in html
    assert "预测侧 TOP10" in html
    assert "实测组成（仅直接测量）" in html
    assert "未分解实测残差" in html
    assert "建议优先实测" in html
    assert "实测侧降级估计" not in html
    assert "scale-predicted-weights-to-observed-e2e" not in html
    assert "证据等级与适用范围" in html
    assert "模块汇总" in html
    assert "不确定区间" in html
    assert "第1层 / 注意力" in html or "其他组件" in html
    assert "框架/调度/未归因" in html

    promoted = _tiered_iteration_input()
    promoted["iteration_report"]["predicted"]["permitted_use"] = (  # type: ignore[index]
        "acceptance-and-calibration"
    )
    with pytest.raises(GapReportError, match="grade-permitted-use"):
        compose_gap_report(promoted)


def test_reconciliation_metrics_and_diagnosis_are_policy_gated() -> None:
    report = compose_gap_report(_input())

    assert report["predicted"]["reconciliation"] == {
        "status": "reconciled",
        "e2e_ns": 2000.0,
        "selected_ns": 1775.0,
        "all_attributed_ns": 2070.0,
        "other_ns": 295.0,
        "unattributed_ns": 20.0,
        "overlap_ns": 90.0,
        "accounted_e2e_ns": 2000.0,
        "residual_ns": 0.0,
    }
    assert report["metrics"]["e2e_absolute_gap_ns"] == 100
    assert report["metrics"]["e2e_ratio"] == pytest.approx(1.05)
    assert report["metrics"]["combined_uncertainty_ns"] == pytest.approx(
        (10**2 + 20**2) ** 0.5
    )
    assert report["metrics"]["frontier_efficiency"] == pytest.approx(2000 / 2100)
    assert report["metrics"]["relative_prediction_error"] == pytest.approx(100 / 2100)
    assert report["diagnosis"]["triggered"] == []
    assert report["drilldown"]["kind"] == "evidence-boundary"
    assert report["drilldown"]["evidence_boundary"] == (
        "diagnostic-classification-evidence-missing"
    )


def test_lower_bound_and_unavailable_side_fail_closed_without_fake_scores() -> None:
    document = _input()
    document["predicted"]["bound_kind"] = "lower-bound"  # type: ignore[index]
    document["observed"] = {
        "identity": deepcopy(document["identity"]),
        "status": "unavailable",
        "e2e_duration_ns": None,
        "items": [],
        "reason_code": "profiling-overhead-ablation-missing",
        "evidence_boundaries": ["exact-identity-profiling-ablation-missing"],
        "required_next_measurement": "collect exact-identity paired holdout",
        "evidence_refs": ["run://observation"],
    }

    report = compose_gap_report(document)

    assert report["observed"]["selected"] == []
    assert {row["stable_path"] for row in report["gap_table"]} == {
        row["stable_path"] for row in report["predicted"]["selected"]
    }
    assert all(row["observed_time_ns"] is None for row in report["gap_table"])
    assert report["metrics"]["relative_prediction_error"] is None
    assert report["metrics"]["frontier_efficiency"] is None
    assert report["diagnosis"]["status"] == "unavailable"
    assert report["drilldown"] == {
        "kind": "evidence-boundary",
        "stable_path": None,
        "evidence_boundaries": {
            "predicted": [],
            "observed": ["exact-identity-profiling-ablation-missing"],
        },
        "required_next_measurement": {
            "predicted": None,
            "observed": "collect exact-identity paired holdout",
        },
    }
    assert "exact-identity-profiling-ablation-missing" in render_gap_report_html(report)
    assert "exact-identity-profiling-ablation-missing" in render_gap_report_html(report)


def test_bundle_is_immutable_replayable_and_tamper_detected(tmp_path: Path) -> None:
    document = _unavailable_input()
    run = write_gap_report_bundle(
        tmp_path,
        run_id="issue49-test-gap-report-v1",
        document=document,
    )

    assert verify_run_bundle(run)["passed"] is True
    manifest = json.loads((run / "run.manifest.json").read_text())
    assert {artifact["role"] for artifact in manifest["artifacts"]} == {
        "e2e-gap-report-input",
        "e2e-gap-report",
        "html-report",
    }
    with pytest.raises(FileExistsError):
        write_gap_report_bundle(
            tmp_path,
            run_id="issue49-test-gap-report-v1",
            document=document,
        )

    report_path = run / "comparison/e2e-gap-report.json"
    tampered = json.loads(report_path.read_text())
    tampered["metrics"]["e2e_absolute_gap_ns"] = 1
    report_path.write_text(json.dumps(tampered), encoding="utf-8")
    verification = verify_run_bundle(run)
    assert verification["passed"] is False
    assert "digest mismatch: comparison/e2e-gap-report.json" in verification["failures"]


def test_tiered_bundle_requires_locked_replay_contract(tmp_path: Path) -> None:
    with pytest.raises(GapReportError, match="replay-contract-required"):
        write_gap_report_bundle(
            tmp_path,
            run_id="issue49-test-tiered-gap-report-v2",
            document=_tiered_iteration_input(),
        )


def test_published_tiered_values_replay_frozen_cost_and_observation_sources(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    builder_path = (
        root
        / "goal_process/issue-49-e2e-gap-report/build_gap_report_bundle.py"
    )
    spec = importlib.util.spec_from_file_location("issue49_gap_builder", builder_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    document = module.build_document()

    replayed = derive_tiered_iteration_report(document, root)
    assert document["iteration_report"] == replayed
    assert len(replayed["predicted"]["items"]) == 52
    assert replayed["predicted"]["evidence_grade"] == "D"
    assert replayed["observed"]["evidence_grade"] == "B"
    assert replayed["observed"]["e2e_duration_ns"] == 1_921_530.0
    assert replayed["observed"]["uncertainty_interval_ns"] == [
        1_911_720.0,
        1_924_885.0,
    ]
    assert replayed["observed"]["component_method"] == "direct-measurements-only"
    assert replayed["observed"]["items"] == []
    assert replayed["observed"]["residual"]["duration_ns"] == 1_921_530.0

    tampered = deepcopy(document)
    tampered["iteration_report"]["predicted"]["items"][0]["duration_ns"] += 1  # type: ignore[index]
    with pytest.raises(GapReportError, match="source-replay-mismatch"):
        write_gap_report_bundle(
            tmp_path,
            run_id="issue49-tampered-tiered-report-v1",
            document=tampered,
        )

    wrong_identity = deepcopy(document)
    wrong_identity["identity"]["hardware_cohort"] = "forged-cohort"  # type: ignore[index]
    with pytest.raises(GapReportError, match="identity-mismatch"):
        derive_tiered_iteration_report(wrong_identity, root)

    wrong_policy = deepcopy(document)
    wrong_policy["iteration_report_derivation"]["observation_component_model"] = {  # type: ignore[index]
        "policy_id": "lie",
        "version": "999",
        "purpose": "acceptance",
    }
    with pytest.raises(GapReportError, match="observation-component-model"):
        derive_tiered_iteration_report(wrong_policy, root)

    forged_measured_component = deepcopy(document)
    forged_measured_component["iteration_report"]["observed"]["items"] = [  # type: ignore[index]
        {
            **forged_measured_component["iteration_report"]["predicted"]["items"][0],  # type: ignore[index]
            "evidence_grade": "D",
            "generation_stage": "diagnostic-attribution",
        }
    ]
    with pytest.raises(GapReportError, match="source-replay-mismatch"):
        write_gap_report_bundle(
            tmp_path,
            run_id="issue49-forged-observed-component-v2",
            document=forged_measured_component,
        )

    forged_authority = deepcopy(document)
    forged_authority["predicted"]["items"][0]["stable_path"] = "forged/path"  # type: ignore[index]
    with pytest.raises(GapReportError, match="predicted-authority-replay"):
        derive_tiered_iteration_report(forged_authority, root)

    forged_provenance = deepcopy(document)
    forged_provenance["predicted"]["evidence_refs"] = ["run://forged"]  # type: ignore[index]
    with pytest.raises(GapReportError, match="predicted-authority-replay"):
        derive_tiered_iteration_report(forged_provenance, root)

    suppressed_measurement = deepcopy(document)
    suppressed_measurement["observed"]["required_next_measurement"] = "无需补测"  # type: ignore[index]
    with pytest.raises(GapReportError, match="observed-authority-replay"):
        derive_tiered_iteration_report(suppressed_measurement, root)


def test_rejects_inclusive_parent_mixed_with_descendant_as_additive_item() -> None:
    document = _input()
    document["predicted"]["items"].append(  # type: ignore[index,union-attr]
        {
            "stable_path": "semantic/model",
            "operation_class": "Transformer",
            "status": "known",
            "duration_ns": 2000,
            "inclusive": True,
            "accounting_interval": [120, 130],
            "evidence_refs": ["run://prediction/parent"],
        }
    )

    with pytest.raises(GapReportError, match="inclusive-parent-is-navigation-only"):
        compose_gap_report(document)


def test_non_reconciling_side_fails_closed_and_missing_uncertainty_never_triggers() -> None:
    document = _input()
    document["predicted"]["unattributed_ns"] = None  # type: ignore[index]
    document["predicted"]["items"][0]["standard_uncertainty_ns"] = None  # type: ignore[index]

    report = compose_gap_report(document)

    assert report["predicted"]["reconciliation"]["status"] == "unknown"
    assert report["status"] == "structured-unknown"
    row = next(
        row for row in report["gap_table"]
        if row["stable_path"] == "semantic/model/leaf_00"
    )
    assert row["combined_uncertainty_ns"] is None
    assert row["diagnosis_eligible"] is False
    assert all(
        item["stable_path"] != "semantic/model/leaf_00"
        for item in report["diagnosis"]["triggered"]
    )


def test_side_identity_exclusivity_scope_drilldown_and_evidence_classification() -> None:
    document = _input()
    document["observed"]["identity"]["shape"] = [1, 1, 1]  # type: ignore[index]
    with pytest.raises(GapReportError, match="side-identity-mismatch"):
        compose_gap_report(document)

    document = _input()
    document["predicted"]["items"][0]["accounting_id"] = "shared"  # type: ignore[index]
    document["predicted"]["items"][1]["accounting_id"] = "shared"  # type: ignore[index]
    with pytest.raises(GapReportError, match="non-mutually-exclusive-items"):
        compose_gap_report(document)

    document = _input()
    document["predicted"]["items"][0]["accounting_interval"] = [5, 15]  # type: ignore[index]
    with pytest.raises(GapReportError, match="non-mutually-exclusive-items"):
        compose_gap_report(document)

    document = _input()
    document["scopes"] = [
        {
            "stable_path": "semantic/model",
            "kind": "inclusive-navigation",
            "children": [f"semantic/model/leaf_{index:02d}" for index in range(12)],
            "children_accounting": "non-overlapping",
            "evidence_refs": [f"run://prediction@sha256:{'a' * 64}"],
        }
    ]
    document["diagnostic_evidence"] = {
        "semantic/model/leaf_11": {
            "classification": "implementation-headroom",
            "reason_code": "qualified-frontier-vs-observed-candidate",
            "evidence_refs": [
                f"run://prediction@sha256:{'a' * 64}#p11",
                f"run://observation@sha256:{'b' * 64}#o11",
            ],
        }
    }
    report = compose_gap_report(document)
    assert report["drilldown"]["navigation_scope"] == "semantic/model"
    assert len(report["drilldown"]["non_overlapping_children"]) == 12
    classified = next(
        row for row in report["diagnosis"]["triggered"]
        if row["stable_path"] == "semantic/model/leaf_11"
    )
    assert classified["classification"] == "implementation-headroom"
    assert classified["classification_evidence_refs"] == [
        f"run://prediction@sha256:{'a' * 64}#p11",
        f"run://observation@sha256:{'b' * 64}#o11",
    ]


def test_published_authority_recursively_verifies_locked_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    run = (
        root
        / "goal_process/issue-49-e2e-gap-report/evidence/runs"
        / "issue49-20260814T0345Z-e2e-gap-report-v6"
    )

    verification = verify_run_bundle(run)

    assert verification["passed"] is True
    assert sha256((run / "run.manifest.json").read_bytes()).hexdigest() == (
        "d582877c0f095e5a8a918a28e2888c71ce1b28ea7746738cac8f62fbc7f1ea10"
    )
    assert sha256((run / "reports/report.html").read_bytes()).hexdigest() == (
        "36fdc7a22cfafd7065577e582a11948204fa44eb3e786e78c0f8f9f71659b2c4"
    )
    manifest = json.loads((run / "run.manifest.json").read_text())
    assert [source["run_id"] for source in manifest["source_bundles"]] == [
        "issue48-20260814T0002Z-schedule-frontier-unknown-v2",
        "issue47-ascend-observed-decomposition-20260813-v1",
    ]
    assert all(source["verification_passed"] for source in manifest["source_bundles"])


def test_published_chinese_iteration_report_has_complete_numeric_components() -> None:
    root = Path(__file__).resolve().parents[1]
    run = (
        root
        / "goal_process/issue-49-e2e-gap-report/evidence/runs"
        / "issue49-20260814T0730Z-e2e-gap-report-v12"
    )

    assert verify_run_bundle(run)["passed"] is True
    manifest = json.loads((run / "run.manifest.json").read_text())
    assert {artifact["role"] for artifact in manifest["artifacts"]} == {
        "e2e-gap-report-input",
        "e2e-gap-report",
        "e2e-components-csv",
        "html-report",
    }
    report = json.loads(
        (run / "comparison/e2e-gap-report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "structured-unknown"
    assert report["iteration_status"] == "numeric-report-values-available"
    assert report["report_values"]["predicted"]["e2e_duration_ns"] > 0
    assert report["report_values"]["observed"]["e2e_duration_ns"] == 1_921_530.0
    assert len(report["report_values"]["predicted"]["all_items"]) == 52
    assert len(report["report_values"]["observed"]["all_items"]) == 52
    assert all(
        item["duration_ns"] is not None
        for side in report["report_values"].values()
        for item in side["all_items"]
    )
    assert len(
        (run / "comparison/e2e-components.csv").read_text().splitlines()
    ) == 53
    assert (run / "comparison/e2e-components.csv").read_text().splitlines()[
        0
    ].startswith(
        "stable_path,operation_class,selected_in_top10_union,predicted_time_ns"
    )
    html_before_payload = (
        (run / "reports/report.html")
        .read_text(encoding="utf-8")
        .split('<script type="application/json"', 1)[0]
    )
    assert "实测侧 TOP10（降级估计）" in html_before_payload
    assert ">unavailable<" not in html_before_payload
    assert ">unknown<" not in html_before_payload

    forged_run = run.parent / ".pytest-v12-missing-manifest-sources"
    assert not forged_run.exists()
    try:
        shutil.copytree(run, forged_run)
        forged_manifest_path = forged_run / "run.manifest.json"
        forged_manifest = json.loads(forged_manifest_path.read_text())
        forged_manifest.pop("source_bundles")
        forged_manifest_path.write_text(json.dumps(forged_manifest))
        verification = verify_run_bundle(forged_run)
        assert verification["passed"] is False
        assert "E2E gap report source lineage mismatch" in verification["failures"]
    finally:
        shutil.rmtree(forged_run, ignore_errors=True)


def test_published_direct_measurement_report_never_estimates_observed_components() -> None:
    root = Path(__file__).resolve().parents[1]
    run = (
        root
        / "goal_process/issue-49-e2e-gap-report/evidence/runs"
        / "issue49-20260817T025339Z-e2e-gap-report-v13"
    )

    assert verify_run_bundle(run)["passed"] is True
    report = json.loads(
        (run / "comparison/e2e-gap-report.json").read_text(encoding="utf-8")
    )
    observed = report["report_values"]["observed"]
    assert report["schema"] == "groundupscale.dev/e2e-gap-report/v1alpha3"
    assert observed["e2e_duration_ns"] == 1_921_530.0
    assert observed["uncertainty_interval_ns"] == [1_911_720.0, 1_924_885.0]
    assert observed["component_method"] == "direct-measurements-only"
    assert observed["all_items"] == []
    assert observed["top10"] == []
    assert observed["reconciliation"]["residual_ns"] == 1_921_530.0
    assert observed["reconciliation"]["share_total"] == 1.0
    assert report["iteration_gap_table"] == []
    assert len(report["prediction_measurement_priorities"]) == 10
    html = (run / "reports/report.html").read_text(encoding="utf-8")
    assert "未分解实测残差" in html
    assert "实测侧降级估计" not in html
    assert "scale-predicted-weights-to-observed-e2e" not in html
    csv_text = (run / "comparison/e2e-components.csv").read_text(encoding="utf-8")
    assert csv_text.count("not-measured") == 52
    assert csv_text.count("measured-residual") == 1


def test_module_cli_publishes_same_machine_and_human_projection(tmp_path: Path) -> None:
    document = _unavailable_input()
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(document), encoding="utf-8")

    subprocess.run(
        [
            "python",
            "-m",
            "groundupscale.gap_report",
            str(input_path),
            "--artifact-store",
            str(tmp_path),
            "--run-id",
            "issue49-cli-contract-v1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert verify_run_bundle(tmp_path / "runs/issue49-cli-contract-v1")["passed"] is True


def test_two_layer_demo_one_click_builder_allocates_unique_run_id() -> None:
    root = Path(__file__).resolve().parents[1]
    run: Path | None = None
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    root
                    / "goal_process/issue-49-e2e-gap-report/build_gap_report_bundle.py"
                ),
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

        run = Path(completed.stdout.strip())
        assert run.parent == (
            root / "goal_process/issue-49-e2e-gap-report/evidence/runs"
        ).resolve()
        assert run.name.startswith("issue49-")
        assert verify_run_bundle(run)["passed"] is True
        report = json.loads(
            (run / "comparison/e2e-gap-report.json").read_text(encoding="utf-8")
        )
        assert report["schema"] == "groundupscale.dev/e2e-gap-report/v1alpha3"
        assert len(report["report_values"]["predicted"]["all_items"]) == 52
        assert report["report_values"]["observed"]["all_items"] == []
        assert report["report_values"]["observed"]["reconciliation"][
            "residual_ns"
        ] == 1_921_530.0
        html = (run / "reports/report.html").read_text(encoding="utf-8")
        visible_html = html.split('<script type="application/json"', 1)[0]
        assert "两层 Transformer 预测—实测迭代报告" in html
        assert "实测组成（仅直接测量）" in html
        assert "实测侧降级估计" not in html
        assert "scale-predicted-weights-to-observed-e2e" not in html
        assert "基线样本中位数与四分位区间" in visible_html
        assert "实测 E2E 减去直接实测组件" in visible_html
        assert "benchmark-median-with-iqr" not in visible_html
        assert "measured-e2e-minus-direct-components" not in visible_html
        csv_rows = (run / "comparison/e2e-components.csv").read_text().splitlines()
        assert len(csv_rows) == 54
        assert sum("not-measured" in row for row in csv_rows[1:]) == 52
        residual_row = next(row for row in csv_rows[1:] if "measured-residual" in row)
        assert "MeasuredE2EResidual" in residual_row
        assert "1921530.0" in residual_row
    finally:
        if run is not None and run.is_dir():
            shutil.rmtree(run)
