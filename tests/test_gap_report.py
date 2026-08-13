from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from groundupscale.gap_report import (
    GapReportError,
    compose_gap_report,
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
            "evidence_refs": [f"run://prediction/p{index}"],
        }
        for index in range(12)
    ]
    observed = [
        {
            "stable_path": f"semantic/model/leaf_{index:02d}",
            "operation_class": "MatMul",
            "duration_ns": 100 + index * 10,
            "standard_uncertainty_ns": 3,
            "evidence_refs": [f"run://observation/o{index}"],
        }
        for index in range(12)
    ]
    return {
        "schema": "groundupscale.dev/e2e-gap-report-input/v1alpha1",
        "identity": {
            "case": "two-layer-prefill",
            "shape": [1, 512, 512],
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
            "evidence_refs": ["run://prediction"],
        },
        "observed": {
            "identity": {
                "case": "two-layer-prefill", "shape": [1, 512, 512],
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
            "evidence_refs": ["run://observation"],
        },
        "source_bundles": [
            {"run_id": "prediction"},
            {"run_id": "observation"},
        ],
    }


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
    assert edge["predicted_evidence_refs"] == ["run://prediction/p11"]
    assert edge["observed_evidence_refs"] == ["run://observation/o11"]


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
    assert report["drilldown"]["kind"] == "none"


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
    run = write_gap_report_bundle(
        tmp_path,
        run_id="issue49-test-gap-report-v1",
        document=_input(),
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
            document=_input(),
        )

    report_path = run / "comparison/e2e-gap-report.json"
    tampered = json.loads(report_path.read_text())
    tampered["metrics"]["e2e_absolute_gap_ns"] += 1
    report_path.write_text(json.dumps(tampered), encoding="utf-8")
    verification = verify_run_bundle(run)
    assert verification["passed"] is False
    assert "digest mismatch: comparison/e2e-gap-report.json" in verification["failures"]


def test_rejects_inclusive_parent_mixed_with_descendant_as_additive_item() -> None:
    document = _input()
    document["predicted"]["items"].append(  # type: ignore[index,union-attr]
        {
            "stable_path": "semantic/model",
            "operation_class": "Transformer",
            "status": "known",
            "duration_ns": 2000,
            "inclusive": True,
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
    document["scopes"] = [
        {
            "stable_path": "semantic/model",
            "kind": "inclusive-navigation",
            "children": [f"semantic/model/leaf_{index:02d}" for index in range(12)],
            "children_accounting": "non-overlapping",
            "evidence_refs": ["run://prediction"],
        }
    ]
    document["diagnostic_evidence"] = {
        "semantic/model/leaf_11": {
            "classification": "implementation-headroom",
            "reason_code": "qualified-frontier-vs-observed-candidate",
            "evidence_refs": ["run://prediction/p11", "run://observation/o11"],
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
        "run://prediction/p11", "run://observation/o11"
    ]


def test_published_authority_recursively_verifies_locked_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    run = (
        root
        / "goal_process/issue-49-e2e-gap-report/evidence/runs"
        / "issue49-20260814T0245Z-e2e-gap-report-v4"
    )

    verification = verify_run_bundle(run)

    assert verification["passed"] is True
    manifest = json.loads((run / "run.manifest.json").read_text())
    assert [source["run_id"] for source in manifest["source_bundles"]] == [
        "issue48-20260814T0002Z-schedule-frontier-unknown-v2",
        "issue47-ascend-observed-decomposition-20260813-v1",
    ]
    assert all(source["verification_passed"] for source in manifest["source_bundles"])


def test_module_cli_publishes_same_machine_and_human_projection(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(_input()), encoding="utf-8")

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
