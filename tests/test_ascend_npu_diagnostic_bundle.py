from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import sys

import pytest

from groundupscale.diagnostics import (
    DiagnosticBundleIntegrityError,
    diagnose_run_bundle,
    render_diagnostic_report,
)
from groundupscale.run_bundle import verify_run_bundle


REPOSITORY_ROOT = Path(__file__).parents[1]
AUTHORITATIVE_BUNDLE = (
    REPOSITORY_ROOT
    / "goal_process"
    / "issue-32-ascend-diagnostic-bundle"
    / "evidence"
    / "runs"
    / "issue32-ascend-910b2-diagnostic-v1"
)


def test_real_ascend_bundle_replays_four_axes_and_evidence_qualified_verdicts(
) -> None:
    verification = verify_run_bundle(AUTHORITATIVE_BUNDLE)
    assert verification["passed"] is True
    assert verification["failures"] == []
    assert verification["artifact_count"] > 20

    result = diagnose_run_bundle(AUTHORITATIVE_BUNDLE)
    axes = result["axes"]
    assert set(axes) == {
        "resource_physical_floor",
        "operator_achievable_frontier",
        "schedule_achievable_frontier",
        "observation",
    }
    assert {axis["status"] for axis in axes.values()} == {"known"}
    assert axes["resource_physical_floor"]["may_be_unattainable"] is True
    assert axes["resource_physical_floor"]["value_ns"] == pytest.approx(
        13_998.515,
        rel=1e-6,
    )
    assert axes["operator_achievable_frontier"]["value_ns"] == pytest.approx(
        16_331.5,
    )
    assert axes["schedule_achievable_frontier"]["value_ns"] >= axes[
        "operator_achievable_frontier"
    ]["value_ns"]
    assert axes["observation"]["value_ns"] > axes[
        "schedule_achievable_frontier"
    ]["value_ns"]
    assert result["comparisons"]["physical_floor_to_observation"][
        "prediction_error_ns"
    ] is None
    assert result["adapter_contract"]["lanes"] == {
        "pair_id": "issue32-ascend-paired-lanes",
        "baseline_lane_id": "issue32-ascend-baseline",
        "diagnostic_lane_id": "issue32-ascend-diagnostic",
        "diagnostic_frontier_eligible": False,
        "reason_code": "profiling-overhead-error-budget-exceeded",
    }

    verdicts = {
        verdict["stable_path"]: verdict
        for verdict in result["performance_diagnosis_verdicts"]
    }
    integration = verdicts[
        "semantic/model/two-layer-transformer/transformer/"
        "layer-0/attention/q-proj"
    ]
    assert integration["verdict"] == "integration_overhead"
    assert integration["ledger"]["status"] == "conserved"
    assert integration["ledger"]["parent_span_total_included_ns"] == 0
    assert {
        leaf["kind"] for leaf in integration["ledger"]["leaves"]
    } >= {"operator", "copy", "dispatch", "sync", "profiling"}
    assert integration["surface_action"]["action"] == "preserve"
    assert integration["surface_action"][
        "operator_achievable_frontier_ns"
    ] == {
        "before": axes["operator_achievable_frontier"]["value_ns"],
        "after": axes["operator_achievable_frontier"]["value_ns"],
    }

    insufficient = verdicts[
        "semantic/model/two-layer-transformer/transformer/"
        "layer-0/attention/k-proj"
    ]
    assert insufficient["verdict"] == "insufficient_evidence"
    assert all(
        gate["gate_id"] != "direct-correctness-violation"
        for gate in insufficient["gates"]["satisfied"]
    )

    confirmed = verdicts[
        "semantic/model/two-layer-transformer/transformer/"
        "layer-0/attention/v-proj-negative-control"
    ]
    assert confirmed["verdict"] == "confirmed_bug"
    assert any(
        gate["gate_id"] == "direct-correctness-violation"
        for gate in confirmed["gates"]["satisfied"]
    )
    assert {
        reason_code
        for counterexample in confirmed["counterexamples"]
        for reason_code in counterexample.get(
            "reason_codes", [counterexample.get("reason_code")]
        )
    } >= {
        "performance-gap-is-not-direct-defect-evidence",
        "proxy-anomaly-is-not-direct-defect-evidence",
        "single-fluctuation-is-not-reproducible",
    }


def test_cli_json_and_human_report_drill_down_to_raw_bundle() -> None:
    expected = diagnose_run_bundle(AUTHORITATIVE_BUNDLE)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(AUTHORITATIVE_BUNDLE),
            "--json",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == expected

    report = render_diagnostic_report(expected)
    for expected_text in (
        f"derivation: {expected['derivation']['derivation_id']}",
        "qualification policy: issue32-frontier-qualification/v2",
        "Frontier Anchor issue32-ascend-matmul-square-512",
        "candidate search: winner=torch.matmul",
        "raw bundle: run-bundle://issue32-ascend-910b2-diagnostic-v1",
        "source run issue31-operator-frontier-v3: operator-frontier",
        "Shape Disambiguation Probe issue32-q-proj-integration: complete",
        "ablation remove-profiling(profiling)",
        "satisfied gates: diagnostic-trigger-met",
        "Operator Achievable Frontier preserved",
        "Performance Diagnosis Verdict",
    ):
        assert expected_text in report


def test_missing_ablation_artifact_fails_closed(tmp_path: Path) -> None:
    tampered = tmp_path / "tampered-diagnostic-bundle"
    shutil.copytree(AUTHORITATIVE_BUNDLE, tampered)
    manifest = json.loads(
        (tampered / "run.manifest.json").read_text(encoding="utf-8")
    )
    artifact = next(
        item
        for item in manifest["artifacts"]
        if item.get("uri") == "artifact://issue-32/integration-copy-1"
    )
    (tampered / artifact["path"]).unlink()

    verification = verify_run_bundle(tampered)
    assert verification["passed"] is False
    assert verification["failures"]
    with pytest.raises(DiagnosticBundleIntegrityError):
        diagnose_run_bundle(tampered)
