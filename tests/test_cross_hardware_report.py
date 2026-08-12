from __future__ import annotations

from groundupscale.cross_hardware import (
    CROSS_HARDWARE_REPORT_SCHEMA,
    compare_cross_hardware,
    render_cross_hardware_report,
)


def _diagnosis(*, cohort: str, device: str, observation: float = 1000.0) -> dict:
    evidence = {
        "hardware": {
            "device": device,
            "partition": "default",
            "topology": "single-node",
            "software": "runtime-v1",
        },
        "resolved_ir": {"operation": "MatMul", "stable_path": "semantic/transformer/q_proj"},
        "cohort_id": cohort,
        "execution_domain": {
            "shape": {"m": 512, "n": 512, "k": 512},
            "dtype": "float32",
            "layout": "row-major-contiguous",
            "execution_mode": "eager",
            "threads": 1,
        },
        "correctness": {"passed": True, "evidence_ref": "artifact://correctness"},
        "environment": {
            "eligible": True,
            "evidence_ref": "artifact://preflight",
        },
        "measurement_capability_manifest": {
            "status": "complete",
            "evidence_ref": "artifact://capabilities",
        },
        "cohort_evidence": {"status": "qualified", "evidence_ref": "artifact://cohort"},
        "timing_plan": {"evidence_ref": "artifact://timing-plan"},
        "baseline_timing_lane": {
            "observation_validity": "QUALIFIED",
            "completion_boundary": {"closed": True},
            "timer": {"source": "monotonic", "resolution_ns": 1},
            "raw_samples_ns": [950.0, 1000.0, 1050.0],
            "warmup": {"converged": True, "iterations": 5},
            "evidence_ref": "artifact://timing",
        },
        "policies": {"qualification": {"policy_id": "qualification", "version": "1"}},
        "frontier_anchors": [
            {
                "anchor_id": f"anchor-{cohort}",
                "frontier_role": "ACTIVE",
                "evidence_ref": "artifact://anchor",
            }
        ],
        "single_node_schedule": {"schedule_id": "schedule-1", "evidence_ref": "artifact://schedule"},
    }
    return {
        "schema": "groundupscale.dev/diagnostic-result/v1alpha1",
        "run_id": f"run-{cohort}",
        "status": "complete",
        "axes": {
            "operator_achievable_frontier": {
                "status": "known",
                "value_ns": 400.0,
                "anchor_id": f"anchor-{cohort}",
                "evidence_refs": ["artifact://anchor"],
            },
            "observation": {
                "status": "known",
                "value_ns": observation,
                "evidence_refs": ["artifact://timing"],
            },
            "resource_physical_floor": {"status": "known", "value_ns": 200.0},
            "schedule_achievable_frontier": {"status": "known", "value_ns": 450.0},
        },
        "comparisons": {
            "operator_frontier_to_observation": {
                "combined_uncertainty_ns": 50.0,
                "distance_ns": observation - 400.0,
            }
        },
        "capability_surface_queries": [
            {
                "query_id": "surface-query-1",
                "status": "known",
                "surface": {"surface_id": f"surface-{cohort}", "version": "1"},
                "evidence_refs": ["artifact://surface"],
            }
        ],
        "evidence": evidence,
        "digests": {"input_sha256": f"input-{cohort}", "evidence_sha256": f"evidence-{cohort}"},
        "derivation": {"derivation_id": f"derivation-{cohort}"},
        "performance_diagnosis_verdicts": [
            {"verdict": "insufficient_evidence", "evidence_refs": ["artifact://verdict"]}
        ],
    }


def test_compare_cross_hardware_reports_efficiency_uncertainty_and_index() -> None:
    report = compare_cross_hardware(
        _diagnosis(cohort="m4-cohort", device="Apple M4 CPU"),
        _diagnosis(cohort="ascend-cohort", device="Ascend 910B2"),
    )

    assert report["schema"] == CROSS_HARDWARE_REPORT_SCHEMA
    assert report["status"] == "complete"
    assert report["shape_comparison"] == {"status": "matched", "shape": {"m": 512, "n": 512, "k": 512}}
    assert report["cohorts"]["m4"]["independent"] is True
    assert report["metrics"]["m4"]["frontier_efficiency"] == 0.4
    assert report["metrics"]["ascend"]["combined_uncertainty_ns"] == 50.0
    assert report["evidence_index"]["m4"]["surface_refs"] == ["artifact://surface"]
    assert report["cross_hardware_comparison"]["combined_uncertainty_ns"] == 70.71067811865476
    rendered = render_cross_hardware_report(report)
    assert "Frontier Efficiency" in rendered
    assert "m4-cohort" in rendered


def test_compare_cross_hardware_rejects_shape_or_cohort_reuse() -> None:
    changed_shape = _diagnosis(cohort="ascend-cohort", device="Ascend 910B2")
    changed_shape["evidence"]["execution_domain"]["shape"]["m"] = 513
    report = compare_cross_hardware(_diagnosis(cohort="m4-cohort", device="Apple M4 CPU"), changed_shape)
    assert report["status"] == "insufficient_evidence"
    assert report["shape_comparison"]["status"] == "unknown"
    assert report["shape_comparison"]["reason_code"] == "exact-shape-mismatch"

    same_cohort = _diagnosis(cohort="m4-cohort", device="Ascend 910B2")
    report = compare_cross_hardware(_diagnosis(cohort="m4-cohort", device="Apple M4 CPU"), same_cohort)
    assert report["status"] == "insufficient_evidence"
    assert report["cohorts"]["ascend"]["independent"] is False
    assert report["cohorts"]["ascend"]["reason_code"] == "hardware-cohort-reused"


def test_compare_cross_hardware_rejects_stable_path_mismatch() -> None:
    npu = _diagnosis(cohort="ascend-cohort", device="Ascend 910B2")
    npu["evidence"]["resolved_ir"]["stable_path"] = "semantic/transformer/k_proj"
    report = compare_cross_hardware(_diagnosis(cohort="m4-cohort", device="Apple M4 CPU"), npu)
    assert report["status"] == "insufficient_evidence"
    assert "semantic-operation-mismatch" in report["gate"]["reason_codes"]


def test_compare_cross_hardware_fails_closed_when_required_evidence_is_missing() -> None:
    npu = _diagnosis(cohort="ascend-cohort", device="Ascend 910B2")
    npu["evidence"].pop("measurement_capability_manifest")
    npu["capability_surface_queries"][0]["status"] = "unknown"
    report = compare_cross_hardware(_diagnosis(cohort="m4-cohort", device="Apple M4 CPU"), npu)

    assert report["status"] == "insufficient_evidence"
    assert report["sides"]["ascend"]["evidence_quality"]["status"] == "unknown"
    assert "missing-measurement-capability-manifest" in report["sides"]["ascend"]["evidence_quality"]["reason_codes"]
    assert report["metrics"]["ascend"]["frontier_efficiency"]["status"] == "unknown"


def test_load_cross_hardware_inputs_keeps_source_identity(tmp_path) -> None:
    m4 = tmp_path / "m4.json"
    npu = tmp_path / "npu.json"
    import json

    m4.write_text(json.dumps(_diagnosis(cohort="m4-cohort", device="Apple M4 CPU")))
    npu.write_text(json.dumps(_diagnosis(cohort="ascend-cohort", device="Ascend 910B2")))

    from groundupscale.cross_hardware import compare_cross_hardware_inputs

    report = compare_cross_hardware_inputs(m4, npu)
    assert report["status"] == "complete"
    assert report["evidence_index"]["m4"]["source"] == str(m4.resolve())


def test_script_projects_cross_hardware_report(tmp_path, capsys) -> None:
    import json
    import runpy
    import sys

    m4 = tmp_path / "m4.json"
    npu = tmp_path / "npu.json"
    m4.write_text(json.dumps(_diagnosis(cohort="m4-cohort", device="Apple M4 CPU")))
    npu.write_text(json.dumps(_diagnosis(cohort="ascend-cohort", device="Ascend 910B2")))
    old_argv = sys.argv
    try:
        sys.argv = ["compare-cross-hardware.py", str(m4), str(npu), "--json"]
        try:
            runpy.run_path("scripts/compare-cross-hardware.py", run_name="__main__")
        except SystemExit as error:
            assert error.code == 0
    finally:
        sys.argv = old_argv
    output = json.loads(capsys.readouterr().out)
    assert output["schema"] == CROSS_HARDWARE_REPORT_SCHEMA
    assert output["status"] == "complete"


def test_bundle_without_diagnostic_artifact_is_reported_as_unknown(tmp_path) -> None:
    from groundupscale.cross_hardware import compare_cross_hardware_inputs

    report = compare_cross_hardware_inputs(tmp_path, tmp_path / "missing.json")
    assert report["status"] == "insufficient_evidence"
    assert report["sides"]["m4"]["evidence_quality"]["status"] == "unknown"
    assert report["sides"]["ascend"]["evidence_quality"]["status"] == "unknown"
