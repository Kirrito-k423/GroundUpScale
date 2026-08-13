from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import pytest

from groundupscale.environment import evaluate_environment_validity
from groundupscale.cli import main
from groundupscale.pipeline import compile_analysis_plan
from groundupscale.run_bundle import (
    EnvironmentValidityError,
    RunBundleExistsError,
    RunBundleWriter,
    verify_run_bundle,
)


REPOSITORY_ROOT = Path(__file__).parents[1]


def _valid_preflight() -> dict[str, object]:
    return evaluate_environment_validity(
        {
            "platform": {
                "system": "Darwin",
                "machine": "arm64",
                "logical_cpu_count": 10,
            },
            "power": {"source": "ac", "battery_percent": 100.0},
            "thermal": {"status": "nominal"},
            "load": {
                "one_minute": 1.0,
                "five_minutes": 1.0,
                "fifteen_minutes": 1.0,
            },
            "competitors": {
                "sample_interval_seconds": 1.0,
                "sample_count": 3,
                "total_cpu_percent_samples": [0.0, 0.0, 0.0],
                "top": [],
            },
        }
    )


def test_run_bundle_is_atomic_self_describing_and_digest_verifiable(
    tmp_path: Path,
) -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )
    writer = RunBundleWriter(compiled)
    run = writer.run(
        tmp_path,
        run_id="test-cpu-run",
        samples_override=4,
        warmup_override=0,
        windows_per_sample=1,
        target_window_ns=1,
        environment_validity=_valid_preflight(),
        require_valid_environment=True,
    )

    assert run == tmp_path / "runs/test-cpu-run"
    manifest = json.loads((run / "run.manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["device"] == "cpu"
    assert manifest["environment_validity"] == "passed"
    assert manifest["hardware_cohort"].startswith("hvc-")
    assert manifest["stages"]["duration_prediction"] == (
        "phase-capabilities-incomplete"
    )
    assert manifest["stages"]["prediction_observation_comparison"] == "completed"
    assert len(manifest["artifacts"]) >= 17
    roles = {artifact["role"] for artifact in manifest["artifacts"]}
    assert {
        "resolved-input-lock",
        "model-ir",
        "workload-ir",
        "semantic-ir",
        "cost-ir",
        "hardware-backend-prediction",
        "prediction",
        "prediction-observation-comparison",
        "benchmark-observation",
        "observation-trace",
        "alignment-map",
        "memory-observation",
        "correctness-observation",
        "alias-materialization-evidence",
        "error-attribution",
        "explanation-graph",
        "html-report",
    } <= roles
    verification = verify_run_bundle(run)
    assert verification["passed"]
    assert verification["artifact_count"] == len(manifest["artifacts"])
    alias_evidence = json.loads(
        (run / "observation/alias-materialization.json").read_text(encoding="utf-8")
    )
    assert alias_evidence["status"] == "qualified"
    assert len(alias_evidence["operations"]) == 16
    assert {
        operation["decision"] for operation in alias_evidence["operations"]
    } == {"alias-preserving"}
    assert alias_evidence["hardware_cohort"] == manifest["hardware_cohort"]
    trace_lines = (run / "observation/observation.trace.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(trace_lines) == 60
    assert all(json.loads(line)["stable_path"].startswith("semantic/") for line in trace_lines)
    explanation = json.loads(
        (run / "prediction/explanation.graph.json").read_text(encoding="utf-8")
    )
    assert len(explanation["entrypoints"]["latency"]) == 10
    assert explanation["entrypoints"]["peak_memory"]
    assert explanation["entrypoints"]["hardware_duration_bound"] == [
        "metric:hardware-empirical-time-floor"
    ]
    assert len(
        explanation["entrypoints"]["prediction_observation_comparison"]
    ) == 11
    e2e_explanation = next(
        node
        for node in explanation["nodes"]
        if node["id"] == "comparison:latency:two-layer-prefill"
    )
    assert e2e_explanation["error_status"] == (
        "not-evaluable-phase-capabilities-incomplete"
    )
    missing_compute = next(
        node
        for node in explanation["nodes"]
        if node["id"] == "capability:missing-fp32-flops-per-second"
    )
    assert missing_compute["status"] == "unknown"
    softmax_candidate_node = next(
        node
        for node in explanation["nodes"]
        if node["kind"] == "implementation-candidate"
        and node["stable_path"].endswith("/layer_0/attention/softmax")
    )
    softmax_phase_nodes = [
        node
        for node in explanation["nodes"]
        if node["kind"] == "operator-phase"
        and node["candidate_id"] == softmax_candidate_node["id"]
    ]
    assert [node["phase_name"] for node in softmax_phase_nodes] == [
        "max_reduce",
        "subtract",
        "exp",
        "sum_reduce",
        "normalize",
    ]
    assert sum(
        edge["kind"] == "phase_depends_on"
        and edge["source"] in {node["id"] for node in softmax_phase_nodes}
        for edge in explanation["edges"]
    ) == 4
    assert explanation["calibration_status"] == "not-yet-applied"
    prediction = json.loads(
        (run / "prediction/metrics.json").read_text(encoding="utf-8")
    )
    assert prediction["schema"] == "groundupscale.dev/prediction/v1alpha2"
    assert prediction["duration_status"] == "phase-capabilities-incomplete"
    assert prediction["duration"]["full_duration_ns"] is None
    assert prediction["duration"]["compulsory_bytes"] == 37_756_928
    assert prediction["duration"]["materialized_bytes"] == 289_415_168
    assert prediction["duration"]["empirical_hardware_floor_ns"] is None
    assert prediction["duration"]["resource_physical_floor_ns"] is None
    assert prediction["layout_execution"] == {
        "status": "qualified",
        "authoritative_artifact": "observation/alias-materialization.json",
        "evidence_version_id": alias_evidence["evidence_version_id"],
        "schedule": alias_evidence["schedule"],
        "decomposition": alias_evidence["decomposition"],
        "policy": (
            "View/Transpose candidate duration is unknown in the hardware backend "
            "until this runtime audit qualifies the selected candidate"
        ),
    }
    assert prediction["duration"]["schedule"] == "serialized-unfused"
    assert prediction["duration"]["ideal_dag_hardware_floor_ns"] is None
    inputs_lock = json.loads(
        (run / "resolved/inputs.lock.json").read_text(encoding="utf-8")
    )
    assert inputs_lock["documents"]["hardware_capability_profiles"][0][
        "metadata"
    ]["name"] == "apple-m4-cpu-local"
    comparison = json.loads(
        (run / "comparison/predicted-vs-observed.json").read_text(
            encoding="utf-8"
        )
    )
    assert comparison["schema"] == (
        "groundupscale.dev/prediction-observation-comparison/v1alpha2"
    )
    assert comparison["status"] == "exploratory-estimate-with-observation"
    assert comparison["summary"] == {
        "aligned_latency_cases": 6,
        "evaluable_latency_errors": 0,
        "evaluable_memory_errors": 1,
        "provisional_latency_comparisons": 6,
        "qualified_frontier_comparisons": 0,
    }
    decomposition = comparison["latency_decomposition"]
    assert decomposition["predicted"]["available"] is False
    assert decomposition["predicted"]["top10"] == []
    assert len(decomposition["observed"]["top10"]) == 10
    assert decomposition["observed"]["statistic"] == (
        "single-diagnostic-trace"
    )
    assert len(decomposition["joined"]) == 10
    assert all(
        item["predicted_time_ns"] is None
        and item["evidence_quality"]
        in {"not-modeled", "unattributed-evidence-boundary"}
        for item in decomposition["joined"]
    )
    assert decomposition["largest_discrepancy"] is None
    e2e_comparison = next(
        item
        for item in comparison["latency_cases"]
        if item["case_id"] == "two-layer-prefill"
    )
    assert e2e_comparison["predicted"]["empirical_hardware_floor_ns"] is None
    assert e2e_comparison["predicted"]["resource_physical_floor_ns"] is None
    assert e2e_comparison["predicted"]["minimum_work_flops"] == 9_710_850_048
    assert e2e_comparison["predicted"]["compulsory_bytes"] == 37_756_928
    assert e2e_comparison["predicted"]["materialized_bytes"] == 289_415_168
    assert e2e_comparison["predicted"]["schedule"] == "serialized-unfused"
    assert e2e_comparison["predicted"]["ideal_dag_hardware_floor_ns"] is None
    assert e2e_comparison["predicted"]["limiting_resource"] is None
    assert e2e_comparison["predicted"]["resource_limiting_resource"] is None
    assert e2e_comparison["predicted"]["full_duration_ns"] is None
    assert e2e_comparison["observed"]["median_ns"] > 0
    assert e2e_comparison["comparison"]["relative_prediction_error"] is None
    assert e2e_comparison["comparison"]["error_status"] == (
        "not-evaluable-phase-capabilities-incomplete"
    )
    q_proj_comparison = next(
        item
        for item in comparison["latency_cases"]
        if item["case_id"] == "matmul-q-proj"
    )
    assert q_proj_comparison["predicted"][
        "empirical_hardware_floor_ns"
    ] == pytest.approx(153_527.65853810357)
    assert q_proj_comparison["predicted"]["candidate_count"] == 1
    assert q_proj_comparison["comparison"]["operator_frontier_efficiency"] is None
    assert "operator-frontier-observation-timing-unqualified" in q_proj_comparison[
        "comparison"
    ]["operator_frontier_comparison_reason_codes"]
    softmax_comparison = next(
        item
        for item in comparison["latency_cases"]
        if item["case_id"] == "softmax-attention"
    )
    phase_schedule = softmax_comparison["predicted"]["compound_phase_schedule"]
    assert phase_schedule["policy"] == "serialized-no-chunk"
    assert phase_schedule["chunk_pipeline_contract_id"] is None
    assert [phase["phase_name"] for phase in phase_schedule["phases"]] == [
        "max_reduce",
        "subtract",
        "exp",
        "sum_reduce",
        "normalize",
    ]
    assert phase_schedule["status"] == "unknown"
    assert phase_schedule["selected_duration_ns"] is None
    assert "compute.transcendental.exp.fp32" in phase_schedule[
        "missing_capabilities"
    ]
    assert comparison["memory"]["predicted"][
        "framework_peak_bytes"
    ] == 54_534_144
    assert comparison["memory"]["observed"]["framework_peak_bytes"] > 0
    assert comparison["memory"]["comparison"]["error_status"] == "evaluated"
    memory = json.loads(
        (run / "observation/memory.json").read_text(encoding="utf-8")
    )
    assert memory["authoritative_gate_metric"].endswith(
        "peak_framework_tensor_bytes"
    )
    assert memory["framework_tensor_storage"]["peak_framework_tensor_bytes"] > 0
    environment = json.loads(
        (run / "resolved/environment.json").read_text(encoding="utf-8")
    )
    assert environment["hardware_validity_cohort"]["cohort_id"] == manifest[
        "hardware_cohort"
    ]
    observed_hardware = environment["hardware_validity_cohort"]["device"][
        "observed_identity"
    ]
    assert observed_hardware["status"] == "resolved"
    assert observed_hardware["cpu_brand"] == "Apple M4"
    assert observed_hardware["performance_levels"] == {
        "performance_cores": 4,
        "efficiency_cores": 6,
    }
    assert environment["measurement_preflight"]["eligible"] is True
    assert environment["measurement_preflight"]["policy"]["policy_id"] == (
        "local-apple-silicon-v2"
    )
    assert environment["process"]["pid"] > 0
    assert environment["process"]["session_id"] == "test-cpu-run"
    report = (run / "reports/report.html").read_text(encoding="utf-8")
    assert "GroundUpScale 可解释运行报告" in report
    assert "Resource Physical Floor：not available" in report
    assert "serialized-unfused" in report
    assert "理想 DAG" in report
    assert "时间分解比较契约" in report
    assert "selected hardware floor is unavailable" in report
    assert "实测侧 Top 10" in report
    assert "Top 10 联合差异排名" in report
    assert "最大差异下钻" in report
    assert "时间回收" in report
    assert "single-diagnostic-trace" in report
    assert "不可作为点预测" in report
    assert "evidence=exploratory" in report
    assert "预测—实测对照" in report
    assert "权威值仍 unknown；降级预估仅供规划" in report
    assert "matmul-q-proj" in report
    assert "峰值内存" in report
    assert "two-layer-prefill" in report
    assert "复合算子 Phase 串行构成" in report
    assert "max_reduce" in report
    assert "compute.transcendental.exp.fp32" in report
    assert "资源组合" in report
    assert "限制资源" in report
    assert "能力证据" in report

    with pytest.raises(RunBundleExistsError):
        writer.run(
            tmp_path,
            run_id="test-cpu-run",
            samples_override=4,
            warmup_override=0,
            windows_per_sample=1,
            target_window_ns=1,
            environment_validity=_valid_preflight(),
            require_valid_environment=True,
        )

    explained = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "explain",
            str(run),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert explained.returncode == 0, explained.stderr
    explain_summary = json.loads(explained.stdout)
    assert explain_summary["run_id"] == "test-cpu-run"
    assert len(explain_summary["cases"]) == 10
    assert explain_summary["duration_status"] == "phase-capabilities-incomplete"
    assert explain_summary["hardware_empirical_floor_ns"] is None
    assert explain_summary["hardware_resource_physical_floor_ns"] is None
    assert explain_summary["hardware_floor_schedule"] == "serialized-unfused"
    assert explain_summary["hardware_ideal_dag_floor_ns"] is None
    assert explain_summary["full_duration_ns"] is None
    assert explain_summary["hardware_capability_environment_eligible"] is False
    assert explain_summary["comparison_status"] == (
        "exploratory-estimate-with-observation"
    )
    assert len(explain_summary["latency_comparisons"]) == 10
    assert explain_summary["memory_comparison"]["error_status"] == "evaluated"
    assert explain_summary["memory_comparison"][
        "predicted_framework_peak_bytes"
    ] == 54_534_144
    assert explain_summary["memory_comparison"][
        "observed_framework_peak_bytes"
    ] > 0


def test_run_cli_can_isolate_one_exact_shape_benchmark_case(
    tmp_path: Path,
) -> None:
    exit_code = main(
        [
            "run",
            str(REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--artifact-store",
            str(tmp_path),
            "--run-id",
            "isolated-layer0-qk",
            "--case-id",
            "matmul-layer0-qk",
            "--samples",
            "4",
            "--warmup",
            "0",
            "--windows-per-sample",
            "1",
            "--target-window-ms",
            "0.000001",
            "--require-valid-environment",
            "--json",
        ],
        environment_collector=lambda **_: _valid_preflight(),
    )

    assert exit_code == 0
    run = tmp_path / "runs/isolated-layer0-qk"
    benchmark = json.loads(
        (run / "observation/raw/benchmark.json").read_text(encoding="utf-8")
    )
    assert [case["case_id"] for case in benchmark["cases"]] == [
        "matmul-layer0-qk"
    ]
    correctness = json.loads(
        (run / "observation/correctness.json").read_text(encoding="utf-8")
    )
    assert [case["case_id"] for case in correctness["operator_cases"]] == [
        "matmul-layer0-qk"
    ]
    assert verify_run_bundle(run)["passed"] is True


def test_run_bundle_verifier_rejects_tampered_alias_materialization_summary(
    tmp_path: Path,
) -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )
    run = RunBundleWriter(compiled).run(
        tmp_path,
        run_id="alias-audit-tamper",
        samples_override=4,
        warmup_override=0,
        windows_per_sample=1,
        target_window_ns=1,
    )
    evidence_path = run / "observation/alias-materialization.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["operations"][0]["duration"]["value_ns"] = 1
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in manifest["artifacts"]
        if item["role"] == "alias-materialization-evidence"
    )
    entry["sha256"] = sha256(evidence_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert "evidence version digest mismatch" in verification["failures"]
    assert "unverified alias zero" in verification["failures"]


def test_run_bundle_verifier_cross_checks_layout_prediction_authority(
    tmp_path: Path,
) -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )
    run = RunBundleWriter(compiled).run(
        tmp_path,
        run_id="layout-authority-tamper",
        samples_override=4,
        warmup_override=0,
        windows_per_sample=1,
        target_window_ns=1,
    )
    prediction_path = run / "prediction/metrics.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction["layout_execution"]["status"] = "unknown"
    prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item for item in manifest["artifacts"] if item["role"] == "prediction"
    )
    entry["sha256"] = sha256(prediction_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert "layout execution authority mismatch" in verification["failures"]


def test_required_environment_gate_rejects_before_publishing_a_run(
    tmp_path: Path,
) -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )
    invalid = _valid_preflight()
    invalid["eligible"] = False
    invalid["reason_codes"] = ["total-competing-cpu-above-policy"]

    with pytest.raises(
        EnvironmentValidityError, match="total-competing-cpu-above-policy"
    ):
        RunBundleWriter(compiled).run(
            tmp_path,
            run_id="must-not-run",
            environment_validity=invalid,
            require_valid_environment=True,
        )

    assert not (tmp_path / "runs/must-not-run").exists()


def test_exploratory_phase_estimate_is_visible_without_becoming_authoritative(
    tmp_path: Path,
) -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT,
        REPOSITORY_ROOT
        / "goal_process/mac-transformer-ir-calibration-slice/"
        "mac-cpu-prefill-phase-exploratory.yaml",
    )
    exploratory_measurement = _valid_preflight()
    exploratory_measurement["eligible"] = False
    exploratory_measurement["reason_codes"] = [
        "load-above-policy",
        "total-competing-cpu-above-policy",
    ]
    run = RunBundleWriter(compiled).run(
        tmp_path,
        run_id="exploratory-phase-estimate",
        samples_override=4,
        warmup_override=0,
        windows_per_sample=1,
        target_window_ns=1,
        environment_validity=exploratory_measurement,
        require_valid_environment=False,
    )

    prediction = json.loads(
        (run / "prediction/metrics.json").read_text(encoding="utf-8")
    )
    assert prediction["duration"]["empirical_hardware_floor_ns"] is None
    assert prediction["duration"]["provisional_estimate_ns"] == pytest.approx(
        8_160_245.494435462
    )
    assert prediction["duration"]["provisional_evidence_tier"] == "exploratory"
    comparison = json.loads(
        (run / "comparison/predicted-vs-observed.json").read_text(encoding="utf-8")
    )
    softmax = next(
        item
        for item in comparison["latency_cases"]
        if item["case_id"] == "softmax-attention"
    )
    assert softmax["predicted"]["empirical_hardware_floor_ns"] is None
    assert softmax["predicted"]["provisional_estimate_ns"] == pytest.approx(
        691_693.8713503513
    )
    assert softmax["predicted"]["provisional_evidence_tier"] == "exploratory"
    assert softmax["observed"]["evidence_tier"] == "exploratory"
    assert set(softmax["observed"]["reason_codes"]) == {
        "load-above-policy",
        "total-competing-cpu-above-policy",
    }
    provisional_schedule = softmax["predicted"][
        "compound_provisional_phase_schedule"
    ]
    assert provisional_schedule["selected_duration_ns"] == pytest.approx(
        691_693.8713503513
    )
    assert [phase["phase_name"] for phase in provisional_schedule["phases"]] == [
        "max_reduce",
        "subtract",
        "exp",
        "sum_reduce",
        "normalize",
    ]
    assert provisional_schedule["phases"][2][
        "local_hardware_floor_ns"
    ] == pytest.approx(356_379.8393014934)
    assert softmax["comparison"]["observed_to_provisional_ratio"] > 0
    decomposition = comparison["latency_decomposition"]
    assert decomposition["predicted"]["available"] is True
    assert decomposition["predicted"]["kind"] == (
        "mixed-exact-frontier-and-provisional-estimate"
    )
    assert decomposition["predicted"]["exact_frontier_override_count"] == 3
    layer1_qk = next(
        item
        for item in decomposition["predicted"]["all_items"]
        if item["stable_path"].endswith("/layer_1/attention/qk_matmul")
    )
    assert layer1_qk["time_ns"] == pytest.approx(580_157.4444444445)
    assert layer1_qk["evidence"] == "exact-operator-frontier"
    assert decomposition["comparison_role"] == "exploratory-planning-only"
    assert decomposition["largest_discrepancy"] is None
    assert len(decomposition["predicted"]["top10"]) == 10
    explanation = json.loads(
        (run / "prediction/explanation.graph.json").read_text(encoding="utf-8")
    )
    hardware_metric = next(
        node
        for node in explanation["nodes"]
        if node["id"] == "metric:hardware-empirical-time-floor"
    )
    assert hardware_metric["provisional_estimate_ns"] == pytest.approx(
        8_160_245.494435462
    )
    assert hardware_metric["provisional_evidence_tier"] == "exploratory"
    report = (run / "reports/report.html").read_text(encoding="utf-8")
    assert "降级预估" in report
    assert "exploratory" in report
    assert "0.692" in report
    assert "356.380 μs" in report
    assert "实测证据：exploratory" in report
    assert "预测证据原因" in report
    assert "实测证据原因" in report
    assert "load-above-policy" in report
    assert "total-competing-cpu-above-policy" in report
    assert "预测侧探索性规划 Top 10（非诊断）" in report
    assert "降级预估不产生最大差异诊断" in report
    assert "存在 Exact-Shape Anchor 的行不改变其余行的 exploratory/unknown 边界" in report
    assert "实测/可采纳" not in report


def test_explain_replays_v1alpha1_bundle_without_alpha2_phase_fields(
    tmp_path: Path,
) -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )
    run = RunBundleWriter(compiled).run(
        tmp_path,
        run_id="legacy-v1-run",
        samples_override=4,
        warmup_override=0,
        windows_per_sample=1,
        target_window_ns=1,
        environment_validity=_valid_preflight(),
        require_valid_environment=True,
    )
    prediction_path = run / "prediction/metrics.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction["schema"] = "groundupscale.dev/prediction/v1alpha1"
    duration = prediction["duration"]
    duration["empirical_hardware_floor_ns"] = duration[
        "resource_physical_floor_ns"
    ]
    for field in (
        "resource_physical_floor_ns",
        "schedule",
        "serialized_hardware_floor_ns",
        "critical_path_hardware_floor_ns",
        "resource_hardware_floor_ns",
        "ideal_dag_hardware_floor_ns",
        "resource_limiting_resource",
    ):
        duration.pop(field, None)
    prediction_path.write_text(json.dumps(prediction), encoding="utf-8")

    comparison_path = run / "comparison/predicted-vs-observed.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["schema"] = (
        "groundupscale.dev/prediction-observation-comparison/v1alpha1"
    )
    for item in comparison["latency_cases"]:
        predicted = item["predicted"]
        predicted["empirical_hardware_floor_ns"] = predicted[
            "resource_physical_floor_ns"
        ]
        for field in (
            "materialized_bytes",
            "resource_physical_floor_ns",
            "schedule",
            "serialized_hardware_floor_ns",
            "critical_path_hardware_floor_ns",
            "resource_hardware_floor_ns",
            "ideal_dag_hardware_floor_ns",
            "resource_limiting_resource",
            "compound_phase_schedule",
        ):
            predicted.pop(field, None)
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    explained = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "explain",
            str(run),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert explained.returncode == 0, explained.stderr
    summary = json.loads(explained.stdout)
    assert summary["schema"] == "groundupscale.dev/explain-summary/v1alpha2"
    assert summary["hardware_empirical_floor_ns"] is None
    assert summary["hardware_resource_physical_floor_ns"] is None
    assert all(item["schedule"] is None for item in summary["latency_comparisons"])
