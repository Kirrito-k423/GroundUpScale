from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from groundupscale.environment import evaluate_environment_validity
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
    assert manifest["hardware_cohort"].endswith(
        "-cpu-env-local-apple-silicon-v2"
    )
    assert manifest["stages"]["duration_prediction"] == (
        "empirical-hardware-lower-bound"
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
        "error-attribution",
        "explanation-graph",
        "html-report",
    } <= roles
    verification = verify_run_bundle(run)
    assert verification["passed"]
    assert verification["artifact_count"] == len(manifest["artifacts"])
    trace_lines = (run / "observation/observation.trace.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(trace_lines) == 60
    assert all(json.loads(line)["stable_path"].startswith("semantic/") for line in trace_lines)
    explanation = json.loads(
        (run / "prediction/explanation.graph.json").read_text(encoding="utf-8")
    )
    assert len(explanation["entrypoints"]["latency"]) == 5
    assert explanation["entrypoints"]["peak_memory"]
    assert explanation["entrypoints"]["hardware_duration_bound"] == [
        "metric:hardware-empirical-time-floor"
    ]
    assert len(
        explanation["entrypoints"]["prediction_observation_comparison"]
    ) == 6
    e2e_explanation = next(
        node
        for node in explanation["nodes"]
        if node["id"] == "comparison:latency:two-layer-prefill"
    )
    assert e2e_explanation["error_status"] == (
        "not-evaluable-hardware-floor"
    )
    missing_compute = next(
        node
        for node in explanation["nodes"]
        if node["id"] == "capability:missing-fp32-flops-per-second"
    )
    assert missing_compute["status"] == "unknown"
    assert explanation["calibration_status"] == "not-yet-applied"
    prediction = json.loads(
        (run / "prediction/metrics.json").read_text(encoding="utf-8")
    )
    assert prediction["duration_status"] == "empirical-hardware-lower-bound"
    assert prediction["duration"]["full_duration_ns"] is None
    assert prediction["duration"]["compulsory_bytes"] == 37_756_928
    assert prediction["duration"]["empirical_hardware_floor_ns"] == (
        pytest.approx(5_553_975.963160658)
    )
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
        "groundupscale.dev/prediction-observation-comparison/v1alpha1"
    )
    assert comparison["status"] == "empirical-hardware-floor-with-observation"
    assert comparison["summary"] == {
        "aligned_latency_cases": 5,
        "evaluable_latency_errors": 0,
        "evaluable_memory_errors": 1,
    }
    e2e_comparison = next(
        item
        for item in comparison["latency_cases"]
        if item["case_id"] == "two-layer-prefill"
    )
    assert e2e_comparison["predicted"][
        "empirical_hardware_floor_ns"
    ] == pytest.approx(5_553_975.963160658)
    assert e2e_comparison["predicted"]["minimum_work_flops"] == 9_710_850_048
    assert e2e_comparison["predicted"]["compulsory_bytes"] == 37_756_928
    assert e2e_comparison["predicted"]["limiting_resource"] == "compute.fp32"
    assert e2e_comparison["predicted"]["full_duration_ns"] is None
    assert e2e_comparison["observed"]["median_ns"] > 0
    assert e2e_comparison["comparison"]["relative_prediction_error"] is None
    assert e2e_comparison["comparison"]["error_status"] == (
        "not-evaluable-hardware-floor"
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
    assert environment["measurement_preflight"]["eligible"] is True
    assert environment["measurement_preflight"]["policy"]["policy_id"] == (
        "local-apple-silicon-v2"
    )
    report = (run / "reports/report.html").read_text(encoding="utf-8")
    assert "GroundUpScale 可解释运行报告" in report
    assert "5.554 ms" in report
    assert "不是当前实现耗时预测" in report
    assert "evidence=exploratory" in report
    assert "预测—实测对照" in report
    assert "不可计算预测误差" in report
    assert "matmul-q-proj" in report
    assert "峰值内存" in report
    assert "two-layer-prefill" in report

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
    assert len(explain_summary["cases"]) == 5
    assert explain_summary["duration_status"] == (
        "empirical-hardware-lower-bound"
    )
    assert explain_summary["hardware_empirical_floor_ns"] == pytest.approx(
        5_553_975.963160658
    )
    assert explain_summary["full_duration_ns"] is None
    assert explain_summary["hardware_capability_environment_eligible"] is False
    assert explain_summary["comparison_status"] == (
        "empirical-hardware-floor-with-observation"
    )
    assert len(explain_summary["latency_comparisons"]) == 5
    assert explain_summary["memory_comparison"]["error_status"] == "evaluated"
    assert explain_summary["memory_comparison"][
        "predicted_framework_peak_bytes"
    ] == 54_534_144
    assert explain_summary["memory_comparison"][
        "observed_framework_peak_bytes"
    ] > 0


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
