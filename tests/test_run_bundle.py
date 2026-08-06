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
    assert manifest["stages"]["duration_prediction"] == "skipped-uncalibrated"
    assert len(manifest["artifacts"]) >= 15
    roles = {artifact["role"] for artifact in manifest["artifacts"]}
    assert {
        "resolved-input-lock",
        "model-ir",
        "workload-ir",
        "semantic-ir",
        "cost-ir",
        "prediction",
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
    assert explanation["calibration_status"] == "not-yet-applied"
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
        "local-apple-silicon-v1"
    )
    report = (run / "reports/report.html").read_text(encoding="utf-8")
    assert "GroundUpScale 可解释运行报告" in report
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


def test_required_environment_gate_rejects_before_publishing_a_run(
    tmp_path: Path,
) -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )
    invalid = _valid_preflight()
    invalid["eligible"] = False
    invalid["reason_codes"] = ["competing-process-above-policy"]

    with pytest.raises(
        EnvironmentValidityError, match="competing-process-above-policy"
    ):
        RunBundleWriter(compiled).run(
            tmp_path,
            run_id="must-not-run",
            environment_validity=invalid,
            require_valid_environment=True,
        )

    assert not (tmp_path / "runs/must-not-run").exists()
