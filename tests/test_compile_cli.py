from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).parents[1]


def test_compile_cli_writes_inspectable_structural_and_semantic_artifacts(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "compile",
            str(REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--output",
            str(tmp_path),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["semantic_operation_count"] == 52
    assert summary["semantic_compilation_fingerprint"]
    for artifact in (
        "model-ir.json",
        "workload-ir.json",
        "semantic-ir.json",
        "cost-ir.json",
        "hardware-prediction.json",
        "provenance.json",
        "compilation.json",
    ):
        path = tmp_path / artifact
        assert path.is_file()
        assert json.loads(path.read_text(encoding="utf-8"))
    assert summary["total_flops"] == 9_710_850_048
    assert summary["parameter_bytes"] == 33_562_624
    assert summary["explicit_activation_bytes"] == 121_634_816
    assert summary["hardware_prediction_status"] == (
        "empirical-hardware-lower-bound"
    )
    hardware_prediction = json.loads(
        (tmp_path / "hardware-prediction.json").read_text(encoding="utf-8")
    )
    assert hardware_prediction["prediction_complete"] is False
    assert hardware_prediction["program_bounds"]["full_duration_ns"] is None
    assert hardware_prediction["program_bounds"][
        "empirical_hardware_floor_ns"
    ] == pytest.approx(5_553_975.963160658)
    assert len(hardware_prediction["measured_capabilities"]) == 2
    assert hardware_prediction["capabilities"]["fp32_flops_per_second"][
        "status"
    ] == "unknown"
    compilation = json.loads(
        (tmp_path / "compilation.json").read_text(encoding="utf-8")
    )
    assert summary["hardware_compilation_fingerprint"] == (
        hardware_prediction["compilation_fingerprint"]
    )
    assert compilation["hardware_compilation_fingerprint"] == (
        hardware_prediction["compilation_fingerprint"]
    )
