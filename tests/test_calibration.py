from __future__ import annotations

import json
from pathlib import Path

import pytest

from groundupscale.calibration import (
    CalibrationError,
    fit_calibration,
    promote_calibration,
    validate_calibration,
)


CASE_IDS = ("e2e", "layer", "matmul", "rmsnorm", "softmax")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _run(
    root: Path,
    run_id: str,
    *,
    delta: float = 0.0,
    noise: float = 0.01,
    device: str = "mps",
    environment_validity: str = "passed",
) -> Path:
    run = root / run_id
    _write_json(
        run / "run.manifest.json",
        {
            "status": "completed",
            "run_id": run_id,
            "device": device,
            "compilation_fingerprint": "semantic-fingerprint",
            "cost_compilation_fingerprint": "cost-fingerprint",
            "hardware_cohort": f"apple-m4-test-{device}",
            "environment_validity": environment_validity,
        },
    )
    cases = []
    for index, case_id in enumerate(CASE_IDS, 1):
        observed = 1_000_000.0 * index * (1.0 + delta)
        cases.append(
            {
                "case_id": case_id,
                "resolved_scope": f"semantic/scope/{case_id}",
                "latency": {
                    "median_ns": observed,
                    "iqr_over_median": noise,
                },
            }
        )
    _write_json(
        run / "observation/raw/benchmark.json",
        {"torch_num_threads": 10, "cases": cases},
    )
    _write_json(
        run / "observation/memory.json",
        {
            "framework_tensor_storage": {
                "peak_framework_tensor_bytes": int(110_000_000 * (1.0 + delta))
            }
        },
    )
    _write_json(
        run / "prediction/metrics.json",
        {"live_set": {"predicted_framework_peak_bytes": 100_000_000}},
    )
    cost_items = [
        {
            "node_id": f"cost-{case_id}",
            "stable_path": f"cost/semantic/scope/{case_id}",
            "operation": "MatMul",
            "metrics": {"flops": index * 100, "materialized_read_bytes": index * 10},
        }
        for index, case_id in enumerate(CASE_IDS, 1)
    ]
    _write_json(
        run / "ir/cost.ir.json",
        {
            "root": {
                "node_id": "root",
                "stable_path": "semantic/root",
                "kind": "root",
                "metrics": {},
                "items": cost_items,
            },
            "summary": {"parameter_bytes": 70_000_000, "buffer_bytes": 10_000_000},
        },
    )
    return run


def test_candidate_uses_only_fit_partition_and_five_holdouts_gate_promotion(
    tmp_path: Path,
) -> None:
    fit_runs = [_run(tmp_path, "fit-1"), _run(tmp_path, "fit-2", delta=0.01)]
    holdouts = [
        _run(tmp_path, f"holdout-{index}", delta=delta)
        for index, delta in enumerate((0.0, 0.01, -0.01, 0.02, -0.02), 1)
    ]

    profile = fit_calibration(fit_runs)
    assert profile["metadata"]["status"] == "candidate"
    assert profile["spec"]["fit_evidence"] == ["fit-1", "fit-2"]
    assert profile["spec"]["memory_model"]["base_predicted_peak_bytes"] == 100_000_000
    validation = validate_calibration(profile, holdouts)
    assert validation["passed"]
    assert validation["valid_holdout_runs"] == 5
    assert validation["quarantined_noisy_runs"] == 0
    assert all(
        case["relative_error"] <= 0.05
        for result in validation["run_results"]
        for case in result["case_results"]
    )
    promoted = promote_calibration(profile, validation)
    assert promoted["metadata"]["status"] == "active"
    assert promoted["spec"]["validation"]["passed"]


def test_noisy_holdout_is_quarantined_and_cannot_satisfy_minimum_count(
    tmp_path: Path,
) -> None:
    profile = fit_calibration([_run(tmp_path, "fit")])
    holdouts = [_run(tmp_path, f"holdout-{index}") for index in range(4)]
    holdouts.append(_run(tmp_path, "noisy", noise=0.031))

    validation = validate_calibration(profile, holdouts)

    assert not validation["passed"]
    assert validation["valid_holdout_runs"] == 4
    assert validation["quarantined_noisy_runs"] == 1
    with pytest.raises(CalibrationError, match="failed holdout"):
        promote_calibration(profile, validation)


def test_fit_rejects_noisy_or_mixed_cohort_evidence(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="noisy fitting evidence"):
        fit_calibration([_run(tmp_path, "noisy-fit", noise=0.031)])
    with pytest.raises(CalibrationError, match="share device"):
        fit_calibration(
            [
                _run(tmp_path, "cpu", device="cpu"),
                _run(tmp_path, "mps", device="mps"),
            ]
        )


def test_fit_and_holdout_overlap_is_rejected(tmp_path: Path) -> None:
    fit = _run(tmp_path, "same-run")
    profile = fit_calibration([fit])

    with pytest.raises(CalibrationError, match="overlap"):
        validate_calibration(profile, [fit])


def test_calibration_rejects_evidence_without_passed_environment_preflight(
    tmp_path: Path,
) -> None:
    unverified = _run(
        tmp_path,
        "unverified",
        environment_validity="not-required",
    )

    with pytest.raises(CalibrationError, match="environment preflight"):
        fit_calibration([unverified])
