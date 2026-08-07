from __future__ import annotations

from pathlib import Path

import pytest

from groundupscale.pipeline import compile_analysis_plan


REPOSITORY_ROOT = Path(__file__).parents[1]


def test_m4_cpu_backend_emits_empirical_algorithm_independent_hardware_floors() -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )

    prediction = compiled.hardware_prediction

    assert prediction is not None
    assert prediction.schema == (
        "groundupscale.dev/hardware-backend-prediction/v1alpha1"
    )
    assert prediction.backend_id == "apple.m4.cpu.resource-envelope"
    assert prediction.placement == "local-m4/cpu"
    assert prediction.status == "empirical-hardware-lower-bound"
    assert prediction.prediction_complete is False
    assert prediction.program_bounds.materialized_bytes == 289_415_168
    assert prediction.program_bounds.compulsory_bytes == 37_756_928
    assert prediction.program_bounds.vendor_memory_time_floor_ns == pytest.approx(
        314_641.06666666665
    )
    assert prediction.program_bounds.empirical_compute_time_ns == pytest.approx(
        5_553_975.963160658
    )
    assert prediction.program_bounds.empirical_memory_time_ns == pytest.approx(
        297_691.0648223506
    )
    assert prediction.program_bounds.empirical_hardware_floor_ns == pytest.approx(
        5_553_975.963160658
    )
    assert prediction.program_bounds.limiting_resource == "compute.fp32"
    assert prediction.program_bounds.full_duration_ns is None
    assert prediction.program_bounds.compute_time.status == "unknown"
    assert prediction.program_bounds.compute_time.reason == (
        "vendor_does_not_publish_frequency_or_fma_issue_rate"
    )

    candidates = tuple(prediction.candidates)
    assert len(candidates) == 52
    q_proj = next(
        candidate
        for candidate in candidates
        if candidate.stable_path.endswith("/attention/q_proj")
    )
    assert q_proj.operation == "MatMul"
    assert q_proj.flops == 268_435_456
    assert q_proj.compulsory_bytes == 3_145_728
    assert q_proj.materialized_bytes == 3_145_728
    assert q_proj.duration.empirical_compute_time_ns == pytest.approx(
        153_527.65853810357
    )
    assert q_proj.duration.empirical_memory_time_ns == pytest.approx(
        24_802.206311951108
    )
    assert q_proj.duration.empirical_hardware_floor_ns == pytest.approx(
        153_527.65853810357
    )
    assert q_proj.duration.full_duration_ns is None
    assert q_proj.duration.status == "empirical-hardware-lower-bound"

    resources = {item.resource: item for item in prediction.measured_capabilities}
    assert resources["compute.fp32"].robust_achievable_rate == pytest.approx(
        1_748_450_139_577.8
    )
    assert resources["memory.shared"].robust_achievable_rate == pytest.approx(
        126_832_587_409.13748
    )
    assert resources["compute.fp32"].environment_eligible is False

    e2e = next(
        bound for bound in prediction.scope_bounds if bound.case_id == "two-layer-prefill"
    )
    assert e2e.compulsory_bytes == 37_756_928
    assert e2e.empirical_hardware_floor_ns == pytest.approx(5_553_975.963160658)


def test_m4_gpu_plan_does_not_silently_reuse_the_cpu_backend() -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-mps-prefill.yaml"
    )

    assert compiled.hardware_prediction is None
