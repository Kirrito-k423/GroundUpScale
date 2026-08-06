from __future__ import annotations

from pathlib import Path

from groundupscale.benchmark import BenchmarkRunner, predict_live_set
from groundupscale.pipeline import compile_analysis_plan


REPOSITORY_ROOT = Path(__file__).parents[1]


def test_cpu_benchmark_runs_all_authored_cases_with_raw_windows() -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )
    observation = BenchmarkRunner(compiled.bundle).run(
        samples_override=4,
        warmup_override=0,
        windows_per_sample=1,
        target_window_ns=1,
    )

    assert observation["device"] == "cpu"
    assert observation["instrumentation_profile"] == "benchmark"
    assert observation["synchronization"] == "measurement-boundaries-only"
    assert [case["case_id"] for case in observation["cases"]] == [
        "matmul-q-proj",
        "rmsnorm-input",
        "softmax-attention",
        "transformer-layer",
        "two-layer-prefill",
    ]
    assert all(case["resolved_scope"].startswith("semantic/") for case in observation["cases"])
    for case in observation["cases"]:
        latency = case["latency"]
        assert len(latency["samples_ns"]) == 4
        assert len(latency["window_samples_ns"]) == 4
        assert latency["median_ns"] > 0
        assert latency["throughput_per_second"] > 0
        assert latency["inner_iterations"] == 1


def test_live_set_uses_alias_roots_and_separates_state_from_activation() -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )
    prediction = predict_live_set(compiled.semantic.semantic_ir, compiled.cost.cost_ir)

    assert prediction["parameter_bytes"] == 33_562_624
    assert prediction["buffer_bytes"] == 2_097_152
    assert prediction["state_bytes"] == 35_659_776
    assert prediction["peak_activation_live_bytes"] > 0
    assert prediction["predicted_framework_peak_bytes"] == (
        prediction["state_bytes"] + prediction["peak_activation_live_bytes"]
    )
    assert any(lifetime["aliases"] for lifetime in prediction["storage_lifetimes"])
    assert all(
        "/state-read/" not in lifetime["stable_path"]
        for lifetime in prediction["storage_lifetimes"]
    )
