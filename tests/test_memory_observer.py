from __future__ import annotations

from pathlib import Path

from groundupscale.benchmark import BenchmarkRunner, observe_tensor_storage_peak
from groundupscale.pipeline import compile_analysis_plan


REPOSITORY_ROOT = Path(__file__).parents[1]


def test_tensor_storage_observer_tracks_framework_peak_without_rss() -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )
    runner = BenchmarkRunner(compiled.bundle)
    model, hidden = runner._model_and_input()
    observation = observe_tensor_storage_peak(
        model, (hidden,), device=runner.device
    )

    assert observation["device"] == "cpu"
    assert observation["peak_framework_tensor_bytes"] > 35_659_776
    assert len(observation["timeline"]) == 54
    assert "process_rss" not in observation
    assert "driver allocations" in observation["excludes"]
