from __future__ import annotations

from pathlib import Path

from groundupscale.benchmark import TraceRunner
from groundupscale.pipeline import compile_analysis_plan


REPOSITORY_ROOT = Path(__file__).parents[1]


def test_trace_aligns_runtime_modules_and_leaves_to_exact_semantic_paths() -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )
    trace = TraceRunner(compiled.bundle, compiled.semantic.semantic_ir).run()

    assert trace["device"] == "cpu"
    assert trace["instrumentation_profile"] == "trace"
    assert trace["synchronization"] == "profile-boundaries-only"
    assert len(trace["events"]) == 60
    assert sum(event["runtime_kind"] == "operation" for event in trace["events"]) == 52
    assert all(event["compiled_node_id"] for event in trace["events"])
    assert all("payload" not in event for event in trace["events"])
    assert all(event["host_duration_ns"] > 0 for event in trace["events"])
    assert trace["alignment_map"]["coverage"] == 1.0
    assert {
        entry["match_rule"] for entry in trace["alignment_map"]["entries"]
    } == {"exact-stable-path"}
    assert trace["error_attribution"]["unattributed_host_ns"] >= 0
    assert trace["memory_observation"]["observer"] == "process_rss"
