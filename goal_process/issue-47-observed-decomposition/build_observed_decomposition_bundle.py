"""Build issue #47's replayable same-boundary observed decomposition."""

from __future__ import annotations

import json
import os
from pathlib import Path

from groundupscale.observed_decomposition import timing_summary
from groundupscale.run_bundle import (
    verify_run_bundle,
    write_schedule_effect_frontier_bundle,
)
from hashlib import sha256


REPOSITORY_ROOT = Path(__file__).parents[2]
ISSUE30 = (
    REPOSITORY_ROOT
    / "goal_process/issue-30-ascend-transformer-demo/evidence/runs"
    / "ascend-910b2-transformer-demo-20260811-v1"
)
ISSUE32 = (
    REPOSITORY_ROOT
    / "goal_process/issue-32-ascend-diagnostic-bundle/evidence/runs"
    / "issue32-ascend-910b2-diagnostic-v1"
)
RUN_ID = "issue47-ascend-observed-decomposition-20260813-v1"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(manifest: dict[str, object], role: str) -> dict[str, object]:
    return next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == role
    )


def build_document() -> dict[str, object]:
    issue30_manifest = _read_json(ISSUE30 / "run.manifest.json")
    issue32_manifest = _read_json(ISSUE32 / "run.manifest.json")
    assert verify_run_bundle(ISSUE30)["passed"] is True
    assert verify_run_bundle(ISSUE32)["passed"] is True
    benchmark_artifact = _artifact(issue30_manifest, "benchmark-observation")
    diagnostic_artifact = _artifact(issue30_manifest, "error-attribution")
    correctness_artifact = _artifact(issue30_manifest, "correctness-observation")
    benchmark = _read_json(ISSUE30 / benchmark_artifact["path"])
    diagnostic = _read_json(ISSUE30 / diagnostic_artifact["path"])
    correctness = _read_json(ISSUE30 / correctness_artifact["path"])
    issue32_evidence_artifact = _artifact(
        issue32_manifest, "diagnostic-evidence"
    )
    issue32_evidence = _read_json(
        ISSUE32 / issue32_evidence_artifact["path"]
    )
    e2e = next(
        case
        for case in benchmark["cases"]
        if case["case_id"] == "two-layer-prefill"
    )
    pair_id = "issue47-ascend-two-layer-prefill-pair-v1"
    baseline_lane_id = "issue47-ascend-baseline"
    diagnostic_lane_id = "issue47-ascend-diagnostic"
    identity = {
        "benchmark_case": "two-layer-prefill",
        "shape": [1, 512, 512],
        "candidate_id": "ascend-two-layer-transformer-pytorch-eager-v1",
        "hardware_cohort": issue30_manifest["hardware_cohort"],
        "completion_boundary": (
            "end-npu-event-synchronize-plus-device-synchronize"
        ),
    }
    return {
        "schema": "groundupscale.dev/schedule-effect-input/v1alpha1",
        "pair_id": pair_id,
        "identity": identity,
        "baseline_timing_lane": {
            "lane_id": baseline_lane_id,
            "pair_id": pair_id,
            "identity": identity,
            "instrumentation_profile": "ascend-npu-baseline-timing-v1",
            "raw_samples_ns": e2e["latency"]["samples_ns"],
            "timing_summary": timing_summary(e2e["latency"]["samples_ns"]),
            "normalized_window_samples_ns": e2e["latency"][
                "normalized_window_samples_ns"
            ],
            "windows_per_sample": e2e["latency"]["windows_per_sample"],
            "warmup": {
                "iterations": e2e["warmup_iterations"],
                "outside_timing_boundary": True,
            },
            "timer": {
                "kind": "npu-event",
                "source": "torch.npu.Event.elapsed_time",
                "resolution_ns": 20.0,
            },
            "synchronization": (
                "end-event-synchronize-plus-device-synchronize"
            ),
            "correctness": {
                "passed": correctness["passed"],
                "semantic_leaf_count": 52,
                "max_absolute_error": correctness["max_absolute_error"],
                "no_cpu_fallback": (
                    correctness["target_audit"]["fallback_enabled"] is False
                ),
                "evidence_ref": (
                    "run-bundle://ascend-910b2-transformer-demo-20260811-v1/"
                    "observation/correctness.json"
                ),
            },
            "source": {
                "run_id": issue30_manifest["run_id"],
                "expected_role": "benchmark-observation",
                "derivation": {
                    "kind": "benchmark-case-latency",
                    "case_id": "two-layer-prefill",
                },
                "evidence_ref": (
                    "run-bundle://ascend-910b2-transformer-demo-20260811-v1/"
                    + benchmark_artifact["path"]
                ),
                "artifact_sha256": benchmark_artifact["sha256"],
            },
        },
        "diagnostic_profiling_lane": {
            "lane_id": diagnostic_lane_id,
            "paired_baseline_lane_id": baseline_lane_id,
            "pair_id": pair_id,
            "identity": identity,
            "instrumentation_profile": "torch-npu-profiler/v1",
            "instrumentation_timing": {
                "clock_domain": "host-monotonic",
                "elapsed_ns": diagnostic["e2e_trace_host_ns"],
                "source": "issue30-single-host-diagnostic-trace",
            },
            "source": {
                "run_id": issue30_manifest["run_id"],
                "expected_role": "error-attribution",
                "derivation": {
                    "kind": "json-field",
                    "field": "e2e_trace_host_ns",
                },
                "evidence_ref": (
                    "run-bundle://ascend-910b2-transformer-demo-20260811-v1/"
                    + diagnostic_artifact["path"]
                ),
                "artifact_sha256": diagnostic_artifact["sha256"],
            },
            "device_timeline_status": {
                "status": "unavailable",
                "reason_code": "profiler-device-timeline-export-incomplete",
                "evidence_ref": (
                    "run-bundle://issue32-ascend-910b2-diagnostic-v1/"
                    + issue32_evidence_artifact["path"]
                ),
            },
            "overhead_ablation": {
                "status": "unavailable",
                "reason_code": "exact-identity-profiling-ablation-missing",
                "instrumentation_profile": "torch-npu-profiler/v1",
                "source_boundary": {
                    "run_id": issue32_manifest["run_id"],
                    "expected_role": "diagnostic-evidence",
                    "evidence_ref": (
                        "run-bundle://issue32-ascend-910b2-diagnostic-v1/"
                        + issue32_evidence_artifact["path"]
                    ),
                    "artifact_sha256": issue32_evidence_artifact["sha256"],
                    "reason_code": (
                        "available-ablation-has-incompatible-case-and-shape"
                    ),
                },
            },
        },
    }


def main() -> None:
    evidence_root = Path(__file__).parent / "evidence"
    destination = evidence_root / "runs" / RUN_ID
    source_runs = [
        {
            "run_id": manifest["run_id"],
            "path": os.path.relpath(source, destination),
            "manifest_sha256": sha256(
                (source / "run.manifest.json").read_bytes()
            ).hexdigest(),
        }
        for source, manifest in (
            (ISSUE30, _read_json(ISSUE30 / "run.manifest.json")),
            (ISSUE32, _read_json(ISSUE32 / "run.manifest.json")),
        )
    ]
    run = write_schedule_effect_frontier_bundle(
        evidence_root,
        run_id=RUN_ID,
        document=build_document(),
        source_runs=source_runs,
    )
    verification = verify_run_bundle(run)
    if verification["passed"] is not True:
        raise RuntimeError(verification["failures"])
    print(json.dumps(verification, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
