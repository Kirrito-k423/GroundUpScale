from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

from groundupscale.run_bundle import (
    verify_run_bundle,
    write_schedule_effect_frontier_bundle,
)


IDENTITY = {
    "benchmark_case": "two-layer-prefill",
    "shape": [1, 512, 512],
    "candidate_id": "ascend-two-layer-prefill-eager-v1",
    "hardware_cohort": "ascend-npu-23b93a89d5fecc79",
    "completion_boundary": "same-stream-device-events-with-stream-sync",
}
PUBLISHED_BUNDLE = (
    Path(__file__).parents[1]
    / "goal_process/issue-47-observed-decomposition/evidence/runs"
    / "issue47-ascend-observed-decomposition-20260813-v1"
)


def _write_json(path: Path, value: object) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _qualified_input() -> dict[str, object]:
    return {
        "schema": "groundupscale.dev/schedule-effect-input/v1alpha1",
        "pair_id": "issue47-pair-qualified",
        "identity": IDENTITY,
        "baseline_timing_lane": {
            "lane_id": "issue47-baseline",
            "pair_id": "issue47-pair-qualified",
            "identity": IDENTITY,
            "instrumentation_profile": "ascend-npu-baseline-timing-v1",
            "raw_samples_ns": [1_900_000, 1_921_530, 1_930_000],
            "warmup": {"iterations": 20, "outside_timing_boundary": True},
            "timer": {"kind": "npu-event", "resolution_ns": 1.0},
            "synchronization": "same-stream-event-then-stream-sync",
            "correctness": {"passed": True, "semantic_leaf_count": 52},
            "source": {
                "run_id": "ascend-910b2-transformer-demo-20260811-v1",
                "evidence_ref": "run-bundle://ascend-910b2-transformer-demo-20260811-v1",
                "artifact_sha256": "1" * 64,
            },
        },
        "diagnostic_profiling_lane": {
            "lane_id": "issue47-diagnostic",
            "paired_baseline_lane_id": "issue47-baseline",
            "pair_id": "issue47-pair-qualified",
            "identity": IDENTITY,
            "instrumentation_profile": "ascend-npu-profiler-device-timeline-v1",
            "instrumentation_timing": {
                "clock_domain": "host-monotonic",
                "elapsed_ns": 50_840_320,
                "source": "torch-npu-profiler-session",
            },
            "device_timeline": {
                "clock_domain": "ascend-device",
                "started_ns": 0,
                "ended_ns": 1_930_000,
                "intervals": [
                    {
                        "span_id": "layer-0",
                        "stable_path": "semantic/workload/layer_0",
                        "kind": "inclusive-parent",
                        "started_ns": 0,
                        "ended_ns": 1_700_000,
                        "parent_span_id": None,
                        "evidence_ref": "artifact://diagnostic/device-timeline#layer-0",
                    },
                    {
                        "span_id": "q-proj",
                        "stable_path": "semantic/workload/layer_0/attention/q_proj",
                        "kind": "leaf",
                        "started_ns": 100_000,
                        "ended_ns": 900_000,
                        "parent_span_id": "layer-0",
                        "evidence_ref": "artifact://diagnostic/device-timeline#q-proj",
                    },
                    {
                        "span_id": "k-proj",
                        "stable_path": "semantic/workload/layer_0/attention/k_proj",
                        "kind": "leaf",
                        "started_ns": 700_000,
                        "ended_ns": 1_300_000,
                        "parent_span_id": "layer-0",
                        "evidence_ref": "artifact://diagnostic/device-timeline#k-proj",
                    },
                ],
                "source": {
                    "collector": "torch-npu-profiler",
                    "evidence_ref": "artifact://diagnostic/device-timeline",
                    "artifact_sha256": "2" * 64,
                },
            },
            "overhead_ablation": {
                "status": "qualified",
                "instrumentation_profile": "ascend-npu-profiler-device-timeline-v1",
                "policy": {
                    "policy_id": "profiling-overhead-error-budget",
                    "version": "1.0.0",
                    "maximum_overhead_ratio": 0.05,
                    "minimum_independent_sessions": 2,
                },
                "selection": {
                    "session_ids": ["selection-1", "selection-2"],
                    "evidence_ref": "artifact://ablation/selection",
                },
                "holdout": {
                    "pair_id": "issue47-pair-qualified",
                    "baseline_lane_id": "issue47-baseline",
                    "diagnostic_lane_id": "issue47-diagnostic",
                    "baseline_session_ids": ["baseline-1", "baseline-2"],
                    "diagnostic_session_ids": ["diagnostic-1", "diagnostic-2"],
                    "baseline_raw_samples_ns": [1_900_000, 1_930_000],
                    "diagnostic_raw_samples_ns": [1_920_000, 1_940_000],
                    "evidence_ref": "artifact://ablation/holdout",
                },
                "evidence_ref": "artifact://ablation/decision",
            },
        },
    }


def test_qualified_pair_publishes_replayable_observed_decomposition(
    tmp_path: Path,
) -> None:
    run = write_schedule_effect_frontier_bundle(
        tmp_path,
        run_id="issue47-qualified-observed-decomposition-v1",
        document=_qualified_input(),
    )

    verification = verify_run_bundle(run)
    assert verification["passed"] is True, verification["failures"]

    manifest = json.loads(
        (run / "run.manifest.json").read_text(encoding="utf-8")
    )
    assert {
        artifact["role"] for artifact in manifest["artifacts"]
    } == {
        "schedule-effect-input",
        "baseline-timing-observation",
        "diagnostic-profiling-observation",
        "observed-decomposition",
    }
    result_artifact = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "observed-decomposition"
    )
    assert result_artifact["inputs"] == [
        "schedule/effects.input.json",
        "observation/baseline-timing.json",
        "observation/diagnostic-profiling.json",
    ]

    baseline = json.loads(
        (run / "schedule/effects.input.json").read_text(encoding="utf-8")
    )["baseline_timing_lane"]
    result = json.loads(
        (run / "observation/observed-decomposition.json").read_text(
            encoding="utf-8"
        )
    )
    assert baseline["raw_samples_ns"] == [1_900_000, 1_921_530, 1_930_000]
    assert result["baseline_e2e_observation"]["median_ns"] == 1_921_530
    assert result["diagnostic_instrumentation_timing"] == {
        "clock_domain": "host-monotonic",
        "elapsed_ns": 50_840_320,
        "source": "torch-npu-profiler-session",
    }
    assert result["observed_decomposition"]["status"] == "available"
    assert result["observed_decomposition"]["e2e_duration_ns"] == 1_930_000
    assert result["observed_decomposition"]["reconciliation"] == {
        "all_attributed_ns": 1_200_000,
        "unattributed_ns": 730_000,
        "overlap_ns": 200_000,
        "reconciled_e2e_ns": 1_930_000,
    }
    inclusive_parent = result["observed_decomposition"]["inclusive_parents"][0]
    assert inclusive_parent["stable_path"] == "semantic/workload/layer_0"
    assert inclusive_parent["additive"] is False
    assert inclusive_parent["duration_ns"] == 1_700_000


def test_failed_ablation_preserves_baseline_and_marks_decomposition_unavailable(
    tmp_path: Path,
) -> None:
    document = deepcopy(_qualified_input())
    document["pair_id"] = "issue47-pair-over-budget"
    baseline = document["baseline_timing_lane"]
    diagnostic = document["diagnostic_profiling_lane"]
    baseline["pair_id"] = document["pair_id"]
    diagnostic["pair_id"] = document["pair_id"]
    diagnostic["overhead_ablation"]["holdout"]["pair_id"] = document[
        "pair_id"
    ]
    diagnostic["overhead_ablation"]["holdout"][
        "diagnostic_raw_samples_ns"
    ] = [2_900_000, 2_940_000]

    run = write_schedule_effect_frontier_bundle(
        tmp_path,
        run_id="issue47-over-budget-observed-decomposition-v1",
        document=document,
    )

    verification = verify_run_bundle(run)
    assert verification["passed"] is True, verification["failures"]
    result = json.loads(
        (run / "observation/observed-decomposition.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["baseline_e2e_observation"]["status"] == "valid"
    assert result["baseline_e2e_observation"]["median_ns"] == 1_921_530
    observed = result["observed_decomposition"]
    assert observed["status"] == "unavailable"
    assert observed["reason_code"] == "profiling-overhead-error-budget-exceeded"
    assert observed["e2e_duration_ns"] is None
    assert observed["leaves"] == []
    assert observed["reconciliation"] == {
        "all_attributed_ns": None,
        "unattributed_ns": None,
        "overlap_ns": None,
        "reconciled_e2e_ns": None,
    }
    assert "independent paired baseline/diagnostic holdout" in observed[
        "required_next_measurement"
    ]


def test_missing_ablation_and_device_timeline_publish_structured_unknown(
    tmp_path: Path,
) -> None:
    document = deepcopy(_qualified_input())
    document["pair_id"] = "issue47-pair-missing-ablation"
    baseline = document["baseline_timing_lane"]
    diagnostic = document["diagnostic_profiling_lane"]
    baseline["pair_id"] = document["pair_id"]
    diagnostic["pair_id"] = document["pair_id"]
    diagnostic["source"] = diagnostic["device_timeline"]["source"]
    diagnostic.pop("device_timeline")
    diagnostic["overhead_ablation"] = {"status": "not_provided"}

    run = write_schedule_effect_frontier_bundle(
        tmp_path,
        run_id="issue47-missing-ablation-observed-decomposition-v1",
        document=document,
    )

    assert verify_run_bundle(run)["passed"] is True
    result = json.loads(
        (run / "observation/observed-decomposition.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["baseline_e2e_observation"]["status"] == "valid"
    observed = result["observed_decomposition"]
    assert observed["status"] == "unavailable"
    assert observed["reason_code"] == "profiling-overhead-ablation-missing"
    assert observed["leaves"] == []


def test_resigned_decomposition_or_pairing_tamper_fails_public_verifier(
    tmp_path: Path,
) -> None:
    source = write_schedule_effect_frontier_bundle(
        tmp_path / "source",
        run_id="issue47-tamper-source-v1",
        document=_qualified_input(),
    )

    decomposition_tampered = tmp_path / "decomposition-tampered"
    shutil.copytree(source, decomposition_tampered)
    decomposition_path = (
        decomposition_tampered / "observation/observed-decomposition.json"
    )
    decomposition = json.loads(decomposition_path.read_text(encoding="utf-8"))
    decomposition["observed_decomposition"]["leaves"][0]["duration_ns"] += 1
    manifest_path = decomposition_tampered / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "observed-decomposition"
    )["sha256"] = _write_json(decomposition_path, decomposition)
    _write_json(manifest_path, manifest)
    verification = verify_run_bundle(decomposition_tampered)
    assert verification["passed"] is False
    assert "observed decomposition derivation mismatch" in verification["failures"]

    pairing_tampered = tmp_path / "pairing-tampered"
    shutil.copytree(source, pairing_tampered)
    input_path = pairing_tampered / "schedule/effects.input.json"
    schedule_input = json.loads(input_path.read_text(encoding="utf-8"))
    schedule_input["diagnostic_profiling_lane"]["identity"][
        "candidate_id"
    ] = "different-candidate"
    manifest_path = pairing_tampered / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "schedule-effect-input"
    )["sha256"] = _write_json(input_path, schedule_input)
    _write_json(manifest_path, manifest)
    verification = verify_run_bundle(pairing_tampered)
    assert verification["passed"] is False
    assert "observed decomposition replay failed" in verification["failures"]


def test_published_ascend_evidence_preserves_frozen_baseline_boundary() -> None:
    verification = verify_run_bundle(PUBLISHED_BUNDLE)
    assert verification["passed"] is True, verification["failures"]
    result = json.loads(
        (PUBLISHED_BUNDLE / "observation/observed-decomposition.json").read_text(
            encoding="utf-8"
        )
    )
    baseline = result["baseline_e2e_observation"]
    assert baseline["median_ns"] == 1_921_530
    assert baseline["raw_sample_count"] == 20
    assert baseline["warmup"] == {
        "iterations": 20,
        "outside_timing_boundary": True,
    }
    assert baseline["correctness"]["semantic_leaf_count"] == 52
    assert baseline["correctness"]["no_cpu_fallback"] is True
    assert result["diagnostic_instrumentation_timing"]["elapsed_ns"] == (
        50_840_320
    )
    assert result["observed_decomposition"]["status"] == "unavailable"
    assert result["observed_decomposition"]["reason_code"] == (
        "profiling-overhead-error-budget-exceeded"
    )
    assert result["observed_decomposition"]["evidence_boundaries"] == [
        "profiling-overhead-error-budget-exceeded",
        "profiler-device-timeline-export-incomplete",
    ]
    assert result["observed_decomposition"]["reconciliation"] == {
        "all_attributed_ns": None,
        "unattributed_ns": None,
        "overlap_ns": None,
        "reconciled_e2e_ns": None,
    }
