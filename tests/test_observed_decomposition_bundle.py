from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
import os
from pathlib import Path

from groundupscale.observed_decomposition import timing_summary
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
    document = {
        "schema": "groundupscale.dev/schedule-effect-input/v1alpha1",
        "pair_id": "issue47-pair-qualified",
        "identity": IDENTITY,
        "baseline_timing_lane": {
            "lane_id": "issue47-baseline",
            "pair_id": "issue47-pair-qualified",
            "identity": IDENTITY,
            "instrumentation_profile": "ascend-npu-baseline-timing-v1",
            "raw_samples_ns": [1_900_000, 1_921_530, 1_930_000],
            "normalized_window_samples_ns": [
                [1_895_000, 1_900_000],
                [1_920_000, 1_921_530],
                [1_925_000, 1_930_000],
            ],
            "windows_per_sample": 2,
            "warmup": {"iterations": 20, "outside_timing_boundary": True},
            "timer": {"kind": "npu-event", "resolution_ns": 1.0},
            "synchronization": "same-stream-event-then-stream-sync",
            "correctness": {"passed": True, "semantic_leaf_count": 52},
            "source": {
                "run_id": "ascend-910b2-transformer-demo-20260811-v1",
                "evidence_ref": (
                    "run-bundle://ascend-910b2-transformer-demo-20260811-v1/"
                    "baseline.json"
                ),
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
                    "run_id": "issue47-synthetic-diagnostic-source-v1",
                    "collector": "torch-npu-profiler",
                    "evidence_ref": (
                        "run-bundle://issue47-synthetic-diagnostic-source-v1/"
                        "diagnostic.json"
                    ),
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
                    "maximum_iqr_fraction_of_median": 0.05,
                    "minimum_independent_sessions": 2,
                },
                "selection": {
                    "session_ids": ["selection-1", "selection-2"],
                    "evidence_ref": "artifact://ablation/selection",
                },
                "holdout": {
                    "identity": IDENTITY,
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
    baseline = document["baseline_timing_lane"]
    baseline["timing_summary"] = timing_summary(baseline["raw_samples_ns"])
    return document


def _source_runs(
    tmp_path: Path, destination: Path, document: dict[str, object]
) -> list[dict[str, str]]:
    sources = [
        (
            document["baseline_timing_lane"]["source"],
            "baseline-observation",
        ),
        (
            document["diagnostic_profiling_lane"]["device_timeline"]["source"],
            "diagnostic-observation",
        ),
    ]
    result = []
    for source, role in sources:
        source_root = tmp_path / "sources" / source["run_id"]
        source_root.mkdir(parents=True, exist_ok=True)
        relative_artifact = source["evidence_ref"].split(
            f"run-bundle://{source['run_id']}/", 1
        )[1]
        artifact_path = source_root / relative_artifact
        artifact_path.write_text("{}\n", encoding="utf-8")
        source["artifact_sha256"] = hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest()
        manifest = {
            "schema": "groundupscale.dev/run-manifest/v1alpha1",
            "run_id": source["run_id"],
            "status": "completed",
            "artifacts": [
                {
                    "role": role,
                    "path": relative_artifact,
                    "media_type": "application/json",
                    "schema": None,
                    "sha256": source["artifact_sha256"],
                    "produced_by": "issue47-synthetic-source@1",
                }
            ],
        }
        manifest_path = source_root / "run.manifest.json"
        _write_json(manifest_path, manifest)
        result.append(
            {
                "run_id": source["run_id"],
                "path": os.path.relpath(source_root, destination),
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
            }
        )
    return result


def _write_bundle(
    tmp_path: Path, *, run_id: str, document: dict[str, object]
) -> Path:
    destination = tmp_path / "runs" / run_id
    return write_schedule_effect_frontier_bundle(
        tmp_path,
        run_id=run_id,
        document=document,
        source_runs=_source_runs(tmp_path, destination, document),
    )


def test_qualified_pair_publishes_replayable_observed_decomposition(
    tmp_path: Path,
) -> None:
    run = _write_bundle(
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

    run = _write_bundle(
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
    assert "independent exact-identity paired baseline/diagnostic holdout" in observed[
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
    synthetic_timeline = diagnostic["device_timeline"]
    diagnostic.pop("device_timeline")
    diagnostic["overhead_ablation"] = {"status": "not_provided"}

    diagnostic["device_timeline"] = synthetic_timeline
    source_runs = _source_runs(
        tmp_path,
        tmp_path / "runs/issue47-missing-ablation-observed-decomposition-v1",
        document,
    )
    diagnostic.pop("device_timeline")
    run = write_schedule_effect_frontier_bundle(
        tmp_path,
        run_id="issue47-missing-ablation-observed-decomposition-v1",
        document=document,
        source_runs=source_runs,
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
    source = _write_bundle(
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
        "exact-identity-profiling-ablation-missing"
    )
    assert result["observed_decomposition"]["evidence_boundaries"] == [
        "exact-identity-profiling-ablation-missing",
        "profiler-device-timeline-export-incomplete",
    ]
    assert result["observed_decomposition"]["reconciliation"] == {
        "all_attributed_ns": None,
        "unattributed_ns": None,
        "overlap_ns": None,
        "reconciled_e2e_ns": None,
    }
    assert "export a complete same-identity Ascend device timeline" in result[
        "observed_decomposition"
    ]["required_next_measurement"]
    assert baseline["timing_summary"] == timing_summary(
        json.loads(
            (PUBLISHED_BUNDLE / "schedule/effects.input.json").read_text(
                encoding="utf-8"
            )
        )["baseline_timing_lane"]["raw_samples_ns"]
    )


def test_ablation_requires_exact_identity_and_uncertainty_budget(
    tmp_path: Path,
) -> None:
    for scenario in ("identity", "uncertainty"):
        document = deepcopy(_qualified_input())
        if scenario == "identity":
            holdout = document["diagnostic_profiling_lane"][
                "overhead_ablation"
            ]["holdout"]
            holdout["identity"] = deepcopy(holdout["identity"])
            holdout["identity"]["shape"] = [1, 256, 512]
        else:
            document["diagnostic_profiling_lane"]["overhead_ablation"][
                "holdout"
            ]["diagnostic_raw_samples_ns"] = [1_600_000, 2_200_000]
        run = _write_bundle(
            tmp_path / scenario,
            run_id=f"issue47-{scenario}-ablation-v1",
            document=document,
        )
        result = json.loads(
            (run / "observation/observed-decomposition.json").read_text(
                encoding="utf-8"
            )
        )
        assert result["observed_decomposition"]["status"] == "unavailable"
        assert result["observed_decomposition"]["reason_code"] in {
            "profiling-overhead-ablation-unqualified",
            "profiling-overhead-error-budget-exceeded",
        }


def test_malformed_manifest_and_lane_fail_closed_without_exception(
    tmp_path: Path,
) -> None:
    document = _qualified_input()
    source = _write_bundle(
        tmp_path / "source",
        run_id="issue47-malformed-source-v1",
        document=document,
    )
    malformed_artifact = tmp_path / "malformed-artifact"
    shutil.copytree(source, malformed_artifact)
    manifest_path = malformed_artifact / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0].pop("path")
    _write_json(manifest_path, manifest)
    verification = verify_run_bundle(malformed_artifact)
    assert verification["passed"] is False
    assert "invalid artifact entry" in verification["failures"]

    malformed_lane = tmp_path / "malformed-lane"
    shutil.copytree(source, malformed_lane)
    input_path = malformed_lane / "schedule/effects.input.json"
    schedule_input = json.loads(input_path.read_text(encoding="utf-8"))
    schedule_input["baseline_timing_lane"] = []
    manifest_path = malformed_lane / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "schedule-effect-input"
    )["sha256"] = _write_json(input_path, schedule_input)
    _write_json(manifest_path, manifest)
    verification = verify_run_bundle(malformed_lane)
    assert verification["passed"] is False
    assert "invalid schedule effect measurement lanes" in verification["failures"]


def test_fully_resigned_source_digest_forgery_fails_verifier(
    tmp_path: Path,
) -> None:
    document = _qualified_input()
    source = _write_bundle(
        tmp_path / "source",
        run_id="issue47-source-forgery-v1",
        document=document,
    )
    tampered = tmp_path / "fully-resigned"
    shutil.copytree(source, tampered)
    input_path = tampered / "schedule/effects.input.json"
    schedule_input = json.loads(input_path.read_text(encoding="utf-8"))
    forged = "f" * 64
    schedule_input["baseline_timing_lane"]["source"][
        "artifact_sha256"
    ] = forged
    baseline_path = tampered / "observation/baseline-timing.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["source"]["artifact_sha256"] = forged
    from groundupscale.observed_decomposition import compose_observed_decomposition

    result = compose_observed_decomposition(schedule_input)
    result_path = tampered / "observation/observed-decomposition.json"
    manifest_path = tampered / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    updates = {
        "schedule-effect-input": _write_json(input_path, schedule_input),
        "baseline-timing-observation": _write_json(baseline_path, baseline),
        "observed-decomposition": _write_json(result_path, result),
    }
    for artifact in manifest["artifacts"]:
        if artifact["role"] in updates:
            artifact["sha256"] = updates[artifact["role"]]
    _write_json(manifest_path, manifest)
    verification = verify_run_bundle(tampered)
    assert verification["passed"] is False
    assert "schedule effect source lineage mismatch" in verification["failures"]
