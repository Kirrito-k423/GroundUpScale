from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from groundupscale.diagnostics import (
    DiagnosticBundleIntegrityError,
    diagnose_run_bundle,
)
from _diagnostic_test_support import (
    canonical_digest as _canonical_digest,
    write_json as _write_json,
)


def _write_frozen_m4_bundle(
    tmp_path: Path,
    *,
    invalid_qualification: str | None = None,
    invalid_observation: str | None = None,
    invalid_schedule_policy: bool = False,
    incomplete_execution_domain: bool = False,
    incomplete_identity: str | None = None,
    projection_corruption: str | None = None,
    missing_layer: str | None = None,
    observation_samples_ns: list[int] | None = None,
    resource_evidence_complete: bool = True,
    resource_units_match: bool = True,
) -> Path:
    run = tmp_path / "m4-exact-shape"
    inputs = {
        "resolved_configuration": (
            {}
            if incomplete_identity == "configuration"
            else {
                "analysis_plan": "mac-cpu-prefill",
                "benchmark_case": "matmul-q-proj",
            }
        ),
        "resolved_ir": (
            {}
            if incomplete_identity == "ir"
            else {
                "semantic_node": (
                    "semantic/workload/main/model-call/model/layers/0/q_proj"
                ),
                "operation": "MatMul",
            }
        ),
        "hardware": (
            {}
            if incomplete_identity == "hardware"
            else {
                "device": "Apple M4 CPU",
                "partition": "host",
                "topology": "single-socket",
                "software": "torch-2.13-cpu",
                "power_clock": {
                    "power_policy": "balanced",
                    "clock_policy": "automatic",
                },
            }
        ),
        "cohort_id": "apple-m4-cpu-darwin-arm64-torch2.13-baseline-v1",
        "execution_domain": (
            {}
            if incomplete_execution_domain
            else {
                "shape": {"m": 128, "k": 256, "n": 256},
                "dtype": "float32",
                "layout": "row-major-contiguous",
                "alignment_bytes": 16,
                "threads": 10,
                "execution_mode": "eager",
                "affinity": "all-performance-cores",
                "numa": "single-domain",
                "context": "default",
                "stream": "not_applicable",
                "concurrency": 1,
            }
        ),
    }
    candidate_id = (
        "" if invalid_qualification == "blank-identity" else "torch.matmul.cpu"
    )
    evidence = {
        "candidate": {
            "candidate_id": candidate_id,
            "family": "pytorch-cpu-matmul",
            "coverage": "C1_SAME_FAMILY",
            "implementation_digest": "sha256:candidate-001",
            "exact_shape_best_of_correct": {
                "passed": invalid_qualification != "best-of-correct",
                "winner_candidate_id": candidate_id,
                "eligible_candidate_ids": [candidate_id],
                "search_session_ids": (
                    ["holdout-session-001"]
                    if invalid_qualification == "holdout-overlap"
                    else ["search-session-001"]
                ),
                "evidence_ref": "artifact://frontier/candidate-search.json",
            },
        },
        "correctness": {
            "passed": invalid_qualification != "correctness",
            "oracle": "torch-reference-float64",
            "policy_ref": "policy://m4-matmul-correctness/v1",
            "evidence_ref": "artifact://observation/correctness.json",
        },
        "environment": {
            "eligible": invalid_qualification != "environment",
            "preflight_ref": "artifact://resolved/environment.json",
        },
        "measurement_adapter": {
            "adapter_id": "apple-m4-cpu",
            "adapter_version": "v1",
            "protocol_id": "exact-shape-diagnostic",
            "protocol_version": "v1",
            "evidence_ref": "artifact://adapter/apple-m4-cpu.json",
        },
        "measurement_capability_manifest": {
            "manifest_id": "manifest-apple-m4-cpu-v1",
            "adapter_id": "apple-m4-cpu",
            "cohort_id": inputs["cohort_id"],
            "fields": [
                {
                    "field": "timer.primary",
                    "status": "measured",
                    "required_for_anchor": True,
                    "source": "mach-continuous-time",
                    "scope": "exact-shape-operator",
                    "attribution": "direct",
                    "intrusion": "baseline",
                    "value": 1,
                },
                {
                    "field": "counter.l2_cache_misses",
                    "status": "not_requested",
                    "required_for_anchor": False,
                    "source": "apple-m4-cpu",
                    "scope": "exact-shape-operator",
                    "attribution": "direct",
                    "intrusion": "diagnostic",
                },
            ],
            "evidence_ref": "artifact://adapter/apple-m4-cpu-capabilities.json",
        },
        "communication_identity": {"status": "not_applicable"},
        "baseline_timing_lane": {
            "lane_id": "baseline-m4-q-proj-001",
            "pair_id": "lane-pair-m4-q-proj-001",
            "cohort_id": inputs["cohort_id"],
            "candidate_id": candidate_id,
            "execution_domain": inputs["execution_domain"],
            "instrumentation_profile": "baseline-timing/v1",
            "observation_validity": (
                "QUARANTINED"
                if invalid_observation == "validity"
                else "COLLECTED"
            ),
            "frontier_role": "NONE",
            "completion_boundary": {
                "kind": "synchronous-cpu-call-return",
                "closed": invalid_observation != "boundary",
                "threadpool_joined": True,
            },
            "timer": {
                "source": (
                    "" if invalid_observation == "timer" else "mach-continuous-time"
                ),
                "resolution_ns": 1,
                "monotonic": True,
            },
            "warmup": {"iterations": 5, "converged": True},
            "raw_samples_ns": (
                observation_samples_ns
                if observation_samples_ns is not None
                else [1_560_000, 1_600_000, 1_640_000]
            ),
            "excluded_samples": [],
            "evidence_ref": "artifact://observation/raw/benchmark.json",
        },
        "diagnostic_profiling_lane": {
            "lane_id": "diagnostic-m4-q-proj-001",
            "pair_id": "lane-pair-m4-q-proj-001",
            "paired_baseline_lane_id": "baseline-m4-q-proj-001",
            "cohort_id": inputs["cohort_id"],
            "candidate_id": candidate_id,
            "execution_domain": inputs["execution_domain"],
            "instrumentation_profile": "diagnostic-counters/v1",
            "observation_validity": "COLLECTED",
            "frontier_role": "NONE",
            "completion_boundary": {
                "kind": "synchronous-cpu-call-return",
                "closed": True,
                "threadpool_joined": True,
            },
            "timer": {
                "source": "mach-continuous-time",
                "resolution_ns": 1,
                "monotonic": True,
            },
            "raw_samples_ns": [1_700_000, 1_720_000, 1_740_000],
            "overhead_ablation": {"status": "not_provided"},
            "evidence_ref": "artifact://observation/raw/diagnostic.json",
        },
        "frontier_anchors": [
            {
                "anchor_id": "anchor-m4-q-proj-001",
                "observation_validity": "QUALIFIED",
                "frontier_role": "ACTIVE",
                "candidate_id": candidate_id,
                "cohort_id": inputs["cohort_id"],
                "execution_domain": inputs["execution_domain"],
                "correctness_passed": True,
                "baseline_lane_id": "baseline-m4-q-proj-holdout-001",
                "instrumentation_profile": "baseline-timing/v1",
                "completion_boundary": {
                    "kind": "synchronous-cpu-call-return",
                    "closed": True,
                    "threadpool_joined": True,
                },
                "timer": (
                    None
                    if invalid_qualification == "timer"
                    else {
                        "source": "mach-continuous-time",
                        "resolution_ns": 1,
                        "monotonic": True,
                    }
                ),
                "warmup": {
                    "iterations": 5,
                    "converged": invalid_qualification != "warmup",
                },
                "raw_timing_ns": [1_180_000, 1_200_000, 1_220_000],
                "holdout": {
                    "passed": True,
                    "latency_ns": 1_200_000,
                    "session_ids": (
                        ["holdout-session-001"]
                        if invalid_qualification == "repeatability"
                        else ["holdout-session-001", "holdout-session-002"]
                    ),
                    "evidence_ref": "artifact://frontier/anchor-holdout.json",
                },
                "evidence_ref": "artifact://frontier/anchor-m4-q-proj-001.json",
            }
        ],
        "resource_physical_floor": {
            "status": "known",
            "value_ns": 320_000,
            "may_be_unattainable": True,
            "policy_ref": "policy://m4-resource-floor/v1",
            "evidence_refs": ["artifact://prediction/hardware-backend.json"],
            "combination": "max-explicit-overlap",
            "resource_terms": (
                [
                    {
                        "resource": "compute.fp32",
                        "validated_rate_resource": "compute.fp32",
                        "minimum_demand": 320_000_000,
                        "demand_unit": (
                            "flop" if resource_units_match else "byte"
                        ),
                        "validated_rate_per_second": 1_000_000_000_000,
                        "rate_unit": "flop/s",
                        "validated": True,
                        "cohort_id": inputs["cohort_id"],
                        "execution_domain": inputs["execution_domain"],
                        "evidence_ref": (
                            "artifact://prediction/hardware-backend.json"
                        ),
                    }
                ]
                if resource_evidence_complete
                else []
            ),
        },
        "single_node_schedule": {
            "schedule_id": "schedule-m4-q-proj-single-node-v1",
            "version": "1",
            "candidate_id": candidate_id,
            "dependencies": [],
            "transformations": [],
            "overlap_claims": [],
            "evidence_refs": ["artifact://schedule/single-node.json"],
        },
        "policies": {
            "qualification": {
                "policy_id": "anchor-qualification",
                "version": "v1",
                "scope": "MatMul/float32/Apple-M4-CPU",
                "minimum_independent_sessions": 2,
                "change_reason": "first production exact-Shape slice",
                "revalidation": "on cohort or implementation change",
            },
            "observation": {
                "policy_id": (
                    ""
                    if invalid_observation == "policy"
                    else "baseline-observation"
                ),
                "version": "v1",
                "scope": "synchronous CPU operator",
                "change_reason": "first production exact-Shape slice",
                "revalidation": "on timer or completion-boundary change",
            },
            "schedule": {
                "policy_id": (
                    "" if invalid_schedule_policy else "single-node-schedule"
                ),
                "version": "v1",
                "scope": "one unfused operator node",
                "change_reason": "first production exact-Shape slice",
                "revalidation": "on dependency or overlap change",
            },
            "cohort": {
                "policy_id": "hardware-validity-cohort",
                "version": "v1",
                "scope": "exact-shape adapter evidence",
                "change_reason": "make cohort split and retry explicit",
                "revalidation": "on stable identity or transient health change",
                "maximum_retry_attempts": 2,
            },
        },
    }
    evidence["cohort_evidence"] = {
        "reference_cohort_id": inputs["cohort_id"],
        "reference_identity": {
            "device": inputs["hardware"].get("device"),
            "partition": inputs["hardware"].get("partition"),
            "topology": inputs["hardware"].get("topology"),
            "software": inputs["hardware"].get("software"),
            "power_clock": inputs["hardware"].get("power_clock"),
            "numeric_execution": {
                key: inputs["execution_domain"].get(key)
                for key in (
                    "dtype",
                    "layout",
                    "alignment_bytes",
                    "threads",
                    "execution_mode",
                )
            },
            "timer_protocol": {
                "source": "mach-continuous-time",
                "resolution_ns": 1,
                "monotonic": True,
                "completion_kind": "synchronous-cpu-call-return",
                "duration_reducer": None,
                "adapter_id": "apple-m4-cpu",
                "adapter_version": "v1",
                "protocol_id": "exact-shape-diagnostic",
                "protocol_version": "v1",
            },
            "execution_context": {
                key: inputs["execution_domain"].get(key)
                for key in (
                    "affinity",
                    "numa",
                    "context",
                    "stream",
                    "concurrency",
                )
            },
            "communication": evidence["communication_identity"],
        },
        "transient_failures": [],
        "retry_attempt": 1,
        "evidence_ref": "artifact://cohort/apple-m4-cpu.json",
    }
    evidence["cohort_evidence"]["observed_identity"] = evidence[
        "cohort_evidence"
    ]["reference_identity"]
    evidence["measurement_adapter"]["operation_evidence"] = [
        {
            "operation": operation,
            "evidence_ref": f"artifact://adapter-operations/{operation}.json",
        }
        for operation in (
            "discover_capabilities",
            "fingerprint_cohort",
            "preflight",
            "build_timing_plan",
            "collect",
        )
    ]
    evidence["timing_plan"] = {
        "case": {
            "benchmark_case": inputs["resolved_configuration"].get(
                "benchmark_case"
            ),
            "semantic_node": inputs["resolved_ir"].get("semantic_node"),
            "execution_domain": inputs["execution_domain"],
        },
        "pair_id": "lane-pair-m4-q-proj-001",
        "baseline_lane_id": "baseline-m4-q-proj-001",
        "diagnostic_lane_id": "diagnostic-m4-q-proj-001",
        "completion_boundary": evidence["baseline_timing_lane"][
            "completion_boundary"
        ],
        "evidence_ref": "artifact://adapter-operations/timing-plan.json",
    }
    if projection_corruption == "configuration-type":
        inputs["resolved_configuration"] = "invalid"
    elif projection_corruption == "ir-type":
        inputs["resolved_ir"] = "invalid"
    elif projection_corruption == "hardware-type":
        inputs["hardware"] = "invalid"
    elif projection_corruption == "missing-lane-id":
        evidence["baseline_timing_lane"].pop("lane_id")
    if invalid_qualification == "minimum-session-policy":
        evidence["policies"]["qualification"][
            "minimum_independent_sessions"
        ] = 1
    missing_evidence_key = {
        "resource_physical_floor": "resource_physical_floor",
        "operator_achievable_frontier": "frontier_anchors",
        "schedule_achievable_frontier": "single_node_schedule",
        "observation": "baseline_timing_lane",
    }.get(missing_layer)
    if missing_evidence_key is not None:
        evidence.pop(missing_evidence_key)
    document = {
        "schema": "groundupscale.dev/diagnostic-evidence/v1alpha1",
        **inputs,
        **evidence,
        "digests": {
            "input_sha256": _canonical_digest(inputs),
            "evidence_sha256": _canonical_digest(evidence),
        },
    }
    evidence_path = run / "diagnostic/evidence.json"
    evidence_artifact_digest = _write_json(evidence_path, document)
    manifest = {
        "schema": "groundupscale.dev/run-manifest/v1alpha1",
        "run_id": "m4-exact-shape",
        "status": "completed",
        "device": "cpu",
        "hardware_cohort": inputs["cohort_id"],
        "artifacts": [
            {
                "role": "diagnostic-evidence",
                "path": "diagnostic/evidence.json",
                "schema": document["schema"],
                "media_type": "application/json",
                "sha256": evidence_artifact_digest,
                "produced_by": "groundupscale-test-fixture",
                "inputs": [],
            }
        ],
    }
    _write_json(run / "run.manifest.json", manifest)
    return run


def test_m4_exact_shape_bundle_projects_four_independent_axes_and_evidence(
    tmp_path: Path,
) -> None:
    run = _write_frozen_m4_bundle(tmp_path)

    result = diagnose_run_bundle(run)

    assert result["status"] == "complete"
    assert result["axes"] == {
        "resource_physical_floor": {
            "status": "known",
            "value_ns": 320_000,
            "may_be_unattainable": True,
            "evidence_refs": ["artifact://prediction/hardware-backend.json"],
        },
        "operator_achievable_frontier": {
            "status": "known",
            "value_ns": 1_200_000,
            "anchor_id": "anchor-m4-q-proj-001",
            "candidate_id": "torch.matmul.cpu",
            "evidence_refs": ["artifact://frontier/anchor-m4-q-proj-001.json"],
        },
        "schedule_achievable_frontier": {
            "status": "known",
            "value_ns": 1_200_000,
            "schedule_id": "schedule-m4-q-proj-single-node-v1",
            "operator_frontier_ref": "anchor-m4-q-proj-001",
            "evidence_refs": ["artifact://schedule/single-node.json"],
        },
        "observation": {
            "status": "known",
            "value_ns": 1_600_000,
            "observation_validity": "COLLECTED",
            "frontier_role": "NONE",
            "lane_id": "baseline-m4-q-proj-001",
            "evidence_refs": ["artifact://observation/raw/benchmark.json"],
        },
    }
    assert result["comparisons"]["physical_floor_to_observation"] == {
        "distance_ns": 1_280_000,
        "prediction_error_ns": None,
        "error_status": "not-evaluable-physical-floor",
    }
    assert result["comparisons"]["operator_frontier_to_observation"] == {
        "distance_ns": 400_000,
        "prediction_error_ns": 400_000,
        "error_status": "evaluated",
    }
    assert result["evidence"]["resolved_configuration"]["analysis_plan"] == (
        "mac-cpu-prefill"
    )
    assert result["evidence"]["resolved_ir"]["operation"] == "MatMul"
    assert result["evidence"]["hardware"]["device"] == "Apple M4 CPU"
    assert result["evidence"]["cohort_id"].startswith("apple-m4-cpu")
    assert result["evidence"]["execution_domain"]["shape"] == {
        "m": 128,
        "k": 256,
        "n": 256,
    }
    assert result["evidence"]["candidate"]["candidate_id"] == (
        "torch.matmul.cpu"
    )
    assert result["evidence"]["correctness"]["passed"] is True
    assert result["evidence"]["baseline_timing_lane"]["completion_boundary"] == {
        "kind": "synchronous-cpu-call-return",
        "closed": True,
        "threadpool_joined": True,
    }
    assert result["evidence"]["baseline_timing_lane"]["raw_samples_ns"] == [
        1_560_000,
        1_600_000,
        1_640_000,
    ]
    assert result["evidence"]["policies"]["qualification"]["policy_id"] == (
        "anchor-qualification"
    )
    assert result["evidence"]["policies"]["qualification"]["version"] == "v1"
    assert len(result["digests"]["input_sha256"]) == 64
    assert len(result["digests"]["evidence_sha256"]) == 64
    assert len(result["derivation"]["derivation_id"]) == 64


@pytest.mark.parametrize(
    ("missing_layer", "reason_code"),
    [
        ("resource_physical_floor", "missing-resource-physical-floor"),
        (
            "operator_achievable_frontier",
            "no-qualified-active-exact-shape-anchor",
        ),
        (
            "schedule_achievable_frontier",
            "missing-single-node-schedule-evidence",
        ),
        ("observation", "missing-baseline-timing-lane"),
    ],
)
def test_missing_axis_is_preserved_as_structured_unknown(
    tmp_path: Path, missing_layer: str, reason_code: str
) -> None:
    run = _write_frozen_m4_bundle(tmp_path, missing_layer=missing_layer)

    result = diagnose_run_bundle(run)

    assert result["status"] == "partial"
    assert result["axes"][missing_layer] == {
        "status": "unknown",
        "reason_code": reason_code,
        "evidence_refs": [],
    }


def test_cli_machine_and_human_views_project_the_same_derivation(
    tmp_path: Path,
) -> None:
    run = _write_frozen_m4_bundle(tmp_path)
    direct = diagnose_run_bundle(run)

    machine = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(run),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    human = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(run),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert machine.returncode == 0, machine.stderr
    assert json.loads(machine.stdout) == direct
    assert human.returncode == 0, human.stderr
    assert direct["derivation"]["derivation_id"] in human.stdout
    assert "Resource Physical Floor: 0.320 ms" in human.stdout
    assert "Operator Achievable Frontier: 1.200 ms" in human.stdout
    assert "Schedule Achievable Frontier: 1.200 ms" in human.stdout
    assert "Observation: 1.600 ms" in human.stdout
    assert "not prediction error" in human.stdout
    assert direct["evidence"]["cohort_id"] in human.stdout
    assert '"dtype":"float32"' in human.stdout
    assert "baseline-m4-q-proj-001" in human.stdout
    assert "anchor-qualification/v1" in human.stdout
    assert direct["digests"]["input_sha256"] in human.stdout
    assert direct["digests"]["evidence_sha256"] in human.stdout


def test_replay_is_deterministic_and_authored_digest_rejects_tampering(
    tmp_path: Path,
) -> None:
    run = _write_frozen_m4_bundle(tmp_path)

    assert diagnose_run_bundle(run) == diagnose_run_bundle(run)

    evidence_path = run / "diagnostic/evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["baseline_timing_lane"]["raw_samples_ns"].append(9_999_999)
    changed_artifact_digest = _write_json(evidence_path, evidence)
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["sha256"] = changed_artifact_digest
    _write_json(manifest_path, manifest)

    with pytest.raises(
        DiagnosticBundleIntegrityError,
        match="diagnostic evidence_sha256 mismatch",
    ):
        diagnose_run_bundle(run)

    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(run),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "diagnostic evidence_sha256 mismatch" in rejected.stderr


def test_slower_observation_does_not_lower_the_active_operator_frontier(
    tmp_path: Path,
) -> None:
    first = diagnose_run_bundle(
        _write_frozen_m4_bundle(
            tmp_path / "first",
            observation_samples_ns=[1_260_000, 1_300_000, 1_340_000],
        )
    )
    slower = diagnose_run_bundle(
        _write_frozen_m4_bundle(
            tmp_path / "slower",
            observation_samples_ns=[2_260_000, 2_300_000, 2_340_000],
        )
    )

    assert first["axes"]["observation"]["value_ns"] == 1_300_000
    assert slower["axes"]["observation"]["value_ns"] == 2_300_000
    assert first["axes"]["operator_achievable_frontier"] == slower["axes"][
        "operator_achievable_frontier"
    ]


@pytest.mark.parametrize(
    "invalid_qualification",
    [
        "environment",
        "correctness",
        "warmup",
        "timer",
        "best-of-correct",
        "repeatability",
        "minimum-session-policy",
        "blank-identity",
        "holdout-overlap",
    ],
)
def test_incomplete_anchor_qualification_cannot_produce_a_frontier(
    tmp_path: Path, invalid_qualification: str
) -> None:
    result = diagnose_run_bundle(
        _write_frozen_m4_bundle(
            tmp_path, invalid_qualification=invalid_qualification
        )
    )

    assert result["axes"]["operator_achievable_frontier"] == {
        "status": "unknown",
        "reason_code": "no-qualified-active-exact-shape-anchor",
        "evidence_refs": [],
    }
    assert result["axes"]["schedule_achievable_frontier"] == {
        "status": "unknown",
        "reason_code": "operator-frontier-unknown",
        "evidence_refs": [],
    }


def test_physical_floor_without_demand_and_validated_rate_is_unknown(
    tmp_path: Path,
) -> None:
    result = diagnose_run_bundle(
        _write_frozen_m4_bundle(tmp_path, resource_evidence_complete=False)
    )

    assert result["axes"]["resource_physical_floor"] == {
        "status": "unknown",
        "reason_code": "incomplete-resource-physical-floor-evidence",
        "evidence_refs": [],
    }


def test_resource_floor_with_mismatched_units_is_unknown(tmp_path: Path) -> None:
    result = diagnose_run_bundle(
        _write_frozen_m4_bundle(tmp_path, resource_units_match=False)
    )

    assert result["axes"]["resource_physical_floor"] == {
        "status": "unknown",
        "reason_code": "resource-physical-floor-unit-mismatch",
        "evidence_refs": [],
    }


@pytest.mark.parametrize(
    "invalid_observation",
    ["validity", "boundary", "timer", "policy"],
)
def test_invalid_observation_evidence_is_unknown(
    tmp_path: Path, invalid_observation: str
) -> None:
    result = diagnose_run_bundle(
        _write_frozen_m4_bundle(
            tmp_path, invalid_observation=invalid_observation
        )
    )

    assert result["axes"]["observation"] == {
        "status": "unknown",
        "reason_code": "invalid-baseline-timing-lane",
        "evidence_refs": [],
    }


def test_invalid_schedule_policy_fails_closed(tmp_path: Path) -> None:
    result = diagnose_run_bundle(
        _write_frozen_m4_bundle(tmp_path, invalid_schedule_policy=True)
    )

    assert result["axes"]["schedule_achievable_frontier"] == {
        "status": "unknown",
        "reason_code": "invalid-schedule-policy",
        "evidence_refs": [],
    }


def test_incomplete_execution_domain_cannot_produce_known_axes(
    tmp_path: Path,
) -> None:
    result = diagnose_run_bundle(
        _write_frozen_m4_bundle(tmp_path, incomplete_execution_domain=True)
    )

    assert result["axes"]["resource_physical_floor"]["status"] == "unknown"
    assert result["axes"]["operator_achievable_frontier"]["status"] == "unknown"
    assert result["axes"]["schedule_achievable_frontier"]["status"] == "unknown"
    assert result["axes"]["observation"] == {
        "status": "unknown",
        "reason_code": "incomplete-execution-domain",
        "evidence_refs": [],
    }


@pytest.mark.parametrize(
    "incomplete_identity",
    ["configuration", "ir", "hardware"],
)
def test_incomplete_required_identity_fails_closed_without_breaking_report(
    tmp_path: Path, incomplete_identity: str
) -> None:
    run = _write_frozen_m4_bundle(
        tmp_path, incomplete_identity=incomplete_identity
    )

    result = diagnose_run_bundle(run)

    assert result["status"] == "partial"
    assert all(axis["status"] == "unknown" for axis in result["axes"].values())
    assert {
        axis["reason_code"] for axis in result["axes"].values()
    } == {"incomplete-diagnostic-identity"}
    human = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(run),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert human.returncode == 0, human.stderr
    assert "incomplete-diagnostic-identity" in human.stdout


@pytest.mark.parametrize(
    "projection_corruption",
    ["configuration-type", "ir-type", "hardware-type", "missing-lane-id"],
)
def test_partial_machine_result_always_has_a_safe_human_projection(
    tmp_path: Path, projection_corruption: str
) -> None:
    run = _write_frozen_m4_bundle(
        tmp_path, projection_corruption=projection_corruption
    )

    result = diagnose_run_bundle(run)

    assert result["status"] == "partial"
    human = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(run),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert human.returncode == 0, human.stderr
    assert result["derivation"]["derivation_id"] in human.stdout
    assert "unknown" in human.stdout
