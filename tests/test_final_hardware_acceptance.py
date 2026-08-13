from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import runpy

import pytest

from groundupscale.final_hardware_acceptance import (
    FinalAcceptanceError,
    compose_final_acceptance,
    write_final_acceptance_bundle,
)
from groundupscale.run_bundle import verify_run_bundle


IDENTITY = {
    "model_spec_sha256": "a" * 64,
    "workload_spec_sha256": "b" * 64,
    "analysis_case_sha256": "c" * 64,
    "deployment_intent_sha256": "d" * 64,
    "case": "two-layer-prefill",
    "shape": [1, 512, 512],
    "dtype": "float32",
    "hardware_cohort": "ascend-npu-23b93a89d5fecc79",
    "completion_boundary": "end-npu-event-synchronize-plus-device-synchronize",
}


def _document(*, schedule_status: str = "known") -> dict[str, object]:
    samples = [900.0, 1000.0, 1100.0]
    schedule = {
        "status": schedule_status,
        "selected_complete_schedule_duration_ns": 800.0 if schedule_status == "known" else None,
        "standard_uncertainty_ns": 20.0 if schedule_status == "known" else None,
        "bound_kind": "schedule-achievable-frontier",
        "stable_paths": ["semantic/model/layer_0/op"],
        "leaves": [{
            "stable_path": "semantic/model/layer_0/op",
            "duration_ns": 800.0 if schedule_status == "known" else None,
            "selected_candidate_id": "candidate-a" if schedule_status == "known" else None,
            "evidence_refs": ["run://qualification-a#candidate-a"],
            "standard_uncertainty_ns": 20.0,
            "event_id": "event-a",
        }],
        "edges": [],
        "policy": {"resource_contention": "explicit", "implicit_fusion": "forbidden"},
        "surfaces": [{"operation_class": "MatMul", "anchor_ids": ["anchor-a"], "candidate_ids": ["candidate-a"]}],
        "missing_evidence": [] if schedule_status == "known" else ["operator.matmul.exact-domain"],
        "execution_ir": {
            "status": "known" if schedule_status == "known" else "unknown",
            "critical_path_duration_ns": 800.0 if schedule_status == "known" else None,
            "physical_events": [{"event_id": "event-a", "duration_ns": 800.0}] if schedule_status == "known" else [],
            "dependency_edges": [],
            "resource_claims": [{"event_id": "event-a", "resource_id": "npu:0", "claim_kind": "exclusive"}] if schedule_status == "known" else [],
            "transformations": [],
        },
    }
    return {
        "schema": "groundupscale.dev/final-hardware-acceptance-input/v1alpha1",
        "identity": IDENTITY,
        "source_bundles": [],
        "construction_run_ids": ["construction-a", "qualification-a", "decomposition-a"],
        "source_bundles": [
            {"run_id": run_id, "bundle_kind": kind, "path": run_id,
             "manifest_sha256": "f" * 64, "verification_passed": True,
             "identity": IDENTITY, "source_role": role, "semantic_contract": {"path": "contract.json"}}
            for run_id, kind, role in (
                ("construction-a", "model-e2e-frontier", "schedule-frontier"),
                ("qualification-a", "schedule-effect-frontier", "observed-decomposition"),
                ("decomposition-a", "e2e-gap-report", "gap-report"),
                ("issue50-independent-holdout-a", "transformer-demo", "independent-holdout"),
            )
        ],
        "source_identities": [
            {"run_id": run_id, "identity": IDENTITY}
            for run_id in ("construction-a", "qualification-a", "decomposition-a", "issue50-independent-holdout-a")
        ],
        "schedule": schedule,
        "holdout": {
            "run_id": "issue50-independent-holdout-a",
            "identity": IDENTITY,
            "raw_samples_ns": samples,
            "sample_count": 3,
            "median_ns": 1000.0,
            "iqr_ns": 100.0,
            "standard_uncertainty_ns": 100.0 / 1.349,
            "observation_digest": sha256(
                json.dumps(samples, separators=(",", ":")).encode()
            ).hexdigest(),
            "warmup": {"iterations": 20, "outside_timing_boundary": True},
            "timer": {"primary": "torch.npu.Event.elapsed_time", "unit": "ns"},
            "synchronization": {
                "protocol": IDENTITY["completion_boundary"],
                "passed": True,
            },
            "correctness": {
                "passed": True,
                "no_cpu_fallback": True,
                "semantic_leaf_count": 52,
            },
            "environment": {"device": "npu:0", "visibility": "0", "lock_session": {
                "schema": "groundupscale.dev/ascend-host-lock-session/v1alpha1",
                "issue": 50, "run_id": "issue50-independent-holdout-a",
                "hardware_cohort": IDENTITY["hardware_cohort"],
                "ascend_rt_visible_devices": "0", "logical_device": "npu:0",
                "whole_host_exclusive": True,
                "lock_path": "/home/t00906153/.groundupscale/locks/ascend-910b2-host.lock",
                "wrapper_path": "/home/t00906153/.groundupscale/bin/with-ascend-lock",
                "wrapper_sha256": "22d43618f1c616b2ff70570944c7447cd851aac98bfedb111b7912fc36b94787",
                "measurement_started_at": "2026-08-14T01:00:00+08:00",
                "measurement_ended_at": "2026-08-14T01:01:00+08:00",
                "owner": "issue=50 pid=1 host=test started=2026-08-14T01:00:00+08:00",
            }},
            "gates": {
                "environment": "passed", "correctness": "passed",
                "no_cpu_fallback": "passed", "timing": "passed",
                "synchronization": "passed", "execution_contract": "passed",
            },
        },
        "decomposition": {
            "status": "available", "stable_paths": ["semantic/model/layer_0/op"],
            "reconciliation": {"observed_e2e_ns": 1000.0, "accounted_e2e_ns": 1000.0, "residual_ns": 0.0},
        },
    }


def test_final_acceptance_publishes_numeric_metrics_only_for_complete_evidence() -> None:
    result = compose_final_acceptance(_document())

    assert result["status"] == "accepted"
    assert result["metrics"] == {
        "selected_complete_schedule_achievable_frontier_ns": 800.0,
        "qualified_e2e_observation_ns": 1000.0,
        "absolute_gap_ns": 200.0,
        "relative_gap": pytest.approx(0.2),
        "observation_to_frontier_ratio": pytest.approx(1.25),
        "combined_uncertainty_ns": pytest.approx((20.0**2 + (100.0 / 1.349)**2) ** 0.5),
        "frontier_efficiency": pytest.approx(0.8),
    }


def test_final_acceptance_fail_closes_at_exact_schedule_boundary() -> None:
    document = _document(schedule_status="unknown")
    result = compose_final_acceptance(document)

    assert result["status"] == "structured-unknown"
    assert set(result["metrics"].values()) == {None}
    assert result["evidence_boundary"] == {
        "schedule": ["operator.matmul.exact-domain"],
        "holdout": [],
        "decomposition": [],
    }


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda d: d["holdout"]["raw_samples_ns"].__setitem__(0, 1.0), "holdout-sample-summary-mismatch"),
        (lambda d: d["schedule"]["leaves"][0].__setitem__("duration_ns", 700.0), "schedule-duration-reconciliation-mismatch"),
        (lambda d: d["schedule"]["leaves"][0].__setitem__("selected_candidate_id", "candidate-b"), "selected-candidate-not-in-surface"),
        (lambda d: d["schedule"].__setitem__("edges", [["missing", "semantic/model/layer_0/op"]]), "schedule-edge-stable-path-mismatch"),
        (lambda d: d["decomposition"]["reconciliation"].__setitem__("residual_ns", 3.0), "decomposition-reconciliation-mismatch"),
    ],
)
def test_final_acceptance_rejects_semantic_tampering(mutation, reason: str) -> None:
    document = _document()
    mutation(document)
    with pytest.raises(FinalAcceptanceError, match=reason):
        compose_final_acceptance(document)


def test_final_acceptance_bundle_replays_without_npu_and_rejects_rehashed_tamper() -> None:
    repository = Path(__file__).parents[1]
    namespace = runpy.run_path(
        str(repository / "goal_process/issue-50-final-hardware-acceptance/build_final_acceptance.py")
    )
    document = namespace["build"]()
    with tempfile.TemporaryDirectory(dir=repository) as temporary:
        run = write_final_acceptance_bundle(
            temporary, run_id="issue50-final-acceptance-test-a", document=document
        )
        verification = verify_run_bundle(run)
        assert verification["passed"] is True, verification

        result_path = run / "acceptance/final-hardware-acceptance.json"
        result = json.loads(result_path.read_text())
        result["metrics"]["absolute_gap_ns"] = 1.0
        result_path.write_text(json.dumps(result, sort_keys=True) + "\n")
        manifest_path = run / "run.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        next(a for a in manifest["artifacts"] if a["path"] == "acceptance/final-hardware-acceptance.json")["sha256"] = sha256(result_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

        verification = verify_run_bundle(run)
        assert verification["passed"] is False
        assert "final hardware acceptance derivation mismatch" in verification["failures"]


def test_final_acceptance_rejects_coordinated_rehashed_stable_path_attack() -> None:
    repository = Path(__file__).parents[1]
    namespace = runpy.run_path(
        str(repository / "goal_process/issue-50-final-hardware-acceptance/build_final_acceptance.py")
    )
    document = namespace["build"]()
    document["schedule"]["stable_paths"][0] = "semantic/tampered"
    document["schedule"]["leaves"][0]["stable_path"] = "semantic/tampered"
    document["source_bundles"][0]["semantic_contract"]["stable_paths"][0] = "semantic/tampered"
    with tempfile.TemporaryDirectory(dir=repository) as temporary:
        run = write_final_acceptance_bundle(
            temporary, run_id="issue50-coordinated-stable-path-attack", document=document
        )
        verification = verify_run_bundle(run)
    assert verification["passed"] is False
    assert "final hardware acceptance source semantic contract mismatch" in verification["failures"]


@pytest.mark.parametrize("attack", ["schedule", "decomposition", "identity"])
def test_final_acceptance_rejects_coordinated_source_contract_attacks(
    attack: str,
) -> None:
    repository = Path(__file__).parents[1]
    namespace = runpy.run_path(
        str(repository / "goal_process/issue-50-final-hardware-acceptance/build_final_acceptance.py")
    )
    document = namespace["build"]()
    if attack == "schedule":
        paths = document["schedule"]["stable_paths"]
        forged = [[paths[0], paths[1]]]
        document["schedule"]["edges"] = forged
        document["source_bundles"][0]["semantic_contract"]["edges"] = forged
    elif attack == "decomposition":
        forged = {"observed_e2e_ns": 1.0, "accounted_e2e_ns": 1.0, "residual_ns": 0.0}
        document["decomposition"]["reconciliation"] = forged
        document["source_bundles"][1]["semantic_contract"]["final_decomposition"]["reconciliation"] = forged
    else:
        forged = {**document["identity"], "model_spec_sha256": "0" * 64}
        document["identity"] = forged
        document["holdout"]["identity"] = forged
        for source in document["source_bundles"]:
            source["identity"] = forged
        for source in document["source_identities"]:
            source["identity"] = forged
    with tempfile.TemporaryDirectory(dir=repository) as temporary:
        run = write_final_acceptance_bundle(
            temporary, run_id=f"issue50-coordinated-{attack}-attack", document=document
        )
        verification = verify_run_bundle(run)
    assert verification["passed"] is False
    assert "final hardware acceptance source semantic contract mismatch" in verification["failures"]


def test_final_acceptance_rejects_known_schedule_with_missing_evidence() -> None:
    document = _document()
    document["schedule"]["missing_evidence"] = ["operator.matmul.exact-domain"]
    with pytest.raises(FinalAcceptanceError, match="known-schedule-contains-missing-evidence"):
        compose_final_acceptance(document)


def test_final_acceptance_requires_locked_sources_and_same_identity() -> None:
    document = _document()
    document["source_bundles"] = []
    with pytest.raises(FinalAcceptanceError, match="final-acceptance-requires-locked-sources"):
        compose_final_acceptance(document)

    document = _document()
    document["source_identities"][0]["identity"] = {**IDENTITY, "dtype": "float16"}
    result = compose_final_acceptance(document)
    assert result["status"] == "structured-unknown"
    assert result["evidence_boundary"]["schedule"] == [
        "source-identity-mismatch:construction-a"
    ]


def test_final_acceptance_preserves_failed_holdout_gates_as_boundary() -> None:
    document = _document(schedule_status="unknown")
    document["holdout"]["gates"]["timing"] = "failed"
    result = compose_final_acceptance(document)
    assert "holdout-gate:timing" in result["evidence_boundary"]["holdout"]

    document = _document(schedule_status="unknown")
    document["holdout"]["gates"]["environment"] = "failed"
    result = compose_final_acceptance(document)
    assert "holdout-gate:environment" in result["evidence_boundary"]["holdout"]


def test_final_acceptance_requires_validated_schedule_execution_ir() -> None:
    document = _document()
    document["schedule"]["execution_ir"]["resource_claims"] = []
    with pytest.raises(FinalAcceptanceError, match="invalid-selected-schedule-execution-ir"):
        compose_final_acceptance(document)

    document = _document()
    document["schedule"]["execution_ir"]["dependency_edges"] = [["ghost-a", "ghost-b"]]
    with pytest.raises(FinalAcceptanceError, match="invalid-selected-schedule-execution-ir"):
        compose_final_acceptance(document)

    document = _document()
    document["schedule"]["execution_ir"]["resource_claims"] = [{"garbage": True}]
    with pytest.raises(FinalAcceptanceError, match="invalid-selected-schedule-execution-ir"):
        compose_final_acceptance(document)

    document = _document()
    document["schedule"]["execution_ir"]["transformations"] = ["arbitrary"]
    with pytest.raises(FinalAcceptanceError, match="invalid-selected-schedule-execution-ir"):
        compose_final_acceptance(document)

    document = _document()
    document["schedule"]["execution_ir"]["transformations"] = [
        {"event_id": "event-a", "kind": "arbitrary"}
    ]
    with pytest.raises(FinalAcceptanceError, match="invalid-selected-schedule-execution-ir"):
        compose_final_acceptance(document)
