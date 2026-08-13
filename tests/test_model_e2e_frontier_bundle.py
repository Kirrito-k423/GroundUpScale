from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from math import inf, nan
from pathlib import Path
import subprocess

import pytest

from groundupscale.benchmark.reference import (
    ReferenceConfig,
    SemanticLeaf,
    TwoLayerTransformer,
)
from groundupscale.model_e2e_frontier import (
    load_model_e2e_frontier_report,
    write_model_e2e_frontier_bundle,
)
from groundupscale.issue48_composition import compose_issue48_input
from groundupscale.run_bundle import verify_run_bundle


def _candidate(
    candidate_id: str,
    operation_class: str,
    duration_ns: int,
    stable_path: str,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "stable_path": stable_path,
        "operation_class": operation_class,
        "duration_ns": duration_ns,
        "standard_uncertainty_ns": 1,
        "resource_claims": [
            {
                "resource_id": "synthetic.device",
                "duration_ns": duration_ns,
                "evidence_refs": [f"fixture://issue-41/{candidate_id}/resource"],
            }
        ],
        "evidence_refs": [f"fixture://issue-41/{candidate_id}"],
    }


def _requirements(path: str, operation: str) -> list[dict[str, object]]:
    slug = path.replace("/", "-")
    phase_classes = {
        "RMSNorm": (
            "elementwise.square.fp32",
            "reduction.sum.fp32",
            "elementwise.mean-scale.fp32",
            "elementwise.epsilon-add.fp32",
            "transcendental.rsqrt.fp32",
            "elementwise.input-scale.fp32",
            "elementwise.weight-scale.fp32",
        ),
        "Softmax": (
            "reduction.max.fp32",
            "elementwise.subtract.fp32",
            "transcendental.exp.fp32",
            "reduction.sum.fp32",
            "elementwise.normalize.fp32",
        ),
    }
    classes = phase_classes.get(operation, (f"operator.{operation.lower()}.fp32",))
    if operation in {"View", "Transpose"}:
        classes = ("alias-preserving.view",)
    return [
        {
            "operation_class": operation_class,
            "required_evidence": f"exact candidate for {operation_class}",
            "candidate": _candidate(
                f"candidate-{slug}-phase-{index}",
                operation_class,
                0 if operation in {"View", "Transpose"} else 100 + index,
                path,
            ),
        }
        for index, operation_class in enumerate(classes)
    ]


def _document() -> dict[str, object]:
    config = ReferenceConfig(
        batch_size=1,
        sequence_length=512,
        hidden_size=512,
        heads=8,
        head_dim=64,
        intermediate_size=2048,
        layers=2,
        model_root="semantic/workload/transformer/prefill/model/transformer",
    )
    model = TwoLayerTransformer(config, seed=20260813)
    leaves = [
        {
            "stable_path": module.stable_path,
            "operation_class": module.operation,
            "requirements": _requirements(module.stable_path, module.operation),
        }
        for module in model.modules()
        if isinstance(module, SemanticLeaf)
    ]
    for leaf in leaves:
        leaf["mandatory_operation_classes"] = [
            requirement["operation_class"]
            for requirement in leaf["requirements"]
        ]
    schedule_effects = [
        {
            "effect_id": "device-dispatch",
            "operation_class": "schedule.device-dispatch",
            "required_evidence": "same-boundary device dispatch candidate",
            "candidate": _candidate(
                "schedule-device-dispatch",
                "schedule.device-dispatch",
                500,
                "schedule/device-dispatch",
            ),
        },
        {
            "effect_id": "device-synchronization",
            "operation_class": "schedule.device-synchronization",
            "required_evidence": "same-boundary synchronization candidate",
            "candidate": _candidate(
                "schedule-device-synchronization",
                "schedule.device-synchronization",
                700,
                "schedule/device-synchronization",
            ),
        },
    ]
    ordered_candidate_ids = [
        requirement["candidate"]["candidate_id"]
        for leaf in leaves
        for requirement in leaf["requirements"]
    ] + [effect["candidate"]["candidate_id"] for effect in schedule_effects]
    return {
        "schema": "groundupscale.dev/model-e2e-frontier-input/v1alpha1",
        "evidence": {
            "classification": "deterministic-synthetic",
            "source_issue": "#41",
            "promotion_eligible": False,
            "hardware_cohort": "synthetic-ascend-910b2-contract",
            "evidence_refs": ["fixture://issue-41/full-demo"],
        },
        "model": {
            "model_id": "two-layer-transformer-prefill",
            "expected_semantic_leaf_count": 52,
            "repeated_layer_indices": [0, 1],
            "semantic_leaves": leaves,
        },
        "schedule": {
            "policy_id": "fixture://issue-41/serialized-unfused",
            "version": "1",
            "kind": "serialized-unfused",
            "mandatory_effect_ids": [
                "device-dispatch",
                "device-synchronization",
            ],
            "mandatory_effects": schedule_effects,
            "dependencies": [
                {
                    "source": source,
                    "target": target,
                    "evidence_refs": ["fixture://issue-41/serialized-order"],
                }
                for source, target in zip(
                    ordered_candidate_ids, ordered_candidate_ids[1:]
                )
            ],
            "evidence_refs": ["fixture://issue-41/schedule-policy"],
        },
        "axes": {
            "resource_physical_floor": {
                "status": "known",
                "value_ns": 4_000,
                "evidence_refs": ["fixture://issue-41/resource-floor"],
            },
            "observation": {
                "status": "known",
                "value_ns": 25_000,
                "evidence_refs": ["fixture://issue-41/observation"],
            },
        },
        "uncertainty": {
            "policy_id": "fixture://issue-41/root-sum-square",
            "version": "1",
            "combination": "root-sum-square",
            "schedule_component_ns": 10,
            "observation_component_ns": 20,
            "evidence_refs": ["fixture://issue-41/uncertainty"],
        },
    }


def test_complete_two_layer_bundle_publishes_one_numeric_model_result(
    tmp_path: Path,
) -> None:
    run = write_model_e2e_frontier_bundle(
        _document(), tmp_path, run_id="issue-41-full-demo-contract"
    )

    verification = verify_run_bundle(run)
    published = load_model_e2e_frontier_report(run)
    result = published["machine_result"]

    assert verification["passed"] is True
    assert result["status"] == "complete"
    assert result["coverage"]["semantic_leaf_count"] == 52
    paths = [leaf["stable_path"] for leaf in result["coverage"]["predicted_leaves"]]
    assert len(paths) == len(set(paths)) == 52
    assert sum("/layer_0/" in path for path in paths) == 26
    assert sum("/layer_1/" in path for path in paths) == 26
    assert result["axes"]["schedule_achievable_frontier"] == {
        "status": "known",
        "value_ns": result["schedule"]["selected_feasible_duration_ns"],
        "evidence_refs": ["fixture://issue-41/schedule-policy"],
    }
    assert result["comparison"]["relative_prediction_error"] is not None
    assert result["schedule"]["serialized_unfused_duration_ns"] == result[
        "schedule"
    ]["selected_feasible_duration_ns"]
    assert result["schedule"]["critical_path_duration_ns"] == result["schedule"][
        "selected_feasible_duration_ns"
    ]
    assert result["schedule"]["shared_resource_duration_ns"] == result["schedule"][
        "selected_feasible_duration_ns"
    ]
    assert result["schedule"]["ideal_dag_duration_ns"] == result["schedule"][
        "selected_feasible_duration_ns"
    ]
    assert len(result["schedule"]["explicit_dependencies"]) == (
        len(result["schedule"]["physical_events"]) - 1
    )
    assert all(event["resource_claims"] for event in result["schedule"]["physical_events"])
    assert "Resource Physical Floor" in published["human_report"]
    assert "Operator Achievable Frontier" in published["human_report"]
    assert "Schedule Achievable Frontier" in published["human_report"]
    assert "E2E Observation" in published["human_report"]

    explained = subprocess.run(
        ["uv", "run", "groundupscale", "explain", str(run), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert explained.returncode == 0, explained.stderr
    summary = json.loads(explained.stdout)
    assert summary["bundle_kind"] == "model-e2e-frontier"
    assert summary["semantic_leaf_count"] == 52
    assert summary["axes"] == result["axes"]
    assert summary["comparison"] == result["comparison"]


def test_missing_rmsnorm_phase_publishes_replayable_structured_unknown(
    tmp_path: Path,
) -> None:
    document = _document()
    rmsnorm = next(
        leaf
        for leaf in document["model"]["semantic_leaves"]
        if leaf["operation_class"] == "RMSNorm"
    )
    rsqrt = next(
        requirement
        for requirement in rmsnorm["requirements"]
        if requirement["operation_class"] == "transcendental.rsqrt.fp32"
    )
    del rsqrt["candidate"]

    run = write_model_e2e_frontier_bundle(
        document, tmp_path, run_id="issue-41-missing-rmsnorm-phase"
    )
    published = load_model_e2e_frontier_report(run)
    result = published["machine_result"]

    assert verify_run_bundle(run)["passed"] is True
    assert result["status"] == "unknown"
    assert result["axes"]["resource_physical_floor"]["status"] == "known"
    assert result["axes"]["operator_achievable_frontier"]["status"] == "unknown"
    assert result["axes"]["schedule_achievable_frontier"] == {
        "status": "unknown",
        "value_ns": None,
        "reason_code": "mandatory-model-evidence-missing",
        "missing_operation_classes": ["transcendental.rsqrt.fp32"],
    }
    assert result["axes"]["observation"]["status"] == "known"
    assert result["comparison"]["relative_prediction_error"] is None
    assert result["comparison"]["error_status"] == "unknown-incomplete-schedule-frontier"
    assert result["missing_evidence"] == [
        {
            "stable_path": rmsnorm["stable_path"],
            "operation_class": "transcendental.rsqrt.fp32",
            "required_evidence": "exact candidate for transcendental.rsqrt.fp32",
        }
    ]
    assert "transcendental.rsqrt.fp32" in published["human_report"]


def test_missing_schedule_effect_does_not_overwrite_operator_frontier(
    tmp_path: Path,
) -> None:
    document = _document()
    synchronization = next(
        effect
        for effect in document["schedule"]["mandatory_effects"]
        if effect["effect_id"] == "device-synchronization"
    )
    del synchronization["candidate"]

    run = write_model_e2e_frontier_bundle(
        document, tmp_path, run_id="issue-41-missing-schedule-effect"
    )
    result = load_model_e2e_frontier_report(run)["machine_result"]

    assert verify_run_bundle(run)["passed"] is True
    assert result["axes"]["resource_physical_floor"]["status"] == "known"
    assert result["axes"]["operator_achievable_frontier"]["status"] == "known"
    assert result["axes"]["schedule_achievable_frontier"] == {
        "status": "unknown",
        "value_ns": None,
        "reason_code": "mandatory-model-evidence-missing",
        "missing_operation_classes": ["schedule.device-synchronization"],
    }
    assert result["axes"]["observation"]["status"] == "known"
    assert result["comparison"]["relative_prediction_error"] is None


def test_missing_elementwise_candidate_publishes_structured_unknown(
    tmp_path: Path,
) -> None:
    document = _document()
    elementwise = next(
        leaf
        for leaf in document["model"]["semantic_leaves"]
        if leaf["operation_class"] == "Add"
    )
    del elementwise["requirements"][0]["candidate"]

    run = write_model_e2e_frontier_bundle(
        document, tmp_path, run_id="issue-41-missing-elementwise"
    )
    result = load_model_e2e_frontier_report(run)["machine_result"]

    assert result["status"] == "unknown"
    assert result["axes"]["schedule_achievable_frontier"]["status"] == "unknown"
    assert result["comparison"]["relative_prediction_error"] is None
    assert result["missing_evidence"] == [
        {
            "stable_path": elementwise["stable_path"],
            "operation_class": "operator.add.fp32",
            "required_evidence": "exact candidate for operator.add.fp32",
        }
    ]


def test_removing_mandatory_requirement_or_effect_fails_closed(tmp_path: Path) -> None:
    missing_requirement = _document()
    missing_requirement["model"]["semantic_leaves"][0]["requirements"].pop()
    with pytest.raises(ValueError, match="mandatory-operation-class-mismatch"):
        write_model_e2e_frontier_bundle(
            missing_requirement, tmp_path, run_id="issue-41-missing-requirement"
        )

    missing_effect = _document()
    missing_effect["schedule"]["mandatory_effects"].pop()
    with pytest.raises(ValueError, match="mandatory-schedule-effect-mismatch"):
        write_model_e2e_frontier_bundle(
            missing_effect, tmp_path, run_id="issue-41-missing-effect-section"
        )


def test_unknown_observation_prevents_complete_result_without_overwriting_schedule(
    tmp_path: Path,
) -> None:
    document = _document()
    document["axes"]["observation"] = {
        "status": "unknown",
        "reason_code": "baseline-observation-missing",
    }
    run = write_model_e2e_frontier_bundle(
        document, tmp_path, run_id="issue-41-observation-unknown"
    )
    result = load_model_e2e_frontier_report(run)["machine_result"]

    assert result["status"] == "unknown"
    assert result["axes"]["schedule_achievable_frontier"]["status"] == "known"
    assert result["axes"]["observation"]["status"] == "unknown"
    assert result["comparison"]["relative_prediction_error"] is None


@pytest.mark.parametrize("value", [0, nan, inf])
def test_invalid_known_observation_fails_closed(
    tmp_path: Path, value: float
) -> None:
    document = _document()
    document["axes"]["observation"]["value_ns"] = value

    with pytest.raises(ValueError, match="invalid-observation-axis"):
        write_model_e2e_frontier_bundle(
            document, tmp_path, run_id="issue-41-invalid-observation"
        )


def test_candidate_cannot_cross_a_stable_path_boundary(tmp_path: Path) -> None:
    document = _document()
    source = document["model"]["semantic_leaves"][0]
    target = document["model"]["semantic_leaves"][1]
    target["requirements"][0]["candidate"]["stable_path"] = source[
        "stable_path"
    ]

    with pytest.raises(
        ValueError, match="candidate-stable-path-mismatch"
    ):
        write_model_e2e_frontier_bundle(
            document, tmp_path, run_id="issue-41-stable-path-mismatch"
        )


def test_synthetic_contract_cannot_be_promotion_eligible(tmp_path: Path) -> None:
    document = _document()
    document["evidence"]["promotion_eligible"] = True

    with pytest.raises(
        ValueError, match="synthetic-evidence-cannot-be-promotion-eligible"
    ):
        write_model_e2e_frontier_bundle(
            document, tmp_path, run_id="issue-41-invalid-promotion"
        )


def test_verifier_rejects_missing_mandatory_section_after_local_rehash(
    tmp_path: Path,
) -> None:
    run = write_model_e2e_frontier_bundle(
        _document(), tmp_path, run_id="issue-41-tampered-complete"
    )
    result_path = run / "comparison" / "model-e2e-frontier.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    del result["schedule"]
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(
        item
        for item in manifest["artifacts"]
        if item["role"] == "prediction-observation-comparison"
    )
    artifact["sha256"] = sha256(result_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert "model E2E comparison derivation mismatch" in verification["failures"]


def test_incomplete_real_composition_retains_replayable_schedule_references(
    tmp_path: Path,
) -> None:
    document = _document()
    document["evidence"] = {
        "classification": "evidence-qualified-composition",
        "source_issue": "#48",
        "promotion_eligible": False,
        "hardware_cohort": "ascend-npu-23b93a89d5fecc79",
        "evidence_refs": ["run://issue42/frontier-v4"],
        "source_bundles": [
            {
                "issue": 42,
                "run_id": "issue42-frontier-v4",
                "bundle_kind": "transformer-matmul-frontier",
                "status": "unknown",
                "hardware_cohort": "ascend-npu-23b93a89d5fecc79",
                "path": "fixture/issue42-frontier-v4",
                "manifest_sha256": "a" * 64,
                "verification_passed": True,
                "verification_failures": [],
            }
        ],
    }
    document["schedule"]["policy_id"] = "issue48-explicit-single-stream-v1"
    document["schedule"]["rejected_optimizations"] = [
        {
            "kind": kind,
            "status": "rejected",
            "reason_code": "missing-explicit-candidate-or-contract-and-direct-evidence",
        }
        for kind in (
            "fusion",
            "overlap",
            "chunk-pipeline",
            "dispatch-hiding",
            "queue-hiding",
            "synchronization-hiding",
        )
    ]
    document["schedule"]["mandatory_effect_ids"] = [
        "device-dispatch",
        "device-queueing",
        "device-transformations",
        "device-copies",
        "device-idle",
        "device-synchronization",
    ]
    document["schedule"]["mandatory_effects"] = [
        {
            "effect_id": effect_id,
            "operation_class": f"schedule.{effect_id}",
            "required_evidence": f"same-boundary {effect_id} evidence",
        }
        for effect_id in document["schedule"]["mandatory_effect_ids"]
    ]
    first = document["model"]["semantic_leaves"][0]
    del first["requirements"][0]["candidate"]

    run = write_model_e2e_frontier_bundle(
        document, tmp_path, run_id="issue48-real-composition-unknown-test"
    )
    result = load_model_e2e_frontier_report(run)["machine_result"]

    assert verify_run_bundle(run)["passed"] is True
    assert result["status"] == "unknown"
    assert result["evidence"]["authority"] == "evidence-qualified-composition"
    assert result["coverage"]["semantic_leaf_count"] == 52
    assert len({leaf["stable_path"] for leaf in result["coverage"]["predicted_leaves"]}) == 52
    assert result["schedule"]["references"] == {
        "serialized_unfused": {"status": "unknown", "duration_ns": None},
        "ideal_dag": {"status": "unknown", "duration_ns": None},
        "selected_feasible": {"status": "unknown", "duration_ns": None},
    }
    assert {item["kind"] for item in result["schedule"]["rejected_optimizations"]} == {
        "fusion",
        "overlap",
        "chunk-pipeline",
        "dispatch-hiding",
        "queue-hiding",
        "synchronization-hiding",
    }
    assert all(item["status"] == "unknown" for item in result["schedule"]["mandatory_effects"])
    assert result["comparison"]["relative_prediction_error"] is None


def test_partial_composition_keeps_resolved_physical_event_provenance() -> None:
    document = _document()
    missing_leaf = document["model"]["semantic_leaves"][1]
    del missing_leaf["requirements"][0]["candidate"]

    result = __import__(
        "groundupscale.model_e2e_frontier", fromlist=["compose_model_e2e_frontier"]
    ).compose_model_e2e_frontier(document)

    assert result["status"] == "unknown"
    assert result["schedule"]["physical_events"]
    for event in result["schedule"]["physical_events"]:
        assert event["candidate_id"]
        assert event["duration_ns"] >= 0
        assert event["standard_uncertainty_ns"] >= 0
        assert event["resource_claims"]
        assert event["evidence_refs"]
        assert "dependency_ids" in event


def test_issue48_composes_real_upstream_boundaries_without_inventing_numbers(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]

    document = compose_issue48_input(repository)
    run = write_model_e2e_frontier_bundle(
        document, tmp_path, run_id="issue48-20260814T0001Z-schedule-frontier"
    )
    result = load_model_e2e_frontier_report(run)["machine_result"]

    assert verify_run_bundle(run)["passed"] is True
    assert result["status"] == "unknown"
    assert result["hardware_cohort"] == "ascend-npu-23b93a89d5fecc79"
    assert result["axes"]["observation"]["value_ns"] == 1_921_530.0
    assert result["comparison"]["relative_prediction_error"] is None
    leaves = result["coverage"]["predicted_leaves"]
    assert len(leaves) == 52
    assert len({leaf["stable_path"] for leaf in leaves}) == 52
    assert sum("/layer_0/" in leaf["stable_path"] for leaf in leaves) == 26
    assert sum("/layer_1/" in leaf["stable_path"] for leaf in leaves) == 26
    assert all(leaf["status"] == "unknown" for leaf in leaves)
    missing_classes = {
        item["operation_class"] for item in result["missing_evidence"]
    }
    assert {
        "operator.matmul.exact-domain",
        "operator.rmsnorm.phase-graph",
        "operator.softmax.phase-graph",
        "operator.add.exact-domain",
        "operator.mul.exact-domain",
        "operator.silu.exact-domain",
        "layout.alias-or-materialization-audit",
        "schedule.device-dispatch",
        "schedule.device-queueing",
        "schedule.device-transformations",
        "schedule.device-copies",
        "schedule.device-idle",
        "schedule.device-synchronization",
    } <= missing_classes
    source_bundles = document["evidence"]["source_bundles"]
    assert {source["issue"] for source in source_bundles} == {30, 42, 43, 44}
    assert all(source["verification_passed"] is True for source in source_bundles)
    assert all(len(source["manifest_sha256"]) == 64 for source in source_bundles)


def test_real_composition_fails_closed_on_unverified_source_metadata() -> None:
    document = compose_issue48_input(Path(__file__).resolve().parents[1])
    document["evidence"]["source_bundles"][0]["verification_passed"] = False

    with pytest.raises(ValueError, match="invalid-model-source-bundles"):
        __import__(
            "groundupscale.model_e2e_frontier",
            fromlist=["compose_model_e2e_frontier"],
        ).compose_model_e2e_frontier(document)
