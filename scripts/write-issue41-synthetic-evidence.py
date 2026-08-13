#!/usr/bin/env python3
"""Write the deterministic public-seam acceptance bundles for issue 41."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from groundupscale.benchmark.reference import (
    ReferenceConfig,
    SemanticLeaf,
    TwoLayerTransformer,
)
from groundupscale.model_e2e_frontier import write_model_e2e_frontier_bundle


ROOT = Path(__file__).parents[1]
ARTIFACT_STORE = ROOT / "goal_process/issue-41-model-e2e-frontier/evidence"


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
    classes = phase_classes.get(
        operation, (f"operator.{operation.lower()}.fp32",)
    )
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


def document() -> dict[str, object]:
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


def main() -> None:
    complete = document()
    write_model_e2e_frontier_bundle(
        complete,
        ARTIFACT_STORE,
        run_id="issue-41-full-demo-contract-v1",
    )
    incomplete = deepcopy(complete)
    rmsnorm = next(
        leaf
        for leaf in incomplete["model"]["semantic_leaves"]
        if leaf["operation_class"] == "RMSNorm"
    )
    rsqrt = next(
        requirement
        for requirement in rmsnorm["requirements"]
        if requirement["operation_class"] == "transcendental.rsqrt.fp32"
    )
    del rsqrt["candidate"]
    write_model_e2e_frontier_bundle(
        incomplete,
        ARTIFACT_STORE,
        run_id="issue-41-missing-rmsnorm-phase-v1",
    )


if __name__ == "__main__":
    main()
