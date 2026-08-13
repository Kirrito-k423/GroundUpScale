"""Compose issue #48 from immutable, already-collected evidence boundaries."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from groundupscale.benchmark.reference import (
    ReferenceConfig,
    SemanticLeaf,
    TwoLayerTransformer,
)
from groundupscale.run_bundle import verify_run_bundle


DEMO_RUN = Path(
    "goal_process/issue-30-ascend-transformer-demo/evidence/runs/"
    "ascend-910b2-transformer-demo-20260811-v1"
)
MATMUL_RUN = Path(
    "goal_process/issue-42-transformer-matmul-frontier/evidence/"
    "issue42-20260813-v1/artifact-store/runs/"
    "issue42-issue42-20260813-v1-transformer-matmul-frontier-v4"
)
RMSNORM_RUN = Path(
    "goal_process/issue-43-ascend-rmsnorm-frontier/evidence/runs/"
    "issue43-20260813Tissue43npu01-rmsnorm-frontier-unknown-v3"
)
SOFTMAX_RUN = Path(
    "goal_process/issue-44-ascend-softmax-phase-frontier/evidence/runs/"
    "issue44-20260813T1945Z-softmax-frontier-unknown-v2"
)
TARGET_COHORT = "ascend-npu-23b93a89d5fecc79"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _manifest_sha(root: Path) -> str:
    return sha256((root / "run.manifest.json").read_bytes()).hexdigest()


def _source(
    repository: Path,
    issue: int,
    relative: Path,
    *,
    require_verified: bool = True,
) -> dict[str, Any]:
    root = repository / relative
    verification = verify_run_bundle(root)
    if require_verified and verification.get("passed") is not True:
        raise ValueError(f"upstream Run Bundle failed verification: {relative}")
    manifest = _load(root / "run.manifest.json")
    return {
        "issue": issue,
        "run_id": manifest["run_id"],
        "bundle_kind": manifest["bundle_kind"],
        "status": manifest["status"],
        "hardware_cohort": manifest.get("hardware_cohort"),
        "path": relative.as_posix(),
        "manifest_sha256": _manifest_sha(root),
        "verification_passed": verification.get("passed") is True,
        "verification_failures": list(verification.get("failures", [])),
    }


def _required_class(operation: str) -> str:
    return {
        "MatMul": "operator.matmul.exact-domain",
        "RMSNorm": "operator.rmsnorm.phase-graph",
        "Softmax": "operator.softmax.phase-graph",
        "Add": "operator.add.exact-domain",
        "Mul": "operator.mul.exact-domain",
        "SiLU": "operator.silu.exact-domain",
        "View": "layout.alias-or-materialization-audit",
        "Transpose": "layout.alias-or-materialization-audit",
    }[operation]


def _boundary(operation: str) -> str:
    return {
        "MatMul": (
            "issue #42 v4 exact domain is structured unknown: one additional "
            "eligible candidate needs three independent search sessions and, "
            "if best-of-correct, three independent holdout sessions"
        ),
        "RMSNorm": (
            "issue #43 v3 phase graph is structured unknown; collect the named "
            "missing phase evidence in the exact Hardware Cohort"
        ),
        "Softmax": (
            "issue #44 qualified bundle belongs to a different Hardware Cohort "
            "and cannot be consumed by this schedule"
        ),
        "Add": (
            "issue #45 does not publish a repository-contained immutable source "
            "Run Bundle covering every Add execution domain"
        ),
        "Mul": (
            "issue #45 does not publish a repository-contained immutable source "
            "Run Bundle covering every Mul execution domain"
        ),
        "SiLU": (
            "issue #45 does not publish a repository-contained immutable source "
            "Run Bundle covering the SiLU execution domain"
        ),
        "View": (
            "issue #46 defines the public audit contract but publishes no "
            "immutable audit Run Bundle for this indexed Stable Path"
        ),
        "Transpose": (
            "issue #46 defines the public audit contract but publishes no "
            "immutable audit Run Bundle for this indexed Stable Path"
        ),
    }[operation]


def _semantic_leaves() -> list[dict[str, object]]:
    config = ReferenceConfig(
        batch_size=1,
        sequence_length=512,
        hidden_size=512,
        heads=8,
        head_dim=64,
        intermediate_size=2048,
        layers=2,
        model_root=(
            "semantic/workload/transformer-prefill/request/model-prefill/"
            "model/transformer"
        ),
    )
    leaves: list[dict[str, object]] = []
    for module in TwoLayerTransformer(config, seed=20260811).modules():
        if not isinstance(module, SemanticLeaf):
            continue
        operation_class = _required_class(module.operation)
        leaves.append(
            {
                "stable_path": module.stable_path,
                "operation_class": module.operation,
                "mandatory_operation_classes": [operation_class],
                "requirements": [
                    {
                        "operation_class": operation_class,
                        "required_evidence": _boundary(module.operation),
                    }
                ],
            }
        )
    if len(leaves) != 52:
        raise ValueError("two-layer Transformer must contain exactly 52 leaves")
    return leaves


def compose_issue48_input(repository: str | Path) -> dict[str, Any]:
    """Build the real #48 model input without promoting incomplete evidence."""

    root = Path(repository).resolve()
    sources = [
        _source(root, 30, DEMO_RUN),
        _source(root, 42, MATMUL_RUN),
        _source(root, 43, RMSNORM_RUN),
        _source(root, 44, SOFTMAX_RUN),
    ]
    demo = root / DEMO_RUN
    comparison = _load(demo / "comparison/predicted-vs-observed.json")
    e2e = next(
        item
        for item in comparison["latency_cases"]
        if item.get("case_id") == "two-layer-prefill"
    )
    observation_ns = e2e["observed"]["median_ns"]
    effects = (
        "device-dispatch",
        "device-queueing",
        "device-transformations",
        "device-copies",
        "device-idle",
        "device-synchronization",
    )
    return {
        "schema": "groundupscale.dev/model-e2e-frontier-input/v1alpha1",
        "evidence": {
            "classification": "evidence-qualified-composition",
            "source_issue": "#48",
            "promotion_eligible": False,
            "hardware_cohort": TARGET_COHORT,
            "evidence_refs": [
                f"run://{source['run_id']}@sha256:{source['manifest_sha256']}"
                for source in sources
            ],
            "source_bundles": sources,
            "unpublished_source_boundaries": [
                {
                    "issue": 45,
                    "reason_code": "no-repository-contained-immutable-run-bundle",
                },
                {
                    "issue": 46,
                    "reason_code": "no-indexed-layout-audit-run-bundle",
                },
            ],
        },
        "model": {
            "model_id": "two-layer-transformer-prefill",
            "expected_semantic_leaf_count": 52,
            "repeated_layer_indices": [0, 1],
            "semantic_leaves": _semantic_leaves(),
        },
        "schedule": {
            "policy_id": "issue48-explicit-single-stream-v1",
            "version": "1",
            "kind": "serialized-unfused",
            "mandatory_effect_ids": list(effects),
            "mandatory_effects": [
                {
                    "effect_id": effect,
                    "operation_class": f"schedule.{effect}",
                    "required_evidence": (
                        "same Completion Boundary direct device evidence; "
                        "host-only diagnostic spans are not device duration"
                    ),
                }
                for effect in effects
            ],
            "dependencies": [],
            "rejected_optimizations": [
                {
                    "kind": kind,
                    "status": "rejected",
                    "reason_code": (
                        "missing-explicit-candidate-or-contract-and-direct-evidence"
                    ),
                }
                for kind in (
                    "fusion",
                    "overlap",
                    "chunk-pipeline",
                    "dispatch-hiding",
                    "queue-hiding",
                    "synchronization-hiding",
                )
            ],
            "evidence_refs": [
                "issue://github/Kirrito-k423/GroundUpScale/48",
                f"run://{sources[0]['run_id']}@sha256:{sources[0]['manifest_sha256']}",
            ],
        },
        "axes": {
            "resource_physical_floor": {
                "status": "unknown",
                "reason_code": "same-boundary-complete-resource-floor-missing",
            },
            "observation": {
                "status": "known",
                "value_ns": observation_ns,
                "evidence_refs": [
                    f"run://{sources[0]['run_id']}@sha256:{sources[0]['manifest_sha256']}"
                    "#comparison/predicted-vs-observed.json/two-layer-prefill"
                ],
            },
        },
        "uncertainty": {
            "policy_id": "issue48-root-sum-square-v1",
            "version": "1",
            "combination": "root-sum-square",
            "schedule_component_ns": 0,
            "observation_component_ns": 0,
            "evidence_refs": [
                "issue://github/Kirrito-k423/GroundUpScale/48#unknown-no-numeric-combination"
            ],
        },
    }
