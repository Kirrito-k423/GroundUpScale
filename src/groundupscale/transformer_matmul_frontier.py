"""Qualify MatMul execution domains used by a Transformer Run Bundle."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from groundupscale.ir import content_fingerprint
from groundupscale.run_bundle import (
    RUN_ID_PATTERN,
    RunBundleExistsError,
    verify_run_bundle,
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _artifact_document(
    root: Path, manifest: dict[str, Any], role: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches = [
        item
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict) and item.get("role") == role
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {role} artifact in {root}")
    entry = matches[0]
    path = (root / str(entry.get("path", ""))).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"invalid {role} artifact path in {root}")
    return _load_json(path), entry


def _walk_items(item: object) -> list[dict[str, Any]]:
    if not isinstance(item, dict):
        return []
    found = [item]
    children = item.get("items")
    if isinstance(children, list):
        for child in children:
            found.extend(_walk_items(child))
    return found


def _contract(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("MatMul tensor contract is missing")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or not all(isinstance(size, int) and size > 0 for size in shape)
    ):
        raise ValueError("MatMul tensor Shape is incomplete")
    return {
        "shape": list(shape),
        "dtype": value.get("dtype"),
        "layout": value.get("layout"),
    }


def _batch_transpose_contract(
    operands: list[dict[str, object]], result: dict[str, object]
) -> dict[str, object]:
    left_shape = list(operands[0]["shape"])
    right_shape = list(operands[1]["shape"])
    return {
        "left_batch_shape": left_shape[:-2],
        "right_batch_shape": right_shape[:-2],
        "result_batch_shape": list(result["shape"])[:-2],
        "left_layout": operands[0]["layout"],
        "right_layout": operands[1]["layout"],
        "result_layout": result["layout"],
        "left_transposed": operands[0]["layout"] == "transposed",
        "right_transposed": operands[1]["layout"] == "transposed",
    }


def _domain_class(local_id: object) -> str:
    if local_id == "qk_matmul":
        return "attention-qk"
    if local_id == "context_matmul":
        return "attention-context"
    if local_id in {"gate_proj", "up_proj"}:
        return "mlp-expand"
    if local_id == "down_proj":
        return "mlp-contract"
    if local_id in {"q_proj", "k_proj", "v_proj", "out_proj"}:
        return "projection"
    raise ValueError(f"unrecognized demo MatMul leaf: {local_id!r}")


def _candidate_family(domain_class: str) -> str:
    if domain_class == "attention-context":
        return "pytorch-ascend-matmul-transpose-contiguous"
    return "pytorch-ascend-matmul"


def _inventory(
    root: Path,
    manifest: dict[str, Any],
    cost_ir: dict[str, Any],
) -> dict[str, object]:
    operations = [
        item
        for item in _walk_items(cost_ir.get("root"))
        if item.get("operation") == "MatMul"
    ]
    domains: dict[str, dict[str, object]] = {}
    for operation in operations:
        operand_types = operation.get("operand_types")
        result_types = operation.get("result_types")
        if (
            not isinstance(operand_types, list)
            or len(operand_types) != 2
            or not isinstance(result_types, list)
            or len(result_types) != 1
        ):
            raise ValueError("demo MatMul has an incomplete tensor contract")
        operands = [_contract(value) for value in operand_types]
        result = _contract(result_types[0])
        domain_class = _domain_class(operation.get("local_id"))
        identity: dict[str, object] = {
            "semantic_operation": "MatMul",
            "operand_contracts": operands,
            "result_contract": result,
            "batch_transpose_contract": _batch_transpose_contract(
                operands, result
            ),
            "dtype": result["dtype"],
            "layout": {
                "operands": [value["layout"] for value in operands],
                "result": result["layout"],
            },
            "candidate_family": _candidate_family(domain_class),
            "execution_mode": "pytorch-eager",
            "runtime_dispatch_regime": "torch-npu-pytorch-eager",
            "hardware_cohort": manifest.get("hardware_cohort"),
        }
        identity_digest = content_fingerprint(identity)
        stable_path = operation.get("stable_path")
        if not isinstance(stable_path, str):
            raise ValueError("demo MatMul Stable Path is missing")
        record = domains.setdefault(
            identity_digest,
            {
                "domain_id": f"matmul-domain:{identity_digest}",
                "domain_class": domain_class,
                "identity": identity,
                "stable_paths": [],
                "source_cost_node_ids": [],
            },
        )
        if record["domain_class"] != domain_class:
            raise ValueError("distinct semantic MatMul domains collapsed")
        record["stable_paths"].append(stable_path)
        record["source_cost_node_ids"].append(operation.get("node_id"))
    ordered_domains = sorted(
        domains.values(), key=lambda item: str(item["domain_class"])
    )
    source_manifest = root / "run.manifest.json"
    return {
        "schema": "groundupscale.dev/matmul-domain-inventory/v1alpha1",
        "source_run_id": manifest.get("run_id"),
        "source_run_manifest_sha256": _sha256(source_manifest),
        "source_hardware_cohort": manifest.get("hardware_cohort"),
        "matmul_leaf_count": len(operations),
        "distinct_domain_count": len(ordered_domains),
        "domains": ordered_domains,
    }


def _frontier_evidence(path: str | Path) -> dict[str, object]:
    root = Path(path).resolve()
    verification = verify_run_bundle(root)
    if verification.get("passed") is not True:
        raise ValueError(f"Frontier Run Bundle failed verification: {root}")
    manifest = _load_json(root / "run.manifest.json")
    if manifest.get("bundle_kind") != "operator-frontier":
        raise ValueError(f"not an Operator Frontier Run Bundle: {root}")
    qualification, _ = _artifact_document(
        root, manifest, "operator-frontier-qualification"
    )
    surface = qualification.get("surface")
    if not isinstance(surface, dict):
        raise ValueError(f"Operator Frontier Surface is missing: {root}")
    return {
        "root": str(root),
        "manifest_sha256": _sha256(root / "run.manifest.json"),
        "run_id": manifest.get("run_id"),
        "hardware_cohort": manifest.get("hardware_cohort"),
        "qualification_status": qualification.get("status"),
        "surface": surface,
    }


def _mismatch_reasons(
    identity: dict[str, object], evidence: dict[str, object]
) -> list[str]:
    surface = evidence["surface"]
    assert isinstance(surface, dict)
    domain = surface.get("domain")
    work_formula = surface.get("work_formula")
    reasons: list[str] = []
    if evidence["qualification_status"] != "qualified":
        reasons.append("surface-not-qualified")
    if evidence["hardware_cohort"] != identity["hardware_cohort"]:
        reasons.append("hardware-cohort-mismatch")
    if not isinstance(domain, dict):
        reasons.append("incomplete-surface-domain")
        return reasons
    if domain.get("semantic_operation") != identity["semantic_operation"]:
        reasons.append("semantic-operation-mismatch")
    if domain.get("dtype") != identity["dtype"]:
        reasons.append("dtype-mismatch")
    if domain.get("execution_mode") != identity["execution_mode"]:
        reasons.append("execution-mode-mismatch")
    if domain.get("kernel_dispatch_regime") != identity["candidate_family"]:
        reasons.append("candidate-family-or-dispatch-mismatch")
    if not isinstance(work_formula, dict) or work_formula.get("kind") not in {
        "matmul-2mnk-fixed-nk"
    }:
        reasons.append("batch-transpose-contract-mismatch")
    layout = identity.get("layout")
    if (
        not isinstance(layout, dict)
        or domain.get("layout")
        != layout.get("result")
        or any(
            operand_layout != domain.get("layout")
            for operand_layout in layout.get("operands", [])
        )
    ):
        reasons.append("layout-mismatch")
    return reasons


def _qualification(
    inventory: dict[str, object],
    frontier_evidence: Sequence[dict[str, object]],
) -> dict[str, object]:
    queries: list[dict[str, object]] = []
    for raw_domain in inventory["domains"]:
        domain = dict(raw_domain)
        identity = dict(domain["identity"])
        considered = [
            {
                "run_id": evidence["run_id"],
                "qualification_status": evidence["qualification_status"],
                "surface_id": evidence["surface"].get("surface_id"),
                "surface_version": evidence["surface"].get("version"),
            }
            for evidence in frontier_evidence
        ]
        mismatch_sets = [
            _mismatch_reasons(identity, evidence)
            for evidence in frontier_evidence
        ]
        reason_codes = sorted(
            {
                reason
                for mismatch in mismatch_sets
                for reason in mismatch
            }
        )
        if not frontier_evidence:
            reason_codes = ["no-evidence-qualified-surface-or-exact-anchor"]
        query: dict[str, object] = {
            "domain_id": domain["domain_id"],
            "domain_class": domain["domain_class"],
            "stable_paths": domain["stable_paths"],
            "domain_identity": identity,
            "status": "unknown",
            "latency_ns": None,
            "effective_rate": None,
            "reason_codes": reason_codes,
            "considered_evidence": considered,
            "minimum_next_measurement": {
                "operation": "MatMul",
                "domain_id": domain["domain_id"],
                "domain_identity": identity,
                "response_target": "latency",
                "search_holdout_identity": "disjoint",
                "hardware_cohort": identity["hardware_cohort"],
            },
        }
        queries.append(query)
    qualified_count = sum(query["status"] == "known" for query in queries)
    required_count = len(queries)
    return {
        "schema": (
            "groundupscale.dev/transformer-matmul-frontier-qualification/"
            "v1alpha1"
        ),
        "status": "qualified" if qualified_count == required_count else "unknown",
        "source_run_id": inventory["source_run_id"],
        "hardware_cohort": inventory["source_hardware_cohort"],
        "required_domain_count": required_count,
        "qualified_domain_count": qualified_count,
        "coverage_fraction": qualified_count / required_count,
        "response_target": "latency",
        "effective_rate_derivation": (
            "declared_work_flop / latency_seconds; latency is primary"
        ),
        "domain_queries": queries,
        "source_frontier_runs": [
            {
                "run_id": evidence["run_id"],
                "manifest_sha256": evidence["manifest_sha256"],
            }
            for evidence in frontier_evidence
        ],
    }


def verify_transformer_matmul_frontier_derivation(
    root: Path,
    manifest: dict[str, Any],
    inventory: dict[str, Any],
    qualification: dict[str, Any],
) -> bool:
    source = manifest.get("source_transformer_run")
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        return False
    source_root = (root / source["path"]).resolve()
    source_manifest_path = source_root / "run.manifest.json"
    if (
        not source_manifest_path.is_file()
        or _sha256(source_manifest_path) != source.get("manifest_sha256")
    ):
        return False
    source_manifest = _load_json(source_manifest_path)
    if source_manifest.get("run_id") != source.get("run_id"):
        return False
    cost_ir, _ = _artifact_document(source_root, source_manifest, "cost-ir")
    expected_inventory = _inventory(source_root, source_manifest, cost_ir)
    if inventory != expected_inventory:
        return False
    frontier_records = manifest.get("frontier_runs")
    if not isinstance(frontier_records, list):
        return False
    evidence: list[dict[str, object]] = []
    for record in frontier_records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            return False
        evidence_item = _frontier_evidence(record["path"])
        if (
            evidence_item["run_id"] != record.get("run_id")
            or evidence_item["manifest_sha256"]
            != record.get("manifest_sha256")
        ):
            return False
        evidence.append(evidence_item)
    return qualification == _qualification(expected_inventory, evidence)


class TransformerMatmulFrontierBundleWriter:
    """Publish the complete demo MatMul domain inventory and match status."""

    def run(
        self,
        artifact_store: str | Path,
        *,
        run_id: str,
        transformer_run: str | Path,
        frontier_runs: Sequence[str | Path],
    ) -> Path:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError(f"unsafe run_id: {run_id!r}")
        source_root = Path(transformer_run).resolve()
        verification = verify_run_bundle(source_root)
        if verification.get("passed") is not True:
            raise ValueError("source Transformer Run Bundle failed verification")
        source_manifest = _load_json(source_root / "run.manifest.json")
        if (
            source_manifest.get("bundle_kind") != "transformer-demo"
            or source_manifest.get("status") != "completed"
            or source_manifest.get("device") != "npu:0"
        ):
            raise ValueError("source is not a completed Transformer demo")
        cost_ir, _ = _artifact_document(source_root, source_manifest, "cost-ir")
        inventory = _inventory(source_root, source_manifest, cost_ir)
        evidence = [_frontier_evidence(path) for path in frontier_runs]
        qualification = _qualification(inventory, evidence)

        runs_root = Path(artifact_store).resolve() / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        destination = runs_root / run_id
        if destination.exists():
            raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
        inventory_path = temporary / "frontier/matmul-domains.json"
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_bytes(_json_bytes(inventory))
        qualification_path = temporary / "frontier/qualification.json"
        qualification_path.write_bytes(_json_bytes(qualification))
        manifest = {
            "schema": "groundupscale.dev/run-manifest/v1alpha1",
            "bundle_kind": "transformer-matmul-frontier",
            "run_id": run_id,
            "status": qualification["status"],
            "created_at": datetime.now(UTC).isoformat(),
            "device": "ascend-npu",
            "hardware_cohort": source_manifest["hardware_cohort"],
            "source_transformer_run": {
                "run_id": source_manifest["run_id"],
                "manifest_sha256": inventory["source_run_manifest_sha256"],
                "path": os.path.relpath(source_root, destination),
            },
            "frontier_runs": [
                {
                    "run_id": item["run_id"],
                    "manifest_sha256": item["manifest_sha256"],
                    "path": os.path.relpath(str(item["root"]), destination),
                }
                for item in evidence
            ],
            "artifacts": [
                {
                    "role": "matmul-domain-inventory",
                    "path": "frontier/matmul-domains.json",
                    "media_type": "application/json",
                    "schema": inventory["schema"],
                    "sha256": _sha256(inventory_path),
                    "produced_by": "groundupscale-transformer-matmul-frontier-v1",
                    "inputs": [],
                },
                {
                    "role": "transformer-matmul-frontier-qualification",
                    "path": "frontier/qualification.json",
                    "media_type": "application/json",
                    "schema": qualification["schema"],
                    "sha256": _sha256(qualification_path),
                    "produced_by": "groundupscale-transformer-matmul-frontier-v1",
                    "inputs": ["matmul-domain-inventory"],
                },
            ],
            "immutability": (
                "writer refuses an existing run_id; source and artifact digests "
                "are authoritative"
            ),
        }
        (temporary / "run.manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(temporary, destination)
        return destination


__all__ = ["TransformerMatmulFrontierBundleWriter"]
