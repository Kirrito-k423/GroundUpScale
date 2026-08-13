"""Qualify MatMul execution domains used by a Transformer Run Bundle."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import median, stdev
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


def _walk_children(item: object) -> list[dict[str, Any]]:
    if not isinstance(item, dict):
        return []
    found = [item]
    children = item.get("children")
    if isinstance(children, list):
        for child in children:
            found.extend(_walk_children(child))
    template = item.get("template")
    if isinstance(template, dict):
        found.extend(_walk_children(template))
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
    inputs_lock: dict[str, Any],
    model_ir: dict[str, Any],
    semantic_ir: dict[str, Any],
    cost_ir: dict[str, Any],
) -> dict[str, object]:
    operations = [
        item
        for item in _walk_items(cost_ir.get("root"))
        if item.get("operation") == "MatMul"
    ]
    documents = inputs_lock.get("documents")
    models = documents.get("models") if isinstance(documents, dict) else None
    workload = documents.get("workload") if isinstance(documents, dict) else None
    if (
        not isinstance(models, dict)
        or len(models) != 1
        or not isinstance(workload, dict)
    ):
        raise ValueError("resolved Model Spec or Workload Spec is missing")
    model_spec = next(iter(models.values()))
    model_root = (
        model_spec.get("spec", {}).get("root")
        if isinstance(model_spec, dict)
        else None
    )
    model_calls = [
        item
        for item in _walk_children(
            workload.get("spec", {}).get("root")
            if isinstance(workload.get("spec"), dict)
            else None
        )
        if item.get("kind") == "model_call"
    ]
    if len(model_calls) != 1 or model_calls[0].get("model", {}).get("path") != (
        "specs/models/two-layer-transformer.yaml"
    ):
        raise ValueError("Workload Spec does not select the Model Spec")
    model_spec_local_ids = {
        item.get("id")
        for item in _walk_children(model_root)
        if item.get("operation") == "MatMul"
    }
    model_ir_paths = {
        item["stable_path"]
        for item in _walk_children(model_ir.get("root"))
        if item.get("operation") == "MatMul"
    }
    semantic_paths = {
        item["stable_path"]
        for item in _walk_items(semantic_ir.get("root"))
        if item.get("operation") == "MatMul"
    }
    cost_semantic_paths = {
        str(item["stable_path"]).removeprefix("cost/")
        for item in operations
    }
    if (
        len(model_spec_local_ids) != 9
        or len(model_ir_paths) != 18
        or semantic_paths != cost_semantic_paths
        or len(semantic_paths) != 18
    ):
        raise ValueError("four-source MatMul inventory mismatch")
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
        metrics = operation.get("metrics")
        declared_work = metrics.get("flops") if isinstance(metrics, dict) else None
        if not isinstance(stable_path, str):
            raise ValueError("demo MatMul Stable Path is missing")
        if (
            not isinstance(declared_work, (int, float))
            or isinstance(declared_work, bool)
            or declared_work <= 0
        ):
            raise ValueError("demo MatMul declared work is missing")
        record = domains.setdefault(
            identity_digest,
            {
                "domain_id": f"matmul-domain:{identity_digest}",
                "domain_class": domain_class,
                "identity": identity,
                "declared_work_flop": float(declared_work),
                "stable_paths": [],
                "source_cost_node_ids": [],
            },
        )
        if record["domain_class"] != domain_class:
            raise ValueError("distinct semantic MatMul domains collapsed")
        if record["declared_work_flop"] != float(declared_work):
            raise ValueError("MatMul domain has inconsistent declared work")
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
        "source_closure": {
            "model_spec_matmul_definition_count": len(model_spec_local_ids),
            "model_ir_indexed_matmul_count": len(model_ir_paths),
            "workload_model_call_count": len(model_calls),
            "semantic_ir_matmul_count": len(semantic_paths),
            "cost_ir_matmul_count": len(operations),
        },
        "matmul_leaf_count": len(operations),
        "distinct_domain_count": len(ordered_domains),
        "domains": ordered_domains,
    }


def _frontier_evidence(path: str | Path) -> dict[str, object]:
    root = Path(path).resolve()
    if root.is_file():
        document = _load_json(root)
        schema = document.get("schema")
        if schema == (
            "groundupscale.dev/transformer-matmul-exact-anchor/v1alpha1"
        ):
            identity = document.get("domain_identity")
            latency_ns = document.get("latency_ns")
            search_ids = document.get("search_run_ids")
            holdout_ids = document.get("holdout_run_ids")
            if (
                document.get("status") != "qualified"
                or document.get("response_target") != "latency"
                or not isinstance(identity, dict)
                or document.get("domain_identity_digest")
                != content_fingerprint(identity)
                or not isinstance(latency_ns, (int, float))
                or isinstance(latency_ns, bool)
                or latency_ns <= 0
                or not isinstance(search_ids, list)
                or not isinstance(holdout_ids, list)
                or len(search_ids) < 3
                or len(holdout_ids) < 3
                or len(set(search_ids + holdout_ids))
                != len(search_ids) + len(holdout_ids)
            ):
                raise ValueError(f"invalid exact Anchor evidence: {root}")
            return {
                "root": str(root),
                "manifest_sha256": _sha256(root),
                "run_id": document.get("evidence_id"),
                "evidence_kind": "exact-anchor",
                "hardware_cohort": document.get("hardware_cohort"),
                "qualification_status": "qualified",
                "surface": {},
                "anchor": document,
            }
        if (
            schema
            == "groundupscale.dev/operator-frontier-qualification-unknown/v1alpha1"
            and document.get("status") == "unknown"
        ):
            return {
                "root": str(root),
                "manifest_sha256": _sha256(root),
                "run_id": document.get("published_run_bundle"),
                "evidence_kind": "incomplete-surface",
                "hardware_cohort": document.get("hardware_cohort"),
                "qualification_status": "unknown",
                "surface": {},
                "anchor": None,
            }
        raise ValueError(f"unsupported Frontier evidence document: {root}")
    verification = verify_run_bundle(root)
    if verification.get("passed") is not True:
        raise ValueError(f"Frontier Run Bundle failed verification: {root}")
    manifest = _load_json(root / "run.manifest.json")
    if manifest.get("bundle_kind") == "transformer-matmul-exact-anchor":
        anchor, _ = _artifact_document(
            root, manifest, "transformer-matmul-exact-anchor"
        )
        return {
            "root": str(root),
            "manifest_sha256": _sha256(root / "run.manifest.json"),
            "run_id": manifest.get("run_id"),
            "evidence_kind": "exact-anchor",
            "hardware_cohort": manifest.get("hardware_cohort"),
            "qualification_status": anchor.get("status"),
            "surface": {},
            "anchor": anchor,
        }
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
        "evidence_kind": "surface",
        "hardware_cohort": manifest.get("hardware_cohort"),
        "qualification_status": qualification.get("status"),
        "surface": surface,
        "anchor": None,
    }


def _measurement_session(path: str | Path) -> dict[str, object]:
    root = Path(path).resolve()
    verification = verify_run_bundle(root)
    if verification.get("passed") is not True:
        raise ValueError(
            f"measurement Run Bundle failed verification: {root}: "
            f"{verification.get('failures')}"
        )
    manifest = _load_json(root / "run.manifest.json")
    if (
        manifest.get("bundle_kind") != "exact-shape-measurement"
        or manifest.get("status") != "completed"
        or manifest.get("device") != "ascend-npu"
    ):
        raise ValueError(f"not a completed Ascend measurement: {root}")
    case, _ = _artifact_document(root, manifest, "benchmark-case")
    candidate, _ = _artifact_document(root, manifest, "candidate-identity")
    execution, _ = _artifact_document(root, manifest, "execution-contract")
    correctness, _ = _artifact_document(
        root, manifest, "correctness-observation"
    )
    raw_timing, _ = _artifact_document(
        root, manifest, "raw-timing-observation"
    )
    environment, _ = _artifact_document(root, manifest, "environment")
    session = environment.get("measurement_session")
    summary = raw_timing.get("summary")
    domain = case.get("domain_identity")
    domain_digest = case.get("domain_identity_digest")
    domain_record = candidate.get("transformer_matmul_domain")
    execution_domain = execution.get("transformer_matmul_domain")
    observation_validity = manifest.get("observation_validity")
    if (
        not isinstance(domain, dict)
        or domain_digest != content_fingerprint(domain)
        or domain_record
        != {
            "identity": domain,
            "identity_digest": domain_digest,
            "declared_work_flop": case.get("declared_work_flop"),
        }
        or execution_domain != domain_record
        or candidate.get("candidate_family") != domain.get("candidate_family")
        or candidate.get("execution_mode") != domain.get("execution_mode")
        or manifest.get("hardware_cohort") != domain.get("hardware_cohort")
        or correctness.get("status") != "passed"
        or not isinstance(observation_validity, dict)
        or observation_validity.get("status") != "valid"
        or observation_validity.get("timing_quality") != "passed"
        or not isinstance(summary, dict)
        or not isinstance(summary.get("median"), (int, float))
        or not isinstance(session, dict)
        or not isinstance(session.get("process_id"), int)
        or not isinstance(session.get("process_started_at"), str)
    ):
        raise ValueError(f"incomplete Transformer MatMul evidence: {root}")
    return {
        "root": str(root),
        "run_id": manifest["run_id"],
        "manifest_sha256": _sha256(root / "run.manifest.json"),
        "hardware_cohort": manifest["hardware_cohort"],
        "domain_identity": domain,
        "domain_identity_digest": domain_digest,
        "declared_work_flop": case["declared_work_flop"],
        "candidate_id": candidate["candidate_id"],
        "candidate_family": candidate["candidate_family"],
        "candidate_digest": candidate["candidate_digest"],
        "median_ns": float(summary["median"]),
        "session_identity": [
            session["process_id"],
            session["process_started_at"],
        ],
    }


def _exact_anchor(
    search: Sequence[dict[str, object]],
    holdout: Sequence[dict[str, object]],
    *,
    evidence_id: str,
) -> dict[str, object]:
    if len(search) < 3 or len(holdout) < 3:
        raise ValueError("at least three search and holdout sessions are required")
    sessions = [*search, *holdout]
    identity_fields = (
        "hardware_cohort",
        "domain_identity",
        "domain_identity_digest",
        "declared_work_flop",
        "candidate_id",
        "candidate_family",
        "candidate_digest",
    )
    if any(
        item[field] != sessions[0][field]
        for item in sessions[1:]
        for field in identity_fields
    ):
        raise ValueError("search and holdout domain identities differ")
    run_ids = [str(item["run_id"]) for item in sessions]
    process_identities = [tuple(item["session_identity"]) for item in sessions]
    if (
        len(set(run_ids)) != len(run_ids)
        or len(set(process_identities)) != len(process_identities)
    ):
        raise ValueError("search and holdout sessions are not independent")

    def relative_range(items: Sequence[dict[str, object]]) -> float:
        values = [float(item["median_ns"]) for item in items]
        center = float(median(values))
        return (max(values) - min(values)) / center

    search_center = float(median([item["median_ns"] for item in search]))
    holdout_values = [float(item["median_ns"]) for item in holdout]
    holdout_center = float(median(holdout_values))
    search_range = relative_range(search)
    holdout_range = relative_range(holdout)
    lane_gap = abs(search_center - holdout_center) / holdout_center
    reasons = []
    if search_range > 0.10:
        reasons.append("search-repeatability-failed")
    if holdout_range > 0.10:
        reasons.append("independent-holdout-repeatability-failed")
    if lane_gap > 0.10:
        reasons.append("search-holdout-relative-gap-failed")
    status = "unknown" if reasons else "qualified"
    return {
        "schema": "groundupscale.dev/transformer-matmul-exact-anchor/v1alpha1",
        "evidence_id": evidence_id,
        "status": status,
        "response_target": "latency",
        "hardware_cohort": sessions[0]["hardware_cohort"],
        "domain_identity": sessions[0]["domain_identity"],
        "domain_identity_digest": sessions[0]["domain_identity_digest"],
        "declared_work_flop": sessions[0]["declared_work_flop"],
        "candidate_id": sessions[0]["candidate_id"],
        "candidate_family": sessions[0]["candidate_family"],
        "candidate_digest": sessions[0]["candidate_digest"],
        "latency_ns": holdout_center if status == "qualified" else None,
        "standard_uncertainty_ns": (
            float(stdev(holdout_values))
            if status == "qualified" and len(holdout_values) > 1
            else None
        ),
        "reason_codes": reasons,
        "repeatability": {
            "search_relative_range": search_range,
            "holdout_relative_range": holdout_range,
            "search_holdout_relative_gap": lane_gap,
            "maximum_relative_range": 0.10,
        },
        "estimator": "median(independent-holdout-session-medians)",
        "search_run_ids": [item["run_id"] for item in search],
        "holdout_run_ids": [item["run_id"] for item in holdout],
        "session_evidence": {
            "search": [
                {key: item[key] for key in ("run_id", "manifest_sha256", "median_ns")}
                for item in search
            ],
            "holdout": [
                {key: item[key] for key in ("run_id", "manifest_sha256", "median_ns")}
                for item in holdout
            ],
        },
    }


def _mismatch_reasons(
    identity: dict[str, object], evidence: dict[str, object]
) -> list[str]:
    if evidence.get("evidence_kind") == "exact-anchor":
        anchor = evidence.get("anchor")
        if not isinstance(anchor, dict):
            return ["exact-anchor-invalid"]
        reasons = []
        if evidence["qualification_status"] != "qualified":
            reasons.append("exact-anchor-not-qualified")
        if evidence["hardware_cohort"] != identity["hardware_cohort"]:
            reasons.append("hardware-cohort-mismatch")
        if anchor.get("domain_identity_digest") != content_fingerprint(identity):
            reasons.append("complete-domain-identity-mismatch")
        if anchor.get("domain_identity") != identity:
            reasons.append("complete-domain-contract-mismatch")
        return reasons
    if evidence.get("evidence_kind") == "incomplete-surface":
        return ["surface-not-qualified"]
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
                **(
                    {"run_id": evidence["run_id"]}
                    if evidence["evidence_kind"] == "surface"
                    else {
                        "evidence_id": evidence["run_id"],
                        "evidence_kind": evidence["evidence_kind"],
                    }
                ),
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
        compatible_unqualified = [
            (evidence, mismatches)
            for evidence, mismatches in zip(
                frontier_evidence, mismatch_sets, strict=True
            )
            if evidence.get("evidence_kind") == "exact-anchor"
            and mismatches == ["exact-anchor-not-qualified"]
        ]
        if compatible_unqualified:
            reason_codes = ["exact-anchor-not-qualified"]
        if not frontier_evidence:
            reason_codes = ["no-evidence-qualified-surface-or-exact-anchor"]
        matches = [
            evidence
            for evidence, mismatches in zip(
                frontier_evidence, mismatch_sets, strict=True
            )
            if not mismatches
        ]
        if len(matches) > 1:
            reason_codes = ["ambiguous-qualified-surface-or-exact-anchor"]
            matches = []
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
        if compatible_unqualified:
            compatible_anchor = compatible_unqualified[0][0].get("anchor")
            if isinstance(compatible_anchor, dict):
                query["evidence_boundary"] = {
                    "reason_codes": compatible_anchor.get("reason_codes", []),
                    "repeatability": compatible_anchor.get("repeatability"),
                    "additional_rounds_allowed": False,
                }
        if len(matches) == 1:
            match = matches[0]
            anchor = match.get("anchor")
            if isinstance(anchor, dict):
                latency_ns = float(anchor["latency_ns"])
                declared_work = float(domain["declared_work_flop"])
                query.update(
                    {
                        "status": "known",
                        "latency_ns": latency_ns,
                        "effective_rate": declared_work * 1_000_000_000 / latency_ns,
                        "effective_rate_derivation": (
                            "declared_work_flop / latency_seconds"
                        ),
                        "reason_codes": [],
                        "minimum_next_measurement": None,
                        "selected_evidence_id": match["run_id"],
                    }
                )
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


def _contiguous_stride(shape: list[int]) -> list[int]:
    stride: list[int] = []
    running = 1
    for dimension in reversed(shape):
        stride.append(running)
        running *= dimension
    return list(reversed(stride))


def _storage_contract(
    contract: dict[str, object],
    *,
    domain_class: str,
    operand_index: int,
) -> dict[str, object]:
    shape = list(contract["shape"])
    layout = contract["layout"]
    if domain_class == "attention-qk" and operand_index == 0:
        storage_shape = [1, 512, 8, 64]
        permutation = [0, 2, 1, 3]
    elif domain_class == "attention-qk" and operand_index == 1:
        storage_shape = [1, 512, 8, 64]
        permutation = [0, 2, 3, 1]
    elif domain_class == "attention-context" and operand_index == 1:
        storage_shape = [1, 512, 8, 64]
        permutation = [0, 2, 1, 3]
    elif layout == "transposed":
        raise ValueError("unrecognized Transformer transpose provenance")
    else:
        storage_shape = shape
        permutation = list(range(len(shape)))
    return {
        "logical_shape": shape,
        "storage_shape": storage_shape,
        "storage_stride": _contiguous_stride(storage_shape),
        "permutation": permutation,
        "layout": layout,
    }


def transformer_matmul_measurement_case(
    transformer_run: str | Path,
    *,
    domain_class: str,
    seed: int,
    warmup_iterations: int,
    repetitions: int,
    inner_iterations: int,
) -> dict[str, object]:
    """Derive one exact measurement case from the verified public inventory."""

    source_root = Path(transformer_run).resolve()
    if verify_run_bundle(source_root).get("passed") is not True:
        raise ValueError("source Transformer Run Bundle failed verification")
    manifest = _load_json(source_root / "run.manifest.json")
    inputs_lock, _ = _artifact_document(
        source_root, manifest, "resolved-input-lock"
    )
    model_ir, _ = _artifact_document(source_root, manifest, "model-ir")
    semantic_ir, _ = _artifact_document(source_root, manifest, "semantic-ir")
    cost_ir, _ = _artifact_document(source_root, manifest, "cost-ir")
    inventory = _inventory(
        source_root, manifest, inputs_lock, model_ir, semantic_ir, cost_ir
    )
    domains = [
        item
        for item in inventory["domains"]
        if item["domain_class"] == domain_class
    ]
    if len(domains) != 1:
        raise ValueError(f"unknown or ambiguous MatMul domain: {domain_class}")
    domain = domains[0]
    identity = domain["identity"]
    operands = identity["operand_contracts"]
    result = identity["result_contract"]
    candidate = (
        "torch.matmul.transpose-1-2-contiguous"
        if domain_class == "attention-context"
        else "torch.matmul"
    )
    rank = len(result["shape"])
    return {
        "schema": "groundupscale.dev/exact-shape-matmul-case/v1alpha2",
        "operation": "MatMul",
        "shape": {
            "left": operands[0]["shape"],
            "right": operands[1]["shape"],
            "result": result["shape"],
        },
        "dtype": identity["dtype"],
        "layout": identity["layout"],
        "operand_storage_contracts": [
            _storage_contract(
                operands[0], domain_class=domain_class, operand_index=0
            ),
            _storage_contract(
                operands[1], domain_class=domain_class, operand_index=1
            ),
        ],
        "result_transform": {
            "permutation": (
                [0, 2, 1, 3]
                if domain_class == "attention-context"
                else list(range(rank))
            ),
            "materialize_contiguous": domain_class == "attention-context",
        },
        "domain_identity": identity,
        "domain_identity_digest": content_fingerprint(identity),
        "declared_work_flop": domain["declared_work_flop"],
        "seed": seed,
        "candidate": candidate,
        "warmup_iterations": warmup_iterations,
        "repetitions": repetitions,
        "inner_iterations": inner_iterations,
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
    source_verification = verify_run_bundle(source_root)
    if (
        source_verification.get("passed") is not True
        or source_manifest.get("run_id") != source.get("run_id")
        or source_manifest.get("bundle_kind") != "transformer-demo"
        or source_manifest.get("status") != "completed"
        or source_manifest.get("device") != "npu:0"
        or source_manifest.get("hardware_cohort")
        != manifest.get("hardware_cohort")
    ):
        return False
    inputs_lock, _ = _artifact_document(
        source_root, source_manifest, "resolved-input-lock"
    )
    model_ir, _ = _artifact_document(source_root, source_manifest, "model-ir")
    semantic_ir, _ = _artifact_document(
        source_root, source_manifest, "semantic-ir"
    )
    cost_ir, _ = _artifact_document(source_root, source_manifest, "cost-ir")
    expected_inventory = _inventory(
        source_root,
        source_manifest,
        inputs_lock,
        model_ir,
        semantic_ir,
        cost_ir,
    )
    if inventory != expected_inventory:
        return False
    frontier_records = manifest.get("frontier_runs")
    if not isinstance(frontier_records, list):
        return False
    evidence: list[dict[str, object]] = []
    for record in frontier_records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            return False
        evidence_item = _frontier_evidence((root / record["path"]).resolve())
        if (
            evidence_item["run_id"] != record.get("run_id")
            or evidence_item["manifest_sha256"]
            != record.get("manifest_sha256")
        ):
            return False
        evidence.append(evidence_item)
    return qualification == _qualification(expected_inventory, evidence)


def verify_transformer_matmul_exact_anchor_derivation(
    root: Path,
    manifest: dict[str, Any],
    anchor: dict[str, Any],
) -> bool:
    source_runs = manifest.get("source_runs")
    if not isinstance(source_runs, dict):
        return False

    def sessions(lane: str) -> list[dict[str, object]]:
        records = source_runs.get(lane)
        if not isinstance(records, list):
            raise ValueError("exact Anchor source lane is missing")
        result = []
        for record in records:
            if not isinstance(record, dict) or not isinstance(
                record.get("path"), str
            ):
                raise ValueError("invalid exact Anchor source record")
            item = _measurement_session((root / record["path"]).resolve())
            if (
                item["run_id"] != record.get("run_id")
                or item["manifest_sha256"] != record.get("manifest_sha256")
            ):
                raise ValueError("exact Anchor source identity mismatch")
            result.append(item)
        return result

    expected = _exact_anchor(
        sessions("search"),
        sessions("holdout"),
        evidence_id=str(manifest.get("run_id")),
    )
    if anchor == expected:
        return True
    legacy_expected = dict(expected)
    legacy_expected.pop("reason_codes", None)
    legacy_expected.pop("repeatability", None)
    return expected.get("status") == "qualified" and anchor == legacy_expected


class TransformerMatmulExactAnchorBundleWriter:
    """Qualify one exact Transformer MatMul domain from disjoint sessions."""

    def run(
        self,
        artifact_store: str | Path,
        *,
        run_id: str,
        search_runs: Sequence[str | Path],
        holdout_runs: Sequence[str | Path],
    ) -> Path:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError(f"unsafe run_id: {run_id!r}")
        search = [_measurement_session(path) for path in search_runs]
        holdout = [_measurement_session(path) for path in holdout_runs]
        anchor = _exact_anchor(search, holdout, evidence_id=run_id)
        runs_root = Path(artifact_store).resolve() / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        destination = runs_root / run_id
        if destination.exists():
            raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
        anchor_path = temporary / "frontier/exact-anchor.json"
        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        anchor_path.write_bytes(_json_bytes(anchor))

        def records(items: Sequence[dict[str, object]]) -> list[dict[str, object]]:
            return [
                {
                    "run_id": item["run_id"],
                    "manifest_sha256": item["manifest_sha256"],
                    "path": os.path.relpath(str(item["root"]), destination),
                }
                for item in items
            ]

        manifest = {
            "schema": "groundupscale.dev/run-manifest/v1alpha1",
            "bundle_kind": "transformer-matmul-exact-anchor",
            "run_id": run_id,
            "status": anchor["status"],
            "created_at": datetime.now(UTC).isoformat(),
            "device": "ascend-npu",
            "hardware_cohort": anchor["hardware_cohort"],
            "source_runs": {
                "search": records(search),
                "holdout": records(holdout),
            },
            "artifacts": [
                {
                    "role": "transformer-matmul-exact-anchor",
                    "path": "frontier/exact-anchor.json",
                    "media_type": "application/json",
                    "schema": anchor["schema"],
                    "sha256": _sha256(anchor_path),
                    "produced_by": "groundupscale-transformer-matmul-exact-anchor-v1",
                    "inputs": [],
                }
            ],
            "immutability": (
                "writer refuses an existing run_id; source and artifact digests "
                "are authoritative"
            ),
        }
        (temporary / "run.manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(temporary, destination)
        return destination


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
        inputs_lock, _ = _artifact_document(
            source_root, source_manifest, "resolved-input-lock"
        )
        model_ir, _ = _artifact_document(
            source_root, source_manifest, "model-ir"
        )
        semantic_ir, _ = _artifact_document(
            source_root, source_manifest, "semantic-ir"
        )
        cost_ir, _ = _artifact_document(source_root, source_manifest, "cost-ir")
        inventory = _inventory(
            source_root,
            source_manifest,
            inputs_lock,
            model_ir,
            semantic_ir,
            cost_ir,
        )
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


__all__ = [
    "TransformerMatmulExactAnchorBundleWriter",
    "TransformerMatmulFrontierBundleWriter",
    "transformer_matmul_measurement_case",
]
