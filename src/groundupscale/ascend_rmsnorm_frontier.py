"""Replayable Ascend RMSNorm compound-operation Frontier publication."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite, sqrt
from pathlib import Path
from typing import Iterable

from groundupscale.ir import canonical_data, content_fingerprint
from groundupscale.ir.cost import CostOperation, OperatorPhase
from groundupscale.run_bundle import RUN_ID_PATTERN, RunBundleExistsError


QUALIFICATION_SCHEMA = (
    "groundupscale.dev/compound-operator-frontier-qualification/v1alpha1"
)
PHASE_GRAPH_SCHEMA = "groundupscale.dev/operator-phase-graph/v1alpha1"
DIAGNOSTIC_SCHEMA = "groundupscale.dev/compound-operator-diagnostic/v1alpha1"
PRODUCER = "groundupscale-rmsnorm-frontier-v1"


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            canonical_data(value), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        canonical_data(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _phase_document(phase: OperatorPhase) -> dict[str, object]:
    return canonical_data(phase)


def _validate_domain(
    operation: CostOperation, execution_domain: dict[str, object]
) -> None:
    expected = {
        "stable_path": operation.stable_path,
        "operand_shapes": [list(tensor.shape) for tensor in operation.operand_types],
        "result_shapes": [list(tensor.shape) for tensor in operation.result_types],
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "execution_mode": "pytorch-eager",
        "logical_device": "npu:0",
    }
    if (
        not isinstance(execution_domain.get("hardware_cohort"), str)
        or not execution_domain["hardware_cohort"]
        or any(execution_domain.get(key) != value for key, value in expected.items())
        or any(
            tensor.dtype != "float32" or tensor.layout != "contiguous"
            for tensor in operation.operand_types + operation.result_types
        )
    ):
        raise ValueError("RMSNorm execution domain mismatch")


def _validated_phase_evidence(
    phase: OperatorPhase,
    evidence: dict[str, object],
    execution_domain: dict[str, object],
) -> dict[str, object]:
    evidence_body = {
        key: value for key, value in evidence.items() if key != "input_digest"
    }
    if evidence.get("input_digest") != content_fingerprint(evidence_body):
        raise ValueError("phase evidence input digest mismatch")
    if evidence.get("phase_id") != phase.phase_id:
        raise ValueError("phase evidence identity mismatch")
    if evidence.get("phase_name") != phase.phase_name:
        raise ValueError("phase evidence identity mismatch")
    if evidence.get("operation_class") != phase.operation_class:
        raise ValueError("phase evidence operation class mismatch")
    if evidence.get("evidence_kind") not in {
        "exact-operation-probe",
        "semantically-matching-capability-class",
    }:
        raise ValueError("phase evidence kind is not semantically qualified")
    if evidence.get("execution_domain") != execution_domain:
        raise ValueError("phase evidence execution domain mismatch")
    duration = evidence.get("duration_ns")
    uncertainty = evidence.get("standard_uncertainty_ns")
    candidate = evidence.get("candidate")
    source = evidence.get("source")
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not isfinite(float(duration))
        or duration <= 0
        or not isinstance(uncertainty, (int, float))
        or isinstance(uncertainty, bool)
        or not isfinite(float(uncertainty))
        or uncertainty < 0
        or not isinstance(candidate, dict)
        or not all(
            isinstance(candidate.get(field), str) and candidate[field]
            for field in ("candidate_id", "candidate_family", "candidate_version")
        )
        or not isinstance(source, dict)
        or source.get("correctness") != "passed"
        or source.get("stability") != "passed"
        or source.get("lane") != "independent-holdout"
        or not isinstance(source.get("run_id"), str)
        or not isinstance(source.get("artifact_ref"), str)
        or not str(source["artifact_ref"]).startswith("artifact://")
        or not isinstance(source.get("run_manifest_sha256"), str)
        or len(str(source["run_manifest_sha256"])) != 64
    ):
        raise ValueError("phase evidence qualification gate failed")
    return canonical_data(evidence)


class RmsNormOperatorFrontierBundleWriter:
    """Publish known or structured-unknown RMSNorm phase evidence."""

    def run(
        self,
        artifact_store: str | Path,
        *,
        run_id: str,
        operation: CostOperation,
        execution_domain: dict[str, object],
        phase_evidence: Iterable[dict[str, object]],
    ) -> Path:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(f"unsafe run_id: {run_id!r}")
        if operation.operation != "RMSNorm" or operation.phase_graph is None:
            raise ValueError("an explicit RMSNorm Operator Phase Graph is required")
        _validate_domain(operation, execution_domain)
        graph = operation.phase_graph
        evidence_by_phase: dict[str, dict[str, object]] = {}
        for evidence in phase_evidence:
            phase_id = evidence.get("phase_id")
            if not isinstance(phase_id, str) or phase_id in evidence_by_phase:
                raise ValueError("duplicate or invalid phase evidence identity")
            evidence_by_phase[phase_id] = evidence
        unknown_ids = set(evidence_by_phase) - {
            phase.phase_id for phase in graph.phases
        }
        if unknown_ids:
            raise ValueError("phase evidence does not belong to the RMSNorm graph")

        validated_evidence: list[dict[str, object]] = []
        scheduled_phases: list[dict[str, object]] = []
        missing_evidence: list[dict[str, object]] = []
        for phase in graph.phases:
            evidence = evidence_by_phase.get(phase.phase_id)
            if evidence is None:
                missing_evidence.append(
                    {
                        "phase_id": phase.phase_id,
                        "phase_name": phase.phase_name,
                        "operation_class": phase.operation_class,
                        "required_evidence": (
                            "semantically matching capability class or exact operation "
                            "probe for the complete execution domain"
                        ),
                    }
                )
                scheduled_phases.append(
                    {
                        **_phase_document(phase),
                        "status": "unknown",
                        "candidate": None,
                        "local_duration_ns": None,
                        "standard_uncertainty_ns": None,
                        "resource_composition": "unknown",
                        "evidence_ref": None,
                    }
                )
                continue
            validated = _validated_phase_evidence(
                phase, evidence, execution_domain
            )
            validated_evidence.append(validated)
            source = validated["source"]
            assert isinstance(source, dict)
            scheduled_phases.append(
                {
                    **_phase_document(phase),
                    "status": "known",
                    "candidate": validated["candidate"],
                    "local_duration_ns": validated["duration_ns"],
                    "standard_uncertainty_ns": validated[
                        "standard_uncertainty_ns"
                    ],
                    "resource_composition": (
                        "exact-operation-duration"
                        if validated["evidence_kind"] == "exact-operation-probe"
                        else "max(matching-compute-capability,memory-pattern-floor)"
                    ),
                    "evidence_ref": (
                        f"run-bundle://{source['run_id']}#"
                        f"{source['artifact_ref']}"
                    ),
                }
            )

        complete = not missing_evidence
        total = (
            sum(float(item["local_duration_ns"]) for item in scheduled_phases)
            if complete
            else None
        )
        standard_uncertainty = (
            sqrt(
                sum(
                    float(item["standard_uncertainty_ns"]) ** 2
                    for item in scheduled_phases
                )
            )
            if complete
            else None
        )
        frontier = {
            "status": "known" if complete else "unknown",
            "duration_ns": total,
            "standard_uncertainty_ns": standard_uncertainty,
            "composition_policy": "serialized-no-chunk",
            "formula": "sum(phase.local_duration_ns)",
        }
        schedule = {
            "policy": "serialized-no-chunk",
            "chunk_pipeline_contract_id": None,
            "overlap_evidence_refs": [],
            "phases": scheduled_phases,
            "selected_duration_ns": total,
        }
        phase_graph_document = {
            "schema": PHASE_GRAPH_SCHEMA,
            "graph_id": graph.graph_id,
            "operation": operation.operation,
            "cost_node_id": operation.node_id,
            "stable_path": operation.stable_path,
            "execution_domain": execution_domain,
            "phases": [_phase_document(phase) for phase in graph.phases],
            "output_phase_ids": list(graph.output_phase_ids),
        }
        phase_graph_document["input_digest"] = _canonical_digest(
            phase_graph_document
        )
        qualification = {
            "schema": QUALIFICATION_SCHEMA,
            "status": "qualified" if complete else "unknown",
            "operation": operation.operation,
            "stable_path": operation.stable_path,
            "hardware_cohort": execution_domain["hardware_cohort"],
            "execution_domain": execution_domain,
            "phase_graph_ref": "artifact://frontier/phase-graph.json",
            "phase_graph_digest": phase_graph_document["input_digest"],
            "phase_evidence": validated_evidence,
            "source_evidence_digest": _canonical_digest(validated_evidence),
            "selected_candidate": {
                "candidate_id": f"ascend-rmsnorm-explicit-phase-graph-{run_id}",
                "candidate_version": "v1",
                "phase_schedule": schedule,
            },
            "operator_frontier": frontier,
            "missing_evidence": missing_evidence,
            "uncertainty_policy": {
                "policy_id": "independent-phase-standard-uncertainty-rss-v1",
                "combination": "root-sum-square",
                "target_coverage": 0.6827,
            },
        }
        qualification["input_digest"] = _canonical_digest(qualification)
        diagnostic = {
            "schema": DIAGNOSTIC_SCHEMA,
            "qualification_ref": "artifact://frontier/qualification.json",
            "qualification_digest": qualification["input_digest"],
            "operation": operation.operation,
            "stable_path": operation.stable_path,
            "hardware_cohort": execution_domain["hardware_cohort"],
            "operator_frontier": frontier,
            "missing_evidence": missing_evidence,
        }
        diagnostic["input_digest"] = _canonical_digest(diagnostic)

        runs_root = Path(artifact_store).resolve() / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        destination = runs_root / run_id
        if destination.exists():
            raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
        artifacts: list[dict[str, object]] = []
        documents = (
            (
                "operator-phase-graph",
                "frontier/phase-graph.json",
                phase_graph_document,
                [],
            ),
            (
                "compound-operator-frontier-qualification",
                "frontier/qualification.json",
                qualification,
                ["operator-phase-graph"],
            ),
            (
                "compound-operator-diagnostic",
                "diagnostic/evidence.json",
                diagnostic,
                ["compound-operator-frontier-qualification"],
            ),
        )
        for role, relative, document, inputs in documents:
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_json_bytes(document))
            artifacts.append(
                {
                    "role": role,
                    "path": relative,
                    "media_type": "application/json",
                    "schema": document["schema"],
                    "sha256": _sha256(path),
                    "produced_by": PRODUCER,
                    "inputs": inputs,
                }
            )
        manifest = {
            "schema": "groundupscale.dev/run-manifest/v1alpha1",
            "bundle_kind": "compound-operator-frontier",
            "run_id": run_id,
            "status": "completed",
            "created_at": datetime.now(UTC).isoformat(),
            "device": "ascend-npu",
            "hardware_cohort": execution_domain["hardware_cohort"],
            "operation": "RMSNorm",
            "stable_path": operation.stable_path,
            "qualification_status": qualification["status"],
            "producer_lineage": {"producer": PRODUCER},
            "artifacts": artifacts,
            "immutability": (
                "writer refuses an existing run_id; artifact and source digests "
                "are authoritative"
            ),
        }
        (temporary / "run.manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(temporary, destination)
        return destination


__all__ = ["RmsNormOperatorFrontierBundleWriter"]
