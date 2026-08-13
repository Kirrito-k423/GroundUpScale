"""Replayable Ascend RMSNorm phase measurement and Frontier publication."""

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
from groundupscale.run_bundle import (
    RUN_ID_PATTERN,
    RunBundleExistsError,
    verify_run_bundle,
)


PHASE_OBSERVATION_SCHEMA = (
    "groundupscale.dev/operator-phase-capability-observation/v1alpha1"
)
QUALIFICATION_SCHEMA = (
    "groundupscale.dev/compound-operator-frontier-qualification/v1alpha1"
)
PHASE_GRAPH_SCHEMA = "groundupscale.dev/operator-phase-graph/v1alpha1"
DIAGNOSTIC_SCHEMA = "groundupscale.dev/compound-operator-diagnostic/v1alpha1"
MEASUREMENT_PRODUCER = "groundupscale-rmsnorm-phase-measurement-v1"
FRONTIER_PRODUCER = "groundupscale-rmsnorm-frontier-v1"


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(canonical_data(value), ensure_ascii=False, indent=2, sort_keys=True)
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


def _validate_domain(operation: CostOperation, domain: dict[str, object]) -> None:
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
        not isinstance(domain.get("hardware_cohort"), str)
        or not domain["hardware_cohort"]
        or any(domain.get(key) != value for key, value in expected.items())
        or any(
            tensor.dtype != "float32" or tensor.layout != "contiguous"
            for tensor in operation.operand_types + operation.result_types
        )
    ):
        raise ValueError("RMSNorm execution domain mismatch")


def _local_constraints(
    *,
    evidence_kind: str,
    compute_or_exact_duration_ns: float,
    memory_pattern_floor_ns: float,
) -> tuple[dict[str, float | None], float]:
    if evidence_kind == "exact-operation-probe":
        constraints = {
            "exact_operation_duration_ns": compute_or_exact_duration_ns,
            "matching_compute_capability_duration_ns": None,
            "memory_pattern_floor_ns": memory_pattern_floor_ns,
        }
    elif evidence_kind == "semantically-matching-capability-class":
        constraints = {
            "exact_operation_duration_ns": None,
            "matching_compute_capability_duration_ns": compute_or_exact_duration_ns,
            "memory_pattern_floor_ns": memory_pattern_floor_ns,
        }
    else:
        raise ValueError("phase evidence kind is not semantically qualified")
    return constraints, max(compute_or_exact_duration_ns, memory_pattern_floor_ns)


class RmsNormPhaseMeasurementBundleWriter:
    """Publish one immutable phase observation from one independent session."""

    def run(
        self,
        artifact_store: str | Path,
        *,
        run_id: str,
        phase: OperatorPhase,
        execution_domain: dict[str, object],
        lane: str,
        evidence_kind: str,
        candidate: dict[str, str],
        compute_or_exact_duration_ns: float,
        memory_pattern_floor_ns: float,
        compute_or_exact_capability_profile_ref: str,
        memory_pattern_capability_profile_ref: str,
        standard_uncertainty_ns: float,
        raw_samples_ns: Iterable[float],
        memory_pattern_raw_samples_ns: Iterable[float],
        correctness: str = "passed",
        timing_quality: str = "passed",
        run_metadata: dict[str, object] | None = None,
        compilation_fingerprint: str,
    ) -> Path:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(f"unsafe run_id: {run_id!r}")
        if lane not in {"search", "independent-holdout"}:
            raise ValueError("phase measurement lane must be search or independent-holdout")
        samples = [float(value) for value in raw_samples_ns]
        memory_samples = [float(value) for value in memory_pattern_raw_samples_ns]
        numeric = (
            compute_or_exact_duration_ns,
            memory_pattern_floor_ns,
            standard_uncertainty_ns,
            *samples,
            *memory_samples,
        )
        if (
            correctness != "passed"
            or timing_quality != "passed"
            or len(samples) < 3
            or len(memory_samples) < 3
            or any(not isfinite(value) or value <= 0 for value in numeric[:2])
            or not isfinite(standard_uncertainty_ns)
            or standard_uncertainty_ns < 0
            or any(not isfinite(value) or value <= 0 for value in (*samples, *memory_samples))
            or not all(
                isinstance(candidate.get(field), str) and candidate[field]
                for field in ("candidate_id", "candidate_family", "candidate_version")
            )
            or not compilation_fingerprint
        ):
            raise ValueError("phase measurement qualification gate failed")
        capability_profile_refs = {
            "compute_or_exact": compute_or_exact_capability_profile_ref,
            "memory_pattern": memory_pattern_capability_profile_ref,
        }
        if (
            not all(
                isinstance(reference, str) and reference
                for reference in capability_profile_refs.values()
            )
            or len(set(capability_profile_refs.values())) != 2
        ):
            raise ValueError("phase measurement requires independent capability-profile references")
        constraints, local_duration = _local_constraints(
            evidence_kind=evidence_kind,
            compute_or_exact_duration_ns=compute_or_exact_duration_ns,
            memory_pattern_floor_ns=memory_pattern_floor_ns,
        )
        metadata = dict(run_metadata or {})
        metadata.setdefault("finished_at", datetime.now(UTC).isoformat())
        observation: dict[str, object] = {
            "schema": PHASE_OBSERVATION_SCHEMA,
            "phase_id": phase.phase_id,
            "phase_name": phase.phase_name,
            "operation_class": phase.operation_class,
            "required_compute_capability": phase.compute_capability_resource,
            "required_memory_capability": phase.memory_capability_resource,
            "evidence_kind": evidence_kind,
            "execution_domain": execution_domain,
            "candidate": candidate,
            "constraints": constraints,
            "capability_profile_refs": capability_profile_refs,
            "local_duration_ns": local_duration,
            "resource_composition": "max(compute-or-exact,memory-pattern-floor)",
            "standard_uncertainty_ns": standard_uncertainty_ns,
            "raw_samples_ns": samples,
            "raw_samples_by_constraint": {
                "compute_or_exact": samples,
                "memory_pattern": memory_samples,
            },
            "correctness": correctness,
            "timing_quality": timing_quality,
            "lane": lane,
            "run_metadata": metadata,
            "compilation_fingerprint": compilation_fingerprint,
        }
        observation["input_digest"] = content_fingerprint(observation)
        runs_root = Path(artifact_store).resolve() / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        destination = runs_root / run_id
        if destination.exists():
            raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
        path = temporary / "observation/phase-capability.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_json_bytes(observation))
        manifest = {
            "schema": "groundupscale.dev/run-manifest/v1alpha1",
            "bundle_kind": "operator-phase-measurement",
            "run_id": run_id,
            "status": "completed",
            "created_at": datetime.now(UTC).isoformat(),
            "device": "ascend-npu",
            "hardware_cohort": execution_domain["hardware_cohort"],
            "operation": "RMSNorm",
            "phase_id": phase.phase_id,
            "phase_name": phase.phase_name,
            "lane": lane,
            "compilation_fingerprint": compilation_fingerprint,
            "producer_lineage": {"producer": MEASUREMENT_PRODUCER},
            "artifacts": [
                {
                    "role": "operator-phase-capability-observation",
                    "path": "observation/phase-capability.json",
                    "media_type": "application/json",
                    "schema": PHASE_OBSERVATION_SCHEMA,
                    "sha256": _sha256(path),
                    "produced_by": MEASUREMENT_PRODUCER,
                    "inputs": [],
                }
            ],
            "immutability": "writer refuses an existing run_id; digests are authoritative",
        }
        (temporary / "run.manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(temporary, destination)
        return destination


def _phase_observation(source: Path) -> tuple[dict[str, object], dict[str, object]]:
    verification = verify_run_bundle(source)
    if verification.get("passed") is not True:
        raise ValueError(f"source phase Run Bundle failed verification: {source}")
    manifest_path = source / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    matches = [
        item
        for item in artifacts
        if isinstance(item, dict)
        and item.get("role") == "operator-phase-capability-observation"
    ] if isinstance(artifacts, list) else []
    if (
        manifest.get("bundle_kind") != "operator-phase-measurement"
        or manifest.get("status") != "completed"
        or manifest.get("device") != "ascend-npu"
        or len(matches) != 1
    ):
        raise ValueError("invalid source phase Run Bundle identity")
    observation_path = (source / str(matches[0]["path"])).resolve()
    if source not in observation_path.parents or not observation_path.is_file():
        raise ValueError("invalid source phase observation path")
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    return manifest, observation


def _critical_path(
    phases: list[dict[str, object]], durations: dict[str, float]
) -> tuple[list[str], float]:
    by_id = {str(phase["phase_id"]): phase for phase in phases}
    if len(by_id) != len(phases):
        raise ValueError("duplicate phase identity")
    completed: set[str] = set()
    longest: dict[str, float] = {}
    order: list[str] = []
    while len(completed) < len(by_id):
        ready = sorted(
            phase_id
            for phase_id, phase in by_id.items()
            if phase_id not in completed
            and set(phase["predecessor_phase_ids"]) <= completed
        )
        if not ready:
            raise ValueError("operator phase graph is cyclic or unresolved")
        for phase_id in ready:
            predecessors = list(by_id[phase_id]["predecessor_phase_ids"])
            longest[phase_id] = durations[phase_id] + max(
                (longest[str(item)] for item in predecessors), default=0.0
            )
            completed.add(phase_id)
            order.append(phase_id)
    predecessors = {
        str(item)
        for phase in phases
        for item in phase["predecessor_phase_ids"]
    }
    sinks = set(by_id) - predecessors
    if not sinks:
        raise ValueError("operator phase graph has no output")
    return order, max(longest[phase_id] for phase_id in sinks)


class RmsNormOperatorFrontierBundleWriter:
    """Qualify independent phase source Runs and publish the compound Frontier."""

    def run(
        self,
        artifact_store: str | Path,
        *,
        run_id: str,
        operation: CostOperation,
        execution_domain: dict[str, object],
        source_runs: Iterable[str | Path],
        compilation_fingerprint: str,
        supersedes_run: str | Path | None = None,
    ) -> Path:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(f"unsafe run_id: {run_id!r}")
        if operation.operation != "RMSNorm" or operation.phase_graph is None:
            raise ValueError("an explicit RMSNorm Operator Phase Graph is required")
        if not isinstance(compilation_fingerprint, str) or not compilation_fingerprint:
            raise ValueError("cost compilation fingerprint is required")
        _validate_domain(operation, execution_domain)
        graph = operation.phase_graph
        phases = [canonical_data(phase) for phase in graph.phases]
        sources = [Path(path).resolve() for path in source_runs]
        observations: dict[tuple[str, str], tuple[Path, dict[str, object], dict[str, object]]] = {}
        for source in sources:
            manifest, observation = _phase_observation(source)
            key = (str(observation.get("phase_id")), str(observation.get("lane")))
            if key in observations:
                raise ValueError("duplicate phase evidence lane")
            observations[key] = (source, manifest, observation)

        source_records: list[dict[str, object]] = [
            {
                "run_id": manifest["run_id"],
                "lane": observation["lane"],
                "path": None,
                "manifest_sha256": _sha256(source / "run.manifest.json"),
                "phase_id": observation["phase_id"],
                "candidate": observation["candidate"],
                "compilation_fingerprint": observation["compilation_fingerprint"],
            }
            for source, manifest, observation in observations.values()
        ]
        evidence: list[dict[str, object]] = []
        scheduled: list[dict[str, object]] = []
        missing: list[dict[str, object]] = []
        for phase, phase_document in zip(graph.phases, phases, strict=True):
            search = observations.get((phase.phase_id, "search"))
            holdout = observations.get((phase.phase_id, "independent-holdout"))
            if search is None or holdout is None:
                missing.append(
                    {
                        "phase_id": phase.phase_id,
                        "phase_name": phase.phase_name,
                        "operation_class": phase.operation_class,
                        "required_evidence": (
                            "verified search and independent-holdout Run Bundles for a "
                            "semantically matching capability class or exact operation probe"
                        ),
                    }
                )
                scheduled.append(
                    {
                        **phase_document,
                        "status": "unknown",
                        "candidate": None,
                        "constraints": None,
                        "capability_profile_refs": None,
                        "local_duration_ns": None,
                        "standard_uncertainty_ns": None,
                        "resource_composition": "unknown",
                        "evidence_refs": [],
                    }
                )
                continue
            lane_records = []
            for source, manifest, observation in (search, holdout):
                observation_body = {
                    key: value
                    for key, value in observation.items()
                    if key != "input_digest"
                }
                constraints = observation.get("constraints")
                compute_or_exact = (
                    constraints.get("exact_operation_duration_ns")
                    or constraints.get("matching_compute_capability_duration_ns")
                    if isinstance(constraints, dict)
                    else None
                )
                memory_floor = (
                    constraints.get("memory_pattern_floor_ns")
                    if isinstance(constraints, dict)
                    else None
                )
                capability_refs = observation.get("capability_profile_refs")
                if (
                    observation.get("input_digest")
                    != content_fingerprint(observation_body)
                    or observation.get("phase_id") != phase.phase_id
                    or observation.get("phase_name") != phase.phase_name
                    or observation.get("operation_class") != phase.operation_class
                    or observation.get("required_compute_capability")
                    != phase.compute_capability_resource
                    or observation.get("required_memory_capability")
                    != phase.memory_capability_resource
                    or observation.get("execution_domain") != execution_domain
                    or observation.get("correctness") != "passed"
                    or observation.get("timing_quality") != "passed"
                    or not isinstance(compute_or_exact, (int, float))
                    or not isinstance(memory_floor, (int, float))
                    or observation.get("local_duration_ns")
                    != max(float(compute_or_exact), float(memory_floor))
                    or not isinstance(capability_refs, dict)
                    or set(capability_refs) != {"compute_or_exact", "memory_pattern"}
                    or not all(
                        isinstance(reference, str) and reference
                        for reference in capability_refs.values()
                    )
                    or len(set(capability_refs.values())) != 2
                    or observation.get("compilation_fingerprint")
                    != compilation_fingerprint
                    or manifest.get("compilation_fingerprint")
                    != compilation_fingerprint
                    or manifest.get("hardware_cohort")
                    != execution_domain["hardware_cohort"]
                ):
                    raise ValueError("source phase evidence identity or semantics mismatch")
                record = next(
                    item
                    for item in source_records
                    if item["run_id"] == manifest["run_id"]
                )
                lane_records.append(record)
            if search[2]["candidate"] != holdout[2]["candidate"]:
                raise ValueError("search and holdout candidate identity mismatch")
            qualified = canonical_data(holdout[2])
            evidence.append(qualified)
            scheduled.append(
                {
                    **phase_document,
                    "status": "known",
                    "candidate": qualified["candidate"],
                    "constraints": qualified["constraints"],
                    "capability_profile_refs": qualified[
                        "capability_profile_refs"
                    ],
                    "local_duration_ns": qualified["local_duration_ns"],
                    "standard_uncertainty_ns": qualified["standard_uncertainty_ns"],
                    "resource_composition": qualified["resource_composition"],
                    "evidence_refs": [
                        f"run-bundle://{record['run_id']}"
                        "#artifact://observation/phase-capability.json"
                        for record in lane_records
                    ],
                }
            )

        complete = not missing
        durations = {
            str(item["phase_id"]): float(item["local_duration_ns"])
            for item in scheduled
            if item["local_duration_ns"] is not None
        }
        topological_order: list[str] = []
        critical_path_ns: float | None = None
        if complete:
            topological_order, critical_path_ns = _critical_path(phases, durations)
        uncertainty = (
            sqrt(sum(float(item["standard_uncertainty_ns"]) ** 2 for item in scheduled))
            if complete
            else None
        )
        frontier = {
            "status": "known" if complete else "unknown",
            "duration_ns": critical_path_ns,
            "standard_uncertainty_ns": uncertainty,
            "composition_policy": "dependency-critical-path-no-chunk",
            "formula": "max_path(sum(phase.local_duration_ns))",
        }
        schedule = {
            "policy": "dependency-critical-path-no-chunk",
            "chunk_pipeline_contract_id": None,
            "overlap_evidence_refs": [],
            "phases": scheduled,
            "topological_phase_ids": topological_order,
            "serialized_duration_ns": sum(durations.values()) if complete else None,
            "critical_path_duration_ns": critical_path_ns,
            "selected_duration_ns": critical_path_ns,
        }
        graph_document: dict[str, object] = {
            "schema": PHASE_GRAPH_SCHEMA,
            "graph_id": graph.graph_id,
            "operation": operation.operation,
            "cost_node_id": operation.node_id,
            "stable_path": operation.stable_path,
            "execution_domain": execution_domain,
            "compilation_fingerprint": compilation_fingerprint,
            "phases": phases,
            "output_phase_ids": list(graph.output_phase_ids),
        }
        graph_document["input_digest"] = _canonical_digest(graph_document)
        qualification: dict[str, object] = {
            "schema": QUALIFICATION_SCHEMA,
            "status": "qualified" if complete else "unknown",
            "operation": operation.operation,
            "stable_path": operation.stable_path,
            "hardware_cohort": execution_domain["hardware_cohort"],
            "execution_domain": execution_domain,
            "compilation_fingerprint": compilation_fingerprint,
            "phase_graph_ref": "artifact://frontier/phase-graph.json",
            "phase_graph_digest": graph_document["input_digest"],
            "phase_evidence": evidence,
            "source_runs": source_records,
            "source_evidence_digest": _canonical_digest(source_records),
            "selected_candidate": {
                "candidate_id": f"ascend-rmsnorm-explicit-phase-graph-{run_id}",
                "candidate_version": "v1",
                "phase_schedule": schedule,
            },
            "operator_frontier": frontier,
            "missing_evidence": missing,
            "uncertainty_policy": {
                "policy_id": "independent-phase-standard-uncertainty-rss-v1",
                "combination": "root-sum-square",
                "target_coverage": 0.6827,
            },
        }
        qualification["input_digest"] = _canonical_digest(qualification)
        diagnostic: dict[str, object] = {
            "schema": DIAGNOSTIC_SCHEMA,
            "qualification_ref": "artifact://frontier/qualification.json",
            "qualification_digest": qualification["input_digest"],
            "operation": operation.operation,
            "stable_path": operation.stable_path,
            "hardware_cohort": execution_domain["hardware_cohort"],
            "operator_frontier": frontier,
            "missing_evidence": missing,
        }
        diagnostic["input_digest"] = _canonical_digest(diagnostic)

        runs_root = Path(artifact_store).resolve() / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        destination = runs_root / run_id
        if destination.exists():
            raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
        sources_by_run_id = {
            json.loads((source / "run.manifest.json").read_text(encoding="utf-8"))[
                "run_id"
            ]: source
            for source in sources
        }
        for record in source_records:
            record["path"] = os.path.relpath(
                sources_by_run_id[str(record["run_id"])], destination
            )
        supersession: dict[str, object] | None = None
        if supersedes_run is not None:
            superseded = Path(supersedes_run).resolve()
            superseded_manifest = superseded / "run.manifest.json"
            if not superseded_manifest.is_file():
                raise ValueError("superseded Run Bundle manifest is required")
            superseded_identity = json.loads(
                superseded_manifest.read_text(encoding="utf-8")
            )
            if superseded_identity.get("bundle_kind") != "compound-operator-frontier":
                raise ValueError("superseded Run Bundle kind mismatch")
            supersession = {
                "run_id": superseded_identity["run_id"],
                "path": os.path.relpath(superseded, destination),
                "manifest_sha256": _sha256(superseded_manifest),
            }
            qualification["supersedes"] = supersession
        # Source paths changed from placeholders, therefore digests above must bind final records.
        qualification["source_evidence_digest"] = _canonical_digest(source_records)
        qualification["input_digest"] = _canonical_digest(
            {key: value for key, value in qualification.items() if key != "input_digest"}
        )
        diagnostic["qualification_digest"] = qualification["input_digest"]
        diagnostic["input_digest"] = _canonical_digest(
            {key: value for key, value in diagnostic.items() if key != "input_digest"}
        )
        temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
        artifacts: list[dict[str, object]] = []
        for role, relative, document, inputs in (
            ("operator-phase-graph", "frontier/phase-graph.json", graph_document, []),
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
        ):
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
                    "produced_by": FRONTIER_PRODUCER,
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
            "compilation_fingerprint": compilation_fingerprint,
            "source_runs": source_records,
            "supersedes": supersession,
            "producer_lineage": {"producer": FRONTIER_PRODUCER},
            "artifacts": artifacts,
            "immutability": "writer refuses an existing run_id; source and artifact digests are authoritative",
        }
        (temporary / "run.manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(temporary, destination)
        return destination


__all__ = [
    "RmsNormOperatorFrontierBundleWriter",
    "RmsNormPhaseMeasurementBundleWriter",
]
