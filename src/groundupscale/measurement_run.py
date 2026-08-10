"""Immutable Run Bundle orchestration for Measurement Adapters."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

from groundupscale.measurement_contract import MeasurementAdapter
from groundupscale.run_bundle import RunBundleExistsError


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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


def _producer_lineage(adapter: MeasurementAdapter) -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[2]
    adapter_module = sys.modules[adapter.__class__.__module__]
    import groundupscale.run_bundle as run_bundle_module

    source_paths = {
        Path(__file__).resolve(),
        Path(str(adapter_module.__file__)).resolve(),
        Path(str(run_bundle_module.__file__)).resolve(),
    }
    source_paths.update(
        Path(path).resolve()
        for path in getattr(adapter_module, "PRODUCER_SOURCE_PATHS", ())
    )
    digest = sha256()
    source_files: list[dict[str, str]] = []
    for path in sorted(source_paths):
        relative_path = path.relative_to(repository_root).as_posix()
        content = path.read_bytes()
        content_digest = sha256(content).hexdigest()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_digest.encode("ascii"))
        source_files.append(
            {"path": relative_path, "sha256": content_digest}
        )
    try:
        package_version = version("groundupscale")
    except PackageNotFoundError:
        package_version = "0.1.0"
    return {
        "schema": "groundupscale.dev/producer-lineage/v1alpha1",
        "producer": "groundupscale",
        "package_version": package_version,
        "source_sha256": digest.hexdigest(),
        "source_files": source_files,
        "source_state": "content-addressed-working-tree",
    }


class MeasurementRunBundleWriter:
    """Execute the portable five-step seam and publish one atomic bundle."""

    def __init__(self, adapter: MeasurementAdapter) -> None:
        self.adapter = adapter

    def run(
        self,
        artifact_store: str | Path,
        *,
        case: dict[str, object],
        run_id: str,
    ) -> Path:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(f"unsafe run_id: {run_id!r}")
        runs_root = Path(artifact_store).resolve() / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        destination = runs_root / run_id
        if destination.exists():
            raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
        artifacts: list[dict[str, Any]] = []
        producer_lineage = _producer_lineage(self.adapter)
        produced_by = (
            f"groundupscale@{producer_lineage['package_version']}+"
            f"source.{str(producer_lineage['source_sha256'])[:16]}"
        )

        def write_json(
            role: str,
            relative_path: str,
            document: dict[str, object],
            *,
            inputs: tuple[str, ...] = (),
        ) -> None:
            path = temporary / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_json_bytes(document))
            artifacts.append(
                {
                    "role": role,
                    "path": relative_path,
                    "media_type": "application/json",
                    "schema": document["schema"],
                    "sha256": _sha256(path),
                    "produced_by": produced_by,
                    "inputs": list(inputs),
                }
            )

        capabilities = dict(self.adapter.discover_capabilities())
        cohort = dict(self.adapter.fingerprint_cohort())
        preflight = dict(self.adapter.preflight())

        write_json("benchmark-case", "resolved/case.json", case)
        write_json(
            "measurement-capability-manifest",
            "adapter/capabilities.json",
            capabilities,
        )
        write_json("hardware-cohort", "adapter/cohort.json", cohort)
        write_json(
            "measurement-preflight",
            "adapter/preflight.json",
            preflight,
            inputs=("measurement-capability-manifest", "hardware-cohort"),
        )
        if preflight.get("eligible") is not True:
            evidence_refs = [
                capabilities["evidence_ref"],
                cohort["evidence_ref"],
                preflight["evidence_ref"],
            ]
            failure = {
                "schema": "groundupscale.dev/measurement-failure/v1alpha1",
                "status": "blocked",
                "device": "ascend-npu",
                "logical_device": preflight["logical_device"],
                "failed_operation": "preflight",
                "reason_codes": list(preflight.get("reason_codes", [])),
                "evidence_refs": evidence_refs,
            }
            write_json(
                "measurement-failure",
                "adapter/failure.json",
                failure,
                inputs=(
                    "measurement-capability-manifest",
                    "hardware-cohort",
                    "measurement-preflight",
                ),
            )
            blocked_operations = {
                "schema": (
                    "groundupscale.dev/measurement-operation-evidence/"
                    "v1alpha1"
                ),
                "operations": [
                    {
                        "operation": operation,
                        "status": document["status"],
                        "evidence_ref": document["evidence_ref"],
                    }
                    for operation, document in (
                        ("discover_capabilities", capabilities),
                        ("fingerprint_cohort", cohort),
                        ("preflight", preflight),
                    )
                ],
            }
            write_json(
                "measurement-operation-evidence",
                "adapter/operations.json",
                blocked_operations,
                inputs=("measurement-failure",),
            )
            blocked_manifest = {
                "schema": "groundupscale.dev/run-manifest/v1alpha1",
                "bundle_kind": "exact-shape-measurement",
                "run_id": run_id,
                "status": "blocked",
                "created_at": datetime.now(UTC).isoformat(),
                "device": "ascend-npu",
                "hardware_cohort": None,
                "adapter": {
                    "adapter_id": "ascend-npu",
                    "adapter_version": "v1",
                    "protocol_id": "exact-shape-measurement",
                    "protocol_version": "v1",
                },
                "producer_lineage": producer_lineage,
                "artifacts": artifacts,
                "immutability": (
                    "writer refuses an existing run_id; artifact digests are "
                    "authoritative"
                ),
            }
            (temporary / "run.manifest.json").write_bytes(
                _json_bytes(blocked_manifest)
            )
            os.replace(temporary, destination)
            return destination

        timing_plan = dict(self.adapter.build_timing_plan(case))
        collection = dict(self.adapter.collect(case, timing_plan))
        write_json(
            "timing-plan",
            "adapter/timing-plan.json",
            timing_plan,
            inputs=("benchmark-case", "measurement-capability-manifest"),
        )
        write_json(
            "measurement-collection",
            "adapter/collection.json",
            collection,
            inputs=("timing-plan", "measurement-preflight"),
        )
        environment = {
            "schema": "groundupscale.dev/environment/v1alpha1",
            "device": "ascend-npu",
            "logical_device": preflight["logical_device"],
            "software": cohort["software_evidence"],
            "cohort_identity_software": cohort["software"],
            "preflight": preflight,
            "policy": "allowlisted fields only; no credentials or ambient dump",
        }
        write_json(
            "environment",
            "resolved/environment.json",
            environment,
            inputs=("measurement-preflight", "hardware-cohort"),
        )
        component_artifacts = (
            ("candidate-identity", "observation/candidate.json", "candidate_identity"),
            ("input-corpus", "resolved/input-corpus.json", "input_corpus"),
            ("execution-contract", "resolved/execution-contract.json", "execution_contract"),
            (
                "instrumentation-profile",
                "resolved/instrumentation-profile.json",
                "instrumentation_profile",
            ),
            ("correctness-observation", "observation/correctness.json", "correctness"),
            ("raw-timing-observation", "observation/raw-timing.json", "raw_timing"),
            ("memory-observation", "observation/memory.json", "memory"),
            ("completion-boundary", "observation/completion-boundary.json", "completion_boundary"),
        )
        for role, relative_path, key in component_artifacts:
            component = collection[key]
            if not isinstance(component, dict):
                raise ValueError(f"collection has invalid {key}")
            write_json(
                role,
                relative_path,
                component,
                inputs=("measurement-collection",),
            )
        operations = {
            "schema": "groundupscale.dev/measurement-operation-evidence/v1alpha1",
            "operations": [
                {
                    "operation": operation,
                    "evidence_ref": document["evidence_ref"],
                }
                for operation, document in (
                    ("discover_capabilities", capabilities),
                    ("fingerprint_cohort", cohort),
                    ("preflight", preflight),
                    ("build_timing_plan", timing_plan),
                    ("collect", collection),
                )
            ],
        }
        write_json(
            "measurement-operation-evidence",
            "adapter/operations.json",
            operations,
            inputs=(
                "measurement-capability-manifest",
                "hardware-cohort",
                "measurement-preflight",
                "timing-plan",
                "measurement-collection",
            ),
        )
        manifest = {
            "schema": "groundupscale.dev/run-manifest/v1alpha1",
            "bundle_kind": "exact-shape-measurement",
            "run_id": run_id,
            "status": "completed",
            "created_at": datetime.now(UTC).isoformat(),
            "device": "ascend-npu",
            "hardware_cohort": cohort["cohort_id"],
            "adapter": {
                "adapter_id": capabilities["adapter_id"],
                "adapter_version": capabilities["adapter_version"],
                "protocol_id": capabilities["protocol_id"],
                "protocol_version": capabilities["protocol_version"],
            },
            "observation_validity": {
                "status": (
                    "valid"
                    if collection["timing_quality"]["status"] == "passed"
                    else "quarantined"
                ),
                "correctness": collection["correctness"]["status"],
                "completion_boundary": (
                    "closed"
                    if collection["completion_boundary"]["closed"] is True
                    else "open"
                ),
                "raw_timing_sample_count": len(
                    collection["raw_timing"]["samples"]
                ),
                "timing_quality": collection["timing_quality"]["status"],
                "reason_codes": list(
                    collection["timing_quality"]["reason_codes"]
                ),
            },
            "frontier_role": {
                "status": "not-evaluated",
                "reason_code": "issue-28-does-not-promote-frontier",
            },
            "producer_lineage": producer_lineage,
            "artifacts": artifacts,
            "immutability": (
                "writer refuses an existing run_id; artifact digests are authoritative"
            ),
        }
        (temporary / "run.manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(temporary, destination)
        return destination


__all__ = ["MeasurementRunBundleWriter"]
