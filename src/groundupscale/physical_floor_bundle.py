"""Immutable replay bundle joining an Ascend MatMul floor and observation."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from groundupscale.backends.apple_m4_cpu import _scope_matches
from groundupscale.backends.ascend_910b2 import matmul_problem_shape
from groundupscale.ir import canonical_data
from groundupscale.pipeline import CompiledAnalysis
from groundupscale.physical_floor_report import render_physical_floor_report
from groundupscale.run_bundle import (
    RUN_ID_PATTERN,
    RunBundleExistsError,
    verify_run_bundle,
)


COMPARISON_SCHEMA = (
    "groundupscale.dev/physical-floor-observation-comparison/v1alpha1"
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            canonical_data(value), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


class PhysicalFloorComparisonBundleWriter:
    """Replay one verified exact-Shape measurement against the compiled floor."""

    def __init__(self, compiled: CompiledAnalysis) -> None:
        self.compiled = compiled

    def run(
        self,
        artifact_store: str | Path,
        *,
        measurement_bundle: str | Path,
        run_id: str,
    ) -> Path:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(f"unsafe run_id: {run_id!r}")
        source = Path(measurement_bundle).resolve()
        verification = verify_run_bundle(source)
        if not verification["passed"]:
            raise ValueError(
                "source Measurement Run Bundle failed verification: "
                + "; ".join(verification["failures"])
            )
        source_manifest_path = source / "run.manifest.json"
        source_manifest = _read_json(source_manifest_path)
        if (
            source_manifest.get("bundle_kind") != "exact-shape-measurement"
            or source_manifest.get("status") != "completed"
        ):
            raise ValueError("source must be a completed exact-Shape Measurement Run Bundle")
        prediction = self.compiled.hardware_prediction
        if prediction is None or prediction.backend_id != (
            "huawei.ascend.910b2.resource-envelope"
        ):
            raise ValueError("compiled analysis has no Ascend 910B2 hardware prediction")
        if source_manifest.get("hardware_cohort") not in {
            item.hardware_cohort for item in prediction.measured_capabilities
        }:
            raise ValueError("measurement and Hardware Capability Profile cohorts differ")

        source_case = _read_json(source / "resolved/case.json")
        source_cohort = _read_json(source / "adapter/cohort.json")
        correctness = _read_json(source / "observation/correctness.json")
        raw_timing = _read_json(source / "observation/raw-timing.json")
        completion = _read_json(source / "observation/completion-boundary.json")
        candidate_identity = _read_json(source / "observation/candidate.json")
        if (
            source_case.get("operation") != "MatMul"
            or source_case.get("dtype") != "float32"
            or source_case.get("layout") != "row-major-contiguous"
            or correctness.get("status") != "passed"
            or completion.get("closed") is not True
            or candidate_identity.get("cpu_fallback") is not False
            or source_manifest.get("observation_validity", {}).get("status")
            != "valid"
        ):
            raise ValueError("measurement does not satisfy the comparison contract")

        benchmark_cases = [
            case
            for document in self.compiled.bundle.benchmark_cases
            for case in document.spec.cases
        ]
        if len(benchmark_cases) != 1:
            raise ValueError("comparison plan must select exactly one Benchmark Case")
        benchmark_case = benchmark_cases[0]
        scope_bound = next(
            (
                bound
                for bound in prediction.scope_bounds
                if bound.case_id == benchmark_case.id
            ),
            None,
        )
        operations_by_id = {
            operation.node_id: operation
            for operation in self.compiled.cost.cost_ir.walk_operations()
        }
        scoped_candidates = tuple(
            candidate
            for candidate in prediction.candidates
            if _scope_matches(
                operations_by_id[candidate.cost_node_id],
                benchmark_case.scope,
            )
        )
        if scope_bound is None or len(scoped_candidates) != 1:
            raise ValueError("Benchmark Stable Path must resolve to one MatMul candidate")
        candidate = scoped_candidates[0]
        shape = source_case.get("shape")
        if not isinstance(shape, dict):
            raise ValueError("measurement has no exact MatMul Shape")
        left = shape.get("left")
        right = shape.get("right")
        problem_shape = matmul_problem_shape(
            operations_by_id[candidate.cost_node_id]
        )
        if (
            not isinstance(left, list)
            or not isinstance(right, list)
            or problem_shape is None
            or left != [problem_shape[0], problem_shape[1]]
            or right != [problem_shape[1], problem_shape[2]]
        ):
            raise ValueError("Measurement exact Shape does not match Cost IR MatMul")
        if scope_bound.empirical_hardware_floor_ns is None:
            raise ValueError("selected MatMul Physical Floor is unknown")

        quality_statuses = {
            item.quality_status for item in prediction.measured_capabilities
        }
        quality_reasons = sorted(
            {
                reason
                for item in prediction.measured_capabilities
                for reason in item.quality_reason_codes
            }
        )
        if len(quality_statuses) != 1:
            raise ValueError("capability resources have inconsistent quality status")
        floor = scope_bound.empirical_hardware_floor_ns
        observed = float(raw_timing["summary"]["median"])
        comparison = {
            "schema": COMPARISON_SCHEMA,
            "status": "physical-floor-with-observation",
            "case_id": benchmark_case.id,
            "stable_path": benchmark_case.scope,
            "hardware_cohort": source_manifest["hardware_cohort"],
            "theoretical_capability": {
                "fp32_flops_per_second": canonical_data(
                    prediction.capabilities.fp32_flops_per_second
                ),
                "peak_memory_bandwidth_bytes_per_second": canonical_data(
                    prediction.capabilities.peak_memory_bandwidth_bytes_per_second
                ),
            },
            "physical_floor": {
                "status": prediction.status,
                "kind": "algorithm-independent-resource-physical-floor",
                "minimum_work_flops": scope_bound.flops,
                "compulsory_bytes": scope_bound.compulsory_bytes,
                "compute_time_ns": scope_bound.empirical_compute_time_ns,
                "memory_time_ns": scope_bound.empirical_memory_time_ns,
                "resource_physical_floor_ns": floor,
                "limiting_resource": scope_bound.limiting_resource,
                "full_duration_ns": None,
                "formula": scope_bound.formula,
                "assumptions": list(scope_bound.assumptions),
                "quality": {
                    "status": next(iter(quality_statuses)),
                    "reason_codes": quality_reasons,
                },
                "capabilities": canonical_data(prediction.measured_capabilities),
            },
            "operator_frontier": {
                "status": "unknown",
                "value_ns": None,
                "reason_code": "not-qualified-by-issue-29",
            },
            "observation": {
                "status": "known",
                "quality": source_manifest["observation_validity"]["status"],
                "median_ns": observed,
                "q1_ns": raw_timing["summary"]["q1"],
                "q3_ns": raw_timing["summary"]["q3"],
                "iqr_over_median": raw_timing["summary"][
                    "iqr_fraction_of_median"
                ],
                "timer_source": raw_timing["timer_source"],
                "timer_resolution_ns": raw_timing["timer_resolution_ns"],
                "completion_boundary": "closed",
                "candidate": candidate_identity["candidate_id"],
                "source_run_id": source_manifest["run_id"],
                "source_manifest_sha256": _sha256(source_manifest_path),
            },
            "comparison": {
                "observation_minus_physical_floor_ns": observed - floor,
                "observed_to_physical_floor_ratio": observed / floor,
                "relative_prediction_error": None,
                "error_status": (
                    "not-evaluable-physical-floor-is-not-a-duration-prediction"
                ),
                "interpretation": "optimization-headroom-not-prediction-error",
            },
            "unsupported_regions": {
                "count": len(prediction.unsupported_regions),
                "status": "partial-unknown",
                "regions": canonical_data(prediction.unsupported_regions),
            },
        }
        explanation = {
            "schema": "groundupscale.dev/explanation-graph/v1alpha1",
            "entrypoints": {
                benchmark_case.scope: [
                    "metric:resource-physical-floor",
                    "metric:observation",
                    "comparison:headroom",
                ]
            },
            "nodes": [
                {
                    "id": "scope:matmul",
                    "kind": "stable-path",
                    "stable_path": benchmark_case.scope,
                },
                {
                    "id": "metric:minimum-work",
                    "kind": "resource-demand",
                    "value": scope_bound.flops,
                    "unit": "FLOP",
                },
                {
                    "id": "metric:compulsory-bytes",
                    "kind": "resource-demand",
                    "value": scope_bound.compulsory_bytes,
                    "unit": "B",
                },
                {
                    "id": "metric:resource-physical-floor",
                    "kind": "resource-physical-floor",
                    "value_ns": floor,
                    "full_duration_ns": None,
                    "quality": comparison["physical_floor"]["quality"],
                    "hardware_cohort": comparison["hardware_cohort"],
                    "assumptions": comparison["physical_floor"]["assumptions"],
                    "capabilities": comparison["physical_floor"]["capabilities"],
                },
                {
                    "id": "metric:observation",
                    "kind": "observation",
                    "value_ns": observed,
                    "completion_boundary": "closed",
                    "source_run_id": source_manifest["run_id"],
                    "hardware_cohort": comparison["hardware_cohort"],
                },
                {
                    "id": "comparison:headroom",
                    "kind": "optimization-headroom",
                    **comparison["comparison"],
                },
                {
                    "id": "summary:unsupported-regions",
                    "kind": "partial-unknown",
                    "count": len(prediction.unsupported_regions),
                },
            ],
            "edges": [
                {"source": "scope:matmul", "target": "metric:minimum-work"},
                {"source": "scope:matmul", "target": "metric:compulsory-bytes"},
                {"source": "metric:minimum-work", "target": "metric:resource-physical-floor"},
                {"source": "metric:compulsory-bytes", "target": "metric:resource-physical-floor"},
                {"source": "metric:resource-physical-floor", "target": "comparison:headroom"},
                {"source": "metric:observation", "target": "comparison:headroom"},
            ],
        }

        runs_root = Path(artifact_store).resolve() / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        destination = runs_root / run_id
        if destination.exists():
            raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
        artifacts: list[dict[str, object]] = []

        def write_bytes(
            role: str,
            relative_path: str,
            payload: bytes,
            *,
            schema: str,
            media_type: str = "application/json",
            inputs: tuple[str, ...] = (),
        ) -> None:
            path = temporary / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            artifacts.append(
                {
                    "role": role,
                    "path": relative_path,
                    "media_type": media_type,
                    "schema": schema,
                    "sha256": _sha256(path),
                    "produced_by": "groundupscale@0.1.0",
                    "inputs": list(inputs),
                }
            )

        def write_json(
            role: str,
            relative_path: str,
            value: object,
            schema: str,
            inputs: tuple[str, ...] = (),
        ) -> None:
            write_bytes(
                role,
                relative_path,
                _json_bytes(value),
                schema=schema,
                inputs=inputs,
            )

        bundle = self.compiled.bundle
        inputs_lock = {
            "schema": "groundupscale.dev/resolved-input-lock/v1alpha1",
            "sources": bundle.sources,
            "documents": {
                "analysis_plan": bundle.plan,
                "workload": bundle.workload,
                "analysis_case": bundle.analysis_case,
                "deployment_intent": bundle.deployment_intent,
                "hardware": bundle.hardware,
                "hardware_capability_profiles": bundle.hardware_capability_profiles,
                "fabric_graph": bundle.fabric_graph,
                "benchmark_cases": bundle.benchmark_cases,
                "models": bundle.models,
            },
        }
        write_json("resolved-input-lock", "resolved/inputs.lock.json", inputs_lock, inputs_lock["schema"])
        write_json("cost-ir", "ir/cost.ir.json", self.compiled.cost.cost_ir, self.compiled.cost.cost_ir.schema, ("resolved-input-lock",))
        write_json("hardware-backend-prediction", "prediction/hardware-backend.json", prediction, prediction.schema, ("cost-ir", "resolved-input-lock"))
        for role, relative_path, document in (
            ("source-measurement-manifest", "source/run.manifest.json", source_manifest),
            ("source-benchmark-case", "source/case.json", source_case),
            ("source-hardware-cohort", "source/cohort.json", source_cohort),
            ("source-correctness-observation", "source/correctness.json", correctness),
            ("source-raw-timing-observation", "source/raw-timing.json", raw_timing),
            ("source-completion-boundary", "source/completion-boundary.json", completion),
            ("source-candidate-identity", "source/candidate.json", candidate_identity),
        ):
            write_json(role, relative_path, document, str(document["schema"]))
        write_json("physical-floor-observation-comparison", "comparison/physical-floor-vs-observation.json", comparison, comparison["schema"], ("hardware-backend-prediction", "source-raw-timing-observation", "source-completion-boundary"))
        write_json("explanation-graph", "prediction/explanation.graph.json", explanation, explanation["schema"], ("physical-floor-observation-comparison",))
        report = render_physical_floor_report(comparison).encode("utf-8")
        write_bytes("html-report", "reports/report.html", report, schema="groundupscale.dev/html-report/v1alpha1", media_type="text/html", inputs=("explanation-graph", "physical-floor-observation-comparison"))
        manifest = {
            "schema": "groundupscale.dev/run-manifest/v1alpha1",
            "bundle_kind": "physical-floor-observation-comparison",
            "run_id": run_id,
            "status": "completed",
            "created_at": datetime.now(UTC).isoformat(),
            "device": "ascend-npu",
            "hardware_cohort": source_manifest["hardware_cohort"],
            "stable_path": benchmark_case.scope,
            "source_measurement": {
                "run_id": source_manifest["run_id"],
                "manifest_sha256": _sha256(source_manifest_path),
                "verification": "passed",
            },
            "stages": {
                "resource_physical_floor": prediction.status,
                "full_implementation_duration": "unknown",
                "observation_comparison": "completed",
                "unsupported_regions": "partial-unknown",
            },
            "artifacts": artifacts,
            "immutability": (
                "writer refuses an existing run_id; artifact digests are authoritative"
            ),
        }
        (temporary / "run.manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(temporary, destination)
        return destination


__all__ = ["PhysicalFloorComparisonBundleWriter"]
