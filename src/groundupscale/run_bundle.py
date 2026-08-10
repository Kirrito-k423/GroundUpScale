"""Atomic, immutable, self-describing local Run Bundle writer."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import tempfile
from typing import Any

import torch

from groundupscale.benchmark import (
    BenchmarkRunner,
    ReferenceRunner,
    TraceRunner,
    build_prediction_observation_comparison,
    observe_tensor_storage_peak,
)
from groundupscale.benchmark.explanation import (
    build_explanation_graph,
    render_report_html,
)
from groundupscale.benchmark.prediction import predict_live_set
from groundupscale.ir import canonical_data
from groundupscale.measurement_contract import COHORT_IDENTITY_DIMENSIONS
from groundupscale.pipeline import CompiledAnalysis
from groundupscale.physical_floor_report import render_physical_floor_report


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

EXACT_SHAPE_MEASUREMENT_REQUIRED_ROLES = frozenset(
    {
        "benchmark-case",
        "measurement-capability-manifest",
        "hardware-cohort",
        "measurement-preflight",
        "timing-plan",
        "measurement-collection",
        "environment",
        "candidate-identity",
        "input-corpus",
        "execution-contract",
        "instrumentation-profile",
        "correctness-observation",
        "raw-timing-observation",
        "memory-observation",
        "completion-boundary",
        "measurement-operation-evidence",
    }
)

EXACT_SHAPE_MEASUREMENT_BLOCKED_REQUIRED_ROLES = frozenset(
    {
        "benchmark-case",
        "measurement-capability-manifest",
        "hardware-cohort",
        "measurement-preflight",
        "measurement-failure",
        "measurement-operation-evidence",
    }
)

PHYSICAL_FLOOR_COMPARISON_REQUIRED_ROLES = frozenset(
    {
        "resolved-input-lock",
        "cost-ir",
        "hardware-backend-prediction",
        "source-measurement-manifest",
        "source-benchmark-case",
        "source-hardware-cohort",
        "source-correctness-observation",
        "source-raw-timing-observation",
        "source-completion-boundary",
        "source-candidate-identity",
        "physical-floor-observation-comparison",
        "explanation-graph",
        "html-report",
    }
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(canonical_data(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _json_line_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            canonical_data(value),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_path_matches_scope(candidate_path: object, scope: object) -> bool:
    if not isinstance(candidate_path, str) or not isinstance(scope, str):
        return False
    if not scope.startswith("model/"):
        return False
    parts = scope.split("/", 2)
    if len(parts) != 3:
        return False
    marker = f"/model/{parts[2]}"
    normalized = candidate_path.removeprefix("cost/")
    return normalized.endswith(marker) or marker + "/" in normalized


def _has_fields(document: object, expected: dict[str, object]) -> bool:
    return isinstance(document, dict) and all(
        document.get(key) == value for key, value in expected.items()
    )


def _linear_percentile(samples: list[int], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _measurement_timing_summary(
    samples: list[int],
) -> dict[str, float | int]:
    median = statistics.median(samples)
    q1 = _linear_percentile(samples, 0.25)
    q3 = _linear_percentile(samples, 0.75)
    median_absolute_deviation = statistics.median(
        abs(sample - median) for sample in samples
    )
    return {
        "count": len(samples),
        "minimum": min(samples),
        "p10": _linear_percentile(samples, 0.10),
        "q1": q1,
        "median": median,
        "q3": q3,
        "p90": _linear_percentile(samples, 0.90),
        "maximum": max(samples),
        "iqr": q3 - q1,
        "iqr_fraction_of_median": (q3 - q1) / median,
        "median_absolute_deviation": median_absolute_deviation,
        "mad_fraction_of_median": median_absolute_deviation / median,
    }


def _measurement_timing_quality(
    summary: dict[str, float | int],
    *,
    timer_resolution_ns: float,
) -> dict[str, object]:
    timer_resolution_fraction = timer_resolution_ns / float(summary["median"])
    reason_codes: list[str] = []
    if float(summary["iqr_fraction_of_median"]) > 0.10:
        reason_codes.append("session-dispersion-exceeds-policy")
    if timer_resolution_fraction > 0.01:
        reason_codes.append("timer-resolution-exceeds-policy")
    return {
        "schema": "groundupscale.dev/timing-quality/v1alpha1",
        "policy_id": "issue28-session-dispersion-v1",
        "status": "passed" if not reason_codes else "quarantined",
        "observed_iqr_fraction_of_median": summary[
            "iqr_fraction_of_median"
        ],
        "maximum_iqr_fraction_of_median": 0.10,
        "timer_resolution_ns": timer_resolution_ns,
        "timer_resolution_fraction_of_median": timer_resolution_fraction,
        "maximum_timer_resolution_fraction_of_median": 0.01,
        "excluded_samples": 0,
        "reason_codes": reason_codes,
    }


def _default_run_id(device: str, fingerprint: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{device}-{fingerprint[:8]}"


class RunBundleExistsError(FileExistsError):
    pass


class EnvironmentValidityError(RuntimeError):
    pass


class RunBundleWriter:
    def __init__(self, compiled: CompiledAnalysis, seed: int = 20260806) -> None:
        self.compiled = compiled
        self.seed = seed

    def run(
        self,
        artifact_store: str | Path,
        *,
        run_id: str | None = None,
        samples_override: int | None = None,
        warmup_override: int | None = None,
        windows_per_sample: int = 5,
        target_window_ns: int = 20_000_000,
        environment_validity: dict[str, Any] | None = None,
        require_valid_environment: bool = False,
    ) -> Path:
        benchmark_runner = BenchmarkRunner(self.compiled.bundle, seed=self.seed)
        device = benchmark_runner.device
        selected_run_id = run_id or _default_run_id(
            device, self.compiled.cost.compilation_fingerprint
        )
        if not RUN_ID_PATTERN.fullmatch(selected_run_id):
            raise ValueError(f"unsafe run_id: {selected_run_id!r}")
        if environment_validity is not None and environment_validity.get(
            "schema"
        ) != "groundupscale.dev/environment-validity/v1alpha1":
            raise EnvironmentValidityError(
                "environment validity report has an unsupported schema"
            )
        if require_valid_environment and not (
            environment_validity is not None
            and environment_validity.get("eligible") is True
        ):
            reason_codes = (
                environment_validity.get("reason_codes", ["preflight-not-supplied"])
                if environment_validity is not None
                else ["preflight-not-supplied"]
            )
            raise EnvironmentValidityError(
                "trusted measurement environment is ineligible: "
                + ", ".join(str(reason) for reason in reason_codes)
            )
        if environment_validity is None:
            preflight_status = "not-required"
            preflight_artifact: dict[str, Any] = {
                "schema": "groundupscale.dev/environment-validity/v1alpha1",
                "eligible": None,
                "status": "not-collected",
                "reason_codes": ["preflight-not-requested"],
            }
        else:
            preflight_status = (
                "passed"
                if environment_validity.get("eligible") is True
                else "failed-not-required"
            )
            preflight_artifact = environment_validity
        environment_policy_id = "unverified"
        if preflight_status == "passed":
            policy_metadata = preflight_artifact.get("policy")
            candidate_policy_id = (
                policy_metadata.get("policy_id")
                if isinstance(policy_metadata, dict)
                else None
            )
            if not isinstance(candidate_policy_id, str) or not RUN_ID_PATTERN.fullmatch(
                candidate_policy_id
            ):
                raise EnvironmentValidityError(
                    "eligible environment report has no valid policy_id"
                )
            environment_policy_id = candidate_policy_id
        runs_root = Path(artifact_store).resolve() / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        destination = runs_root / selected_run_id
        if destination.exists():
            raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{selected_run_id}.", dir=runs_root)
        )
        artifacts: list[dict[str, Any]] = []

        def write_bytes(
            role: str,
            relative: str,
            payload: bytes,
            *,
            media_type: str,
            schema: str,
            inputs: tuple[str, ...] = (),
        ) -> None:
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            artifacts.append(
                {
                    "role": role,
                    "path": relative,
                    "media_type": media_type,
                    "schema": schema,
                    "sha256": _sha256(path),
                    "produced_by": "groundupscale@0.1.0",
                    "inputs": list(inputs),
                }
            )

        def write_json(
            role: str,
            relative: str,
            value: Any,
            schema: str,
            inputs: tuple[str, ...] = (),
        ) -> None:
            write_bytes(
                role,
                relative,
                _json_bytes(value),
                media_type="application/json",
                schema=schema,
                inputs=inputs,
            )

        try:
            benchmark = benchmark_runner.run(
                samples_override=samples_override,
                warmup_override=warmup_override,
                windows_per_sample=windows_per_sample,
                target_window_ns=target_window_ns,
            )
            trace = TraceRunner(
                self.compiled.bundle,
                self.compiled.semantic.semantic_ir,
                seed=self.seed,
            ).run()
            memory_model, memory_input = benchmark_runner._model_and_input()
            tensor_storage_memory = observe_tensor_storage_peak(
                memory_model, (memory_input,), device=device
            )
            live_set = predict_live_set(
                self.compiled.semantic.semantic_ir, self.compiled.cost.cost_ir
            )
            comparison = build_prediction_observation_comparison(
                hardware_prediction=self.compiled.hardware_prediction,
                benchmark=benchmark,
                live_set=live_set,
                tensor_storage_observation=tensor_storage_memory,
            )
            explanation = build_explanation_graph(
                self.compiled.cost.cost_ir,
                benchmark,
                trace,
                live_set,
                self.compiled.hardware_prediction,
                comparison,
            )
            reference_runner = ReferenceRunner.from_analysis_bundle(
                self.compiled.bundle, seed=self.seed
            )
            if device == "mps":
                correctness_result = reference_runner.compare_cpu_mps(
                    atol=1e-4, rtol=1e-3
                )
                correctness = {
                    "schema": "groundupscale.dev/correctness-observation/v1alpha1",
                    "passed": correctness_result.passed,
                    "max_absolute_error": correctness_result.max_absolute_error,
                    "max_relative_error": correctness_result.max_relative_error,
                    "atol": correctness_result.atol,
                    "rtol": correctness_result.rtol,
                    "cpu_output_sha256": correctness_result.cpu.output_sha256,
                    "target_output_sha256": correctness_result.mps.output_sha256,
                    "target_audit": canonical_data(correctness_result.mps.audit),
                }
            else:
                target_run = reference_runner.run_device("cpu")
                correctness = {
                    "schema": "groundupscale.dev/correctness-observation/v1alpha1",
                    "passed": True,
                    "target_output_sha256": target_run.output_sha256,
                    "target_audit": canonical_data(target_run.audit),
                }

            bundle = self.compiled.bundle
            resolved_documents = {
                "analysis_plan": bundle.plan,
                "workload": bundle.workload,
                "analysis_case": bundle.analysis_case,
                "deployment_intent": bundle.deployment_intent,
                "hardware": bundle.hardware,
                "hardware_capability_profiles": (
                    bundle.hardware_capability_profiles
                ),
                "fabric_graph": bundle.fabric_graph,
                "benchmark_cases": bundle.benchmark_cases,
                "models": bundle.models,
            }
            inputs_lock = {
                "schema": "groundupscale.dev/resolved-input-lock/v1alpha1",
                "sources": bundle.sources,
                "documents": resolved_documents,
            }
            environment = {
                "schema": "groundupscale.dev/environment/v1alpha1",
                "captured_at": datetime.now(UTC).isoformat(),
                "device": device,
                "python": platform.python_version(),
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "machine": platform.machine(),
                },
                "torch": {
                    "version": torch.__version__,
                    "num_threads": torch.get_num_threads(),
                    "num_interop_threads": torch.get_num_interop_threads(),
                    "mps_built": bool(torch.backends.mps.is_built()),
                    "mps_available": bool(torch.backends.mps.is_available()),
                },
                "measurement_preflight": preflight_artifact,
                "policy": "allowlisted fields only; no unrestricted environment dump",
            }
            model_payload: Any = (
                self.compiled.models[0]
                if len(self.compiled.models) == 1
                else {"models": self.compiled.models}
            )
            write_json("resolved-input-lock", "resolved/inputs.lock.json", inputs_lock, inputs_lock["schema"])
            write_json("environment", "resolved/environment.json", environment, environment["schema"])
            write_json("model-ir", "ir/model.ir.json", model_payload, "groundupscale.dev/model-ir/v1alpha1", ("resolved-input-lock",))
            write_json("workload-ir", "ir/workload.ir.json", self.compiled.workload, "groundupscale.dev/workload-ir/v1alpha1", ("resolved-input-lock",))
            write_json("semantic-ir", "ir/semantic.ir.json", self.compiled.semantic.semantic_ir, self.compiled.semantic.semantic_ir.schema, ("model-ir", "workload-ir"))
            write_json("cost-ir", "ir/cost.ir.json", self.compiled.cost.cost_ir, self.compiled.cost.cost_ir.schema, ("semantic-ir",))
            if self.compiled.hardware_prediction is not None:
                write_json(
                    "hardware-backend-prediction",
                    "prediction/hardware-backend.json",
                    self.compiled.hardware_prediction,
                    self.compiled.hardware_prediction.schema,
                    ("cost-ir", "resolved-input-lock"),
                )
            duration_status = (
                self.compiled.hardware_prediction.status
                if self.compiled.hardware_prediction is not None
                else "uncalibrated"
            )
            prediction = {
                "schema": "groundupscale.dev/prediction/v1alpha1",
                "cost_summary": self.compiled.cost.cost_ir.summary,
                "live_set": live_set,
                "duration_status": duration_status,
                "duration": (
                    self.compiled.hardware_prediction.program_bounds
                    if self.compiled.hardware_prediction is not None
                    else None
                ),
                "hardware_backend": (
                    {
                        "backend_id": self.compiled.hardware_prediction.backend_id,
                        "backend_version": self.compiled.hardware_prediction.backend_version,
                        "prediction_complete": (
                            self.compiled.hardware_prediction.prediction_complete
                        ),
                        "artifact": "prediction/hardware-backend.json",
                    }
                    if self.compiled.hardware_prediction is not None
                    else None
                ),
            }
            write_json("prediction", "prediction/metrics.json", prediction, prediction["schema"], ("cost-ir",))
            write_json("explanation-graph", "prediction/explanation.graph.json", explanation, explanation["schema"], ("prediction", "prediction-observation-comparison", "benchmark-observation", "alignment-map"))
            write_json("benchmark-observation", "observation/raw/benchmark.json", benchmark, benchmark["schema"], ("resolved-input-lock", "environment"))
            trace_lines = b"".join(
                _json_line_bytes(event) for event in trace["events"]
            )
            write_bytes("observation-trace", "observation/observation.trace.jsonl", trace_lines, media_type="application/x-ndjson", schema="groundupscale.dev/observation-span/v1alpha1", inputs=("resolved-input-lock", "environment"))
            write_json("alignment-map", "observation/alignment.map.json", trace["alignment_map"], trace["alignment_map"]["schema"], ("semantic-ir", "observation-trace"))
            memory_observation = {
                "schema": "groundupscale.dev/memory-observation/v1alpha1",
                "framework_tensor_storage": tensor_storage_memory,
                "runtime_point_samples": trace["memory_observation"],
                "authoritative_gate_metric": "framework_tensor_storage.peak_framework_tensor_bytes",
            }
            write_json("memory-observation", "observation/memory.json", memory_observation, memory_observation["schema"], ("observation-trace", "semantic-ir"))
            write_json(
                "prediction-observation-comparison",
                "comparison/predicted-vs-observed.json",
                comparison,
                comparison["schema"],
                (
                    "prediction",
                    *(
                        ("hardware-backend-prediction",)
                        if self.compiled.hardware_prediction is not None
                        else ()
                    ),
                    "benchmark-observation",
                    "memory-observation",
                ),
            )
            write_json("correctness-observation", "observation/correctness.json", correctness, correctness["schema"], ("resolved-input-lock", "environment"))
            write_json("error-attribution", "comparison/error-attribution.json", trace["error_attribution"], trace["error_attribution"]["schema"], ("benchmark-observation", "alignment-map"))
            report = render_report_html(
                run_id=selected_run_id,
                device=device,
                benchmark=benchmark,
                trace=trace,
                live_set=live_set,
                explanation=explanation,
                comparison=comparison,
            )
            write_bytes("html-report", "reports/report.html", report.encode("utf-8"), media_type="text/html", schema="groundupscale.dev/html-report/v1alpha1", inputs=("explanation-graph", "prediction-observation-comparison"))

            hardware_names = "-".join(
                document.metadata.name for document in bundle.hardware
            )
            manifest = {
                "schema": "groundupscale.dev/run-manifest/v1alpha1",
                "run_id": selected_run_id,
                "status": "completed",
                "created_at": datetime.now(UTC).isoformat(),
                "compilation_fingerprint": self.compiled.semantic.compilation_fingerprint,
                "cost_compilation_fingerprint": self.compiled.cost.compilation_fingerprint,
                "hardware_compilation_fingerprint": (
                    self.compiled.hardware_prediction.compilation_fingerprint
                    if self.compiled.hardware_prediction is not None
                    else None
                ),
                "hardware_cohort": (
                    f"{hardware_names}-{platform.release()}-torch{torch.__version__}-"
                    f"{device}-env-{environment_policy_id}"
                ),
                "device": device,
                "environment_validity": preflight_status,
                "seed": self.seed,
                "stages": {
                    "compilation": "completed",
                    "structural_prediction": "completed",
                    "duration_prediction": duration_status,
                    "prediction_observation_comparison": "completed",
                    "benchmark": "completed",
                    "trace": "completed",
                    "calibration": "skipped-not-requested",
                },
                "artifacts": artifacts,
                "immutability": "writer refuses an existing run_id; artifact digests are authoritative",
            }
            (temporary / "run.manifest.json").write_bytes(_json_bytes(manifest))
            os.replace(temporary, destination)
            return destination
        except Exception:
            # A failed temporary bundle is intentionally not published as completed evidence.
            # Its path is preserved for diagnosis and cannot collide with a future Run ID.
            failure = {
                "schema": "groundupscale.dev/run-failure/v1alpha1",
                "run_id": selected_run_id,
                "status": "failed-before-publication",
                "captured_at": datetime.now(UTC).isoformat(),
            }
            (temporary / "failure.json").write_bytes(_json_bytes(failure))
            raise


def verify_run_bundle(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    manifest_path = root / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        artifacts = []
        failures.append("manifest artifacts must be a list")
    exact_shape = manifest.get("bundle_kind") == "exact-shape-measurement"
    floor_comparison = (
        manifest.get("bundle_kind") == "physical-floor-observation-comparison"
    )
    structured_bundle = exact_shape or floor_comparison
    completed_measurement = exact_shape and manifest.get("status") == "completed"
    role_counts: dict[object, int] = {}
    if structured_bundle:
        for artifact in artifacts:
            if isinstance(artifact, dict):
                role = artifact.get("role")
                role_counts[role] = role_counts.get(role, 0) + 1
        if floor_comparison:
            required_roles = PHYSICAL_FLOOR_COMPARISON_REQUIRED_ROLES
        else:
            required_roles = (
                EXACT_SHAPE_MEASUREMENT_REQUIRED_ROLES
                if manifest.get("status") == "completed"
                else EXACT_SHAPE_MEASUREMENT_BLOCKED_REQUIRED_ROLES
            )
        for role, count in sorted(
            (str(role), count)
            for role, count in role_counts.items()
            if role in required_roles and count > 1
        ):
            failures.append(f"duplicate artifact role: {role}")
        present_roles = {
            role for role in role_counts if isinstance(role, str)
        }
        for role in sorted(required_roles - present_roles):
            failures.append(f"missing required artifact role: {role}")

    documents_by_role: dict[str, dict[str, object]] = {}
    paths_by_role: dict[str, str] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            failures.append("invalid artifact entry")
            continue
        artifact_path = (root / artifact["path"]).resolve()
        if root not in artifact_path.parents:
            failures.append(f"path escapes bundle: {artifact['path']}")
        elif not artifact_path.is_file():
            failures.append(f"missing artifact: {artifact['path']}")
        elif _sha256(artifact_path) != artifact["sha256"]:
            failures.append(f"digest mismatch: {artifact['path']}")
        if (
            structured_bundle
            and artifact_path.is_file()
            and artifact.get("media_type") == "application/json"
        ):
            try:
                artifact_document = json.loads(
                    artifact_path.read_text(encoding="utf-8")
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                failures.append(f"invalid JSON artifact: {artifact['path']}")
            else:
                if not isinstance(artifact_document, dict):
                    failures.append(f"invalid JSON artifact: {artifact['path']}")
                    continue
                if artifact_document.get("schema") != artifact.get("schema"):
                    failures.append(f"schema mismatch: {artifact['path']}")
                role = artifact.get("role")
                if (
                    isinstance(role, str)
                    and role_counts.get(role) == 1
                ):
                    documents_by_role[role] = artifact_document
                    paths_by_role[role] = str(artifact["path"])

    if floor_comparison:
        comparison = documents_by_role.get(
            "physical-floor-observation-comparison"
        )
        source_manifest = documents_by_role.get("source-measurement-manifest")
        source_cohort = documents_by_role.get("source-hardware-cohort")
        correctness = documents_by_role.get("source-correctness-observation")
        raw_timing = documents_by_role.get("source-raw-timing-observation")
        completion = documents_by_role.get("source-completion-boundary")
        source_candidate = documents_by_role.get("source-candidate-identity")
        hardware_prediction = documents_by_role.get(
            "hardware-backend-prediction"
        )
        explanation = documents_by_role.get("explanation-graph")
        if comparison is not None:
            if comparison.get("hardware_cohort") != manifest.get(
                "hardware_cohort"
            ):
                failures.append("comparison hardware cohort mismatch")
            if comparison.get("stable_path") != manifest.get("stable_path"):
                failures.append("comparison Stable Path mismatch")
            physical_floor = comparison.get("physical_floor")
            theoretical = comparison.get("theoretical_capability")
            operator_frontier = comparison.get("operator_frontier")
            comparison_result = comparison.get("comparison")
            observation = comparison.get("observation")
            if (
                not isinstance(physical_floor, dict)
                or physical_floor.get("resource_physical_floor_ns") is None
                or physical_floor.get("full_duration_ns") is not None
            ):
                failures.append("invalid Resource Physical Floor semantics")
            if (
                not isinstance(theoretical, dict)
                or not all(
                    isinstance(value, dict) and value.get("status") == "unknown"
                    for value in theoretical.values()
                )
                or not isinstance(operator_frontier, dict)
                or operator_frontier.get("status") != "unknown"
                or operator_frontier.get("value_ns") is not None
            ):
                failures.append("non-overwriting result layers mismatch")
            if (
                not isinstance(comparison_result, dict)
                or comparison_result.get("relative_prediction_error") is not None
                or comparison_result.get("interpretation")
                != "optimization-headroom-not-prediction-error"
            ):
                failures.append("invalid Physical Floor comparison semantics")
            if isinstance(observation, dict) and isinstance(raw_timing, dict):
                summary = raw_timing.get("summary")
                if (
                    not isinstance(summary, dict)
                    or observation.get("median_ns") != summary.get("median")
                    or observation.get("completion_boundary") != "closed"
                ):
                    failures.append("comparison observation mismatch")
        if source_manifest is not None:
            if (
                source_manifest.get("bundle_kind") != "exact-shape-measurement"
                or source_manifest.get("status") != "completed"
                or source_manifest.get("hardware_cohort")
                != manifest.get("hardware_cohort")
            ):
                failures.append("source measurement manifest mismatch")
            source_metadata = manifest.get("source_measurement")
            source_artifact = next(
                (
                    artifact
                    for artifact in artifacts
                    if isinstance(artifact, dict)
                    and artifact.get("role") == "source-measurement-manifest"
                ),
                None,
            )
            if (
                not isinstance(source_metadata, dict)
                or not isinstance(source_artifact, dict)
                or source_metadata.get("run_id") != source_manifest.get("run_id")
                or source_metadata.get("manifest_sha256")
                != source_artifact.get("sha256")
            ):
                failures.append("source measurement digest mismatch")
            source_artifacts = source_manifest.get("artifacts")
            copied_source_roles = {
                "source-benchmark-case": "benchmark-case",
                "source-hardware-cohort": "hardware-cohort",
                "source-correctness-observation": "correctness-observation",
                "source-raw-timing-observation": "raw-timing-observation",
                "source-completion-boundary": "completion-boundary",
                "source-candidate-identity": "candidate-identity",
            }
            if not isinstance(source_artifacts, list):
                failures.append("source measurement artifacts must be a list")
            else:
                for copied_role, source_role in copied_source_roles.items():
                    copied_entries = [
                        artifact
                        for artifact in artifacts
                        if isinstance(artifact, dict)
                        and artifact.get("role") == copied_role
                    ]
                    source_entries = [
                        artifact
                        for artifact in source_artifacts
                        if isinstance(artifact, dict)
                        and artifact.get("role") == source_role
                    ]
                    if len(copied_entries) != 1 or len(source_entries) != 1:
                        failures.append(
                            f"source artifact role mismatch: {source_role}"
                        )
                        continue
                    copied_entry = copied_entries[0]
                    source_entry = source_entries[0]
                    if (
                        copied_entry.get("sha256") != source_entry.get("sha256")
                        or copied_entry.get("schema") != source_entry.get("schema")
                    ):
                        failures.append(
                            f"source artifact digest mismatch: {source_role}"
                        )
        if (
            source_cohort is not None
            and source_cohort.get("cohort_id") != manifest.get("hardware_cohort")
        ):
            failures.append("source hardware cohort mismatch")
        if correctness is not None and correctness.get("status") != "passed":
            failures.append("source correctness mismatch")
        if completion is not None and completion.get("closed") is not True:
            failures.append("source Completion Boundary mismatch")
        if (
            source_candidate is not None
            and source_candidate.get("cpu_fallback") is not False
        ):
            failures.append("source candidate identity mismatch")
        if hardware_prediction is not None:
            measured = hardware_prediction.get("measured_capabilities")
            if (
                not isinstance(measured, list)
                or not measured
                or any(
                    not isinstance(item, dict)
                    or item.get("hardware_cohort")
                    != manifest.get("hardware_cohort")
                    for item in measured
                )
            ):
                failures.append("hardware capability cohort mismatch")
        if explanation is not None:
            entrypoints = explanation.get("entrypoints")
            if (
                not isinstance(entrypoints, dict)
                or manifest.get("stable_path") not in entrypoints
            ):
                failures.append("Explanation Graph Stable Path mismatch")
        if all(
            document is not None
            for document in (
                comparison,
                source_manifest,
                raw_timing,
                source_candidate,
                hardware_prediction,
                explanation,
            )
        ):
            try:
                assert comparison is not None
                assert source_manifest is not None
                assert raw_timing is not None
                assert source_candidate is not None
                assert hardware_prediction is not None
                assert explanation is not None
                scope = comparison["stable_path"]
                case_id = comparison["case_id"]
                scope_matches = [
                    item
                    for item in hardware_prediction["scope_bounds"]
                    if isinstance(item, dict)
                    and item.get("case_id") == case_id
                    and item.get("scope") == scope
                ]
                candidate_matches = [
                    item
                    for item in hardware_prediction["candidates"]
                    if isinstance(item, dict)
                    and _candidate_path_matches_scope(
                        item.get("stable_path"), scope
                    )
                ]
                if len(scope_matches) != 1 or len(candidate_matches) != 1:
                    failures.append("physical floor derivation mismatch")
                else:
                    scope_bound = scope_matches[0]
                    candidate = candidate_matches[0]
                    candidate_duration = candidate["duration"]
                    measured = hardware_prediction["measured_capabilities"]
                    quality_statuses = {
                        item["quality_status"] for item in measured
                    }
                    quality_reasons = sorted(
                        {
                            reason
                            for item in measured
                            for reason in item["quality_reason_codes"]
                        }
                    )
                    if len(quality_statuses) != 1:
                        raise ValueError(
                            "inconsistent Hardware Capability quality statuses"
                        )
                    expected_quality = {
                        "status": next(iter(quality_statuses)),
                        "reason_codes": quality_reasons,
                    }
                    expected_floor = {
                        "status": hardware_prediction["status"],
                        "kind": "algorithm-independent-resource-physical-floor",
                        "minimum_work_flops": scope_bound["flops"],
                        "compulsory_bytes": scope_bound["compulsory_bytes"],
                        "compute_time_ns": scope_bound[
                            "empirical_compute_time_ns"
                        ],
                        "memory_time_ns": scope_bound[
                            "empirical_memory_time_ns"
                        ],
                        "resource_physical_floor_ns": scope_bound[
                            "empirical_hardware_floor_ns"
                        ],
                        "limiting_resource": scope_bound["limiting_resource"],
                        "full_duration_ns": None,
                        "formula": scope_bound["formula"],
                        "assumptions": scope_bound["assumptions"],
                        "quality": expected_quality,
                        "capabilities": measured,
                    }
                    candidate_consistent = (
                        candidate.get("flops") == scope_bound["flops"]
                        and candidate.get("compulsory_bytes")
                        == scope_bound["compulsory_bytes"]
                        and _has_fields(
                            candidate_duration,
                            {
                                "empirical_compute_time_ns": scope_bound[
                                    "empirical_compute_time_ns"
                                ],
                                "empirical_memory_time_ns": scope_bound[
                                    "empirical_memory_time_ns"
                                ],
                                "empirical_hardware_floor_ns": scope_bound[
                                    "empirical_hardware_floor_ns"
                                ],
                                "limiting_resource": scope_bound[
                                    "limiting_resource"
                                ],
                                "full_duration_ns": None,
                                "assumptions": scope_bound["assumptions"],
                            },
                        )
                    )
                    if not candidate_consistent or not _has_fields(
                        comparison.get("physical_floor"), expected_floor
                    ):
                        failures.append("physical floor derivation mismatch")

                    capability = hardware_prediction["capabilities"]
                    expected_theoretical = {
                        "fp32_flops_per_second": capability[
                            "fp32_flops_per_second"
                        ],
                        "peak_memory_bandwidth_bytes_per_second": capability[
                            "peak_memory_bandwidth_bytes_per_second"
                        ],
                    }
                    if comparison.get("theoretical_capability") != (
                        expected_theoretical
                    ):
                        failures.append("theoretical capability derivation mismatch")
                    expected_unsupported = {
                        "count": len(hardware_prediction["unsupported_regions"]),
                        "status": "partial-unknown",
                        "regions": hardware_prediction["unsupported_regions"],
                    }
                    if comparison.get("unsupported_regions") != expected_unsupported:
                        failures.append("unsupported region derivation mismatch")

                    source_manifest_artifact = next(
                        artifact
                        for artifact in artifacts
                        if isinstance(artifact, dict)
                        and artifact.get("role")
                        == "source-measurement-manifest"
                    )
                    summary = raw_timing["summary"]
                    expected_observation = {
                        "status": "known",
                        "quality": source_manifest["observation_validity"][
                            "status"
                        ],
                        "median_ns": summary["median"],
                        "q1_ns": summary["q1"],
                        "q3_ns": summary["q3"],
                        "iqr_over_median": summary[
                            "iqr_fraction_of_median"
                        ],
                        "timer_source": raw_timing["timer_source"],
                        "timer_resolution_ns": raw_timing["timer_resolution_ns"],
                        "completion_boundary": "closed",
                        "candidate": source_candidate["candidate_id"],
                        "source_run_id": source_manifest["run_id"],
                        "source_manifest_sha256": source_manifest_artifact[
                            "sha256"
                        ],
                    }
                    observation = comparison.get("observation")
                    if not _has_fields(observation, expected_observation):
                        if (
                            isinstance(observation, dict)
                            and observation.get("candidate")
                            != source_candidate.get("candidate_id")
                        ):
                            failures.append("comparison source candidate mismatch")
                        else:
                            failures.append("comparison observation derivation mismatch")

                    floor_value = expected_floor[
                        "resource_physical_floor_ns"
                    ]
                    observed_value = expected_observation["median_ns"]
                    expected_comparison = {
                        "observation_minus_physical_floor_ns": (
                            observed_value - floor_value
                        ),
                        "observed_to_physical_floor_ratio": (
                            observed_value / floor_value
                        ),
                        "relative_prediction_error": None,
                        "error_status": (
                            "not-evaluable-physical-floor-is-not-a-duration-prediction"
                        ),
                        "interpretation": (
                            "optimization-headroom-not-prediction-error"
                        ),
                    }
                    if comparison.get("comparison") != expected_comparison:
                        failures.append("comparison headroom derivation mismatch")

                    explanation_nodes = explanation.get("nodes")
                    nodes_by_id = {
                        node["id"]: node
                        for node in explanation_nodes
                        if isinstance(node, dict) and isinstance(node.get("id"), str)
                    }
                    expected_entrypoints = {
                        scope: [
                            "metric:resource-physical-floor",
                            "metric:observation",
                            "comparison:headroom",
                        ]
                    }
                    expected_node_fields = {
                        "scope:matmul": {
                            "kind": "stable-path",
                            "stable_path": scope,
                        },
                        "metric:minimum-work": {
                            "kind": "resource-demand",
                            "value": scope_bound["flops"],
                            "unit": "FLOP",
                        },
                        "metric:compulsory-bytes": {
                            "kind": "resource-demand",
                            "value": scope_bound["compulsory_bytes"],
                            "unit": "B",
                        },
                        "metric:resource-physical-floor": {
                            "kind": "resource-physical-floor",
                            "value_ns": floor_value,
                            "full_duration_ns": None,
                            "quality": expected_quality,
                            "hardware_cohort": comparison["hardware_cohort"],
                            "assumptions": scope_bound["assumptions"],
                            "capabilities": measured,
                        },
                        "metric:observation": {
                            "kind": "observation",
                            "value_ns": observed_value,
                            "completion_boundary": "closed",
                            "source_run_id": source_manifest["run_id"],
                            "hardware_cohort": comparison["hardware_cohort"],
                        },
                        "comparison:headroom": {
                            "kind": "optimization-headroom",
                            **expected_comparison,
                        },
                        "summary:unsupported-regions": {
                            "kind": "partial-unknown",
                            "count": expected_unsupported["count"],
                        },
                    }
                    expected_edges = [
                        {
                            "source": "scope:matmul",
                            "target": "metric:minimum-work",
                        },
                        {
                            "source": "scope:matmul",
                            "target": "metric:compulsory-bytes",
                        },
                        {
                            "source": "metric:minimum-work",
                            "target": "metric:resource-physical-floor",
                        },
                        {
                            "source": "metric:compulsory-bytes",
                            "target": "metric:resource-physical-floor",
                        },
                        {
                            "source": "metric:resource-physical-floor",
                            "target": "comparison:headroom",
                        },
                        {
                            "source": "metric:observation",
                            "target": "comparison:headroom",
                        },
                    ]
                    explanation_consistent = (
                        isinstance(explanation_nodes, list)
                        and len(nodes_by_id) == len(explanation_nodes)
                        and explanation.get("entrypoints") == expected_entrypoints
                        and explanation.get("edges") == expected_edges
                        and all(
                            _has_fields(nodes_by_id.get(node_id), expected)
                            for node_id, expected in expected_node_fields.items()
                        )
                    )
                    if not explanation_consistent:
                        failures.append("Explanation Graph derivation mismatch")
            except (
                AssertionError,
                KeyError,
                StopIteration,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ):
                failures.append("comparison derivation verification failed")

            report_artifact = next(
                (
                    artifact
                    for artifact in artifacts
                    if isinstance(artifact, dict)
                    and artifact.get("role") == "html-report"
                ),
                None,
            )
            if isinstance(report_artifact, dict):
                report_path = (root / str(report_artifact["path"])).resolve()
                try:
                    expected_report = render_physical_floor_report(comparison)
                    actual_report = report_path.read_text(encoding="utf-8")
                except (KeyError, OSError, TypeError, ValueError):
                    failures.append("HTML report derivation mismatch")
                else:
                    if actual_report != expected_report:
                        failures.append("HTML report derivation mismatch")

    if exact_shape:
        cohort = documents_by_role.get("hardware-cohort")
        if cohort is not None:
            cohort_path = paths_by_role["hardware-cohort"]
            if cohort.get("cohort_id") != manifest.get("hardware_cohort"):
                failures.append(f"hardware cohort mismatch: {cohort_path}")
            identity = {
                dimension: cohort.get(dimension)
                for dimension in COHORT_IDENTITY_DIMENSIONS
            }
            encoded_identity = json.dumps(
                identity,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            cohort_digest = sha256(encoded_identity).hexdigest()
            if completed_measurement and (
                cohort.get("cohort_digest") != cohort_digest
                or cohort.get("cohort_id")
                != f"ascend-npu-{cohort_digest[:16]}"
            ):
                failures.append(f"cohort digest mismatch: {cohort_path}")

        capabilities = documents_by_role.get(
            "measurement-capability-manifest"
        )
        if capabilities is not None:
            capability_path = paths_by_role[
                "measurement-capability-manifest"
            ]
            if capabilities.get("cohort_id") != manifest.get(
                "hardware_cohort"
            ):
                failures.append(
                    f"hardware cohort mismatch: {capability_path}"
                )
            adapter_identity = manifest.get("adapter", {})
            if completed_measurement and (
                not isinstance(adapter_identity, dict) or any(
                capabilities.get(key) != adapter_identity.get(key)
                for key in (
                    "adapter_id",
                    "adapter_version",
                    "protocol_id",
                    "protocol_version",
                )
                )
            ):
                failures.append(f"adapter mismatch: {capability_path}")

        for role in (
            "measurement-capability-manifest",
            "measurement-preflight",
            "timing-plan",
            "measurement-collection",
            "environment",
            "measurement-failure",
        ):
            document = documents_by_role.get(role)
            if (
                document is not None
                and document.get("device") != manifest.get("device")
            ):
                failures.append(f"device mismatch: {paths_by_role[role]}")

        preflight = documents_by_role.get("measurement-preflight")
        if (
            completed_measurement
            and preflight is not None
            and preflight.get("cohort_id") != manifest.get("hardware_cohort")
        ):
            failures.append(
                "hardware cohort mismatch: "
                f"{paths_by_role['measurement-preflight']}"
            )
        expected_logical_device = (
            preflight.get("logical_device") if preflight is not None else None
        )
        for role in (
            "timing-plan",
            "measurement-collection",
            "environment",
            "measurement-failure",
        ):
            document = documents_by_role.get(role)
            if (
                document is not None
                and document.get("logical_device") != expected_logical_device
            ):
                failures.append(
                    f"logical device mismatch: {paths_by_role[role]}"
                )

        environment = documents_by_role.get("environment")
        if environment is not None:
            environment_path = paths_by_role["environment"]
            if environment.get("preflight") != preflight:
                failures.append(
                    f"environment preflight mismatch: {environment_path}"
                )
            if cohort is not None and (
                environment.get("software") != cohort.get("software_evidence")
                or environment.get("cohort_identity_software")
                != cohort.get("software")
            ):
                failures.append(
                    f"environment cohort mismatch: {environment_path}"
                )

        collection = documents_by_role.get("measurement-collection")
        if collection is not None:
            component_roles = {
                "candidate_identity": "candidate-identity",
                "input_corpus": "input-corpus",
                "execution_contract": "execution-contract",
                "instrumentation_profile": "instrumentation-profile",
                "correctness": "correctness-observation",
                "raw_timing": "raw-timing-observation",
                "memory": "memory-observation",
                "completion_boundary": "completion-boundary",
            }
            for key, role in component_roles.items():
                component = documents_by_role.get(role)
                if component is not None and collection.get(key) != component:
                    failures.append(
                        f"collection component mismatch: {paths_by_role[role]}"
                    )
            candidate = documents_by_role.get("candidate-identity")
            contract = documents_by_role.get("execution-contract")
            if (
                candidate is not None
                and candidate.get("candidate_device")
                != expected_logical_device
            ):
                failures.append(
                    f"logical device mismatch: {paths_by_role['candidate-identity']}"
                )
            if (
                contract is not None
                and contract.get("logical_device") != expected_logical_device
            ):
                failures.append(
                    f"logical device mismatch: {paths_by_role['execution-contract']}"
                )

            correctness = documents_by_role.get("correctness-observation")
            raw_timing = documents_by_role.get("raw-timing-observation")
            completion = documents_by_role.get("completion-boundary")
            timing_plan = documents_by_role.get("timing-plan")
            samples = (
                raw_timing.get("samples", [])
                if raw_timing is not None
                else []
            )
            repetitions = (
                timing_plan.get("repetitions")
                if timing_plan is not None
                else None
            )
            summary = (
                raw_timing.get("summary", {})
                if raw_timing is not None
                else {}
            )
            timer_resolution = (
                raw_timing.get("timer_resolution_ns")
                if raw_timing is not None
                else None
            )
            recomputed_summary = (
                _measurement_timing_summary(samples)
                if isinstance(samples, list)
                and bool(samples)
                and all(
                    isinstance(sample, int)
                    and not isinstance(sample, bool)
                    and sample > 0
                    for sample in samples
                )
                else None
            )
            recomputed_timing_quality = (
                _measurement_timing_quality(
                    recomputed_summary,
                    timer_resolution_ns=float(timer_resolution),
                )
                if recomputed_summary is not None
                and isinstance(timer_resolution, (int, float))
                and not isinstance(timer_resolution, bool)
                and timer_resolution > 0
                else None
            )
            raw_timing_valid = (
                isinstance(samples, list)
                and bool(samples)
                and all(
                    isinstance(sample, int)
                    and not isinstance(sample, bool)
                    and sample > 0
                    for sample in samples
                )
                and isinstance(repetitions, int)
                and len(samples) == repetitions
                and isinstance(summary, dict)
                and summary == recomputed_summary
                and collection.get("timing_quality")
                == recomputed_timing_quality
            )
            timing_quality_status = (
                recomputed_timing_quality.get("status")
                if recomputed_timing_quality is not None
                else "quarantined"
            )
            timing_reason_codes = (
                list(recomputed_timing_quality["reason_codes"])
                if recomputed_timing_quality is not None
                else ["invalid-timing-evidence"]
            )
            expected_observation_validity = {
                "status": (
                    "valid"
                    if timing_quality_status == "passed"
                    else "quarantined"
                ),
                "correctness": "passed",
                "completion_boundary": "closed",
                "raw_timing_sample_count": len(samples),
                "timing_quality": timing_quality_status,
                "reason_codes": timing_reason_codes,
            }
            if (
                collection.get("status") != "completed"
                or correctness is None
                or correctness.get("status") != "passed"
                or completion is None
                or completion.get("closed") is not True
                or not raw_timing_valid
                or manifest.get("observation_validity")
                != expected_observation_validity
            ):
                failures.append("observation validity mismatch")

        operations = documents_by_role.get(
            "measurement-operation-evidence"
        )
        if operations is not None:
            declared_paths = {
                artifact.get("path")
                for artifact in artifacts
                if isinstance(artifact, dict)
            }
            operation_items = operations.get("operations", [])
            if not isinstance(operation_items, list):
                operation_items = []
                failures.append("invalid measurement operation evidence")
            expected_operation_roles = [
                ("discover_capabilities", "measurement-capability-manifest"),
                ("fingerprint_cohort", "hardware-cohort"),
                ("preflight", "measurement-preflight"),
            ]
            if completed_measurement:
                expected_operation_roles.extend(
                    [
                        ("build_timing_plan", "timing-plan"),
                        ("collect", "measurement-collection"),
                    ]
                )
            actual_operation_refs = [
                (
                    item.get("operation"),
                    item.get("evidence_ref"),
                )
                for item in operation_items
                if isinstance(item, dict)
            ]
            expected_operation_refs = [
                (
                    operation,
                    f"artifact://{paths_by_role.get(role, '')}",
                )
                for operation, role in expected_operation_roles
            ]
            if actual_operation_refs != expected_operation_refs:
                failures.append("measurement operation evidence mismatch")
            for operation in operation_items:
                reference = (
                    operation.get("evidence_ref")
                    if isinstance(operation, dict)
                    else None
                )
                referenced_path = (
                    reference[len("artifact://") :].split("#", 1)[0]
                    if isinstance(reference, str)
                    and reference.startswith("artifact://")
                    else None
                )
                if referenced_path not in declared_paths:
                    failures.append(f"missing evidence reference: {reference}")

        producer_lineage = manifest.get("producer_lineage")
        if isinstance(producer_lineage, dict):
            source_files = producer_lineage.get("source_files", [])
            lineage_digest = sha256()
            if isinstance(source_files, list):
                for source_file in sorted(
                    source_files,
                    key=lambda item: str(item.get("path"))
                    if isinstance(item, dict)
                    else "",
                ):
                    if not isinstance(source_file, dict):
                        continue
                    lineage_digest.update(str(source_file.get("path")).encode("utf-8"))
                    lineage_digest.update(b"\0")
                    lineage_digest.update(
                        str(source_file.get("sha256")).encode("ascii")
                    )
            expected_source_digest = lineage_digest.hexdigest()
            if producer_lineage.get("source_sha256") != expected_source_digest:
                failures.append("producer lineage digest mismatch")
            producer_suffix = expected_source_digest[:16]
            for artifact in artifacts:
                if isinstance(artifact, dict) and producer_suffix not in str(
                    artifact.get("produced_by")
                ):
                    failures.append(
                        f"producer lineage mismatch: {artifact.get('path')}"
                    )
        else:
            failures.append("missing producer lineage")
    return {
        "schema": "groundupscale.dev/run-verification/v1alpha1",
        "run_id": manifest["run_id"],
        "passed": not failures,
        "artifact_count": len(manifest["artifacts"]),
        "failures": failures,
    }


__all__ = [
    "EnvironmentValidityError",
    "RunBundleExistsError",
    "RunBundleWriter",
    "verify_run_bundle",
]
