"""Atomic, immutable, self-describing local Run Bundle writer."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
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
from groundupscale.execution_runtime import ExecutionRuntime
from groundupscale.ir import canonical_data, content_fingerprint
from groundupscale.measurement_contract import COHORT_IDENTITY_DIMENSIONS
from groundupscale.physical_floor_report import render_physical_floor_report
from groundupscale.pipeline import CompiledAnalysis

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

OPERATOR_FRONTIER_REQUIRED_ROLES = frozenset(
    {
        "operator-frontier-qualification",
        "diagnostic-evidence",
    }
)

TRANSFORMER_DEMO_COMPLETED_REQUIRED_ROLES = frozenset(
    {
        "resolved-input-lock",
        "environment",
        "measurement-capability-manifest",
        "hardware-cohort",
        "measurement-preflight",
        "model-ir",
        "workload-ir",
        "semantic-ir",
        "cost-ir",
        "hardware-backend-prediction",
        "prediction",
        "benchmark-observation",
        "observation-trace",
        "alignment-map",
        "memory-observation",
        "prediction-observation-comparison",
        "correctness-observation",
        "execution-contract",
        "transfer-observation",
        "error-attribution",
        "explanation-graph",
        "html-report",
    }
)

TRANSFORMER_DEMO_BLOCKED_REQUIRED_ROLES = frozenset(
    {
        "resolved-input-lock",
        "measurement-capability-manifest",
        "hardware-cohort",
        "measurement-preflight",
        "execution-failure",
    }
)

TRANSFORMER_DEMO_FAILED_REQUIRED_ROLES = (
    TRANSFORMER_DEMO_BLOCKED_REQUIRED_ROLES | {"transfer-observation"}
)
TRANSFORMER_DEMO_PRODUCER = "groundupscale@0.1.0"


def _transformer_demo_producer_lineage() -> dict[str, object]:
    return {
        "producer": TRANSFORMER_DEMO_PRODUCER,
        "source": "python://groundupscale.run_bundle",
        "artifact_lineage": "manifest.artifacts[].produced_by",
    }


@dataclass(frozen=True)
class NpuRunEvidence:
    capabilities: dict[str, object]
    cohort: dict[str, object]
    preflight: dict[str, object]


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


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        canonical_data(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


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


def _sysctl_value(name: str) -> str | None:
    if platform.system() != "Darwin":
        return None
    completed = subprocess.run(
        ["/usr/sbin/sysctl", "-n", name],
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _observed_hardware_identity() -> dict[str, Any]:
    values = {
        "model": _sysctl_value("hw.model"),
        "cpu_brand": _sysctl_value("machdep.cpu.brand_string"),
        "physical_cpu": _sysctl_value("hw.physicalcpu"),
        "logical_cpu": _sysctl_value("hw.logicalcpu"),
        "performance_cores": _sysctl_value("hw.perflevel0.physicalcpu"),
        "efficiency_cores": _sysctl_value("hw.perflevel1.physicalcpu"),
    }
    if any(value is None for value in values.values()):
        return {
            "status": "unresolved",
            "source": "sysctl",
            "reason_codes": ["required-apple-silicon-identity-unavailable"],
        }
    return {
        "status": "resolved",
        "source": "sysctl",
        "model": values["model"],
        "cpu_brand": values["cpu_brand"],
        "physical_cpu": int(str(values["physical_cpu"])),
        "logical_cpu": int(str(values["logical_cpu"])),
        "performance_levels": {
            "performance_cores": int(str(values["performance_cores"])),
            "efficiency_cores": int(str(values["efficiency_cores"])),
        },
    }


def _hardware_validity_cohort(
    *,
    bundle: Any,
    benchmark: dict[str, Any],
    device: str,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    policy = preflight.get("policy")
    observations = preflight.get("observations")
    power = observations.get("power") if isinstance(observations, dict) else None
    thermal = observations.get("thermal") if isinstance(observations, dict) else None
    hardware_names = [document.metadata.name for document in bundle.hardware]
    observed_hardware = _observed_hardware_identity()
    identity = {
        "schema": "groundupscale.dev/hardware-validity-cohort/v1alpha1",
        "device": {
            "hardware": hardware_names,
            "device": device,
            "observed_identity": observed_hardware,
            "identity_sha256": content_fingerprint(observed_hardware),
            "partition": "whole-device",
            "topology": "single-socket-unified-memory",
        },
        "software": {
            "os": platform.system(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_build_sha256": content_fingerprint(torch.__config__.show()),
            "operator_library": "pytorch-aten",
        },
        "numeric_execution": {
            "dtype": "float32",
            "execution_mode": "eager",
            "threads": benchmark.get("torch_num_threads"),
            "interop_threads": torch.get_num_interop_threads(),
            "affinity": "os-managed-unpinned",
            "numa": "single-socket-unified-memory",
            "context": "single-process-cpu",
            "stream": "not-applicable-cpu",
            "concurrency": "single-operator-no-overlap",
        },
        "power_clock": {
            "policy_id": policy.get("policy_id") if isinstance(policy, dict) else None,
            "power_source": power.get("source") if isinstance(power, dict) else None,
            "thermal_status": thermal.get("status") if isinstance(thermal, dict) else None,
            "clock_policy": "macos-default-managed",
        },
        "timing": {
            "timing_scope": "host_visible_completion",
            "completion_boundary": "synchronous-cpu-call-return",
            "timer_source": "time.perf_counter_ns",
            "timer_monotonic": bool(time.get_clock_info("perf_counter").monotonic),
            "instrumentation_profile": benchmark.get("instrumentation_profile"),
            "measurement_protocol": benchmark.get("measurement_protocol"),
            "adapter": {"name": "groundupscale-run-bundle", "version": "1.0.0"},
        },
    }
    identity["cohort_id"] = f"hvc-{content_fingerprint(identity)}"
    return identity


class RunBundleExistsError(FileExistsError):
    pass


class EnvironmentValidityError(RuntimeError):
    pass


def write_blocked_transformer_run(
    compiled: CompiledAnalysis,
    artifact_store: str | Path,
    *,
    run_id: str,
    npu_evidence: NpuRunEvidence,
) -> Path:
    """Publish immutable compatibility evidence when an NPU run cannot start."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"unsafe run_id: {run_id!r}")
    runs_root = Path(artifact_store).resolve() / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    destination = runs_root / run_id
    if destination.exists():
        raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
    artifacts: list[dict[str, Any]] = []

    def write_json(
        role: str,
        relative: str,
        value: object,
        schema: str,
        inputs: tuple[str, ...] = (),
    ) -> None:
        path = temporary / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_json_bytes(value))
        artifacts.append(
            {
                "role": role,
                "path": relative,
                "media_type": "application/json",
                "schema": schema,
                "sha256": _sha256(path),
                "produced_by": "groundupscale@0.1.0",
                "inputs": list(inputs),
            }
        )

    capabilities = npu_evidence.capabilities
    cohort = npu_evidence.cohort
    preflight = npu_evidence.preflight
    reason_code_values: list[object] = []
    for document in (capabilities, cohort, preflight):
        document_reasons = document.get("reason_codes")
        if isinstance(document_reasons, list):
            reason_code_values.extend(document_reasons)
    reason_codes = sorted({str(reason) for reason in reason_code_values}) or [
        "ascend-npu-compatibility-failed"
    ]
    bundle = compiled.bundle
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
    failure = {
        "schema": "groundupscale.dev/transformer-execution-failure/v1alpha1",
        "status": "compatibility-failed",
        "device": preflight.get("logical_device", "npu:0"),
        "reason_codes": reason_codes,
        "failed_before_execution": True,
        "preserved_evidence": [
            "measurement-capability-manifest",
            "hardware-cohort",
            "measurement-preflight",
        ],
    }
    write_json(
        "resolved-input-lock",
        "resolved/inputs.lock.json",
        inputs_lock,
        inputs_lock["schema"],
    )
    write_json(
        "measurement-capability-manifest",
        "adapter/capabilities.json",
        capabilities,
        str(capabilities["schema"]),
    )
    write_json(
        "hardware-cohort",
        "adapter/cohort.json",
        cohort,
        str(cohort["schema"]),
        ("measurement-capability-manifest",),
    )
    write_json(
        "measurement-preflight",
        "adapter/preflight.json",
        preflight,
        str(preflight["schema"]),
        ("measurement-capability-manifest", "hardware-cohort"),
    )
    write_json(
        "execution-failure",
        "observation/execution-failure.json",
        failure,
        str(failure["schema"]),
        ("measurement-preflight", "resolved-input-lock"),
    )
    manifest = {
        "schema": "groundupscale.dev/run-manifest/v1alpha1",
        "bundle_kind": "transformer-demo",
        "producer_lineage": _transformer_demo_producer_lineage(),
        "run_id": run_id,
        "status": "blocked",
        "created_at": datetime.now(UTC).isoformat(),
        "compilation_fingerprint": compiled.semantic.compilation_fingerprint,
        "cost_compilation_fingerprint": compiled.cost.compilation_fingerprint,
        "hardware_compilation_fingerprint": (
            compiled.hardware_prediction.compilation_fingerprint
            if compiled.hardware_prediction is not None
            else None
        ),
        "hardware_cohort": cohort.get("cohort_id"),
        "device": preflight.get("logical_device", "npu:0"),
        "reason_codes": reason_codes,
        "stages": {
            "compilation": "completed",
            "compatibility": "failed",
            "benchmark": "not-started",
            "trace": "not-started",
        },
        "artifacts": artifacts,
        "immutability": (
            "writer refuses an existing run_id; artifact digests are authoritative"
        ),
    }
    (temporary / "run.manifest.json").write_bytes(_json_bytes(manifest))
    os.replace(temporary, destination)
    return destination


class RunBundleWriter:
    def __init__(
        self,
        compiled: CompiledAnalysis,
        seed: int = 20260806,
        *,
        execution_runtime: ExecutionRuntime | None = None,
        npu_evidence: NpuRunEvidence | None = None,
    ) -> None:
        self.compiled = compiled
        self.seed = seed
        self.execution_runtime = execution_runtime
        self.npu_evidence = npu_evidence

    def run(
        self,
        artifact_store: str | Path,
        *,
        run_id: str | None = None,
        samples_override: int | None = None,
        warmup_override: int | None = None,
        windows_per_sample: int = 5,
        target_window_ns: int = 20_000_000,
        selected_case_ids: tuple[str, ...] | None = None,
        environment_validity: dict[str, Any] | None = None,
        require_valid_environment: bool = False,
    ) -> Path:
        benchmark_runner = BenchmarkRunner(
            self.compiled.bundle,
            seed=self.seed,
            execution_runtime=self.execution_runtime,
        )
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

        benchmark: dict[str, Any] | None = None
        trace: dict[str, Any] | None = None
        memory_observation: dict[str, Any] | None = None
        comparison: dict[str, Any] | None = None
        explanation: dict[str, Any] | None = None
        correctness: dict[str, Any] | None = None
        current_stage = "benchmark"
        try:
            benchmark = benchmark_runner.run(
                samples_override=samples_override,
                warmup_override=warmup_override,
                windows_per_sample=windows_per_sample,
                target_window_ns=target_window_ns,
                selected_case_ids=selected_case_ids,
            )
            cohort = (
                self.npu_evidence.cohort
                if self.npu_evidence is not None
                else _hardware_validity_cohort(
                    bundle=self.compiled.bundle,
                    benchmark=benchmark,
                    device=device,
                    preflight=preflight_artifact,
                )
            )
            benchmark["hardware_validity_cohort"] = cohort
            reference_runner = ReferenceRunner.from_analysis_bundle(
                self.compiled.bundle, seed=self.seed
            )
            operator_cases = [
                {
                    "case_id": case["case_id"],
                    "stable_path": case["resolved_scope"],
                    "candidate_identity": case["candidate_identity"],
                    "input_corpus": case["input_corpus"],
                    "execution_contract": case["execution_contract"],
                    "correctness": case["operator_correctness"],
                }
                for case in benchmark["cases"]
                if case.get("mode") == "operator"
            ]
            current_stage = "correctness"
            compatibility_error: RuntimeError | None = None
            if self.execution_runtime is not None:
                try:
                    correctness_result = reference_runner.compare_cpu_target(
                        self.execution_runtime, atol=0.001, rtol=0.001
                    )
                except RuntimeError as error:
                    compatibility_error = error
                    full_model_correctness = {
                        "status": "failed",
                        "reason_codes": [str(error)],
                    }
                else:
                    target_audit = canonical_data(correctness_result.mps.audit)
                    target_audit["leaf_output_devices"] = dict(
                        correctness_result.mps.audit.leaf_output_devices
                    )
                    target_audit["leaf_output_contracts"] = {
                        path: canonical_data(contract)
                        for path, contract in (
                            correctness_result.mps.audit.leaf_output_contracts
                        )
                    }
                    full_model_correctness = {
                        "status": (
                            "passed" if correctness_result.passed else "failed"
                        ),
                        "max_absolute_error": correctness_result.max_absolute_error,
                        "max_relative_error": correctness_result.max_relative_error,
                        "atol": correctness_result.atol,
                        "rtol": correctness_result.rtol,
                        "oracle": "cpu-float32-same-seed-same-weights",
                        "cpu_output_sha256": correctness_result.cpu.output_sha256,
                        "target_output_sha256": correctness_result.mps.output_sha256,
                        "target_audit": target_audit,
                    }
                    if not correctness_result.passed:
                        compatibility_error = RuntimeError(
                            "cpu-correctness-oracle-failed"
                        )
                    elif correctness_result.mps.audit.fallback_enabled:
                        compatibility_error = RuntimeError("cpu-fallback-detected")
            elif device == "mps":
                correctness_result = reference_runner.compare_cpu_mps(
                    atol=1e-4, rtol=1e-3
                )
                full_model_correctness = {
                    "status": "passed" if correctness_result.passed else "failed",
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
                full_model_correctness = {
                    "status": "not_evaluated_single_provider",
                    "reason_codes": ["independent-full-model-oracle-not-configured"],
                    "target_output_sha256": target_run.output_sha256,
                    "target_audit": canonical_data(target_run.audit),
                }
            operator_correctness_passed = all(
                case.get("correctness", {}).get("status") == "passed"
                for case in operator_cases
                if case.get("candidate_identity", {}).get("status")
                == "resolved"
            )
            correctness = {
                "schema": "groundupscale.dev/correctness-observation/v1alpha2",
                "passed": operator_correctness_passed
                and full_model_correctness.get("status") != "failed",
                "scope": "operator-specific-records-are-authoritative",
                "hardware_cohort": cohort["cohort_id"],
                "operator_cases": operator_cases,
                "full_model": full_model_correctness,
            }
            if "target_audit" in full_model_correctness:
                correctness.update(
                    {
                        key: full_model_correctness[key]
                        for key in (
                            "max_absolute_error",
                            "max_relative_error",
                            "atol",
                            "rtol",
                            "oracle",
                            "cpu_output_sha256",
                            "target_output_sha256",
                            "target_audit",
                        )
                        if key in full_model_correctness
                    }
                )
            current_stage = "trace"
            trace = TraceRunner(
                self.compiled.bundle,
                self.compiled.semantic.semantic_ir,
                seed=self.seed,
                execution_runtime=self.execution_runtime,
            ).run()
            current_stage = "memory-observation"
            memory_model, memory_input = benchmark_runner._model_and_input()
            tensor_storage_memory = observe_tensor_storage_peak(
                memory_model,
                (memory_input,),
                device=device,
                execution_runtime=self.execution_runtime,
            )
            memory_observation = {
                "schema": "groundupscale.dev/memory-observation/v1alpha1",
                "framework_tensor_storage": tensor_storage_memory,
                "runtime_point_samples": trace["memory_observation"],
                "authoritative_gate_metric": (
                    "framework_tensor_storage.peak_framework_tensor_bytes"
                ),
            }
            if self.execution_runtime is not None:
                runtime_memory = self.execution_runtime.memory_snapshot()
                current_rss_bytes = runtime_memory.get(
                    "process_current_rss_bytes",
                    runtime_memory.get("process_rss_bytes"),
                )
                peak_rss_bytes = runtime_memory.get(
                    "process_peak_observed_rss_bytes",
                    current_rss_bytes,
                )
                memory_observation.update(
                    {
                        "logical_tensor_live_set": tensor_storage_memory,
                        "framework_device_memory": {
                            "current_allocated_bytes": runtime_memory.get(
                                "framework_current_allocated_bytes"
                            ),
                            "reserved_bytes": runtime_memory.get(
                                "framework_reserved_bytes"
                            ),
                            "peak_allocated_bytes": runtime_memory.get(
                                "framework_max_allocated_bytes"
                            ),
                            "peak_reset_before_run": bool(
                                getattr(
                                    self.execution_runtime,
                                    "allocator_peak_reset_before_run",
                                    False,
                                )
                            ),
                            "attribution": (
                                "framework-owned-device-allocator"
                            ),
                        },
                        "process_memory": {
                            "current_rss_bytes": current_rss_bytes,
                            "peak_rss_bytes": peak_rss_bytes,
                            "peak_kind": "maximum-observed-point-sample",
                            "attribution": "process-wide-host-rss",
                        },
                    }
                )
            live_set = predict_live_set(
                self.compiled.semantic.semantic_ir, self.compiled.cost.cost_ir
            )
            current_stage = "comparison"
            comparison = build_prediction_observation_comparison(
                hardware_prediction=self.compiled.hardware_prediction,
                benchmark=benchmark,
                trace=trace,
                live_set=live_set,
                tensor_storage_observation=tensor_storage_memory,
                observation_evidence_tier=(
                    "qualified"
                    if preflight_status == "passed"
                    else "exploratory"
                    if environment_validity is not None
                    else "unverified"
                ),
                observation_reason_codes=tuple(
                    str(reason)
                    for reason in preflight_artifact.get("reason_codes", [])
                ),
                observation_hardware_cohort=str(cohort["cohort_id"]),
                observation_operator_cases=tuple(operator_cases),
            )
            explanation = build_explanation_graph(
                self.compiled.cost.cost_ir,
                benchmark,
                trace,
                live_set,
                self.compiled.hardware_prediction,
                comparison,
            )
            assert benchmark is not None
            assert trace is not None
            assert memory_observation is not None
            assert comparison is not None
            assert explanation is not None
            assert correctness is not None
            if compatibility_error is not None:
                current_stage = "correctness"
                raise compatibility_error
            current_stage = "publication"
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
                "operator_frontier_profiles": bundle.operator_frontier_profiles,
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
                "process": {
                    "pid": os.getpid(),
                    "session_id": selected_run_id,
                },
                "measurement_preflight": preflight_artifact,
                "hardware_validity_cohort": cohort,
                "policy": "allowlisted fields only; no unrestricted environment dump",
            }
            if self.execution_runtime is not None:
                environment["accelerator_runtime"] = (
                    self.execution_runtime.environment()
                )
            model_payload: Any = (
                self.compiled.models[0]
                if len(self.compiled.models) == 1
                else {"models": self.compiled.models}
            )
            write_json("resolved-input-lock", "resolved/inputs.lock.json", inputs_lock, inputs_lock["schema"])
            write_json("environment", "resolved/environment.json", environment, environment["schema"])
            if self.npu_evidence is not None:
                capabilities = self.npu_evidence.capabilities
                cohort = self.npu_evidence.cohort
                preflight = self.npu_evidence.preflight
                write_json(
                    "measurement-capability-manifest",
                    "adapter/capabilities.json",
                    capabilities,
                    str(capabilities["schema"]),
                    ("resolved-input-lock",),
                )
                write_json(
                    "hardware-cohort",
                    "adapter/cohort.json",
                    cohort,
                    str(cohort["schema"]),
                    ("measurement-capability-manifest",),
                )
                write_json(
                    "measurement-preflight",
                    "adapter/preflight.json",
                    preflight,
                    str(preflight["schema"]),
                    ("measurement-capability-manifest", "hardware-cohort"),
                )
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
                "schema": "groundupscale.dev/prediction/v1alpha2",
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
                    "observation-trace",
                    "memory-observation",
                ),
            )
            write_json("correctness-observation", "observation/correctness.json", correctness, correctness["schema"], ("resolved-input-lock", "environment"))
            if self.execution_runtime is not None:
                execution_contract = {
                    "schema": "groundupscale.dev/transformer-execution-contract/v1alpha1",
                    "device": device,
                    "semantic_leaf_count": correctness["target_audit"][
                        "semantic_leaf_count"
                    ],
                    "semantic_operations": sorted(
                        {
                            operation.operation
                            for operation in self.compiled.semantic.semantic_ir.walk_operations()
                        }
                    ),
                    "dtype": self.compiled.bundle.analysis_case.spec.shape.dtype,
                    "shape": self.compiled.bundle.analysis_case.spec.shape,
                    "baseline_timing": {
                        "timer_source": self.execution_runtime.timer_source,
                        "timer_resolution_ns": (
                            self.execution_runtime.timer_resolution_ns
                        ),
                        "completion_protocol": (
                            self.execution_runtime.completion_protocol
                        ),
                        "warmup": "outside-timed-region",
                        "per_module_synchronization": False,
                    },
                    "diagnostic_profiling": {
                        "lane": "separate",
                        "timing_is_frontier_eligible": False,
                    },
                }
                write_json(
                    "execution-contract",
                    "resolved/execution-contract.json",
                    execution_contract,
                    execution_contract["schema"],
                    (
                        "resolved-input-lock",
                        "measurement-preflight",
                        "benchmark-observation",
                    ),
                )
                transfers = self.execution_runtime.transfer_evidence()
                write_json(
                    "transfer-observation",
                    "observation/transfers.json",
                    transfers,
                    str(transfers["schema"]),
                    ("execution-contract",),
                )
            write_json("error-attribution", "comparison/error-attribution.json", trace["error_attribution"], trace["error_attribution"]["schema"], ("benchmark-observation", "alignment-map"))
            report = render_report_html(
                run_id=selected_run_id,
                device=device,
                benchmark=benchmark,
                trace=trace,
                live_set=live_set,
                explanation=explanation,
                comparison=comparison,
                memory_observation=memory_observation,
            )
            write_bytes("html-report", "reports/report.html", report.encode("utf-8"), media_type="text/html", schema="groundupscale.dev/html-report/v1alpha2", inputs=("explanation-graph", "prediction-observation-comparison"))

            manifest = {
                "schema": "groundupscale.dev/run-manifest/v1alpha1",
                **(
                    {
                        "bundle_kind": "transformer-demo",
                        "producer_lineage": (
                            _transformer_demo_producer_lineage()
                        ),
                    }
                    if self.execution_runtime is not None
                    else {}
                ),
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
                    self.npu_evidence.cohort.get("cohort_id")
                    if self.npu_evidence is not None
                    else cohort["cohort_id"]
                ),
                "device": device,
                "environment_validity": preflight_status,
                "seed": self.seed,
                "stages": {
                    "compilation": "completed",
                    **(
                        {"compatibility": "passed"}
                        if self.execution_runtime is not None
                        else {}
                    ),
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
        except Exception as error:
            if self.execution_runtime is not None:
                existing_roles = {
                    artifact["role"]
                    for artifact in artifacts
                    if isinstance(artifact, dict) and "role" in artifact
                }

                def write_json_once(
                    role: str,
                    relative: str,
                    value: object,
                    schema: str,
                    inputs: tuple[str, ...] = (),
                ) -> None:
                    if role not in existing_roles:
                        write_json(role, relative, value, schema, inputs)
                        existing_roles.add(role)

                def write_bytes_once(
                    role: str,
                    relative: str,
                    value: bytes,
                    *,
                    media_type: str,
                    schema: str,
                    inputs: tuple[str, ...] = (),
                ) -> None:
                    if role not in existing_roles:
                        write_bytes(
                            role,
                            relative,
                            value,
                            media_type=media_type,
                            schema=schema,
                            inputs=inputs,
                        )
                        existing_roles.add(role)

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
                        "hardware_capability_profiles": (
                            bundle.hardware_capability_profiles
                        ),
                        "fabric_graph": bundle.fabric_graph,
                        "benchmark_cases": bundle.benchmark_cases,
                        "models": bundle.models,
                    },
                }
                write_json_once(
                    "resolved-input-lock",
                    "resolved/inputs.lock.json",
                    inputs_lock,
                    inputs_lock["schema"],
                )
                if self.npu_evidence is not None:
                    capabilities = self.npu_evidence.capabilities
                    cohort = self.npu_evidence.cohort
                    preflight = self.npu_evidence.preflight
                    write_json_once(
                        "measurement-capability-manifest",
                        "adapter/capabilities.json",
                        capabilities,
                        str(capabilities["schema"]),
                        ("resolved-input-lock",),
                    )
                    write_json_once(
                        "hardware-cohort",
                        "adapter/cohort.json",
                        cohort,
                        str(cohort["schema"]),
                        ("measurement-capability-manifest",),
                    )
                    write_json_once(
                        "measurement-preflight",
                        "adapter/preflight.json",
                        preflight,
                        str(preflight["schema"]),
                        (
                            "measurement-capability-manifest",
                            "hardware-cohort",
                        ),
                    )
                if benchmark is not None:
                    write_json_once(
                        "benchmark-observation",
                        "observation/raw/benchmark.json",
                        benchmark,
                        str(benchmark["schema"]),
                        ("resolved-input-lock",),
                    )
                if trace is not None:
                    trace_lines = b"".join(
                        _json_line_bytes(event) for event in trace["events"]
                    )
                    write_bytes_once(
                        "observation-trace",
                        "observation/observation.trace.jsonl",
                        trace_lines,
                        media_type="application/x-ndjson",
                        schema="groundupscale.dev/observation-span/v1alpha1",
                        inputs=("resolved-input-lock",),
                    )
                    alignment = trace["alignment_map"]
                    write_json_once(
                        "alignment-map",
                        "observation/alignment.map.json",
                        alignment,
                        str(alignment["schema"]),
                        ("observation-trace",),
                    )
                    attribution = trace["error_attribution"]
                    write_json_once(
                        "error-attribution",
                        "comparison/error-attribution.json",
                        attribution,
                        str(attribution["schema"]),
                        ("benchmark-observation", "alignment-map"),
                    )
                if memory_observation is not None:
                    write_json_once(
                        "memory-observation",
                        "observation/memory.json",
                        memory_observation,
                        str(memory_observation["schema"]),
                        ("observation-trace",),
                    )
                if comparison is not None:
                    write_json_once(
                        "prediction-observation-comparison",
                        "comparison/predicted-vs-observed.json",
                        comparison,
                        str(comparison["schema"]),
                        ("benchmark-observation", "memory-observation"),
                    )
                if explanation is not None:
                    write_json_once(
                        "explanation-graph",
                        "prediction/explanation.graph.json",
                        explanation,
                        str(explanation["schema"]),
                        ("prediction-observation-comparison",),
                    )
                if correctness is not None:
                    write_json_once(
                        "correctness-observation",
                        "observation/correctness.json",
                        correctness,
                        str(correctness["schema"]),
                        ("resolved-input-lock",),
                    )
                transfers = self.execution_runtime.transfer_evidence()
                write_json_once(
                    "transfer-observation",
                    "observation/transfers.json",
                    transfers,
                    str(transfers["schema"]),
                    ("resolved-input-lock",),
                )
                raw_reason = str(error).strip()
                reason = (
                    raw_reason
                    if raw_reason
                    and len(raw_reason) <= 160
                    and all(character.isprintable() for character in raw_reason)
                    else f"execution-failed:{type(error).__name__}"
                )
                failure = {
                    "schema": (
                        "groundupscale.dev/transformer-execution-failure/"
                        "v1alpha1"
                    ),
                    "run_id": selected_run_id,
                    "status": "compatibility-failed",
                    "device": device,
                    "failed_stage": current_stage,
                    "failed_before_execution": False,
                    "reason_codes": [reason],
                    "error_type": type(error).__name__,
                    "captured_at": datetime.now(UTC).isoformat(),
                }
                write_json_once(
                    "execution-failure",
                    "observation/execution-failure.json",
                    failure,
                    failure["schema"],
                    (
                        "resolved-input-lock",
                        "measurement-preflight",
                        "transfer-observation",
                    ),
                )
                manifest = {
                    "schema": "groundupscale.dev/run-manifest/v1alpha1",
                    "bundle_kind": "transformer-demo",
                    "producer_lineage": _transformer_demo_producer_lineage(),
                    "run_id": selected_run_id,
                    "status": "compatibility-failed",
                    "created_at": datetime.now(UTC).isoformat(),
                    "compilation_fingerprint": (
                        self.compiled.semantic.compilation_fingerprint
                    ),
                    "cost_compilation_fingerprint": (
                        self.compiled.cost.compilation_fingerprint
                    ),
                    "hardware_compilation_fingerprint": (
                        self.compiled.hardware_prediction.compilation_fingerprint
                        if self.compiled.hardware_prediction is not None
                        else None
                    ),
                    "hardware_cohort": (
                        self.npu_evidence.cohort.get("cohort_id")
                        if self.npu_evidence is not None
                        else None
                    ),
                    "device": device,
                    "reason_codes": [reason],
                    "stages": {
                        "compilation": "completed",
                        "compatibility": "failed",
                        current_stage: "failed",
                    },
                    "artifacts": artifacts,
                    "immutability": (
                        "writer refuses an existing run_id; artifact digests "
                        "are authoritative"
                    ),
                }
                (temporary / "run.manifest.json").write_bytes(
                    _json_bytes(manifest)
                )
                os.replace(temporary, destination)
                return destination
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
    if manifest.get("source_manifest_integrity") == "required":
        source_runs = manifest.get("source_runs")
        seen_source_ids: set[str] = set()
        seen_source_paths: set[Path] = set()
        if not isinstance(source_runs, list) or not source_runs:
            failures.append("source manifest integrity requires source_runs")
        else:
            for source in source_runs:
                if not isinstance(source, dict):
                    failures.append("invalid source manifest lineage")
                    continue
                source_id = source.get("run_id")
                relative_path = source.get("path")
                manifest_digest = source.get("manifest_sha256")
                if (
                    not isinstance(source_id, str)
                    or not source_id
                    or source_id in seen_source_ids
                    or not isinstance(relative_path, str)
                    or not relative_path
                    or Path(relative_path).is_absolute()
                    or not isinstance(manifest_digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", manifest_digest) is None
                ):
                    failures.append("invalid source manifest lineage")
                    continue
                source_root = (root / relative_path).resolve()
                source_manifest_path = source_root / "run.manifest.json"
                if source_root in seen_source_paths:
                    failures.append("duplicate source manifest lineage path")
                    continue
                seen_source_ids.add(source_id)
                seen_source_paths.add(source_root)
                if not source_manifest_path.is_file():
                    failures.append(f"missing source Run manifest: {source_id}")
                    continue
                if _sha256(source_manifest_path) != manifest_digest:
                    failures.append(
                        f"source Run manifest digest mismatch: {source_id}"
                    )
                    continue
                try:
                    source_manifest = json.loads(
                        source_manifest_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    failures.append(f"invalid source Run manifest: {source_id}")
                    continue
                if (
                    not isinstance(source_manifest, dict)
                    or source_manifest.get("run_id") != source_id
                    or source_manifest.get("status") != "completed"
                ):
                    failures.append(f"source Run identity mismatch: {source_id}")
    exact_shape = manifest.get("bundle_kind") == "exact-shape-measurement"
    floor_comparison = (
        manifest.get("bundle_kind") == "physical-floor-observation-comparison"
    )
    transformer_demo = manifest.get("bundle_kind") == "transformer-demo"
    operator_frontier = manifest.get("bundle_kind") == "operator-frontier"
    structured_bundle = (
        exact_shape or floor_comparison or transformer_demo or operator_frontier
    )
    completed_measurement = exact_shape and manifest.get("status") == "completed"
    role_counts: dict[object, int] = {}
    if structured_bundle:
        for artifact in artifacts:
            if isinstance(artifact, dict):
                role = artifact.get("role")
                role_counts[role] = role_counts.get(role, 0) + 1
        if operator_frontier:
            required_roles = OPERATOR_FRONTIER_REQUIRED_ROLES
        elif transformer_demo:
            if manifest.get("status") == "completed":
                required_roles = TRANSFORMER_DEMO_COMPLETED_REQUIRED_ROLES
            elif manifest.get("status") == "blocked":
                required_roles = TRANSFORMER_DEMO_BLOCKED_REQUIRED_ROLES
            else:
                required_roles = TRANSFORMER_DEMO_FAILED_REQUIRED_ROLES
        elif floor_comparison:
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
        if transformer_demo:
            lineage = manifest.get("producer_lineage")
            if (
                not isinstance(lineage, dict)
                or lineage.get("producer") != TRANSFORMER_DEMO_PRODUCER
                or any(
                    isinstance(artifact, dict)
                    and artifact.get("produced_by")
                    != TRANSFORMER_DEMO_PRODUCER
                    for artifact in artifacts
                )
                ):
                    failures.append("invalid transformer demo producer lineage")

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

    if operator_frontier:
        qualification = documents_by_role.get(
            "operator-frontier-qualification"
        )
        diagnostic = documents_by_role.get("diagnostic-evidence")
        source_runs = manifest.get("source_runs")
        if (
            manifest.get("status") != "completed"
            or manifest.get("device") != "ascend-npu"
            or not isinstance(source_runs, list)
            or not source_runs
            or not isinstance(qualification, dict)
            or qualification.get("status") != "qualified"
            or qualification.get("hardware_cohort")
            != manifest.get("hardware_cohort")
            or qualification.get("source_runs") != source_runs
            or not isinstance(diagnostic, dict)
        ):
            failures.append("invalid operator Frontier bundle identity")
        else:
            surface = qualification.get("surface")
            surfaces = diagnostic.get("capability_surfaces")
            surface_ref = manifest.get("surface")
            if (
                not isinstance(surface, dict)
                or surfaces != [surface]
                or not isinstance(surface_ref, dict)
                or surface_ref
                != {
                    "surface_id": surface.get("surface_id"),
                    "version": surface.get("version"),
                    "input_digest": surface.get("input_digest"),
                }
                or surface.get("cohort_id") != manifest.get("hardware_cohort")
                or diagnostic.get("cohort_id") != manifest.get("hardware_cohort")
            ):
                failures.append("operator Frontier Surface identity mismatch")
            if isinstance(surface, dict):
                expected_surface_digest = surface.get("input_digest")
                surface_body = {
                    key: value
                    for key, value in surface.items()
                    if key != "input_digest"
                }
                if expected_surface_digest != _canonical_digest(surface_body):
                    failures.append(
                        "operator Frontier Surface input digest mismatch"
                    )
            policy = qualification.get("policy")
            if isinstance(policy, dict):
                expected_policy_digest = policy.get("input_digest")
                policy_body = {
                    key: value
                    for key, value in policy.items()
                    if key != "input_digest"
                }
                if expected_policy_digest != _canonical_digest(policy_body):
                    failures.append(
                        "operator Frontier qualification policy digest mismatch"
                    )
            else:
                failures.append("invalid operator Frontier qualification policy")
            diagnostic_input_keys = (
                "resolved_configuration",
                "resolved_ir",
                "hardware",
                "cohort_id",
                "execution_domain",
            )
            if all(key in diagnostic for key in diagnostic_input_keys):
                diagnostic_inputs = {
                    key: diagnostic[key] for key in diagnostic_input_keys
                }
                diagnostic_evidence = {
                    key: value
                    for key, value in diagnostic.items()
                    if key
                    not in {*diagnostic_input_keys, "schema", "digests"}
                }
                expected_digests = diagnostic.get("digests")
                actual_digests = {
                    "input_sha256": _canonical_digest(diagnostic_inputs),
                    "evidence_sha256": _canonical_digest(
                        diagnostic_evidence
                    ),
                }
                if not isinstance(expected_digests, dict) or any(
                    expected_digests.get(name) != digest
                    for name, digest in actual_digests.items()
                ):
                    failures.append(
                        "operator Frontier diagnostic evidence digest mismatch"
                    )
            else:
                failures.append(
                    "operator Frontier diagnostic evidence digest mismatch"
                )
            seen_source_ids: set[str] = set()
            seen_source_paths: set[Path] = set()
            for source in source_runs:
                if not isinstance(source, dict):
                    failures.append("invalid operator Frontier source Run")
                    continue
                source_id = source.get("run_id")
                relative_path = source.get("path")
                if (
                    not isinstance(source_id, str)
                    or not source_id
                    or source_id in seen_source_ids
                    or not isinstance(relative_path, str)
                    or not relative_path
                    or Path(relative_path).is_absolute()
                ):
                    failures.append("invalid operator Frontier source Run")
                    continue
                source_root = (root / relative_path).resolve()
                source_manifest_path = source_root / "run.manifest.json"
                if source_root in seen_source_paths:
                    failures.append("duplicate operator Frontier source Run path")
                    continue
                seen_source_ids.add(source_id)
                seen_source_paths.add(source_root)
                if not source_manifest_path.is_file():
                    failures.append(
                        f"missing operator Frontier source Run: {source_id}"
                    )
                    continue
                if _sha256(source_manifest_path) != source.get(
                    "manifest_sha256"
                ):
                    failures.append(
                        f"source Run Manifest digest mismatch: {source_id}"
                    )
                    continue
                source_manifest = json.loads(
                    source_manifest_path.read_text(encoding="utf-8")
                )
                if (
                    source_manifest.get("run_id") != source_id
                    or source_manifest.get("bundle_kind")
                    != "exact-shape-measurement"
                    or source_manifest.get("hardware_cohort")
                    != manifest.get("hardware_cohort")
                ):
                    failures.append(
                        f"operator Frontier source Run identity mismatch: {source_id}"
                    )
                    continue
                source_verification = verify_run_bundle(source_root)
                if source_verification.get("passed") is not True:
                    failures.append(
                        f"operator Frontier source Run failed verification: {source_id}"
                    )

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
            correctness_status = (
                correctness.get("status")
                if isinstance(correctness, dict)
                else None
            )
            expected_observation_validity = {
                "status": (
                    "rejected"
                    if correctness_status != "passed"
                    else (
                        "valid"
                        if timing_quality_status == "passed"
                        else "quarantined"
                    )
                ),
                "correctness": correctness_status,
                "completion_boundary": "closed",
                "raw_timing_sample_count": len(samples),
                "timing_quality": timing_quality_status,
                "reason_codes": [
                    *(
                        ["candidate-correctness-failed"]
                        if correctness_status != "passed"
                        else []
                    ),
                    *timing_reason_codes,
                ],
            }
            if (
                collection.get("status") != "completed"
                or correctness is None
                or correctness_status not in {"passed", "failed"}
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
    "NpuRunEvidence",
    "RunBundleExistsError",
    "RunBundleWriter",
    "verify_run_bundle",
    "write_blocked_transformer_run",
]
