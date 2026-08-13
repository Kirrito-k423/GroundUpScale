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
MODEL_E2E_FRONTIER_REQUIRED_ROLES = frozenset(
    {
        "model-e2e-frontier-input",
        "prediction-observation-comparison",
        "html-report",
    }
)
TRANSFORMER_MATMUL_FRONTIER_REQUIRED_ROLES = frozenset(
    {
        "transformer-matmul-execution-ir",
        "matmul-domain-inventory",
        "transformer-matmul-frontier-qualification",
    }
)
LEGACY_TRANSFORMER_MATMUL_FRONTIER_REQUIRED_ROLES = frozenset(
    {
        "matmul-domain-inventory",
        "transformer-matmul-frontier-qualification",
    }
)
TRANSFORMER_MATMUL_EXACT_ANCHOR_REQUIRED_ROLES = frozenset(
    {"transformer-matmul-exact-anchor"}
)
TRANSFORMER_MATMUL_SURFACE_REQUIRED_ROLES = frozenset(
    {"transformer-matmul-surface"}
)
COMPOUND_OPERATOR_FRONTIER_REQUIRED_ROLES = frozenset(
    {
        "operator-phase-graph",
        "compound-operator-frontier-qualification",
        "compound-operator-diagnostic",
    }
)
OPERATOR_PHASE_MEASUREMENT_REQUIRED_ROLES = frozenset(
    {"operator-phase-capability-observation"}
)
EVIDENCE_DATASET_SCHEMA = "groundupscale.dev/evidence-dataset/v1alpha1"

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


_SOFTMAX_PHASE_SPECS = (
    ("max_reduce", "torch.amax", "compute.reduction.max.fp32"),
    ("subtract", "torch.sub", "compute.elementwise.subtract.fp32"),
    ("exp", "torch.exp", "compute.transcendental.exp.fp32"),
    ("sum_reduce", "torch.sum", "compute.reduction.sum.fp32"),
    ("normalize", "torch.div", "compute.elementwise.divide.fp32"),
)
_SOFTMAX_INPUT_ROLES = {
    "max_reduce": ["softmax_input"],
    "subtract": ["softmax_input", "row_max"],
    "exp": ["centered_logits"],
    "sum_reduce": ["exponentials"],
    "normalize": ["exponentials", "row_sum"],
}
_SOFTMAX_OUTPUT_ROLES = {
    "max_reduce": ["row_max"],
    "subtract": ["centered_logits"],
    "exp": ["exponentials"],
    "sum_reduce": ["row_sum"],
    "normalize": ["softmax_output"],
}
_SOFTMAX_LEGACY_NORMALIZATION_MANIFESTS = frozenset({
    "196748d01d508ae414d414f492eedd0b54e45b260f0cfa4102ffa789b2e36a3d",
    "42dd7a1004309672f42ce8e7ef5d8d7747f4f90537fe720b11be98bf8d4df36f",
    "4f98dc94c76e5fb6b71b8950e3152203fe6c1fe729b5b3119ea37e9791a14304",
    "65f7db7d0b8d762bd66f9e01f6f85fd12b288f76924be1c1ac297bb7aa9828cd",
    "67163f7b599fa89300cf71a9bbaa3ac692861521b78af6bb82f141e3b19ef24d",
    "7d533c056c4a5bfbb59cd7de5489cac83097ace8dfae1bb966f96795003233ea",
    "87d19ebf8058adf70248d2315b2e3895e99cc75e6dba8764b527e8726b2af38c",
    "b14bad53362f009001ddbe4579d3ce6b3a51ef7d968fd9a617a1c69429040e39",
    "c7dc79d99aeb5f87e084068c1352b5cdb1678ba9f3e8adb9597db15bd5b65f6e",
    "f2a2cdb1e4408b1e687c5d2297fa2221b0465220742504107ca64ba1203676e5",
})


def _softmax_source_replay(
    root: Path, source_runs: list[object]
) -> tuple[dict[str, dict[str, dict[str, object]]], list[str]]:
    replay: dict[str, dict[str, dict[str, object]]] = {}
    failures: list[str] = []
    for source in source_runs:
        if not isinstance(source, dict):
            continue
        relative_path = source.get("path")
        if not isinstance(relative_path, str) or Path(relative_path).is_absolute():
            continue
        source_root = (root / relative_path).resolve()
        try:
            manifest = json.loads((source_root / "run.manifest.json").read_text())
            documents: dict[str, dict[str, object]] = {}
            for artifact in manifest.get("artifacts", []):
                if isinstance(artifact, dict) and isinstance(artifact.get("role"), str):
                    document = json.loads((source_root / str(artifact["path"])).read_text())
                    if isinstance(document, dict):
                        documents[str(artifact["role"])] = document
            case = documents["benchmark-case"]
            candidate = documents["candidate-identity"]
            timing = documents["raw-timing-observation"]
            environment = documents["environment"]
            phase = str(case["phase"])
            lane = str(source.get("lane"))
            samples = timing["samples"]
            session = environment["measurement_session"]
            expected_spec = next(item for item in _SOFTMAX_PHASE_SPECS if item[0] == phase)
            candidate_body = dict(candidate)
            candidate_digest = candidate_body.pop("candidate_digest")
            record = {
                "run_id": manifest["run_id"],
                "manifest_sha256": _sha256(source_root / "run.manifest.json"),
                "lane": lane,
                "phase": phase,
                "shape": case["shape"],
                "axis": case["axis"],
                "dtype": case["dtype"],
                "layout": case["layout"],
                "logical_device": documents["execution-contract"]["logical_device"],
                "execution_mode": documents["execution-contract"]["execution_mode"],
                "cohort": manifest["hardware_cohort"],
                "candidate_id": candidate["candidate_id"],
                "candidate_family": candidate["candidate_family"],
                "candidate_digest": candidate_digest,
                "capability_class": candidate["capability_class"],
                "median_ns": float(statistics.median(samples)),
                "standard_uncertainty_ns": float(statistics.stdev(samples)),
                "process_identity": (
                    session["process_id"], session["process_started_at"]
                ),
            }
            if (
                source.get("run_id") != record["run_id"]
                or source.get("manifest_sha256") != record["manifest_sha256"]
                or source.get("phase") != phase
                or lane not in {"search", "holdout"}
                or candidate_digest != content_fingerprint(candidate_body)
                or (candidate["candidate_id"], candidate["capability_class"])
                != expected_spec[1:]
            ):
                failures.append(f"Softmax source lineage mismatch: {source.get('run_id')}")
                continue
            if lane in replay.setdefault(phase, {}):
                failures.append(f"duplicate Softmax {phase} {lane} source")
                continue
            replay[phase][lane] = record
        except (KeyError, StopIteration, OSError, ValueError, TypeError, json.JSONDecodeError):
            failures.append(f"invalid Softmax source replay: {source.get('run_id')}")
    return replay, failures


def _semantic_softmax_paths(value: object) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        if value.get("operation") == "Softmax" and isinstance(value.get("stable_path"), str):
            paths.append(str(value["stable_path"]))
        for child in value.values():
            paths.extend(_semantic_softmax_paths(child))
    elif isinstance(value, list):
        for child in value:
            paths.extend(_semantic_softmax_paths(child))
    return sorted(paths)


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
    model_e2e_frontier = (
        manifest.get("bundle_kind") == "model-e2e-frontier"
    )
    compound_operator_frontier = (
        manifest.get("bundle_kind") == "compound-operator-frontier"
    )
    transformer_matmul_frontier = (
        manifest.get("bundle_kind") == "transformer-matmul-frontier"
    )
    transformer_matmul_exact_anchor = (
        manifest.get("bundle_kind") == "transformer-matmul-exact-anchor"
    )
    transformer_matmul_surface = (
        manifest.get("bundle_kind") == "transformer-matmul-surface"
    )
    operator_phase_measurement = (
        manifest.get("bundle_kind") == "operator-phase-measurement"
    )
    structured_bundle = (
        exact_shape
        or floor_comparison
        or transformer_demo
        or operator_frontier
        or model_e2e_frontier
        or transformer_matmul_frontier
        or transformer_matmul_exact_anchor
        or transformer_matmul_surface
        or compound_operator_frontier
        or operator_phase_measurement
        or operator_phase_measurement
    )
    supersedes = manifest.get("supersedes")
    enforce_supersession = (
        transformer_matmul_frontier or transformer_matmul_exact_anchor
    ) and root.name.endswith("-v4")
    if enforce_supersession:
        seen_superseded_ids: set[str] = set()
        seen_superseded_paths: set[Path] = set()
        if not isinstance(supersedes, list) or not supersedes:
            failures.append("invalid supersession lineage")
        else:
            for record in supersedes:
                if not isinstance(record, dict):
                    failures.append("invalid supersession lineage")
                    continue
                run_id = record.get("run_id")
                relative_path = record.get("path")
                digest = record.get("manifest_sha256")
                if (
                    not isinstance(run_id, str)
                    or not run_id
                    or run_id in seen_superseded_ids
                    or not isinstance(relative_path, str)
                    or not relative_path
                    or Path(relative_path).is_absolute()
                    or not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                ):
                    failures.append("invalid supersession lineage")
                    continue
                superseded_root = (root / relative_path).resolve()
                superseded_manifest_path = superseded_root / "run.manifest.json"
                if superseded_root in seen_superseded_paths:
                    failures.append("invalid supersession lineage")
                    continue
                seen_superseded_ids.add(run_id)
                seen_superseded_paths.add(superseded_root)
                if not superseded_manifest_path.is_file():
                    failures.append("supersession lineage mismatch")
                    continue
                try:
                    superseded_manifest = json.loads(
                        superseded_manifest_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    failures.append("supersession lineage mismatch")
                    continue
                if (
                    _sha256(superseded_manifest_path) != digest
                    or superseded_manifest.get("run_id") != run_id
                ):
                    failures.append("supersession lineage mismatch")
    completed_measurement = exact_shape and manifest.get("status") == "completed"
    role_counts: dict[object, int] = {}
    if structured_bundle:
        for artifact in artifacts:
            if isinstance(artifact, dict):
                role = artifact.get("role")
                role_counts[role] = role_counts.get(role, 0) + 1
        if model_e2e_frontier:
            required_roles = MODEL_E2E_FRONTIER_REQUIRED_ROLES
        elif transformer_matmul_exact_anchor:
            required_roles = TRANSFORMER_MATMUL_EXACT_ANCHOR_REQUIRED_ROLES
        elif transformer_matmul_surface:
            required_roles = TRANSFORMER_MATMUL_SURFACE_REQUIRED_ROLES
        elif transformer_matmul_frontier:
            required_roles = (
                LEGACY_TRANSFORMER_MATMUL_FRONTIER_REQUIRED_ROLES
                if manifest.get("run_id") in {
                    "issue42-issue42-20260813-v1-transformer-matmul-frontier",
                    "issue42-issue42-20260813-v1-transformer-matmul-frontier-final",
                }
                else TRANSFORMER_MATMUL_FRONTIER_REQUIRED_ROLES
            )
        elif operator_phase_measurement:
            required_roles = OPERATOR_PHASE_MEASUREMENT_REQUIRED_ROLES
        elif compound_operator_frontier:
            required_roles = COMPOUND_OPERATOR_FRONTIER_REQUIRED_ROLES
        elif operator_frontier:
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

    if model_e2e_frontier:
        source = documents_by_role.get("model-e2e-frontier-input")
        comparison = documents_by_role.get(
            "prediction-observation-comparison"
        )
        report_entry = next(
            (
                artifact
                for artifact in artifacts
                if isinstance(artifact, dict)
                and artifact.get("role") == "html-report"
            ),
            None,
        )
        if (
            manifest.get("status") not in {"complete", "unknown"}
            or not isinstance(source, dict)
            or not isinstance(comparison, dict)
            or not isinstance(report_entry, dict)
            or manifest.get("hardware_cohort")
            != comparison.get("hardware_cohort")
        ):
            failures.append("invalid model E2E Frontier bundle identity")
        else:
            try:
                from groundupscale.model_e2e_frontier import (
                    compose_model_e2e_frontier,
                    render_model_e2e_frontier_report,
                )

                expected = compose_model_e2e_frontier(source)
                report_path = (root / str(report_entry["path"])).resolve()
                actual_report = report_path.read_text(encoding="utf-8")
                expected_report = render_model_e2e_frontier_report(expected)
            except (OSError, UnicodeDecodeError, ValueError) as error:
                failures.append(f"invalid model E2E Frontier evidence: {error}")
            else:
                if comparison != expected:
                    failures.append("model E2E comparison derivation mismatch")
                if actual_report != expected_report:
                    failures.append("model E2E human report projection mismatch")
                if manifest.get("status") != expected.get("status"):
                    failures.append("model E2E manifest status mismatch")

    if operator_frontier:
        qualification = documents_by_role.get(
            "operator-frontier-qualification"
        )
        diagnostic = documents_by_role.get("diagnostic-evidence")
        source_dataset = documents_by_role.get("source-dataset-manifest")
        source_runs = manifest.get("source_runs")
        if (
            manifest.get("status") != "completed"
            or manifest.get("device") != "ascend-npu"
            or not isinstance(source_runs, list)
            or not source_runs
            or not isinstance(qualification, dict)
            or qualification.get("status") not in {"qualified", "rejected", "unknown"}
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

            if (
                isinstance(surface, dict)
                and surface.get(
                    "qualification_status",
                    "qualified",
                )
                != qualification.get("status")
            ):
                failures.append("operator Frontier qualification status mismatch")
            if (
                isinstance(surface, dict)
                and qualification.get("anchors") != surface.get("anchors")
            ):
                failures.append("operator Frontier qualification anchors mismatch")
            if (
                qualification.get("status") == "qualified"
                and (
                    not isinstance(surface, dict)
                    or not isinstance(surface.get("anchors"), list)
                    or not surface["anchors"]
                    or not isinstance(surface.get("cells"), list)
                    or not any(
                        isinstance(cell, dict)
                        and cell.get("status") == "retained"
                        for cell in surface["cells"]
                    )
                )
            ):
                failures.append("invalid qualified operator Frontier qualification")
            if (
                qualification.get("status") == "rejected"
                and (
                    not isinstance(surface, dict)
                    or surface.get("qualification_status") != "rejected"
                    or surface.get("rejection_reason_code")
                    != qualification.get("reason_code")
                    or not isinstance(qualification.get("reason_code"), str)
                    or not qualification["reason_code"]
                    or surface.get("anchors") != []
                    or surface.get("cells") != []
                    or not isinstance(qualification.get("stopping_decision"), dict)
                    or qualification["stopping_decision"].get("status") != "stopped"
                )
            ):
                failures.append("invalid rejected operator Frontier qualification")
            if (
                qualification.get("status") == "unknown"
                and (
                    not isinstance(surface, dict)
                    or surface.get("qualification_status") != "unknown"
                    or surface.get("qualification_reason_code")
                    != qualification.get("reason_code")
                    or not isinstance(qualification.get("reason_code"), str)
                    or not qualification["reason_code"]
                    or surface.get("anchors") != []
                    or surface.get("cells") != []
                    or not isinstance(qualification.get("stopping_decision"), dict)
                    or qualification["stopping_decision"].get("status") != "stopped"
                )
            ):
                failures.append("invalid unknown operator Frontier qualification")
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
                phase_graph = surface.get("operator_phase_graph")
                if isinstance(phase_graph, dict):
                    replay, replay_failures = _softmax_source_replay(root, source_runs)
                    failures.extend(replay_failures)
                    phases = phase_graph.get("phases")
                    composition = phase_graph.get("composition")
                    expected_names = [item[0] for item in _SOFTMAX_PHASE_SPECS]
                    expected_capabilities = [item[2] for item in _SOFTMAX_PHASE_SPECS]
                    source_ids = {
                        item.get("run_id")
                        for item in source_runs
                        if isinstance(item, dict)
                    }
                    valid_phases = (
                        isinstance(phases, list)
                        and len(phases) == 5
                        and all(isinstance(phase, dict) for phase in phases)
                        and [phase.get("phase_name") for phase in phases]
                        == expected_names
                        and [
                            phase.get("required_capability_class")
                            for phase in phases
                        ]
                        == expected_capabilities
                        and [phase.get("predecessor_phase_ids") for phase in phases]
                        == [
                            [],
                            ["softmax-phase:max_reduce"],
                            ["softmax-phase:subtract"],
                            ["softmax-phase:exp"],
                            ["softmax-phase:sum_reduce"],
                        ]
                        and all(
                            phase.get("local_composition")
                            == "exact-operation-probe"
                            and isinstance(phase.get("candidate"), dict)
                            and isinstance(
                                phase["candidate"].get("candidate_digest"), str
                            )
                            and set(phase.get("source_run_ids", [])) <= source_ids
                            and len(phase.get("source_run_ids", [])) >= 2
                            and len(phase.get("source_digests", []))
                            == len(phase.get("source_run_ids", []))
                            for phase in phases
                        )
                    )
                    if valid_phases and qualification.get("status") == "qualified":
                        process_identities: set[tuple[object, object]] = set()
                        replay_domains: set[str] = set()
                        for phase_document, spec in zip(phases, _SOFTMAX_PHASE_SPECS):
                            phase_name = spec[0]
                            lanes = replay.get(phase_name, {})
                            search = lanes.get("search")
                            holdout = lanes.get("holdout")
                            if search is None or holdout is None:
                                valid_phases = False
                                continue
                            process_identities.update(
                                (tuple(search["process_identity"]), tuple(holdout["process_identity"]))
                            )
                            for item in (search, holdout):
                                replay_domains.add(
                                    _canonical_digest(
                                        {key: item[key] for key in (
                                            "shape", "axis", "dtype", "layout",
                                            "logical_device", "execution_mode", "cohort",
                                        )}
                                    )
                                )
                            expected_source_ids = [search["run_id"], holdout["run_id"]]
                            expected_source_digests = [
                                search["manifest_sha256"], holdout["manifest_sha256"]
                            ]
                            expected_uncertainty = math.hypot(
                                float(search["standard_uncertainty_ns"]),
                                float(holdout["standard_uncertainty_ns"]),
                            )
                            if not (
                                phase_document.get("source_run_ids") == expected_source_ids
                                and phase_document.get("source_digests") == expected_source_digests
                                and phase_document.get("selected_duration_ns") == holdout["median_ns"]
                                and phase_document.get("standard_uncertainty_ns") == expected_uncertainty
                                and phase_document["candidate"].get("candidate_id") == holdout["candidate_id"]
                                and phase_document["candidate"].get("candidate_family") == holdout["candidate_family"]
                                and phase_document["candidate"].get("candidate_digest") == holdout["candidate_digest"]
                                and phase_document.get("resource_demands") == {
                                    "exact_operation_invocations": 1,
                                    "declared_elements": math.prod(holdout["shape"]),
                                    "capability_class": spec[2],
                                }
                                and phase_document.get("input_roles") == _SOFTMAX_INPUT_ROLES[phase_name]
                                and phase_document.get("output_roles") == _SOFTMAX_OUTPUT_ROLES[phase_name]
                                and phase_document.get("assumptions") == [
                                    "fixed-shape-float32-contiguous",
                                    "candidate-invocation-includes-operand-data-movement",
                                    "no-fusion-no-chunk-pipeline-no-cross-phase-overlap",
                                ]
                                and phase_document.get("provenance") == {
                                    "semantic_ir_sha256": surface.get("source_demo", {}).get("semantic_ir_sha256"),
                                    "source_run_ids": expected_source_ids,
                                }
                            ):
                                valid_phases = False
                        if len(process_identities) != 10 or len(replay_domains) != 1:
                            valid_phases = False

                        source_demo = surface.get("source_demo")
                        stable_paths = phase_graph.get("stable_paths")
                        if isinstance(source_demo, dict):
                            demo_relative_path = source_demo.get("path")
                            demo_root = (
                                (root / demo_relative_path).resolve()
                                if isinstance(demo_relative_path, str)
                                and not Path(demo_relative_path).is_absolute()
                                else root
                            )
                            try:
                                demo_manifest_path = demo_root / "run.manifest.json"
                                demo_manifest = json.loads(demo_manifest_path.read_text())
                                semantic_entry = next(
                                    item for item in demo_manifest["artifacts"]
                                    if item.get("role") == "semantic-ir"
                                )
                                semantic_path = demo_root / semantic_entry["path"]
                                semantic = json.loads(semantic_path.read_text())
                                demo_valid = (
                                    verify_run_bundle(demo_root).get("passed") is True
                                    and source_demo.get("manifest_sha256") == _sha256(demo_manifest_path)
                                    and source_demo.get("semantic_ir_sha256") == _sha256(semantic_path)
                                    and source_demo.get("semantic_ir_path") == semantic_entry["path"]
                                    and stable_paths == _semantic_softmax_paths(semantic.get("root"))
                                )
                            except (OSError, KeyError, StopIteration, TypeError, json.JSONDecodeError):
                                demo_valid = False
                        else:
                            demo_valid = False
                        if not demo_valid:
                            valid_phases = False

                        session = surface.get("measurement_session")
                        session_valid = (
                            isinstance(session, dict)
                            and session.get("schema") == "groundupscale.dev/ascend-host-lock-session/v1alpha1"
                            and session.get("issue") == 44
                            and session.get("owner_start") == session.get("owner_end")
                            and "issue=44 " in str(session.get("owner_start"))
                            and session.get("device_visibility") == "0"
                            and session.get("hardware_cohort") == manifest.get("hardware_cohort")
                            and session.get("wrapper_sha256")
                            == "22d43618f1c616b2ff70570944c7447cd851aac98bfedb111b7912fc36b94787"
                            and str(session.get("started_at")) < str(session.get("ended_at"))
                        )
                        if not session_valid:
                            valid_phases = False
                    if not isinstance(composition, dict) or composition.get(
                        "rule"
                    ) != "serialized-critical-path-sum":
                        valid_phases = False
                    if qualification.get("status") == "qualified":
                        if valid_phases:
                            expected_duration = sum(
                                float(phase["selected_duration_ns"])
                                for phase in phases
                            )
                            expected_uncertainty = math.sqrt(
                                sum(
                                    float(phase["standard_uncertainty_ns"]) ** 2
                                    for phase in phases
                                )
                            )
                            composition_matches = (
                                composition.get("status") == "qualified"
                                and composition.get("operator_frontier_ns")
                                == expected_duration
                                and composition.get("standard_uncertainty_ns")
                                == expected_uncertainty
                                and composition.get("missing_evidence") == []
                                and phase_graph.get("fusion_contract") is None
                                and phase_graph.get("chunk_pipeline_contract")
                                is None
                            )
                        else:
                            composition_matches = False
                    else:
                        missing = (
                            composition.get("missing_evidence")
                            if isinstance(composition, dict)
                            else None
                        )
                        actual_missing = [
                            {
                                "phase_name": phase,
                                "required_capability_class": capability,
                                "reason_code": "missing-mandatory-phase-evidence",
                            }
                            for phase, _, capability in _SOFTMAX_PHASE_SPECS
                            if set(replay.get(phase, {})) != {"search", "holdout"}
                        ]
                        replay_records = [
                            record for lanes in replay.values() for record in lanes.values()
                        ]
                        replay_domains = {
                            _canonical_digest({key: record[key] for key in (
                                "shape", "axis", "dtype", "layout", "logical_device",
                                "execution_mode", "cohort",
                            )})
                            for record in replay_records
                        }
                        expected_reason = (
                            "mandatory-phase-domain-mismatch"
                            if len(replay_domains) > 1
                            else (
                                "missing-mandatory-phase-evidence"
                                if actual_missing
                                else "legacy-synthetic-operand-domain"
                            )
                        )
                        if expected_reason == "legacy-synthetic-operand-domain":
                            actual_missing = [
                                {
                                    "phase_name": phase,
                                    "required_capability_class": capability,
                                    "reason_code": "missing-real-chain-operand-evidence",
                                }
                                for phase, _, capability in _SOFTMAX_PHASE_SPECS
                                if phase in {"exp", "sum_reduce", "normalize"}
                            ]
                        composition_matches = bool(
                            isinstance(composition, dict)
                            and composition.get("status") == "unknown"
                            and composition.get("operator_frontier_ns") is None
                            and composition.get("standard_uncertainty_ns") is None
                            and isinstance(missing, list)
                            and missing == actual_missing
                            and qualification.get("reason_code") == expected_reason
                        )
                    if qualification.get("status") == "unknown":
                        valid_phases = phases == []
                    if not valid_phases or not composition_matches:
                        failures.append(
                            "Softmax Operator Frontier composition mismatch"
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
            dataset_members = (
                source_dataset.get("members")
                if isinstance(source_dataset, dict)
                and source_dataset.get("schema")
                == EVIDENCE_DATASET_SCHEMA
                else None
            )
            if isinstance(source_dataset, dict):
                dataset_body = {
                    key: value
                    for key, value in source_dataset.items()
                    if key != "dataset_digest"
                }
                archive = source_dataset.get("archive")
                archive_uri = (
                    archive.get("uri") if isinstance(archive, dict) else None
                )
                archive_sha256 = (
                    archive.get("sha256")
                    if isinstance(archive, dict)
                    else None
                )
                if (
                    source_dataset.get("dataset_digest")
                    != _canonical_digest(dataset_body)
                    or not isinstance(archive_uri, str)
                    or not archive_uri.startswith("https://github.com/")
                    or not isinstance(archive_sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", archive_sha256) is None
                    or archive_sha256 not in archive_uri
                    or not isinstance(dataset_members, list)
                    or not dataset_members
                    or len(
                        {
                            member.get("run_id")
                            for member in dataset_members
                            if isinstance(member, dict)
                        }
                    )
                    != len(dataset_members)
                    or any(
                        not isinstance(member, dict)
                        or not isinstance(member.get("run_id"), str)
                        or re.fullmatch(
                            r"[0-9a-f]{64}",
                            str(member.get("manifest_sha256")),
                        )
                        is None
                        or not isinstance(member.get("artifact_uri"), str)
                        or not member["artifact_uri"].startswith(
                            f"{archive_uri}#"
                        )
                        for member in dataset_members
                    )
                ):
                    failures.append("invalid content-addressed evidence dataset")
                qualification_dataset = qualification.get("source_dataset")
                dataset_artifact_ref = (
                    f"artifact://{paths_by_role['source-dataset-manifest']}"
                    if "source-dataset-manifest" in paths_by_role
                    else None
                )
                if (
                    not isinstance(qualification_dataset, dict)
                    or qualification_dataset
                    != {
                        "evidence_ref": dataset_artifact_ref,
                        "dataset_digest": source_dataset.get("dataset_digest"),
                        "archive_sha256": archive_sha256,
                    }
                    or diagnostic.get("source_dataset_ref")
                    != dataset_artifact_ref
                ):
                    failures.append(
                        "operator Frontier source dataset identity mismatch"
                    )
            dataset_member_index = (
                {
                    member.get("run_id"): member
                    for member in dataset_members
                    if isinstance(member, dict)
                    and isinstance(member.get("run_id"), str)
                }
                if isinstance(dataset_members, list)
                else {}
            )
            for source in source_runs:
                if not isinstance(source, dict):
                    failures.append("invalid operator Frontier source Run")
                    continue
                source_id = source.get("run_id")
                relative_path = source.get("path")
                artifact_uri = source.get("artifact_uri")
                manifest_digest = source.get("manifest_sha256")
                if (
                    not isinstance(source_id, str)
                    or not source_id
                    or source_id in seen_source_ids
                    or not isinstance(manifest_digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", manifest_digest) is None
                ):
                    failures.append("invalid operator Frontier source Run")
                    continue
                seen_source_ids.add(source_id)
                if artifact_uri is not None:
                    member = dataset_member_index.get(source_id)
                    if (
                        relative_path is not None
                        or not isinstance(artifact_uri, str)
                        or not artifact_uri.startswith("https://github.com/")
                        or not isinstance(member, dict)
                        or member.get("manifest_sha256") != manifest_digest
                        or member.get("artifact_uri") != artifact_uri
                    ):
                        failures.append(
                            f"content-addressed source lineage mismatch: {source_id}"
                        )
                    continue
                if (
                    not isinstance(relative_path, str)
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

    if transformer_matmul_frontier:
        inventory = documents_by_role.get("matmul-domain-inventory")
        qualification = documents_by_role.get(
            "transformer-matmul-frontier-qualification"
        )
        identity_matches = (
            isinstance(inventory, dict)
            and isinstance(qualification, dict)
            and manifest.get("status") == qualification.get("status")
            and manifest.get("hardware_cohort")
            == qualification.get("hardware_cohort")
            == inventory.get("source_hardware_cohort")
        )
        if not identity_matches:
            failures.append("Transformer MatMul Frontier identity mismatch")
        try:
            from groundupscale.transformer_matmul_frontier import (
                verify_transformer_matmul_frontier_derivation,
            )

            derivation_matches = (
                identity_matches
                and verify_transformer_matmul_frontier_derivation(
                    root, manifest, inventory, qualification
                )
            )
        except (KeyError, OSError, TypeError, ValueError):
            derivation_matches = False
        if not derivation_matches:
            failures.append("Transformer MatMul Frontier derivation mismatch")

    if transformer_matmul_exact_anchor:
        anchor = documents_by_role.get("transformer-matmul-exact-anchor")
        identity_matches = (
            isinstance(anchor, dict)
            and manifest.get("status") == anchor.get("status")
            and anchor.get("status") in {"qualified", "unknown"}
            and manifest.get("device") == "ascend-npu"
            and manifest.get("hardware_cohort")
            == anchor.get("hardware_cohort")
        )
        if not identity_matches:
            failures.append("Transformer MatMul exact Anchor identity mismatch")
        try:
            from groundupscale.transformer_matmul_frontier import (
                verify_transformer_matmul_exact_anchor_derivation,
            )

            derivation_matches = (
                identity_matches
                and verify_transformer_matmul_exact_anchor_derivation(
                    root, manifest, anchor
                )
            )
        except (KeyError, OSError, TypeError, ValueError):
            derivation_matches = False
        if not derivation_matches:
            failures.append("Transformer MatMul exact Anchor derivation mismatch")

    if transformer_matmul_surface:
        surface = documents_by_role.get("transformer-matmul-surface")
        identity_matches = (
            isinstance(surface, dict)
            and manifest.get("status") == surface.get("status") == "qualified"
            and manifest.get("device") == "ascend-npu"
            and manifest.get("hardware_cohort") == surface.get("hardware_cohort")
        )
        if not identity_matches:
            failures.append("Transformer MatMul Surface identity mismatch")
        try:
            from groundupscale.transformer_matmul_frontier import (
                verify_transformer_matmul_surface_derivation,
            )

            derivation_matches = (
                identity_matches
                and verify_transformer_matmul_surface_derivation(
                    root, manifest, surface
                )
            )
        except (KeyError, OSError, TypeError, ValueError):
            derivation_matches = False
        if not derivation_matches:
            failures.append("Transformer MatMul Surface derivation mismatch")

    if compound_operator_frontier:
        graph = documents_by_role.get("operator-phase-graph")
        qualification = documents_by_role.get(
            "compound-operator-frontier-qualification"
        )
        diagnostic = documents_by_role.get("compound-operator-diagnostic")
        if (
            manifest.get("status") != "completed"
            or manifest.get("device") != "ascend-npu"
            or manifest.get("operation") != "RMSNorm"
            or not isinstance(graph, dict)
            or not isinstance(qualification, dict)
            or not isinstance(diagnostic, dict)
            or qualification.get("operation") != "RMSNorm"
            or qualification.get("hardware_cohort")
            != manifest.get("hardware_cohort")
            or qualification.get("stable_path") != manifest.get("stable_path")
            or qualification.get("compilation_fingerprint")
            != manifest.get("compilation_fingerprint")
            or not isinstance(manifest.get("compilation_fingerprint"), str)
            or not manifest.get("compilation_fingerprint")
            or graph.get("compilation_fingerprint")
            != manifest.get("compilation_fingerprint")
            or manifest.get("source_runs") != qualification.get("source_runs")
            or diagnostic.get("operation") != "RMSNorm"
            or diagnostic.get("hardware_cohort")
            != manifest.get("hardware_cohort")
            or diagnostic.get("stable_path") != manifest.get("stable_path")
        ):
            failures.append("invalid compound operator Frontier bundle identity")
        else:
            for label, document in (
                ("phase graph", graph),
                ("qualification", qualification),
                ("diagnostic", diagnostic),
            ):
                expected_digest = document.get("input_digest")
                body = {
                    key: value
                    for key, value in document.items()
                    if key != "input_digest"
                }
                if expected_digest != _canonical_digest(body):
                    failures.append(
                        f"compound operator Frontier {label} digest mismatch"
                    )
            phase_evidence = qualification.get("phase_evidence")
            source_runs = qualification.get("source_runs")
            supersedes = manifest.get("supersedes")
            if supersedes is not None:
                superseded_path = (
                    supersedes.get("path")
                    if isinstance(supersedes, dict)
                    else None
                )
                superseded_root = (
                    (root / superseded_path).resolve()
                    if isinstance(superseded_path, str)
                    and not Path(superseded_path).is_absolute()
                    else None
                )
                superseded_manifest_path = (
                    superseded_root / "run.manifest.json"
                    if superseded_root is not None
                    else None
                )
                superseded_manifest = (
                    json.loads(
                        superseded_manifest_path.read_text(encoding="utf-8")
                    )
                    if superseded_manifest_path is not None
                    and superseded_manifest_path.is_file()
                    else None
                )
                if (
                    not isinstance(supersedes, dict)
                    or qualification.get("supersedes") != supersedes
                    or not isinstance(superseded_manifest, dict)
                    or superseded_manifest.get("run_id")
                    != supersedes.get("run_id")
                    or superseded_manifest.get("bundle_kind")
                    != "compound-operator-frontier"
                    or _sha256(superseded_manifest_path)
                    != supersedes.get("manifest_sha256")
                ):
                    failures.append(
                        "compound operator Frontier superseded Run mismatch"
                    )
            if (
                not isinstance(source_runs, list)
                or qualification.get("source_evidence_digest")
                != _canonical_digest(source_runs)
            ):
                failures.append(
                    "compound operator Frontier source evidence digest mismatch"
                )
            if (
                qualification.get("phase_graph_ref")
                != f"artifact://{paths_by_role.get('operator-phase-graph')}"
                or qualification.get("phase_graph_digest")
                != graph.get("input_digest")
            ):
                failures.append(
                    "compound operator Frontier phase graph identity mismatch"
                )
            if (
                diagnostic.get("qualification_ref")
                != (
                    "artifact://"
                    + str(
                        paths_by_role.get(
                            "compound-operator-frontier-qualification"
                        )
                    )
                )
                or diagnostic.get("qualification_digest")
                != qualification.get("input_digest")
                or diagnostic.get("operator_frontier")
                != qualification.get("operator_frontier")
                or diagnostic.get("missing_evidence")
                != qualification.get("missing_evidence")
            ):
                failures.append(
                    "compound operator Frontier diagnostic identity mismatch"
                )
            phases = graph.get("phases")
            candidate = qualification.get("selected_candidate")
            schedule = (
                candidate.get("phase_schedule")
                if isinstance(candidate, dict)
                else None
            )
            scheduled_phases = (
                schedule.get("phases") if isinstance(schedule, dict) else None
            )
            frontier = qualification.get("operator_frontier")
            missing = qualification.get("missing_evidence")
            graph_phase_ids = (
                [phase.get("phase_id") for phase in phases]
                if isinstance(phases, list)
                and all(isinstance(phase, dict) for phase in phases)
                else []
            )
            schedule_phase_ids = (
                [phase.get("phase_id") for phase in scheduled_phases]
                if isinstance(scheduled_phases, list)
                and all(isinstance(phase, dict) for phase in scheduled_phases)
                else []
            )
            evidence_by_phase = (
                {
                    evidence.get("phase_id"): evidence
                    for evidence in phase_evidence
                    if isinstance(evidence, dict)
                    and isinstance(evidence.get("phase_id"), str)
                }
                if isinstance(phase_evidence, list)
                else {}
            )
            scheduled_by_phase = (
                {
                    phase.get("phase_id"): phase
                    for phase in scheduled_phases
                    if isinstance(phase, dict)
                    and isinstance(phase.get("phase_id"), str)
                }
                if isinstance(scheduled_phases, list)
                else {}
            )
            phase_evidence_mismatch = False
            if len(evidence_by_phase) != len(phase_evidence or []):
                phase_evidence_mismatch = True
            else:
                for phase_id, evidence in evidence_by_phase.items():
                    phase = scheduled_by_phase.get(phase_id)
                    evidence_body = {
                        key: value
                        for key, value in evidence.items()
                        if key != "input_digest"
                    }
                    constraints = evidence.get("constraints")
                    exact = constraints.get("exact_operation_duration_ns") if isinstance(constraints, dict) else None
                    matching = constraints.get("matching_compute_capability_duration_ns") if isinstance(constraints, dict) else None
                    memory = constraints.get("memory_pattern_floor_ns") if isinstance(constraints, dict) else None
                    compute_or_exact = exact if exact is not None else matching
                    if (
                        evidence.get("input_digest")
                        != content_fingerprint(evidence_body)
                        or not isinstance(phase, dict)
                        or phase.get("phase_name") != evidence.get("phase_name")
                        or phase.get("operation_class")
                        != evidence.get("operation_class")
                        or phase.get("candidate") != evidence.get("candidate")
                        or phase.get("constraints") != constraints
                        or phase.get("capability_profile_refs")
                        != evidence.get("capability_profile_refs")
                        or phase.get("local_duration_ns")
                        != evidence.get("local_duration_ns")
                        or phase.get("standard_uncertainty_ns")
                        != evidence.get("standard_uncertainty_ns")
                        or not isinstance(compute_or_exact, (int, float))
                        or not isinstance(memory, (int, float))
                        or evidence.get("local_duration_ns")
                        != max(float(compute_or_exact), float(memory))
                        or phase.get("resource_composition")
                        != "max(compute-or-exact,memory-pattern-floor)"
                        or not isinstance(phase.get("evidence_refs"), list)
                        or len(phase["evidence_refs"]) != 2
                    ):
                        phase_evidence_mismatch = True
                        break
            if phase_evidence_mismatch:
                failures.append(
                    "compound operator Frontier phase evidence mismatch"
                )
            graph_by_phase = (
                {
                    phase.get("phase_id"): phase
                    for phase in phases
                    if isinstance(phase, dict)
                    and isinstance(phase.get("phase_id"), str)
                }
                if isinstance(phases, list)
                else {}
            )
            graph_schedule_mismatch = any(
                not isinstance(scheduled_by_phase.get(phase_id), dict)
                or any(
                    scheduled_by_phase[phase_id].get(field)
                    != graph_phase.get(field)
                    for field in (
                        "phase_name",
                        "operation_class",
                        "compute_capability_resource",
                        "memory_capability_resource",
                        "predecessor_phase_ids",
                        "input_roles",
                        "output_roles",
                        "minimum_flops",
                        "logical_read_bytes",
                        "logical_write_bytes",
                    )
                )
                for phase_id, graph_phase in graph_by_phase.items()
            )
            source_lane_keys = {
                (source.get("phase_id"), source.get("lane"))
                for source in source_runs
                if isinstance(source, dict)
            } if isinstance(source_runs, list) else set()
            expected_missing = [
                {
                    "phase_id": phase_id,
                    "phase_name": graph_phase.get("phase_name"),
                    "operation_class": graph_phase.get("operation_class"),
                    "required_evidence": (
                        "verified search and independent-holdout Run Bundles with "
                        "replayable compute-or-exact and memory-pattern capability "
                        "profiles matching this phase"
                    ),
                }
                for phase_id, graph_phase in graph_by_phase.items()
                if (phase_id, "search") not in source_lane_keys
                or (phase_id, "independent-holdout") not in source_lane_keys
            ]
            expected_missing_ids = {
                item["phase_id"] for item in expected_missing
            }
            unknown_schedule_mismatch = any(
                not isinstance(scheduled_by_phase.get(phase_id), dict)
                or scheduled_by_phase[phase_id].get("status") != "unknown"
                or scheduled_by_phase[phase_id].get("candidate") is not None
                or scheduled_by_phase[phase_id].get("constraints") is not None
                or scheduled_by_phase[phase_id].get("capability_profile_refs")
                is not None
                or scheduled_by_phase[phase_id].get("local_duration_ns") is not None
                or scheduled_by_phase[phase_id].get("standard_uncertainty_ns")
                is not None
                or scheduled_by_phase[phase_id].get("resource_composition")
                != "unknown"
                or scheduled_by_phase[phase_id].get("evidence_refs") != []
                for phase_id in expected_missing_ids
            )
            if (
                missing != expected_missing
                or unknown_schedule_mismatch
                or set(evidence_by_phase) != set(graph_by_phase) - expected_missing_ids
            ):
                failures.append(
                    "compound operator Frontier missing evidence mismatch"
                )
            known_durations = (
                [phase.get("local_duration_ns") for phase in scheduled_phases]
                if isinstance(scheduled_phases, list)
                and all(isinstance(phase, dict) for phase in scheduled_phases)
                else []
            )
            known_uncertainties = (
                [
                    phase.get("standard_uncertainty_ns")
                    for phase in scheduled_phases
                ]
                if isinstance(scheduled_phases, list)
                and all(isinstance(phase, dict) for phase in scheduled_phases)
                else []
            )
            complete = bool(
                not expected_missing
                and missing == expected_missing
                and known_durations
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and value > 0
                    for value in known_durations
                )
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and value >= 0
                    for value in known_uncertainties
                )
            )
            expected_duration = (
                None
            )
            topological_order: list[str] = []
            if complete:
                completed: set[str] = set()
                longest: dict[str, float] = {}
                phase_durations = {
                    str(phase["phase_id"]): float(phase["local_duration_ns"])
                    for phase in scheduled_phases
                }
                while len(completed) < len(graph_by_phase):
                    ready = sorted(
                        phase_id
                        for phase_id, phase in graph_by_phase.items()
                        if phase_id not in completed
                        and set(phase.get("predecessor_phase_ids", [])) <= completed
                    )
                    if not ready:
                        break
                    for phase_id in ready:
                        predecessors = graph_by_phase[phase_id].get(
                            "predecessor_phase_ids", []
                        )
                        longest[phase_id] = phase_durations[phase_id] + max(
                            (longest[str(item)] for item in predecessors),
                            default=0.0,
                        )
                        completed.add(phase_id)
                        topological_order.append(phase_id)
                predecessor_ids = {
                    str(item)
                    for phase in graph_by_phase.values()
                    for item in phase.get("predecessor_phase_ids", [])
                }
                sinks = set(graph_by_phase) - predecessor_ids
                declared_outputs = set(graph.get("output_phase_ids", []))
                if len(completed) == len(graph_by_phase) and sinks == declared_outputs:
                    expected_duration = max(longest[phase_id] for phase_id in sinks)
            expected_uncertainty = (
                math.sqrt(
                    sum(float(value) ** 2 for value in known_uncertainties)
                )
                if complete
                else None
            )
            expected_status = "known" if complete else "unknown"
            if (
                graph_phase_ids != schedule_phase_ids
                or graph_schedule_mismatch
                or len(graph_phase_ids) != 7
                or not isinstance(schedule, dict)
                or schedule.get("policy") != "dependency-critical-path-no-chunk"
                or schedule.get("chunk_pipeline_contract_id") is not None
                or schedule.get("overlap_evidence_refs") != []
                or schedule.get("topological_phase_ids") != topological_order
                or schedule.get("serialized_duration_ns")
                != (sum(float(value) for value in known_durations) if complete else None)
                or schedule.get("critical_path_duration_ns") != expected_duration
                or schedule.get("selected_duration_ns") != expected_duration
                or not isinstance(frontier, dict)
                or frontier.get("status") != expected_status
                or frontier.get("duration_ns") != expected_duration
                or frontier.get("standard_uncertainty_ns")
                != expected_uncertainty
                or frontier.get("composition_policy")
                != "dependency-critical-path-no-chunk"
                or frontier.get("formula")
                != "max_path(sum(phase.local_duration_ns))"
                or qualification.get("status")
                != ("qualified" if complete else "unknown")
            ):
                failures.append(
                    "compound operator Frontier phase schedule mismatch"
                )

            if isinstance(source_runs, list):
                seen_lanes: set[tuple[object, object]] = set()
                source_observations: dict[tuple[object, object], dict[str, object]] = {}
                for source in source_runs:
                    if not isinstance(source, dict):
                        failures.append("invalid compound operator Frontier source Run")
                        continue
                    source_id = source.get("run_id")
                    source_path = source.get("path")
                    key = (source.get("phase_id"), source.get("lane"))
                    if (
                        key in seen_lanes
                        or source.get("lane") not in {"search", "independent-holdout"}
                        or not isinstance(source_id, str)
                        or not isinstance(source_path, str)
                        or Path(source_path).is_absolute()
                    ):
                        failures.append("invalid compound operator Frontier source Run")
                        continue
                    seen_lanes.add(key)
                    source_root = (root / source_path).resolve()
                    source_manifest_path = source_root / "run.manifest.json"
                    if (
                        not source_manifest_path.is_file()
                        or _sha256(source_manifest_path)
                        != source.get("manifest_sha256")
                        or verify_run_bundle(source_root).get("passed") is not True
                    ):
                        failures.append(
                            f"compound operator Frontier source Run failed verification: {source_id}"
                        )
                        continue
                    source_manifest = json.loads(
                        source_manifest_path.read_text(encoding="utf-8")
                    )
                    if (
                        source_manifest.get("run_id") != source_id
                        or source_manifest.get("bundle_kind")
                        != "operator-phase-measurement"
                        or source_manifest.get("hardware_cohort")
                        != manifest.get("hardware_cohort")
                        or source_manifest.get("phase_id") != source.get("phase_id")
                        or source_manifest.get("lane") != source.get("lane")
                        or source_manifest.get("compilation_fingerprint")
                        != manifest.get("compilation_fingerprint")
                        or source.get("compilation_fingerprint")
                        != manifest.get("compilation_fingerprint")
                    ):
                        failures.append(
                            f"compound operator Frontier source Run identity mismatch: {source_id}"
                        )
                        continue
                    source_artifacts = source_manifest.get("artifacts")
                    observation_artifact = next(
                        (
                            item
                            for item in source_artifacts
                            if isinstance(item, dict)
                            and item.get("role")
                            == "operator-phase-capability-observation"
                        ),
                        None,
                    ) if isinstance(source_artifacts, list) else None
                    observation_path = (
                        source_root / str(observation_artifact.get("path"))
                    ).resolve() if isinstance(observation_artifact, dict) else None
                    if (
                        observation_path is None
                        or source_root not in observation_path.parents
                        or not observation_path.is_file()
                    ):
                        failures.append(
                            f"compound operator Frontier source observation mismatch: {source_id}"
                        )
                        continue
                    source_observation = json.loads(
                        observation_path.read_text(encoding="utf-8")
                    )
                    if (
                        source.get("candidate")
                        != source_observation.get("candidate")
                        or source.get("phase_id")
                        != source_observation.get("phase_id")
                        or source.get("lane") != source_observation.get("lane")
                        or source.get("compilation_fingerprint")
                        != source_observation.get("compilation_fingerprint")
                    ):
                        failures.append(
                            f"compound operator Frontier source observation mismatch: {source_id}"
                        )
                        continue
                    source_observations[key] = source_observation

                for phase_id, evidence in evidence_by_phase.items():
                    holdout = source_observations.get(
                        (phase_id, "independent-holdout")
                    )
                    search = source_observations.get((phase_id, "search"))
                    scheduled_phase = scheduled_by_phase.get(phase_id)
                    matching_sources = [
                        source
                        for source in source_runs
                        if isinstance(source, dict)
                        and source.get("phase_id") == phase_id
                    ]
                    refs_by_lane = {
                        source.get("lane"): (
                            f"run-bundle://{source.get('run_id')}"
                            "#artifact://observation/phase-capability.json"
                        )
                        for source in matching_sources
                    }
                    evidence_body = (
                        {
                            key: value
                            for key, value in holdout.items()
                            if key != "input_digest"
                        }
                        if isinstance(holdout, dict)
                        else None
                    )
                    source_evidence = (
                        {**evidence_body, "input_digest": holdout.get("input_digest")}
                        if isinstance(evidence_body, dict)
                        and isinstance(holdout, dict)
                        else None
                    )
                    if (
                        not isinstance(search, dict)
                        or not isinstance(holdout, dict)
                        or evidence != source_evidence
                        or search.get("candidate") != holdout.get("candidate")
                        or not isinstance(scheduled_phase, dict)
                        or scheduled_phase.get("evidence_refs")
                        != [
                            refs_by_lane.get("search"),
                            refs_by_lane.get("independent-holdout"),
                        ]
                    ):
                        failures.append(
                            f"compound operator Frontier source observation mismatch: {phase_id}"
                        )

    if operator_phase_measurement:
        observation = documents_by_role.get(
            "operator-phase-capability-observation"
        )
        if (
            manifest.get("status") != "completed"
            or manifest.get("device") != "ascend-npu"
            or manifest.get("operation") != "RMSNorm"
            or not isinstance(observation, dict)
            or observation.get("phase_id") != manifest.get("phase_id")
            or observation.get("phase_name") != manifest.get("phase_name")
            or observation.get("lane") != manifest.get("lane")
            or not isinstance(observation.get("execution_domain"), dict)
            or observation["execution_domain"].get("hardware_cohort")
            != manifest.get("hardware_cohort")
            or observation.get("correctness") != "passed"
            or observation.get("timing_quality") != "passed"
            or observation.get("compilation_fingerprint")
            != manifest.get("compilation_fingerprint")
            or not isinstance(manifest.get("compilation_fingerprint"), str)
            or not manifest.get("compilation_fingerprint")
        ):
            failures.append("invalid operator phase measurement identity")
        else:
            body = {
                key: value
                for key, value in observation.items()
                if key != "input_digest"
            }
            constraints = observation.get("constraints")
            capability_refs = observation.get("capability_profile_refs")
            raw_by_constraint = observation.get("raw_samples_by_constraint")
            constraint_profiles = observation.get("constraint_profiles")
            exact = constraints.get("exact_operation_duration_ns") if isinstance(constraints, dict) else None
            matching = constraints.get("matching_compute_capability_duration_ns") if isinstance(constraints, dict) else None
            memory = constraints.get("memory_pattern_floor_ns") if isinstance(constraints, dict) else None
            compute_or_exact = exact if exact is not None else matching
            if (
                observation.get("input_digest") != content_fingerprint(body)
                or not isinstance(compute_or_exact, (int, float))
                or not isinstance(memory, (int, float))
                or observation.get("local_duration_ns")
                != max(float(compute_or_exact), float(memory))
                or observation.get("resource_composition")
                != "max(compute-or-exact,memory-pattern-floor)"
                or not isinstance(capability_refs, dict)
                or capability_refs
                != {
                    "compute_or_exact": (
                        "artifact://observation/phase-capability.json"
                        "#constraint_profiles.compute_or_exact"
                    ),
                    "memory_pattern": (
                        "artifact://observation/phase-capability.json"
                        "#constraint_profiles.memory_pattern"
                    ),
                }
                or not isinstance(raw_by_constraint, dict)
                or set(raw_by_constraint) != {"compute_or_exact", "memory_pattern"}
                or not all(
                    isinstance(samples, list)
                    and len(samples) >= 3
                    and all(
                        isinstance(value, (int, float))
                        and math.isfinite(float(value))
                        and value > 0
                        for value in samples
                    )
                    for samples in raw_by_constraint.values()
                )
                or not isinstance(constraint_profiles, dict)
                or set(constraint_profiles) != {"compute_or_exact", "memory_pattern"}
                or not all(
                    isinstance(profile, dict)
                    for profile in constraint_profiles.values()
                )
                or constraint_profiles["compute_or_exact"].get(
                    "measurement_policy"
                ) != "median-ns-v1"
                or constraint_profiles["memory_pattern"].get(
                    "measurement_policy"
                ) != "median-ns-v1"
                or constraint_profiles["compute_or_exact"].get("probe_kind")
                != observation.get("evidence_kind")
                or constraint_profiles["memory_pattern"].get("probe_kind")
                != "memory-pattern-probe"
                or constraint_profiles["compute_or_exact"].get(
                    "capability_resource"
                )
                != (
                    observation.get("operation_class")
                    if observation.get("evidence_kind") == "exact-operation-probe"
                    else observation.get("required_compute_capability")
                )
                or constraint_profiles["memory_pattern"].get(
                    "capability_resource"
                ) != observation.get("required_memory_capability")
                or constraint_profiles["compute_or_exact"].get("samples_ns")
                != raw_by_constraint.get("compute_or_exact")
                or constraint_profiles["memory_pattern"].get("samples_ns")
                != raw_by_constraint.get("memory_pattern")
                or constraint_profiles["compute_or_exact"].get("summary_ns")
                != compute_or_exact
                or constraint_profiles["memory_pattern"].get("summary_ns")
                != memory
                or statistics.median(
                    constraint_profiles["compute_or_exact"].get("samples_ns", [])
                ) != compute_or_exact
                or statistics.median(
                    constraint_profiles["memory_pattern"].get("samples_ns", [])
                ) != memory
            ):
                failures.append("invalid operator phase measurement composition")

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
            aggregate_samples = (
                raw_timing.get("aggregate_samples_ns")
                if raw_timing is not None
                else None
            )
            normalization = (
                raw_timing.get("normalization")
                if raw_timing is not None
                else None
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
            if aggregate_samples is not None or normalization is not None:
                divisor = (
                    normalization.get("divisor")
                    if isinstance(normalization, dict)
                    else None
                )
                raw_timing_valid = bool(
                    raw_timing_valid
                    and isinstance(aggregate_samples, list)
                    and len(aggregate_samples) == len(samples)
                    and all(
                        isinstance(sample, int)
                        and not isinstance(sample, bool)
                        and sample > 0
                        for sample in aggregate_samples
                    )
                    and isinstance(divisor, int)
                    and not isinstance(divisor, bool)
                    and timing_plan is not None
                    and divisor == timing_plan.get("inner_iterations")
                    and normalization.get("rounding")
                    == "round-half-to-even-nanoseconds"
                    and normalization.get("aggregate_unit") == "nanoseconds"
                    and samples
                    == [round(sample / divisor) for sample in aggregate_samples]
                )
            elif (
                contract is not None
                and contract.get("operation") == "SoftmaxPhase"
                and isinstance(contract.get("inner_iterations"), int)
                and contract["inner_iterations"] > 1
                and _sha256(manifest_path)
                not in _SOFTMAX_LEGACY_NORMALIZATION_MANIFESTS
            ):
                raw_timing_valid = False
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
