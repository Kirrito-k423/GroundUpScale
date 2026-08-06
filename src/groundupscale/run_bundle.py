"""Atomic, immutable, self-describing local Run Bundle writer."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import tempfile
from typing import Any

import torch

from groundupscale.benchmark import (
    BenchmarkRunner,
    ReferenceRunner,
    TraceRunner,
    observe_tensor_storage_peak,
)
from groundupscale.benchmark.explanation import (
    build_explanation_graph,
    render_report_html,
)
from groundupscale.benchmark.prediction import predict_live_set
from groundupscale.ir import canonical_data
from groundupscale.pipeline import CompiledAnalysis


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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


def _default_run_id(device: str, fingerprint: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{device}-{fingerprint[:8]}"


class RunBundleExistsError(FileExistsError):
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
    ) -> Path:
        benchmark_runner = BenchmarkRunner(self.compiled.bundle, seed=self.seed)
        device = benchmark_runner.device
        selected_run_id = run_id or _default_run_id(
            device, self.compiled.cost.compilation_fingerprint
        )
        if not RUN_ID_PATTERN.fullmatch(selected_run_id):
            raise ValueError(f"unsafe run_id: {selected_run_id!r}")
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
            explanation = build_explanation_graph(
                self.compiled.cost.cost_ir, benchmark, trace, live_set
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
            prediction = {
                "schema": "groundupscale.dev/prediction/v1alpha1",
                "cost_summary": self.compiled.cost.cost_ir.summary,
                "live_set": live_set,
                "duration_status": "uncalibrated",
            }
            write_json("prediction", "prediction/metrics.json", prediction, prediction["schema"], ("cost-ir",))
            write_json("explanation-graph", "prediction/explanation.graph.json", explanation, explanation["schema"], ("prediction", "benchmark-observation", "alignment-map"))
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
            write_json("correctness-observation", "observation/correctness.json", correctness, correctness["schema"], ("resolved-input-lock", "environment"))
            write_json("error-attribution", "comparison/error-attribution.json", trace["error_attribution"], trace["error_attribution"]["schema"], ("benchmark-observation", "alignment-map"))
            report = render_report_html(
                run_id=selected_run_id,
                device=device,
                benchmark=benchmark,
                trace=trace,
                live_set=live_set,
                explanation=explanation,
            )
            write_bytes("html-report", "reports/report.html", report.encode("utf-8"), media_type="text/html", schema="groundupscale.dev/html-report/v1alpha1", inputs=("explanation-graph",))

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
                "hardware_cohort": f"{hardware_names}-{platform.release()}-torch{torch.__version__}-{device}",
                "device": device,
                "seed": self.seed,
                "stages": {
                    "compilation": "completed",
                    "structural_prediction": "completed",
                    "duration_prediction": "skipped-uncalibrated",
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
    for artifact in manifest["artifacts"]:
        artifact_path = (root / artifact["path"]).resolve()
        if root not in artifact_path.parents:
            failures.append(f"path escapes bundle: {artifact['path']}")
        elif not artifact_path.is_file():
            failures.append(f"missing artifact: {artifact['path']}")
        elif _sha256(artifact_path) != artifact["sha256"]:
            failures.append(f"digest mismatch: {artifact['path']}")
    return {
        "schema": "groundupscale.dev/run-verification/v1alpha1",
        "run_id": manifest["run_id"],
        "passed": not failures,
        "artifact_count": len(manifest["artifacts"]),
        "failures": failures,
    }


__all__ = [
    "RunBundleExistsError",
    "RunBundleWriter",
    "verify_run_bundle",
]
