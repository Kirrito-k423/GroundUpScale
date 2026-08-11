from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Mapping
import warnings

import pytest
import torch
from torch import Tensor, nn

from groundupscale.cli import main
from groundupscale.execution_runtime import execute_with_npu_cpu_fallback_guard
from groundupscale.pipeline import compile_analysis_plan
from groundupscale.run_bundle import verify_run_bundle


REPOSITORY_ROOT = Path(__file__).parents[1]
ASCEND_DEMO_PLAN = REPOSITORY_ROOT / "specs/plans/ascend-npu-transformer-demo.yaml"
ASCEND_PROFILE_COHORT = "ascend-npu-23b93a89d5fecc79"


def _semantic_operations(compiled: object) -> list[tuple[str, str]]:
    semantic = compiled.semantic.semantic_ir  # type: ignore[attr-defined]
    return [
        (operation.stable_path, operation.operation)
        for operation in semantic.walk_operations()
    ]


def _benchmark_cases(compiled: object) -> list[tuple[str, str, str]]:
    bundle = compiled.bundle  # type: ignore[attr-defined]
    return [
        (case.id, case.scope, case.mode)
        for document in bundle.benchmark_cases
        for case in document.spec.cases
    ]


def test_ascend_demo_reuses_m4_logical_specs_shape_and_benchmark_cases() -> None:
    mac = compile_analysis_plan(
        REPOSITORY_ROOT,
        REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml",
    )
    ascend = compile_analysis_plan(
        REPOSITORY_ROOT,
        ASCEND_DEMO_PLAN,
    )

    for shared_path in (
        "specs/models/two-layer-transformer.yaml",
        "specs/workloads/prefill.yaml",
        "specs/analysis-cases/fixed-prefill.yaml",
        "specs/benchmarks/core-prefill.yaml",
    ):
        assert ascend.bundle.sources[shared_path] == mac.bundle.sources[shared_path]

    assert ascend.bundle.analysis_case.spec.shape == mac.bundle.analysis_case.spec.shape
    assert _semantic_operations(ascend) == _semantic_operations(mac)
    assert len(_semantic_operations(ascend)) == 52
    assert _benchmark_cases(ascend) == _benchmark_cases(mac)
    semantic_operation_kinds = {
        operation for _stable_path, operation in _semantic_operations(ascend)
    }
    hardware_capabilities = ascend.bundle.hardware[0].spec.devices[0].capabilities
    assert hardware_capabilities is not None
    assert set(hardware_capabilities.supported_operations) == semantic_operation_kinds
    assert ascend.hardware_prediction is not None
    assert {candidate.operation for candidate in ascend.hardware_prediction.candidates} == {
        "MatMul"
    }


class _UnavailableAscendAdapter:
    def discover_capabilities(self) -> Mapping[str, object]:
        return {
            "schema": "groundupscale.dev/measurement-capability-manifest/v1alpha1",
            "status": "blocked",
            "device": "ascend-npu",
            "logical_device": "npu:0",
            "reason_codes": ["torch-npu-unavailable"],
        }

    def fingerprint_cohort(self) -> Mapping[str, object]:
        return {
            "schema": "groundupscale.dev/hardware-cohort/v1alpha1",
            "status": "blocked",
            "device": "ascend-npu",
            "logical_device": "npu:0",
            "reason_codes": ["torch-npu-unavailable"],
        }

    def preflight(self) -> Mapping[str, object]:
        return {
            "schema": "groundupscale.dev/measurement-preflight/v1alpha1",
            "status": "blocked",
            "eligible": False,
            "device": "ascend-npu",
            "logical_device": "npu:0",
            "reason_codes": ["torch-npu-unavailable"],
        }


def test_ascend_demo_preserves_a_verified_blocked_bundle_when_npu_is_unavailable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "run",
            str(ASCEND_DEMO_PLAN),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--artifact-store",
            str(tmp_path),
            "--run-id",
            "unavailable-npu-demo",
            "--json",
        ],
        measurement_adapter_factory=lambda *_args, **_kwargs: (
            _UnavailableAscendAdapter()
        ),
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    run = Path(summary["run_bundle"])
    manifest = json.loads((run / "run.manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_kind"] == "transformer-demo"
    assert manifest["status"] == "blocked"
    assert manifest["device"] == "npu:0"
    assert manifest["stages"]["compatibility"] == "failed"
    assert manifest["reason_codes"] == ["torch-npu-unavailable"]
    assert {
        "resolved-input-lock",
        "measurement-capability-manifest",
        "hardware-cohort",
        "measurement-preflight",
        "execution-failure",
    } <= {artifact["role"] for artifact in manifest["artifacts"]}
    failure = json.loads(
        (run / "observation/execution-failure.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "compatibility-failed"
    assert failure["reason_codes"] == ["torch-npu-unavailable"]
    assert verify_run_bundle(run)["passed"] is True


class _AvailableAscendAdapter(_UnavailableAscendAdapter):
    def discover_capabilities(self) -> Mapping[str, object]:
        return {
            "schema": "groundupscale.dev/measurement-capability-manifest/v1alpha1",
            "status": "completed",
            "device": "ascend-npu",
            "logical_device": "npu:0",
            "supported_operations": [
                "MatMul",
                "Add",
                "RMSNorm",
                "Softmax",
                "SiLU",
                "Mul",
                "View",
                "Transpose",
            ],
        }

    def fingerprint_cohort(self) -> Mapping[str, object]:
        return {
            "schema": "groundupscale.dev/hardware-cohort/v1alpha1",
            "status": "completed",
            "device": "ascend-npu",
            "logical_device": "npu:0",
            "cohort_id": ASCEND_PROFILE_COHORT,
        }

    def preflight(self) -> Mapping[str, object]:
        return {
            "schema": "groundupscale.dev/measurement-preflight/v1alpha1",
            "status": "passed",
            "eligible": True,
            "device": "ascend-npu",
            "logical_device": "npu:0",
            "cohort_id": ASCEND_PROFILE_COHORT,
            "reason_codes": [],
        }


class _FakeNpuExecutionRuntime:
    logical_device = "npu:0"
    device_type = "npu"
    timer_source = "fake.npu.Event.elapsed_time"
    timer_resolution_ns = 20.0
    completion_protocol = "fake-event-synchronize-plus-device-synchronize"

    def __init__(self) -> None:
        self._transfers: list[dict[str, object]] = []

    def prepare_model(self, model: nn.Module, *, lane: str) -> nn.Module:
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        buffer_bytes = sum(
            buffer.numel() * buffer.element_size() for buffer in model.buffers()
        )
        self._transfers.append(
            {
                "lane": lane,
                "kind": "weights-and-buffers-host-to-device",
                "bytes": parameter_bytes + buffer_bytes,
            }
        )
        return model

    def prepare_tensor(self, tensor: Tensor, *, lane: str, role: str) -> Tensor:
        self._transfers.append(
            {
                "lane": lane,
                "kind": f"{role}-host-to-device",
                "bytes": tensor.numel() * tensor.element_size(),
            }
        )
        return tensor

    def copy_to_cpu(self, tensor: Tensor, *, lane: str, role: str) -> Tensor:
        self._transfers.append(
            {
                "lane": lane,
                "kind": f"{role}-device-to-host",
                "bytes": tensor.numel() * tensor.element_size(),
            }
        )
        return tensor.detach().cpu()

    def synchronize(self) -> None:
        return None

    def execute_checked(self, invoke: object) -> Tensor:
        return execute_with_npu_cpu_fallback_guard(invoke)  # type: ignore[arg-type,return-value]

    def execute_timed(self, invoke: object, *, iterations: int) -> dict[str, int]:
        self.synchronize()
        started = time.perf_counter_ns()
        for _ in range(iterations):
            self.execute_checked(invoke)
        launch_ended = time.perf_counter_ns()
        self.synchronize()
        completed = time.perf_counter_ns()
        elapsed = max(1, completed - started)
        return {
            "primary_elapsed_ns": elapsed,
            "host_launch_ns": max(1, launch_ended - started),
            "device_completion_wait_ns": max(0, completed - launch_ended),
            "host_completion_ns": elapsed,
        }

    def tensor_device(self, _tensor: Tensor) -> str:
        return self.logical_device

    def tensor_device_type(self, _tensor: Tensor) -> str:
        return self.device_type

    def memory_snapshot(self) -> dict[str, int]:
        return {
            "process_current_rss_bytes": 120_000_000,
            "process_peak_observed_rss_bytes": 123_000_000,
            "framework_current_allocated_bytes": 64_000_000,
            "framework_reserved_bytes": 96_000_000,
            "framework_max_allocated_bytes": 80_000_000,
        }

    def environment(self) -> dict[str, object]:
        return {
            "runtime": "fake-torch-npu",
            "device_name": "Fake Ascend 910B2",
            "logical_device": self.logical_device,
        }

    def transfer_evidence(self) -> dict[str, object]:
        return {
            "schema": "groundupscale.dev/transfer-observation/v1alpha1",
            "records": list(self._transfers),
        }


def test_ascend_demo_runs_complete_model_and_replays_verified_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _FakeNpuExecutionRuntime()
    exit_code = main(
        [
            "run",
            str(ASCEND_DEMO_PLAN),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--artifact-store",
            str(tmp_path),
            "--run-id",
            "complete-npu-demo",
            "--samples",
            "4",
            "--warmup",
            "0",
            "--windows-per-sample",
            "1",
            "--target-window-ms",
            "0.000001",
            "--json",
        ],
        measurement_adapter_factory=lambda *_args, **_kwargs: (
            _AvailableAscendAdapter()
        ),
        execution_runtime_factory=lambda *_args, **_kwargs: runtime,
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    run = Path(summary["run_bundle"])
    manifest = json.loads((run / "run.manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_kind"] == "transformer-demo"
    assert manifest["status"] == "completed"
    assert manifest["device"] == "npu:0"
    assert manifest["hardware_cohort"] == ASCEND_PROFILE_COHORT
    assert manifest["stages"]["compatibility"] == "passed"
    assert manifest["producer_lineage"]["producer"] == "groundupscale@0.1.0"
    roles = {artifact["role"] for artifact in manifest["artifacts"]}
    assert {
        "measurement-capability-manifest",
        "hardware-cohort",
        "measurement-preflight",
        "execution-contract",
        "transfer-observation",
        "correctness-observation",
        "benchmark-observation",
        "observation-trace",
        "memory-observation",
        "prediction-observation-comparison",
        "explanation-graph",
        "html-report",
    } <= roles

    correctness = json.loads(
        (run / "observation/correctness.json").read_text(encoding="utf-8")
    )
    assert correctness["passed"] is True
    assert correctness["atol"] == 0.001
    assert correctness["rtol"] == 0.001
    assert correctness["target_audit"]["semantic_leaf_count"] == 52
    assert set(correctness["target_audit"]["leaf_output_devices"].values()) == {
        "npu:0"
    }
    leaf_contracts = correctness["target_audit"]["leaf_output_contracts"]
    assert len(leaf_contracts) == 52
    assert {contract["dtype"] for contract in leaf_contracts.values()} == {
        "float32"
    }
    assert {contract["device"] for contract in leaf_contracts.values()} == {
        "npu:0"
    }
    assert correctness["target_audit"]["input_contract"] == {
        "device": "npu:0",
        "dtype": "float32",
        "shape": [1, 512, 512],
        "stride": [262144, 512, 1],
        "is_contiguous": True,
    }
    assert correctness["target_audit"]["output_contract"] == {
        "device": "npu:0",
        "dtype": "float32",
        "shape": [1, 512, 512],
        "stride": [262144, 512, 1],
        "is_contiguous": True,
    }
    assert correctness["target_audit"]["fallback_enabled"] is False

    benchmark = json.loads(
        (run / "observation/raw/benchmark.json").read_text(encoding="utf-8")
    )
    assert [case["case_id"] for case in benchmark["cases"]] == [
        "matmul-q-proj",
        "rmsnorm-input",
        "softmax-attention",
        "transformer-layer",
        "two-layer-prefill",
    ]
    assert benchmark["instrumentation_profile"] == "baseline-timing"
    assert benchmark["diagnostic_profiling"] == "separate-artifact"
    assert all(
        case["timing_boundaries"]["primary_timer"]
        == "fake.npu.Event.elapsed_time"
        for case in benchmark["cases"]
    )
    assert all(case["timing_boundaries"]["host_launch_ns"] for case in benchmark["cases"])
    assert all(
        case["timing_boundaries"]["host_completion_ns"]
        for case in benchmark["cases"]
    )

    memory = json.loads(
        (run / "observation/memory.json").read_text(encoding="utf-8")
    )
    assert memory["logical_tensor_live_set"]["peak_framework_tensor_bytes"] > 0
    assert memory["framework_device_memory"]["peak_allocated_bytes"] == 80_000_000
    assert memory["process_memory"]["current_rss_bytes"] == 120_000_000
    assert memory["process_memory"]["peak_rss_bytes"] == 123_000_000
    report = (run / "reports/report.html").read_text(encoding="utf-8")
    assert "逻辑张量 live set" in report
    assert "框架设备内存" in report
    assert "80,000,000 B" in report
    assert "进程 RSS" in report
    assert "123,000,000 B" in report

    transfers = json.loads(
        (run / "observation/transfers.json").read_text(encoding="utf-8")
    )
    transfer_kinds = {record["kind"] for record in transfers["records"]}
    assert "weights-and-buffers-host-to-device" in transfer_kinds
    assert "input-host-to-device" in transfer_kinds
    assert "output-device-to-host" in transfer_kinds
    assert verify_run_bundle(run)["passed"] is True

    assert main(["verify-run", str(run), "--json"]) == 0
    verify_summary = json.loads(capsys.readouterr().out)
    assert verify_summary["passed"] is True
    assert main(["explain", str(run), "--json"]) == 0
    explain_summary = json.loads(capsys.readouterr().out)
    assert len(explain_summary["cases"]) == 5
    assert explain_summary["comparison_status"]

    manifest["producer_lineage"]["producer"] = "tampered-producer"
    (run / "run.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lineage_tampered = verify_run_bundle(run)
    assert lineage_tampered["passed"] is False
    assert "invalid transformer demo producer lineage" in lineage_tampered[
        "failures"
    ]
    manifest["producer_lineage"]["producer"] = "groundupscale@0.1.0"
    manifest["artifacts"] = [
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] != "transfer-observation"
    ]
    (run / "run.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered = verify_run_bundle(run)
    assert tampered["passed"] is False
    assert "missing required artifact role: transfer-observation" in tampered[
        "failures"
    ]


class _UnsupportedOperatorRuntime(_FakeNpuExecutionRuntime):
    def execute_timed(self, invoke: object, *, iterations: int) -> dict[str, int]:
        raise RuntimeError("unsupported-operator:Softmax")


class _DtypeSubstitutingRuntime(_FakeNpuExecutionRuntime):
    def prepare_model(self, model: nn.Module, *, lane: str) -> nn.Module:
        return super().prepare_model(model, lane=lane).double()

    def prepare_tensor(self, tensor: Tensor, *, lane: str, role: str) -> Tensor:
        return super().prepare_tensor(tensor, lane=lane, role=role).double()


class _HiddenCpuFallbackRuntime(_FakeNpuExecutionRuntime):
    def execute_checked(self, invoke: object) -> Tensor:
        def invoke_with_warning() -> Tensor:
            warnings.warn(
                "CAUTION: The operator 'aten::unsupported' is not currently "
                "supported on the NPU backend and will fall back to run on the "
                "CPU. (function npu_cpu_fallback)",
                UserWarning,
            )
            return invoke()  # type: ignore[operator,no-any-return]

        return super().execute_checked(invoke_with_warning)


class _MismatchedCohortAdapter(_AvailableAscendAdapter):
    def fingerprint_cohort(self) -> Mapping[str, object]:
        cohort = dict(super().fingerprint_cohort())
        cohort["cohort_id"] = "ascend-npu-another-cohort"
        return cohort

    def preflight(self) -> Mapping[str, object]:
        preflight = dict(super().preflight())
        preflight["cohort_id"] = "ascend-npu-another-cohort"
        return preflight


def test_ascend_demo_rejects_a_cohort_outside_the_plan_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "run",
            str(ASCEND_DEMO_PLAN),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--artifact-store",
            str(tmp_path),
            "--run-id",
            "mismatched-cohort-demo",
            "--samples",
            "4",
            "--warmup",
            "0",
            "--windows-per-sample",
            "1",
            "--target-window-ms",
            "0.000001",
            "--json",
        ],
        measurement_adapter_factory=lambda *_args, **_kwargs: (
            _MismatchedCohortAdapter()
        ),
        execution_runtime_factory=lambda *_args, **_kwargs: (
            _FakeNpuExecutionRuntime()
        ),
    )

    assert exit_code == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "blocked"
    assert summary["reason_codes"] == ["hardware-cohort-profile-mismatch"]
    run = Path(summary["run_bundle"])
    assert verify_run_bundle(run)["passed"] is True


def test_ascend_demo_preserves_execution_compatibility_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _UnsupportedOperatorRuntime()
    exit_code = main(
        [
            "run",
            str(ASCEND_DEMO_PLAN),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--artifact-store",
            str(tmp_path),
            "--run-id",
            "unsupported-npu-demo",
            "--samples",
            "4",
            "--warmup",
            "0",
            "--windows-per-sample",
            "1",
            "--target-window-ms",
            "0.000001",
            "--json",
        ],
        measurement_adapter_factory=lambda *_args, **_kwargs: (
            _AvailableAscendAdapter()
        ),
        execution_runtime_factory=lambda *_args, **_kwargs: runtime,
    )

    assert exit_code == 2
    summary = json.loads(capsys.readouterr().out)
    run = Path(summary["run_bundle"])
    manifest = json.loads((run / "run.manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "compatibility-failed"
    assert manifest["reason_codes"] == ["unsupported-operator:Softmax"]
    assert manifest["stages"]["compatibility"] == "failed"
    assert manifest["stages"]["benchmark"] == "failed"
    failure = json.loads(
        (run / "observation/execution-failure.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "compatibility-failed"
    assert failure["failed_stage"] == "benchmark"
    assert failure["failed_before_execution"] is False
    assert failure["reason_codes"] == ["unsupported-operator:Softmax"]
    roles = {artifact["role"] for artifact in manifest["artifacts"]}
    assert {
        "resolved-input-lock",
        "measurement-capability-manifest",
        "hardware-cohort",
        "measurement-preflight",
        "transfer-observation",
        "execution-failure",
    } <= roles
    assert verify_run_bundle(run)["passed"] is True


def test_ascend_demo_rejects_dtype_or_layout_substitution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "run",
            str(ASCEND_DEMO_PLAN),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--artifact-store",
            str(tmp_path),
            "--run-id",
            "dtype-substitution-demo",
            "--samples",
            "4",
            "--warmup",
            "0",
            "--windows-per-sample",
            "1",
            "--target-window-ms",
            "0.000001",
            "--json",
        ],
        measurement_adapter_factory=lambda *_args, **_kwargs: (
            _AvailableAscendAdapter()
        ),
        execution_runtime_factory=lambda *_args, **_kwargs: (
            _DtypeSubstitutingRuntime()
        ),
    )

    assert exit_code == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "compatibility-failed"
    assert summary["reason_codes"] == ["dtype-layout-substitution-detected"]
    run = Path(summary["run_bundle"])
    manifest = json.loads((run / "run.manifest.json").read_text(encoding="utf-8"))
    failure = json.loads(
        (run / "observation/execution-failure.json").read_text(encoding="utf-8")
    )
    assert failure["failed_stage"] == "correctness"
    roles = {artifact["role"] for artifact in manifest["artifacts"]}
    assert {
        "benchmark-observation",
        "observation-trace",
        "alignment-map",
        "memory-observation",
        "prediction-observation-comparison",
    } <= roles
    assert verify_run_bundle(run)["passed"] is True


def test_ascend_demo_rejects_hidden_torch_npu_cpu_fallback_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "run",
            str(ASCEND_DEMO_PLAN),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--artifact-store",
            str(tmp_path),
            "--run-id",
            "hidden-cpu-fallback-demo",
            "--samples",
            "4",
            "--warmup",
            "0",
            "--windows-per-sample",
            "1",
            "--target-window-ms",
            "0.000001",
            "--json",
        ],
        measurement_adapter_factory=lambda *_args, **_kwargs: (
            _AvailableAscendAdapter()
        ),
        execution_runtime_factory=lambda *_args, **_kwargs: (
            _HiddenCpuFallbackRuntime()
        ),
    )

    assert exit_code == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["reason_codes"] == ["cpu-fallback-detected"]
    run = Path(summary["run_bundle"])
    failure = json.loads(
        (run / "observation/execution-failure.json").read_text(encoding="utf-8")
    )
    assert failure["failed_stage"] == "benchmark"
    assert verify_run_bundle(run)["passed"] is True


def test_ascend_demo_preserves_runtime_initialization_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_runtime_initialization(_device: str) -> _FakeNpuExecutionRuntime:
        raise RuntimeError("npu-context-initialization-failed")

    exit_code = main(
        [
            "run",
            str(ASCEND_DEMO_PLAN),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--artifact-store",
            str(tmp_path),
            "--run-id",
            "runtime-init-failure-demo",
            "--json",
        ],
        measurement_adapter_factory=lambda *_args, **_kwargs: (
            _AvailableAscendAdapter()
        ),
        execution_runtime_factory=fail_runtime_initialization,
    )

    assert exit_code == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "blocked"
    assert summary["reason_codes"] == ["npu-context-initialization-failed"]
    run = Path(summary["run_bundle"])
    failure = json.loads(
        (run / "observation/execution-failure.json").read_text(encoding="utf-8")
    )
    assert failure["failed_before_execution"] is True
    assert verify_run_bundle(run)["passed"] is True
