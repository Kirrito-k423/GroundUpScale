from __future__ import annotations

import builtins
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import NoReturn

from groundupscale.measurement_contract import (
    HardwareValidityIdentity,
    MeasurementAdapter,
    MeasurementCapabilityManifest,
)
from groundupscale.run_bundle import verify_run_bundle


class _FakeNpuRuntime:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def set_device(self, index: int) -> None:
        self.selected_device = index

    def current_device(self) -> int:
        return self.selected_device

    def get_device_name(self, index: int) -> str:
        assert index == self.selected_device
        return "Ascend910B2"


class _UnavailableNpuRuntime:
    def is_available(self) -> bool:
        return False

    def device_count(self) -> int:
        return 0

    def set_device(self, index: int) -> None:
        raise AssertionError("an unavailable NPU must not be selected")


class _SelectionFailureNpuRuntime:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def set_device(self, index: int) -> None:
        raise RuntimeError("device is not ready")


def _available_runtime() -> tuple[object, object]:
    npu = _FakeNpuRuntime()
    npu.selected_device = 0
    torch = SimpleNamespace(__version__="2.7.1", npu=npu)
    torch_npu = SimpleNamespace(__version__="2.7.1")
    return torch, torch_npu


def _unavailable_runtime() -> tuple[object, object]:
    torch = SimpleNamespace(__version__="2.7.1", npu=_UnavailableNpuRuntime())
    torch_npu = SimpleNamespace(__version__="2.7.1")
    return torch, torch_npu


def _selection_failure_runtime() -> tuple[object, object]:
    torch = SimpleNamespace(
        __version__="2.7.1",
        npu=_SelectionFailureNpuRuntime(),
    )
    torch_npu = SimpleNamespace(__version__="2.7.1")
    return torch, torch_npu


def _unsupported_runtime_pair() -> tuple[object, object]:
    npu = _FakeNpuRuntime()
    npu.selected_device = 0
    torch = SimpleNamespace(__version__="2.8.0", npu=npu)
    torch_npu = SimpleNamespace(__version__="2.8.0")
    return torch, torch_npu


def _complete_system_probe(logical_device_index: int) -> dict[str, object]:
    assert logical_device_index == 0
    return {
        "schema": "groundupscale.dev/ascend-system-probe/v1alpha1",
        "status": "completed",
        "hardware": {
            "machine_name": "test-npu-host",
            "device_name": "910B2",
            "device_version": "V1",
            "vdie_id": "test-vdie-id",
            "logical_device": "npu:0",
            "physical_device": {"npu_id": 0, "chip_id": 0},
            "chip_logic_id": 0,
            "pcie_bus_id": "0000:C1:00.0",
            "hbm_capacity_bytes": 65_536 * 1024 * 1024,
            "hbm_clock_mhz": 1600,
            "real_time_power_watts": 91.7,
            "sources": {
                "device_identity": "artifact://adapter/cohort.json#system_probe/command_snapshots/npu_board",
                "memory": "artifact://adapter/cohort.json#system_probe/command_snapshots/npu_memory",
                "power": "artifact://adapter/cohort.json#system_probe/command_snapshots/npu_power",
                "frequency": "artifact://adapter/cohort.json#system_probe/command_snapshots/npu_memory",
            },
        },
        "software": {
            "os": {"value": "openEuler 22.03", "source": "file:///etc/os-release"},
            "kernel": {"value": "5.10-test", "source": "python://platform.release"},
            "driver": {"value": "25.3.rc1", "source": "file:///usr/local/Ascend/driver/version.info"},
            "firmware": {"value": "7.7.0.10.220", "source": "file:///usr/local/Ascend/firmware/version.info"},
            "cann": {"value": "8.5.0", "source": "file:///usr/local/Ascend/ascend-toolkit/latest/compiler/version.info"},
        },
        "health": {
            "status": "OK",
            "competing_process": False,
            "throttling": "unknown",
            "source": "artifact://adapter/cohort.json#system_probe/command_snapshots/npu_health",
        },
        "topology": {
            "topology_sha256": "a" * 64,
            "hccs_sha256": "b" * 64,
            "cpu_affinity": "192-223",
            "numa_node": "6",
            "sources": {
                "topology": "artifact://adapter/cohort.json#system_probe/command_snapshots/npu_topology",
                "hccs": "artifact://adapter/cohort.json#system_probe/command_snapshots/npu_hccs",
            },
        },
        "power_clock": {
            "power_policy": "unsupported(npu-smi-work-mode-query)",
            "clock_policy": "hbm=1600MHz;ai-core=unknown",
        },
        "command_snapshots": {
            "npu_board": {"status": "measured", "stdout_sha256": "c" * 64},
            "npu_memory": {"status": "measured", "stdout_sha256": "d" * 64},
            "npu_power": {"status": "measured", "stdout_sha256": "e" * 64},
            "npu_health": {"status": "measured", "stdout_sha256": "f" * 64},
            "npu_topology": {"status": "measured", "stdout_sha256": "1" * 64},
            "npu_hccs": {"status": "measured", "stdout_sha256": "2" * 64},
        },
        "evidence_ref": "artifact://adapter/cohort.json#system_probe",
    }


def _exact_shape_case() -> dict[str, object]:
    return {
        "schema": "groundupscale.dev/exact-shape-matmul-case/v1alpha1",
        "operation": "MatMul",
        "shape": {"left": [2, 3], "right": [3, 4]},
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "seed": 20260810,
        "candidate": "torch.matmul",
        "warmup_iterations": 2,
        "repetitions": 3,
    }


def _raw_hardware_collection(
    torch: object,
    logical_device_index: int,
    case: dict[str, object],
    timing_plan: dict[str, object],
) -> dict[str, object]:
    return {
        "runtime_device_name": "Ascend910B2",
        "candidate_device": "npu:0",
        "cpu_fallback": False,
        "left_sha256": "1" * 64,
        "right_sha256": "2" * 64,
        "target_output_sha256": "3" * 64,
        "correctness": {
            "status": "passed",
            "oracle": "cpu-float64-matmul",
            "atol": 0.001,
            "rtol": 0.001,
            "max_absolute_error": 0.0002,
            "max_relative_error": 0.0004,
            "finite": True,
            "shape_exact": True,
        },
        "raw_samples_ns": [12_000, 13_000, 14_000],
        "memory": {
            "allocated_bytes_before": 1024,
            "allocated_bytes_after": 2048,
            "reserved_bytes_after": 4096,
            "maximum_allocated_bytes": 8192,
        },
        "device_event_id": "torch-npu-event-pair",
        "stream_id": "default-npu-stream",
    }


def _write_measurement_bundle(root: Path) -> Path:
    from groundupscale.measurement_adapters.ascend_npu import (
        AscendNpuMeasurementAdapter,
    )
    from groundupscale.measurement_run import MeasurementRunBundleWriter

    adapter = AscendNpuMeasurementAdapter(
        runtime_loader=_available_runtime,
        collection_executor=_raw_hardware_collection,
        system_probe=_complete_system_probe,
    )
    return MeasurementRunBundleWriter(adapter).run(
        root,
        case=_exact_shape_case(),
        run_id="ascend-exact-shape-verifier-test",
    )


def _rewrite_json_artifact(
    run: Path,
    manifest: dict[str, object],
    role: str,
    mutate,
) -> None:
    artifact = next(
        item for item in manifest["artifacts"] if item["role"] == role
    )
    artifact_path = run / artifact["path"]
    document = json.loads(artifact_path.read_text(encoding="utf-8"))
    mutate(document)
    artifact_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    artifact["sha256"] = sha256(artifact_path.read_bytes()).hexdigest()


def test_ascend_npu_is_registered_without_importing_torch_npu(
    monkeypatch,
) -> None:
    original_import = builtins.__import__

    def reject_torch_npu(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch_npu" or name.startswith("torch_npu."):
            raise AssertionError("torch_npu must be loaded only for a real NPU run")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch_npu)

    from groundupscale.measurement_adapters import (
        available_measurement_devices,
        create_measurement_adapter,
    )

    assert available_measurement_devices() == ("ascend-npu",)
    adapter = create_measurement_adapter(
        "ascend-npu", logical_device_index=0
    )
    assert isinstance(adapter, MeasurementAdapter)


def test_missing_torch_npu_returns_structured_preflight_blocker() -> None:
    from groundupscale.measurement_adapters.ascend_npu import (
        AscendNpuMeasurementAdapter,
    )

    def missing_runtime() -> NoReturn:
        raise ModuleNotFoundError("No module named 'torch_npu'")

    adapter = AscendNpuMeasurementAdapter(runtime_loader=missing_runtime)

    assert adapter.preflight() == {
        "schema": "groundupscale.dev/measurement-preflight/v1alpha1",
        "operation": "preflight",
        "status": "blocked",
        "eligible": False,
        "device": "ascend-npu",
        "logical_device": "npu:0",
        "reason_codes": ["torch-npu-unavailable"],
        "evidence_ref": "artifact://adapter/preflight.json",
    }


def test_missing_torch_npu_preserves_discovery_and_cohort_failures() -> None:
    from groundupscale.measurement_adapters.ascend_npu import (
        AscendNpuMeasurementAdapter,
    )

    def missing_runtime() -> NoReturn:
        raise ModuleNotFoundError("No module named 'torch_npu'")

    adapter = AscendNpuMeasurementAdapter(runtime_loader=missing_runtime)

    assert adapter.discover_capabilities() == {
        "schema": "groundupscale.dev/measurement-capability-manifest/v1alpha1",
        "operation": "discover_capabilities",
        "status": "blocked",
        "device": "ascend-npu",
        "logical_device": "npu:0",
        "reason_codes": ["torch-npu-unavailable"],
        "evidence_ref": "artifact://adapter/capabilities.json",
    }
    assert adapter.fingerprint_cohort() == {
        "schema": "groundupscale.dev/hardware-cohort/v1alpha1",
        "operation": "fingerprint_cohort",
        "status": "blocked",
        "device": "ascend-npu",
        "logical_device": "npu:0",
        "reason_codes": ["torch-npu-unavailable"],
        "evidence_ref": "artifact://adapter/cohort.json",
    }


def test_installed_runtime_with_no_visible_npu_is_structurally_blocked() -> None:
    from groundupscale.measurement_adapters.ascend_npu import (
        AscendNpuMeasurementAdapter,
    )

    adapter = AscendNpuMeasurementAdapter(runtime_loader=_unavailable_runtime)

    discovery = adapter.discover_capabilities()
    cohort = adapter.fingerprint_cohort()
    preflight = adapter.preflight()

    assert discovery["status"] == "blocked"
    assert discovery["reason_codes"] == ["ascend-npu-unavailable"]
    assert cohort["status"] == "blocked"
    assert cohort["reason_codes"] == ["ascend-npu-unavailable"]
    assert preflight["status"] == "blocked"
    assert preflight["eligible"] is False
    assert preflight["reason_codes"] == ["ascend-npu-unavailable"]


def test_runtime_device_selection_failure_is_structurally_blocked() -> None:
    from groundupscale.measurement_adapters.ascend_npu import (
        AscendNpuMeasurementAdapter,
    )

    adapter = AscendNpuMeasurementAdapter(
        runtime_loader=_selection_failure_runtime,
    )

    assert adapter.discover_capabilities()["reason_codes"] == [
        "ascend-npu-selection-failed:RuntimeError"
    ]
    assert adapter.fingerprint_cohort()["reason_codes"] == [
        "ascend-npu-selection-failed:RuntimeError"
    ]
    assert adapter.preflight()["reason_codes"] == [
        "ascend-npu-selection-failed:RuntimeError"
    ]


def test_unhealthy_or_contended_npu_fails_preflight_closed() -> None:
    from groundupscale.measurement_adapters.ascend_npu import (
        AscendNpuMeasurementAdapter,
    )

    probe = _complete_system_probe(0)
    probe["health"] = dict(probe["health"])
    probe["health"].update({"status": "Warning", "competing_process": True})

    preflight = AscendNpuMeasurementAdapter(
        runtime_loader=_available_runtime,
        system_probe=lambda _: probe,
    ).preflight()

    assert preflight["status"] == "blocked"
    assert preflight["eligible"] is False
    assert preflight["reason_codes"] == [
        "npu-health-not-ok",
        "competing-npu-process",
    ]


def test_process_probe_excludes_the_measurement_process_itself() -> None:
    from groundupscale.measurement_adapters.ascend_npu import (
        _parse_npu_processes,
    )

    summary = """
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
| 0       0                 | 123           | python                   | 105                     |
| 0       0                 | 456           | worker                   | 2048                    |
| No running processes found in NPU 1                                                            |
"""

    assert _parse_npu_processes(summary, physical_npu_id=0, current_pid=123) == [
        {
            "pid": 456,
            "process_name": "worker",
            "process_memory_mb": 2048,
        }
    ]


def test_exact_shape_case_builds_explicit_npu_timing_plan() -> None:
    from groundupscale.measurement_adapters.ascend_npu import (
        AscendNpuMeasurementAdapter,
    )

    case = {
        "schema": "groundupscale.dev/exact-shape-matmul-case/v1alpha1",
        "operation": "MatMul",
        "shape": {"left": [512, 512], "right": [512, 512]},
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "seed": 20260810,
        "candidate": "torch.matmul",
        "warmup_iterations": 20,
        "repetitions": 100,
    }

    plan = AscendNpuMeasurementAdapter().build_timing_plan(case)

    assert plan == {
        "schema": "groundupscale.dev/timing-plan/v1alpha1",
        "operation": "build_timing_plan",
        "status": "ready",
        "device": "ascend-npu",
        "logical_device": "npu:0",
        "case": case,
        "timer": {
            "kind": "device-event",
            "source": "torch.npu.Event.elapsed_time",
            "unit": "nanoseconds",
        },
        "completion_boundary": {
            "kind": "device-event-stream-completion",
            "protocol": "end-event-synchronize-plus-device-synchronize",
        },
        "warmup_iterations": 20,
        "repetitions": 100,
        "sample_exclusion": "none-preserve-all-raw-samples",
        "evidence_ref": "artifact://adapter/timing-plan.json",
    }


def test_available_npu_runtime_passes_structured_preflight() -> None:
    from groundupscale.measurement_adapters.ascend_npu import (
        AscendNpuMeasurementAdapter,
    )

    preflight = AscendNpuMeasurementAdapter(
        runtime_loader=_available_runtime,
        system_probe=_complete_system_probe,
    ).preflight()

    assert preflight == {
        "schema": "groundupscale.dev/measurement-preflight/v1alpha1",
        "operation": "preflight",
        "status": "passed",
        "eligible": True,
        "device": "ascend-npu",
        "logical_device": "npu:0",
        "cohort_id": preflight["cohort_id"],
        "device_name": "Ascend910B2",
        "device_count": 1,
        "runtime_current_device": 0,
        "torch_version": "2.7.1",
        "torch_npu_version": "2.7.1",
        "runtime_compatibility": {
            "status": "compatible",
            "contract_id": "ascend-npu-runtime-v1",
            "environment_kind": "trusted-hardware-calibration",
            "requirements": {
                "python_major_minor": "3.11",
                "torch_major_minor": "2.7",
                "torch_npu_major_minor": "2.7",
                "cann_major_minor": "8.5",
            },
            "observed": {
                "python": preflight["runtime_compatibility"]["observed"]["python"],
                "torch": "2.7.1",
                "torch_npu": "2.7.1",
                "cann": "8.5.0",
            },
            "rule": (
                "all observed major.minor versions equal the declared trusted "
                "NPU contract"
            ),
            "separation_policy": (
                "pyproject torch pin governs portable compiler CI; this "
                "contract governs trusted Ascend measurement"
            ),
        },
        "checks": {
            "system_probe": "completed",
            "health": "OK",
            "competing_process": False,
            "physical_logical_mapping": "resolved",
            "runtime_contract": "compatible",
        },
        "system_probe_ref": "artifact://adapter/cohort.json#system_probe",
        "reason_codes": [],
        "evidence_ref": "artifact://adapter/preflight.json",
    }


def test_preflight_enforces_declared_groundupscale_npu_runtime_contract() -> None:
    from groundupscale.measurement_adapters.ascend_npu import (
        AscendNpuMeasurementAdapter,
    )

    preflight = AscendNpuMeasurementAdapter(
        runtime_loader=_unsupported_runtime_pair,
        system_probe=_complete_system_probe,
    ).preflight()

    assert preflight["status"] == "blocked"
    assert preflight["eligible"] is False
    assert preflight["runtime_compatibility"]["contract_id"] == (
        "ascend-npu-runtime-v1"
    )
    assert preflight["runtime_compatibility"]["status"] == "incompatible"
    assert preflight["reason_codes"] == [
        "groundupscale-npu-runtime-contract-mismatch"
    ]


def test_capability_discovery_and_cohort_fingerprint_share_identity() -> None:
    from groundupscale.measurement_adapters.ascend_npu import (
        AscendNpuMeasurementAdapter,
    )

    adapter = AscendNpuMeasurementAdapter(
        runtime_loader=_available_runtime,
        system_probe=_complete_system_probe,
    )

    capabilities = adapter.discover_capabilities()
    cohort = adapter.fingerprint_cohort()

    HardwareValidityIdentity.from_document(cohort)
    manifest = MeasurementCapabilityManifest.from_document(
        capabilities,
        adapter_id="ascend-npu",
        cohort_id=cohort["cohort_id"],
    )
    assert capabilities["operation"] == "discover_capabilities"
    assert capabilities["status"] == "completed"
    assert capabilities["evidence_ref"] == (
        "artifact://adapter/capabilities.json"
    )
    assert cohort["operation"] == "fingerprint_cohort"
    assert cohort["status"] == "completed"
    assert cohort["device"] == "910B2/V1/vdie=test-vdie-id"
    assert cohort["partition"].startswith("physical-npu=0/chip=0;")
    assert cohort["evidence_ref"] == "artifact://adapter/cohort.json"
    assert cohort["hardware"]["device_version"] == "V1"
    assert cohort["hardware"]["hbm_capacity_bytes"] == 65_536 * 1024 * 1024
    assert set(cohort["software_evidence"]) >= {
        "os",
        "kernel",
        "driver",
        "firmware",
        "cann",
        "python",
        "torch",
        "torch_npu",
    }
    assert cohort["topology_evidence"]["topology_sha256"] == "a" * 64
    assert cohort["health"]["status"] == "OK"
    assert cohort["system_probe"]["command_snapshots"]
    fields = {field.name: field for field in manifest.fields}
    assert fields["timer.primary"].status.value == "declared"
    assert fields["synchronization.device_stream"].status.value == "declared"
    assert fields["memory.framework"].status.value == "declared"
    assert fields["profiling.operator_timeline"].status.value == (
        "not_requested"
    )


def test_collection_packages_exact_shape_evidence_from_npu_boundary() -> None:
    from groundupscale.measurement_adapters.ascend_npu import (
        AscendNpuMeasurementAdapter,
    )

    case = {
        "schema": "groundupscale.dev/exact-shape-matmul-case/v1alpha1",
        "operation": "MatMul",
        "shape": {"left": [2, 3], "right": [3, 4]},
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "seed": 20260810,
        "candidate": "torch.matmul",
        "warmup_iterations": 2,
        "repetitions": 3,
    }

    def collect_at_hardware_boundary(
        torch: object,
        logical_device_index: int,
        received_case: dict[str, object],
        timing_plan: dict[str, object],
    ) -> dict[str, object]:
        assert logical_device_index == 0
        assert received_case == case
        assert timing_plan["repetitions"] == 3
        return {
            "runtime_device_name": "Ascend910B2",
            "candidate_device": "npu:0",
            "cpu_fallback": False,
            "left_sha256": "1" * 64,
            "right_sha256": "2" * 64,
            "target_output_sha256": "3" * 64,
            "correctness": {
                "status": "passed",
                "oracle": "cpu-float64-matmul",
                "atol": 0.001,
                "rtol": 0.001,
                "max_absolute_error": 0.0002,
                "max_relative_error": 0.0004,
                "finite": True,
                "shape_exact": True,
            },
            "raw_samples_ns": [12_000, 13_000, 14_000],
            "memory": {
                "allocated_bytes_before": 1024,
                "allocated_bytes_after": 2048,
                "reserved_bytes_after": 4096,
                "maximum_allocated_bytes": 8192,
            },
            "device_event_id": "torch-npu-event-pair",
            "stream_id": "default-npu-stream",
        }

    adapter = AscendNpuMeasurementAdapter(
        runtime_loader=_available_runtime,
        collection_executor=collect_at_hardware_boundary,
    )
    timing_plan = dict(adapter.build_timing_plan(case))

    collection = adapter.collect(case, timing_plan)

    assert collection["schema"] == (
        "groundupscale.dev/exact-shape-collection/v1alpha1"
    )
    assert collection["operation"] == "collect"
    assert collection["status"] == "completed"
    assert collection["candidate_identity"] == {
        "schema": "groundupscale.dev/candidate-identity/v1alpha1",
        "candidate_id": "torch.matmul",
        "candidate_family": "pytorch-ascend-matmul",
        "runtime_device_name": "Ascend910B2",
        "candidate_device": "npu:0",
        "cpu_fallback": False,
    }
    assert collection["input_corpus"] == {
        "schema": "groundupscale.dev/input-corpus/v1alpha1",
        "seed": 20260810,
        "initialization": "cpu-torch-randn-fixed-seed",
        "left_shape": [2, 3],
        "right_shape": [3, 4],
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "left_sha256": "1" * 64,
        "right_sha256": "2" * 64,
    }
    assert collection["execution_contract"]["warmup_iterations"] == 2
    assert collection["execution_contract"]["repetitions"] == 3
    assert collection["execution_contract"]["sample_exclusion"] == (
        "none-preserve-all-raw-samples"
    )
    assert collection["correctness"]["status"] == "passed"
    assert collection["correctness"]["target_output_sha256"] == "3" * 64
    assert collection["raw_timing"] == {
        "schema": "groundupscale.dev/raw-timing-observation/v1alpha1",
        "timer_source": "torch.npu.Event.elapsed_time",
        "timer_resolution_ns": 20.0,
        "unit": "nanoseconds",
        "samples": [12_000, 13_000, 14_000],
        "summary": {
            "count": 3,
            "minimum": 12_000,
            "p10": 12_200.0,
            "q1": 12_500.0,
            "median": 13_000,
            "q3": 13_500.0,
            "p90": 13_800.0,
            "maximum": 14_000,
            "iqr": 1_000.0,
            "iqr_fraction_of_median": 1_000 / 13_000,
            "median_absolute_deviation": 1_000,
            "mad_fraction_of_median": 1_000 / 13_000,
        },
    }
    assert collection["timing_quality"] == {
        "schema": "groundupscale.dev/timing-quality/v1alpha1",
        "policy_id": "issue28-session-dispersion-v1",
        "status": "passed",
        "observed_iqr_fraction_of_median": 1_000 / 13_000,
        "maximum_iqr_fraction_of_median": 0.10,
        "timer_resolution_ns": 20.0,
        "timer_resolution_fraction_of_median": 20 / 13_000,
        "maximum_timer_resolution_fraction_of_median": 0.01,
        "excluded_samples": 0,
        "reason_codes": [],
    }
    assert collection["memory"]["maximum_allocated_bytes"] == 8192
    assert collection["completion_boundary"] == {
        "schema": "groundupscale.dev/completion-boundary/v1alpha1",
        "kind": "device-event-stream-completion",
        "closed": True,
        "device_event_id": "torch-npu-event-pair",
        "stream_id": "default-npu-stream",
        "stream_synchronized": True,
        "absolute_timestamps_subtracted": False,
        "protocol": "end-event-synchronize-plus-device-synchronize",
    }
    assert collection["instrumentation_profile"] == {
        "schema": "groundupscale.dev/instrumentation-profile/v1alpha1",
        "profile_id": "ascend-npu-baseline-timing-v1",
        "lane": "baseline-timing",
        "status": "active",
        "intrusion": "minimally-instrumented",
        "collectors": [
            "torch.npu.Event.elapsed_time",
            "torch.npu.memory_*",
            "cpu-float64-correctness-oracle",
        ],
        "diagnostic_profiling": {"status": "not_requested"},
        "synchronization": {
            "before_sample": "device-synchronize",
            "completion": "end-event-synchronize-plus-device-synchronize",
            "per_module_synchronization": False,
        },
        "metadata": {
            "policy": "allowlisted-exact-shape-contract",
            "fields": [
                "shape",
                "dtype",
                "layout",
                "seed",
                "warmup_iterations",
                "repetitions",
                "timer_source",
                "timer_resolution_ns",
            ],
        },
        "accepted_overhead": {
            "rule": "only candidate execution lies between timing events",
            "event_pair": "accepted-primary-timer-overhead",
            "correctness_oracle": "outside-timed-region",
            "memory_queries": "outside-timed-region",
            "diagnostic_profiler": "disabled",
        },
        "evidence_ref": "artifact://resolved/instrumentation-profile.json",
    }
    assert collection["evidence_ref"] == "artifact://adapter/collection.json"


def test_high_dispersion_collection_is_quarantined_without_dropping_samples() -> None:
    from groundupscale.measurement_adapters.ascend_npu import (
        AscendNpuMeasurementAdapter,
    )

    def noisy_collection(*args: object) -> dict[str, object]:
        raw = _raw_hardware_collection(*args)
        raw["raw_samples_ns"] = [100, 100, 1_000]
        return raw

    adapter = AscendNpuMeasurementAdapter(
        runtime_loader=_available_runtime,
        collection_executor=noisy_collection,
    )
    case = _exact_shape_case()
    collection = adapter.collect(
        case,
        dict(adapter.build_timing_plan(case)),
    )

    assert collection["raw_timing"]["samples"] == [100, 100, 1_000]
    assert collection["timing_quality"]["status"] == "quarantined"
    assert collection["timing_quality"]["reason_codes"] == [
        "session-dispersion-exceeds-policy",
        "timer-resolution-exceeds-policy",
    ]


def test_five_adapter_operations_publish_immutable_run_bundle(
    tmp_path: Path,
) -> None:
    from groundupscale.measurement_adapters.ascend_npu import (
        AscendNpuMeasurementAdapter,
    )
    from groundupscale.measurement_run import MeasurementRunBundleWriter

    case = {
        "schema": "groundupscale.dev/exact-shape-matmul-case/v1alpha1",
        "operation": "MatMul",
        "shape": {"left": [2, 3], "right": [3, 4]},
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "seed": 20260810,
        "candidate": "torch.matmul",
        "warmup_iterations": 2,
        "repetitions": 3,
    }

    def collect_at_hardware_boundary(
        torch: object,
        logical_device_index: int,
        received_case: dict[str, object],
        timing_plan: dict[str, object],
    ) -> dict[str, object]:
        return {
            "runtime_device_name": "Ascend910B2",
            "candidate_device": "npu:0",
            "cpu_fallback": False,
            "left_sha256": "1" * 64,
            "right_sha256": "2" * 64,
            "target_output_sha256": "3" * 64,
            "correctness": {
                "status": "passed",
                "oracle": "cpu-float64-matmul",
                "atol": 0.001,
                "rtol": 0.001,
                "max_absolute_error": 0.0002,
                "max_relative_error": 0.0004,
                "finite": True,
                "shape_exact": True,
            },
            "raw_samples_ns": [12_000, 13_000, 14_000],
            "memory": {
                "allocated_bytes_before": 1024,
                "allocated_bytes_after": 2048,
                "reserved_bytes_after": 4096,
                "maximum_allocated_bytes": 8192,
            },
            "device_event_id": "torch-npu-event-pair",
            "stream_id": "default-npu-stream",
        }

    adapter = AscendNpuMeasurementAdapter(
        runtime_loader=_available_runtime,
        collection_executor=collect_at_hardware_boundary,
        system_probe=_complete_system_probe,
    )
    run = MeasurementRunBundleWriter(adapter).run(
        tmp_path,
        case=case,
        run_id="ascend-exact-shape-test",
    )

    manifest = json.loads(
        (run / "run.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["bundle_kind"] == "exact-shape-measurement"
    assert manifest["status"] == "completed"
    assert manifest["device"] == "ascend-npu"
    cohort = json.loads(
        (run / "adapter/cohort.json").read_text(encoding="utf-8")
    )
    assert manifest["hardware_cohort"] == cohort["cohort_id"]
    roles = [artifact["role"] for artifact in manifest["artifacts"]]
    assert roles == [
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
    ]
    assert manifest["observation_validity"] == {
        "status": "valid",
        "correctness": "passed",
        "completion_boundary": "closed",
        "raw_timing_sample_count": 3,
        "timing_quality": "passed",
        "reason_codes": [],
    }
    assert manifest["frontier_role"] == {
        "status": "not-evaluated",
        "reason_code": "issue-28-does-not-promote-frontier",
    }
    assert manifest["producer_lineage"]["source_sha256"]
    assert all(
        manifest["producer_lineage"]["source_sha256"][:16]
        in artifact["produced_by"]
        for artifact in manifest["artifacts"]
    )
    operations = json.loads(
        (run / "adapter/operations.json").read_text(encoding="utf-8")
    )
    assert [item["operation"] for item in operations["operations"]] == [
        "discover_capabilities",
        "fingerprint_cohort",
        "preflight",
        "build_timing_plan",
        "collect",
    ]
    assert all(
        item["evidence_ref"].startswith("artifact://")
        for item in operations["operations"]
    )
    assert verify_run_bundle(run)["passed"] is True


def test_verify_run_fails_when_required_role_is_removed_from_manifest(
    tmp_path: Path,
) -> None:
    run = _write_measurement_bundle(tmp_path)
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    memory = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "memory-observation"
    )
    manifest["artifacts"].remove(memory)
    (run / memory["path"]).unlink()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert "missing required artifact role: memory-observation" in verification[
        "failures"
    ]


def test_verify_run_fails_when_required_role_is_duplicated(
    tmp_path: Path,
) -> None:
    run = _write_measurement_bundle(tmp_path)
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    memory = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "memory-observation"
    )
    manifest["artifacts"].append(dict(memory))
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert "duplicate artifact role: memory-observation" in verification[
        "failures"
    ]


def test_verify_run_fails_when_manifest_cohort_disagrees_with_evidence(
    tmp_path: Path,
) -> None:
    run = _write_measurement_bundle(tmp_path)
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["hardware_cohort"] = "ascend-npu-forged-cohort"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert "hardware cohort mismatch: adapter/cohort.json" in verification[
        "failures"
    ]


def test_verify_run_fails_when_artifact_schema_disagrees_with_manifest(
    tmp_path: Path,
) -> None:
    run = _write_measurement_bundle(tmp_path)
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    memory = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "memory-observation"
    )
    memory_path = run / memory["path"]
    memory_document = json.loads(memory_path.read_text(encoding="utf-8"))
    memory_document["schema"] = "groundupscale.dev/forged-schema/v9"
    memory_path.write_text(
        json.dumps(
            memory_document, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    memory["sha256"] = sha256(memory_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert "schema mismatch: observation/memory.json" in verification[
        "failures"
    ]


def test_verify_run_fails_when_manifest_device_disagrees_with_environment(
    tmp_path: Path,
) -> None:
    run = _write_measurement_bundle(tmp_path)
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    environment = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "environment"
    )
    environment_path = run / environment["path"]
    environment_document = json.loads(
        environment_path.read_text(encoding="utf-8")
    )
    environment_document["device"] = "cpu"
    environment_path.write_text(
        json.dumps(
            environment_document, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    environment["sha256"] = sha256(environment_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert "device mismatch: resolved/environment.json" in verification[
        "failures"
    ]


def test_unavailable_npu_publishes_structured_blocked_attempt(
    tmp_path: Path,
) -> None:
    from groundupscale.measurement_adapters.ascend_npu import (
        AscendNpuMeasurementAdapter,
    )
    from groundupscale.measurement_run import MeasurementRunBundleWriter

    def missing_runtime() -> NoReturn:
        raise ModuleNotFoundError("No module named 'torch_npu'")

    run = MeasurementRunBundleWriter(
        AscendNpuMeasurementAdapter(runtime_loader=missing_runtime)
    ).run(
        tmp_path,
        case=_exact_shape_case(),
        run_id="ascend-unavailable-test",
    )

    manifest = json.loads(
        (run / "run.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "blocked"
    assert manifest["device"] == "ascend-npu"
    assert manifest["hardware_cohort"] is None
    failure = json.loads(
        (run / "adapter/failure.json").read_text(encoding="utf-8")
    )
    assert failure == {
        "schema": "groundupscale.dev/measurement-failure/v1alpha1",
        "status": "blocked",
        "device": "ascend-npu",
        "logical_device": "npu:0",
        "failed_operation": "preflight",
        "reason_codes": ["torch-npu-unavailable"],
        "evidence_refs": [
            "artifact://adapter/capabilities.json",
            "artifact://adapter/cohort.json",
            "artifact://adapter/preflight.json",
        ],
    }
    assert verify_run_bundle(run)["passed"] is True


def test_public_cli_selects_ascend_npu_and_reports_blocked_on_mac(
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "measure",
            "--device",
            "ascend-npu",
            "--m",
            "2",
            "--n",
            "4",
            "--k",
            "3",
            "--warmup",
            "2",
            "--repetitions",
            "3",
            "--artifact-store",
            str(tmp_path),
            "--run-id",
            "cli-ascend-unavailable",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "blocked"
    assert summary["device"] == "ascend-npu"
    assert summary["reason_codes"] == ["torch-npu-unavailable"]
    assert summary["verification_passed"] is True
    run = Path(summary["run_bundle"])
    case = json.loads((run / "resolved/case.json").read_text(encoding="utf-8"))
    assert case["shape"] == {"left": [2, 3], "right": [3, 4]}


def test_verify_run_fails_when_operation_evidence_ref_cannot_replay(
    tmp_path: Path,
) -> None:
    run = _write_measurement_bundle(tmp_path)
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operations_artifact = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "measurement-operation-evidence"
    )
    operations_path = run / operations_artifact["path"]
    operations = json.loads(operations_path.read_text(encoding="utf-8"))
    operations["operations"][-1]["evidence_ref"] = (
        "artifact://adapter/missing-collection.json"
    )
    operations_path.write_text(
        json.dumps(operations, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    operations_artifact["sha256"] = sha256(
        operations_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert (
        "missing evidence reference: "
        "artifact://adapter/missing-collection.json"
    ) in verification["failures"]


def test_verify_run_fails_when_capability_manifest_uses_another_cohort(
    tmp_path: Path,
) -> None:
    run = _write_measurement_bundle(tmp_path)
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capabilities_artifact = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "measurement-capability-manifest"
    )
    capabilities_path = run / capabilities_artifact["path"]
    capabilities = json.loads(
        capabilities_path.read_text(encoding="utf-8")
    )
    capabilities["cohort_id"] = "ascend-npu-foreign-cohort"
    capabilities_path.write_text(
        json.dumps(capabilities, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    capabilities_artifact["sha256"] = sha256(
        capabilities_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert "hardware cohort mismatch: adapter/capabilities.json" in verification[
        "failures"
    ]


def test_verify_run_recomputes_hardware_cohort_digest(tmp_path: Path) -> None:
    run = _write_measurement_bundle(tmp_path)
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _rewrite_json_artifact(
        run,
        manifest,
        "hardware-cohort",
        lambda document: document.update({"partition": "forged-partition"}),
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert "cohort digest mismatch: adapter/cohort.json" in verification[
        "failures"
    ]


def test_verify_run_checks_all_device_bearing_documents(tmp_path: Path) -> None:
    run = _write_measurement_bundle(tmp_path)
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _rewrite_json_artifact(
        run,
        manifest,
        "measurement-collection",
        lambda document: document.update({"device": "cpu"}),
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert "device mismatch: adapter/collection.json" in verification["failures"]


def test_verify_run_checks_preflight_and_environment_cohort_identity(
    tmp_path: Path,
) -> None:
    run = _write_measurement_bundle(tmp_path)
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _rewrite_json_artifact(
        run,
        manifest,
        "measurement-preflight",
        lambda document: document.update({"cohort_id": "forged-cohort"}),
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert "hardware cohort mismatch: adapter/preflight.json" in verification[
        "failures"
    ]
    assert "environment preflight mismatch: resolved/environment.json" in (
        verification["failures"]
    )


def test_verify_run_derives_observation_validity_from_evidence(
    tmp_path: Path,
) -> None:
    run = _write_measurement_bundle(tmp_path)
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def drop_last_sample(document: dict[str, object]) -> None:
        document["samples"].pop()
        document["summary"]["count"] = len(document["samples"])

    def drop_last_collection_sample(document: dict[str, object]) -> None:
        drop_last_sample(document["raw_timing"])

    _rewrite_json_artifact(
        run,
        manifest,
        "raw-timing-observation",
        drop_last_sample,
    )
    _rewrite_json_artifact(
        run,
        manifest,
        "measurement-collection",
        drop_last_collection_sample,
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert "observation validity mismatch" in verification["failures"]


def test_verify_run_requires_all_five_operation_evidence_entries(
    tmp_path: Path,
) -> None:
    run = _write_measurement_bundle(tmp_path)
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def remove_collect(document: dict[str, object]) -> None:
        document["operations"] = [
            item
            for item in document["operations"]
            if item["operation"] != "collect"
        ]

    _rewrite_json_artifact(
        run,
        manifest,
        "measurement-operation-evidence",
        remove_collect,
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert "measurement operation evidence mismatch" in verification[
        "failures"
    ]
