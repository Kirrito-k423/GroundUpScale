"""Ascend NPU Measurement Adapter with lazy runtime loading."""

from __future__ import annotations

import importlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
from collections.abc import Callable, Mapping
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from groundupscale.ir import content_fingerprint
from groundupscale.measurement_adapters import ascend_npu_runtime
from groundupscale.measurement_adapters.ascend_npu_runtime import (
    assess_ascend_npu_runtime,
)

RuntimeLoader = Callable[[], tuple[object, object]]
CollectionExecutor = Callable[
    [object, int, dict[str, object], dict[str, object]],
    dict[str, object],
]
SystemProbe = Callable[[int], dict[str, object]]
PRODUCER_SOURCE_PATHS = (
    Path(str(ascend_npu_runtime.__file__)).resolve(),
)


_MATMUL_CANDIDATES: dict[str, dict[str, object]] = {
    "torch.matmul": {
        "candidate_family": "pytorch-ascend-matmul",
        "operator_entrypoint": "torch.matmul",
        "compilation_parameters": {
            "compiler": "pytorch-eager",
            "graph_compilation": False,
        },
        "tuning_parameters": {},
    },
    "torch.matmul.k-split-2": {
        "candidate_family": "pytorch-ascend-matmul-k-split",
        "operator_entrypoint": "two torch.matmul calls plus torch.add",
        "compilation_parameters": {
            "compiler": "pytorch-eager",
            "graph_compilation": False,
        },
        "tuning_parameters": {
            "k_partitions": 2,
            "split_axis": "k",
        },
    },
}


def _load_runtime() -> tuple[object, object]:
    torch = importlib.import_module("torch")
    torch_npu = importlib.import_module("torch_npu")
    return torch, torch_npu


def _parse_colon_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip():
            values[key.strip()] = value.strip()
    return values


def _parse_equals_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip():
            values[key.strip()] = value.strip().strip('"')
    return values


def _command_snapshot(*command: str) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {
            "status": "collection_failed",
            "command": list(command),
            "error": type(error).__name__,
        }
    stdout = completed.stdout
    return {
        "status": "measured" if completed.returncode == 0 else "collection_failed",
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": completed.stderr,
        "stdout_sha256": sha256(stdout.encode("utf-8")).hexdigest(),
    }


def _file_snapshot(path: str) -> dict[str, object]:
    source = f"file://{path}"
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return {
            "status": "collection_failed",
            "source": source,
            "error": type(error).__name__,
        }
    return {
        "status": "measured",
        "source": source,
        "text": text,
        "sha256": sha256(text.encode("utf-8")).hexdigest(),
    }


def _physical_npu_id(logical_device_index: int) -> int:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        return logical_device_index
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if logical_device_index >= len(entries) or not entries[logical_device_index].isdigit():
        raise ValueError("invalid-ascend-visible-device-mapping")
    return int(entries[logical_device_index])


def _parse_npu_processes(
    text: str,
    *,
    physical_npu_id: int,
    current_pid: int,
) -> list[dict[str, object]]:
    process_pattern = re.compile(
        r"^\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|"
        r"\s*([^|]+?)\s*\|\s*(\d+)\s*\|",
        re.MULTILINE,
    )
    return [
        {
            "pid": int(match.group(2)),
            "process_name": match.group(3).strip(),
            "process_memory_mb": int(match.group(4)),
        }
        for match in process_pattern.finditer(text)
        if int(match.group(1)) == physical_npu_id
        and int(match.group(2)) != current_pid
    ]


def _percentile(samples: list[int], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _timing_summary(samples: list[int]) -> dict[str, float | int]:
    median = statistics.median(samples)
    q1 = _percentile(samples, 0.25)
    q3 = _percentile(samples, 0.75)
    median_absolute_deviation = statistics.median(
        abs(sample - median) for sample in samples
    )
    return {
        "count": len(samples),
        "minimum": min(samples),
        "p10": _percentile(samples, 0.10),
        "q1": q1,
        "median": median,
        "q3": q3,
        "p90": _percentile(samples, 0.90),
        "maximum": max(samples),
        "iqr": q3 - q1,
        "iqr_fraction_of_median": (q3 - q1) / median,
        "median_absolute_deviation": median_absolute_deviation,
        "mad_fraction_of_median": median_absolute_deviation / median,
    }


def _collect_system_probe(logical_device_index: int) -> dict[str, object]:
    try:
        physical_npu_id = _physical_npu_id(logical_device_index)
    except ValueError as error:
        return {
            "schema": "groundupscale.dev/ascend-system-probe/v1alpha1",
            "status": "blocked",
            "reason_codes": [str(error)],
            "evidence_ref": "artifact://adapter/cohort.json#system_probe",
        }
    npu = str(physical_npu_id)
    chip = "0"
    snapshots = {
        "npu_summary": _command_snapshot("npu-smi", "info"),
        "npu_board": _command_snapshot(
            "npu-smi", "info", "-t", "board", "-i", npu, "-c", chip
        ),
        "npu_memory": _command_snapshot(
            "npu-smi", "info", "-t", "memory", "-i", npu, "-c", chip
        ),
        "npu_power": _command_snapshot(
            "npu-smi", "info", "-t", "power", "-i", npu, "-c", chip
        ),
        "npu_health": _command_snapshot(
            "npu-smi", "info", "-t", "health", "-i", npu, "-c", chip
        ),
        "npu_sensors": _command_snapshot(
            "npu-smi", "info", "-t", "sensors", "-i", npu, "-c", chip
        ),
        "npu_work_mode": _command_snapshot(
            "npu-smi", "info", "-t", "work-mode", "-i", npu, "-c", chip
        ),
        "npu_topology": _command_snapshot(
            "npu-smi", "info", "-t", "topo", "-i", npu, "-c", chip
        ),
        "npu_hccs": _command_snapshot(
            "npu-smi", "info", "-t", "hccs", "-i", npu, "-c", chip
        ),
        "npu_mapping": _command_snapshot("npu-smi", "info", "-m"),
        "driver_version": _file_snapshot("/usr/local/Ascend/driver/version.info"),
        "firmware_version": _file_snapshot(
            "/usr/local/Ascend/firmware/version.info"
        ),
        "cann_version": _file_snapshot(
            "/usr/local/Ascend/ascend-toolkit/latest/compiler/version.info"
        ),
        "os_release": _file_snapshot("/etc/os-release"),
    }
    required = (
        "npu_summary",
        "npu_board",
        "npu_memory",
        "npu_power",
        "npu_health",
        "npu_topology",
        "npu_hccs",
        "npu_mapping",
        "driver_version",
        "firmware_version",
        "cann_version",
        "os_release",
    )
    failed = [name for name in required if snapshots[name]["status"] != "measured"]
    if failed:
        return {
            "schema": "groundupscale.dev/ascend-system-probe/v1alpha1",
            "status": "blocked",
            "reason_codes": [f"system-probe-failed:{name}" for name in failed],
            "command_snapshots": snapshots,
            "evidence_ref": "artifact://adapter/cohort.json#system_probe",
        }

    def stdout(name: str) -> str:
        return str(snapshots[name]["stdout"])

    board = _parse_colon_values(stdout("npu_board"))
    memory = _parse_colon_values(stdout("npu_memory"))
    power = _parse_colon_values(stdout("npu_power"))
    health = _parse_colon_values(stdout("npu_health"))
    required_board = ("Chip Name", "Chip Version", "VDie ID", "PCIe Bus Info")
    if any(not board.get(field) for field in required_board):
        return {
            "schema": "groundupscale.dev/ascend-system-probe/v1alpha1",
            "status": "blocked",
            "reason_codes": ["incomplete-board-identity"],
            "command_snapshots": snapshots,
            "evidence_ref": "artifact://adapter/cohort.json#system_probe",
        }
    mapping_pattern = re.compile(
        rf"^\s*{physical_npu_id}\s+0\s+(\d+)\s+Ascend\s+(.+?)\s*$",
        re.MULTILINE,
    )
    mapping_match = mapping_pattern.search(stdout("npu_mapping"))
    if mapping_match is None:
        return {
            "schema": "groundupscale.dev/ascend-system-probe/v1alpha1",
            "status": "blocked",
            "reason_codes": ["missing-physical-logical-device-mapping"],
            "command_snapshots": snapshots,
            "evidence_ref": "artifact://adapter/cohort.json#system_probe",
        }
    topology_text = stdout("npu_topology")
    hccs_text = stdout("npu_hccs")
    hccs_values = _parse_colon_values(hccs_text)
    stable_hccs = {
        key: hccs_values.get(key, "unknown")
        for key in (
            "hccs lane mode",
            "hccs link lane list",
            "hccs link speed",
        )
    }
    stable_hccs_text = json.dumps(
        stable_hccs,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    affinity_match = re.search(
        rf"^NPU{physical_npu_id}\s+.*?\s+(\d+(?:-\d+)?)\s*$",
        topology_text,
        re.MULTILINE,
    )
    work_mode = stdout("npu_work_mode")
    if "does not support querying work-mode" in work_mode.lower():
        power_policy = "unsupported(npu-smi-work-mode-query)"
    elif work_mode.strip():
        power_policy = (
            "declared(npu-smi-work-mode-sha256="
            f"{sha256(work_mode.encode('utf-8')).hexdigest()})"
        )
    else:
        power_policy = "unknown(empty-work-mode-query)"
    required_memory = ("HBM Capacity(MB)", "HBM Clock Speed(MHz)")
    required_power = ("NPU Real-time Power(W)",)
    if any(not memory.get(field) for field in required_memory) or any(
        not power.get(field) for field in required_power
    ):
        return {
            "schema": "groundupscale.dev/ascend-system-probe/v1alpha1",
            "status": "blocked",
            "reason_codes": ["incomplete-memory-power-state"],
            "command_snapshots": snapshots,
            "evidence_ref": "artifact://adapter/cohort.json#system_probe",
        }
    try:
        hbm_capacity_bytes = int(memory["HBM Capacity(MB)"]) * 1024 * 1024
        hbm_clock_mhz = int(memory["HBM Clock Speed(MHz)"])
        real_time_power_watts = float(power["NPU Real-time Power(W)"])
    except ValueError:
        return {
            "schema": "groundupscale.dev/ascend-system-probe/v1alpha1",
            "status": "blocked",
            "reason_codes": ["invalid-memory-power-state"],
            "command_snapshots": snapshots,
            "evidence_ref": "artifact://adapter/cohort.json#system_probe",
        }
    source_root = "artifact://adapter/cohort.json#system_probe/command_snapshots"
    driver = _parse_equals_values(str(snapshots["driver_version"]["text"]))
    firmware = _parse_equals_values(str(snapshots["firmware_version"]["text"]))
    cann = _parse_equals_values(str(snapshots["cann_version"]["text"]))
    os_release = _parse_equals_values(str(snapshots["os_release"]["text"]))
    pcie_numa_path = (
        "/sys/bus/pci/devices/"
        f"{board['PCIe Bus Info'].lower()}/numa_node"
    )
    snapshots["pcie_numa_node"] = _file_snapshot(pcie_numa_path)
    if snapshots["pcie_numa_node"]["status"] != "measured":
        return {
            "schema": "groundupscale.dev/ascend-system-probe/v1alpha1",
            "status": "blocked",
            "reason_codes": ["system-probe-failed:pcie_numa_node"],
            "command_snapshots": snapshots,
            "evidence_ref": "artifact://adapter/cohort.json#system_probe",
        }
    numa_node = str(snapshots["pcie_numa_node"]["text"]).strip()
    competing_processes = _parse_npu_processes(
        stdout("npu_summary"),
        physical_npu_id=physical_npu_id,
        current_pid=os.getpid(),
    )
    return {
        "schema": "groundupscale.dev/ascend-system-probe/v1alpha1",
        "status": "completed",
        "hardware": {
            "machine_name": platform.node(),
            "device_name": board["Chip Name"],
            "device_version": board["Chip Version"],
            "vdie_id": board["VDie ID"],
            "logical_device": f"npu:{logical_device_index}",
            "physical_device": {"npu_id": physical_npu_id, "chip_id": 0},
            "chip_logic_id": int(mapping_match.group(1)),
            "pcie_bus_id": board["PCIe Bus Info"],
            "firmware_reported_by_board": board.get("Firmware Version", "unknown"),
            "hbm_capacity_bytes": hbm_capacity_bytes,
            "hbm_clock_mhz": hbm_clock_mhz,
            "real_time_power_watts": real_time_power_watts,
            "sources": {
                "machine_name": "python://platform.node",
                "device_identity": f"{source_root}/npu_board",
                "logical_device": "environment://ASCEND_RT_VISIBLE_DEVICES",
                "device_mapping": f"{source_root}/npu_mapping",
                "memory": f"{source_root}/npu_memory",
                "power": f"{source_root}/npu_power",
                "frequency": f"{source_root}/npu_memory",
            },
        },
        "software": {
            "os": {
                "value": os_release.get("PRETTY_NAME", os_release.get("NAME", "unknown")),
                "source": str(snapshots["os_release"]["source"]),
            },
            "kernel": {"value": platform.release(), "source": "python://platform.release"},
            "driver": {
                "value": driver.get("Version", driver.get("package_version", "unknown")),
                "source": str(snapshots["driver_version"]["source"]),
            },
            "firmware": {
                "value": firmware.get("Version", firmware.get("package_version", "unknown")),
                "source": str(snapshots["firmware_version"]["source"]),
            },
            "cann": {
                "value": cann.get("Version", "unknown"),
                "source": str(snapshots["cann_version"]["source"]),
            },
        },
        "health": {
            "status": health.get("Health Status", "unknown"),
            "competing_process": bool(competing_processes),
            "competing_processes": competing_processes,
            "throttling": "unknown",
            "sources": {
                "health": f"{source_root}/npu_health",
                "processes": f"{source_root}/npu_summary",
            },
        },
        "topology": {
            "topology_sha256": sha256(topology_text.encode("utf-8")).hexdigest(),
            "hccs_sha256": sha256(stable_hccs_text.encode("utf-8")).hexdigest(),
            "cpu_affinity": affinity_match.group(1) if affinity_match else "unknown",
            "numa_node": numa_node,
            "sources": {
                "topology": f"{source_root}/npu_topology",
                "hccs": f"{source_root}/npu_hccs",
                "numa_node": str(snapshots["pcie_numa_node"]["source"]),
            },
        },
        "power_clock": {
            "power_policy": power_policy,
            "clock_policy": f"hbm={hbm_clock_mhz}MHz;ai-core=unknown",
            "sources": {
                "power_policy": f"{source_root}/npu_work_mode",
                "clock_policy": f"{source_root}/npu_memory",
            },
        },
        "command_snapshots": snapshots,
        "evidence_ref": "artifact://adapter/cohort.json#system_probe",
    }


def _collect_exact_shape_matmul(
    torch: object,
    logical_device_index: int,
    case: dict[str, object],
    timing_plan: dict[str, object],
) -> dict[str, object]:
    runtime: Any = torch
    shape = case.get("shape")
    candidate_id = case.get("candidate")
    if (
        case.get("operation") != "MatMul"
        or candidate_id not in _MATMUL_CANDIDATES
        or case.get("dtype") != "float32"
        or case.get("layout") != "row-major-contiguous"
        or not isinstance(shape, dict)
    ):
        raise ValueError("unsupported exact-Shape MatMul case")
    left_shape = shape.get("left")
    right_shape = shape.get("right")
    if (
        not isinstance(left_shape, list)
        or not isinstance(right_shape, list)
        or len(left_shape) != 2
        or len(right_shape) != 2
        or left_shape[1] != right_shape[0]
        or not all(
            isinstance(dimension, int) and dimension > 0
            for dimension in (*left_shape, *right_shape)
        )
    ):
        raise ValueError("invalid exact-Shape MatMul dimensions")
    warmup_iterations = timing_plan.get("warmup_iterations")
    repetitions = timing_plan.get("repetitions")
    inner_iterations = timing_plan.get("inner_iterations", 1)
    if (
        not isinstance(warmup_iterations, int)
        or warmup_iterations < 0
        or not isinstance(repetitions, int)
        or repetitions < 1
        or not isinstance(inner_iterations, int)
        or inner_iterations < 1
    ):
        raise ValueError("invalid timing plan iteration counts")

    runtime.npu.set_device(logical_device_index)
    logical_device = f"npu:{logical_device_index}"
    generator = runtime.Generator(device="cpu").manual_seed(int(case["seed"]))
    left_cpu = runtime.randn(
        tuple(left_shape), dtype=runtime.float32, generator=generator
    )
    right_cpu = runtime.randn(
        tuple(right_shape), dtype=runtime.float32, generator=generator
    )
    oracle = runtime.matmul(left_cpu.double(), right_cpu.double())
    allocated_before = int(runtime.npu.memory_allocated())
    left = left_cpu.to(logical_device)
    right = right_cpu.to(logical_device)
    runtime.npu.synchronize()

    def invoke_candidate() -> object:
        if candidate_id == "torch.matmul":
            return runtime.matmul(left, right)
        split = int(left_shape[1]) // 2
        if split == 0:
            raise ValueError("k-split candidate requires k >= 2")
        return runtime.matmul(left[:, :split], right[:split, :]) + runtime.matmul(
            left[:, split:], right[split:, :]
        )

    def measure() -> tuple[object, int]:
        start = runtime.npu.Event(enable_timing=True)
        end = runtime.npu.Event(enable_timing=True)
        runtime.npu.synchronize()
        start.record()
        result = None
        for _ in range(inner_iterations):
            result = invoke_candidate()
        end.record()
        end.synchronize()
        runtime.npu.synchronize()
        elapsed_ns = round(
            float(start.elapsed_time(end))
            * 1_000_000
            / inner_iterations
        )
        assert result is not None
        return result, elapsed_ns

    for _ in range(warmup_iterations):
        warmup_result, _ = measure()
        if warmup_result.device.type != "npu":
            raise RuntimeError("cpu-fallback-detected")

    actual = invoke_candidate()
    runtime.npu.synchronize()
    if actual.device.type != "npu":
        raise RuntimeError("cpu-fallback-detected")
    actual_cpu = actual.cpu().double()
    absolute_error = (actual_cpu - oracle).abs()
    relative_error = absolute_error / oracle.abs().clamp_min(1e-12)
    finite = bool(
        runtime.isfinite(actual_cpu).all() and runtime.isfinite(oracle).all()
    )
    shape_exact = tuple(actual_cpu.shape) == tuple(oracle.shape)
    atol = 0.001
    rtol = 0.001
    passed = bool(
        finite
        and shape_exact
        and runtime.allclose(actual_cpu, oracle, atol=atol, rtol=rtol)
    )
    samples: list[int] = []
    for _ in range(repetitions):
        result, elapsed_ns = measure()
        if result.device.type != "npu":
            raise RuntimeError("cpu-fallback-detected")
        if elapsed_ns <= 0:
            raise RuntimeError("invalid-primary-timer-sample")
        samples.append(elapsed_ns)

    def tensor_digest(tensor: Any) -> str:
        return sha256(tensor.contiguous().numpy().tobytes()).hexdigest()

    memory = {
        "allocated_bytes_before": allocated_before,
        "allocated_bytes_after": int(runtime.npu.memory_allocated()),
        "reserved_bytes_after": int(runtime.npu.memory_reserved()),
        "maximum_allocated_bytes": int(runtime.npu.max_memory_allocated()),
    }

    def pointer_alignment(pointer: int) -> int:
        return pointer & -pointer

    return {
        "runtime_device_name": str(
            runtime.npu.get_device_name(logical_device_index)
        ),
        "candidate_device": str(actual.device),
        "cpu_fallback": actual.device.type == "cpu",
        "minimum_alignment_bytes": min(
            pointer_alignment(int(left.data_ptr())),
            pointer_alignment(int(right.data_ptr())),
        ),
        "left_sha256": tensor_digest(left_cpu),
        "right_sha256": tensor_digest(right_cpu),
        "target_output_sha256": tensor_digest(actual_cpu),
        "correctness": {
            "status": "passed" if passed else "failed",
            "oracle": "cpu-float64-matmul",
            "atol": atol,
            "rtol": rtol,
            "max_absolute_error": float(absolute_error.max().item()),
            "max_relative_error": float(relative_error.max().item()),
            "finite": finite,
            "shape_exact": shape_exact,
        },
        "raw_samples_ns": samples,
        "memory": memory,
        "device_event_id": "per-sample-torch-npu-event-pair",
        "stream_id": "default-npu-stream",
    }


class AscendNpuMeasurementAdapter:
    def __init__(
        self,
        *,
        logical_device_index: int = 0,
        runtime_loader: RuntimeLoader = _load_runtime,
        collection_executor: CollectionExecutor = _collect_exact_shape_matmul,
        system_probe: SystemProbe = _collect_system_probe,
    ) -> None:
        if logical_device_index < 0:
            raise ValueError("logical_device_index must be non-negative")
        self.logical_device_index = logical_device_index
        self._runtime_loader = runtime_loader
        self._collection_executor = collection_executor
        self._system_probe = system_probe
        self._cached_cohort: dict[str, object] | None = None

    def discover_capabilities(self) -> Mapping[str, object]:
        try:
            cohort = self.fingerprint_cohort()
        except (ImportError, ModuleNotFoundError):
            return {
                "schema": (
                    "groundupscale.dev/measurement-capability-manifest/"
                    "v1alpha1"
                ),
                "operation": "discover_capabilities",
                "status": "blocked",
                "device": "ascend-npu",
                "logical_device": f"npu:{self.logical_device_index}",
                "reason_codes": ["torch-npu-unavailable"],
                "evidence_ref": "artifact://adapter/capabilities.json",
            }
        if cohort.get("status") != "completed":
            return {
                "schema": (
                    "groundupscale.dev/measurement-capability-manifest/"
                    "v1alpha1"
                ),
                "operation": "discover_capabilities",
                "status": "blocked",
                "device": "ascend-npu",
                "logical_device": f"npu:{self.logical_device_index}",
                "reason_codes": list(
                    cohort.get("reason_codes", ["ascend-npu-unavailable"])
                ),
                "evidence_ref": "artifact://adapter/capabilities.json",
            }

        def field(
            name: str,
            status: str,
            *,
            required_for_anchor: bool,
            source: str,
            value: object | None = None,
        ) -> dict[str, object]:
            result: dict[str, object] = {
                "field": name,
                "status": status,
                "required_for_anchor": required_for_anchor,
                "source": source,
                "scope": "exact-shape MatMul",
                "attribution": "single Ascend NPU logical device",
                "intrusion": "baseline" if required_for_anchor else "optional",
            }
            if status in {"measured", "derived", "declared"}:
                result["value"] = value
            return result

        return {
            "schema": (
                "groundupscale.dev/measurement-capability-manifest/v1alpha1"
            ),
            "operation": "discover_capabilities",
            "status": "completed",
            "manifest_id": f"ascend-npu-{cohort['cohort_id']}-v1",
            "adapter_id": "ascend-npu",
            "adapter_version": "v1",
            "protocol_id": "exact-shape-measurement",
            "protocol_version": "v1",
            "cohort_id": cohort["cohort_id"],
            "device": "ascend-npu",
            "logical_device": f"npu:{self.logical_device_index}",
            "supported_execution_profiles": [
                "exact-shape-matmul",
                "two-layer-transformer-demo",
            ],
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
            "execution_profiles": [
                {
                    "profile_id": "exact-shape-matmul",
                    "protocol_id": "exact-shape-measurement",
                    "execution_domain": {
                        "operation": "MatMul",
                        "dtype": "float32",
                        "layout": "row-major-contiguous",
                    },
                    "correctness": {
                        "oracle": "cpu-float64-matmul",
                        "atol": 0.001,
                        "rtol": 0.001,
                    },
                },
                {
                    "profile_id": "two-layer-transformer-demo",
                    "protocol_id": "two-layer-transformer-demo",
                    "execution_domain": {
                        "model": "two-layer-transformer",
                        "workload": "transformer-prefill",
                        "shape": {
                            "B": 1,
                            "S": 512,
                            "H": 512,
                            "NH": 8,
                            "D": 64,
                            "I": 2048,
                        },
                        "dtype": "float32",
                        "layout": "row-major-contiguous",
                        "semantic_operations": [
                            "MatMul",
                            "Add",
                            "RMSNorm",
                            "Softmax",
                            "SiLU",
                            "Mul",
                            "View",
                            "Transpose",
                        ],
                    },
                    "correctness": {
                        "oracle": "cpu-float32-same-seed-same-weights",
                        "atol": 0.001,
                        "rtol": 0.001,
                        "cpu_fallback_policy": (
                            "warning-is-compatibility-failure"
                        ),
                        "dtype_layout_substitution": (
                            "compatibility-failure"
                        ),
                    },
                    "instrumentation": {
                        "baseline_timing": "npu-event",
                        "diagnostic_profiling": (
                            "separate-non-frontier-lane"
                        ),
                    },
                },
            ],
            "fields": [
                field(
                    "timer.primary",
                    "declared",
                    required_for_anchor=True,
                    source="python://torch.npu.Event.elapsed_time",
                    value={"unit": "nanoseconds", "resolution_ns": 20.0},
                ),
                field(
                    "synchronization.device_stream",
                    "declared",
                    required_for_anchor=True,
                    source="python://torch.npu.synchronize",
                    value="end-event-synchronize-plus-device-synchronize",
                ),
                field(
                    "completion.boundary",
                    "declared",
                    required_for_anchor=True,
                    source="python://torch.npu.Event.synchronize",
                    value="device-event-stream-completion",
                ),
                field(
                    "correctness.cpu_oracle",
                    "declared",
                    required_for_anchor=True,
                    source="python://torch.float64.matmul",
                    value="cpu-float64-matmul",
                ),
                field(
                    "raw_timing.samples",
                    "declared",
                    required_for_anchor=True,
                    source="python://torch.npu.Event.elapsed_time",
                    value="all-samples-preserved",
                ),
                field(
                    "memory.framework",
                    "declared",
                    required_for_anchor=False,
                    source="python://torch.npu.memory_*",
                    value="allocated-reserved-maximum-allocated",
                ),
                field(
                    "memory.hbm_device_wide",
                    "measured",
                    required_for_anchor=False,
                    source=(
                        "artifact://adapter/cohort.json#system_probe/"
                        "hardware/sources/memory"
                    ),
                    value={
                        "capacity_bytes": cohort["hardware"]["hbm_capacity_bytes"],
                    },
                ),
                field(
                    "power.device_wide",
                    "measured",
                    required_for_anchor=False,
                    source=(
                        "artifact://adapter/cohort.json#system_probe/"
                        "hardware/sources/power"
                    ),
                    value={
                        "real_time_watts": cohort["hardware"][
                            "real_time_power_watts"
                        ],
                        "policy": cohort["power_clock"]["power_policy"],
                    },
                ),
                field(
                    "frequency.hbm",
                    "measured",
                    required_for_anchor=False,
                    source=(
                        "artifact://adapter/cohort.json#system_probe/"
                        "hardware/sources/frequency"
                    ),
                    value={
                        "clock_mhz": cohort["hardware"]["hbm_clock_mhz"],
                    },
                ),
                field(
                    "transfer.h2d",
                    "not_requested",
                    required_for_anchor=False,
                    source="python://tensor.to-npu",
                ),
                field(
                    "transfer.d2h",
                    "not_requested",
                    required_for_anchor=False,
                    source="python://tensor.to-cpu",
                ),
                field(
                    "profiling.operator_timeline",
                    "not_requested",
                    required_for_anchor=False,
                    source="python://torch.profiler",
                ),
                field(
                    "counter.ai_core",
                    "not_requested",
                    required_for_anchor=False,
                    source="tool://ascend-profiler",
                ),
            ],
            "evidence_ref": "artifact://adapter/capabilities.json",
        }

    def fingerprint_cohort(self) -> Mapping[str, object]:
        if self._cached_cohort is not None:
            return deepcopy(self._cached_cohort)
        try:
            torch, torch_npu = self._runtime_loader()
        except (ImportError, ModuleNotFoundError):
            return {
                "schema": "groundupscale.dev/hardware-cohort/v1alpha1",
                "operation": "fingerprint_cohort",
                "status": "blocked",
                "device": "ascend-npu",
                "logical_device": f"npu:{self.logical_device_index}",
                "reason_codes": ["torch-npu-unavailable"],
                "evidence_ref": "artifact://adapter/cohort.json",
            }
        npu: Any = getattr(torch, "npu")
        if not npu.is_available() or npu.device_count() <= self.logical_device_index:
            return {
                "schema": "groundupscale.dev/hardware-cohort/v1alpha1",
                "operation": "fingerprint_cohort",
                "status": "blocked",
                "device": "ascend-npu",
                "logical_device": f"npu:{self.logical_device_index}",
                "reason_codes": ["ascend-npu-unavailable"],
                "evidence_ref": "artifact://adapter/cohort.json",
            }
        try:
            npu.set_device(self.logical_device_index)
        except (OSError, RuntimeError, ValueError) as error:
            blocked = {
                "schema": "groundupscale.dev/hardware-cohort/v1alpha1",
                "operation": "fingerprint_cohort",
                "status": "blocked",
                "device": "ascend-npu",
                "logical_device": f"npu:{self.logical_device_index}",
                "reason_codes": [
                    "ascend-npu-selection-failed:"
                    f"{type(error).__name__}"
                ],
                "evidence_ref": "artifact://adapter/cohort.json",
            }
            self._cached_cohort = blocked
            return deepcopy(blocked)
        try:
            system_probe = self._system_probe(self.logical_device_index)
        except (OSError, RuntimeError, ValueError) as error:
            system_probe = {
                "schema": "groundupscale.dev/ascend-system-probe/v1alpha1",
                "status": "blocked",
                "reason_codes": [
                    f"system-probe-failed:{type(error).__name__}"
                ],
                "evidence_ref": "artifact://adapter/cohort.json#system_probe",
            }
        if system_probe.get("status") != "completed":
            blocked = {
                "schema": "groundupscale.dev/hardware-cohort/v1alpha1",
                "operation": "fingerprint_cohort",
                "status": "blocked",
                "device": "ascend-npu",
                "logical_device": f"npu:{self.logical_device_index}",
                "reason_codes": list(
                    system_probe.get("reason_codes", ["system-probe-failed"])
                ),
                "system_probe": deepcopy(system_probe),
                "evidence_ref": "artifact://adapter/cohort.json",
            }
            self._cached_cohort = blocked
            return deepcopy(blocked)

        hardware = deepcopy(system_probe["hardware"])
        software_evidence = deepcopy(system_probe["software"])
        software_evidence.update(
            {
                "python": {
                    "value": platform.python_version(),
                    "source": "python://platform.python_version",
                },
                "torch": {
                    "value": str(getattr(torch, "__version__")),
                    "source": "python://torch.__version__",
                },
                "torch_npu": {
                    "value": str(getattr(torch_npu, "__version__")),
                    "source": "python://torch_npu.__version__",
                },
            }
        )
        health = deepcopy(system_probe["health"])
        topology_evidence = deepcopy(system_probe["topology"])
        visibility = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "all")
        stable_identity: dict[str, object] = {
            "device": (
                f"{hardware['device_name']}/{hardware['device_version']}/"
                f"vdie={hardware['vdie_id']}"
            ),
            "partition": (
                f"physical-npu={hardware['physical_device']['npu_id']}/"
                f"chip={hardware['physical_device']['chip_id']};"
                f"logical-device=npu:{self.logical_device_index};"
                f"ASCEND_RT_VISIBLE_DEVICES={visibility};"
                f"chip-logic-id={hardware['chip_logic_id']}"
            ),
            "topology": (
                f"pcie={hardware['pcie_bus_id']};"
                f"topology-sha256={topology_evidence['topology_sha256']};"
                f"hccs-sha256={topology_evidence['hccs_sha256']}"
            ),
            "software": (
                ";".join(
                    f"{name}={software_evidence[name]['value']}"
                    for name in sorted(software_evidence)
                )
            ),
            "numeric_execution": {
                "dtype": "float32",
                "layout": "row-major-contiguous",
                "alignment_bytes": 64,
                "threads": 1,
                "execution_mode": "pytorch-eager",
            },
            "timer_protocol": {
                "source": "torch.npu.Event.elapsed_time",
                "resolution_ns": 20.0,
                "monotonic": True,
                "completion_kind": "device-event-stream-completion",
                "duration_reducer": None,
                "adapter_id": "ascend-npu",
                "adapter_version": "v1",
                "protocol_id": "exact-shape-measurement",
                "protocol_version": "v1",
            },
            "power_clock": deepcopy(system_probe["power_clock"]),
            "execution_context": {
                "affinity": str(topology_evidence["cpu_affinity"]),
                "numa": str(topology_evidence["numa_node"]),
                "context": "pytorch-eager",
                "stream": "default-npu-stream",
                "concurrency": 1,
            },
            "communication": {"status": "not_applicable"},
        }
        encoded = json.dumps(
            stable_identity,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        cohort_digest = sha256(encoded).hexdigest()
        completed = {
            "schema": "groundupscale.dev/hardware-cohort/v1alpha1",
            "operation": "fingerprint_cohort",
            "status": "completed",
            **stable_identity,
            "runtime_device_name": str(
                npu.get_device_name(self.logical_device_index)
            ),
            "hardware": hardware,
            "software_evidence": software_evidence,
            "health": health,
            "topology_evidence": topology_evidence,
            "system_probe": deepcopy(system_probe),
            "cohort_digest": cohort_digest,
            "cohort_id": f"ascend-npu-{cohort_digest[:16]}",
            "evidence_ref": "artifact://adapter/cohort.json",
        }
        self._cached_cohort = completed
        return deepcopy(completed)

    def preflight(self) -> Mapping[str, object]:
        try:
            torch, torch_npu = self._runtime_loader()
        except (ImportError, ModuleNotFoundError):
            return {
                "schema": "groundupscale.dev/measurement-preflight/v1alpha1",
                "operation": "preflight",
                "status": "blocked",
                "eligible": False,
                "device": "ascend-npu",
                "logical_device": f"npu:{self.logical_device_index}",
                "reason_codes": ["torch-npu-unavailable"],
                "evidence_ref": "artifact://adapter/preflight.json",
            }
        npu: Any = getattr(torch, "npu")
        if not npu.is_available() or npu.device_count() <= self.logical_device_index:
            return {
                "schema": "groundupscale.dev/measurement-preflight/v1alpha1",
                "operation": "preflight",
                "status": "blocked",
                "eligible": False,
                "device": "ascend-npu",
                "logical_device": f"npu:{self.logical_device_index}",
                "reason_codes": ["ascend-npu-unavailable"],
                "evidence_ref": "artifact://adapter/preflight.json",
            }
        cohort = self.fingerprint_cohort()
        if cohort.get("status") != "completed":
            return {
                "schema": "groundupscale.dev/measurement-preflight/v1alpha1",
                "operation": "preflight",
                "status": "blocked",
                "eligible": False,
                "device": "ascend-npu",
                "logical_device": f"npu:{self.logical_device_index}",
                "reason_codes": list(
                    cohort.get("reason_codes", ["cohort-fingerprint-failed"])
                ),
                "evidence_ref": "artifact://adapter/preflight.json",
            }
        torch_version = str(getattr(torch, "__version__"))
        torch_npu_version = str(getattr(torch_npu, "__version__"))
        runtime_compatibility = assess_ascend_npu_runtime(
            python_version=platform.python_version(),
            torch_version=torch_version,
            torch_npu_version=torch_npu_version,
            cann_version=cohort["software_evidence"]["cann"]["value"],
        )
        reason_codes: list[str] = []
        if cohort["health"]["status"] != "OK":
            reason_codes.append("npu-health-not-ok")
        if cohort["health"]["competing_process"] is not False:
            reason_codes.append("competing-npu-process")
        if runtime_compatibility["status"] != "compatible":
            reason_codes.append(
                "groundupscale-npu-runtime-contract-mismatch"
            )
        if reason_codes:
            return {
                "schema": "groundupscale.dev/measurement-preflight/v1alpha1",
                "operation": "preflight",
                "status": "blocked",
                "eligible": False,
                "device": "ascend-npu",
                "logical_device": f"npu:{self.logical_device_index}",
                "cohort_id": cohort["cohort_id"],
                "runtime_compatibility": runtime_compatibility,
                "reason_codes": reason_codes,
                "evidence_ref": "artifact://adapter/preflight.json",
            }
        npu.set_device(self.logical_device_index)
        return {
            "schema": "groundupscale.dev/measurement-preflight/v1alpha1",
            "operation": "preflight",
            "status": "passed",
            "eligible": True,
            "device": "ascend-npu",
            "logical_device": f"npu:{self.logical_device_index}",
            "cohort_id": cohort["cohort_id"],
            "device_name": npu.get_device_name(self.logical_device_index),
            "device_count": npu.device_count(),
            "runtime_current_device": npu.current_device(),
            "torch_version": torch_version,
            "torch_npu_version": torch_npu_version,
            "runtime_compatibility": runtime_compatibility,
            "checks": {
                "system_probe": "completed",
                "health": "OK",
                "competing_process": False,
                "physical_logical_mapping": "resolved",
                "runtime_contract": "compatible",
            },
            "system_probe_ref": (
                "artifact://adapter/cohort.json#system_probe"
            ),
            "reason_codes": [],
            "evidence_ref": "artifact://adapter/preflight.json",
        }

    def build_timing_plan(
        self, case: dict[str, object]
    ) -> Mapping[str, object]:
        return {
            "schema": "groundupscale.dev/timing-plan/v1alpha1",
            "operation": "build_timing_plan",
            "status": "ready",
            "device": "ascend-npu",
            "logical_device": f"npu:{self.logical_device_index}",
            "case": deepcopy(case),
            "timer": {
                "kind": "device-event",
                "source": "torch.npu.Event.elapsed_time",
                "unit": "nanoseconds",
            },
            "completion_boundary": {
                "kind": "device-event-stream-completion",
                "protocol": (
                    "end-event-synchronize-plus-device-synchronize"
                ),
            },
            "warmup_iterations": case["warmup_iterations"],
            "repetitions": case["repetitions"],
            "inner_iterations": case.get("inner_iterations", 1),
            "sample_exclusion": "none-preserve-all-raw-samples",
            "evidence_ref": "artifact://adapter/timing-plan.json",
        }

    def collect(
        self,
        case: dict[str, object],
        timing_plan: dict[str, object],
    ) -> Mapping[str, object]:
        torch, torch_npu = self._runtime_loader()
        raw = self._collection_executor(
            torch,
            self.logical_device_index,
            deepcopy(case),
            deepcopy(timing_plan),
        )
        samples = list(raw["raw_samples_ns"])
        timing_summary = _timing_summary(samples)
        inner_iterations = int(timing_plan.get("inner_iterations", 1))
        timer_resolution_ns = max(1.0, 20.0 / inner_iterations)
        timer_resolution_fraction = (
            timer_resolution_ns / timing_summary["median"]
        )
        timing_reason_codes: list[str] = []
        if timing_summary["iqr_fraction_of_median"] > 0.10:
            timing_reason_codes.append("session-dispersion-exceeds-policy")
        if timer_resolution_fraction > 0.01:
            timing_reason_codes.append("timer-resolution-exceeds-policy")
        timing_quality = {
            "schema": "groundupscale.dev/timing-quality/v1alpha1",
            "policy_id": "issue28-session-dispersion-v1",
            "status": "passed" if not timing_reason_codes else "quarantined",
            "observed_iqr_fraction_of_median": timing_summary[
                "iqr_fraction_of_median"
            ],
            "maximum_iqr_fraction_of_median": 0.10,
            "timer_resolution_ns": timer_resolution_ns,
            "timer_resolution_fraction_of_median": timer_resolution_fraction,
            "maximum_timer_resolution_fraction_of_median": 0.01,
            "excluded_samples": 0,
            "reason_codes": timing_reason_codes,
        }
        correctness = dict(raw["correctness"])
        correctness.update(
            {
                "schema": (
                    "groundupscale.dev/correctness-observation/v1alpha1"
                ),
                "target_output_sha256": raw["target_output_sha256"],
            }
        )
        memory = dict(raw["memory"])
        memory["schema"] = "groundupscale.dev/memory-observation/v1alpha1"
        candidate_id = str(case["candidate"])
        candidate_spec = _MATMUL_CANDIDATES.get(candidate_id)
        if candidate_spec is None:
            raise ValueError(f"unsupported MatMul candidate: {candidate_id}")
        candidate_identity = {
            "schema": "groundupscale.dev/candidate-identity/v1alpha1",
            "candidate_id": candidate_id,
            "candidate_family": candidate_spec["candidate_family"],
            "build_identity": {
                "framework": "torch",
                "framework_version": str(getattr(torch, "__version__", "unknown")),
                "extension": "torch_npu",
                "extension_version": str(
                    getattr(torch_npu, "__version__", "unknown")
                ),
                "operator_entrypoint": candidate_spec["operator_entrypoint"],
            },
            "runtime_identity": {
                "runtime_device_name": raw["runtime_device_name"],
                "logical_device": f"npu:{self.logical_device_index}",
                "candidate_device": raw["candidate_device"],
            },
            "execution_mode": "pytorch-eager",
            "shape": deepcopy(case["shape"]),
            "dtype": case["dtype"],
            "layout": case["layout"],
            "minimum_alignment_bytes": raw.get(
                "minimum_alignment_bytes", 64
            ),
            "compilation_parameters": deepcopy(
                candidate_spec["compilation_parameters"]
            ),
            "tuning_parameters": deepcopy(
                candidate_spec["tuning_parameters"]
            ),
            "runtime_device_name": raw["runtime_device_name"],
            "candidate_device": raw["candidate_device"],
            "cpu_fallback": raw["cpu_fallback"],
            "evidence_ref": "artifact://observation/candidate.json",
        }
        candidate_identity["candidate_digest"] = content_fingerprint(
            candidate_identity
        )
        return {
            "schema": "groundupscale.dev/exact-shape-collection/v1alpha1",
            "operation": "collect",
            "status": "completed",
            "device": "ascend-npu",
            "logical_device": f"npu:{self.logical_device_index}",
            "candidate_identity": candidate_identity,
            "input_corpus": {
                "schema": "groundupscale.dev/input-corpus/v1alpha1",
                "seed": case["seed"],
                "initialization": "cpu-torch-randn-fixed-seed",
                "left_shape": case["shape"]["left"],
                "right_shape": case["shape"]["right"],
                "dtype": case["dtype"],
                "layout": case["layout"],
                "left_sha256": raw["left_sha256"],
                "right_sha256": raw["right_sha256"],
            },
            "execution_contract": {
                "schema": "groundupscale.dev/execution-contract/v1alpha1",
                "operation": case["operation"],
                "shape": deepcopy(case["shape"]),
                "dtype": case["dtype"],
                "layout": case["layout"],
                "candidate": case["candidate"],
                "logical_device": f"npu:{self.logical_device_index}",
                "warmup_iterations": timing_plan["warmup_iterations"],
                "repetitions": timing_plan["repetitions"],
                "inner_iterations": inner_iterations,
                "sample_exclusion": timing_plan["sample_exclusion"],
                "timer": deepcopy(timing_plan["timer"]),
                "completion_protocol": deepcopy(
                    timing_plan["completion_boundary"]
                ),
            },
            "instrumentation_profile": {
                "schema": (
                    "groundupscale.dev/instrumentation-profile/v1alpha1"
                ),
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
                    "completion": (
                        "end-event-synchronize-plus-device-synchronize"
                    ),
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
                        "inner_iterations",
                        "timer_source",
                        "timer_resolution_ns",
                    ],
                },
                "accepted_overhead": {
                    "rule": (
                        "only candidate executions lie between timing events"
                    ),
                    "event_pair": "accepted-primary-timer-overhead",
                    "correctness_oracle": "outside-timed-region",
                    "memory_queries": "outside-timed-region",
                    "diagnostic_profiler": "disabled",
                },
                "evidence_ref": (
                    "artifact://resolved/instrumentation-profile.json"
                ),
            },
            "correctness": correctness,
            "raw_timing": {
                "schema": (
                    "groundupscale.dev/raw-timing-observation/v1alpha1"
                ),
                "timer_source": "torch.npu.Event.elapsed_time",
                "timer_resolution_ns": timer_resolution_ns,
                "unit": "nanoseconds",
                "sample_derivation": (
                    "device-event-elapsed-ns / inner_iterations"
                ),
                "samples": samples,
                "summary": timing_summary,
            },
            "timing_quality": timing_quality,
            "memory": memory,
            "completion_boundary": {
                "schema": (
                    "groundupscale.dev/completion-boundary/v1alpha1"
                ),
                "kind": "device-event-stream-completion",
                "closed": True,
                "device_event_id": raw["device_event_id"],
                "stream_id": raw["stream_id"],
                "stream_synchronized": True,
                "absolute_timestamps_subtracted": False,
                "protocol": (
                    "end-event-synchronize-plus-device-synchronize"
                ),
            },
            "evidence_ref": "artifact://adapter/collection.json",
        }


__all__ = ["AscendNpuMeasurementAdapter"]
