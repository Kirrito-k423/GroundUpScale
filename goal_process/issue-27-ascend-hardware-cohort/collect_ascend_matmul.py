#!/usr/bin/env python3
"""Collect and merge Issue #27 real Ascend exact-Shape evidence.

This is a ticket-scoped evidence harness, not the public Measurement Adapter
planned by Issue #28.  ``collect`` must run on the target Ascend host.  ``merge``
combines independent process sessions only when their complete cohort identity
is identical.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
from typing import Any


SCHEMA = (
    "groundupscale.dev/ascend-exact-shape-hardware-cohort-evidence/v1alpha1"
)
SESSION_SCHEMA = "groundupscale.dev/ascend-exact-shape-session/v1alpha1"
ADAPTER_ID = "issue27-one-off-ascend-probe"
ADAPTER_VERSION = "v1"
PROTOCOL_ID = "ascend-exact-shape-matmul"
PROTOCOL_VERSION = "v1"
EXPECTED_WARMUP_ITERATIONS = 20
EXPECTED_REPETITIONS = 100
MAX_SESSION_IQR_FRACTION = 0.10
MAX_SESSION_MEDIAN_DEVIATION_FRACTION = 0.05
DISPERSION_POLICY_ID = "issue27-raw-timing-repeatability-v1"
ALLOWED_UNAVAILABLE_STATUSES = {
    "unsupported",
    "permission_denied",
    "not_requested",
    "not_applicable",
    "collection_failed",
    "unknown",
}


class ProbeBlocked(RuntimeError):
    """The real-device probe cannot honestly produce completed evidence."""


def canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.link(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def run_command(*command: str) -> dict[str, object]:
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_sha256": sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": sha256(completed.stderr.encode("utf-8")).hexdigest(),
    }


def require_command(snapshot: dict[str, object], name: str) -> str:
    if snapshot["returncode"] != 0:
        raise ProbeBlocked(f"required-command-failed:{name}")
    return str(snapshot["stdout"])


def parse_colon_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            values[key] = value
    return values


def parse_equals_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def parse_npu_mapping(
    text: str, *, npu_id: int, chip_id: int
) -> dict[str, object]:
    for line in text.splitlines():
        columns = re.split(r"\s{2,}", line.strip())
        if len(columns) < 4 or not columns[0].isdigit() or not columns[1].isdigit():
            continue
        if int(columns[0]) != npu_id or int(columns[1]) != chip_id:
            continue
        if not columns[2].isdigit():
            raise ProbeBlocked("selected-device-has-no-chip-logic-id")
        return {
            "npu_id": int(columns[0]),
            "chip_id": int(columns[1]),
            "chip_logic_id": int(columns[2]),
            "chip_name": columns[3],
        }
    raise ProbeBlocked("selected-device-missing-from-npu-mapping")


def require_visible_device_mapping(
    mapping: dict[str, object], visible_device: str | None
) -> None:
    expected = str(mapping["chip_logic_id"])
    if visible_device != expected:
        raise ProbeBlocked("physical-to-logical-device-selection-mismatch")


def read_snapshot(path: str) -> dict[str, object]:
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as error:
        return {
            "path": path,
            "status": "collection_failed",
            "error": type(error).__name__,
        }
    return {
        "path": path,
        "status": "measured",
        "text": text,
        "sha256": sha256(text.encode("utf-8")).hexdigest(),
    }


def source_ref(session_id: str, section: str) -> str:
    return f"artifact://sessions/{session_id}.json#{section}"


def percentile(samples: list[int], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def timing_summary(samples: list[int]) -> dict[str, float | int]:
    median = statistics.median(samples)
    q1 = percentile(samples, 0.25)
    q3 = percentile(samples, 0.75)
    median_absolute_deviation = statistics.median(
        abs(sample - median) for sample in samples
    )
    return {
        "count": len(samples),
        "minimum": min(samples),
        "p10": percentile(samples, 0.10),
        "q1": q1,
        "median": median,
        "q3": q3,
        "p90": percentile(samples, 0.90),
        "maximum": max(samples),
        "iqr": q3 - q1,
        "iqr_fraction_of_median": (q3 - q1) / median,
        "median_absolute_deviation": median_absolute_deviation,
        "mad_fraction_of_median": median_absolute_deviation / median,
    }


def largest_common_power_of_two(values: list[int]) -> int:
    alignment = 1
    while alignment < 4096 and all(value % (alignment * 2) == 0 for value in values):
        alignment *= 2
    return alignment


def empirical_event_resolution_ns(elapsed_ms: list[float]) -> tuple[float, dict[str, object]]:
    integer_ns = sorted({int(round(value * 1_000_000)) for value in elapsed_ms})
    differences = [
        current - previous
        for previous, current in zip(integer_ns, integer_ns[1:])
        if current > previous
    ]
    if not differences:
        raise ProbeBlocked("device-event-resolution-not-observable")
    resolution = differences[0]
    for difference in differences[1:]:
        resolution = math.gcd(resolution, difference)
    if resolution <= 0:
        raise ProbeBlocked("invalid-device-event-resolution")
    return float(resolution), {
        "method": "gcd-of-distinct-integer-nanosecond-event-sample-differences",
        "documented_api_output_unit": "milliseconds-float",
        "distinct_observed_values": len(integer_ns),
        "difference_count": len(differences),
        "observed_resolution_ns": resolution,
    }


def process_identity() -> dict[str, object]:
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="utf-8"
    ).strip()
    fields = Path(f"/proc/{os.getpid()}/stat").read_text(
        encoding="utf-8"
    ).split()
    return {
        "pid": os.getpid(),
        "process_start_ticks_since_boot": int(fields[21]),
        "boot_id": boot_id,
        "python_executable": str(Path(sys.executable).resolve()),
    }


def collect_command_snapshots(npu_id: int, chip_id: int) -> dict[str, object]:
    npu = str(npu_id)
    chip = str(chip_id)
    snapshots = {
        "npu_summary_before": run_command("npu-smi", "info"),
        "npu_board": run_command(
            "npu-smi", "info", "-t", "board", "-i", npu, "-c", chip
        ),
        "npu_memory_before": run_command(
            "npu-smi", "info", "-t", "memory", "-i", npu, "-c", chip
        ),
        "npu_usages_before": run_command(
            "npu-smi", "info", "-t", "usages", "-i", npu, "-c", chip
        ),
        "npu_sensors_before": run_command(
            "npu-smi", "info", "-t", "sensors", "-i", npu, "-c", chip
        ),
        "npu_health_before": run_command(
            "npu-smi", "info", "-t", "health", "-i", npu, "-c", chip
        ),
        "npu_power_before": run_command(
            "npu-smi", "info", "-t", "power", "-i", npu, "-c", chip
        ),
        "npu_work_mode": run_command(
            "npu-smi", "info", "-t", "work-mode", "-i", npu, "-c", chip
        ),
        "npu_topology": run_command("npu-smi", "info", "-t", "topo", "-i", npu, "-c", chip),
        "npu_hccs": run_command("npu-smi", "info", "-t", "hccs", "-i", npu, "-c", chip),
        "npu_mapping": run_command("npu-smi", "info", "-m"),
        "npu_inventory": run_command("npu-smi", "info", "-l"),
    }
    board = parse_colon_values(
        require_command(snapshots["npu_board"], "npu-board")
    )
    bus = board.get("PCIe Bus Info")
    if not bus:
        raise ProbeBlocked("missing-pcie-bus-id")
    snapshots["lspci_device"] = run_command("lspci", "-s", bus, "-nn", "-vv")
    return snapshots


def collect_static_identity(
    *,
    machine_name: str,
    session_id: str,
    npu_id: int,
    chip_id: int,
    snapshots: dict[str, object],
    torch_version: str,
    torch_npu_version: str,
    python_version: str,
) -> tuple[dict[str, object], dict[str, object]]:
    board_text = require_command(snapshots["npu_board"], "npu-board")
    memory_text = require_command(snapshots["npu_memory_before"], "npu-memory")
    topology_text = require_command(snapshots["npu_topology"], "npu-topology")
    hccs_text = require_command(snapshots["npu_hccs"], "npu-hccs")
    lspci_text = require_command(snapshots["lspci_device"], "lspci-device")
    board = parse_colon_values(board_text)
    memory = parse_colon_values(memory_text)
    topology_digest = sha256(topology_text.encode("utf-8")).hexdigest()
    stable_hccs_lines = [
        line.strip()
        for line in hccs_text.splitlines()
        if line.strip().startswith(
            (
                "hccs health status",
                "hccs lane mode",
                "hccs link lane list",
                "hccs link speed",
            )
        )
    ]
    if not stable_hccs_lines:
        raise ProbeBlocked("missing-stable-hccs-identity")
    hccs_digest = canonical_digest(stable_hccs_lines)
    serial_match = re.search(r"Device Serial Number\s+([^\n]+)", lspci_text)
    numa_match = re.search(r"NUMA node:\s*(\d+)", lspci_text)

    required_board_fields = (
        "Chip Name",
        "Chip Version",
        "VDie ID",
        "PCIe Bus Info",
        "Firmware Version",
    )
    if any(not board.get(field) for field in required_board_fields):
        raise ProbeBlocked("incomplete-board-identity")

    driver_snapshot = read_snapshot("/usr/local/Ascend/driver/version.info")
    firmware_snapshot = read_snapshot("/usr/local/Ascend/firmware/version.info")
    cann_snapshot = read_snapshot(
        "/usr/local/Ascend/ascend-toolkit/latest/compiler/version.info"
    )
    os_snapshot = read_snapshot("/etc/os-release")
    for name, snapshot in (
        ("driver", driver_snapshot),
        ("firmware", firmware_snapshot),
        ("cann", cann_snapshot),
        ("os", os_snapshot),
    ):
        if snapshot["status"] != "measured":
            raise ProbeBlocked(f"missing-software-identity:{name}")

    driver = parse_equals_values(str(driver_snapshot["text"]))
    firmware = parse_equals_values(str(firmware_snapshot["text"]))
    cann = parse_equals_values(str(cann_snapshot["text"]))
    os_release = parse_equals_values(str(os_snapshot["text"]))
    snapshots["driver_version"] = driver_snapshot
    snapshots["firmware_version"] = firmware_snapshot
    snapshots["cann_version"] = cann_snapshot
    snapshots["os_release"] = os_snapshot

    hbm_capacity_mb = int(memory["HBM Capacity(MB)"])
    sources = {
        "device_identity": source_ref(session_id, "command_snapshots/npu_board"),
        "pcie_and_numa": source_ref(session_id, "command_snapshots/lspci_device"),
        "topology": source_ref(session_id, "command_snapshots/npu_topology"),
        "hccs": source_ref(session_id, "command_snapshots/npu_hccs"),
        "memory": source_ref(session_id, "command_snapshots/npu_memory_before"),
        "power": source_ref(session_id, "command_snapshots/npu_power_before"),
        "frequency": source_ref(session_id, "command_snapshots/npu_memory_before"),
    }
    hardware = {
        "machine_name": machine_name,
        "device_name": board["Chip Name"],
        "device_version": board["Chip Version"],
        "logical_device": "npu:0",
        "physical_device": {"npu_id": npu_id, "chip_id": chip_id},
        "pcie_bus_id": board["PCIe Bus Info"],
        "vdie_id": board["VDie ID"],
        "device_serial_number": serial_match.group(1).strip() if serial_match else "unknown",
        "firmware_reported_by_board": board["Firmware Version"],
        "numa_node": int(numa_match.group(1)) if numa_match else "unknown",
        "hbm_capacity_bytes": hbm_capacity_mb * 1024 * 1024,
        "topology_sha256": topology_digest,
        "hccs_sha256": hccs_digest,
        "fallback_detected": False,
        "sources": sources,
    }
    software = {
        "os": {
            "value": os_release.get("PRETTY_NAME", os_release.get("NAME", "unknown")),
            "source": source_ref(session_id, "command_snapshots/os_release"),
        },
        "kernel": {
            "value": platform.release(),
            "source": "python://platform.release",
        },
        "driver": {
            "value": driver.get("Version", driver.get("package_version", "unknown")),
            "source": source_ref(session_id, "command_snapshots/driver_version"),
        },
        "firmware": {
            "value": firmware.get("Version", firmware.get("package_version", "unknown")),
            "source": source_ref(session_id, "command_snapshots/firmware_version"),
        },
        "cann": {
            "value": cann.get("Version", "unknown"),
            "source": source_ref(session_id, "command_snapshots/cann_version"),
        },
        "python": {"value": python_version, "source": "python://sys.version"},
        "torch": {"value": torch_version, "source": "python://torch.__version__"},
        "torch_npu": {
            "value": torch_npu_version,
            "source": "python://torch_npu.__version__",
        },
    }
    return hardware, software


def observation_field(
    name: str,
    status: str,
    *,
    source: str,
    value: object | None = None,
    required_for_anchor: bool = False,
    scope: str = "exact-shape-matmul",
    attribution: str = "direct",
    intrusion: str = "baseline",
    derivation: dict[str, object] | None = None,
    notes: list[str] | None = None,
) -> dict[str, object]:
    field: dict[str, object] = {
        "field": name,
        "status": status,
        "required_for_anchor": required_for_anchor,
        "source": source,
        "scope": scope,
        "attribution": attribution,
        "intrusion": intrusion,
    }
    if status not in ALLOWED_UNAVAILABLE_STATUSES:
        field["value"] = value
    if derivation is not None:
        field["derivation"] = derivation
    if notes:
        field["notes"] = notes
    return field


def measure_device_interval(torch: Any, invoke: Any) -> tuple[Any, float, int]:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    host_start = time_perf_counter_ns()
    start.record()
    result = invoke()
    end.record()
    end.synchronize()
    torch.npu.synchronize()
    host_end = time_perf_counter_ns()
    return result, float(start.elapsed_time(end)), host_end - host_start


def measure_transfer(torch: Any, *, direction: str, source: Any) -> dict[str, object]:
    if direction == "h2d":
        invoke = lambda: source.to("npu:0")
    elif direction == "d2h":
        invoke = lambda: source.to("cpu")
    else:
        raise ValueError(direction)
    result, elapsed_ms, host_visible_ns = measure_device_interval(torch, invoke)
    return {
        "direction": direction,
        "raw_elapsed_ms": elapsed_ms,
        "elapsed_ns": int(round(elapsed_ms * 1_000_000)),
        "host_visible_elapsed_ns": host_visible_ns,
        "bytes": int(result.numel() * result.element_size()),
        "completion_boundary": "end-event-synchronize-plus-device-synchronize",
    }


def collect_session(args: argparse.Namespace) -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    blocked: dict[str, object] = {
        "schema": SESSION_SCHEMA,
        "session_id": args.session_id,
        "status": "blocked",
        "started_at": started_at,
        "reason_codes": [],
    }
    try:
        snapshots = collect_command_snapshots(args.npu_id, args.chip_id)
        mapping = parse_npu_mapping(
            require_command(snapshots["npu_mapping"], "npu-mapping"),
            npu_id=args.npu_id,
            chip_id=args.chip_id,
        )
        work_mode_output = require_command(
            snapshots["npu_work_mode"], "npu-work-mode"
        ).strip()
        if "does not support querying work-mode" in work_mode_output.lower():
            power_policy_status = "unsupported"
            power_policy_identity = "unsupported(npu-smi-work-mode-query)"
        elif work_mode_output:
            power_policy_status = "declared"
            power_policy_identity = (
                "declared(npu-smi-work-mode-sha256="
                f"{sha256(work_mode_output.encode('utf-8')).hexdigest()})"
            )
        else:
            raise ProbeBlocked("empty-power-policy-evidence")
        import torch
        import torch_npu

        if not torch_npu.npu.is_available():
            raise ProbeBlocked("npu-unavailable")
        if torch_npu.npu.device_count() != 1:
            raise ProbeBlocked("visible-device-selection-not-singleton")
        visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        require_visible_device_mapping(mapping, visible_devices)
        torch.npu.set_device(0)

        board = parse_colon_values(
            require_command(snapshots["npu_board"], "npu-board")
        )
        health = parse_colon_values(
            require_command(snapshots["npu_health_before"], "npu-health")
        )
        usages = parse_colon_values(
            require_command(snapshots["npu_usages_before"], "npu-usages")
        )
        sensors = parse_colon_values(
            require_command(snapshots["npu_sensors_before"], "npu-sensors")
        )
        memory_before = parse_colon_values(
            require_command(snapshots["npu_memory_before"], "npu-memory")
        )
        power_before = parse_colon_values(
            require_command(snapshots["npu_power_before"], "npu-power")
        )
        summary = require_command(snapshots["npu_summary_before"], "npu-summary")
        if health.get("Health Status") != "OK":
            raise ProbeBlocked("npu-health-not-ok")
        no_process_marker = f"No running processes found in NPU {args.npu_id}"
        if no_process_marker not in summary:
            raise ProbeBlocked("selected-npu-has-competing-process")

        hardware, software = collect_static_identity(
            machine_name=args.machine_name,
            session_id=args.session_id,
            npu_id=args.npu_id,
            chip_id=args.chip_id,
            snapshots=snapshots,
            torch_version=str(torch.__version__),
            torch_npu_version=str(torch_npu.__version__),
            python_version=sys.version,
        )
        runtime_device_name = str(torch.npu.get_device_name(0))
        if torch.npu.current_device() != 0:
            raise ProbeBlocked("logical-device-selection-mismatch")
        if (
            str(board["Chip Name"]) not in runtime_device_name
            or str(mapping["chip_name"]).replace(" ", "") not in runtime_device_name
        ):
            raise ProbeBlocked("runtime-and-board-device-name-mismatch")
        hardware["device_selection"] = {
            "environment_variable": "ASCEND_RT_VISIBLE_DEVICES",
            "environment_value": visible_devices,
            "physical_npu_id": args.npu_id,
            "physical_chip_id": args.chip_id,
            "visible_chip_logic_id": mapping["chip_logic_id"],
            "logical_device_index": 0,
            "logical_device": "npu:0",
            "runtime_device_name": runtime_device_name,
            "runtime_current_device": torch.npu.current_device(),
            "host_mapping_record": mapping,
            "mapping_source": source_ref(
                args.session_id, "command_snapshots/npu_mapping"
            ),
        }

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        left_cpu = torch.randn(
            (args.m, args.k), dtype=torch.float32, generator=generator
        )
        right_cpu = torch.randn(
            (args.k, args.n), dtype=torch.float32, generator=generator
        )
        oracle = torch.matmul(left_cpu.double(), right_cpu.double())
        left = left_cpu.to("npu:0")
        right = right_cpu.to("npu:0")
        torch.npu.synchronize()
        alignment_bytes = largest_common_power_of_two(
            [int(left.data_ptr()), int(right.data_ptr())]
        )

        warmup_elapsed_ms: list[float] = []
        for _ in range(args.warmup):
            warmup_result, elapsed_ms, _ = measure_device_interval(
                torch, lambda: torch.matmul(left, right)
            )
            if warmup_result.device.type != "npu":
                raise ProbeBlocked("cpu-fallback-detected")
            warmup_elapsed_ms.append(elapsed_ms)

        actual = torch.matmul(left, right)
        torch.npu.synchronize()
        if actual.device.type != "npu":
            raise ProbeBlocked("cpu-fallback-detected")
        actual_cpu = actual.cpu().double()
        absolute_error = (actual_cpu - oracle).abs()
        relative_error = absolute_error / oracle.abs().clamp_min(1e-12)
        finite = bool(torch.isfinite(actual_cpu).all() and torch.isfinite(oracle).all())
        shape_exact = tuple(actual_cpu.shape) == tuple(oracle.shape)
        passed = bool(
            finite
            and shape_exact
            and torch.allclose(
                actual_cpu,
                oracle,
                atol=args.atol,
                rtol=args.rtol,
            )
        )
        correctness = {
            "status": "passed" if passed else "failed",
            "oracle": "cpu-float64-matmul",
            "candidate": "torch.matmul",
            "candidate_device": str(actual.device),
            "cpu_fallback": actual.device.type == "cpu",
            "shape_exact": shape_exact,
            "finite": finite,
            "atol": args.atol,
            "rtol": args.rtol,
            "max_absolute_error": float(absolute_error.max().item()),
            "max_relative_error": float(relative_error.max().item()),
        }
        if not passed:
            raise ProbeBlocked("cpu-correctness-oracle-failed")

        raw_elapsed_ms: list[float] = []
        raw_samples_ns: list[int] = []
        host_visible_samples_ns: list[int] = []
        for _ in range(args.repeats):
            result, elapsed_ms, host_visible_ns = measure_device_interval(
                torch, lambda: torch.matmul(left, right)
            )
            if result.device.type != "npu":
                raise ProbeBlocked("cpu-fallback-detected")
            raw_elapsed_ms.append(elapsed_ms)
            raw_samples_ns.append(int(round(elapsed_ms * 1_000_000)))
            host_visible_samples_ns.append(host_visible_ns)

        h2d = measure_transfer(torch, direction="h2d", source=left_cpu)
        d2h = measure_transfer(torch, direction="d2h", source=left)
        memory_framework = {
            "allocated_bytes": int(torch.npu.memory_allocated()),
            "reserved_bytes": int(torch.npu.memory_reserved()),
            "maximum_allocated_bytes": int(torch.npu.max_memory_allocated()),
        }
        after_snapshots = {
            "npu_memory_after": run_command(
                "npu-smi",
                "info",
                "-t",
                "memory",
                "-i",
                str(args.npu_id),
                "-c",
                str(args.chip_id),
            ),
            "npu_usages_after": run_command(
                "npu-smi",
                "info",
                "-t",
                "usages",
                "-i",
                str(args.npu_id),
                "-c",
                str(args.chip_id),
            ),
            "npu_sensors_after": run_command(
                "npu-smi",
                "info",
                "-t",
                "sensors",
                "-i",
                str(args.npu_id),
                "-c",
                str(args.chip_id),
            ),
            "npu_health_after": run_command(
                "npu-smi",
                "info",
                "-t",
                "health",
                "-i",
                str(args.npu_id),
                "-c",
                str(args.chip_id),
            ),
            "npu_power_after": run_command(
                "npu-smi",
                "info",
                "-t",
                "power",
                "-i",
                str(args.npu_id),
                "-c",
                str(args.chip_id),
            ),
        }
        snapshots.update(after_snapshots)
        memory_after = parse_colon_values(
            require_command(after_snapshots["npu_memory_after"], "npu-memory-after")
        )
        usages_after = parse_colon_values(
            require_command(after_snapshots["npu_usages_after"], "npu-usages-after")
        )
        health_after = parse_colon_values(
            require_command(after_snapshots["npu_health_after"], "npu-health-after")
        )
        power_after = parse_colon_values(
            require_command(after_snapshots["npu_power_after"], "npu-power-after")
        )
        if health_after.get("Health Status") != "OK":
            raise ProbeBlocked("npu-health-not-ok-after-probe")

        timer_resolution_ns, resolution_evidence = empirical_event_resolution_ns(
            raw_elapsed_ms
        )
        event_source_name = inspect.getsourcefile(torch.npu.Event)
        if event_source_name is None:
            raise ProbeBlocked("torch-npu-event-source-unavailable")
        event_source_path = Path(event_source_name).resolve()
        event_id = "per-sample-start-end-event-pair"
        stream_id = "logical-npu:0-default-stream"
        completion_boundary = {
            "kind": "device-event-stream-completion",
            "closed": True,
            "device_event_id": event_id,
            "stream_id": stream_id,
            "stream_synchronized": True,
            "absolute_timestamps_subtracted": False,
            "protocol": [
                "device-wide synchronize before sample",
                "record start event on default stream",
                "enqueue exactly one torch.matmul",
                "record end event on the same stream",
                "synchronize end event and device before reading elapsed time",
            ],
        }
        timer = {
            "kind": "device-event",
            "source": "torch.npu.Event.elapsed_time",
            "resolution_ns": timer_resolution_ns,
            "resolution_kind": "empirically-observed-output-step",
            "resolution_evidence": resolution_evidence,
            "source_evidence": {
                "installed_source_path": str(event_source_path),
                "installed_source_sha256": file_digest(event_source_path),
                "official_api_reference": (
                    "https://www.hiascend.com/document/detail/en/"
                    "canncommercial/850/API/appdevgapi/aclcppdevg_03_0090.html"
                ),
            },
            "monotonic": True,
            "device_event_id": event_id,
            "stream_id": stream_id,
        }
        clock = time_clock_info()
        execution_domain = {
            "semantic_operation": "matrix-multiply",
            "shape": {"left": [args.m, args.k], "right": [args.k, args.n]},
            "dtype": "float32",
            "accumulation_mode": "torch-npu-provider-default-fp32",
            "layout": "row-major-contiguous",
            "strides": {
                "left": list(left.stride()),
                "right": list(right.stride()),
                "result": list(actual.stride()),
            },
            "alignment_bytes": alignment_bytes,
            "threads": 1,
            "execution_mode": "torch-eager",
            "affinity": f"npu-smi-topology-cpu-affinity-for-npu{args.npu_id}",
            "numa": f"node-{hardware['numa_node']}",
            "context": "torch-npu-default-device-context",
            "stream": stream_id,
            "concurrency": 1,
        }
        software_identity = ";".join(
            f"{key}={software[key]['value']}" for key in sorted(software)
        )
        measurement_policy = {
            "policy_id": DISPERSION_POLICY_ID,
            "warmup_iterations": args.warmup,
            "repetitions": args.repeats,
            "maximum_session_iqr_fraction_of_median": MAX_SESSION_IQR_FRACTION,
            "maximum_session_median_deviation_fraction": (
                MAX_SESSION_MEDIAN_DEVIATION_FRACTION
            ),
            "sample_exclusion": "none-preserve-all-raw-samples",
            "frontier_qualification": "not-evaluated-by-issue-27",
        }
        cohort_identity = {
            "device": (
                f"{hardware['device_name']}-{hardware['device_version']}"
                f"/vdie={hardware['vdie_id']}"
            ),
            "partition": (
                f"physical-npu-{args.npu_id}/chip-{args.chip_id}/"
                f"chip-logic-{mapping['chip_logic_id']}/"
                f"ascend-rt-visible-devices={visible_devices}"
            ),
            "topology": (
                f"machine={args.machine_name};bus={hardware['pcie_bus_id']};"
                f"numa={hardware['numa_node']};"
                f"topology_sha256={hardware['topology_sha256']};"
                f"hccs_sha256={hardware['hccs_sha256']}"
            ),
            "software": software_identity,
            "numeric_execution": {
                "dtype": execution_domain["dtype"],
                "layout": execution_domain["layout"],
                "alignment_bytes": alignment_bytes,
                "threads": 1,
                "execution_mode": execution_domain["execution_mode"],
            },
            "timer_protocol": {
                "source": timer["source"],
                "resolution_ns": timer["resolution_ns"],
                "monotonic": timer["monotonic"],
                "completion_kind": completion_boundary["kind"],
                "duration_reducer": None,
                "adapter_id": ADAPTER_ID,
                "adapter_version": ADAPTER_VERSION,
                "protocol_id": PROTOCOL_ID,
                "protocol_version": PROTOCOL_VERSION,
                "measurement_policy": measurement_policy,
            },
            "power_clock": {
                "power_policy": power_policy_identity,
                "clock_policy": (
                    f"hbm={memory_before.get('HBM Clock Speed(MHz)', 'unknown')}MHz;"
                    "ai-core-frequency=unsupported-by-npu-smi-query-set"
                ),
            },
            "execution_context": {
                "affinity": execution_domain["affinity"],
                "numa": execution_domain["numa"],
                "context": execution_domain["context"],
                "stream": execution_domain["stream"],
                "concurrency": execution_domain["concurrency"],
            },
            "communication": {"status": "not_applicable"},
        }
        cohort_digest = canonical_digest(cohort_identity)
        cohort_id = f"ascend-910b2-issue27-{cohort_digest[:16]}"
        session_ref = source_ref(args.session_id, "session")
        device_summary = timing_summary(raw_samples_ns)
        host_summary = timing_summary(host_visible_samples_ns)
        session_dispersion_passed = (
            device_summary["iqr_fraction_of_median"]
            <= MAX_SESSION_IQR_FRACTION
        )
        timing_quality = {
            "policy_id": DISPERSION_POLICY_ID,
            "status": "passed" if session_dispersion_passed else "quarantined",
            "observed_iqr_fraction_of_median": device_summary[
                "iqr_fraction_of_median"
            ],
            "maximum_iqr_fraction_of_median": MAX_SESSION_IQR_FRACTION,
            "excluded_samples": 0,
            "reason_codes": (
                []
                if session_dispersion_passed
                else ["session-dispersion-exceeds-policy"]
            ),
        }
        fields = [
            observation_field(
                "timer.primary",
                "measured",
                source=session_ref,
                value=device_summary["median"],
                required_for_anchor=True,
                notes=[
                    "value is median device-event nanoseconds converted from the "
                    "documented millisecond float output",
                    f"empirical output step is {timer_resolution_ns:g} ns",
                ],
            ),
            observation_field(
                "timer.host_visible_completion",
                "measured",
                source="python://time.perf_counter_ns",
                value=host_summary["median"],
                notes=[f"clock resolution is {clock['resolution_ns']} ns"],
            ),
            observation_field(
                "synchronization.device_stream",
                "declared",
                source=session_ref,
                value=completion_boundary,
            ),
            observation_field(
                "memory.framework",
                "measured",
                source="python://torch.npu.memory_*",
                value=memory_framework,
            ),
            observation_field(
                "memory.hbm_device_wide",
                "measured",
                source=source_ref(args.session_id, "command_snapshots/npu_memory_after"),
                value={
                    "capacity_mb": int(memory_after["HBM Capacity(MB)"]),
                    "usage_rate_percent": int(usages_after["HBM Usage Rate(%)"]),
                },
                scope="selected-device-device-wide",
                attribution="device-wide-not-operator-attributed",
                intrusion="diagnostic-after-baseline",
            ),
            observation_field(
                "power.device_wide",
                "measured",
                source=source_ref(args.session_id, "command_snapshots/npu_power_after"),
                value=float(power_after["NPU Real-time Power(W)"]),
                scope="selected-device-device-wide",
                attribution="device-wide-not-operator-attributed",
                intrusion="diagnostic-after-baseline",
            ),
            observation_field(
                "power.policy",
                power_policy_status,
                source=source_ref(
                    args.session_id, "command_snapshots/npu_work_mode"
                ),
                value=(
                    power_policy_identity
                    if power_policy_status == "declared"
                    else None
                ),
                scope="selected-device",
                attribution="device-wide",
                intrusion="preflight",
                required_for_anchor=True,
                notes=[
                    "Issue #27 preserves unavailable policy evidence and does "
                    "not claim Frontier eligibility"
                ],
            ),
            observation_field(
                "frequency.hbm",
                "measured",
                source=source_ref(args.session_id, "command_snapshots/npu_memory_before"),
                value=int(memory_before["HBM Clock Speed(MHz)"]),
                scope="selected-device",
                attribution="device-wide",
                intrusion="preflight",
            ),
            observation_field(
                "frequency.ai_core",
                "unsupported",
                source="npu-smi://25.3.rc1/query-types",
                scope="selected-device",
                attribution="none",
                intrusion="preflight",
                notes=["npu-smi exposes HBM clock but no AI Core frequency query type"],
            ),
            observation_field(
                "transfer.h2d",
                "measured",
                source=session_ref,
                value=h2d,
                scope="isolated-tensor-copy",
            ),
            observation_field(
                "transfer.d2h",
                "measured",
                source=session_ref,
                value=d2h,
                scope="isolated-tensor-copy",
            ),
            observation_field(
                "attribution.device_kernel",
                "derived",
                source=session_ref,
                value="isolated-single-matmul-device-event-interval",
                attribution="isolated-operation-not-kernel-name",
                derivation={
                    "method": "one-matmul-enqueued-between-same-stream-device-events",
                    "input_evidence_refs": [
                        source_ref(args.session_id, "raw_elapsed_ms"),
                        source_ref(args.session_id, "completion_boundary"),
                    ],
                },
                notes=["does not identify the underlying kernel binary or tiling"],
            ),
            observation_field(
                "profiling.operator_timeline",
                "not_requested" if hasattr(torch_npu, "profiler") else "unsupported",
                source="python://torch_npu.profiler",
                attribution="none",
                intrusion="diagnostic-profiling-lane",
                notes=[
                    "API capability was probed; profiling was intentionally excluded "
                    "from baseline"
                ],
            ),
            observation_field(
                "counter.ai_core",
                "unsupported",
                source="npu-smi://25.3.rc1/query-types",
                scope="selected-device",
                attribution="none",
                intrusion="diagnostic-profiling-lane",
                notes=[
                    "the available npu-smi query set exposes device-wide AI Core "
                    "utilization, not an operator-attributed hardware counter"
                ],
            ),
        ]
        manifest = {
            "manifest_id": f"issue27-ascend-capabilities-{cohort_digest[:16]}-v1",
            "adapter_id": ADAPTER_ID,
            "cohort_id": cohort_id,
            "fields": fields,
            "evidence_ref": source_ref(args.session_id, "measurement_capability_manifest"),
        }
        environment = {
            "status": "eligible-for-issue27-evidence-collection",
            "frontier_eligibility": (
                "eligible"
                if power_policy_status == "declared"
                else "ineligible-missing-power-policy-evidence"
            ),
            "health_before": health,
            "health_after": health_after,
            "usage_before": usages,
            "sensors_before": sensors,
            "power_before": power_before,
            "power_after": power_after,
            "hbm_before": memory_before,
            "hbm_after": memory_after,
            "selected_npu_competing_processes": 0,
            "power_policy_status": power_policy_status,
            "power_policy_source": source_ref(
                args.session_id, "command_snapshots/npu_work_mode"
            ),
            "ai_core_frequency_status": "unsupported",
        }
        session = {
            "schema": SESSION_SCHEMA,
            "session_id": args.session_id,
            "status": "completed",
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "process_identity": process_identity(),
            "hardware": hardware,
            "software": software,
            "execution_domain": execution_domain,
            "cohort_identity": cohort_identity,
            "cohort_digest": cohort_digest,
            "cohort_id": cohort_id,
            "measurement_capability_manifest": manifest,
            "measurement_policy": measurement_policy,
            "environment": environment,
            "benchmark_case": {
                "operation": "MatMul",
                "seed": args.seed,
                "shape": execution_domain["shape"],
                "dtype": execution_domain["dtype"],
                "layout": execution_domain["layout"],
                "candidate": "torch.matmul",
                "input_initialization": "cpu-torch-randn-fixed-seed",
            },
            "correctness": correctness,
            "warmup": {
                "iterations": args.warmup,
                "synchronized": True,
                "raw_elapsed_ms": warmup_elapsed_ms,
            },
            "timer": timer,
            "host_timer": clock,
            "completion_boundary": completion_boundary,
            "raw_elapsed_ms": raw_elapsed_ms,
            "raw_samples_ns": raw_samples_ns,
            "host_visible_samples_ns": host_visible_samples_ns,
            "summary_ns": device_summary,
            "timing_quality": timing_quality,
            "host_visible_summary_ns": host_summary,
            "transfer_probe": {"h2d": h2d, "d2h": d2h},
            "memory_framework": memory_framework,
            "command_snapshots": snapshots,
        }
        if not session_dispersion_passed:
            session["status"] = "quarantined"
            session["reason_codes"] = timing_quality["reason_codes"]
        write_json(args.output, session)
        return 0 if session_dispersion_passed else 2
    except Exception as error:
        reason = str(error) if isinstance(error, ProbeBlocked) else "runtime-error"
        blocked["reason_codes"] = [reason]
        blocked["error_type"] = type(error).__name__
        blocked["error_message"] = str(error)[:500]
        blocked["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output, blocked)
        return 2


def time_perf_counter_ns() -> int:
    import time

    return time.perf_counter_ns()


def time_clock_info() -> dict[str, object]:
    import time

    info = time.get_clock_info("perf_counter")
    return {
        "source": info.implementation,
        "monotonic": info.monotonic,
        "adjustable": info.adjustable,
        "resolution_ns": info.resolution * 1_000_000_000,
    }


def validate_completed_session(document: dict[str, Any], *, source_name: str) -> None:
    if document.get("schema") != SESSION_SCHEMA:
        raise ProbeBlocked(f"invalid-session-schema:{source_name}")
    if document.get("status") != "completed":
        raise ProbeBlocked(f"session-not-completed:{source_name}")
    identity = document.get("cohort_identity")
    if not isinstance(identity, dict):
        raise ProbeBlocked(f"missing-cohort-identity:{source_name}")
    calculated_digest = canonical_digest(identity)
    if document.get("cohort_digest") != calculated_digest:
        raise ProbeBlocked(f"invalid-cohort-digest:{source_name}")
    expected_cohort_id = f"ascend-910b2-issue27-{calculated_digest[:16]}"
    if document.get("cohort_id") != expected_cohort_id:
        raise ProbeBlocked(f"invalid-cohort-id:{source_name}")
    manifest = document.get("measurement_capability_manifest")
    if not isinstance(manifest, dict) or (
        manifest.get("adapter_id") != ADAPTER_ID
        or manifest.get("cohort_id") != expected_cohort_id
    ):
        raise ProbeBlocked(f"invalid-capability-manifest-identity:{source_name}")
    manifest_fields = manifest.get("fields")
    power_policy_field = (
        next(
            (
                field
                for field in manifest_fields
                if isinstance(field, dict) and field.get("field") == "power.policy"
            ),
            None,
        )
        if isinstance(manifest_fields, list)
        else None
    )
    if not isinstance(power_policy_field, dict):
        raise ProbeBlocked(f"missing-power-policy-capability:{source_name}")
    benchmark = document.get("benchmark_case")
    if benchmark != {
        "operation": "MatMul",
        "seed": 20260810,
        "shape": {"left": [512, 512], "right": [512, 512]},
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "candidate": "torch.matmul",
        "input_initialization": "cpu-torch-randn-fixed-seed",
    }:
        raise ProbeBlocked(f"invalid-frozen-benchmark-case:{source_name}")
    correctness = document.get("correctness")
    if not isinstance(correctness, dict) or (
        correctness.get("status") != "passed"
        or correctness.get("oracle") != "cpu-float64-matmul"
        or correctness.get("candidate") != "torch.matmul"
        or correctness.get("candidate_device") != "npu:0"
        or correctness.get("cpu_fallback") is not False
        or correctness.get("shape_exact") is not True
        or correctness.get("finite") is not True
        or correctness.get("atol") != 1e-3
        or correctness.get("rtol") != 1e-3
    ):
        raise ProbeBlocked(f"invalid-correctness-or-device-evidence:{source_name}")
    policy = document.get("measurement_policy")
    if not isinstance(policy, dict) or (
        policy.get("policy_id") != DISPERSION_POLICY_ID
        or policy.get("warmup_iterations") != EXPECTED_WARMUP_ITERATIONS
        or policy.get("repetitions") != EXPECTED_REPETITIONS
        or policy.get("maximum_session_iqr_fraction_of_median")
        != MAX_SESSION_IQR_FRACTION
        or policy.get("maximum_session_median_deviation_fraction")
        != MAX_SESSION_MEDIAN_DEVIATION_FRACTION
        or policy.get("sample_exclusion")
        != "none-preserve-all-raw-samples"
    ):
        raise ProbeBlocked(f"invalid-measurement-policy:{source_name}")
    if identity.get("timer_protocol", {}).get("measurement_policy") != policy:
        raise ProbeBlocked(f"measurement-policy-not-in-cohort:{source_name}")
    warmup = document.get("warmup")
    if not isinstance(warmup, dict) or (
        warmup.get("iterations") != EXPECTED_WARMUP_ITERATIONS
        or warmup.get("synchronized") is not True
        or not isinstance(warmup.get("raw_elapsed_ms"), list)
        or len(warmup["raw_elapsed_ms"]) != EXPECTED_WARMUP_ITERATIONS
        or not all(
            isinstance(sample, (int, float)) and sample > 0
            for sample in warmup["raw_elapsed_ms"]
        )
    ):
        raise ProbeBlocked(f"invalid-warmup-evidence:{source_name}")
    boundary = document.get("completion_boundary")
    timer = document.get("timer")
    if not isinstance(boundary, dict) or not isinstance(timer, dict) or (
        boundary.get("kind") != "device-event-stream-completion"
        or boundary.get("closed") is not True
        or boundary.get("stream_synchronized") is not True
        or boundary.get("absolute_timestamps_subtracted") is not False
        or boundary.get("device_event_id")
        != "per-sample-start-end-event-pair"
        or boundary.get("stream_id") != "logical-npu:0-default-stream"
        or timer.get("source") != "torch.npu.Event.elapsed_time"
        or timer.get("resolution_ns") != identity["timer_protocol"]["resolution_ns"]
        or timer.get("resolution_kind")
        != "empirically-observed-output-step"
    ):
        raise ProbeBlocked(f"invalid-timer-or-completion-boundary:{source_name}")
    process = document.get("process_identity")
    if not isinstance(process, dict) or not all(
        key in process
        for key in ("pid", "process_start_ticks_since_boot", "boot_id", "python_executable")
    ):
        raise ProbeBlocked(f"missing-process-identity:{source_name}")
    samples = document.get("raw_samples_ns")
    raw_elapsed_ms = document.get("raw_elapsed_ms")
    if (
        not isinstance(samples, list)
        or len(samples) != EXPECTED_REPETITIONS
        or not all(isinstance(sample, int) and sample > 0 for sample in samples)
        or not isinstance(raw_elapsed_ms, list)
        or len(raw_elapsed_ms) != EXPECTED_REPETITIONS
        or [int(round(float(value) * 1_000_000)) for value in raw_elapsed_ms]
        != samples
    ):
        raise ProbeBlocked(f"invalid-raw-samples:{source_name}")
    calculated_resolution, _ = empirical_event_resolution_ns(raw_elapsed_ms)
    if calculated_resolution != timer.get("resolution_ns"):
        raise ProbeBlocked(f"invalid-observed-timer-step:{source_name}")
    calculated_summary = timing_summary(samples)
    if document.get("summary_ns") != calculated_summary:
        raise ProbeBlocked(f"invalid-timing-summary:{source_name}")
    timing_quality = document.get("timing_quality")
    if not isinstance(timing_quality, dict) or (
        timing_quality.get("policy_id") != DISPERSION_POLICY_ID
        or timing_quality.get("status") != "passed"
        or timing_quality.get("excluded_samples") != 0
        or timing_quality.get("reason_codes") != []
        or timing_quality.get("observed_iqr_fraction_of_median")
        != calculated_summary["iqr_fraction_of_median"]
        or calculated_summary["iqr_fraction_of_median"]
        > MAX_SESSION_IQR_FRACTION
    ):
        raise ProbeBlocked(f"session-dispersion-exceeds-policy:{source_name}")
    host_samples = document.get("host_visible_samples_ns")
    if not isinstance(host_samples, list) or len(host_samples) != EXPECTED_REPETITIONS:
        raise ProbeBlocked(f"invalid-host-visible-samples:{source_name}")
    hardware = document.get("hardware")
    snapshots = document.get("command_snapshots")
    if not isinstance(hardware, dict) or not isinstance(snapshots, dict):
        raise ProbeBlocked(f"missing-device-mapping-evidence:{source_name}")
    physical = hardware.get("physical_device")
    selection = hardware.get("device_selection")
    mapping_snapshot = snapshots.get("npu_mapping")
    if (
        not isinstance(physical, dict)
        or not isinstance(selection, dict)
        or not isinstance(mapping_snapshot, dict)
    ):
        raise ProbeBlocked(f"missing-device-mapping-evidence:{source_name}")
    parsed_mapping = parse_npu_mapping(
        require_command(mapping_snapshot, "npu-mapping"),
        npu_id=int(physical.get("npu_id", -1)),
        chip_id=int(physical.get("chip_id", -1)),
    )
    expected_partition = (
        f"physical-npu-{physical.get('npu_id')}/chip-{physical.get('chip_id')}/"
        f"chip-logic-{parsed_mapping['chip_logic_id']}/"
        f"ascend-rt-visible-devices={selection.get('environment_value')}"
    )
    if (
        selection.get("host_mapping_record") != parsed_mapping
        or selection.get("physical_npu_id") != physical.get("npu_id")
        or selection.get("physical_chip_id") != physical.get("chip_id")
        or selection.get("environment_variable")
        != "ASCEND_RT_VISIBLE_DEVICES"
        or selection.get("environment_value")
        != str(parsed_mapping["chip_logic_id"])
        or selection.get("visible_chip_logic_id")
        != parsed_mapping["chip_logic_id"]
        or selection.get("logical_device_index") != 0
        or selection.get("logical_device") != "npu:0"
        or selection.get("runtime_current_device") != 0
        or str(parsed_mapping["chip_name"]).replace(" ", "")
        not in str(selection.get("runtime_device_name"))
        or identity.get("partition") != expected_partition
    ):
        raise ProbeBlocked(f"inconsistent-device-mapping-evidence:{source_name}")
    environment = document.get("environment")
    work_mode_snapshot = snapshots.get("npu_work_mode")
    if not isinstance(work_mode_snapshot, dict):
        raise ProbeBlocked(f"missing-power-policy-snapshot:{source_name}")
    work_mode_output = require_command(
        work_mode_snapshot, "npu-work-mode"
    ).strip()
    if "does not support querying work-mode" in work_mode_output.lower():
        expected_power_policy_status = "unsupported"
        expected_power_policy_identity = "unsupported(npu-smi-work-mode-query)"
        expected_frontier_eligibility = (
            "ineligible-missing-power-policy-evidence"
        )
    elif work_mode_output:
        expected_power_policy_status = "declared"
        expected_power_policy_identity = (
            "declared(npu-smi-work-mode-sha256="
            f"{sha256(work_mode_output.encode('utf-8')).hexdigest()})"
        )
        expected_frontier_eligibility = "eligible"
    else:
        raise ProbeBlocked(f"empty-power-policy-evidence:{source_name}")
    expected_power_policy_source = source_ref(
        str(document.get("session_id")), "command_snapshots/npu_work_mode"
    )
    power_clock = identity.get("power_clock")
    if not isinstance(environment, dict) or not isinstance(power_clock, dict) or (
        environment.get("power_policy_status") != expected_power_policy_status
        or environment.get("frontier_eligibility")
        != expected_frontier_eligibility
        or environment.get("power_policy_source")
        != expected_power_policy_source
        or power_clock.get("power_policy") != expected_power_policy_identity
        or power_policy_field.get("status") != expected_power_policy_status
        or power_policy_field.get("source") != expected_power_policy_source
        or power_policy_field.get("required_for_anchor") is not True
        or (
            expected_power_policy_status == "unsupported"
            and "value" in power_policy_field
        )
        or (
            expected_power_policy_status == "declared"
            and power_policy_field.get("value") != expected_power_policy_identity
        )
    ):
        raise ProbeBlocked(f"invalid-power-policy-status:{source_name}")


def merge_sessions(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise ProbeBlocked("immutable-output-already-exists")
    resolved_paths = [path.resolve() for path in args.sessions]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ProbeBlocked("duplicate-session-source")
    sessions: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, object]] = []
    for path in args.sessions:
        document = json.loads(path.read_text(encoding="utf-8"))
        validate_completed_session(document, source_name=path.name)
        sessions.append(document)
        source_artifacts.append(
            {
                "path": f"sessions/{path.name}",
                "sha256": file_digest(path),
                "schema": SESSION_SCHEMA,
                "session_id": document["session_id"],
            }
        )
    if len(sessions) < 3:
        raise ProbeBlocked("fewer-than-three-independent-sessions")
    session_ids = [session["session_id"] for session in sessions]
    process_identities = [
        canonical_digest(session["process_identity"]) for session in sessions
    ]
    if len(set(session_ids)) != len(session_ids):
        raise ProbeBlocked("duplicate-session-id")
    if len(set(process_identities)) != len(process_identities):
        raise ProbeBlocked("duplicate-process-session")
    if len({canonical_digest(session["benchmark_case"]) for session in sessions}) != 1:
        raise ProbeBlocked("benchmark-case-changed-between-sessions")
    if len({canonical_digest(session["execution_domain"]) for session in sessions}) != 1:
        raise ProbeBlocked("execution-domain-changed-between-sessions")
    cohort_digests = {session["cohort_digest"] for session in sessions}
    cohort_ids = {session["cohort_id"] for session in sessions}
    if len(cohort_digests) != 1 or len(cohort_ids) != 1:
        raise ProbeBlocked("cohort-identity-changed-between-sessions")
    session_medians = [
        float(session["summary_ns"]["median"]) for session in sessions
    ]
    median_of_session_medians = float(statistics.median(session_medians))
    session_median_deviation_fractions = [
        abs(value - median_of_session_medians) / median_of_session_medians
        for value in session_medians
    ]
    maximum_session_median_deviation_fraction = max(
        session_median_deviation_fractions
    )
    if (
        maximum_session_median_deviation_fraction
        > MAX_SESSION_MEDIAN_DEVIATION_FRACTION
    ):
        raise ProbeBlocked("session-median-deviation-exceeds-policy")

    first = sessions[0]
    raw_session_values = [
        sample for session in sessions for sample in session["raw_samples_ns"]
    ]
    evidence = {
        "schema": SCHEMA,
        "ticket": 27,
        "status": "completed",
        "scope": {
            "kind": "real-device-evidence-only",
            "formal_measurement_adapter": "out-of-scope-issue-28",
            "frontier_promotion": "not-evaluated",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardware": first["hardware"],
        "software": first["software"],
        "execution_domain": first["execution_domain"],
        "cohort_identity": first["cohort_identity"],
        "cohort_digest": first["cohort_digest"],
        "cohort_id": first["cohort_id"],
        "measurement_capability_manifest": first[
            "measurement_capability_manifest"
        ],
        "benchmark_case": first["benchmark_case"],
        "sessions": [
            {
                key: session[key]
                for key in (
                    "session_id",
                    "status",
                    "started_at",
                    "completed_at",
                    "process_identity",
                    "cohort_digest",
                    "measurement_policy",
                    "environment",
                    "correctness",
                    "warmup",
                    "timer",
                    "host_timer",
                    "completion_boundary",
                    "raw_elapsed_ms",
                    "raw_samples_ns",
                    "host_visible_samples_ns",
                    "summary_ns",
                    "timing_quality",
                    "host_visible_summary_ns",
                    "transfer_probe",
                    "memory_framework",
                )
            }
            for session in sessions
        ],
        "cross_session_summary_ns": timing_summary(raw_session_values),
        "cohort_repeatability": {
            "status": "identity-stable",
            "scope": "cohort-identity-only",
            "independent_process_sessions": len(sessions),
            "unique_cohort_digests": sorted(cohort_digests),
            "identity_rule": "canonical-sha256-of-all-stable-cohort-dimensions",
        },
        "timing_repeatability": {
            "status": "passed",
            "policy_id": DISPERSION_POLICY_ID,
            "independent_process_sessions": len(sessions),
            "session_medians_ns": session_medians,
            "median_of_session_medians_ns": median_of_session_medians,
            "session_median_deviation_fractions": (
                session_median_deviation_fractions
            ),
            "observed_maximum_session_median_deviation_fraction": (
                maximum_session_median_deviation_fraction
            ),
            "maximum_session_median_deviation_fraction": (
                MAX_SESSION_MEDIAN_DEVIATION_FRACTION
            ),
            "maximum_session_iqr_fraction_of_median": MAX_SESSION_IQR_FRACTION,
            "excluded_samples": 0,
            "frontier_qualification": "not-evaluated-by-issue-27",
        },
        "source_artifacts": source_artifacts,
        "governance": {
            "credentials_recorded": False,
            "cpu_fallback_allowed": False,
            "diagnostic_profiling_used_for_baseline": False,
            "next_ticket_started": False,
        },
    }
    evidence["evidence_digest"] = canonical_digest(evidence)
    write_json(args.output, evidence)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--session-id", required=True)
    collect.add_argument("--machine-name", default="A2-AK-225")
    collect.add_argument("--npu-id", type=int, default=0)
    collect.add_argument("--chip-id", type=int, default=0)
    collect.add_argument("--m", type=int, default=512)
    collect.add_argument("--k", type=int, default=512)
    collect.add_argument("--n", type=int, default=512)
    collect.add_argument("--seed", type=int, default=20260810)
    collect.add_argument(
        "--warmup", type=int, default=EXPECTED_WARMUP_ITERATIONS
    )
    collect.add_argument("--repeats", type=int, default=EXPECTED_REPETITIONS)
    collect.add_argument("--atol", type=float, default=1e-3)
    collect.add_argument("--rtol", type=float, default=1e-3)
    collect.set_defaults(handler=collect_session)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("sessions", type=Path, nargs="+")
    merge.set_defaults(handler=merge_sessions)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except Exception as error:
        print(f"blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
