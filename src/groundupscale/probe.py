"""Reproducible CPU/MPS capability probe used before performance modeling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
import platform
import statistics
import time
from typing import Any

import psutil
import torch
from torch import Tensor
from torch.nn import functional as functional


PROBE_SCHEMA = "groundupscale.dev/environment-probe/v1alpha1"
REQUIRED_OPERATIONS = (
    "matmul",
    "add",
    "rmsnorm",
    "softmax",
    "silu",
    "mul",
    "view",
    "transpose",
)


def _operations(x: Tensor, weight: Tensor) -> dict[str, Callable[[], Tensor]]:
    width = x.shape[-1]
    return {
        "matmul": lambda: torch.matmul(x, weight),
        "add": lambda: torch.add(x, weight),
        "rmsnorm": lambda: functional.rms_norm(x, (width,), eps=1e-6),
        "softmax": lambda: torch.softmax(x, dim=-1),
        "silu": lambda: functional.silu(x),
        "mul": lambda: torch.mul(x, weight),
        "view": lambda: x.view(width, width),
        "transpose": lambda: x.transpose(0, 1),
    }


def _synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _run_operations(x: Tensor, weight: Tensor) -> dict[str, Tensor]:
    return {name: operation() for name, operation in _operations(x, weight).items()}


def _correctness(
    expected: dict[str, Tensor],
    actual: dict[str, Tensor],
    *,
    atol: float,
    rtol: float,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_OPERATIONS:
        reference = expected[name]
        observed = actual[name].to("cpu")
        absolute = (reference - observed).abs()
        relative = absolute / reference.abs().clamp_min(atol)
        max_absolute_error = float(absolute.max().item())
        max_relative_error = float(relative.max().item())
        passed = bool(torch.allclose(reference, observed, atol=atol, rtol=rtol))
        results[name] = {
            "status": "passed" if passed else "failed",
            "max_absolute_error": max_absolute_error,
            "max_relative_error": max_relative_error,
            "atol": atol,
            "rtol": rtol,
        }
    return results


def _latency_summary(
    window_samples: list[list[int]], inner_iterations: int
) -> dict[str, Any]:
    normalized_windows = [
        [window / inner_iterations for window in sample_windows]
        for sample_windows in window_samples
    ]
    samples = [float(statistics.median(windows)) for windows in normalized_windows]
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    median = float(statistics.median(samples))
    iqr = float(quartiles[2] - quartiles[0])
    return {
        "samples": samples,
        "window_samples": window_samples,
        "normalized_window_samples": normalized_windows,
        "inner_iterations": inner_iterations,
        "median": median,
        "q1": float(quartiles[0]),
        "q3": float(quartiles[2]),
        "iqr": iqr,
        "iqr_over_median": iqr / median,
    }


def _process_memory(before: int) -> dict[str, Any]:
    after = psutil.Process().memory_info().rss
    return {
        "observer": "process_rss",
        "before_bytes": before,
        "after_bytes": after,
        "delta_bytes": after - before,
        "attribution": "process-wide; diagnostic only",
    }


def _mps_memory(before: dict[str, int]) -> dict[str, Any]:
    after = {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
    }
    return {
        "observer": "torch_mps_allocator",
        "before": before,
        "after": after,
        "current_allocated_delta_bytes": (
            after["current_allocated_bytes"] - before["current_allocated_bytes"]
        ),
        "driver_allocated_delta_bytes": (
            after["driver_allocated_bytes"] - before["driver_allocated_bytes"]
        ),
        "attribution": (
            "current_allocated is framework-attributed; driver_allocated is reported "
            "separately and is not treated as logical tensor memory"
        ),
    }


def _probe_device(
    device: str,
    cpu_x: Tensor,
    cpu_weight: Tensor,
    expected: dict[str, Tensor],
    *,
    warmup: int,
    repeats: int,
    inner_iterations: int,
    windows_per_sample: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    process_memory_before = psutil.Process().memory_info().rss
    mps_memory_before: dict[str, int] | None = None
    if device == "mps":
        torch.mps.empty_cache()
        _synchronize(device)
        mps_memory_before = {
            "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
        }

    x = cpu_x.to(device)
    weight = cpu_weight.to(device)
    actual = _run_operations(x, weight)
    _synchronize(device)
    operations = _correctness(expected, actual, atol=atol, rtol=rtol)

    for _ in range(warmup):
        _run_operations(x, weight)
    _synchronize(device)

    samples: list[list[int]] = []
    for _ in range(repeats):
        windows: list[int] = []
        for _ in range(windows_per_sample):
            _synchronize(device)
            started = time.perf_counter_ns()
            for _ in range(inner_iterations):
                actual = _run_operations(x, weight)
            _synchronize(device)
            windows.append(time.perf_counter_ns() - started)
        samples.append(windows)

    memory = (
        _mps_memory(mps_memory_before)
        if mps_memory_before is not None
        else _process_memory(process_memory_before)
    )
    return {
        "available": True,
        "device": str(x.device),
        "dtype": str(x.dtype).removeprefix("torch."),
        "operations": operations,
        "latency_ns": _latency_summary(samples, inner_iterations),
        "memory": memory,
    }


def run_environment_probe(
    devices: Sequence[str] = ("cpu",),
    *,
    warmup: int = 5,
    repeats: int = 20,
    inner_iterations: int = 1,
    windows_per_sample: int = 1,
    matrix_size: int = 512,
    seed: int = 20260806,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Probe required operations without hiding unavailable or failed devices."""
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats < 4:
        raise ValueError("repeats must be at least 4 to calculate an IQR")
    if inner_iterations <= 0:
        raise ValueError("inner_iterations must be positive")
    if windows_per_sample <= 0:
        raise ValueError("windows_per_sample must be positive")
    if matrix_size <= 0:
        raise ValueError("matrix_size must be positive")
    unsupported = sorted(set(devices) - {"cpu", "mps"})
    if unsupported:
        raise ValueError(f"unsupported devices: {', '.join(unsupported)}")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    cpu_x = torch.randn(matrix_size, matrix_size, generator=generator, dtype=torch.float32)
    cpu_weight = torch.randn(
        matrix_size, matrix_size, generator=generator, dtype=torch.float32
    )
    expected = _run_operations(cpu_x, cpu_weight)
    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    reports: dict[str, Any] = {}

    for device in dict.fromkeys(devices):
        if device == "mps" and not mps_available:
            reports[device] = {
                "available": False,
                "reason": "torch.backends.mps.is_available() returned false",
                "operations": {},
            }
            continue
        try:
            reports[device] = _probe_device(
                device,
                cpu_x,
                cpu_weight,
                expected,
                warmup=warmup,
                repeats=repeats,
                inner_iterations=inner_iterations,
                windows_per_sample=windows_per_sample,
                atol=atol,
                rtol=rtol,
            )
        except Exception as error:  # The report must preserve the runtime failure.
            reports[device] = {
                "available": True,
                "operations": {},
                "error": f"{type(error).__name__}: {error}",
            }

    ok = all(
        report.get("available", False)
        and not report.get("error")
        and all(
            operation["status"] == "passed"
            for operation in report.get("operations", {}).values()
        )
        for report in reports.values()
    )
    return {
        "schema": PROBE_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "ok": ok,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "torch": {
            "version": torch.__version__,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "mps": {"built": mps_built, "available": mps_available},
        "configuration": {
            "seed": seed,
            "warmup": warmup,
            "repeats": repeats,
            "inner_iterations": inner_iterations,
            "windows_per_sample": windows_per_sample,
            "matrix_size": matrix_size,
            "dtype": "float32",
            "atol": atol,
            "rtol": rtol,
        },
        "devices": reports,
    }
