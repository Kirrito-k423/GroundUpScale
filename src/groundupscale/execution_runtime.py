"""Explicit execution boundary for trusted accelerator runs."""

from __future__ import annotations

from collections.abc import Callable
import importlib
import time
from typing import Any, Protocol, TypeVar
import warnings

import psutil
from torch import Tensor, nn


_ResultT = TypeVar("_ResultT")
_CPU_FALLBACK_WARNING = (
    r".*not currently supported on the NPU backend.*"
    r"fall back to run on the CPU.*"
)


def execute_with_npu_cpu_fallback_guard(
    invoke: Callable[[], _ResultT],
) -> _ResultT:
    """Turn torch_npu's eager CPU-fallback warning into a closed failure."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message=_CPU_FALLBACK_WARNING,
            category=UserWarning,
        )
        try:
            return invoke()
        except UserWarning as error:
            message = str(error)
            if (
                "not currently supported on the NPU backend" in message
                and "fall back to run on the CPU" in message
            ):
                raise RuntimeError("cpu-fallback-detected") from error
            raise


class ExecutionRuntime(Protocol):
    logical_device: str
    device_type: str
    timer_source: str
    timer_resolution_ns: float
    completion_protocol: str
    allocator_peak_reset_before_run: bool

    def prepare_model(self, model: nn.Module, *, lane: str) -> nn.Module: ...

    def prepare_tensor(self, tensor: Tensor, *, lane: str, role: str) -> Tensor: ...

    def copy_to_cpu(self, tensor: Tensor, *, lane: str, role: str) -> Tensor: ...

    def execute_checked(self, invoke: Callable[[], Tensor]) -> Tensor: ...

    def synchronize(self) -> None: ...

    def execute_timed(
        self, invoke: Callable[[], Tensor], *, iterations: int
    ) -> dict[str, int]: ...

    def tensor_device(self, tensor: Tensor) -> str: ...

    def tensor_device_type(self, tensor: Tensor) -> str: ...

    def memory_snapshot(self) -> dict[str, int]: ...

    def environment(self) -> dict[str, object]: ...

    def transfer_evidence(self) -> dict[str, object]: ...


class AscendNpuExecutionRuntime:
    """Torch-NPU execution with event timing and explicit transfer evidence."""

    device_type = "npu"
    timer_source = "torch.npu.Event.elapsed_time"
    timer_resolution_ns = 20.0
    completion_protocol = "end-event-synchronize-plus-device-synchronize"
    allocator_peak_reset_before_run = True

    def __init__(self, logical_device_index: int = 0) -> None:
        if logical_device_index < 0:
            raise ValueError("logical_device_index must be non-negative")
        self._torch: Any = importlib.import_module("torch")
        self._torch_npu: Any = importlib.import_module("torch_npu")
        self._index = logical_device_index
        self.logical_device = f"npu:{logical_device_index}"
        self._torch.npu.set_device(logical_device_index)
        self._torch.npu.reset_peak_memory_stats()
        self._process_peak_observed_rss_bytes = (
            psutil.Process().memory_info().rss
        )
        self._transfers: list[dict[str, object]] = []

    @staticmethod
    def _tensor_bytes(tensor: Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def prepare_model(self, model: nn.Module, *, lane: str) -> nn.Module:
        parameter_bytes = sum(self._tensor_bytes(value) for value in model.parameters())
        buffer_bytes = sum(self._tensor_bytes(value) for value in model.buffers())
        prepared = model.to(self.logical_device)
        self._transfers.append(
            {
                "lane": lane,
                "kind": "weights-and-buffers-host-to-device",
                "source": "cpu",
                "destination": self.logical_device,
                "parameter_bytes": parameter_bytes,
                "buffer_bytes": buffer_bytes,
                "bytes": parameter_bytes + buffer_bytes,
            }
        )
        return prepared

    def prepare_tensor(self, tensor: Tensor, *, lane: str, role: str) -> Tensor:
        prepared = tensor.to(self.logical_device)
        self._transfers.append(
            {
                "lane": lane,
                "kind": f"{role}-host-to-device",
                "source": "cpu",
                "destination": self.logical_device,
                "bytes": self._tensor_bytes(tensor),
            }
        )
        return prepared

    def copy_to_cpu(self, tensor: Tensor, *, lane: str, role: str) -> Tensor:
        self.synchronize()
        copied = tensor.detach().cpu()
        self._transfers.append(
            {
                "lane": lane,
                "kind": f"{role}-device-to-host",
                "source": self.logical_device,
                "destination": "cpu",
                "bytes": self._tensor_bytes(tensor),
            }
        )
        return copied

    def synchronize(self) -> None:
        self._torch.npu.synchronize()

    def execute_checked(self, invoke: Callable[[], Tensor]) -> Tensor:
        return execute_with_npu_cpu_fallback_guard(invoke)

    def execute_timed(
        self, invoke: Callable[[], Tensor], *, iterations: int
    ) -> dict[str, int]:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        start_event = self._torch.npu.Event(enable_timing=True)
        end_event = self._torch.npu.Event(enable_timing=True)
        self.synchronize()
        host_started = time.perf_counter_ns()
        start_event.record()
        for _ in range(iterations):
            result = self.execute_checked(invoke)
            if self.tensor_device_type(result) != "npu":
                raise RuntimeError("cpu-fallback-detected")
        end_event.record()
        host_launch_ended = time.perf_counter_ns()
        end_event.synchronize()
        self.synchronize()
        host_completed = time.perf_counter_ns()
        primary_elapsed_ns = int(
            round(float(start_event.elapsed_time(end_event)) * 1_000_000)
        )
        if primary_elapsed_ns <= 0:
            raise RuntimeError("invalid-primary-timer-sample")
        return {
            "primary_elapsed_ns": primary_elapsed_ns,
            "host_launch_ns": max(1, host_launch_ended - host_started),
            "device_completion_wait_ns": max(
                0, host_completed - host_launch_ended
            ),
            "host_completion_ns": max(1, host_completed - host_started),
        }

    def tensor_device(self, tensor: Tensor) -> str:
        return str(tensor.device)

    def tensor_device_type(self, tensor: Tensor) -> str:
        return str(tensor.device.type)

    def memory_snapshot(self) -> dict[str, int]:
        current_rss_bytes = psutil.Process().memory_info().rss
        self._process_peak_observed_rss_bytes = max(
            self._process_peak_observed_rss_bytes,
            current_rss_bytes,
        )
        return {
            "process_rss_bytes": current_rss_bytes,
            "process_current_rss_bytes": current_rss_bytes,
            "process_peak_observed_rss_bytes": (
                self._process_peak_observed_rss_bytes
            ),
            "framework_current_allocated_bytes": int(
                self._torch.npu.memory_allocated()
            ),
            "framework_reserved_bytes": int(self._torch.npu.memory_reserved()),
            "framework_max_allocated_bytes": int(
                self._torch.npu.max_memory_allocated()
            ),
        }

    def environment(self) -> dict[str, object]:
        return {
            "runtime": "torch-npu",
            "torch_version": str(self._torch.__version__),
            "torch_npu_version": str(self._torch_npu.__version__),
            "device_name": str(self._torch.npu.get_device_name(self._index)),
            "logical_device": self.logical_device,
            "allocator_peak_reset_before_run": True,
            "cpu_fallback_policy": "warning-is-compatibility-failure",
        }

    def transfer_evidence(self) -> dict[str, object]:
        return {
            "schema": "groundupscale.dev/transfer-observation/v1alpha1",
            "logical_device": self.logical_device,
            "records": list(self._transfers),
        }


def create_execution_runtime(device: str) -> ExecutionRuntime:
    if not device.startswith("npu:"):
        raise ValueError(f"no explicit accelerator runtime for {device!r}")
    return AscendNpuExecutionRuntime(int(device.partition(":")[2]))


__all__ = [
    "AscendNpuExecutionRuntime",
    "ExecutionRuntime",
    "create_execution_runtime",
    "execute_with_npu_cpu_fallback_guard",
]
