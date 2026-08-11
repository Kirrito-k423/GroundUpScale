"""Cross-device observation of live framework Tensor storages."""

from __future__ import annotations

from typing import Any
import weakref

import torch
from torch import Tensor, nn

from groundupscale.benchmark.measurement import synchronize
from groundupscale.benchmark.reference import SemanticLeaf
from groundupscale.execution_runtime import ExecutionRuntime


def _tensors(value: Any) -> list[Tensor]:
    if isinstance(value, Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [item for item in value if isinstance(item, Tensor)]
    return []


def observe_tensor_storage_peak(
    model: nn.Module,
    inputs: tuple[Tensor, ...],
    *,
    device: str,
    execution_runtime: ExecutionRuntime | None = None,
) -> dict[str, Any]:
    """Observe live Tensor storages without retaining Tensor payloads.

    This observer counts unique storages referenced by model state, inputs, and
    operation outputs at every semantic leaf boundary. It deliberately excludes
    allocator reservations, driver memory, kernel-private workspace, and Python RSS.
    """

    storage_refs: dict[
        tuple[str, int, int], list[weakref.ReferenceType[Tensor]]
    ] = {}
    timeline: list[dict[str, Any]] = []
    handles: list[Any] = []

    def register(tensor: Tensor) -> None:
        storage = tensor.untyped_storage()
        tensor_device = (
            execution_runtime.tensor_device(tensor)
            if execution_runtime is not None
            else str(tensor.device)
        )
        key = (tensor_device, storage.data_ptr(), storage.nbytes())
        storage_refs.setdefault(key, []).append(weakref.ref(tensor))

    for parameter in model.parameters():
        register(parameter)
    for buffer in model.buffers():
        register(buffer)
    for tensor in inputs:
        register(tensor)

    def snapshot(stable_path: str) -> None:
        live = [
            key
            for key, references in storage_refs.items()
            if any(reference() is not None for reference in references)
        ]
        timeline.append(
            {
                "stable_path": stable_path,
                "live_storage_bytes": sum(key[2] for key in live),
                "live_storage_count": len(live),
            }
        )

    def hook(module: SemanticLeaf, hook_inputs: tuple[Any, ...], output: Any) -> None:
        for tensor in _tensors(output):
            register(tensor)
        snapshot(module.stable_path)

    for module in model.modules():
        if isinstance(module, SemanticLeaf):
            handles.append(module.register_forward_hook(hook, always_call=True))
    snapshot("before-forward")
    try:
        with torch.inference_mode():
            output = (
                execution_runtime.execute_checked(lambda: model(*inputs))
                if execution_runtime is not None
                else model(*inputs)
            )
            synchronize(device, execution_runtime)
    finally:
        for handle in handles:
            handle.remove()
    register(output)
    snapshot("after-forward")
    peak = max(timeline, key=lambda item: item["live_storage_bytes"])
    return {
        "schema": "groundupscale.dev/tensor-storage-observation/v1alpha1",
        "observer": "weakref-live-unique-tensor-storage",
        "device": device,
        "peak_framework_tensor_bytes": peak["live_storage_bytes"],
        "peak_stable_path": peak["stable_path"],
        "timeline": timeline,
        "includes": ["parameters", "buffers", "input", "live operation outputs"],
        "excludes": [
            "allocator reservation and fragmentation",
            "driver allocations",
            "kernel-private workspace",
            "framework/runtime code and Python RSS",
        ],
        "observer_effect": "forward hooks and weak references; use for memory attribution, not latency truth",
    }


__all__ = ["observe_tensor_storage_peak"]
