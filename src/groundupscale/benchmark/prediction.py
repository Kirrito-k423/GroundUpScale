"""Explainable hardware-independent live-set prediction for the reference slice."""

from __future__ import annotations

from math import prod
from typing import Any

from groundupscale.ir import CostProgram, SemanticProgram


DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "int64": 8, "bool": 1}


def _bytes(dtype: str, shape: tuple[int, ...]) -> int:
    return DTYPE_BYTES[dtype] * prod(shape)


def predict_live_set(semantic: SemanticProgram, cost: CostProgram) -> dict[str, Any]:
    operations = list(semantic.walk_operations())
    operation_order = {operation.node_id: index for index, operation in enumerate(operations)}
    values = {value.value_id: value for value in semantic.values}

    def storage_root(value_id: str) -> str:
        seen: set[str] = set()
        current = value_id
        while values[current].alias_of is not None:
            if current in seen:
                raise ValueError(f"alias cycle at {current}")
            seen.add(current)
            current = values[current].alias_of  # type: ignore[assignment]
        return current

    lifetimes: dict[str, dict[str, Any]] = {}
    for value in semantic.values:
        if "/state-read/" in value.stable_path:
            continue
        root_id = storage_root(value.value_id)
        root = values[root_id]
        start = operation_order.get(root.producer_id or "", -1)
        consumer_positions = [
            operation_order[consumer]
            for consumer in value.consumer_ids
            if consumer in operation_order
        ]
        external_consumer = any(
            consumer not in operation_order for consumer in value.consumer_ids
        )
        end = (
            len(operations)
            if external_consumer
            else max(consumer_positions, default=start)
        )
        lifetime = lifetimes.setdefault(
            root_id,
            {
                "storage_id": root_id,
                "stable_path": root.stable_path,
                "bytes": _bytes(root.tensor.dtype, root.tensor.shape),
                "start_operation": start,
                "end_operation": end,
                "aliases": [],
            },
        )
        lifetime["start_operation"] = min(lifetime["start_operation"], start)
        lifetime["end_operation"] = max(lifetime["end_operation"], end)
        if value.value_id != root_id:
            lifetime["aliases"].append(value.value_id)

    timeline: list[dict[str, Any]] = []
    peak_activation = 0
    peak_index = -1
    for index in range(-1, len(operations) + 1):
        live = [
            lifetime
            for lifetime in lifetimes.values()
            if lifetime["start_operation"] <= index <= lifetime["end_operation"]
        ]
        live_bytes = sum(lifetime["bytes"] for lifetime in live)
        if live_bytes > peak_activation:
            peak_activation = live_bytes
            peak_index = index
        timeline.append(
            {
                "operation_index": index,
                "operation_stable_path": (
                    operations[index].stable_path
                    if 0 <= index < len(operations)
                    else "before" if index < 0 else "after"
                ),
                "activation_live_bytes": live_bytes,
                "live_storage_ids": [lifetime["storage_id"] for lifetime in live],
            }
        )

    state_bytes = cost.summary.parameter_bytes + cost.summary.buffer_bytes
    return {
        "schema": "groundupscale.dev/live-set-prediction/v1alpha1",
        "parameter_bytes": cost.summary.parameter_bytes,
        "buffer_bytes": cost.summary.buffer_bytes,
        "state_bytes": state_bytes,
        "peak_activation_live_bytes": peak_activation,
        "predicted_framework_peak_bytes": state_bytes + peak_activation,
        "peak_operation_index": peak_index,
        "peak_operation_stable_path": timeline[peak_index + 1][
            "operation_stable_path"
        ],
        "storage_lifetimes": list(lifetimes.values()),
        "timeline": timeline,
        "exclusions": [
            "kernel-private workspace",
            "allocator reservation and fragmentation",
            "driver allocations",
            "framework/runtime code and Python object memory",
        ],
    }
