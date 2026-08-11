"""Structured diagnostic tracing and identity alignment."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import psutil
import torch
from torch import Tensor, nn

from groundupscale.benchmark.measurement import BenchmarkRunner, synchronize
from groundupscale.benchmark.reference import SemanticLeaf
from groundupscale.execution_runtime import ExecutionRuntime
from groundupscale.ir import SemanticOperation, SemanticProgram, SemanticRegion
from groundupscale.specs import AnalysisBundle


def _tensor_metadata(
    value: Any, execution_runtime: ExecutionRuntime | None = None
) -> list[dict[str, Any]]:
    tensors: list[Tensor] = []
    if isinstance(value, Tensor):
        tensors.append(value)
    elif isinstance(value, (tuple, list)):
        tensors.extend(item for item in value if isinstance(item, Tensor))
    return [
        {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "device": (
                execution_runtime.tensor_device(tensor)
                if execution_runtime is not None
                else str(tensor.device)
            ),
            "layout": str(tensor.layout).removeprefix("torch."),
            "is_contiguous": tensor.is_contiguous(),
        }
        for tensor in tensors
    ]


def _memory_snapshot(
    device: str, execution_runtime: ExecutionRuntime | None = None
) -> dict[str, int]:
    if execution_runtime is not None:
        return execution_runtime.memory_snapshot()
    result = {"process_rss_bytes": psutil.Process().memory_info().rss}
    if device == "mps":
        result.update(
            {
                "framework_current_allocated_bytes": int(
                    torch.mps.current_allocated_memory()
                ),
                "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            }
        )
    return result


def _semantic_identities(program: SemanticProgram) -> dict[str, tuple[str, str]]:
    identities = {program.root.stable_path: (program.root.node_id, program.root.kind)}
    for item in program.root.walk_items():
        identities[item.stable_path] = (
            item.node_id,
            item.operation if isinstance(item, SemanticOperation) else item.kind,
        )
    return identities


def _union_duration(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


@dataclass
class _OpenSpan:
    span_id: str
    parent_span_id: str
    stable_path: str
    started_ns: int
    inputs: list[dict[str, Any]]


class TraceRunner:
    """Collects one diagnostic execution without per-module device synchronization."""

    def __init__(
        self,
        bundle: AnalysisBundle,
        semantic: SemanticProgram,
        seed: int = 20260806,
        *,
        execution_runtime: ExecutionRuntime | None = None,
    ) -> None:
        self.bundle = bundle
        self.semantic = semantic
        self.seed = seed
        self.execution_runtime = execution_runtime

    def run(self) -> dict[str, Any]:
        benchmark = BenchmarkRunner(
            self.bundle,
            seed=self.seed,
            execution_runtime=self.execution_runtime,
            lane="diagnostic-profiling",
        )
        device = benchmark.device
        model, hidden = benchmark._model_and_input()
        identities = _semantic_identities(self.semantic)
        model_call_regions = [
            item
            for item in self.semantic.root.walk_items()
            if isinstance(item, SemanticRegion) and item.kind == "model_call"
        ]
        if len(model_call_regions) != 1:
            raise ValueError("reference trace requires exactly one model_call region")
        e2e_region = model_call_regions[0]
        events: list[dict[str, Any]] = []
        handles: list[Any] = []
        stack: list[str] = ["span-00000"]
        frames: dict[int, _OpenSpan] = {}
        counter = 0

        def pre_hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            nonlocal counter
            counter += 1
            span_id = f"span-{counter:05d}"
            frame = _OpenSpan(
                span_id=span_id,
                parent_span_id=stack[-1],
                stable_path=str(module.stable_path),
                started_ns=time.perf_counter_ns(),
                inputs=_tensor_metadata(inputs, self.execution_runtime),
            )
            frames[id(module)] = frame
            stack.append(span_id)

        def post_hook(
            module: nn.Module, inputs: tuple[Any, ...], output: Any
        ) -> None:
            frame = frames.pop(id(module))
            ended_ns = time.perf_counter_ns()
            if not stack or stack[-1] != frame.span_id:
                raise RuntimeError(f"unbalanced trace span for {frame.stable_path}")
            stack.pop()
            node_id, semantic_kind = identities.get(frame.stable_path, (None, None))
            events.append(
                {
                    "schema": "groundupscale.dev/observation-span/v1alpha1",
                    "span_id": frame.span_id,
                    "parent_span_id": frame.parent_span_id,
                    "stable_path": frame.stable_path,
                    "compiled_node_id": node_id,
                    "semantic_kind": semantic_kind,
                    "runtime_kind": (
                        "operation" if isinstance(module, SemanticLeaf) else "module"
                    ),
                    "operation": (
                        module.operation if isinstance(module, SemanticLeaf) else None
                    ),
                    "host_started_ns": frame.started_ns,
                    "host_ended_ns": ended_ns,
                    "host_duration_ns": ended_ns - frame.started_ns,
                    "clock_domain": (
                        "host-synchronous" if device == "cpu" else "host-enqueue"
                    ),
                    "inputs": frame.inputs,
                    "outputs": _tensor_metadata(output, self.execution_runtime),
                    "memory": _memory_snapshot(device, self.execution_runtime),
                    "instrumentation_profile": "trace",
                }
            )

        for module in model.modules():
            if hasattr(module, "stable_path"):
                handles.append(module.register_forward_pre_hook(pre_hook))
                handles.append(module.register_forward_hook(post_hook, always_call=True))

        before_memory = _memory_snapshot(device, self.execution_runtime)
        synchronize(device, self.execution_runtime)
        e2e_started = time.perf_counter_ns()
        try:
            with torch.inference_mode():
                output = (
                    self.execution_runtime.execute_checked(lambda: model(hidden))
                    if self.execution_runtime is not None
                    else model(hidden)
                )
                synchronize(device, self.execution_runtime)
        finally:
            for handle in handles:
                handle.remove()
        e2e_ended = time.perf_counter_ns()
        after_memory = _memory_snapshot(device, self.execution_runtime)
        if frames or stack != ["span-00000"]:
            raise RuntimeError("trace hooks did not close cleanly")
        events.append(
            {
                "schema": "groundupscale.dev/observation-span/v1alpha1",
                "span_id": "span-00000",
                "parent_span_id": None,
                "stable_path": e2e_region.stable_path,
                "compiled_node_id": e2e_region.node_id,
                "semantic_kind": e2e_region.kind,
                "runtime_kind": "e2e",
                "operation": None,
                "host_started_ns": e2e_started,
                "host_ended_ns": e2e_ended,
                "host_duration_ns": e2e_ended - e2e_started,
                "clock_domain": "host-synchronized-boundary",
                "inputs": _tensor_metadata(hidden, self.execution_runtime),
                "outputs": _tensor_metadata(output, self.execution_runtime),
                "memory": after_memory,
                "instrumentation_profile": "trace",
            }
        )
        events.sort(key=lambda event: (event["host_started_ns"], event["span_id"]))

        alignment_entries = []
        for event in events:
            node_id, semantic_kind = identities.get(event["stable_path"], (None, None))
            alignment_entries.append(
                {
                    "span_id": event["span_id"],
                    "stable_path": event["stable_path"],
                    "compiled_node_ids": [node_id] if node_id is not None else [],
                    "semantic_kind": semantic_kind,
                    "match_rule": "exact-stable-path" if node_id is not None else "unattributed",
                    "confidence": 1.0 if node_id is not None else 0.0,
                }
            )
        matched = sum(bool(entry["compiled_node_ids"]) for entry in alignment_entries)
        leaf_intervals = [
            (event["host_started_ns"], event["host_ended_ns"])
            for event in events
            if event["runtime_kind"] == "operation"
        ]
        leaf_union = _union_duration(leaf_intervals)
        e2e_duration = e2e_ended - e2e_started
        memory_key = (
            "framework_current_allocated_bytes"
            if device == "mps" or self.execution_runtime is not None
            else "process_rss_bytes"
        )
        peak_observed = max(event["memory"][memory_key] for event in events)
        return {
            "schema": "groundupscale.dev/trace-observation/v1alpha1",
            "device": device,
            "instrumentation_profile": "trace",
            "synchronization": "profile-boundaries-only",
            "events": events,
            "alignment_map": {
                "schema": "groundupscale.dev/alignment-map/v1alpha1",
                "entries": alignment_entries,
                "matched_spans": matched,
                "total_spans": len(alignment_entries),
                "coverage": matched / len(alignment_entries),
            },
            "error_attribution": {
                "schema": "groundupscale.dev/error-attribution/v1alpha1",
                "e2e_trace_host_ns": e2e_duration,
                "leaf_host_interval_union_ns": leaf_union,
                "unattributed_host_ns": max(0, e2e_duration - leaf_union),
                "unattributed_reason": (
                    "Python/composite/runtime overhead plus device wait; accelerator leaf "
                    "spans are host enqueue ranges and are not treated as device durations"
                ),
                "alignment_coverage": matched / len(alignment_entries),
            },
            "memory_observation": {
                "observer": (
                    "framework_device_current_allocated"
                    if device == "mps" or self.execution_runtime is not None
                    else "process_rss"
                ),
                "before": before_memory,
                "after": after_memory,
                "peak_observed_bytes": peak_observed,
                "attribution": (
                    "framework-attributed point samples"
                    if device == "mps" or self.execution_runtime is not None else
                    "process-wide diagnostic; not framework-attributed"
                ),
            },
        }


__all__ = ["TraceRunner"]
