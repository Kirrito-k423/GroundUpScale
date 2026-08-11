"""Minimally intrusive Benchmark Case execution on CPU and MPS."""

from __future__ import annotations

import math
import os
import statistics
import time
from typing import Any, Callable, cast

import psutil
import torch
from torch import Tensor, nn

from groundupscale.benchmark.reference import (
    ReferenceConfig,
    SemanticLeaf,
    TwoLayerTransformer,
)
from groundupscale.execution_runtime import ExecutionRuntime
from groundupscale.schemas.v1alpha1 import BenchmarkDefinition
from groundupscale.specs import AnalysisBundle


def resolve_device(bundle: AnalysisBundle) -> str:
    placements = {binding.placement for binding in bundle.deployment_intent.spec.bindings}
    fabric_nodes = {node.id: node for node in bundle.fabric_graph.spec.nodes}
    devices: set[str] = set()
    for placement in placements:
        try:
            kind = fabric_nodes[placement].device
        except KeyError as error:
            raise ValueError(f"deployment placement {placement!r} is absent from FabricGraph") from error
        if kind == "gpu":
            devices.add("mps")
        elif kind.startswith("npu-"):
            devices.add(kind.replace("npu-", "npu:", 1))
        else:
            devices.add(kind)
    if len(devices) != 1 or not (
        next(iter(devices)) in {"cpu", "mps"}
        or next(iter(devices)).startswith("npu:")
    ):
        raise ValueError(
            "reference slice requires exactly one CPU, MPS, or NPU placement: "
            f"{devices}"
        )
    device = next(iter(devices))
    if device == "mps":
        fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").lower()
        if fallback in {"1", "true", "yes"}:
            raise RuntimeError("refusing MPS benchmark with fallback enabled")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
    return device


def synchronize(
    device: str, execution_runtime: ExecutionRuntime | None = None
) -> None:
    if execution_runtime is not None:
        execution_runtime.synchronize()
        return
    if device == "mps":
        torch.mps.synchronize()


def _case_definitions(bundle: AnalysisBundle) -> tuple[BenchmarkDefinition, ...]:
    return tuple(
        case
        for document in bundle.benchmark_cases
        for case in document.spec.cases
    )


def resolve_module_scope(model: nn.Module, scope: str) -> nn.Module:
    modules = [
        (module, stable_path)
        for module in model.modules()
        if isinstance((stable_path := getattr(module, "stable_path", None)), str)
    ]
    exact = [module for module, stable_path in modules if stable_path == scope]
    if len(exact) == 1:
        return exact[0]
    if scope.startswith("model/"):
        parts = scope.split("/", 2)
        if len(parts) == 3:
            suffix = f"/model/{parts[2]}"
            matches = [
                module
                for module, stable_path in modules
                if stable_path.endswith(suffix)
            ]
            if len(matches) == 1:
                return matches[0]
    raise KeyError(f"benchmark scope {scope!r} does not resolve to one runtime module")


def _latency_summary(
    window_samples_ns: list[list[int]], inner_iterations: int
) -> dict[str, Any]:
    normalized = [
        [window / inner_iterations for window in windows]
        for windows in window_samples_ns
    ]
    samples = [float(statistics.median(windows)) for windows in normalized]
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    median = float(statistics.median(samples))
    iqr = float(quartiles[2] - quartiles[0])
    return {
        "samples_ns": samples,
        "window_samples_ns": window_samples_ns,
        "normalized_window_samples_ns": normalized,
        "inner_iterations": inner_iterations,
        "windows_per_sample": len(window_samples_ns[0]),
        "median_ns": median,
        "q1_ns": float(quartiles[0]),
        "q3_ns": float(quartiles[2]),
        "iqr_ns": iqr,
        "iqr_over_median": iqr / median,
        "throughput_per_second": 1_000_000_000.0 / median,
    }


def _memory_snapshot(
    device: str, execution_runtime: ExecutionRuntime | None = None
) -> dict[str, int | str]:
    if execution_runtime is not None:
        return dict(execution_runtime.memory_snapshot())
    snapshot: dict[str, int | str] = {
        "process_rss_bytes": psutil.Process().memory_info().rss,
    }
    if device == "mps":
        snapshot.update(
            {
                "framework_current_allocated_bytes": int(
                    torch.mps.current_allocated_memory()
                ),
                "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            }
        )
    return snapshot


class BenchmarkRunner:
    """Runs authored Benchmark Cases without per-operation synchronization."""

    def __init__(
        self,
        bundle: AnalysisBundle,
        seed: int = 20260806,
        *,
        execution_runtime: ExecutionRuntime | None = None,
        lane: str = "baseline-timing",
    ) -> None:
        self.bundle = bundle
        self.seed = seed
        self.config = ReferenceConfig.from_analysis_bundle(bundle)
        self.device = resolve_device(bundle)
        self.execution_runtime = execution_runtime
        self.lane = lane
        if self.device.startswith("npu:") and execution_runtime is None:
            raise RuntimeError("NPU execution requires an explicit ExecutionRuntime")
        if (
            execution_runtime is not None
            and execution_runtime.logical_device != self.device
        ):
            raise ValueError(
                "ExecutionRuntime logical device does not match Deployment Intent"
            )

    def _model_and_input(self) -> tuple[TwoLayerTransformer, Tensor]:
        model = TwoLayerTransformer(self.config, self.seed)
        generator = torch.Generator(device="cpu").manual_seed(self.seed + 1)
        hidden = torch.randn(
            self.config.batch_size,
            self.config.sequence_length,
            self.config.hidden_size,
            generator=generator,
            dtype=self.config.dtype,
        )
        if self.execution_runtime is not None:
            model = cast(
                TwoLayerTransformer,
                self.execution_runtime.prepare_model(model, lane=self.lane).eval(),
            )
            hidden = self.execution_runtime.prepare_tensor(
                hidden, lane=self.lane, role="input"
            )
        else:
            target = torch.device(self.device)
            model = model.to(target).eval()
            hidden = hidden.to(target)
        return model, hidden

    def _synchronize(self) -> None:
        synchronize(self.device, self.execution_runtime)

    def _execute_timed(
        self, invoke: Callable[[], Tensor], *, iterations: int
    ) -> dict[str, int]:
        if self.execution_runtime is not None:
            return self.execution_runtime.execute_timed(
                invoke, iterations=iterations
            )
        self._synchronize()
        started = time.perf_counter_ns()
        for _ in range(iterations):
            invoke()
        self._synchronize()
        elapsed = max(1, time.perf_counter_ns() - started)
        return {
            "primary_elapsed_ns": elapsed,
            "host_launch_ns": elapsed,
            "device_completion_wait_ns": 0,
            "host_completion_ns": elapsed,
        }

    def _invocations(
        self, model: TwoLayerTransformer, hidden: Tensor
    ) -> dict[str, tuple[str, Callable[[], Tensor]]]:
        definitions = _case_definitions(self.bundle)
        captured: dict[str, tuple[Tensor, ...]] = {}
        handles: list[Any] = []
        targets: dict[str, nn.Module] = {}
        for case in definitions:
            if case.mode == "e2e":
                continue
            target = resolve_module_scope(model, case.scope)
            targets[case.id] = target

            def capture(
                module: nn.Module,
                inputs: tuple[Any, ...],
                *,
                case_id: str = case.id,
            ) -> None:
                if case_id not in captured:
                    captured[case_id] = tuple(
                        value.detach().clone()
                        for value in inputs
                        if isinstance(value, Tensor)
                    )

            handles.append(target.register_forward_pre_hook(capture))
        try:
            with torch.inference_mode():
                model(hidden)
                self._synchronize()
        finally:
            for handle in handles:
                handle.remove()

        invocations: dict[str, tuple[str, Callable[[], Tensor]]] = {}
        for case in definitions:
            if case.mode == "e2e":
                def invoke_model(
                    model: TwoLayerTransformer = model,
                    hidden: Tensor = hidden,
                ) -> Tensor:
                    return model(hidden)

                invocations[case.id] = (
                    model.stable_path,
                    invoke_model,
                )
                continue
            target = targets[case.id]
            inputs = captured[case.id]

            def invoke_target(
                target: nn.Module = target,
                inputs: tuple[Tensor, ...] = inputs,
            ) -> Tensor:
                return cast(Tensor, target(*inputs))

            invocations[case.id] = (
                str(target.stable_path),
                invoke_target,
            )
        return invocations

    def run(
        self,
        *,
        samples_override: int | None = None,
        warmup_override: int | None = None,
        windows_per_sample: int = 5,
        target_window_ns: int = 20_000_000,
        maximum_inner_iterations: int = 1_000,
    ) -> dict[str, Any]:
        if samples_override is not None and samples_override < 4:
            raise ValueError("samples_override must be at least 4")
        if windows_per_sample <= 0:
            raise ValueError("windows_per_sample must be positive")
        model, hidden = self._model_and_input()
        invocations = self._invocations(model, hidden)
        results: list[dict[str, Any]] = []

        with torch.inference_mode():
            for definition in _case_definitions(self.bundle):
                resolved_scope, invoke = invocations[definition.id]
                warmup = (
                    definition.warmup_iterations
                    if warmup_override is None
                    else warmup_override
                )
                samples = definition.samples if samples_override is None else samples_override
                for _ in range(warmup):
                    invoke()
                self._synchronize()

                pilot_iterations = 10
                pilot = self._execute_timed(
                    invoke, iterations=pilot_iterations
                )
                single_ns = max(
                    1,
                    pilot["primary_elapsed_ns"] // pilot_iterations,
                )
                inner_iterations = (
                    max(
                        1,
                        min(
                            maximum_inner_iterations,
                            math.ceil(target_window_ns / single_ns),
                        ),
                    )
                    if definition.mode == "operator"
                    else 1
                )
                before_memory = _memory_snapshot(
                    self.device, self.execution_runtime
                )
                windows: list[list[int]] = []
                timing_boundaries: dict[str, list[int]] = {
                    "host_launch_ns": [],
                    "device_completion_wait_ns": [],
                    "host_completion_ns": [],
                }
                for _ in range(samples):
                    sample_windows: list[int] = []
                    for _ in range(windows_per_sample):
                        timing = self._execute_timed(
                            invoke, iterations=inner_iterations
                        )
                        sample_windows.append(timing["primary_elapsed_ns"])
                        for field in timing_boundaries:
                            timing_boundaries[field].append(timing[field])
                    windows.append(sample_windows)
                after_memory = _memory_snapshot(
                    self.device, self.execution_runtime
                )
                result = {
                        "case_id": definition.id,
                        "authored_scope": definition.scope,
                        "resolved_scope": resolved_scope,
                        "mode": definition.mode,
                        "warmup_iterations": warmup,
                        "pilot_iterations": pilot_iterations,
                        "samples": samples,
                        "latency": _latency_summary(windows, inner_iterations),
                        "memory_observation": {
                            "before": before_memory,
                            "after": after_memory,
                            "attribution": (
                                "point-in-time allocator/RSS diagnostic; peak live-set is "
                                "modeled and separately observed in trace mode"
                            ),
                        },
                    }
                if self.execution_runtime is not None:
                    result["timing_boundaries"] = {
                        "primary_timer": self.execution_runtime.timer_source,
                        "timer_resolution_ns": (
                            self.execution_runtime.timer_resolution_ns
                        ),
                        "completion_protocol": (
                            self.execution_runtime.completion_protocol
                        ),
                        **timing_boundaries,
                    }
                results.append(result)
        observation: dict[str, Any] = {
            "schema": "groundupscale.dev/benchmark-observation/v1alpha1",
            "device": self.device,
            "instrumentation_profile": (
                "baseline-timing"
                if self.execution_runtime is not None
                else "benchmark"
            ),
            "synchronization": "measurement-boundaries-only",
            "seed": self.seed,
            "torch_num_threads": torch.get_num_threads(),
            "cases": results,
        }
        if self.execution_runtime is not None:
            observation.update(
                {
                    "lane": "baseline-timing",
                    "diagnostic_profiling": "separate-artifact",
                    "primary_timer": self.execution_runtime.timer_source,
                    "completion_boundary": {
                        "kind": "device-event-stream-completion",
                        "closed": True,
                        "protocol": self.execution_runtime.completion_protocol,
                        "per_module_synchronization": False,
                    },
                }
            )
        return observation


__all__ = [
    "BenchmarkRunner",
    "resolve_device",
    "resolve_module_scope",
    "synchronize",
]
