"""Minimally intrusive Benchmark Case execution on CPU and MPS."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from hashlib import sha256
import inspect
from pathlib import Path
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
from groundupscale.benchmark.frontier_evidence import EXACT_FRONTIER_TIMING_POLICY
from groundupscale.execution_runtime import ExecutionRuntime
from groundupscale.schemas.v1alpha1 import BenchmarkDefinition
from groundupscale.specs import AnalysisBundle
from groundupscale.ir import content_fingerprint


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


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _torch_runtime_identity() -> dict[str, Any]:
    runtime = Path(str(torch._C.__file__)).resolve()
    torch_cpu = Path(torch.__file__).resolve().parent / "lib/libtorch_cpu.dylib"
    dispatch_binaries = []
    if torch_cpu.is_file():
        dispatch_binaries.append(
            {
                "role": "cpu-dispatch-provider",
                "name": torch_cpu.name,
                "sha256": _file_sha256(torch_cpu),
            }
        )
    return {
        "runtime_binary": runtime.name,
        "runtime_binary_sha256": _file_sha256(runtime),
        "torch_version": str(torch.__version__),
        "torch_build_config_sha256": sha256(
            torch.__config__.show().encode("utf-8")
        ).hexdigest(),
        "dispatch_provider_binaries": dispatch_binaries,
    }


def _tensor_bytes_digest(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tensor_contract(tensor: Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "layout": "row-major-contiguous" if tensor.is_contiguous() else "strided",
        "minimum_alignment_bytes": 64 if tensor.data_ptr() % 64 == 0 else 0,
    }


def _candidate_and_correctness(
    target: nn.Module,
    inputs: tuple[Tensor, ...],
    invoke: Callable[[], Tensor],
    *,
    seed: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    operation = str(getattr(target, "operation", "unknown"))
    if operation != "MatMul" or device != "cpu":
        return (
            {"status": "unsupported", "reason_codes": ["candidate-identity-unsupported"]},
            {"status": "unsupported", "reason_codes": ["input-corpus-unsupported"]},
            {"status": "not_evaluated", "reason_codes": ["exact-oracle-unsupported"]},
            {"status": "unsupported", "reason_codes": ["execution-contract-unsupported"]},
        )
    weight = getattr(target, "weight", None)
    equation = getattr(target, "equation", None)
    if isinstance(weight, Tensor) and len(inputs) == 1:
        operands = (*inputs, weight)
        reference = torch.matmul(inputs[0].double(), weight.detach().double())
        candidate_operation = "torch.matmul"
        family = "torch.matmul.cpu.fp32"
    elif weight is None and len(inputs) == 2:
        operands = inputs
        reference = torch.matmul(inputs[0].double(), inputs[1].double())
        if equation == "bhqk,bhkd->bqhd":
            reference = reference.transpose(1, 2).contiguous()
            candidate_operation = "torch.matmul+transpose-contiguous"
            family = "torch.matmul.transpose-contiguous.cpu.fp32"
        elif equation is None:
            candidate_operation = "torch.matmul"
            family = "torch.matmul.cpu.fp32"
        else:
            raise ValueError(f"unsupported exact MatMul equation: {equation}")
    else:
        raise ValueError("exact MatMul evidence requires one state or two tensor operands")
    actual = invoke().detach()
    actual64 = actual.double()
    absolute_error = (actual64 - reference).abs()
    relative_error = absolute_error / reference.abs().clamp_min(1e-12)
    atol = 1e-5
    rtol = 1e-4
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    shape_matches = tuple(actual.shape) == tuple(reference.shape)
    passed = bool(
        finite
        and shape_matches
        and torch.allclose(actual64, reference, atol=atol, rtol=rtol)
    )
    source = inspect.getsource(type(target)).encode("utf-8")
    runtime = _torch_runtime_identity()
    candidate_identity = {
        "schema": "groundupscale.dev/operator-candidate-identity/v1alpha1",
        "status": "resolved",
        "family": family,
        "provider": "pytorch",
        "operation": candidate_operation,
        "module_class": f"{type(target).__module__}.{type(target).__qualname__}",
        "module_source_sha256": sha256(source).hexdigest(),
        "runtime": runtime,
        "dispatch_provider_binaries": runtime["dispatch_provider_binaries"],
        "dispatch_mode": "eager",
        "workspace_policy": "framework-managed-stable-after-warmup",
    }
    if weight is None:
        candidate_identity["equation"] = equation
    candidate_identity["candidate_digest"] = content_fingerprint(candidate_identity)
    tensor_digests = [_tensor_bytes_digest(value) for value in operands]
    input_corpus = {
        "schema": "groundupscale.dev/input-corpus-identity/v1alpha1",
        "status": "resolved",
        "generator": "groundupscale.reference.TwoLayerTransformer",
        "seed": seed,
        "distribution": "deterministic-normal-fp32",
        "tensor_sha256": tensor_digests,
    }
    input_corpus["input_corpus_digest"] = content_fingerprint(input_corpus)
    tensors = [*operands, actual]
    execution_contract = {
        "schema": "groundupscale.dev/operator-execution-contract/v1alpha1",
        "status": "resolved",
        "operand_contracts": [_tensor_contract(value) for value in operands],
        "result_contract": _tensor_contract(actual),
        "execution_mode": "eager",
        "cache_state": "warm-reused-inputs-and-weights",
        "working_set_bytes": sum(value.numel() * value.element_size() for value in tensors),
        "concurrency": "single-operator-no-overlap",
    }
    execution_contract["execution_contract_digest"] = content_fingerprint(
        execution_contract
    )
    correctness = {
        "schema": "groundupscale.dev/operator-correctness-evidence/v1alpha1",
        "status": "passed" if passed else "failed",
        "candidate_family": family,
        "candidate_digest": candidate_identity["candidate_digest"],
        "input_corpus_digest": input_corpus["input_corpus_digest"],
        "oracle": {
            "policy_id": "matmul-fp32-float64-oracle-v1",
            "version": "1.0.0",
            "provider": "torch.float64.matmul",
            "atol": atol,
            "rtol": rtol,
            "accumulation_dtype": "float64",
            "invariants": ["shape-exact", "finite-output"],
        },
        "max_absolute_error": float(absolute_error.max().item()),
        "max_relative_error": float(relative_error.max().item()),
        "shape_matches": shape_matches,
        "finite": finite,
        "actual_output_sha256": _tensor_bytes_digest(actual),
        "reference_output_sha256": _tensor_bytes_digest(reference),
    }
    return candidate_identity, input_corpus, correctness, execution_contract


def _warmup_convergence(
    invoke: Callable[[], Tensor], *, device: str, warmup_iterations: int
) -> dict[str, Any]:
    count = int(EXACT_FRONTIER_TIMING_POLICY["convergence_window_count"])
    iterations = int(
        EXACT_FRONTIER_TIMING_POLICY["convergence_iterations_per_window"]
    )
    windows: list[float] = []
    for _ in range(count):
        synchronize(device)
        started = time.perf_counter_ns()
        for _ in range(iterations):
            invoke()
        synchronize(device)
        windows.append((time.perf_counter_ns() - started) / iterations)
    first = float(statistics.median(windows[:3]))
    last = float(statistics.median(windows[-3:]))
    drift = abs(last - first) / last
    return {
        "policy": EXACT_FRONTIER_TIMING_POLICY,
        "warmup_iterations": warmup_iterations,
        "window_samples_ns": windows,
        "median_drift": drift,
        "converged": bool(
            warmup_iterations
            >= EXACT_FRONTIER_TIMING_POLICY["minimum_warmup_iterations"]
            and drift
            <= EXACT_FRONTIER_TIMING_POLICY["maximum_warmup_median_drift"]
        ),
    }


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
    ) -> dict[
        str, tuple[str, Callable[[], Tensor], nn.Module, tuple[Tensor, ...]]
    ]:
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
                if self.execution_runtime is not None:
                    self.execution_runtime.execute_checked(
                        lambda: model(hidden)
                    )
                else:
                    model(hidden)
                self._synchronize()
        finally:
            for handle in handles:
                handle.remove()

        invocations: dict[
            str, tuple[str, Callable[[], Tensor], nn.Module, tuple[Tensor, ...]]
        ] = {}
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
                    model,
                    (hidden,),
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
                target,
                inputs,
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
        selected_case_ids: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        if samples_override is not None and samples_override < 4:
            raise ValueError("samples_override must be at least 4")
        if windows_per_sample <= 0:
            raise ValueError("windows_per_sample must be positive")
        model, hidden = self._model_and_input()
        invocations = self._invocations(model, hidden)
        definitions = _case_definitions(self.bundle)
        selected = set(selected_case_ids or ())
        available = {definition.id for definition in definitions}
        unknown = selected - available
        if unknown:
            raise ValueError(
                "unknown Benchmark Case ids: " + ", ".join(sorted(unknown))
            )
        results: list[dict[str, Any]] = []

        with torch.inference_mode():
            for definition in definitions:
                if selected and definition.id not in selected:
                    continue
                resolved_scope, invoke, target, inputs = invocations[definition.id]
                warmup = (
                    definition.warmup_iterations
                    if warmup_override is None
                    else warmup_override
                )
                samples = definition.samples if samples_override is None else samples_override
                for _ in range(warmup):
                    if self.execution_runtime is not None:
                        self.execution_runtime.execute_checked(invoke)
                    else:
                        invoke()
                self._synchronize()

                (
                    candidate_identity,
                    input_corpus,
                    operator_correctness,
                    execution_contract,
                ) = _candidate_and_correctness(
                    target,
                    inputs,
                    invoke,
                    seed=self.seed,
                    device=self.device,
                )
                warmup_convergence = _warmup_convergence(
                    invoke,
                    device=self.device,
                    warmup_iterations=warmup,
                )

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
                        "candidate_identity": candidate_identity,
                        "input_corpus": input_corpus,
                        "operator_correctness": operator_correctness,
                        "execution_contract": execution_contract,
                        "warmup_convergence": warmup_convergence,
                        "timing_contract": {
                            "policy": EXACT_FRONTIER_TIMING_POLICY,
                            "timing_scope": "host_visible_completion",
                            "completion_boundary": "synchronous-cpu-call-return",
                            "timer": {
                                "source": "time.perf_counter_ns",
                                "monotonic": bool(
                                    time.get_clock_info("perf_counter").monotonic
                                ),
                                "resolution_ns": (
                                    time.get_clock_info("perf_counter").resolution
                                    * 1_000_000_000
                                ),
                            },
                            "instrumentation_profile": "benchmark",
                            "exclusions": [],
                        },
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
            "measurement_protocol": {
                "protocol_id": "groundupscale-cpu-baseline-timing",
                "version": "1.0.0",
            },
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
