"""CPU dispatch/intercept and instrumentation-overhead ablation experiments."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import math
import platform
import random
import statistics
import time
from typing import Any, Callable, Iterator

import numpy as np
import torch
from torch import Tensor, nn

from groundupscale.benchmark.measurement import BenchmarkRunner
from groundupscale.benchmark.trace import _memory_snapshot, _tensor_metadata
from groundupscale.specs import AnalysisBundle


MATMUL_FACTORS = (1, 2, 4, 8, 16, 32)
INSTRUMENTATION_MODES = (
    "off",
    "empty-hooks",
    "timing-hooks",
    "full-trace-hooks",
)


def _summary(samples_ns: list[float]) -> dict[str, Any]:
    quartiles = statistics.quantiles(samples_ns, n=4, method="inclusive")
    median_ns = float(statistics.median(samples_ns))
    return {
        "samples_ns": samples_ns,
        "sample_count": len(samples_ns),
        "median_ns": median_ns,
        "q1_ns": float(quartiles[0]),
        "q3_ns": float(quartiles[2]),
        "iqr_ns": float(quartiles[2] - quartiles[0]),
        "iqr_over_median": (
            float(quartiles[2] - quartiles[0]) / median_ns
            if median_ns
            else None
        ),
    }


def fit_affine_latency(points: list[dict[str, float]]) -> dict[str, float]:
    """Fit duration_ns = intercept_ns + slope_ns_per_flop * work_flops."""

    if len(points) < 2:
        raise ValueError("affine latency fit requires at least two points")
    work = np.asarray([point["work_flops"] for point in points], dtype=np.float64)
    duration = np.asarray([point["median_ns"] for point in points], dtype=np.float64)
    centered_work = work - np.mean(work)
    centered_duration = duration - np.mean(duration)
    denominator = float(np.sum(centered_work**2))
    if denominator == 0:
        raise ValueError("affine latency fit requires distinct work values")
    slope_ns_per_flop = float(
        np.sum(centered_work * centered_duration) / denominator
    )
    intercept_ns = float(np.mean(duration) - slope_ns_per_flop * np.mean(work))
    fitted = intercept_ns + slope_ns_per_flop * work
    residual_sum = float(np.sum((duration - fitted) ** 2))
    total_sum = float(np.sum((duration - np.mean(duration)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum else 1.0
    return {
        "intercept_ns": intercept_ns,
        "slope_ns_per_flop": slope_ns_per_flop,
        "rate_flops_per_second": (
            1_000_000_000.0 / slope_ns_per_flop
            if slope_ns_per_flop > 0
            else float("inf")
        ),
        "r_squared": r_squared,
        "residual_sum_squares": residual_sum,
    }


def _bootstrap_affine(
    records: list[dict[str, Any]], *, seed: int, iterations: int = 1_000
) -> dict[str, Any]:
    generator = random.Random(seed)
    intercepts: list[float] = []
    rates: list[float] = []
    for _ in range(iterations):
        points = []
        for record in records:
            samples = record["latency"]["samples_ns"]
            resampled = [generator.choice(samples) for _ in samples]
            points.append(
                {
                    "work_flops": float(record["flops"]),
                    "median_ns": float(statistics.median(resampled)),
                }
            )
        fit = fit_affine_latency(points)
        intercepts.append(fit["intercept_ns"])
        rates.append(fit["rate_flops_per_second"])
    intercepts.sort()
    rates.sort()

    def percentile(values: list[float], fraction: float) -> float:
        index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
        return float(values[index])

    return {
        "iterations": iterations,
        "intercept_ns_95ci": [
            percentile(intercepts, 0.025),
            percentile(intercepts, 0.975),
        ],
        "rate_flops_per_second_95ci": [
            percentile(rates, 0.025),
            percentile(rates, 0.975),
        ],
    }


def _calibrate_inner_iterations(
    invoke: Callable[[], Tensor], target_window_ns: int
) -> int:
    pilot_iterations = 3
    started_ns = time.perf_counter_ns()
    for _ in range(pilot_iterations):
        invoke()
    duration_ns = max(1, time.perf_counter_ns() - started_ns)
    per_call_ns = duration_ns / pilot_iterations
    return max(1, min(1_000, math.ceil(target_window_ns / per_call_ns)))


def run_matmul_scaling_ablation(
    *,
    samples: int,
    warmup: int,
    target_window_ns: int,
    seed: int,
) -> dict[str, Any]:
    """Scale only M/sequence so work grows exactly with the authored factors."""

    hidden = 512
    base_sequence = 512
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = torch.randn(hidden, hidden, generator=generator, dtype=torch.float32)
    inputs = {
        factor: torch.randn(
            1,
            base_sequence * factor,
            hidden,
            generator=generator,
            dtype=torch.float32,
        )
        for factor in MATMUL_FACTORS
    }
    invocations = {
        factor: (lambda factor=factor: torch.matmul(inputs[factor], weight))
        for factor in MATMUL_FACTORS
    }
    with torch.inference_mode():
        for factor in MATMUL_FACTORS:
            for _ in range(warmup):
                invocations[factor]()
        inner_iterations = {
            factor: _calibrate_inner_iterations(
                invocations[factor], target_window_ns
            )
            for factor in MATMUL_FACTORS
        }
        samples_by_factor: dict[int, list[float]] = {
            factor: [] for factor in MATMUL_FACTORS
        }
        order_generator = random.Random(seed + 1)
        for _ in range(samples):
            order = list(MATMUL_FACTORS)
            order_generator.shuffle(order)
            for factor in order:
                iterations = inner_iterations[factor]
                started_ns = time.perf_counter_ns()
                for _ in range(iterations):
                    invocations[factor]()
                duration_ns = time.perf_counter_ns() - started_ns
                samples_by_factor[factor].append(duration_ns / iterations)

        empty_input = torch.empty((1, 0, hidden), dtype=torch.float32)
        empty_invoke = lambda: torch.matmul(empty_input, weight)
        empty_inner = _calibrate_inner_iterations(empty_invoke, target_window_ns)
        empty_samples = []
        for _ in range(samples):
            started_ns = time.perf_counter_ns()
            for _ in range(empty_inner):
                empty_invoke()
            empty_samples.append(
                (time.perf_counter_ns() - started_ns) / empty_inner
            )

    records = []
    for factor in MATMUL_FACTORS:
        sequence = base_sequence * factor
        flops = 2 * sequence * hidden * hidden
        latency = _summary(samples_by_factor[factor])
        records.append(
            {
                "factor": factor,
                "input_shape": [1, sequence, hidden],
                "weight_shape": [hidden, hidden],
                "output_shape": [1, sequence, hidden],
                "flops": flops,
                "flop_factor": factor,
                "inner_iterations": inner_iterations[factor],
                "latency": latency,
                "effective_flops_per_second": (
                    flops * 1_000_000_000.0 / latency["median_ns"]
                ),
            }
        )
    fit_points = [
        {"work_flops": float(record["flops"]), "median_ns": record["latency"]["median_ns"]}
        for record in records
    ]
    fit = fit_affine_latency(fit_points)
    fit.update(_bootstrap_affine(records, seed=seed + 2))
    large_records = [record for record in records if record["factor"] >= 4]
    large_fit = fit_affine_latency(
        [
            {
                "work_flops": float(record["flops"]),
                "median_ns": record["latency"]["median_ns"],
            }
            for record in large_records
        ]
    )
    large_fit.update(_bootstrap_affine(large_records, seed=seed + 3))
    baseline_ns = records[0]["latency"]["median_ns"]
    for record in records:
        record["latency_over_1x"] = record["latency"]["median_ns"] / baseline_ns
        record["latency_growth_minus_flop_factor"] = (
            record["latency_over_1x"] - record["flop_factor"]
        )
    intercept_ci = fit["intercept_ns_95ci"]
    fit["intercept_share_of_1x_median"] = fit["intercept_ns"] / baseline_ns
    fit["material_positive_intercept"] = bool(
        intercept_ci[0] > 0 and fit["intercept_ns"] >= baseline_ns * 0.05
    )
    return {
        "hypothesis": (
            "preallocated CPU MatMul has a fixed dispatch component plus a work-proportional component"
        ),
        "controlled_dimension": "M/sequence only; K=N=512 remain fixed",
        "timed_region": "torch.matmul invocation and output allocation only",
        "input_and_weight_preallocated": True,
        "explicit_h2d_in_timed_region": False,
        "device": "cpu",
        "dtype": "float32",
        "records": records,
        "zero_work_matmul": {
            "input_shape": [1, 0, hidden],
            "inner_iterations": empty_inner,
            "latency": _summary(empty_samples),
            "share_of_1x_median": (
                statistics.median(empty_samples) / baseline_ns
            ),
        },
        "affine_fit_all_shapes": fit,
        "affine_fit_factors_4_to_32": large_fit,
        "interpretation_guardrail": (
            "a positive intercept may include Python, dispatcher, allocator, and thread-pool costs; "
            "it is not evidence of H2D because all tensors are resident on CPU"
        ),
    }


@contextmanager
def _instrumentation(
    model: nn.Module,
    mode: str,
) -> Iterator[dict[str, Any]]:
    if mode not in INSTRUMENTATION_MODES:
        raise ValueError(f"unsupported instrumentation mode: {mode}")
    handles: list[Any] = []
    events: list[dict[str, Any]] = []
    starts: dict[int, int] = {}

    if mode != "off":
        def pre_hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if mode in {"timing-hooks", "full-trace-hooks"}:
                starts[id(module)] = time.perf_counter_ns()
            if mode == "full-trace-hooks":
                _tensor_metadata(inputs)

        def post_hook(
            module: nn.Module, inputs: tuple[Any, ...], output: Any
        ) -> None:
            if mode == "timing-hooks":
                events.append(
                    {"duration_ns": time.perf_counter_ns() - starts.pop(id(module))}
                )
            elif mode == "full-trace-hooks":
                events.append(
                    {
                        "duration_ns": time.perf_counter_ns() - starts.pop(id(module)),
                        "outputs": _tensor_metadata(output),
                        "memory": _memory_snapshot("cpu"),
                    }
                )

        for module in model.modules():
            if hasattr(module, "stable_path"):
                handles.append(module.register_forward_pre_hook(pre_hook))
                handles.append(module.register_forward_hook(post_hook, always_call=True))
    try:
        yield {"events": events, "registered_hook_handles": len(handles)}
    finally:
        for handle in handles:
            handle.remove()


def _bootstrap_ratio(
    baseline: list[float], candidate: list[float], *, seed: int, iterations: int = 1_000
) -> list[float]:
    if len(baseline) != len(candidate):
        raise ValueError("paired ratio bootstrap requires equal sample counts")
    generator = random.Random(seed)
    ratios = []
    for _ in range(iterations):
        indices = [generator.randrange(len(baseline)) for _ in baseline]
        baseline_median = statistics.median(baseline[index] for index in indices)
        candidate_median = statistics.median(candidate[index] for index in indices)
        ratios.append(candidate_median / baseline_median)
    ratios.sort()
    return [
        float(ratios[round((len(ratios) - 1) * 0.025)]),
        float(ratios[round((len(ratios) - 1) * 0.975)]),
    ]


def run_instrumentation_ablation(
    bundle: AnalysisBundle,
    *,
    samples: int,
    warmup: int,
    seed: int,
) -> dict[str, Any]:
    runner = BenchmarkRunner(bundle, seed=seed)
    if runner.device != "cpu":
        raise ValueError("instrumentation ablation currently requires CPU placement")
    model, hidden = runner._model_and_input()
    invocations = runner._invocations(model, hidden)
    e2e_invocations = [
        (case_id, invoke)
        for case_id, (stable_path, invoke) in invocations.items()
        if stable_path == model.stable_path
    ]
    if len(e2e_invocations) != 1:
        raise ValueError(f"expected one E2E invocation, found {len(e2e_invocations)}")
    case_id, invoke = e2e_invocations[0]
    active_hooks_after_capture = sum(
        len(module._forward_pre_hooks) + len(module._forward_hooks)
        for module in model.modules()
    )
    samples_by_mode: dict[str, list[float]] = {
        mode: [] for mode in INSTRUMENTATION_MODES
    }
    hook_counts: dict[str, int] = {}
    with torch.inference_mode():
        for mode in INSTRUMENTATION_MODES:
            with _instrumentation(model, mode) as diagnostics:
                hook_counts[mode] = diagnostics["registered_hook_handles"]
                for _ in range(warmup):
                    invoke()
        order_generator = random.Random(seed + 10)
        for _ in range(samples):
            order = list(INSTRUMENTATION_MODES)
            order_generator.shuffle(order)
            for mode in order:
                with _instrumentation(model, mode):
                    started_ns = time.perf_counter_ns()
                    invoke()
                    samples_by_mode[mode].append(
                        float(time.perf_counter_ns() - started_ns)
                    )

    modes = []
    baseline = samples_by_mode["off"]
    baseline_median = statistics.median(baseline)
    for index, mode in enumerate(INSTRUMENTATION_MODES):
        latency = _summary(samples_by_mode[mode])
        ratio_ci = _bootstrap_ratio(
            baseline,
            samples_by_mode[mode],
            seed=seed + 20 + index,
        )
        modes.append(
            {
                "mode": mode,
                "registered_hook_handles": hook_counts[mode],
                "latency": latency,
                "median_over_off": latency["median_ns"] / baseline_median,
                "median_over_off_95ci": ratio_ci,
                "statistically_detected_overhead": bool(
                    mode != "off" and ratio_ci[0] > 1.0
                ),
                "material_overhead_over_5_percent": bool(
                    mode != "off" and ratio_ci[0] > 1.05
                ),
            }
        )
    return {
        "hypothesis": "forward-hook instrumentation inflates E2E latency",
        "e2e_case_id": case_id,
        "modes_interleaved": True,
        "hook_registration_and_removal_outside_timed_region": True,
        "active_hooks_after_formal_input_capture": active_hooks_after_capture,
        "formal_benchmark_timed_path_has_hooks": active_hooks_after_capture != 0,
        "formal_benchmark_timed_path_prints_per_module": False,
        "trace_is_a_separate_execution_from_benchmark": True,
        "modes": modes,
        "verdict": {
            "formal_benchmark_contaminated_by_hooks": active_hooks_after_capture != 0,
            "empty_hooks_material": next(
                item for item in modes if item["mode"] == "empty-hooks"
            )["material_overhead_over_5_percent"],
            "timing_hooks_material": next(
                item for item in modes if item["mode"] == "timing-hooks"
            )["material_overhead_over_5_percent"],
            "full_trace_hooks_material": next(
                item for item in modes if item["mode"] == "full-trace-hooks"
            )["material_overhead_over_5_percent"],
            "full_trace_hooks_overhead_detected": next(
                item for item in modes if item["mode"] == "full-trace-hooks"
            )["statistically_detected_overhead"],
        },
    }


def run_cpu_ablation(
    bundle: AnalysisBundle,
    *,
    samples: int = 25,
    warmup: int = 10,
    target_window_ns: int = 20_000_000,
    seed: int = 20260807,
) -> dict[str, Any]:
    if samples < 8:
        raise ValueError("samples must be at least 8")
    return {
        "schema": "groundupscale.dev/cpu-overhead-ablation/v1alpha1",
        "created_at": datetime.now(UTC).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "protocol": {
            "samples": samples,
            "warmup": warmup,
            "target_window_ns": target_window_ns,
            "seed": seed,
        },
        "matmul_scaling": run_matmul_scaling_ablation(
            samples=samples,
            warmup=warmup,
            target_window_ns=target_window_ns,
            seed=seed,
        ),
        "instrumentation": run_instrumentation_ablation(
            bundle,
            samples=samples,
            warmup=warmup,
            seed=seed,
        ),
    }


__all__ = ["fit_affine_latency", "run_cpu_ablation"]
