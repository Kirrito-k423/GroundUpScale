"""Lazy real-device multi-Shape resource probes for one Ascend NPU cohort."""

from __future__ import annotations

from collections.abc import Callable
import importlib
import math
import platform
import statistics
from typing import Any

from groundupscale.benchmark.hardware_microbenchmark import OBSERVATION_SCHEMA
from groundupscale.schemas.v1alpha1 import (
    HardwareBenchmarkSuiteDocument,
    HardwareProbeSpec,
)


CaseExecutor = Callable[
    [HardwareProbeSpec, tuple[int, ...], int, int], dict[str, object]
]


class AscendNpuHardwareMicrobenchmarkRunner:
    """Measure matrix compute and HBM copy capacity without eager torch_npu import."""

    def __init__(
        self,
        suite: HardwareBenchmarkSuiteDocument,
        *,
        environment: dict[str, object],
        cohort: dict[str, object],
        cohort_evidence: dict[str, str],
        logical_device_index: int = 0,
        seed: int = 20260810,
        case_executor: CaseExecutor | None = None,
        software: dict[str, str] | None = None,
    ) -> None:
        if suite.spec.target.device != f"npu-{logical_device_index}":
            raise ValueError(
                "Ascend suite target must match the selected logical NPU device"
            )
        if logical_device_index < 0:
            raise ValueError("logical_device_index must be non-negative")
        if not isinstance(cohort.get("cohort_id"), str):
            raise ValueError("completed Ascend cohort evidence is required")
        self.suite = suite
        self.environment = environment
        self.cohort = cohort
        self.cohort_evidence = cohort_evidence
        self.logical_device_index = logical_device_index
        self.seed = seed
        self._case_executor = case_executor or self._execute_case
        self._software = software
        self._runtime: tuple[Any, Any] | None = None

    def _load_runtime(self) -> tuple[Any, Any]:
        if self._runtime is None:
            torch = importlib.import_module("torch")
            torch_npu = importlib.import_module("torch_npu")
            self._runtime = (torch, torch_npu)
        return self._runtime

    @staticmethod
    def _timing_summary(samples_ns: list[float]) -> dict[str, float]:
        quartiles = statistics.quantiles(samples_ns, n=4, method="inclusive")
        median_ns = float(statistics.median(samples_ns))
        iqr_ns = float(quartiles[2] - quartiles[0])
        return {
            "median_ns": median_ns,
            "q1_ns": float(quartiles[0]),
            "q3_ns": float(quartiles[2]),
            "iqr_ns": iqr_ns,
            "iqr_over_median": iqr_ns / median_ns,
        }

    def _time_invocation(self, invoke: Callable[[], Any], inner_iterations: int) -> float:
        torch, _ = self._load_runtime()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        start.record()
        result = None
        for _ in range(inner_iterations):
            result = invoke()
        end.record()
        end.synchronize()
        torch.npu.synchronize()
        if result is not None and result.device.type != "npu":
            raise RuntimeError("cpu-fallback-detected")
        return float(start.elapsed_time(end)) * 1_000_000 / inner_iterations

    def _execute_case(
        self,
        probe: HardwareProbeSpec,
        shape: tuple[int, ...],
        concurrency: int,
        case_index: int,
    ) -> dict[str, object]:
        if concurrency != 1:
            raise ValueError("initial Ascend probes require concurrency=1")
        torch, _ = self._load_runtime()
        torch.npu.set_device(self.logical_device_index)
        logical_device = f"npu:{self.logical_device_index}"
        generator = torch.Generator(device="cpu").manual_seed(
            self.seed + case_index
        )
        if probe.kind == "matrix_multiply":
            m, k, n = shape
            left_cpu = torch.randn(
                (m, k), dtype=torch.float32, generator=generator
            )
            right_cpu = torch.randn(
                (k, n), dtype=torch.float32, generator=generator
            )
            left = left_cpu.to(logical_device)
            right = right_cpu.to(logical_device)

            def invoke() -> Any:
                return torch.matmul(left, right)

            oracle = torch.matmul(left_cpu.double(), right_cpu.double())
            actual = invoke().cpu().double()
            absolute_error = (actual - oracle).abs()
            relative_error = absolute_error / oracle.abs().clamp_min(1e-12)
            atol = 0.001
            rtol = 0.001
            passed = bool(
                torch.isfinite(actual).all()
                and torch.isfinite(oracle).all()
                and tuple(actual.shape) == tuple(oracle.shape)
                and torch.allclose(actual, oracle, atol=atol, rtol=rtol)
            )
            correctness = {
                "status": "passed" if passed else "failed",
                "oracle": "cpu-float64-matmul",
                "atol": atol,
                "rtol": rtol,
                "max_absolute_error": float(absolute_error.max().item()),
                "max_relative_error": float(relative_error.max().item()),
            }
            work = 2 * m * k * n
            unit = "FLOP/s"
            implementation = "pytorch-ascend-matmul"
            assumptions = ["one multiply and one add count as two FLOPs"]
        elif probe.kind == "memory_copy":
            elements = shape[0]
            source_cpu = torch.randn(
                elements, dtype=torch.float32, generator=generator
            )
            source = source_cpu.to(logical_device)
            output = torch.empty_like(source)

            def invoke() -> Any:
                return output.copy_(source)

            actual = invoke().cpu()
            passed = bool(torch.equal(actual, source_cpu))
            correctness = {
                "status": "passed" if passed else "failed",
                "oracle": "exact-cpu-source-copy",
            }
            work = 2 * elements * 4
            unit = "B/s"
            implementation = "pytorch-ascend-hbm-copy"
            assumptions = ["copy counts one compulsory read and one compulsory write"]
        else:
            raise ValueError(f"unsupported Ascend hardware probe kind: {probe.kind}")
        torch.npu.synchronize()
        for _ in range(self.suite.spec.warmup_iterations):
            invoke()
        torch.npu.synchronize()
        pilot_ns = max(self._time_invocation(invoke, 1), 1.0)
        target_ns = float(self.suite.spec.target_window_ms) * 1_000_000
        inner_iterations = max(
            1,
            min(
                self.suite.spec.maximum_inner_iterations,
                math.ceil(target_ns / pilot_ns),
            ),
        )
        samples_ns = [
            self._time_invocation(invoke, inner_iterations)
            for _ in range(self.suite.spec.samples)
        ]
        summary = self._timing_summary(samples_ns)
        stable = summary["iqr_over_median"] <= 0.10
        eligible = stable and correctness["status"] == "passed"
        reason = (
            None
            if eligible
            else "correctness-failed"
            if correctness["status"] != "passed"
            else "unstable-measurement"
        )
        torch.npu.empty_cache()
        return {
            "shape": list(shape),
            "threads": concurrency,
            "work": work,
            "unit": unit,
            "implementation": implementation,
            "inner_iterations": inner_iterations,
            "samples_ns": samples_ns,
            **summary,
            "achieved_rate": work * 1_000_000_000 / summary["median_ns"],
            "eligible": eligible,
            "eligibility": {
                "maximum_iqr_over_median": 0.10,
                "reason": reason,
            },
            "correctness": correctness,
            "timer": {
                "source": "torch.npu.Event.elapsed_time",
                "resolution_ns": 20.0,
            },
            "assumptions": assumptions,
        }

    def run(self) -> dict[str, object]:
        probes: list[dict[str, object]] = []
        case_index = 0
        eligible_shape_count_by_resource: dict[str, int] = {}
        for probe in self.suite.spec.probes:
            cases: list[dict[str, object]] = []
            for shape in probe.shapes:
                for concurrency in probe.thread_counts:
                    cases.append(
                        self._case_executor(
                            probe, shape, concurrency, case_index
                        )
                    )
                    case_index += 1
            units = {case["unit"] for case in cases}
            if len(units) != 1:
                raise ValueError(f"probe {probe.id!r} emitted inconsistent units")
            eligible_shape_count_by_resource[probe.resource] = len(
                {
                    tuple(case["shape"])
                    for case in cases
                    if case.get("eligible") is True
                }
            )
            probes.append(
                {
                    "probe_id": probe.id,
                    "probe_kind": probe.kind,
                    "resource": probe.resource,
                    "dtype": probe.dtype,
                    "unit": next(iter(units)),
                    "cases": cases,
                }
            )

        reason_codes = [
            str(reason) for reason in self.environment.get("reason_codes", [])
        ]
        if self.environment.get("eligible") is not True:
            reason_codes.append("measurement-preflight-ineligible")
        for resource, count in eligible_shape_count_by_resource.items():
            if count < 10:
                reason_codes.append(f"insufficient-eligible-shapes:{resource}")
        power_clock = self.cohort.get("power_clock")
        power_policy = (
            power_clock.get("power_policy")
            if isinstance(power_clock, dict)
            else None
        )
        power_unknown = not isinstance(power_policy, str) or power_policy.startswith(
            "unsupported"
        )
        if power_unknown:
            reason_codes.append("power-policy-unobserved")
        blocking_reasons = [
            reason
            for reason in reason_codes
            if reason != "power-policy-unobserved"
        ]
        quality_status = (
            "quarantined"
            if blocking_reasons
            else "exploratory"
            if power_unknown
            else "qualified"
        )
        if self._software is None:
            torch, torch_npu = self._load_runtime()
            software = {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "torch_npu": str(torch_npu.__version__),
                "platform": platform.platform(),
            }
        else:
            software = dict(self._software)
        operation_classes = [
            "MatMul" if probe.kind == "matrix_multiply" else probe.kind
            for probe in self.suite.spec.probes
        ]
        return {
            "schema": OBSERVATION_SCHEMA,
            "suite": {
                "name": self.suite.metadata.name,
                "version": self.suite.metadata.version,
            },
            "target": self.suite.spec.target.model_dump(mode="json"),
            "hardware_cohort": self.cohort["cohort_id"],
            "cohort_evidence": dict(self.cohort_evidence),
            "environment": dict(self.environment),
            "software": software,
            "validity_domain": {
                "operation_classes": operation_classes,
                "dtype": "float32",
                "layout": "row-major-contiguous",
                "logical_device": f"npu:{self.logical_device_index}",
                "execution_mode": "pytorch-eager",
                "shape_support": "observed-stratified-shapes-only",
            },
            "uncertainty": {
                "method": "per-shape-median-cross-shape-quantiles",
                "robust_quantile": 0.8,
                "optimistic_quantile": 0.95,
                "maximum_iqr_over_median": 0.10,
            },
            "quality": {
                "status": quality_status,
                "reason_codes": sorted(set(reason_codes)),
                "eligible_shape_count_by_resource": (
                    eligible_shape_count_by_resource
                ),
            },
            "statistics": {
                "per_shape": "median",
                "stability": "IQR/median <= 10%",
                "probe_envelope": "P80 and P95 per Shape",
                "resource_envelope": "maximum probe P80/P95",
            },
            "probes": probes,
        }


__all__ = ["AscendNpuHardwareMicrobenchmarkRunner"]
