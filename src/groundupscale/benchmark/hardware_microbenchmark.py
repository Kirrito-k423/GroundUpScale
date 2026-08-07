"""Hardware-resource microbenchmarks and capability-envelope aggregation."""

from __future__ import annotations

from collections import defaultdict
import ctypes
from datetime import UTC, datetime
from hashlib import sha256
import math
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time
from typing import Any

import torch

from groundupscale.ir import content_fingerprint
from groundupscale.schemas.v1alpha1 import (
    HardwareBenchmarkSuiteDocument,
    HardwareProbeSpec,
)


OBSERVATION_SCHEMA = (
    "groundupscale.dev/hardware-microbenchmark-observation/v1alpha1"
)
CAPABILITY_PROFILE_API_VERSION = "groundupscale.dev/v1alpha1"


class CapabilityAggregationError(ValueError):
    """Raw probe evidence cannot support a hardware capability envelope."""


class _NativeArm64Kernels:
    _library: ctypes.CDLL | None = None

    @classmethod
    def load(cls) -> ctypes.CDLL:
        if cls._library is not None:
            return cls._library
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("native scalar FMA probe requires Darwin arm64")
        source = Path(__file__).with_name("native") / "arm64_cpu_kernels.c"
        source_digest = sha256(source.read_bytes()).hexdigest()[:16]
        cache = Path(tempfile.gettempdir()) / "groundupscale-native"
        cache.mkdir(parents=True, exist_ok=True)
        library_path = cache / f"arm64-cpu-kernels-{source_digest}.dylib"
        if not library_path.exists():
            subprocess.run(
                [
                    "xcrun",
                    "clang",
                    "-O3",
                    "-dynamiclib",
                    "-arch",
                    "arm64",
                    str(source),
                    "-o",
                    str(library_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        library = ctypes.CDLL(str(library_path))
        library.groundupscale_scalar_fma.argtypes = [ctypes.c_uint64]
        library.groundupscale_scalar_fma.restype = ctypes.c_float
        cls._library = library
        return library


class HardwareMicrobenchmarkRunner:
    """Execute a strict multi-Shape CPU suite and retain every raw sample."""

    def __init__(
        self,
        suite: HardwareBenchmarkSuiteDocument,
        *,
        environment: dict[str, Any],
        seed: int = 20260807,
    ) -> None:
        if suite.spec.target.device != "cpu":
            raise ValueError("the initial hardware microbenchmark runner supports CPU only")
        self.suite = suite
        self.environment = environment
        self.seed = seed

    @staticmethod
    def _make_invocation(
        probe: HardwareProbeSpec, shape: tuple[int, ...], seed: int
    ) -> tuple[Any, int, str, str, tuple[str, ...]]:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if probe.kind == "matrix_multiply":
            m, k, n = shape
            left = torch.randn((m, k), dtype=torch.float32, generator=generator)
            right = torch.randn((k, n), dtype=torch.float32, generator=generator)
            output = torch.empty((m, n), dtype=torch.float32)
            return (
                lambda: torch.mm(left, right, out=output),
                2 * m * k * n,
                "FLOP/s",
                "pytorch-accelerate-matrix-multiply",
                ("one multiply and one add count as two FLOPs",),
            )
        elements = shape[0]
        if probe.kind == "vector_fma":
            base = torch.randn(elements, dtype=torch.float32, generator=generator)
            left = torch.randn(elements, dtype=torch.float32, generator=generator)
            right = torch.randn(elements, dtype=torch.float32, generator=generator)
            output = torch.empty(elements, dtype=torch.float32)
            return (
                lambda: torch.addcmul(base, left, right, out=output),
                2 * elements,
                "FLOP/s",
                "pytorch-elementwise-vector-fma",
                ("one elementwise multiply and add count as two FLOPs",),
            )
        if probe.kind == "scalar_fma":
            library = _NativeArm64Kernels.load()

            def scalar_chain() -> float:
                return float(library.groundupscale_scalar_fma(elements))

            return (
                scalar_chain,
                16 * elements,
                "FLOP/s",
                "native-arm64-scalar-fma",
                (
                    "eight independent scalar FMADD instructions execute per loop iteration",
                    "one scalar multiply and add count as two FLOPs",
                ),
            )
        source = torch.randn(elements, dtype=torch.float32, generator=generator)
        output = torch.empty(elements, dtype=torch.float32)
        if probe.kind == "memory_copy":
            return (
                lambda: output.copy_(source),
                2 * elements * 4,
                "B/s",
                "pytorch-inplace-memory-copy",
                ("copy counts one compulsory read and one compulsory write",),
            )
        if probe.kind == "memory_triad":
            left = torch.randn(elements, dtype=torch.float32, generator=generator)
            right = torch.randn(elements, dtype=torch.float32, generator=generator)
            return (
                lambda: torch.addcmul(source, left, right, out=output),
                4 * elements * 4,
                "B/s",
                "pytorch-elementwise-memory-triad",
                ("triad counts three compulsory reads and one compulsory write",),
            )
        raise ValueError(f"unsupported hardware probe kind: {probe.kind}")

    @staticmethod
    def _time_windows(invoke: Any, inner_iterations: int) -> int:
        started = time.perf_counter_ns()
        for _ in range(inner_iterations):
            invoke()
        return time.perf_counter_ns() - started

    def _run_case(
        self,
        probe: HardwareProbeSpec,
        shape: tuple[int, ...],
        threads: int,
        case_index: int,
    ) -> dict[str, Any]:
        torch.set_num_threads(threads)
        invoke, work, unit, implementation, assumptions = self._make_invocation(
            probe, shape, self.seed + case_index
        )
        for _ in range(self.suite.spec.warmup_iterations):
            invoke()
        pilot_ns = max(self._time_windows(invoke, 1), 1)
        target_ns = int(self.suite.spec.target_window_ms * 1_000_000)
        inner_iterations = max(
            1,
            min(
                self.suite.spec.maximum_inner_iterations,
                math.ceil(target_ns / pilot_ns),
            ),
        )
        samples_ns = [
            self._time_windows(invoke, inner_iterations) / inner_iterations
            for _ in range(self.suite.spec.samples)
        ]
        median_ns = float(statistics.median(samples_ns))
        quartiles = statistics.quantiles(samples_ns, n=4, method="inclusive")
        iqr_ns = float(quartiles[2] - quartiles[0])
        iqr_over_median = iqr_ns / median_ns
        return {
            "shape": list(shape),
            "threads": threads,
            "work": work,
            "unit": unit,
            "implementation": implementation,
            "inner_iterations": inner_iterations,
            "samples_ns": samples_ns,
            "median_ns": median_ns,
            "q1_ns": float(quartiles[0]),
            "q3_ns": float(quartiles[2]),
            "iqr_ns": iqr_ns,
            "iqr_over_median": iqr_over_median,
            "achieved_rate": work * 1_000_000_000 / median_ns,
            "eligible": iqr_over_median <= 0.10,
            "eligibility": {
                "maximum_iqr_over_median": 0.10,
                "reason": (
                    None if iqr_over_median <= 0.10 else "unstable-measurement"
                ),
            },
            "assumptions": list(assumptions),
        }

    def run(self) -> dict[str, Any]:
        old_threads = torch.get_num_threads()
        try:
            probes: list[dict[str, Any]] = []
            case_index = 0
            for probe in self.suite.spec.probes:
                cases: list[dict[str, Any]] = []
                for shape in probe.shapes:
                    for threads in probe.thread_counts:
                        cases.append(
                            self._run_case(probe, shape, threads, case_index)
                        )
                        case_index += 1
                units = {case["unit"] for case in cases}
                if len(units) != 1:
                    raise ValueError(f"probe {probe.id!r} emitted inconsistent units")
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
        finally:
            torch.set_num_threads(old_threads)
        environment_policy = self.environment.get("policy", {})
        policy_id = (
            environment_policy.get("policy_id", "unverified")
            if isinstance(environment_policy, dict)
            else "unverified"
        )
        cohort = "-".join(
            (
                self.suite.spec.target.hardware,
                self.suite.spec.target.device,
                platform.system().lower(),
                platform.machine().lower(),
                f"torch-{torch.__version__}",
                str(policy_id),
                content_fingerprint(
                    self.suite.metadata.name,
                    self.suite.metadata.version,
                    platform.platform(),
                    torch.__version__,
                    policy_id,
                )[:12],
            )
        )
        return {
            "schema": OBSERVATION_SCHEMA,
            "captured_at": datetime.now(UTC).isoformat(),
            "suite": {
                "name": self.suite.metadata.name,
                "version": self.suite.metadata.version,
            },
            "target": self.suite.spec.target.model_dump(mode="json"),
            "hardware_cohort": cohort,
            "environment": self.environment,
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
            "statistics": {
                "per_shape": "median",
                "stability": "IQR/median <= 10%",
                "probe_envelope": "P80 and P95 of best thread-count rate per Shape",
                "resource_envelope": "maximum probe P80/P95",
            },
            "probes": probes,
        }


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise CapabilityAggregationError("cannot aggregate an empty rate set")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    remainder = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * remainder


def _shape_identity(shape: object) -> tuple[object, ...]:
    if not isinstance(shape, list) or not shape:
        raise CapabilityAggregationError("each probe case requires a non-empty shape")
    return tuple(shape)


def aggregate_capability_envelope(
    observation: dict[str, Any],
    *,
    profile_name: str,
    profile_version: str,
    source_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Promote diverse probes into resource-keyed P80/P95 capacity envelopes."""

    if observation.get("schema") != OBSERVATION_SCHEMA:
        raise CapabilityAggregationError("unsupported observation schema")
    by_resource: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for probe in observation.get("probes", []):
        best_by_shape: dict[tuple[object, ...], tuple[float, int]] = {}
        for case in probe.get("cases", []):
            if case.get("eligible") is not True:
                continue
            identity = _shape_identity(case.get("shape"))
            rate = float(case["achieved_rate"])
            threads = int(case["threads"])
            previous = best_by_shape.get(identity)
            if previous is None or rate > previous[0]:
                best_by_shape[identity] = (rate, threads)
        if len(best_by_shape) < 10:
            raise CapabilityAggregationError(
                f"probe {probe.get('probe_id')!r} requires at least 10 distinct "
                f"eligible Shapes, found {len(best_by_shape)}"
            )
        rates = [rate for rate, _ in best_by_shape.values()]
        envelope = {
            "probe_id": probe["probe_id"],
            "distinct_shape_count": len(best_by_shape),
            "shape_p80_rate": _quantile(rates, 0.80),
            "shape_p95_rate": _quantile(rates, 0.95),
            "shape_best_rates": [
                {
                    "shape": list(shape),
                    "rate": rate,
                    "threads": threads,
                }
                for shape, (rate, threads) in sorted(best_by_shape.items())
            ],
        }
        by_resource[(str(probe["resource"]), str(probe["unit"]))].append(
            envelope
        )

    resources: list[dict[str, Any]] = []
    for (resource, unit), envelopes in sorted(by_resource.items()):
        envelopes.sort(key=lambda item: item["probe_id"])
        robust = max(envelopes, key=lambda item: item["shape_p80_rate"])
        optimistic = max(envelopes, key=lambda item: item["shape_p95_rate"])
        resources.append(
            {
                "resource": resource,
                "unit": unit,
                "robust_achievable_rate": robust["shape_p80_rate"],
                "optimistic_rate": optimistic["shape_p95_rate"],
                "selected_robust_probe": robust["probe_id"],
                "selected_optimistic_probe": optimistic["probe_id"],
                "aggregation": "max(probe_shape_p80)",
                "probe_envelopes": envelopes,
            }
        )

    return {
        "apiVersion": CAPABILITY_PROFILE_API_VERSION,
        "kind": "HardwareCapabilityProfile",
        "metadata": {"name": profile_name, "version": profile_version},
        "spec": {
            "target": observation["target"],
            "hardware_cohort": observation["hardware_cohort"],
            "environment": observation["environment"],
            "source": {
                "path": source_path,
                "sha256": source_sha256,
                "schema": observation["schema"],
                "suite": observation["suite"],
            },
            "resources": resources,
        },
    }


__all__ = [
    "CapabilityAggregationError",
    "HardwareMicrobenchmarkRunner",
    "aggregate_capability_envelope",
]
