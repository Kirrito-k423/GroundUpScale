from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time

import pytest
import torch

from groundupscale.benchmark import ReferenceRunner
from groundupscale.benchmark.reference import TensorMatMul
from groundupscale.compiler import SemanticCompiler
from groundupscale.specs import SpecRepository

from test_semantic_compiler import _request


REPOSITORY_ROOT = Path(__file__).parents[1]


def _best_runtime_ns(invoke: Callable[[], object], samples: int = 5) -> int:
    durations: list[int] = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        invoke()
        durations.append(time.perf_counter_ns() - started)
    return min(durations)


def _runner() -> ReferenceRunner:
    bundle = SpecRepository(REPOSITORY_ROOT).load_analysis_plan(
        REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )
    return ReferenceRunner.from_analysis_bundle(bundle, seed=20260806)


def test_reference_model_matches_cost_state_and_semantic_leaf_counts() -> None:
    runner = _runner()
    cpu = runner.run_device("cpu")

    assert tuple(cpu.output.shape) == (1, 512, 512)
    assert torch.isfinite(cpu.output).all()
    assert cpu.audit.semantic_leaf_count == 52
    assert cpu.audit.parameter_bytes == 33_562_624
    assert cpu.audit.buffer_bytes == 2_097_152
    assert cpu.audit.parameter_devices == ("cpu",)
    assert cpu.audit.output_device == "cpu"
    assert len(cpu.audit.alias_checks) == 16
    assert all(check.aliases_input_storage for check in cpu.audit.alias_checks)
    assert all(
        check.input_storage_identity == check.output_storage_identity
        for check in cpu.audit.alias_checks
    )
    q_transpose = next(
        check
        for check in cpu.audit.alias_checks
        if check.stable_path.endswith("/layer_0/attention/q_transpose")
    )
    assert q_transpose.input_contract.shape == (1, 512, 8, 64)
    assert q_transpose.input_contract.stride == (262144, 512, 64, 1)
    assert q_transpose.output_contract.shape == (1, 8, 512, 64)
    assert q_transpose.output_contract.stride == (262144, 64, 512, 1)
    assert q_transpose.output_contract.layout == "strided"
    semantic_paths = {
        operation.stable_path
        for operation in SemanticCompiler().compile(_request()).semantic_ir.walk_operations()
    }
    assert set(dict(cpu.audit.leaf_output_devices)) == semantic_paths


def test_cpu_reference_is_deterministic() -> None:
    runner = _runner()

    first = runner.run_device("cpu")
    second = runner.run_device("cpu")

    assert first.output_sha256 == second.output_sha256
    torch.testing.assert_close(first.output, second.output, rtol=0, atol=0)


def test_context_matmul_avoids_pathological_query_broadcast_decomposition() -> None:
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        generator = torch.Generator(device="cpu").manual_seed(20260807)
        probabilities = torch.randn(
            (1, 8, 512, 512), dtype=torch.float32, generator=generator
        ).softmax(dim=-1)
        values = torch.randn(
            (1, 512, 8, 64), dtype=torch.float32, generator=generator
        ).transpose(1, 2)
        operation = TensorMatMul("test/context", "bhqk,bhkd->bqhd")

        with torch.inference_mode():
            expected = torch.einsum("bhqk,bhkd->bqhd", probabilities, values)
            actual = operation(probabilities, values)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            for _ in range(3):
                operation(probabilities, values)
                torch.einsum("bhqk,bhkd->bqhd", probabilities, values)
            operation_ns = _best_runtime_ns(
                lambda: operation(probabilities, values)
            )
            reference_ns = _best_runtime_ns(
                lambda: torch.einsum(
                    "bhqk,bhkd->bqhd", probabilities, values
                )
            )

        assert operation_ns <= reference_ns * 8, (
            "context MatMul must retain batched-GEMM granularity; "
            f"implementation={operation_ns} ns, einsum reference={reference_ns} ns"
        )
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS correctness runs only in the trusted local Mac lane",
)
def test_mps_matches_cpu_without_fallback_or_cpu_leaf_outputs() -> None:
    runner = _runner()
    report = runner.compare_cpu_mps(atol=1e-4, rtol=1e-3)

    assert report.passed
    assert report.max_absolute_error <= 1e-4
    assert report.max_relative_error <= 1e-3
    assert report.mps.audit.input_device == "mps:0"
    assert report.mps.audit.output_device == "mps:0"
    assert report.mps.audit.parameter_devices == ("mps:0",)
    assert report.mps.audit.buffer_devices == ("mps:0",)
    assert set(dict(report.mps.audit.leaf_output_devices).values()) == {"mps:0"}
    assert report.mps.audit.fallback_enabled is False
    assert all(check.aliases_input_storage for check in report.mps.audit.alias_checks)
