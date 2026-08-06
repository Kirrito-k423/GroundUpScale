from __future__ import annotations

from pathlib import Path

import torch

from groundupscale.benchmark import ReferenceRunner
from groundupscale.compiler import SemanticCompiler
from groundupscale.specs import SpecRepository

from test_semantic_compiler import _request


REPOSITORY_ROOT = Path(__file__).parents[1]


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
