from __future__ import annotations

from pathlib import Path
import platform

import pytest
import yaml

from groundupscale.benchmark.hardware_microbenchmark import (
    CapabilityAggregationError,
    HardwareMicrobenchmarkRunner,
    aggregate_capability_envelope,
)
from groundupscale.schemas.v1alpha1 import (
    HardwareBenchmarkSuiteDocument,
    HardwareCapabilityProfileDocument,
)
from groundupscale.cli import main
from groundupscale.specs import SpecRepository, SpecValidationError


def _probe(probe_id: str, rates: list[float]) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for shape_index, rate in enumerate(rates):
        shape = [shape_index + 1]
        cases.extend(
            [
                {
                    "shape": shape,
                    "threads": 1,
                    "eligible": True,
                    "achieved_rate": rate / 2,
                },
                {
                    "shape": shape,
                    "threads": 4,
                    "eligible": True,
                    "achieved_rate": rate,
                },
            ]
        )
    return {
        "probe_id": probe_id,
        "resource": "compute.fp32",
        "unit": "FLOP/s",
        "cases": cases,
    }


def test_diverse_probes_promote_a_hardware_resource_envelope() -> None:
    observation = {
        "schema": "groundupscale.dev/hardware-microbenchmark-observation/v1alpha1",
        "suite": {"name": "synthetic-m4", "version": "0.1.0"},
        "target": {"hardware": "apple-m4", "device": "cpu"},
        "hardware_cohort": "synthetic-m4-cpu",
        "environment": {"eligible": True},
        "probes": [
            _probe("vector-fma", [float(value) for value in range(1, 11)]),
            _probe("matrix-fma", [float(value) for value in range(11, 21)]),
        ],
    }

    profile = aggregate_capability_envelope(
        observation,
        profile_name="synthetic-m4-capability",
        profile_version="0.1.0",
        source_path="raw/observation.json",
        source_sha256="a" * 64,
    )

    resource = profile["spec"]["resources"][0]
    assert resource["resource"] == "compute.fp32"
    assert resource["unit"] == "FLOP/s"
    assert resource["robust_achievable_rate"] == pytest.approx(18.2)
    assert resource["optimistic_rate"] == pytest.approx(19.55)
    assert resource["selected_robust_probe"] == "matrix-fma"
    assert resource["aggregation"] == "max(probe_shape_p80)"
    assert resource["probe_envelopes"][0]["distinct_shape_count"] == 10
    assert resource["probe_envelopes"][1]["distinct_shape_count"] == 10
    assert profile["spec"]["source"]["sha256"] == "a" * 64
    assert isinstance(
        HardwareCapabilityProfileDocument.model_validate(profile),
        HardwareCapabilityProfileDocument,
    )


def test_capability_envelope_rejects_fewer_than_ten_distinct_shapes() -> None:
    observation = {
        "schema": "groundupscale.dev/hardware-microbenchmark-observation/v1alpha1",
        "suite": {"name": "synthetic-m4", "version": "0.1.0"},
        "target": {"hardware": "apple-m4", "device": "cpu"},
        "hardware_cohort": "synthetic-m4-cpu",
        "environment": {"eligible": True},
        "probes": [_probe("matrix-fma", [float(value) for value in range(1, 10)])],
    }

    with pytest.raises(CapabilityAggregationError, match="at least 10"):
        aggregate_capability_envelope(
            observation,
            profile_name="invalid",
            profile_version="0.1.0",
            source_path="raw/observation.json",
            source_sha256="b" * 64,
        )


def _suite_document(shape_count: int = 10) -> dict[str, object]:
    return {
        "apiVersion": "groundupscale.dev/v1alpha1",
        "kind": "HardwareBenchmarkSuite",
        "metadata": {"name": "m4-cpu", "version": "0.1.0"},
        "spec": {
            "target": {"hardware": "apple-m4", "device": "cpu"},
            "warmup_iterations": 2,
            "samples": 5,
            "target_window_ms": 10.0,
            "maximum_inner_iterations": 1000,
            "probes": [
                {
                    "id": "matrix-fma",
                    "kind": "matrix_multiply",
                    "resource": "compute.fp32",
                    "dtype": "float32",
                    "shapes": [[value, value, value] for value in range(1, shape_count + 1)],
                    "alignment_boundaries": [5],
                    "thread_counts": [1, 4],
                }
            ],
        },
    }


def test_hardware_benchmark_suite_is_a_strict_yaml_spec(tmp_path: Path) -> None:
    document = _suite_document()
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    loaded = SpecRepository(tmp_path).load_document(path)

    assert isinstance(loaded, HardwareBenchmarkSuiteDocument)
    assert loaded.spec.probes[0].kind == "matrix_multiply"
    assert len(loaded.spec.probes[0].shapes) == 10


def test_hardware_benchmark_suite_rejects_insufficient_shape_coverage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "suite.yaml"
    path.write_text(
        yaml.safe_dump(_suite_document(shape_count=9), sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(SpecValidationError, match="at least 10"):
        SpecRepository(tmp_path).load_document(path)


def test_hardware_benchmark_suite_requires_both_sides_of_alignment_boundaries(
    tmp_path: Path,
) -> None:
    document = _suite_document()
    document["spec"]["probes"][0]["alignment_boundaries"] = [512]  # type: ignore[index]
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(SpecValidationError, match="511/512/513"):
        SpecRepository(tmp_path).load_document(path)


def test_hardware_microbenchmark_runner_emits_rate_evidence_for_every_shape() -> None:
    document = _suite_document()
    probe = document["spec"]["probes"][0]  # type: ignore[index]
    probe.update(  # type: ignore[union-attr]
        {
            "id": "memory-copy",
            "kind": "memory_copy",
            "resource": "memory.shared",
            "shapes": [[value] for value in range(64, 74)],
            "alignment_boundaries": [65],
            "thread_counts": [1],
        }
    )
    document["spec"]["target_window_ms"] = 0.01  # type: ignore[index]
    suite = HardwareBenchmarkSuiteDocument.model_validate(document)

    observation = HardwareMicrobenchmarkRunner(
        suite,
        environment={
            "eligible": True,
            "captured_at": "2026-08-07T00:00:00Z",
            "policy": {"policy_id": "test"},
            "reason_codes": [],
        },
    ).run()

    assert observation["schema"] == (
        "groundupscale.dev/hardware-microbenchmark-observation/v1alpha1"
    )
    assert observation["target"] == {"hardware": "apple-m4", "device": "cpu"}
    assert len(observation["probes"]) == 1
    cases = observation["probes"][0]["cases"]
    assert len(cases) == 10
    assert all(case["work"] == 2 * case["shape"][0] * 4 for case in cases)
    assert all(case["achieved_rate"] > 0 for case in cases)
    assert all(len(case["samples_ns"]) == 5 for case in cases)


@pytest.mark.skipif(platform.machine() != "arm64", reason="ARM64 native scalar probe")
def test_scalar_probe_uses_a_native_fma_kernel_instead_of_framework_dispatch() -> None:
    document = _suite_document()
    probe = document["spec"]["probes"][0]  # type: ignore[index]
    probe.update(  # type: ignore[union-attr]
        {
            "id": "scalar-fma",
            "kind": "scalar_fma",
            "resource": "compute.fp32",
            "shapes": [[value] for value in range(64, 74)],
            "alignment_boundaries": [65],
            "thread_counts": [1],
        }
    )
    document["spec"]["target_window_ms"] = 0.01  # type: ignore[index]
    suite = HardwareBenchmarkSuiteDocument.model_validate(document)

    observation = HardwareMicrobenchmarkRunner(
        suite,
        environment={"eligible": True, "policy": {"policy_id": "test"}},
    ).run()

    cases = observation["probes"][0]["cases"]
    assert all(case["implementation"] == "native-arm64-scalar-fma" for case in cases)
    assert all(case["work"] == 16 * case["shape"][0] for case in cases)
    assert all(case["achieved_rate"] > 0 for case in cases)


def test_hardware_benchmark_cli_writes_raw_observation_and_yaml_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _suite_document()
    probe = document["spec"]["probes"][0]  # type: ignore[index]
    probe.update(  # type: ignore[union-attr]
        {
            "id": "memory-copy",
            "kind": "memory_copy",
            "resource": "memory.shared",
            "shapes": [[value] for value in range(65536, 65546)],
            "alignment_boundaries": [65537],
            "thread_counts": [1],
        }
    )
    document["spec"]["target_window_ms"] = 2.0  # type: ignore[index]
    document["spec"]["samples"] = 9  # type: ignore[index]
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    observation_path = tmp_path / "observation.json"
    profile_path = tmp_path / "profile.yaml"

    def stable_case(
        _runner: HardwareMicrobenchmarkRunner,
        _probe_spec: object,
        shape: tuple[int, ...],
        threads: int,
        _case_index: int,
    ) -> dict[str, object]:
        return {
            "shape": list(shape),
            "threads": threads,
            "work": shape[0] * 8,
            "unit": "B/s",
            "implementation": "deterministic-test-double",
            "inner_iterations": 1,
            "samples_ns": [1000.0] * 9,
            "median_ns": 1000.0,
            "q1_ns": 1000.0,
            "q3_ns": 1000.0,
            "iqr_ns": 0.0,
            "iqr_over_median": 0.0,
            "achieved_rate": float(shape[0] * 8_000_000),
            "eligible": True,
            "eligibility": {
                "maximum_iqr_over_median": 0.10,
                "reason": None,
            },
            "assumptions": ["deterministic CLI wiring test"],
        }

    monkeypatch.setattr(HardwareMicrobenchmarkRunner, "_run_case", stable_case)

    exit_code = main(
        [
            "benchmark-hardware",
            str(suite_path),
            "--repository-root",
            str(tmp_path),
            "--observation-output",
            str(observation_path),
            "--profile-output",
            str(profile_path),
            "--profile-name",
            "test-m4-capability",
            "--json",
        ],
        environment_collector=lambda **_: {
            "eligible": True,
            "captured_at": "2026-08-07T00:00:00Z",
            "policy": {"policy_id": "test"},
            "reason_codes": [],
        },
    )

    assert exit_code == 0
    assert observation_path.is_file()
    profile = SpecRepository(tmp_path).load_document(profile_path)
    assert isinstance(profile, HardwareCapabilityProfileDocument)
    summary = __import__("json").loads(capsys.readouterr().out)
    assert summary["resource_count"] == 1
    assert summary["environment_eligible"] is True


def test_hardware_benchmark_cli_collects_row_reduction_phase_capability(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document = _suite_document()
    probe = document["spec"]["probes"][0]  # type: ignore[index]
    probe.update(  # type: ignore[union-attr]
        {
            "id": "row-reduction-max-fp32",
            "kind": "reduction_max",
            "resource": "compute.reduction.max.fp32",
            "shapes": [[2, width] for width in range(4, 16)],
            "alignment_boundaries": [5],
            "thread_counts": [1],
        }
    )
    document["spec"]["target_window_ms"] = 0.1  # type: ignore[index]
    document["spec"]["samples"] = 4  # type: ignore[index]
    suite_path = tmp_path / "phase-suite.yaml"
    suite_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    observation_path = tmp_path / "phase-observation.json"
    profile_path = tmp_path / "phase-profile.yaml"

    exit_code = main(
        [
            "benchmark-hardware",
            str(suite_path),
            "--repository-root",
            str(tmp_path),
            "--observation-output",
            str(observation_path),
            "--profile-output",
            str(profile_path),
            "--profile-name",
            "phase-profile",
            "--json",
        ],
        environment_collector=lambda **_: {
            "eligible": True,
            "captured_at": "2026-08-09T00:00:00Z",
            "policy": {"policy_id": "test"},
            "reason_codes": [],
        },
    )

    assert exit_code == 0
    summary = yaml.safe_load(capsys.readouterr().out)
    assert summary["resource_count"] == 1
    observation = yaml.safe_load(observation_path.read_text(encoding="utf-8"))
    cases = observation["probes"][0]["cases"]
    assert observation["probes"][0]["resource"] == (
        "compute.reduction.max.fp32"
    )
    assert all(case["unit"] == "FLOP/s" for case in cases)
    assert [case["work"] for case in cases] == [
        2 * (width - 1) for width in range(4, 16)
    ]


def test_hardware_benchmark_cli_collects_complete_compound_phase_resources(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document = _suite_document()
    # Keep every probe above the timer-noise floor while retaining the
    # boundary-1/boundary/boundary+1 coverage required by the public schema.
    row_shapes = [[512, width] for width in range(255, 267)]
    vector_shapes = [[elements] for elements in range(65535, 65547)]
    contracts = (
        ("reduction_sum", "compute.reduction.sum.fp32", row_shapes, 256),
        (
            "elementwise_subtract",
            "compute.elementwise.subtract.fp32",
            row_shapes,
            256,
        ),
        (
            "elementwise_divide",
            "compute.elementwise.divide.fp32",
            row_shapes,
            256,
        ),
        (
            "elementwise_exp",
            "compute.transcendental.exp.fp32",
            vector_shapes,
            65536,
        ),
        (
            "elementwise_square",
            "compute.elementwise.square.fp32",
            vector_shapes,
            65536,
        ),
        ("scalar_divide", "compute.scalar.divide.fp32", vector_shapes, 65536),
        ("scalar_add", "compute.scalar.add.fp32", vector_shapes, 65536),
        ("scalar_rsqrt", "compute.transcendental.rsqrt.fp32", vector_shapes, 65536),
        (
            "elementwise_multiply",
            "compute.elementwise.multiply.fp32",
            vector_shapes,
            65536,
        ),
        (
            "memory_row_reduction",
            "memory.row-reduction.fp32",
            row_shapes,
            256,
        ),
        (
            "memory_broadcast",
            "memory.broadcast-read-write.fp32",
            row_shapes,
            256,
        ),
        (
            "memory_elementwise",
            "memory.elementwise-read-write.fp32",
            vector_shapes,
            65536,
        ),
        (
            "memory_row_scalar",
            "memory.row-scalar-read-write.fp32",
            vector_shapes,
            65536,
        ),
    )
    document["spec"]["probes"] = [  # type: ignore[index]
        {
            "id": f"phase-{kind}",
            "kind": kind,
            "resource": resource,
            "dtype": "float32",
            "shapes": shapes,
            "alignment_boundaries": [boundary],
            "thread_counts": [1],
        }
        for kind, resource, shapes, boundary in contracts
    ]
    document["spec"]["target_window_ms"] = 0.25  # type: ignore[index]
    document["spec"]["samples"] = 5  # type: ignore[index]
    suite_path = tmp_path / "all-phase-suite.yaml"
    suite_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    observation_path = tmp_path / "all-phase-observation.json"
    profile_path = tmp_path / "all-phase-profile.yaml"

    exit_code = main(
        [
            "benchmark-hardware",
            str(suite_path),
            "--repository-root",
            str(tmp_path),
            "--observation-output",
            str(observation_path),
            "--profile-output",
            str(profile_path),
            "--profile-name",
            "all-phase-profile",
            "--json",
        ],
        environment_collector=lambda **_: {
            "eligible": True,
            "captured_at": "2026-08-09T00:00:00Z",
            "policy": {"policy_id": "test"},
            "reason_codes": [],
        },
    )

    assert exit_code == 0
    summary = yaml.safe_load(capsys.readouterr().out)
    assert summary["resource_count"] == len(contracts)
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    resources = {item["resource"]: item for item in profile["spec"]["resources"]}
    assert set(resources) == {resource for _, resource, _, _ in contracts}
    assert all(
        item["unit"] == ("FLOP/s" if resource.startswith("compute.") else "B/s")
        for resource, item in resources.items()
    )
