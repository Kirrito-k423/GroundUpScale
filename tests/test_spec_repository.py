from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
import yaml

from groundupscale.specs import SpecRepository, SpecValidationError


API_VERSION = "groundupscale.dev/v1alpha1"
VERSION = "0.1.0"
REPOSITORY_ROOT = Path(__file__).parents[1]


def _document(kind: str, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "apiVersion": API_VERSION,
        "kind": kind,
        "metadata": {"name": name, "version": VERSION},
        "spec": spec,
    }


def _reference(path: str) -> dict[str, str]:
    return {"path": path, "version": VERSION}


def _write_yaml(root: Path, relative_path: str, document: dict[str, Any]) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _minimal_bundle(root: Path) -> tuple[Path, dict[str, Any]]:
    tensor = {"dtype": "float32", "shape": ["B", "S", "H"], "layout": "contiguous"}
    model = _document(
        "ModelSpec",
        "tiny-transformer",
        {
            "symbols": {
                "B": {"type": "integer", "minimum": 1},
                "S": {"type": "integer", "minimum": 1},
                "H": {"type": "integer", "minimum": 1},
            },
            "constraints": [],
            "root": {
                "id": "transformer",
                "kind": "composite",
                "children": [],
                "entrypoints": [
                    {
                        "name": "prefill",
                        "inputs": [{"name": "hidden", "tensor": tensor}],
                        "outputs": [{"name": "hidden", "tensor": tensor}],
                        "steps": [],
                    }
                ],
            },
        },
    )
    workload = _document(
        "WorkloadSpec",
        "prefill",
        {
            "artifacts": [
                {"name": "hidden-input", "tensor": tensor},
                {"name": "hidden-output", "tensor": tensor},
            ],
            "root": {
                "id": "request",
                "kind": "sequence",
                "children": [
                    {
                        "id": "model-prefill",
                        "kind": "model_call",
                        "model": _reference("models/tiny.yaml"),
                        "entrypoint": "prefill",
                        "inputs": {"hidden": "hidden-input"},
                        "outputs": {"hidden": "hidden-output"},
                    }
                ],
            },
        },
    )
    analysis_case = _document(
        "AnalysisCase",
        "fixed-shape",
        {
            "shape": {
                "kind": "fixed",
                "bindings": {"B": 1, "S": 512, "H": 512},
                "dtype": "float32",
            },
            "driver": {
                "kind": "fixed_iterations",
                "warmup_iterations": 5,
                "measured_iterations": 20,
            },
            "observation_window": {"kind": "iterations", "value": 20},
        },
    )
    deployment = _document(
        "DeploymentIntent",
        "local-cpu",
        {
            "bindings": [
                {"scope": "workload/request/model-prefill", "placement": "m4/cpu"}
            ]
        },
    )
    hardware = _document(
        "HardwareSpec",
        "apple-m4",
        {
            "devices": [
                {
                    "id": "cpu",
                    "kind": "cpu",
                    "vendor": "Apple",
                    "model": "M4",
                    "compute_units": 10,
                    "memory_bytes": 17179869184,
                }
            ]
        },
    )
    fabric = _document(
        "FabricGraph",
        "local-m4",
        {
            "nodes": [{"id": "m4/cpu", "hardware": "apple-m4", "device": "cpu"}],
            "links": [],
        },
    )
    benchmark = _document(
        "BenchmarkCase",
        "tiny-e2e",
        {
            "cases": [
                {
                    "id": "e2e-prefill",
                    "scope": "workload/request/model-prefill",
                    "mode": "e2e",
                    "warmup_iterations": 5,
                    "samples": 20,
                }
            ]
        },
    )

    documents = {
        "models/tiny.yaml": model,
        "workloads/prefill.yaml": workload,
        "analysis/fixed.yaml": analysis_case,
        "deployments/cpu.yaml": deployment,
        "hardware/m4.yaml": hardware,
        "fabrics/local.yaml": fabric,
        "benchmarks/e2e.yaml": benchmark,
    }
    for relative_path, document in documents.items():
        _write_yaml(root, relative_path, document)

    plan = _document(
        "AnalysisPlan",
        "tiny-cpu-prefill",
        {
            "workload": _reference("workloads/prefill.yaml"),
            "analysis_case": _reference("analysis/fixed.yaml"),
            "deployment_intent": _reference("deployments/cpu.yaml"),
            "hardware": [_reference("hardware/m4.yaml")],
            "fabric_graph": _reference("fabrics/local.yaml"),
            "benchmark_cases": [_reference("benchmarks/e2e.yaml")],
        },
    )
    return _write_yaml(root, "plans/plan.yaml", plan), plan


def test_analysis_plan_resolves_all_versioned_yaml_documents(tmp_path: Path) -> None:
    plan_path, _ = _minimal_bundle(tmp_path)

    bundle = SpecRepository(tmp_path).load_analysis_plan(plan_path)

    assert bundle.plan.metadata.name == "tiny-cpu-prefill"
    assert bundle.workload.metadata.name == "prefill"
    assert bundle.analysis_case.metadata.name == "fixed-shape"
    assert bundle.deployment_intent.metadata.name == "local-cpu"
    assert [document.metadata.name for document in bundle.hardware] == ["apple-m4"]
    assert bundle.fabric_graph.metadata.name == "local-m4"
    assert [document.metadata.name for document in bundle.benchmark_cases] == ["tiny-e2e"]
    assert list(bundle.models) == ["tiny-transformer"]
    assert len(bundle.sources) == 8
    assert all(len(source.sha256) == 64 for source in bundle.sources.values())


def test_m4_cpu_hardware_spec_exposes_official_limits_without_inventing_flops() -> None:
    bundle = SpecRepository(REPOSITORY_ROOT).load_analysis_plan(
        REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )

    cpu = next(
        device
        for hardware in bundle.hardware
        for device in hardware.spec.devices
        if device.id == "cpu"
    )
    capabilities = cpu.capabilities

    assert capabilities is not None
    assert [(pool.kind, pool.count) for pool in capabilities.core_pools] == [
        ("performance", 4),
        ("efficiency", 6),
    ]
    assert capabilities.vector.register_bits == 128
    assert capabilities.vector.fp32_fma_flops_per_instruction == 8
    assert capabilities.unified_memory.peak_bandwidth_bytes_per_second == (
        120_000_000_000
    )
    assert capabilities.unified_memory.scope == "soc_shared"
    assert capabilities.theoretical_compute.fp32_flops_per_second.value is None
    assert capabilities.theoretical_compute.fp32_flops_per_second.status == "unknown"
    assert capabilities.theoretical_compute.fp32_flops_per_second.reason == (
        "vendor_does_not_publish_frequency_or_fma_issue_rate"
    )


def test_m4_plan_loads_measured_capabilities_without_overwriting_vendor_facts() -> None:
    bundle = SpecRepository(REPOSITORY_ROOT).load_analysis_plan(
        REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )

    assert len(bundle.hardware_capability_profiles) == 1
    profile = bundle.hardware_capability_profiles[0]
    resources = {item.resource: item for item in profile.spec.resources}
    assert resources["compute.fp32"].robust_achievable_rate > 0
    assert resources["memory.shared"].robust_achievable_rate > 0
    cpu = bundle.hardware[0].spec.devices[0]
    assert cpu.capabilities is not None
    assert cpu.capabilities.theoretical_compute.fp32_flops_per_second.value is None
    assert (
        cpu.capabilities.unified_memory.peak_bandwidth_bytes_per_second
        == 120_000_000_000
    )


def test_capability_profile_must_be_exactly_rederived_from_its_raw_observation(
    tmp_path: Path,
) -> None:
    plan_path, plan = _minimal_bundle(tmp_path)
    profile = yaml.safe_load(
        (REPOSITORY_ROOT / "specs/hardware-capabilities/apple-m4-cpu-local.yaml")
        .read_text(encoding="utf-8")
    )
    source = profile["spec"]["source"]
    raw_bytes = (REPOSITORY_ROOT / source["path"]).read_bytes()
    raw_path = tmp_path / "evidence/raw-observation.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw_bytes)
    source["path"] = "evidence/raw-observation.json"
    source["sha256"] = sha256(raw_bytes).hexdigest()
    profile_path = _write_yaml(tmp_path, "capabilities/profile.yaml", profile)
    plan["spec"]["hardware_capability_profiles"] = [
        {
            "path": "capabilities/profile.yaml",
            "version": profile["metadata"]["version"],
        }
    ]
    _write_yaml(tmp_path, "plans/plan.yaml", plan)

    assert SpecRepository(tmp_path).load_analysis_plan(plan_path)

    tampered = deepcopy(profile)
    tampered["spec"]["environment"]["eligible"] = not profile["spec"][
        "environment"
    ]["eligible"]
    profile_path.write_text(
        yaml.safe_dump(tampered, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(
        SpecValidationError,
        match="derived capability profile does not match raw observation",
    ):
        SpecRepository(tmp_path).load_analysis_plan(plan_path)


def test_m4_measured_memory_p80_is_within_ten_percent_of_comparable_vendor_peak() -> None:
    bundle = SpecRepository(REPOSITORY_ROOT).load_analysis_plan(
        REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )
    cpu = bundle.hardware[0].spec.devices[0]
    assert cpu.capabilities is not None
    vendor_rate = cpu.capabilities.unified_memory.peak_bandwidth_bytes_per_second
    resource = next(
        item
        for item in bundle.hardware_capability_profiles[0].spec.resources
        if item.resource == "memory.shared"
    )

    relative_delta = abs(resource.robust_achievable_rate - vendor_rate) / vendor_rate

    assert relative_delta == pytest.approx(0.05693822840947904)
    assert relative_delta <= 0.10


def test_unknown_fields_are_rejected_with_a_precise_location(tmp_path: Path) -> None:
    _, plan = _minimal_bundle(tmp_path)
    invalid = deepcopy(plan)
    invalid["spec"]["unexpected"] = True
    invalid_path = _write_yaml(tmp_path, "plans/invalid.yaml", invalid)

    with pytest.raises(SpecValidationError, match=r"spec\.unexpected"):
        SpecRepository(tmp_path).load_document(invalid_path)


def test_reference_cannot_escape_repository_root(tmp_path: Path) -> None:
    plan_path, plan = _minimal_bundle(tmp_path)
    invalid = deepcopy(plan)
    invalid["spec"]["workload"] = _reference("../outside.yaml")
    _write_yaml(tmp_path.parent, "outside.yaml", _document("WorkloadSpec", "bad", {}))
    _write_yaml(tmp_path, "plans/escape.yaml", invalid)

    with pytest.raises(SpecValidationError, match="escapes repository root"):
        SpecRepository(tmp_path).load_analysis_plan(tmp_path / "plans/escape.yaml")

    assert plan_path.exists()


def test_reference_version_and_kind_are_verified(tmp_path: Path) -> None:
    _, plan = _minimal_bundle(tmp_path)
    wrong_version = deepcopy(plan)
    wrong_version["spec"]["workload"]["version"] = "9.9.9"
    wrong_version_path = _write_yaml(tmp_path, "plans/wrong-version.yaml", wrong_version)
    with pytest.raises(SpecValidationError, match="expected version 9.9.9"):
        SpecRepository(tmp_path).load_analysis_plan(wrong_version_path)

    wrong_kind = deepcopy(plan)
    wrong_kind["spec"]["workload"] = _reference("models/tiny.yaml")
    wrong_kind_path = _write_yaml(tmp_path, "plans/wrong-kind.yaml", wrong_kind)
    with pytest.raises(SpecValidationError, match="expected WorkloadSpec, found ModelSpec"):
        SpecRepository(tmp_path).load_analysis_plan(wrong_kind_path)


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    _, plan = _minimal_bundle(tmp_path)
    content = yaml.safe_dump(plan, sort_keys=False)
    content = content.replace("kind: AnalysisPlan\n", "kind: AnalysisPlan\nkind: AnalysisPlan\n")
    duplicate_path = tmp_path / "plans/duplicate.yaml"
    duplicate_path.write_text(content, encoding="utf-8")

    with pytest.raises(SpecValidationError, match="duplicate YAML key 'kind'"):
        SpecRepository(tmp_path).load_document(duplicate_path)
