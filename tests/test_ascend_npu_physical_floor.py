from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import shutil

import pytest
import yaml

import groundupscale.cli as cli_module
from groundupscale.benchmark.ascend_hardware_microbenchmark import (
    AscendNpuHardwareMicrobenchmarkRunner,
)
from groundupscale.benchmark.hardware_microbenchmark import (
    aggregate_capability_envelope,
)
from groundupscale.schemas.v1alpha1 import (
    AnalysisPlanDocument,
    DeploymentIntentDocument,
    FabricGraphDocument,
    HardwareCapabilityProfileDocument,
    HardwareBenchmarkSuiteDocument,
    HardwareSpecDocument,
)
from groundupscale.pipeline import compile_analysis_plan
from groundupscale.physical_floor_bundle import PhysicalFloorComparisonBundleWriter
from groundupscale.run_bundle import verify_run_bundle
from groundupscale.specs import SpecRepository, SpecValidationError


REPOSITORY_ROOT = Path(__file__).parents[1]
ASCEND_PLAN = REPOSITORY_ROOT / "specs/plans/ascend-npu-prefill.yaml"
ASCEND_MEASUREMENT_BUNDLE = REPOSITORY_ROOT / (
    "goal_process/issue-28-npu-measurement-adapter/evidence/runs/"
    "ascend-910b2-exact-shape-512-20260810-v1"
)


def _copy_ascend_analysis_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    shutil.copytree(REPOSITORY_ROOT / "specs", repository / "specs")
    evidence = repository / "goal_process/issue-29-ascend-physical-floor/evidence"
    evidence.mkdir(parents=True)
    source_evidence = REPOSITORY_ROOT / (
        "goal_process/issue-29-ascend-physical-floor/evidence"
    )
    for name in (
        "ascend-910b2-resource-observation-20260810-v2.json",
        "ascend-910b2-hardware-cohort-20260810-v2.json",
    ):
        shutil.copy2(source_evidence / name, evidence / name)
    return repository


def _refresh_bundle_artifact_digest(bundle: Path, role: str) -> None:
    manifest_path = bundle / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(
        item for item in manifest["artifacts"] if item["role"] == role
    )
    artifact["sha256"] = sha256((bundle / artifact["path"]).read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_ascend_analysis_plan_keeps_hardware_topology_and_deployment_separate() -> None:
    repository = SpecRepository(REPOSITORY_ROOT)
    plan = repository.load_document(ASCEND_PLAN)
    hardware = repository.load_document("specs/hardware/ascend-910b2.yaml")
    fabric = repository.load_document("specs/fabrics/local-ascend-910b2.yaml")
    deployment = repository.load_document("specs/deployment-intents/ascend-npu.yaml")

    assert isinstance(plan, AnalysisPlanDocument)
    assert isinstance(hardware, HardwareSpecDocument)
    assert isinstance(fabric, FabricGraphDocument)
    assert isinstance(deployment, DeploymentIntentDocument)
    assert hardware.metadata.name == "ascend-910b2"
    assert hardware.spec.devices[0].kind == "npu"
    assert fabric.metadata.name == "local-ascend-910b2"
    assert deployment.metadata.name == "ascend-npu"
    assert deployment.spec.bindings[0].placement == (
        "local-ascend-910b2/npu-0"
    )
    assert plan.spec.hardware_capability_profiles[0].path == (
        "specs/hardware-capabilities/ascend-910b2-npu-local.yaml"
    )


def test_analysis_plan_rejects_tampered_hardware_cohort_evidence(
    tmp_path: Path,
) -> None:
    repository = _copy_ascend_analysis_repository(tmp_path)
    cohort_path = repository / (
        "goal_process/issue-29-ascend-physical-floor/evidence/"
        "ascend-910b2-hardware-cohort-20260810-v2.json"
    )
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort["device_name"] = "tampered"
    cohort_path.write_text(
        json.dumps(cohort, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(SpecValidationError, match="cohort evidence sha256"):
        SpecRepository(repository).load_analysis_plan(
            repository / "specs/plans/ascend-npu-prefill.yaml"
        )


def _resource_probe(
    probe_id: str, resource: str, unit: str, base_rate: float
) -> dict[str, object]:
    return {
        "probe_id": probe_id,
        "probe_kind": "matrix_multiply" if unit == "FLOP/s" else "memory_copy",
        "resource": resource,
        "dtype": "float32",
        "unit": unit,
        "cases": [
            {
                "shape": [index + 1],
                "threads": 1,
                "eligible": True,
                "achieved_rate": base_rate + index,
            }
            for index in range(10)
        ],
    }


def test_capability_profile_preserves_cohort_domain_uncertainty_and_quality() -> None:
    observation = {
        "schema": "groundupscale.dev/hardware-microbenchmark-observation/v1alpha1",
        "suite": {"name": "ascend-910b2-resource-envelope", "version": "0.1.0"},
        "target": {"hardware": "ascend-910b2", "device": "npu-0"},
        "hardware_cohort": "ascend-npu-test-cohort",
        "cohort_evidence": {
            "path": "evidence/cohort.json",
            "sha256": "b" * 64,
            "schema": "groundupscale.dev/hardware-cohort/v1alpha1",
        },
        "environment": {"eligible": True},
        "validity_domain": {
            "operation_classes": ["MatMul", "memory_copy"],
            "dtype": "float32",
            "layout": "row-major-contiguous",
            "logical_device": "npu:0",
            "execution_mode": "pytorch-eager",
            "shape_support": "observed-stratified-shapes-only",
        },
        "uncertainty": {
            "method": "per-shape-median-cross-shape-quantiles",
            "robust_quantile": 0.8,
            "optimistic_quantile": 0.95,
            "maximum_iqr_over_median": 0.1,
        },
        "quality": {
            "status": "qualified",
            "reason_codes": [],
            "eligible_shape_count_by_resource": {
                "compute.fp32": 10,
                "memory.hbm": 10,
            },
        },
        "probes": [
            _resource_probe("matrix-fp32", "compute.fp32", "FLOP/s", 100.0),
            _resource_probe("hbm-copy", "memory.hbm", "B/s", 200.0),
        ],
    }

    profile_data = aggregate_capability_envelope(
        observation,
        profile_name="ascend-910b2-npu-local",
        profile_version="0.1.0",
        source_path="evidence/resource-observation.json",
        source_sha256="a" * 64,
    )
    profile = HardwareCapabilityProfileDocument.model_validate(profile_data)

    assert profile.spec.source.sha256 == "a" * 64
    assert profile.spec.cohort_evidence.sha256 == "b" * 64
    assert profile.spec.validity_domain.operation_classes == (
        "MatMul",
        "memory_copy",
    )
    assert profile.spec.uncertainty.robust_quantile == pytest.approx(0.8)
    assert profile.spec.quality.status == "qualified"
    resources = {item.resource: item for item in profile.spec.resources}
    assert resources["compute.fp32"].robust_achievable_rate == pytest.approx(107.2)
    assert resources["memory.hbm"].optimistic_rate == pytest.approx(208.55)


def _ascend_suite_document() -> HardwareBenchmarkSuiteDocument:
    return HardwareBenchmarkSuiteDocument.model_validate(
        {
            "apiVersion": "groundupscale.dev/v1alpha1",
            "kind": "HardwareBenchmarkSuite",
            "metadata": {
                "name": "ascend-910b2-resource-envelope",
                "version": "0.1.0",
            },
            "spec": {
                "target": {"hardware": "ascend-910b2", "device": "npu-0"},
                "warmup_iterations": 2,
                "samples": 5,
                "target_window_ms": 1.0,
                "maximum_inner_iterations": 10,
                "probes": [
                    {
                        "id": "matrix-fp32",
                        "kind": "matrix_multiply",
                        "resource": "compute.fp32",
                        "dtype": "float32",
                        "shapes": [
                            [value, value, value] for value in range(1, 11)
                        ],
                        "alignment_boundaries": [5],
                        "thread_counts": [1],
                    },
                    {
                        "id": "hbm-copy",
                        "kind": "memory_copy",
                        "resource": "memory.hbm",
                        "dtype": "float32",
                        "shapes": [[value] for value in range(1, 11)],
                        "alignment_boundaries": [5],
                        "thread_counts": [1],
                    },
                ],
            },
        }
    )


def test_ascend_runner_emits_replayable_multi_shape_resource_observation() -> None:
    def stable_case(
        probe: object,
        shape: tuple[int, ...],
        concurrency: int,
        case_index: int,
    ) -> dict[str, object]:
        unit = "FLOP/s" if getattr(probe, "kind") == "matrix_multiply" else "B/s"
        work = 1000 + case_index
        return {
            "shape": list(shape),
            "threads": concurrency,
            "work": work,
            "unit": unit,
            "implementation": "deterministic-ascend-test-double",
            "inner_iterations": 1,
            "samples_ns": [1000.0] * 5,
            "median_ns": 1000.0,
            "q1_ns": 1000.0,
            "q3_ns": 1000.0,
            "iqr_ns": 0.0,
            "iqr_over_median": 0.0,
            "achieved_rate": float(work * 1_000_000),
            "eligible": True,
            "eligibility": {"maximum_iqr_over_median": 0.1, "reason": None},
            "correctness": {"status": "passed"},
            "timer": {
                "source": "torch.npu.Event.elapsed_time",
                "resolution_ns": 20.0,
            },
            "assumptions": ["deterministic test"],
        }

    observation = AscendNpuHardwareMicrobenchmarkRunner(
        _ascend_suite_document(),
        environment={"eligible": True, "reason_codes": []},
        cohort={
            "cohort_id": "ascend-npu-test-cohort",
            "power_clock": {"power_policy": "fixed"},
        },
        cohort_evidence={
            "path": "evidence/cohort.json",
            "sha256": "c" * 64,
            "schema": "groundupscale.dev/hardware-cohort/v1alpha1",
        },
        case_executor=stable_case,
        software={"torch": "2.7.1", "torch_npu": "2.7.1"},
    ).run()

    assert observation["hardware_cohort"] == "ascend-npu-test-cohort"
    assert observation["cohort_evidence"]["sha256"] == "c" * 64
    assert observation["quality"]["status"] == "qualified"
    assert observation["quality"]["eligible_shape_count_by_resource"] == {
        "compute.fp32": 10,
        "memory.hbm": 10,
    }
    assert observation["validity_domain"] == {
        "operation_classes": ["MatMul", "memory_copy"],
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "logical_device": "npu:0",
        "execution_mode": "pytorch-eager",
        "shape_support": "observed-stratified-shapes-only",
    }
    assert len(observation["probes"]) == 2
    assert all(len(probe["cases"]) == 10 for probe in observation["probes"])


def test_benchmark_hardware_cli_selects_the_lazy_ascend_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _ascend_suite_document()
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(suite.model_dump(mode="json", by_alias=True), sort_keys=False),
        encoding="utf-8",
    )
    observation_path = tmp_path / "resource-observation.json"
    cohort_path = tmp_path / "cohort.json"
    profile_path = tmp_path / "profile.yaml"

    class FakeAdapter:
        def fingerprint_cohort(self) -> dict[str, object]:
            return {
                "schema": "groundupscale.dev/hardware-cohort/v1alpha1",
                "status": "completed",
                "cohort_id": "ascend-npu-cli-test",
                "power_clock": {"power_policy": "fixed"},
            }

        def preflight(self) -> dict[str, object]:
            return {
                "schema": "groundupscale.dev/measurement-preflight/v1alpha1",
                "eligible": True,
                "reason_codes": [],
            }

    class FakeRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.cohort_evidence = kwargs["cohort_evidence"]

        def run(self) -> dict[str, object]:
            return {
                "schema": (
                    "groundupscale.dev/hardware-microbenchmark-observation/v1alpha1"
                ),
                "suite": {
                    "name": "ascend-910b2-resource-envelope",
                    "version": "0.1.0",
                },
                "target": {"hardware": "ascend-910b2", "device": "npu-0"},
                "hardware_cohort": "ascend-npu-cli-test",
                "cohort_evidence": self.cohort_evidence,
                "environment": {"eligible": True},
                "validity_domain": {
                    "operation_classes": ["MatMul", "memory_copy"],
                    "dtype": "float32",
                    "layout": "row-major-contiguous",
                    "logical_device": "npu:0",
                    "execution_mode": "pytorch-eager",
                    "shape_support": "observed-stratified-shapes-only",
                },
                "uncertainty": {
                    "method": "per-shape-median-cross-shape-quantiles",
                    "robust_quantile": 0.8,
                    "optimistic_quantile": 0.95,
                    "maximum_iqr_over_median": 0.1,
                },
                "quality": {
                    "status": "qualified",
                    "reason_codes": [],
                    "eligible_shape_count_by_resource": {
                        "compute.fp32": 10,
                        "memory.hbm": 10,
                    },
                },
                "probes": [
                    _resource_probe(
                        "matrix-fp32", "compute.fp32", "FLOP/s", 100.0
                    ),
                    _resource_probe("hbm-copy", "memory.hbm", "B/s", 200.0),
                ],
            }

    monkeypatch.setattr(
        cli_module, "create_measurement_adapter", lambda *_args, **_kwargs: FakeAdapter()
    )
    monkeypatch.setattr(
        cli_module, "AscendNpuHardwareMicrobenchmarkRunner", FakeRunner
    )

    exit_code = cli_module.main(
        [
            "benchmark-hardware",
            str(suite_path),
            "--repository-root",
            str(tmp_path),
            "--observation-output",
            str(observation_path),
            "--cohort-output",
            str(cohort_path),
            "--profile-output",
            str(profile_path),
            "--profile-name",
            "ascend-910b2-npu-local",
            "--json",
        ]
    )

    assert exit_code == 0
    assert cohort_path.is_file()
    profile = HardwareCapabilityProfileDocument.model_validate(
        yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    )
    assert profile.spec.hardware_cohort == "ascend-npu-cli-test"
    assert profile.spec.cohort_evidence is not None
    assert profile.spec.cohort_evidence.sha256 == sha256(
        cohort_path.read_bytes()
    ).hexdigest()


def test_ascend_backend_maps_only_supported_matmul_regions_to_physical_floors() -> None:
    compiled = compile_analysis_plan(REPOSITORY_ROOT, ASCEND_PLAN)

    prediction = compiled.hardware_prediction
    assert prediction is not None
    assert prediction.backend_id == "huawei.ascend.910b2.resource-envelope"
    assert prediction.placement == "local-ascend-910b2/npu-0"
    assert prediction.status == "partial-empirical-hardware-floor-exploratory"
    assert prediction.prediction_complete is False
    assert prediction.capabilities.architecture == "ascend"
    assert prediction.capabilities.fp32_flops_per_second.status == "unknown"
    assert prediction.capabilities.peak_memory_bandwidth_bytes_per_second.status == (
        "unknown"
    )

    resources = {item.resource: item for item in prediction.measured_capabilities}
    assert resources["compute.fp32"].robust_achievable_rate == pytest.approx(
        19_175_995_339_398.254
    )
    assert resources["memory.hbm"].robust_achievable_rate == pytest.approx(
        1_408_047_205_172.7576
    )
    assert resources["compute.fp32"].quality_status == "exploratory"
    assert resources["compute.fp32"].quality_reason_codes == (
        "power-policy-unobserved",
    )

    assert prediction.candidates
    assert all(candidate.operation == "MatMul" for candidate in prediction.candidates)
    q_proj = next(
        candidate
        for candidate in prediction.candidates
        if candidate.stable_path.endswith("/attention/q_proj")
    )
    assert q_proj.flops == 268_435_456
    assert q_proj.compulsory_bytes == 3_145_728
    assert q_proj.duration.empirical_compute_time_ns == pytest.approx(
        13_998.514878963015
    )
    assert q_proj.duration.empirical_memory_time_ns == pytest.approx(
        2_234.1069166172174
    )
    assert q_proj.duration.empirical_hardware_floor_ns == pytest.approx(
        13_998.514878963015
    )
    assert q_proj.duration.full_duration_ns is None
    assert q_proj.duration.status == "empirical-hardware-floor-exploratory"

    assert prediction.unsupported_regions
    assert any(
        region.operation == "RMSNorm"
        and region.reason == "unsupported-ascend-cost-operation"
        for region in prediction.unsupported_regions
    )
    assert any(
        region.stable_path.endswith("/attention/qk_matmul")
        and region.reason == "outside-capability-validity-domain"
        for region in prediction.unsupported_regions
    )
    assert any(
        region.stable_path.endswith("/mlp/gate_proj")
        and region.reason == "outside-capability-observed-shapes"
        for region in prediction.unsupported_regions
    )
    q_scope = next(
        bound
        for bound in prediction.scope_bounds
        if bound.case_id == "ascend-matmul-q-proj"
    )
    assert q_scope.empirical_hardware_floor_ns == pytest.approx(
        q_proj.duration.empirical_hardware_floor_ns
    )
    assert prediction.program_bounds.empirical_hardware_floor_ns is None
    assert prediction.program_bounds.full_duration_ns is None


@pytest.mark.parametrize(
    "profile_mutation",
    ("environment-ineligible", "quality-quarantined", "operation-domain-mismatch"),
)
def test_ascend_backend_rejects_profile_policy_tampering_without_raw_evidence(
    tmp_path: Path,
    profile_mutation: str,
) -> None:
    repository = _copy_ascend_analysis_repository(tmp_path)
    profile_path = repository / (
        "specs/hardware-capabilities/ascend-910b2-npu-local.yaml"
    )
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if profile_mutation == "environment-ineligible":
        profile["spec"]["environment"]["eligible"] = False
    elif profile_mutation == "quality-quarantined":
        profile["spec"]["quality"]["status"] = "quarantined"
    else:
        profile["spec"]["validity_domain"]["operation_classes"] = [
            "memory_copy"
        ]
    profile_path.write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(
        SpecValidationError,
        match="derived capability profile does not match raw observation",
    ):
        compile_analysis_plan(
            repository, repository / "specs/plans/ascend-npu-prefill.yaml"
        )


def test_comparison_bundle_replays_stable_path_floor_quality_and_observation(
    tmp_path: Path,
) -> None:
    compiled = compile_analysis_plan(REPOSITORY_ROOT, ASCEND_PLAN)
    run = PhysicalFloorComparisonBundleWriter(compiled).run(
        tmp_path,
        measurement_bundle=ASCEND_MEASUREMENT_BUNDLE,
        run_id="ascend-matmul-floor-comparison",
    )

    verification = verify_run_bundle(run)
    assert verification["passed"], verification["failures"]
    manifest = __import__("json").loads(
        (run / "run.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["bundle_kind"] == "physical-floor-observation-comparison"
    assert manifest["hardware_cohort"] == "ascend-npu-23b93a89d5fecc79"
    assert any(
        artifact["role"] == "source-candidate-identity"
        for artifact in manifest["artifacts"]
    )
    comparison = __import__("json").loads(
        (run / "comparison/physical-floor-vs-observation.json").read_text(
            encoding="utf-8"
        )
    )
    assert comparison["stable_path"] == (
        "model/two-layer-transformer/transformer/layer_0/attention/q_proj"
    )
    assert comparison["physical_floor"]["minimum_work_flops"] == 268_435_456
    assert comparison["physical_floor"]["compulsory_bytes"] == 3_145_728
    assert comparison["physical_floor"][
        "resource_physical_floor_ns"
    ] == pytest.approx(13_998.514878963015)
    assert comparison["physical_floor"]["full_duration_ns"] is None
    assert comparison["theoretical_capability"]["fp32_flops_per_second"][
        "status"
    ] == "unknown"
    assert comparison["operator_frontier"] == {
        "status": "unknown",
        "value_ns": None,
        "reason_code": "not-qualified-by-issue-29",
    }
    assert comparison["physical_floor"]["quality"] == {
        "status": "exploratory",
        "reason_codes": ["power-policy-unobserved"],
    }
    assert comparison["observation"]["median_ns"] == pytest.approx(82_810.0)
    assert comparison["observation"]["completion_boundary"] == "closed"
    assert comparison["observation"]["quality"] == "valid"
    assert comparison["comparison"]["observed_to_physical_floor_ratio"] == (
        pytest.approx(5.915627515246598)
    )
    assert comparison["comparison"]["relative_prediction_error"] is None
    assert comparison["comparison"]["error_status"] == (
        "not-evaluable-physical-floor-is-not-a-duration-prediction"
    )
    assert comparison["unsupported_regions"]["count"] > 0

    explanation = __import__("json").loads(
        (run / "prediction/explanation.graph.json").read_text(encoding="utf-8")
    )
    assert comparison["stable_path"] in explanation["entrypoints"]
    floor_node = next(
        node
        for node in explanation["nodes"]
        if node["kind"] == "resource-physical-floor"
    )
    observation_node = next(
        node for node in explanation["nodes"] if node["kind"] == "observation"
    )
    assert floor_node["assumptions"] == comparison["physical_floor"]["assumptions"]
    assert floor_node["hardware_cohort"] == manifest["hardware_cohort"]
    assert floor_node["capabilities"] == comparison["physical_floor"]["capabilities"]
    assert observation_node["hardware_cohort"] == manifest["hardware_cohort"]
    report = (run / "reports/report.html").read_text(encoding="utf-8")
    assert "Resource Physical Floor" in report
    assert "完整实现 duration：unknown" in report
    assert "Observation" in report
    assert "power-policy-unobserved" in report
    assert "ascend-npu-23b93a89d5fecc79" in report
    assert "FLOPs are CostIR minimum mathematical work" in report

    comparison["hardware_cohort"] = "tampered-cohort"
    comparison_path = run / "comparison/physical-floor-vs-observation.json"
    comparison_path.write_text(
        __import__("json").dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison_artifact = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "physical-floor-observation-comparison"
    )
    comparison_artifact["sha256"] = sha256(comparison_path.read_bytes()).hexdigest()
    (run / "run.manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered = verify_run_bundle(run)
    assert tampered["passed"] is False
    assert "comparison hardware cohort mismatch" in tampered["failures"]


def test_comparison_rejects_same_flops_from_a_different_matmul_shape(
    tmp_path: Path,
) -> None:
    measurement = tmp_path / "same-flops-different-shape"
    shutil.copytree(ASCEND_MEASUREMENT_BUNDLE, measurement)
    case_path = measurement / "resolved/case.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    case["shape"] = {"left": [256, 512], "right": [512, 1024]}
    case_path.write_text(
        json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_bundle_artifact_digest(measurement, "benchmark-case")
    assert verify_run_bundle(measurement)["passed"]

    compiled = compile_analysis_plan(REPOSITORY_ROOT, ASCEND_PLAN)
    with pytest.raises(ValueError, match="exact Shape"):
        PhysicalFloorComparisonBundleWriter(compiled).run(
            tmp_path,
            measurement_bundle=measurement,
            run_id="reject-same-flops-different-shape",
        )


def test_comparison_verifier_cross_checks_copied_source_artifact_digests(
    tmp_path: Path,
) -> None:
    compiled = compile_analysis_plan(REPOSITORY_ROOT, ASCEND_PLAN)
    run = PhysicalFloorComparisonBundleWriter(compiled).run(
        tmp_path,
        measurement_bundle=ASCEND_MEASUREMENT_BUNDLE,
        run_id="cross-check-source-digests",
    )
    raw_path = run / "source/raw-timing.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["timer_source"] = "tampered-but-internally-digested"
    raw_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_bundle_artifact_digest(run, "source-raw-timing-observation")

    verification = verify_run_bundle(run)
    assert verification["passed"] is False
    assert "source artifact digest mismatch: raw-timing-observation" in (
        verification["failures"]
    )


@pytest.mark.parametrize(
    ("tamper", "expected_failure"),
    (
        ("floor", "physical floor derivation mismatch"),
        ("headroom", "comparison headroom derivation mismatch"),
        ("candidate", "comparison source candidate mismatch"),
        ("explanation", "Explanation Graph derivation mismatch"),
        ("report", "HTML report derivation mismatch"),
    ),
)
def test_comparison_verifier_recomputes_every_derived_projection(
    tmp_path: Path,
    tamper: str,
    expected_failure: str,
) -> None:
    compiled = compile_analysis_plan(REPOSITORY_ROOT, ASCEND_PLAN)
    run = PhysicalFloorComparisonBundleWriter(compiled).run(
        tmp_path,
        measurement_bundle=ASCEND_MEASUREMENT_BUNDLE,
        run_id=f"reject-tampered-{tamper}",
    )
    if tamper in {"floor", "headroom", "candidate"}:
        path = run / "comparison/physical-floor-vs-observation.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        if tamper == "floor":
            document["physical_floor"]["resource_physical_floor_ns"] += 1
        elif tamper == "headroom":
            document["comparison"]["observed_to_physical_floor_ratio"] = 99.0
        else:
            document["observation"]["candidate"] = "tampered-candidate"
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _refresh_bundle_artifact_digest(
            run, "physical-floor-observation-comparison"
        )
    elif tamper == "explanation":
        path = run / "prediction/explanation.graph.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        floor_node = next(
            node
            for node in document["nodes"]
            if node["id"] == "metric:resource-physical-floor"
        )
        floor_node["value_ns"] += 1
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _refresh_bundle_artifact_digest(run, "explanation-graph")
    else:
        path = run / "reports/report.html"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "优化 headroom", "tampered headroom"
            ),
            encoding="utf-8",
        )
        _refresh_bundle_artifact_digest(run, "html-report")

    verification = verify_run_bundle(run)
    assert verification["passed"] is False
    assert expected_failure in verification["failures"]


def test_compare_measurement_and_explain_are_public_cli_seams(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli_module.main(
        [
            "compare-measurement",
            str(ASCEND_PLAN),
            str(ASCEND_MEASUREMENT_BUNDLE),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--artifact-store",
            str(tmp_path),
            "--run-id",
            "ascend-cli-comparison",
            "--json",
        ]
    )

    assert exit_code == 0
    summary = __import__("json").loads(capsys.readouterr().out)
    run = Path(summary["run_bundle"])
    assert summary["status"] == "completed"
    assert summary["stable_path"].endswith("/attention/q_proj")
    assert summary["full_duration_ns"] is None
    assert verify_run_bundle(run)["passed"]

    explain_exit = cli_module.main(["explain", str(run), "--json"])

    assert explain_exit == 0
    explained = __import__("json").loads(capsys.readouterr().out)
    assert explained["bundle_kind"] == "physical-floor-observation-comparison"
    assert explained["resource_physical_floor_ns"] == pytest.approx(
        13_998.514878963015
    )
    assert explained["observation_median_ns"] == pytest.approx(82_810.0)
    assert explained["full_duration_ns"] is None
