from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

import pytest
import yaml
from test_ascend_npu_measurement_adapter import (
    _available_runtime,
    _complete_system_probe,
    _raw_hardware_collection,
)

from groundupscale.diagnostics import diagnose_run_bundle, render_diagnostic_report
from groundupscale.ir import content_fingerprint
from groundupscale.measurement_adapters.ascend_npu import (
    AscendNpuMeasurementAdapter,
)
from groundupscale.measurement_run import MeasurementRunBundleWriter
from groundupscale.operator_frontier import (
    OperatorFrontierBundleWriter,
    OperatorFrontierQualificationError,
)
from groundupscale.run_bundle import verify_run_bundle


def _rewrite_environment_session(
    run: Path,
    *,
    process_id: int,
) -> None:
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(
        item
        for item in manifest["artifacts"]
        if item["role"] == "environment"
    )
    path = run / artifact["path"]
    environment = json.loads(path.read_text(encoding="utf-8"))
    environment["measurement_session"] = {
        "session_id": manifest["run_id"],
        "process_id": process_id,
        "process_started_at": f"2026-08-11T00:00:{process_id:02d}+00:00",
        "python_executable": "/trusted/python",
        "source": "python-process-identity",
    }
    path.write_text(
        json.dumps(environment, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    artifact["sha256"] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _measurement_run(
    root: Path,
    *,
    run_id: str,
    size: int,
    candidate: str,
    median_ns: int,
    process_id: int,
    correctness: str = "passed",
    foreign_cohort: bool = False,
    warmup_iterations: int = 20,
) -> Path:
    case = {
        "schema": "groundupscale.dev/exact-shape-matmul-case/v1alpha1",
        "operation": "MatMul",
        "shape": {"left": [size, size], "right": [size, size]},
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "seed": 20260811,
        "candidate": candidate,
        "warmup_iterations": warmup_iterations,
        "repetitions": 5,
    }
    raw = _raw_hardware_collection(None, 0, case, {})
    raw["left_sha256"] = sha256(f"left-{size}".encode()).hexdigest()
    raw["right_sha256"] = sha256(f"right-{size}".encode()).hexdigest()
    raw["target_output_sha256"] = sha256(
        f"output-{size}-{candidate}".encode()
    ).hexdigest()
    raw["minimum_alignment_bytes"] = 64
    raw["raw_samples_ns"] = [
        median_ns - 200,
        median_ns - 100,
        median_ns,
        median_ns + 100,
        median_ns + 200,
    ]
    raw["correctness"] = {
        **raw["correctness"],
        "status": correctness,
        "max_absolute_error": 0.0002 if correctness == "passed" else 1.0,
        "max_relative_error": 0.0004 if correctness == "passed" else 1.0,
    }
    def system_probe(logical_device_index: int) -> dict[str, object]:
        probe = _complete_system_probe(logical_device_index)
        if foreign_cohort:
            probe["hardware"]["vdie_id"] = "foreign-vdie-id"
        return probe

    adapter = AscendNpuMeasurementAdapter(
        runtime_loader=_available_runtime,
        collection_executor=lambda *args: raw,
        system_probe=system_probe,
    )
    run = MeasurementRunBundleWriter(adapter).run(
        root,
        case=case,
        run_id=run_id,
    )
    _rewrite_environment_session(run, process_id=process_id)
    assert verify_run_bundle(run)["passed"] is True
    return run


def _fixed_nk_measurement_run(
    root: Path,
    *,
    run_id: str,
    m: int,
    median_ns: int,
    process_id: int,
    n: int = 512,
    k: int = 512,
) -> Path:
    case = {
        "schema": "groundupscale.dev/exact-shape-matmul-case/v1alpha1",
        "operation": "MatMul",
        "shape": {"left": [m, k], "right": [k, n]},
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "seed": 20260812,
        "candidate": "torch.matmul",
        "warmup_iterations": 20,
        "repetitions": 5,
    }
    raw = _raw_hardware_collection(None, 0, case, {})
    raw["left_sha256"] = sha256(f"left-{m}-{k}".encode()).hexdigest()
    raw["right_sha256"] = sha256(f"right-{k}-{n}".encode()).hexdigest()
    raw["target_output_sha256"] = sha256(f"output-{m}-{n}".encode()).hexdigest()
    raw["minimum_alignment_bytes"] = 64
    raw["raw_samples_ns"] = [
        median_ns - 2,
        median_ns - 1,
        median_ns,
        median_ns + 1,
        median_ns + 2,
    ]
    adapter = AscendNpuMeasurementAdapter(
        runtime_loader=_available_runtime,
        collection_executor=lambda *args: raw,
        system_probe=_complete_system_probe,
    )
    run = MeasurementRunBundleWriter(adapter).run(root, case=case, run_id=run_id)
    _rewrite_environment_session(run, process_id=process_id)
    assert verify_run_bundle(run)["passed"] is True
    return run


def _fixed_nk_inputs(tmp_path: Path) -> tuple[list[Path], list[Path], list[Path]]:
    measurements = tmp_path / "fixed-nk-measurements"
    search: list[Path] = []
    holdout: list[Path] = []
    confirmation: list[Path] = []
    process_id = 100

    def expected_latency_ns(m: int) -> int:
        declared_work = 2 * m * 512 * 512
        return round(30_000 + declared_work / 524.288)

    for m in (128, 512):
        for lane, target in (("search", search), ("holdout", holdout)):
            for session in range(3):
                target.append(
                    _fixed_nk_measurement_run(
                        measurements,
                        run_id=f"{lane}-m{m}-{session}",
                        m=m,
                        median_ns=expected_latency_ns(m),
                        process_id=process_id,
                    )
                )
                process_id += 1
    for session in range(3):
        confirmation.append(
            _fixed_nk_measurement_run(
                measurements,
                run_id=f"confirmation-m320-{session}",
                m=320,
                median_ns=expected_latency_ns(320),
                process_id=process_id,
            )
        )
        process_id += 1
    return search, holdout, confirmation


def _fixed_nk_policy() -> dict[str, object]:
    policy = _qualification_policy()
    policy["policy_id"] = "test-fixed-nk-latency-response"
    policy["version"] = "v2"
    policy["scope"] = {
        "hardware_cohort": "ascend-npu-febd831c8d07e06f",
        "operation": "MatMul",
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "fixed_n": 512,
        "fixed_k": 512,
        "anchor_m": [128, 512],
        "confirmation_m": 320,
        "candidate_ids": ["torch.matmul"],
    }
    policy["response_target"] = "latency"
    policy["response_kind"] = "setup-plus-throughput"
    policy["response_version"] = "v1"
    policy["maximum_relative_error"] = 0.01
    policy["maximum_setup_fraction_for_steady"] = 0.10
    policy["change_reason"] = "deterministic ticket-35 synthetic fixture"
    policy["evidence_ref"] = "test://ticket-35/fixed-nk-policy-v2"
    return policy


def _rewrite_candidate_identity(run: Path) -> None:
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def mutate(candidate: dict[str, object]) -> None:
        candidate["build_identity"]["operator_entrypoint"] = (
            "changed-runtime-entrypoint"
        )
        candidate.pop("candidate_digest")
        candidate["candidate_digest"] = content_fingerprint(candidate)

    for role in ("candidate-identity", "measurement-collection"):
        artifact = next(
            item for item in manifest["artifacts"] if item["role"] == role
        )
        path = run / artifact["path"]
        document = json.loads(path.read_text(encoding="utf-8"))
        mutate(
            document
            if role == "candidate-identity"
            else document["candidate_identity"]
        )
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        artifact["sha256"] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _rewrite_execution_contract(run: Path) -> None:
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for role in ("execution-contract", "measurement-collection"):
        artifact = next(
            item for item in manifest["artifacts"] if item["role"] == role
        )
        path = run / artifact["path"]
        document = json.loads(path.read_text(encoding="utf-8"))
        contract = (
            document
            if role == "execution-contract"
            else document["execution_contract"]
        )
        contract["dispatch_contract"] = "changed-dispatch-contract"
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        artifact["sha256"] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _rewrite_bundle_artifact(
    run: Path,
    role: str,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(
        item for item in manifest["artifacts"] if item["role"] == role
    )
    path = run / artifact["path"]
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    artifact["sha256"] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _frontier_inputs(
    tmp_path: Path,
    *,
    split_correctness: str = "failed",
    include_split_holdout: bool = False,
    warmup_iterations: int = 20,
) -> tuple[list[Path], list[Path], list[Path]]:
    measurements = tmp_path / "measurements"
    search: list[Path] = []
    holdout: list[Path] = []
    confirmation: list[Path] = []
    process_id = 10
    for size, direct_ns, split_ns in (
        (256, 10_000, 14_000),
        (512, 60_000, 80_000),
    ):
        for session in range(3):
            search.append(
                _measurement_run(
                    measurements,
                    run_id=f"search-{size}-direct-{session}",
                    size=size,
                    candidate="torch.matmul",
                    median_ns=direct_ns + session * 50,
                    process_id=process_id,
                    warmup_iterations=warmup_iterations,
                )
            )
            process_id += 1
            search.append(
                _measurement_run(
                    measurements,
                    run_id=f"search-{size}-split-{session}",
                    size=size,
                    candidate="torch.matmul.k-split-2",
                    median_ns=split_ns + session * 50,
                    process_id=process_id,
                    correctness=split_correctness,
                    warmup_iterations=warmup_iterations,
                )
            )
            process_id += 1
        for session, offset in enumerate((-100, 0, 100)):
            holdout.append(
                _measurement_run(
                    measurements,
                    run_id=f"holdout-{size}-direct-{session}",
                    size=size,
                    candidate="torch.matmul",
                    median_ns=direct_ns + offset,
                    process_id=process_id,
                    warmup_iterations=warmup_iterations,
                )
            )
            process_id += 1
            if include_split_holdout:
                holdout.append(
                    _measurement_run(
                        measurements,
                        run_id=f"holdout-{size}-split-{session}",
                        size=size,
                        candidate="torch.matmul.k-split-2",
                        median_ns=split_ns + offset,
                        process_id=process_id,
                        warmup_iterations=warmup_iterations,
                    )
                )
                process_id += 1
    for session, offset in enumerate((-100, 0, 100)):
        confirmation.append(
            _measurement_run(
                measurements,
                run_id=f"confirm-384-direct-{session}",
                size=384,
                candidate="torch.matmul",
                median_ns=29_000 + offset,
                process_id=process_id,
                warmup_iterations=warmup_iterations,
            )
        )
        process_id += 1
    return search, holdout, confirmation


def _qualification_policy(
    *,
    minimum_candidate_coverage: str = "C0_SINGLE",
    minimum_warmup_iterations: int = 20,
) -> dict[str, object]:
    return {
        "schema": (
            "groundupscale.dev/operator-frontier-qualification-policy/"
            "v1alpha1"
        ),
        "policy_id": "test-ascend-matmul-frontier",
        "version": "v1",
        "scope": {
            "hardware_cohort": "ascend-npu-febd831c8d07e06f",
            "operation": "MatMul",
            "dtype": "float32",
            "layout": "row-major-contiguous",
            "anchor_shapes": [256, 512],
            "confirmation_shape": 384,
            "candidate_ids": [
                "torch.matmul",
                "torch.matmul.k-split-2",
            ],
        },
        "minimum_search_sessions": 3,
        "minimum_holdout_sessions": 3,
        "minimum_confirmation_sessions": 3,
        "minimum_warmup_iterations": minimum_warmup_iterations,
        "maximum_session_median_relative_range": 0.10,
        "minimum_candidate_coverage": minimum_candidate_coverage,
        "holdout_candidate_scope": "all-eligible-candidates",
        "uncertainty_combination": "root-sum-of-squares",
        "target_coverage": 0.68,
        "sample_exclusion": "none-preserve-all-raw-samples",
        "estimator": "median(independent-holdout-session-medians)",
        "change_reason": "deterministic ticket-31 test fixture",
        "revalidation": (
            "on cohort, candidate, execution contract, policy, anchor, "
            "or confirmation change"
        ),
        "evidence_ref": "test://ticket-31/qualification-policy-v1",
    }


def test_qualifies_best_correct_candidates_and_queries_minimal_surface(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _frontier_inputs(tmp_path)

    run = OperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="ascend-matmul-frontier-v1",
        qualification_policy=_qualification_policy(),
        search_runs=search,
        holdout_runs=holdout,
        confirmation_runs=confirmation,
        query_sizes=(512, 384, 640),
    )

    assert verify_run_bundle(run)["passed"] is True
    qualification = json.loads(
        (run / "frontier/qualification.json").read_text(encoding="utf-8")
    )
    assert qualification["status"] == "qualified"
    assert qualification["candidate_coverage"] == {
        "attempted_level": "C2_MULTI_FAMILY",
        "eligible_level": "C0_SINGLE",
        "attempted_candidate_ids": [
            "torch.matmul",
            "torch.matmul.k-split-2",
        ],
        "eligible_candidate_ids": ["torch.matmul"],
        "selected_candidate_id": "torch.matmul",
        "selected_candidate_family": "pytorch-ascend-matmul",
    }
    excluded = {
        item["candidate_id"]: item
        for item in qualification["candidate_records"]
        if item["status"] == "excluded"
    }
    assert excluded["torch.matmul.k-split-2"]["reason_codes"] == [
        "candidate-correctness-failed"
    ]
    direct_256 = next(
        item
        for item in qualification["candidate_records"]
        if item["shape"] == {"s": 256}
        and item["candidate_id"] == "torch.matmul"
    )
    identity = direct_256["candidate_identity"]
    assert identity["candidate_family"] == "pytorch-ascend-matmul"
    assert identity["build_identity"]["framework_version"] == "2.7.1"
    assert identity["runtime_identity"]["candidate_device"] == "npu:0"
    assert identity["execution_mode"] == "pytorch-eager"
    assert identity["dtype"] == "float32"
    assert identity["layout"] == "row-major-contiguous"
    assert identity["minimum_alignment_bytes"] == 64
    assert identity["shape"] == {
        "left": [256, 256],
        "right": [256, 256],
    }
    assert identity["compilation_parameters"]
    assert identity["tuning_parameters"] == {}
    assert len(direct_256["candidate_evidence_digests"]) == 1
    assert len(direct_256["candidate_evidence_digests"][0]) == 64
    assert [anchor["shape"] for anchor in qualification["anchors"]] == [
        {"s": 256},
        {"s": 512},
    ]
    assert all(
        anchor["observation_validity"] == "QUALIFIED"
        and anchor["frontier_role"] == "ACTIVE"
        and len(anchor["search_run_ids"]) == 3
        and len(anchor["holdout_run_ids"]) == 3
        and not set(anchor["search_run_ids"])
        & set(anchor["holdout_run_ids"])
        for anchor in qualification["anchors"]
    )

    result = diagnose_run_bundle(run)
    queries = {
        item["query_shape"]["s"]: item
        for item in result["capability_surface_queries"]
    }
    assert queries[512]["status"] == "exact_anchor"
    assert queries[384]["status"] == "interpolated"
    assert queries[384]["cell_id"].startswith(
        "ascend-matmul-square-256-512-"
    )
    assert queries[640]["status"] == "unknown"
    assert queries[640]["reason_code"] == "outside_validated_domain"

def test_fixed_nk_queries_fail_closed_for_invalid_m_and_mismatched_domain(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _fixed_nk_inputs(tmp_path)
    run = OperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="fixed-nk-invalid-query-v1",
        qualification_policy=_fixed_nk_policy(),
        search_runs=search,
        holdout_runs=holdout,
        confirmation_runs=confirmation,
        query_sizes=(0, -1, 320, 320),
    )
    diagnostic_path = run / "diagnostic/evidence.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    diagnostic["surface_queries"][2]["domain"]["fixed_n"] = 256
    diagnostic["surface_queries"][3]["shape"]["m"] = 320.5
    evidence = {
        key: value
        for key, value in diagnostic.items()
        if key
        not in {
            "schema",
            "resolved_configuration",
            "resolved_ir",
            "hardware",
            "cohort_id",
            "execution_domain",
            "digests",
        }
    }
    diagnostic["digests"]["evidence_sha256"] = sha256(
        json.dumps(
            evidence,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    diagnostic_path.write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostic_artifact = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "diagnostic-evidence"
    )
    diagnostic_artifact["sha256"] = sha256(diagnostic_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    queries = diagnose_run_bundle(run)["capability_surface_queries"]

    assert [query["status"] for query in queries] == ["unknown"] * 4
    assert [query["reason_code"] for query in queries] == [
        "invalid_query_shape",
        "invalid_query_shape",
        "fixed_nk_domain_mismatch",
        "invalid_query_shape",
    ]
    assert all(query["latency"] is None for query in queries)
    assert all(query["effective_rate"] is None for query in queries)


def test_fixed_nk_qualification_rejects_anchor_holdout_outside_error_budget(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _fixed_nk_inputs(tmp_path)
    holdout[:3] = [
        _fixed_nk_measurement_run(
            tmp_path / "fixed-nk-measurements",
            run_id=f"holdout-m128-outside-budget-{session}",
            m=128,
            median_ns=161_000,
            process_id=999 + session,
        )
        for session in range(3)
    ]

    with pytest.raises(OperatorFrontierQualificationError) as captured:
        OperatorFrontierBundleWriter().run(
            tmp_path / "frontier",
            run_id="must-reject-anchor-outside-error-budget",
            qualification_policy=_fixed_nk_policy(),
            search_runs=search,
            holdout_runs=holdout,
            confirmation_runs=confirmation,
            query_sizes=(128, 320),
        )

    assert captured.value.reason_code == "latency-response-error-budget-failed"


def test_fixed_nk_synthetic_evidence_qualifies_latency_response_surface(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _fixed_nk_inputs(tmp_path)

    run = OperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="fixed-nk-latency-response-v1",
        qualification_policy=_fixed_nk_policy(),
        search_runs=search,
        holdout_runs=holdout,
        confirmation_runs=confirmation,
        query_sizes=(128, 320, 640),
    )

    assert verify_run_bundle(run)["passed"] is True
    qualification = json.loads(
        (run / "frontier/qualification.json").read_text(encoding="utf-8")
    )
    surface = qualification["surface"]
    response = surface["cells"][0]["response"]
    assert surface["coordinate"] == {
        "axis": "m",
        "transform": "identity",
        "transform_version": "v1",
    }
    assert surface["work_formula"] == {
        "kind": "matmul-2mnk-fixed-nk",
        "version": "v1",
        "fixed_n": 512,
        "fixed_k": 512,
        "work_unit": "FLOP",
    }
    assert response["target"] == "latency"
    assert response["kind"] == "setup-plus-throughput"
    assert response["version"] == "v1"
    assert response["setup_latency_ns"] == pytest.approx(30_000.0)
    assert response["asymptotic_rate"] == pytest.approx(524_288_000_000.0)
    assert response["shape_regime"]["identity"]
    assert response["shape_regime"]["classification"] == "ramp"
    assert response["fit_evidence_refs"]
    assert response["holdout_evidence_refs"]

    queries = {
        item["query_shape"]["m"]: item
        for item in diagnose_run_bundle(run)["capability_surface_queries"]
    }
    exact = queries[128]
    assert exact["status"] == "exact_anchor"
    assert exact["latency"]["value_ns"] == pytest.approx(158_000.0)
    assert exact["effective_rate"]["value"] == pytest.approx(
        (2 * 128 * 512 * 512) / (158_000.0e-9)
    )
    interior = queries[320]
    assert interior["status"] == "modeled"
    assert interior["latency"]["value_ns"] == pytest.approx(350_000.0)
    assert interior["effective_rate"]["value"] == pytest.approx(
        (2 * 320 * 512 * 512) / (interior["latency"]["value_ns"] * 1e-9)
    )
    assert set(interior["uncertainty"]["components"]) == {
        "anchor_standard_latency_ns",
        "response_model_standard_latency_ns",
        "instrumentation_standard_latency_ns",
        "boundary_standard_latency_ns",
    }
    assert queries[640]["status"] == "unknown"
    assert queries[640]["reason_code"] == "outside_validated_domain"

    report = render_diagnostic_report(diagnose_run_bundle(run))
    assert "Setup Latency=30000.000000 ns" in report
    assert "asymptotic-rate=0.524288000 TFLOP/s" in report
    assert f"Shape Regime={response['shape_regime']['identity']}/ramp" in report
    assert '"response_model_standard_latency_ns"' in report

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    machine = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(run),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    human = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(run),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert machine.returncode == 0, machine.stderr
    assert json.loads(machine.stdout) == diagnose_run_bundle(run)
    assert human.returncode == 0, human.stderr
    assert "Setup Latency=30000.000000 ns" in human.stdout
    assert "asymptotic-rate=0.524288000 TFLOP/s" in human.stdout


def test_qualification_requires_an_explicit_versioned_policy(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _frontier_inputs(tmp_path)

    with pytest.raises(OperatorFrontierQualificationError) as captured:
        OperatorFrontierBundleWriter().run(
            tmp_path / "frontier",
            run_id="must-reject-missing-policy",
            qualification_policy=None,
            search_runs=search,
            holdout_runs=holdout,
            confirmation_runs=confirmation,
            query_sizes=(384,),
        )

    assert captured.value.reason_code == "missing-qualification-policy"


def test_qualification_rejects_unsafe_run_id_before_reading_evidence(
    tmp_path: Path,
) -> None:
    artifact_store = tmp_path / "frontier"

    with pytest.raises(ValueError, match="unsafe run_id"):
        OperatorFrontierBundleWriter().run(
            artifact_store,
            run_id="../escaped",
            qualification_policy=_qualification_policy(),
            search_runs=(tmp_path / "missing-search",),
            holdout_runs=(tmp_path / "missing-holdout",),
            confirmation_runs=(tmp_path / "missing-confirmation",),
            query_sizes=(384,),
        )

    assert not artifact_store.exists()
    assert not (tmp_path / "escaped").exists()


def test_c2_policy_requires_holdout_for_every_eligible_candidate(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _frontier_inputs(
        tmp_path,
        split_correctness="passed",
    )

    with pytest.raises(OperatorFrontierQualificationError) as captured:
        OperatorFrontierBundleWriter().run(
            tmp_path / "frontier",
            run_id="must-reject-incomplete-c2-holdout",
            qualification_policy=_qualification_policy(
                minimum_candidate_coverage="C2_MULTI_FAMILY"
            ),
            search_runs=search,
            holdout_runs=holdout,
            confirmation_runs=confirmation,
            query_sizes=(384,),
        )

    assert captured.value.reason_code == "holdout-candidate-coverage-incomplete"
    assert not (tmp_path / "frontier").exists()


def test_c2_policy_qualifies_all_candidates_before_best_of_correct(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _frontier_inputs(
        tmp_path,
        split_correctness="passed",
        include_split_holdout=True,
    )

    run = OperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="complete-c2-holdout",
        qualification_policy=_qualification_policy(
            minimum_candidate_coverage="C2_MULTI_FAMILY"
        ),
        search_runs=search,
        holdout_runs=holdout,
        confirmation_runs=confirmation,
        query_sizes=(384,),
    )

    qualification = json.loads(
        (run / "frontier/qualification.json").read_text(encoding="utf-8")
    )
    assert qualification["candidate_coverage"]["eligible_level"] == (
        "C2_MULTI_FAMILY"
    )
    assert all(
        len(record["holdout_run_ids"]) == 3
        for record in qualification["candidate_records"]
        if record["status"] == "eligible"
    )
    assert all(
        anchor["candidate_id"] == "torch.matmul"
        for anchor in qualification["anchors"]
    )


def test_qualification_rejects_zero_warmup_evidence(tmp_path: Path) -> None:
    search, holdout, confirmation = _frontier_inputs(
        tmp_path,
        warmup_iterations=0,
    )

    with pytest.raises(OperatorFrontierQualificationError) as captured:
        OperatorFrontierBundleWriter().run(
            tmp_path / "frontier",
            run_id="must-reject-zero-warmup",
            qualification_policy=_qualification_policy(
                minimum_warmup_iterations=1
            ),
            search_runs=search,
            holdout_runs=holdout,
            confirmation_runs=confirmation,
            query_sizes=(384,),
        )

    assert captured.value.reason_code == "warmup-policy-failed"


def test_public_cli_publishes_and_replays_operator_frontier(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _frontier_inputs(tmp_path)
    policy_path = tmp_path / "qualification-policy.json"
    policy_path.write_text(
        json.dumps(_qualification_policy(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    command = [
        sys.executable,
        "-m",
        "groundupscale.cli",
        "qualify-frontier",
        "--policy",
        str(policy_path),
    ]
    for run in search:
        command.extend(("--search-run", str(run)))
    for run in holdout:
        command.extend(("--holdout-run", str(run)))
    for run in confirmation:
        command.extend(("--confirmation-run", str(run)))
    command.extend(
        (
            "--query-size",
            "512",
            "--query-size",
            "384",
            "--query-size",
            "640",
            "--artifact-store",
            str(tmp_path / "frontier-cli"),
            "--run-id",
            "ascend-matmul-frontier-cli-v1",
            "--json",
        )
    )

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "qualified"
    assert summary["hardware_cohort"].startswith("ascend-npu-")
    assert summary["verification_passed"] is True
    assert summary["query_statuses"] == {
        "384": "interpolated",
        "512": "exact_anchor",
        "640": "unknown",
    }


def test_frontier_bundle_rejects_changed_source_run_digest(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _frontier_inputs(tmp_path)
    run = OperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="ascend-matmul-frontier-source-digest-v1",
        qualification_policy=_qualification_policy(),
        search_runs=search,
        holdout_runs=holdout,
        confirmation_runs=confirmation,
        query_sizes=(512, 384),
    )
    source_manifest = search[0] / "run.manifest.json"
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    source["created_at"] = "2030-01-01T00:00:00+00:00"
    source_manifest.write_text(
        json.dumps(source, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert (
        "source Run Manifest digest mismatch: search-256-direct-0"
        in verification["failures"]
    )


def test_frontier_verifier_recomputes_surface_input_digest(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _frontier_inputs(tmp_path)
    run = OperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="must-reject-rehashed-surface-tamper",
        qualification_policy=_qualification_policy(),
        search_runs=search,
        holdout_runs=holdout,
        confirmation_runs=confirmation,
        query_sizes=(384,),
    )

    def mutate_qualification(document: dict[str, object]) -> None:
        surface = document["surface"]
        assert isinstance(surface, dict)
        cells = surface["cells"]
        assert isinstance(cells, list)
        cell = cells[0]
        assert isinstance(cell, dict)
        cell["confirmation_observed_rate"] = 1.0

    def mutate_diagnostic(document: dict[str, object]) -> None:
        surfaces = document["capability_surfaces"]
        assert isinstance(surfaces, list)
        surface = surfaces[0]
        assert isinstance(surface, dict)
        cells = surface["cells"]
        assert isinstance(cells, list)
        cell = cells[0]
        assert isinstance(cell, dict)
        cell["confirmation_observed_rate"] = 1.0

    _rewrite_bundle_artifact(
        run, "operator-frontier-qualification", mutate_qualification
    )
    _rewrite_bundle_artifact(run, "diagnostic-evidence", mutate_diagnostic)

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert "operator Frontier Surface input digest mismatch" in verification[
        "failures"
    ]


def test_frontier_verifier_recomputes_diagnostic_evidence_digest(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _frontier_inputs(tmp_path)
    run = OperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="must-reject-rehashed-diagnostic-tamper",
        qualification_policy=_qualification_policy(),
        search_runs=search,
        holdout_runs=holdout,
        confirmation_runs=confirmation,
        query_sizes=(384,),
    )

    def mutate(document: dict[str, object]) -> None:
        queries = document["surface_queries"]
        assert isinstance(queries, list)
        query = queries[0]
        assert isinstance(query, dict)
        query["shape"] = {"s": 640}

    _rewrite_bundle_artifact(run, "diagnostic-evidence", mutate)

    verification = verify_run_bundle(run)

    assert verification["passed"] is False
    assert "operator Frontier diagnostic evidence digest mismatch" in (
        verification["failures"]
    )


def test_qualification_rejects_mixed_hardware_cohort(tmp_path: Path) -> None:
    search, holdout, confirmation = _frontier_inputs(tmp_path)
    search[0] = _measurement_run(
        tmp_path / "foreign-measurement",
        run_id="search-256-direct-foreign-cohort",
        size=256,
        candidate="torch.matmul",
        median_ns=10_000,
        process_id=999,
        foreign_cohort=True,
    )

    with pytest.raises(OperatorFrontierQualificationError) as captured:
        OperatorFrontierBundleWriter().run(
            tmp_path / "frontier",
            run_id="must-reject-mixed-cohort",
            qualification_policy=_qualification_policy(),
            search_runs=search,
            holdout_runs=holdout,
            confirmation_runs=confirmation,
            query_sizes=(384,),
        )

    assert captured.value.reason_code == "hardware-cohort-mismatch"


def test_qualification_rejects_changed_candidate_identity(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _frontier_inputs(tmp_path)
    _rewrite_candidate_identity(search[0])
    assert verify_run_bundle(search[0])["passed"] is True

    with pytest.raises(OperatorFrontierQualificationError) as captured:
        OperatorFrontierBundleWriter().run(
            tmp_path / "frontier",
            run_id="must-reject-changed-candidate",
            qualification_policy=_qualification_policy(),
            search_runs=search,
            holdout_runs=holdout,
            confirmation_runs=confirmation,
            query_sizes=(384,),
        )

    assert captured.value.reason_code == "candidate-identity-changed"


def test_qualification_rejects_candidate_identity_changed_only_in_holdout(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _frontier_inputs(tmp_path)
    for run in holdout:
        if "holdout-256-" in run.name:
            _rewrite_candidate_identity(run)
            assert verify_run_bundle(run)["passed"] is True

    with pytest.raises(OperatorFrontierQualificationError) as captured:
        OperatorFrontierBundleWriter().run(
            tmp_path / "frontier",
            run_id="must-reject-holdout-candidate-change",
            qualification_policy=_qualification_policy(),
            search_runs=search,
            holdout_runs=holdout,
            confirmation_runs=confirmation,
            query_sizes=(384,),
        )

    assert captured.value.reason_code == "candidate-identity-changed"


def test_qualification_rejects_changed_execution_contract(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _frontier_inputs(tmp_path)
    _rewrite_execution_contract(holdout[0])
    assert verify_run_bundle(holdout[0])["passed"] is True

    with pytest.raises(OperatorFrontierQualificationError) as captured:
        OperatorFrontierBundleWriter().run(
            tmp_path / "frontier",
            run_id="must-reject-changed-contract",
            qualification_policy=_qualification_policy(),
            search_runs=search,
            holdout_runs=holdout,
            confirmation_runs=confirmation,
            query_sizes=(384,),
        )

    assert captured.value.reason_code == "execution-contract-mismatch"


def test_qualification_rejects_cross_shape_execution_protocol_change(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _frontier_inputs(tmp_path)
    for run in (*search, *holdout):
        if "-512-" in run.name:
            _rewrite_execution_contract(run)
            assert verify_run_bundle(run)["passed"] is True

    with pytest.raises(OperatorFrontierQualificationError) as captured:
        OperatorFrontierBundleWriter().run(
            tmp_path / "frontier",
            run_id="must-reject-cross-shape-contract-change",
            qualification_policy=_qualification_policy(),
            search_runs=search,
            holdout_runs=holdout,
            confirmation_runs=confirmation,
            query_sizes=(384,),
        )

    assert captured.value.reason_code == "execution-contract-mismatch"


def test_surface_identity_binds_candidate_and_execution_protocols_while_new_evidence_versions(
    tmp_path: Path,
) -> None:
    def qualify(
        root: Path,
        run_id: str,
        *,
        mutate_candidate: bool = False,
        mutate_execution: bool = False,
    ) -> Path:
        search, holdout, confirmation = _frontier_inputs(root)
        observations = [*search, *holdout, *confirmation]
        if mutate_candidate:
            for source in observations:
                if "torch-matmul-k-split-2" not in source.name:
                    _rewrite_candidate_identity(source)
        if mutate_execution:
            for source in observations:
                _rewrite_execution_contract(source)
        return OperatorFrontierBundleWriter().run(
            root / "frontier",
            run_id=run_id,
            qualification_policy=_qualification_policy(),
            search_runs=search,
            holdout_runs=holdout,
            confirmation_runs=confirmation,
            query_sizes=(384,),
        )

    first = qualify(tmp_path / "first", "first-evidence")
    second = qualify(tmp_path / "second", "second-evidence")
    changed_candidate = qualify(
        tmp_path / "candidate", "changed-candidate", mutate_candidate=True
    )
    changed_execution = qualify(
        tmp_path / "execution", "changed-execution", mutate_execution=True
    )

    def surface_ref(run: Path) -> tuple[str, str]:
        qualification = json.loads(
            (run / "frontier/qualification.json").read_text(encoding="utf-8")
        )
        surface = qualification["surface"]
        return surface["surface_id"], surface["version"]

    first_id, first_version = surface_ref(first)
    second_id, second_version = surface_ref(second)
    candidate_id, _ = surface_ref(changed_candidate)
    execution_id, _ = surface_ref(changed_execution)

    assert first_id == second_id
    assert first_version != second_version
    assert candidate_id != first_id
    assert execution_id != first_id
    assert verify_run_bundle(first)["passed"] is True
    assert verify_run_bundle(second)["passed"] is True
    assert diagnose_run_bundle(first)["capability_surface_queries"][0][
        "status"
    ] == "interpolated"


def test_qualification_requires_disjoint_search_and_holdout_processes(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _frontier_inputs(tmp_path)
    _rewrite_environment_session(holdout[0], process_id=10)
    assert verify_run_bundle(holdout[0])["passed"] is True

    with pytest.raises(OperatorFrontierQualificationError) as captured:
        OperatorFrontierBundleWriter().run(
            tmp_path / "frontier",
            run_id="must-reject-reused-process",
            qualification_policy=_qualification_policy(),
            search_runs=search,
            holdout_runs=holdout,
            confirmation_runs=confirmation,
            query_sizes=(384,),
        )

    assert captured.value.reason_code == "sessions-not-independent"


def test_committed_ascend_frontier_evidence_replays() -> None:
    run = (
        Path(__file__).parents[1]
        / "goal_process/issue-31-ascend-matmul-frontier/evidence/runs"
        / "issue31-operator-frontier-v3"
    )

    assert verify_run_bundle(run)["passed"] is True
    qualification = json.loads(
        (run / "frontier/qualification.json").read_text(encoding="utf-8")
    )
    policy = yaml.safe_load(
        (
            Path(__file__).parents[1]
            / "specs/policies/ascend-910b2-matmul-frontier-qualification-v1.yaml"
        ).read_text(encoding="utf-8")
    )
    embedded_policy = dict(qualification["policy"])
    embedded_policy.pop("input_digest")
    assert embedded_policy == policy
    assert qualification["candidate_coverage"]["eligible_level"] == (
        "C2_MULTI_FAMILY"
    )
    assert len(qualification["source_runs"]) == 27
    assert all(
        len(record["holdout_run_ids"]) == 3
        for record in qualification["candidate_records"]
        if record["status"] == "eligible"
    )
    result = diagnose_run_bundle(run)
    queries = {
        item["query_shape"]["s"]: item
        for item in result["capability_surface_queries"]
    }
    assert queries[512]["status"] == "exact_anchor"
    assert queries[384]["status"] == "interpolated"
    assert queries[640]["status"] == "unknown"
    assert queries[640]["reason_code"] == "outside_validated_domain"
