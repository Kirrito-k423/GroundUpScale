from __future__ import annotations

import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from test_ascend_npu_measurement_adapter import (
    _available_runtime,
    _complete_system_probe,
    _raw_hardware_collection,
)

from groundupscale.diagnostics import diagnose_run_bundle
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
) -> Path:
    case = {
        "schema": "groundupscale.dev/exact-shape-matmul-case/v1alpha1",
        "operation": "MatMul",
        "shape": {"left": [size, size], "right": [size, size]},
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "seed": 20260811,
        "candidate": candidate,
        "warmup_iterations": 20,
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


def _frontier_inputs(tmp_path: Path) -> tuple[list[Path], list[Path], list[Path]]:
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
                    correctness="failed",
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
            )
        )
        process_id += 1
    return search, holdout, confirmation


def test_qualifies_best_correct_candidates_and_queries_minimal_surface(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _frontier_inputs(tmp_path)

    run = OperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="ascend-matmul-frontier-v1",
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
    assert queries[384]["cell_id"] == "ascend-matmul-square-256-512"
    assert queries[640]["status"] == "unknown"
    assert queries[640]["reason_code"] == "outside_validated_domain"


def test_public_cli_publishes_and_replays_operator_frontier(
    tmp_path: Path,
) -> None:
    search, holdout, confirmation = _frontier_inputs(tmp_path)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    command = [
        sys.executable,
        "-m",
        "groundupscale.cli",
        "qualify-frontier",
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
            search_runs=search,
            holdout_runs=holdout,
            confirmation_runs=confirmation,
            query_sizes=(384,),
        )

    assert captured.value.reason_code == "execution-contract-mismatch"


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
        / "issue31-operator-frontier-v1"
    )

    assert verify_run_bundle(run)["passed"] is True
    result = diagnose_run_bundle(run)
    queries = {
        item["query_shape"]["s"]: item
        for item in result["capability_surface_queries"]
    }
    assert queries[512]["status"] == "exact_anchor"
    assert queries[384]["status"] == "interpolated"
    assert queries[640]["status"] == "unknown"
    assert queries[640]["reason_code"] == "outside_validated_domain"
