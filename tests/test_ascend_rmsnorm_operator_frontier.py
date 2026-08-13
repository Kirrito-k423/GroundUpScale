from __future__ import annotations

import json
from hashlib import sha256
from math import sqrt
from pathlib import Path

import pytest

from groundupscale.ascend_rmsnorm_frontier import (
    RmsNormOperatorFrontierBundleWriter,
)
from groundupscale.ir import canonical_data, content_fingerprint
from groundupscale.pipeline import compile_analysis_plan
from groundupscale.run_bundle import verify_run_bundle


REPOSITORY_ROOT = Path(__file__).parents[1]
PLAN = REPOSITORY_ROOT / "specs/plans/ascend-npu-transformer-demo.yaml"
COHORT = "ascend-npu-23b93a89d5fecc79"


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        canonical_data(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _rmsnorm_operation():
    compiled = compile_analysis_plan(REPOSITORY_ROOT, PLAN)
    return next(
        operation
        for operation in compiled.cost.cost_ir.walk_operations()
        if operation.stable_path.endswith("/layer_0/input_norm")
    )


def _execution_domain(operation) -> dict[str, object]:
    return {
        "hardware_cohort": COHORT,
        "stable_path": operation.stable_path,
        "operand_shapes": [list(tensor.shape) for tensor in operation.operand_types],
        "result_shapes": [list(tensor.shape) for tensor in operation.result_types],
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "execution_mode": "pytorch-eager",
        "logical_device": "npu:0",
    }


def _phase_evidence(operation, *, missing: str | None = None):
    assert operation.phase_graph is not None
    domain = _execution_domain(operation)
    evidence = []
    for index, phase in enumerate(operation.phase_graph.phases, start=1):
        if phase.phase_name == missing:
            continue
        body = {
            "schema": "groundupscale.dev/operator-phase-capability/v1alpha1",
            "phase_id": phase.phase_id,
            "phase_name": phase.phase_name,
            "operation_class": phase.operation_class,
            "evidence_kind": "exact-operation-probe",
            "execution_domain": domain,
            "candidate": {
                "candidate_id": f"ascend-rmsnorm-{phase.phase_name}-direct-v1",
                "candidate_family": f"torch-npu.{phase.phase_name}",
                "candidate_version": "v1",
            },
            "duration_ns": float(index * 1_000),
            "standard_uncertainty_ns": float(index * 10),
            "source": {
                "run_id": f"issue43-{phase.phase_name}-holdout-001",
                "run_manifest_sha256": sha256(
                    f"issue43-{phase.phase_name}-holdout-001".encode()
                ).hexdigest(),
                "artifact_ref": "artifact://observation/raw/timing.json",
                "correctness": "passed",
                "stability": "passed",
                "lane": "independent-holdout",
            },
        }
        body["input_digest"] = content_fingerprint(body)
        evidence.append(body)
    return evidence


def _artifact(run: Path, role: str) -> dict[str, object]:
    manifest = json.loads((run / "run.manifest.json").read_text(encoding="utf-8"))
    artifact = next(item for item in manifest["artifacts"] if item["role"] == role)
    return json.loads((run / artifact["path"]).read_text(encoding="utf-8"))


def _rewrite_artifact(run: Path, role: str, mutate) -> None:
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(item for item in manifest["artifacts"] if item["role"] == role)
    path = run / artifact["path"]
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact["sha256"] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rehash_connected_qualification(run: Path) -> None:
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_role = {item["role"]: item for item in manifest["artifacts"]}
    qualification_artifact = by_role["compound-operator-frontier-qualification"]
    qualification_path = run / qualification_artifact["path"]
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["source_evidence_digest"] = _canonical_digest(
        qualification["phase_evidence"]
    )
    qualification["input_digest"] = _canonical_digest(
        {
            key: value
            for key, value in qualification.items()
            if key != "input_digest"
        }
    )
    qualification_path.write_text(
        json.dumps(qualification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    qualification_artifact["sha256"] = sha256(
        qualification_path.read_bytes()
    ).hexdigest()

    diagnostic_artifact = by_role["compound-operator-diagnostic"]
    diagnostic_path = run / diagnostic_artifact["path"]
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    diagnostic["qualification_digest"] = qualification["input_digest"]
    diagnostic["input_digest"] = _canonical_digest(
        {key: value for key, value in diagnostic.items() if key != "input_digest"}
    )
    diagnostic_path.write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostic_artifact["sha256"] = sha256(diagnostic_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_complete_rmsnorm_phase_evidence_publishes_replayable_serial_frontier(
    tmp_path: Path,
) -> None:
    operation = _rmsnorm_operation()
    run = RmsNormOperatorFrontierBundleWriter().run(
        tmp_path,
        run_id="issue43-rmsnorm-complete-fixture-v1",
        operation=operation,
        execution_domain=_execution_domain(operation),
        phase_evidence=_phase_evidence(operation),
    )

    assert verify_run_bundle(run) == {
        "schema": "groundupscale.dev/run-verification/v1alpha1",
        "run_id": "issue43-rmsnorm-complete-fixture-v1",
        "passed": True,
        "artifact_count": 3,
        "failures": [],
    }
    qualification = _artifact(run, "compound-operator-frontier-qualification")
    frontier = qualification["operator_frontier"]
    assert qualification["status"] == "qualified"
    assert frontier == {
        "status": "known",
        "duration_ns": 28_000.0,
        "standard_uncertainty_ns": pytest.approx(
            sqrt(sum((index * 10) ** 2 for index in range(1, 8)))
        ),
        "composition_policy": "serialized-no-chunk",
        "formula": "sum(phase.local_duration_ns)",
    }
    schedule = qualification["selected_candidate"]["phase_schedule"]
    assert [phase["phase_name"] for phase in schedule["phases"]] == [
        "square",
        "reduce_sum",
        "mean_scale",
        "epsilon_add",
        "rsqrt",
        "input_scale",
        "weight_scale",
    ]
    assert [phase["predecessor_phase_ids"] for phase in schedule["phases"]] == [
        [],
        [schedule["phases"][0]["phase_id"]],
        [schedule["phases"][1]["phase_id"]],
        [schedule["phases"][2]["phase_id"]],
        [schedule["phases"][3]["phase_id"]],
        [schedule["phases"][4]["phase_id"]],
        [schedule["phases"][5]["phase_id"]],
    ]
    assert all(
        phase["resource_composition"] == "exact-operation-duration"
        for phase in schedule["phases"]
    )
    assert schedule["chunk_pipeline_contract_id"] is None
    assert schedule["overlap_evidence_refs"] == []
    assert qualification["execution_domain"] == _execution_domain(operation)
    assert qualification["source_evidence_digest"] == _canonical_digest(
        qualification["phase_evidence"]
    )


def test_missing_rsqrt_evidence_publishes_replayable_structured_unknown(
    tmp_path: Path,
) -> None:
    operation = _rmsnorm_operation()
    run = RmsNormOperatorFrontierBundleWriter().run(
        tmp_path,
        run_id="issue43-rmsnorm-missing-rsqrt-fixture-v1",
        operation=operation,
        execution_domain=_execution_domain(operation),
        phase_evidence=_phase_evidence(operation, missing="rsqrt"),
    )

    assert verify_run_bundle(run)["passed"] is True
    qualification = _artifact(run, "compound-operator-frontier-qualification")
    assert qualification["status"] == "unknown"
    assert qualification["operator_frontier"] == {
        "status": "unknown",
        "duration_ns": None,
        "standard_uncertainty_ns": None,
        "composition_policy": "serialized-no-chunk",
        "formula": "sum(phase.local_duration_ns)",
    }
    assert qualification["missing_evidence"] == [
        {
            "phase_id": next(
                phase.phase_id
                for phase in operation.phase_graph.phases
                if phase.phase_name == "rsqrt"
            ),
            "phase_name": "rsqrt",
            "operation_class": "transcendental.rsqrt.fp32",
            "required_evidence": (
                "semantically matching capability class or exact operation probe "
                "for the complete execution domain"
            ),
        }
    ]


def test_matmul_probe_cannot_qualify_rmsnorm_reduction_phase(tmp_path: Path) -> None:
    operation = _rmsnorm_operation()
    evidence = _phase_evidence(operation)
    reduce_sum = next(item for item in evidence if item["phase_name"] == "reduce_sum")
    reduce_sum["operation_class"] = "matmul.fp32"
    reduce_sum["input_digest"] = content_fingerprint(
        {key: value for key, value in reduce_sum.items() if key != "input_digest"}
    )

    with pytest.raises(ValueError, match="phase evidence operation class mismatch"):
        RmsNormOperatorFrontierBundleWriter().run(
            tmp_path,
            run_id="issue43-rmsnorm-matmul-substitution-v1",
            operation=operation,
            execution_domain=_execution_domain(operation),
            phase_evidence=evidence,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hardware_cohort", "ascend-npu-foreign"),
        ("dtype", "float16"),
        ("layout", "strided"),
        ("execution_mode", "compiled-graph"),
    ],
)
def test_phase_evidence_domain_mismatch_fails_closed(
    tmp_path: Path, field: str, value: str
) -> None:
    operation = _rmsnorm_operation()
    evidence = _phase_evidence(operation)
    square = evidence[0]
    evidence_domain = square["execution_domain"]
    assert isinstance(evidence_domain, dict)
    square["execution_domain"] = {**evidence_domain, field: value}
    square["input_digest"] = content_fingerprint(
        {key: item for key, item in square.items() if key != "input_digest"}
    )

    with pytest.raises(ValueError, match="phase evidence execution domain mismatch"):
        RmsNormOperatorFrontierBundleWriter().run(
            tmp_path,
            run_id=f"issue43-rmsnorm-domain-mismatch-{field}",
            operation=operation,
            execution_domain=_execution_domain(operation),
            phase_evidence=evidence,
        )


def test_incomplete_bundle_names_every_missing_mandatory_phase(tmp_path: Path) -> None:
    operation = _rmsnorm_operation()
    evidence = _phase_evidence(operation)
    evidence = [
        item
        for item in evidence
        if item["phase_name"] not in {"reduce_sum", "rsqrt"}
    ]

    run = RmsNormOperatorFrontierBundleWriter().run(
        tmp_path,
        run_id="issue43-rmsnorm-two-missing-phases-v1",
        operation=operation,
        execution_domain=_execution_domain(operation),
        phase_evidence=evidence,
    )

    qualification = _artifact(run, "compound-operator-frontier-qualification")
    assert [item["phase_name"] for item in qualification["missing_evidence"]] == [
        "reduce_sum",
        "rsqrt",
    ]
    assert verify_run_bundle(run)["passed"] is True


def test_verifier_rejects_rehashed_phase_duration_tamper(tmp_path: Path) -> None:
    operation = _rmsnorm_operation()
    run = RmsNormOperatorFrontierBundleWriter().run(
        tmp_path,
        run_id="issue43-rmsnorm-rehashed-tamper-v1",
        operation=operation,
        execution_domain=_execution_domain(operation),
        phase_evidence=_phase_evidence(operation),
    )

    def mutate(document: dict[str, object]) -> None:
        phase_evidence = document["phase_evidence"]
        assert isinstance(phase_evidence, list)
        phase_evidence[0]["duration_ns"] = 1.0

    _rewrite_artifact(run, "compound-operator-frontier-qualification", mutate)

    verification = verify_run_bundle(run)
    assert verification["passed"] is False
    assert "compound operator Frontier source evidence digest mismatch" in verification[
        "failures"
    ]


def test_verifier_replays_phase_evidence_instead_of_trusting_rehashed_digests(
    tmp_path: Path,
) -> None:
    operation = _rmsnorm_operation()
    run = RmsNormOperatorFrontierBundleWriter().run(
        tmp_path,
        run_id="issue43-rmsnorm-deep-rehash-v1",
        operation=operation,
        execution_domain=_execution_domain(operation),
        phase_evidence=_phase_evidence(operation),
    )

    def mutate(document: dict[str, object]) -> None:
        phase_evidence = document["phase_evidence"]
        assert isinstance(phase_evidence, list)
        evidence = phase_evidence[0]
        evidence["duration_ns"] = 1.0
        evidence["input_digest"] = content_fingerprint(
            {key: value for key, value in evidence.items() if key != "input_digest"}
        )

    _rewrite_artifact(run, "compound-operator-frontier-qualification", mutate)
    _rehash_connected_qualification(run)

    verification = verify_run_bundle(run)
    assert verification["passed"] is False
    assert "compound operator Frontier phase evidence mismatch" in verification[
        "failures"
    ]


@pytest.mark.parametrize(
    ("tamper", "expected_failure"),
    [
        ("candidate", "compound operator Frontier qualification digest mismatch"),
        ("uncertainty", "compound operator Frontier phase schedule mismatch"),
        ("dependency", "compound operator Frontier phase graph identity mismatch"),
    ],
)
def test_verifier_rejects_rehashed_qualification_tampering(
    tmp_path: Path, tamper: str, expected_failure: str
) -> None:
    operation = _rmsnorm_operation()
    run = RmsNormOperatorFrontierBundleWriter().run(
        tmp_path,
        run_id=f"issue43-rmsnorm-rehashed-{tamper}-v1",
        operation=operation,
        execution_domain=_execution_domain(operation),
        phase_evidence=_phase_evidence(operation),
    )

    if tamper == "dependency":
        def mutate_graph(document: dict[str, object]) -> None:
            phases = document["phases"]
            assert isinstance(phases, list)
            phases[1]["predecessor_phase_ids"] = []
            document["input_digest"] = _canonical_digest(
                {key: value for key, value in document.items() if key != "input_digest"}
            )

        _rewrite_artifact(
            run,
            "operator-phase-graph",
            mutate_graph,
        )
    else:
        def mutate(document: dict[str, object]) -> None:
            selected = document["selected_candidate"]
            assert isinstance(selected, dict)
            schedule = selected["phase_schedule"]
            assert isinstance(schedule, dict)
            phases = schedule["phases"]
            assert isinstance(phases, list)
            if tamper == "candidate":
                candidate = phases[0]["candidate"]
                assert isinstance(candidate, dict)
                candidate["candidate_id"] = "forged-candidate"
            else:
                document["operator_frontier"]["standard_uncertainty_ns"] = 0.0

        _rewrite_artifact(run, "compound-operator-frontier-qualification", mutate)

    verification = verify_run_bundle(run)
    assert verification["passed"] is False
    assert expected_failure in verification["failures"]
