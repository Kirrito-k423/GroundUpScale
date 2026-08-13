from __future__ import annotations

import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

from test_ascend_npu_measurement_adapter import (
    _available_runtime,
    _complete_system_probe,
    _raw_hardware_collection,
)

from groundupscale.ir import content_fingerprint
from groundupscale.measurement_adapters.ascend_npu import (
    AscendNpuMeasurementAdapter,
)
from groundupscale.measurement_run import MeasurementRunBundleWriter
from groundupscale.operator_frontier import SoftmaxOperatorFrontierBundleWriter
from groundupscale.run_bundle import verify_run_bundle


PHASES = (
    ("max_reduce", "torch.amax", "compute.reduction.max.fp32"),
    ("subtract", "torch.sub", "compute.elementwise.subtract.fp32"),
    ("exp", "torch.exp", "compute.transcendental.exp.fp32"),
    ("sum_reduce", "torch.sum", "compute.reduction.sum.fp32"),
    ("normalize", "torch.div", "compute.elementwise.divide.fp32"),
)
STABLE_PATHS = (
    "semantic/workload/transformer-prefill/request/model-prefill/model/transformer/layer_0/attention/softmax",
    "semantic/workload/transformer-prefill/request/model-prefill/model/transformer/layer_1/attention/softmax",
)
DEMO_BUNDLE = (
    Path(__file__).parents[1]
    / "goal_process/issue-30-ascend-transformer-demo/evidence/runs/"
    "ascend-910b2-transformer-demo-20260811-v1"
)


def _phase_run(
    root: Path,
    *,
    phase: str,
    candidate: str,
    lane: str,
    process_id: int,
    shape: list[int] | None = None,
) -> Path:
    case = {
        "schema": "groundupscale.dev/exact-shape-softmax-phase-case/v1alpha1",
        "operation": "SoftmaxPhase",
        "phase": phase,
        "shape": shape or [1, 8, 512, 512],
        "axis": -1,
        "dtype": "float32",
        "layout": "contiguous",
        "candidate": candidate,
        "seed": 20260813,
        "warmup_iterations": 20,
        "repetitions": 5,
        "inner_iterations": 1,
    }
    raw = _raw_hardware_collection(None, 0, case, {})
    raw.update(
        {
            "input_sha256": sha256(
                f"{phase}-{shape or case['shape']}".encode()
            ).hexdigest(),
            "target_output_sha256": sha256(
                f"output-{phase}-{shape or case['shape']}".encode()
            ).hexdigest(),
            "minimum_alignment_bytes": 64,
            "raw_samples_ns": [
                10_000 + process_id,
                10_100 + process_id,
                10_200 + process_id,
                10_300 + process_id,
                10_400 + process_id,
            ],
        }
    )
    raw["correctness"] = {
        **raw["correctness"],
        "status": "passed",
        "max_absolute_error": 0.0,
        "max_relative_error": 0.0,
    }
    adapter = AscendNpuMeasurementAdapter(
        runtime_loader=_available_runtime,
        collection_executor=lambda *args: raw,
        system_probe=_complete_system_probe,
    )
    run_id = f"issue44-{phase}-{lane}-{process_id}"
    run = MeasurementRunBundleWriter(adapter).run(root, case=case, run_id=run_id)
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    environment_entry = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "environment"
    )
    environment_path = run / environment_entry["path"]
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["measurement_session"] = {
        "session_id": run_id,
        "process_id": process_id,
        "process_started_at": f"2026-08-13T00:00:{process_id:02d}+00:00",
        "python_executable": "/trusted/python",
        "source": "python-process-identity",
    }
    environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environment_entry["sha256"] = sha256(environment_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert verify_run_bundle(run)["passed"] is True
    return run


def _phase_inputs(tmp_path: Path) -> dict[str, dict[str, list[Path]]]:
    result: dict[str, dict[str, list[Path]]] = {}
    process_id = 1
    for phase, candidate, _ in PHASES:
        result[phase] = {}
        for lane in ("search", "holdout"):
            result[phase][lane] = [
                _phase_run(
                    tmp_path / "measurements",
                    phase=phase,
                    candidate=candidate,
                    lane=lane,
                    process_id=process_id,
                )
            ]
            process_id += 1
    return result


def _policy() -> dict[str, object]:
    return {
        "schema": "groundupscale.dev/softmax-phase-qualification-policy/v1alpha1",
        "policy_id": "issue44-softmax-phase-frontier",
        "version": "v1",
        "composition_rule": "serialized-critical-path-sum",
        "minimum_search_sessions_per_phase": 1,
        "minimum_holdout_sessions_per_phase": 1,
        "uncertainty_combination": "root-sum-of-squares",
        "change_reason": "qualify the fixed Transformer demo Softmax domain",
        "revalidation": "on graph, domain, candidate, cohort, or evidence change",
    }


def _session_metadata(
    cohort: str = "ascend-npu-febd831c8d07e06f",
) -> dict[str, object]:
    owner = (
        "issue=44 pid=2226102 host=localhost.localdomain "
        "started=2026-08-13T18:59:55+08:00"
    )
    return {
        "schema": "groundupscale.dev/ascend-host-lock-session/v1alpha1",
        "issue": 44,
        "lock_path": (
            "/home/t00906153/.groundupscale/locks/"
            "ascend-910b2-host.lock"
        ),
        "owner_start": owner,
        "owner_end": owner,
        "started_at": "2026-08-13T18:59:55+08:00",
        "ended_at": "2026-08-13T19:03:24+08:00",
        "device_visibility": "0",
        "hardware_cohort": cohort,
        "wrapper_sha256": (
            "22d43618f1c616b2ff70570944c7447cd851aac98bfedb111b7912fc36b94787"
        ),
    }


def test_complete_softmax_phase_graph_publishes_replayable_frontier(
    tmp_path: Path,
) -> None:
    run = SoftmaxOperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="issue44-softmax-complete-test",
        qualification_policy=_policy(),
        phase_runs=_phase_inputs(tmp_path),
        source_demo_bundle=DEMO_BUNDLE,
        session_metadata=_session_metadata(),
    )

    assert verify_run_bundle(run)["passed"] is True
    qualification = json.loads(
        (run / "frontier/qualification.json").read_text(encoding="utf-8")
    )
    assert qualification["status"] == "qualified"
    surface = qualification["surface"]
    assert surface["domain"] == {
        "semantic_operation": "Softmax",
        "shape": [1, 8, 512, 512],
        "axis": -1,
        "dtype": "float32",
        "layout": "contiguous",
        "execution_mode": "pytorch-eager",
        "logical_device": "npu:0",
    }
    graph = surface["operator_phase_graph"]
    assert graph["stable_paths"] == list(STABLE_PATHS)
    assert [phase["phase_name"] for phase in graph["phases"]] == [
        phase for phase, _, _ in PHASES
    ]
    assert [phase["required_capability_class"] for phase in graph["phases"]] == [
        resource for _, _, resource in PHASES
    ]
    assert graph["composition"]["rule"] == "serialized-critical-path-sum"
    assert graph["composition"]["operator_frontier_ns"] == sum(
        phase["selected_duration_ns"] for phase in graph["phases"]
    )
    assert all(phase["source_digests"] for phase in graph["phases"])


def test_missing_softmax_phase_stays_structured_unknown(tmp_path: Path) -> None:
    phase_runs = _phase_inputs(tmp_path)
    del phase_runs["exp"]
    run = SoftmaxOperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="issue44-softmax-missing-exp-test",
        qualification_policy=_policy(),
        phase_runs=phase_runs,
        source_demo_bundle=DEMO_BUNDLE,
    )

    assert verify_run_bundle(run)["passed"] is True
    qualification = json.loads(
        (run / "frontier/qualification.json").read_text(encoding="utf-8")
    )
    assert qualification["status"] == "unknown"
    assert qualification["surface"]["operator_phase_graph"]["composition"] == {
        "status": "unknown",
        "rule": "serialized-critical-path-sum",
        "operator_frontier_ns": None,
        "standard_uncertainty_ns": None,
        "missing_evidence": [
            {
                "phase_name": "exp",
                "required_capability_class": "compute.transcendental.exp.fp32",
                "reason_code": "missing-mandatory-phase-evidence",
            }
        ],
    }


def test_softmax_phase_domain_mismatch_fails_closed(tmp_path: Path) -> None:
    phase_runs = _phase_inputs(tmp_path)
    phase_runs["normalize"]["holdout"] = [
        _phase_run(
            tmp_path / "mismatch",
            phase="normalize",
            candidate="torch.div",
            lane="holdout",
            process_id=20,
            shape=[1, 8, 256, 256],
        )
    ]
    run = SoftmaxOperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="issue44-softmax-domain-mismatch-test",
        qualification_policy=_policy(),
        phase_runs=phase_runs,
        source_demo_bundle=DEMO_BUNDLE,
    )

    assert verify_run_bundle(run)["passed"] is True
    qualification = json.loads(
        (run / "frontier/qualification.json").read_text(encoding="utf-8")
    )
    assert qualification["status"] == "unknown"
    assert qualification["reason_code"] == "mandatory-phase-domain-mismatch"
    assert qualification["surface"]["operator_phase_graph"]["composition"][
        "operator_frontier_ns"
    ] is None


def test_legacy_softmax_operands_stay_structured_unknown(tmp_path: Path) -> None:
    boundaries = [
        {
            "phase_name": phase,
            "required_capability_class": capability,
            "reason_code": "missing-real-chain-operand-evidence",
        }
        for phase, _, capability in PHASES
        if phase in {"exp", "sum_reduce", "normalize"}
    ]
    run = SoftmaxOperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="issue44-softmax-legacy-operands-test",
        qualification_policy=_policy(),
        phase_runs=_phase_inputs(tmp_path),
        source_demo_bundle=DEMO_BUNDLE,
        session_metadata=_session_metadata(),
        evidence_boundaries=boundaries,
    )
    assert verify_run_bundle(run)["passed"] is True
    qualification = json.loads(
        (run / "frontier/qualification.json").read_text(encoding="utf-8")
    )
    assert qualification["status"] == "unknown"
    assert qualification["reason_code"] == "legacy-synthetic-operand-domain"
    assert qualification["surface"]["operator_phase_graph"]["phases"] == []
    assert qualification["surface"]["operator_phase_graph"]["composition"] == {
        "status": "unknown",
        "rule": "serialized-critical-path-sum",
        "operator_frontier_ns": None,
        "standard_uncertainty_ns": None,
        "missing_evidence": boundaries,
    }


def test_verifier_recomputes_softmax_serial_composition(tmp_path: Path) -> None:
    run = SoftmaxOperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="issue44-softmax-composition-tamper-test",
        qualification_policy=_policy(),
        phase_runs=_phase_inputs(tmp_path),
        source_demo_bundle=DEMO_BUNDLE,
        session_metadata=_session_metadata(),
    )
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qualification_entry = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "operator-frontier-qualification"
    )
    path = run / qualification_entry["path"]
    qualification = json.loads(path.read_text(encoding="utf-8"))
    graph = qualification["surface"]["operator_phase_graph"]
    graph["composition"]["operator_frontier_ns"] += 1
    qualification["surface"].pop("input_digest")
    qualification["surface"]["input_digest"] = content_fingerprint(
        qualification["surface"]
    )
    path.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    qualification_entry["sha256"] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(run)
    assert verification["passed"] is False
    assert "Softmax Operator Frontier composition mismatch" in verification[
        "failures"
    ]


def test_verifier_replays_softmax_source_digest_and_uncertainty(
    tmp_path: Path,
) -> None:
    run = SoftmaxOperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="issue44-softmax-source-replay-tamper-test",
        qualification_policy=_policy(),
        phase_runs=_phase_inputs(tmp_path),
        source_demo_bundle=DEMO_BUNDLE,
        session_metadata=_session_metadata(),
    )
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        artifact for artifact in manifest["artifacts"]
        if artifact["role"] == "operator-frontier-qualification"
    )
    path = run / entry["path"]
    qualification = json.loads(path.read_text(encoding="utf-8"))
    phase = qualification["surface"]["operator_phase_graph"]["phases"][0]
    phase["source_digests"][0] = "0" * 64
    phase["standard_uncertainty_ns"] += 1
    composition = qualification["surface"]["operator_phase_graph"]["composition"]
    composition["standard_uncertainty_ns"] = sum(
        item["standard_uncertainty_ns"] ** 2
        for item in qualification["surface"]["operator_phase_graph"]["phases"]
    ) ** 0.5
    qualification["surface"].pop("input_digest")
    qualification["surface"]["input_digest"] = content_fingerprint(
        qualification["surface"]
    )
    path.write_text(json.dumps(qualification, indent=2, sort_keys=True) + "\n")
    entry["sha256"] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    verification = verify_run_bundle(run)
    assert verification["passed"] is False
    assert "Softmax Operator Frontier composition mismatch" in verification["failures"]


def test_verifier_rejects_rehashed_softmax_demo_and_lock_metadata(
    tmp_path: Path,
) -> None:
    run = SoftmaxOperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="issue44-softmax-lineage-tamper-test",
        qualification_policy=_policy(),
        phase_runs=_phase_inputs(tmp_path),
        source_demo_bundle=DEMO_BUNDLE,
        session_metadata=_session_metadata(),
    )
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        artifact for artifact in manifest["artifacts"]
        if artifact["role"] == "operator-frontier-qualification"
    )
    path = run / entry["path"]
    qualification = json.loads(path.read_text(encoding="utf-8"))
    surface = qualification["surface"]
    surface["source_demo"]["semantic_ir_sha256"] = "0" * 64
    surface["measurement_session"]["device_visibility"] = "1"
    surface.pop("input_digest")
    surface["input_digest"] = content_fingerprint(surface)
    path.write_text(json.dumps(qualification, indent=2, sort_keys=True) + "\n")
    entry["sha256"] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    verification = verify_run_bundle(run)
    assert verification["passed"] is False
    assert "Softmax Operator Frontier composition mismatch" in verification["failures"]


def test_public_measure_cli_accepts_exact_softmax_phase_contract(
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "measure",
            "--device",
            "ascend-npu",
            "--operation",
            "SoftmaxPhase",
            "--phase",
            "exp",
            "--shape",
            "1,8,512,512",
            "--axis",
            "-1",
            "--dtype",
            "float32",
            "--layout",
            "contiguous",
            "--candidate",
            "torch.exp",
            "--artifact-store",
            str(tmp_path),
            "--run-id",
            "issue44-cli-softmax-exp",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "blocked"
    case = json.loads(
        (Path(summary["run_bundle"]) / "resolved/case.json").read_text(
            encoding="utf-8"
        )
    )
    assert case == {
        "schema": "groundupscale.dev/exact-shape-softmax-phase-case/v1alpha1",
        "operation": "SoftmaxPhase",
        "phase": "exp",
        "shape": [1, 8, 512, 512],
        "axis": -1,
        "dtype": "float32",
        "layout": "contiguous",
        "seed": 20260810,
        "candidate": "torch.exp",
        "warmup_iterations": 20,
        "repetitions": 100,
        "inner_iterations": 1,
    }
