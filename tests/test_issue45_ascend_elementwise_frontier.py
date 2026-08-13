from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
from hashlib import sha256
import fcntl
import subprocess
import os
import yaml

import pytest

from groundupscale.operator_shape_semantics import (
    UnsupportedOperatorShape,
    semantics_from_case,
)
from groundupscale.measurement_adapters.ascend_npu import (
    AscendNpuMeasurementAdapter,
    _collect_exact_shape_elementwise,
)
from groundupscale.measurement_run import MeasurementRunBundleWriter
from groundupscale.run_bundle import verify_run_bundle
from groundupscale.operator_frontier import OperatorFrontierBundleWriter
from groundupscale.operator_frontier import OperatorFrontierQualificationError
from groundupscale.diagnostics import diagnose_run_bundle


def _case(operation: str, shape: list[int], *, operand_kind: str) -> dict[str, object]:
    return {
        "schema": "groundupscale.dev/exact-shape-elementwise-case/v1alpha1",
        "operation": operation,
        "shape": {"result": shape},
        "operand_kind": operand_kind,
        "dtype": "float32",
        "layout": "contiguous",
        "candidate": f"torch.{operation.casefold()}",
        "seed": 20260813,
        "warmup_iterations": 20,
        "repetitions": 100,
        "inner_iterations": 100,
    }


@pytest.mark.parametrize(
    ("operation", "shape", "operand_kind", "elements", "work"),
    [
        ("Add", [1, 8, 512, 512], "tensor-broadcast", 2_097_152, 2_097_152),
        ("Add", [1, 512, 512], "tensor-tensor", 262_144, 262_144),
        ("Mul", [1, 8, 512, 512], "tensor-scalar", 2_097_152, 2_097_152),
        ("Mul", [1, 512, 2048], "tensor-tensor", 1_048_576, 1_048_576),
        ("SiLU", [1, 512, 2048], "tensor", 1_048_576, 5_242_880),
    ],
)
def test_elementwise_shape_semantics_preserve_distinct_execution_domains(
    operation: str,
    shape: list[int],
    operand_kind: str,
    elements: int,
    work: int,
) -> None:
    semantics = semantics_from_case(_case(operation, shape, operand_kind=operand_kind))

    assert semantics.operation == operation
    assert semantics.normalized_shape == {"result": shape, "elements": elements}
    assert semantics.domain_facets == {
        "semantic_operation": operation,
        "dtype": "float32",
        "layout": "contiguous",
        "operand_kind": operand_kind,
    }
    assert semantics.declared_work == work
    assert semantics.coordinate_axis == "elements"
    assert semantics.coordinate_value == elements


def test_elementwise_shape_semantics_fail_closed_on_domain_mismatch() -> None:
    case = _case("Add", [1, 512, 512], operand_kind="tensor-tensor")
    case["layout"] = "row-major-contiguous"

    with pytest.raises(UnsupportedOperatorShape, match="contiguous float32"):
        semantics_from_case(case)


def test_measure_cli_builds_exact_elementwise_execution_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from groundupscale import cli

    captured: dict[str, object] = {}

    class FakeWriter:
        def __init__(self, adapter: object) -> None:
            captured["adapter"] = adapter

        def run(
            self, root: Path, *, case: dict[str, object], run_id: str
        ) -> Path:
            captured["case"] = case
            run = root / "runs" / run_id
            run.mkdir(parents=True)
            (run / "run.manifest.json").write_text(
                '{"run_id":"issue45-cli-test","status":"completed",'
                '"device":"ascend-npu","hardware_cohort":"test-cohort"}\n',
                encoding="utf-8",
            )
            return run

    monkeypatch.setattr(cli, "MeasurementRunBundleWriter", FakeWriter)
    monkeypatch.setattr(cli, "verify_run_bundle", lambda run: {"passed": True})
    adapter = SimpleNamespace(name="fake-ascend-adapter")

    exit_code = cli.main(
        [
            "measure",
            "--device",
            "ascend-npu",
            "--operation",
            "Add",
            "--elementwise-shape",
            "1,8,512,512",
            "--operand-kind",
            "tensor-broadcast",
            "--dtype",
            "float32",
            "--layout",
            "contiguous",
            "--candidate",
            "torch.add",
            "--artifact-store",
            str(tmp_path),
            "--run-id",
            "issue45-cli-test",
            "--json",
        ],
        measurement_adapter_factory=lambda *args, **kwargs: adapter,
    )

    assert exit_code == 0
    assert captured["case"] == {
        "schema": "groundupscale.dev/exact-shape-elementwise-case/v1alpha1",
        "operation": "Add",
        "shape": {"result": [1, 8, 512, 512]},
        "operand_kind": "tensor-broadcast",
        "dtype": "float32",
        "layout": "contiguous",
        "seed": 20260810,
        "candidate": "torch.add",
        "warmup_iterations": 20,
        "repetitions": 100,
        "inner_iterations": 1,
    }


@pytest.mark.parametrize(
    ("operation", "operand_kind", "candidate"),
    [
        ("Add", "tensor-broadcast", "torch.add"),
        ("Mul", "tensor-scalar", "torch.mul"),
        ("SiLU", "tensor", "torch.nn.functional.silu"),
    ],
)
def test_elementwise_collection_preserves_correctness_and_timing_boundary(
    operation: str,
    operand_kind: str,
    candidate: str,
) -> None:
    import torch

    class CpuNpuFacade:
        class Event:
            def __init__(self, enable_timing: bool) -> None:
                self.enable_timing = enable_timing

            def record(self) -> None:
                pass

            def synchronize(self) -> None:
                pass

            def elapsed_time(self, end: object) -> float:
                return 0.125

        @staticmethod
        def set_device(index: int) -> None:
            pass

        @staticmethod
        def synchronize() -> None:
            pass

        @staticmethod
        def memory_allocated() -> int:
            return 10

        @staticmethod
        def memory_reserved() -> int:
            return 20

        @staticmethod
        def max_memory_allocated() -> int:
            return 30

        @staticmethod
        def get_device_name(index: int) -> str:
            return "Fake Ascend 910B2"

    class TorchFacade:
        float32 = torch.float32
        npu = CpuNpuFacade()
        Generator = torch.Generator
        randn = staticmethod(torch.randn)
        isfinite = staticmethod(torch.isfinite)
        allclose = staticmethod(torch.allclose)
        add = staticmethod(torch.add)
        mul = staticmethod(torch.mul)
        nn = torch.nn

    case = _case(operation, [2, 4], operand_kind=operand_kind)
    case["candidate"] = candidate
    case["warmup_iterations"] = 1
    case["repetitions"] = 3
    case["inner_iterations"] = 2

    result = _collect_exact_shape_elementwise(
        TorchFacade(),
        0,
        case,
        {
            "warmup_iterations": 1,
            "repetitions": 3,
            "inner_iterations": 2,
        },
        device="cpu",
    )

    assert result["correctness"]["status"] == "passed"
    assert result["cpu_fallback"] is False
    assert result["raw_samples_ns"] == [62500, 62500, 62500]
    assert result["input_sha256"]
    assert result["target_output_sha256"]


def test_adapter_publishes_elementwise_candidate_input_and_lock_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case("Mul", [1, 512, 2048], operand_kind="tensor-tensor")
    raw = {
        "runtime_device_name": "Ascend 910B2",
        "candidate_device": "npu:0",
        "cpu_fallback": False,
        "minimum_alignment_bytes": 512,
        "input_sha256": "a" * 64,
        "other_sha256": "b" * 64,
        "target_output_sha256": "c" * 64,
        "correctness": {
            "status": "passed",
            "oracle": "cpu-float64-elementwise",
            "atol": 1e-6,
            "rtol": 1e-5,
            "max_absolute_error": 0.0,
            "max_relative_error": 0.0,
            "finite": True,
            "shape_exact": True,
        },
        "raw_samples_ns": [1000, 1001, 999],
        "memory": {},
        "device_event_id": "event-pair",
        "stream_id": "default-npu-stream",
    }
    torch_runtime = SimpleNamespace(__version__="2.7.1")
    torch_npu_runtime = SimpleNamespace(__version__="2.7.1")
    adapter = AscendNpuMeasurementAdapter(
        runtime_loader=lambda: (torch_runtime, torch_npu_runtime),
        collection_executor=lambda *args: raw,
        system_probe=lambda index: {},
    )
    owner_file = tmp_path / "ascend.owner"
    owner_file.write_text(
        "issue=45 pid=42 host=test started=2026-08-13T00:00:00Z\n"
    )
    monkeypatch.setenv("GROUNDUPSCALE_ISSUE", "45")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("GROUNDUPSCALE_NPU_LOCK_OWNER_FILE", str(owner_file))
    lock_file = tmp_path / "ascend.lock"
    lock_fd = os.open(lock_file, os.O_RDWR | os.O_CREAT)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    monkeypatch.setenv("GROUNDUPSCALE_NPU_LOCK_FD", str(lock_fd))
    monkeypatch.setenv("GROUNDUPSCALE_NPU_LOCK_PATH", str(lock_file))
    timing_plan = {
        "warmup_iterations": 20,
        "repetitions": 100,
        "inner_iterations": 100,
        "sample_exclusion": "none-preserve-all-raw-samples",
        "timer": {"source": "torch.npu.Event.elapsed_time"},
        "completion_boundary": {
            "kind": "device-event-stream-completion",
            "protocol": "end-event-synchronize-plus-device-synchronize",
        },
    }

    try:
        collection = adapter.collect(case, timing_plan)
    finally:
        os.close(lock_fd)

    candidate = collection["candidate_identity"]
    assert candidate["operation"] == "Mul"
    assert candidate["candidate_id"] == "torch.mul"
    assert candidate["candidate_family"] == "pytorch-ascend-elementwise-mul"
    assert candidate["semantic_domain"]["operand_kind"] == "tensor-tensor"
    corpus = collection["input_corpus"]
    assert corpus["input_sha256"] == "a" * 64
    assert corpus["other_sha256"] == "b" * 64
    assert corpus["operator_shape_identity"] == candidate["operator_shape_identity"]
    assert collection["npu_host_lock"] == {
        "schema": "groundupscale.dev/npu-host-lock-metadata/v1alpha1",
        "status": "held-during-collection",
        "lock_path": str(lock_file),
        "lock_fd": lock_fd,
        "lock_validation": "inherited-fd-inode-and-exclusive-conflict",
        "owner": "issue=45 pid=42 host=test started=2026-08-13T00:00:00Z",
        "collection_finished_at": collection["npu_host_lock"][
            "collection_finished_at"
        ],
        "hardware_cohort": "ascend-npu-23b93a89d5fecc79",
        "device_visibility": "0",
    }


def test_adapter_rejects_owner_file_without_inherited_locked_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner_file = tmp_path / "ascend.owner"
    owner_file.write_text("issue=45 pid=42 host=test started=now\n")
    monkeypatch.setenv("GROUNDUPSCALE_ISSUE", "45")
    monkeypatch.setenv("GROUNDUPSCALE_NPU_LOCK_OWNER_FILE", str(owner_file))
    monkeypatch.delenv("GROUNDUPSCALE_NPU_LOCK_FD", raising=False)

    from groundupscale.measurement_adapters.ascend_npu import _host_lock_metadata

    assert _host_lock_metadata() is None


def _rewrite_environment_session(run: Path, process_id: int) -> None:
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(
        item for item in manifest["artifacts"] if item["role"] == "environment"
    )
    path = run / artifact["path"]
    environment = json.loads(path.read_text(encoding="utf-8"))
    environment["measurement_session"] = {
        "session_id": manifest["run_id"],
        "process_id": process_id,
        "process_started_at": f"2026-08-13T00:00:{process_id:02d}+00:00",
        "python_executable": "/trusted/python",
        "source": "python-process-identity",
    }
    path.write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    artifact["sha256"] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _elementwise_measurement_run(
    root: Path,
    *,
    run_id: str,
    operation: str,
    shape: list[int],
    operand_kind: str,
    median_ns: int,
    process_id: int,
    raw_samples_ns: list[int] | None = None,
) -> Path:
    case = _case(operation, shape, operand_kind=operand_kind)
    candidate = {
        "Add": "torch.add",
        "Mul": "torch.mul",
        "SiLU": "torch.nn.functional.silu",
    }[operation]
    case["candidate"] = candidate
    case["repetitions"] = 5
    case["inner_iterations"] = 100
    raw = {
        "runtime_device_name": "Ascend 910B2",
        "candidate_device": "npu:0",
        "cpu_fallback": False,
        "minimum_alignment_bytes": 512,
        "input_sha256": sha256(f"input-{operation}-{shape}".encode()).hexdigest(),
        "other_sha256": (
            sha256(f"other-{operation}-{shape}".encode()).hexdigest()
            if operand_kind in {"tensor-tensor", "tensor-broadcast"}
            else None
        ),
        "target_output_sha256": sha256(f"output-{operation}-{shape}".encode()).hexdigest(),
        "correctness": {
            "status": "passed",
            "oracle": "cpu-float64-elementwise",
            "atol": 1e-6,
            "rtol": 1e-5,
            "max_absolute_error": 0.0,
            "max_relative_error": 0.0,
            "finite": True,
            "shape_exact": True,
        },
        "raw_samples_ns": raw_samples_ns or [
            median_ns - 2,
            median_ns - 1,
            median_ns,
            median_ns + 1,
            median_ns + 2,
        ],
        "memory": {},
        "device_event_id": "event-pair",
        "stream_id": "default-npu-stream",
    }
    from test_ascend_npu_measurement_adapter import (
        _available_runtime,
        _complete_system_probe,
    )

    adapter = AscendNpuMeasurementAdapter(
        runtime_loader=_available_runtime,
        collection_executor=lambda *args: raw,
        system_probe=_complete_system_probe,
    )
    run = MeasurementRunBundleWriter(adapter).run(root, case=case, run_id=run_id)
    _inject_authoritative_host_lock(run)
    _rewrite_environment_session(run, process_id)
    verification = verify_run_bundle(run)
    assert verification["passed"] is True, verification["failures"]
    return run


def _inject_authoritative_host_lock(run: Path) -> None:
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifact = next(
        item for item in manifest["artifacts"]
        if item["role"] == "measurement-collection"
    )
    collection_path = run / artifact["path"]
    collection = json.loads(collection_path.read_text())
    collection["npu_host_lock"] = {
        "schema": "groundupscale.dev/npu-host-lock-metadata/v1alpha1",
        "status": "held-during-collection",
        "lock_path": (
            "/home/t00906153/.groundupscale/locks/ascend-910b2-host.lock"
        ),
        "owner": "issue=45 pid=1 host=test started=2026-08-13T00:00:00Z",
        "collection_finished_at": "2026-08-13T00:01:00Z",
        "hardware_cohort": manifest["hardware_cohort"],
        "device_visibility": "0",
    }
    collection_path.write_text(json.dumps(collection, indent=2, sort_keys=True) + "\n")
    artifact["sha256"] = sha256(collection_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _remove_host_lock_metadata(run: Path) -> None:
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifact = next(
        item for item in manifest["artifacts"]
        if item["role"] == "measurement-collection"
    )
    collection_path = run / artifact["path"]
    collection = json.loads(collection_path.read_text())
    collection.pop("npu_host_lock", None)
    collection_path.write_text(json.dumps(collection, indent=2, sort_keys=True) + "\n")
    artifact["sha256"] = sha256(collection_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_elementwise_measurement_run_bundle_replays_through_public_verifier(
    tmp_path: Path,
) -> None:
    run = _elementwise_measurement_run(
        tmp_path,
        run_id="issue45-add-residual-search-01",
        operation="Add",
        shape=[1, 512, 512],
        operand_kind="tensor-tensor",
        median_ns=10_000,
        process_id=1,
    )

    result = verify_run_bundle(run)

    assert result["passed"] is True
    case = json.loads((run / "resolved/case.json").read_text())
    candidate = json.loads((run / "observation/candidate.json").read_text())
    assert case["operand_kind"] == "tensor-tensor"
    assert candidate["candidate_id"] == "torch.add"


def _exact_elementwise_policy(
    *, operation: str, shape: list[int], operand_kind: str
) -> dict[str, object]:
    candidate = {
        "Add": "torch.add",
        "Mul": "torch.mul",
        "SiLU": "torch.nn.functional.silu",
    }[operation]
    return {
        "schema": "groundupscale.dev/operator-frontier-qualification-policy/v1alpha1",
        "policy_id": f"issue45-{operation.casefold()}-exact-frontier",
        "version": "v1",
        "scope": {
            "hardware_cohort": "ascend-npu-febd831c8d07e06f",
            "operation": operation,
            "dtype": "float32",
            "layout": "contiguous",
            "operand_kind": operand_kind,
            "sequence_distribution_mode": "exact-only",
            "result_shapes": [shape],
            "candidate_ids": [candidate],
        },
        "minimum_search_sessions": 3,
        "minimum_holdout_sessions": 3,
        "minimum_confirmation_sessions": 3,
        "minimum_warmup_iterations": 20,
        "maximum_session_median_relative_range": 0.10,
        "minimum_candidate_coverage": "C0_SINGLE",
        "holdout_candidate_scope": "all-eligible-candidates",
        "uncertainty_combination": "root-sum-of-squares",
        "target_coverage": 0.68,
        "sample_exclusion": "none-preserve-all-raw-samples",
        "estimator": "median(independent-holdout-session-medians)",
        "change_reason": "issue 45 exact elementwise execution domain",
        "revalidation": "on cohort, domain, candidate, evidence, or policy change",
        "evidence_ref": "artifact://issue-45/policy",
    }


def test_elementwise_frontier_qualifies_exact_domain_and_fails_closed_elsewhere(
    tmp_path: Path,
) -> None:
    measurements = tmp_path / "measurements"
    search = [
        _elementwise_measurement_run(
            measurements,
            run_id=f"issue45-add-residual-search-{session}",
            operation="Add",
            shape=[1, 512, 512],
            operand_kind="tensor-tensor",
            median_ns=10_000 + session,
            process_id=session,
        )
        for session in (1, 2, 3)
    ]
    holdout = [
        _elementwise_measurement_run(
            measurements,
            run_id=f"issue45-add-residual-holdout-{session}",
            operation="Add",
            shape=[1, 512, 512],
            operand_kind="tensor-tensor",
            median_ns=10_010 + session,
            process_id=session + 3,
        )
        for session in (1, 2, 3)
    ]

    run = OperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="issue45-add-residual-frontier-v1",
        qualification_policy=_exact_elementwise_policy(
            operation="Add",
            shape=[1, 512, 512],
            operand_kind="tensor-tensor",
        ),
        search_runs=search,
        holdout_runs=holdout,
        confirmation_runs=[],
        query_sizes=[],
        query_shapes=(
            {"result": [1, 512, 512]},
            {"result": [1, 8, 512, 512]},
        ),
    )

    assert verify_run_bundle(run)["passed"] is True
    qualification = json.loads((run / "frontier/qualification.json").read_text())
    assert qualification["status"] == "qualified"
    assert qualification["surface"]["domain"]["operand_kind"] == "tensor-tensor"
    assert qualification["surface"]["coordinate"]["axis"] == "operator_shape_index"
    assert qualification["anchors"][0]["shape"] == {"operator_shape_index": 1}
    assert qualification["surface"]["cells"][0]["regime_id"] == (
        "operator-shape-exact-anchor"
    )
    result = diagnose_run_bundle(run)
    exact, mismatch = result["capability_surface_queries"]
    assert exact["status"] == "exact_anchor"
    assert exact["latency"]["value_ns"] == pytest.approx(10_012)
    assert mismatch["status"] == "unknown"
    assert mismatch["reason_code"] == "unsupported_sequence_distribution_interpolation"


@pytest.mark.parametrize(
    ("query_override", "reason_code"),
    [
        ({"domain": {"dtype": "float16"}}, "query-domain-mismatch"),
        ({"cohort_id": "other-cohort"}, "query-hardware-cohort-mismatch"),
        ({"candidate_family": "other-family"}, "query-candidate-family-mismatch"),
    ],
)
def test_elementwise_exact_surface_rejects_full_identity_mismatch(
    tmp_path: Path,
    query_override: dict[str, object],
    reason_code: str,
) -> None:
    measurements = tmp_path / "measurements"
    search = [
        _elementwise_measurement_run(
            measurements,
            run_id=f"issue45-add-identity-search-{session}",
            operation="Add",
            shape=[1, 512, 512],
            operand_kind="tensor-tensor",
            median_ns=10_000 + session,
            process_id=session,
        )
        for session in (1, 2, 3)
    ]
    holdout = [
        _elementwise_measurement_run(
            measurements,
            run_id=f"issue45-add-identity-holdout-{session}",
            operation="Add",
            shape=[1, 512, 512],
            operand_kind="tensor-tensor",
            median_ns=10_010 + session,
            process_id=session + 3,
        )
        for session in (1, 2, 3)
    ]
    run = OperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="issue45-add-identity-v1",
        qualification_policy=_exact_elementwise_policy(
            operation="Add",
            shape=[1, 512, 512],
            operand_kind="tensor-tensor",
        ),
        search_runs=search,
        holdout_runs=holdout,
        confirmation_runs=[],
        query_sizes=[],
        query_shapes=({"result": [1, 512, 512]},),
    )
    diagnostic = json.loads((run / "diagnostic/evidence.json").read_text())
    query = diagnostic["surface_queries"][0]
    if "domain" in query_override:
        query["domain"] = {**query["domain"], **query_override["domain"]}
    else:
        query.update(query_override)
    from groundupscale.diagnostics import _query_capability_surface

    diagnosed = _query_capability_surface(
        query, diagnostic["capability_surfaces"][0]
    )

    assert diagnosed["status"] == "unknown"
    assert diagnosed["reason_code"] == reason_code


def test_elementwise_frontier_publishes_structured_unknown_when_sessions_are_unstable(
    tmp_path: Path,
) -> None:
    measurements = tmp_path / "measurements"
    search = [
        _elementwise_measurement_run(
            measurements,
            run_id=f"issue45-mul-gate-search-{session}",
            operation="Mul",
            shape=[1, 512, 2048],
            operand_kind="tensor-tensor",
            median_ns=median_ns,
            process_id=session,
        )
        for session, median_ns in enumerate((10_000, 10_100, 13_000), start=1)
    ]
    holdout = [
        _elementwise_measurement_run(
            measurements,
            run_id=f"issue45-mul-gate-holdout-{session}",
            operation="Mul",
            shape=[1, 512, 2048],
            operand_kind="tensor-tensor",
            median_ns=10_200 + session,
            process_id=session + 3,
        )
        for session in (1, 2, 3)
    ]

    run = OperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="issue45-mul-gate-unstable-v1",
        qualification_policy=_exact_elementwise_policy(
            operation="Mul",
            shape=[1, 512, 2048],
            operand_kind="tensor-tensor",
        ),
        search_runs=search,
        holdout_runs=holdout,
        confirmation_runs=[],
        query_sizes=[],
        query_shapes=({"result": [1, 512, 2048]},),
    )

    verification = verify_run_bundle(run)
    assert verification["passed"] is True, verification["failures"]
    qualification = json.loads((run / "frontier/qualification.json").read_text())
    assert qualification["status"] == "unknown"
    assert qualification["reason_code"] == "exact-shape-session-repeatability-failed"
    query = diagnose_run_bundle(run)["capability_surface_queries"][0]
    assert query["status"] == "unknown"
    assert query["reason_code"] == "exact-shape-session-repeatability-failed"


def test_elementwise_frontier_publishes_structured_unknown_for_quarantined_source(
    tmp_path: Path,
) -> None:
    measurements = tmp_path / "measurements"
    search = [
        _elementwise_measurement_run(
            measurements,
            run_id=f"issue45-silu-gate-search-{session}",
            operation="SiLU",
            shape=[1, 512, 2048],
            operand_kind="tensor",
            median_ns=10_000 + session,
            process_id=session,
            raw_samples_ns=(
                [100, 100, 10_001, 10_001, 10_001]
                if session == 1
                else None
            ),
        )
        for session in (1, 2, 3)
    ]
    holdout = [
        _elementwise_measurement_run(
            measurements,
            run_id=f"issue45-silu-gate-holdout-{session}",
            operation="SiLU",
            shape=[1, 512, 2048],
            operand_kind="tensor",
            median_ns=10_010 + session,
            process_id=session + 3,
        )
        for session in (1, 2, 3)
    ]
    manifest = json.loads((search[0] / "run.manifest.json").read_text())
    assert manifest["observation_validity"]["timing_quality"] == "quarantined"
    assert verify_run_bundle(search[0])["passed"] is True

    run = OperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="issue45-silu-gate-quarantined-v1",
        qualification_policy=_exact_elementwise_policy(
            operation="SiLU",
            shape=[1, 512, 2048],
            operand_kind="tensor",
        ),
        search_runs=search,
        holdout_runs=holdout,
        confirmation_runs=[],
        query_sizes=[],
        query_shapes=({"result": [1, 512, 2048]},),
    )

    verification = verify_run_bundle(run)
    assert verification["passed"] is True, verification["failures"]
    qualification = json.loads((run / "frontier/qualification.json").read_text())
    assert qualification["status"] == "unknown"
    assert qualification["reason_code"] == "exact-shape-source-observation-invalid"
    assert qualification["minimum_next_evidence_boundary"]["kind"] == (
        "replace-invalid-independent-sessions"
    )
    query = diagnose_run_bundle(run)["capability_surface_queries"][0]
    assert query["status"] == "unknown"
    assert query["reason_code"] == "exact-shape-source-observation-invalid"


def test_elementwise_frontier_rejects_source_without_authoritative_host_lock(
    tmp_path: Path,
) -> None:
    measurements = tmp_path / "measurements"
    search = [
        _elementwise_measurement_run(
            measurements,
            run_id=f"issue45-add-lock-search-{session}",
            operation="Add",
            shape=[1, 512, 512],
            operand_kind="tensor-tensor",
            median_ns=10_000 + session,
            process_id=session,
        )
        for session in (1, 2, 3)
    ]
    holdout = [
        _elementwise_measurement_run(
            measurements,
            run_id=f"issue45-add-lock-holdout-{session}",
            operation="Add",
            shape=[1, 512, 512],
            operand_kind="tensor-tensor",
            median_ns=10_010 + session,
            process_id=session + 3,
        )
        for session in (1, 2, 3)
    ]
    _remove_host_lock_metadata(search[0])
    assert verify_run_bundle(search[0])["passed"] is True

    with pytest.raises(
        OperatorFrontierQualificationError,
        match="authoritative Ascend host lock",
    ):
        OperatorFrontierBundleWriter().run(
            tmp_path / "frontier",
            run_id="issue45-add-missing-lock-v1",
            qualification_policy=_exact_elementwise_policy(
                operation="Add",
                shape=[1, 512, 512],
                operand_kind="tensor-tensor",
            ),
            search_runs=search,
            holdout_runs=holdout,
            confirmation_runs=[],
            query_sizes=[],
            query_shapes=({"result": [1, 512, 512]},),
        )


def test_collection_plan_covers_all_five_demo_domains_and_uses_issue_scoped_ids(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    policy = yaml.safe_load(
        (root / "specs/policies/ascend-910b2-elementwise-exact-frontier-v1.yaml").read_text()
    )
    assert {domain["domain_id"] for domain in policy["domains"]} == {
        "add-broadcast-mask",
        "add-residual",
        "mul-attention-scale",
        "mul-mlp-gate",
        "silu-mlp-gate",
    }
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "commands.log"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$ISSUE45_COMMAND_LOG\"\n"
    )
    fake_python.chmod(0o755)
    owner_file = tmp_path / "ascend.owner"
    owner_file.write_text("issue=45 pid=1 host=test started=2026-08-13T00:00:00Z\n")
    completed = subprocess.run(
        [
            "bash",
            str(
                root
                / "goal_process/issue-45-ascend-elementwise-frontier/collect_elementwise_evidence.sh"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GROUNDUPSCALE_ISSUE45_WORKSPACE": str(tmp_path / "workspace"),
            "GROUNDUPSCALE_NPU_PYTHON": str(fake_python),
            "GROUNDUPSCALE_ISSUE45_SESSION_ID": "test-session",
            "GROUNDUPSCALE_NPU_LOCK_OWNER_FILE": str(owner_file),
            "ISSUE45_COMMAND_LOG": str(log),
        },
    )
    assert completed.returncode == 0, completed.stderr
    commands = log.read_text().splitlines()
    assert len(commands) == 30
    assert all("--run-id issue45-" in command for command in commands)
    assert all("-test-session-" in command for command in commands)
    assert all("--warmup 20 --repetitions 100 --inner-iterations 100" in command for command in commands)


def test_inventory_preserves_all_indexed_elementwise_stable_paths() -> None:
    root = Path(__file__).parents[1]
    inventory = yaml.safe_load(
        (
            root
            / "goal_process/issue-45-ascend-elementwise-frontier/elementwise-stable-paths.yaml"
        ).read_text()
    )
    cost_path = root / inventory["source_run_bundle"] / "ir/cost.ir.json"
    assert sha256(cost_path.read_bytes()).hexdigest() == inventory[
        "source_cost_ir_sha256"
    ]
    cost_ir = json.loads(cost_path.read_text())

    def walk(value: object):
        if isinstance(value, dict):
            yield value
            for item in value.values():
                yield from walk(item)
        elif isinstance(value, list):
            for item in value:
                yield from walk(item)

    actual = {
        item["stable_path"]
        for item in walk(cost_ir)
        if item.get("operation") in {"Add", "Mul", "SiLU"}
        and isinstance(item.get("stable_path"), str)
    }
    declared = {
        path
        for domain in inventory["domains"].values()
        for path in domain["stable_paths"]
    }
    assert actual == declared
    assert len(declared) == 12
    assert any("layer_0" in path for path in declared)
    assert any("layer_1" in path for path in declared)


def test_two_layer_elementwise_frontiers_bind_stable_paths_into_schedule(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    inventory_path = (
        root
        / "goal_process/issue-45-ascend-elementwise-frontier/elementwise-stable-paths.yaml"
    )
    inventory = yaml.safe_load(inventory_path.read_text())
    inventory["hardware_cohort"] = "ascend-npu-febd831c8d07e06f"
    inventory_path = tmp_path / "elementwise-stable-paths.yaml"
    inventory_path.write_text(yaml.safe_dump(inventory, sort_keys=False))
    measurements = tmp_path / "measurements"
    frontiers: dict[str, Path] = {}
    for domain_id, domain in inventory["domains"].items():
        search = [
            _elementwise_measurement_run(
                measurements,
                run_id=f"issue45-{domain_id}-schedule-search-{session}",
                operation=domain["operation"],
                shape=domain["result_shape"],
                operand_kind=domain["operand_kind"],
                median_ns=10_000 + session,
                process_id=session,
            )
            for session in (1, 2, 3)
        ]
        holdout = [
            _elementwise_measurement_run(
                measurements,
                run_id=f"issue45-{domain_id}-schedule-holdout-{session}",
                operation=domain["operation"],
                shape=domain["result_shape"],
                operand_kind=domain["operand_kind"],
                median_ns=10_010 + session,
                process_id=session + 3,
            )
            for session in (1, 2, 3)
        ]
        frontiers[domain_id] = OperatorFrontierBundleWriter().run(
            tmp_path / "frontiers",
            run_id=f"issue45-{domain_id}-schedule-v1",
            qualification_policy=_exact_elementwise_policy(
                operation=domain["operation"],
                shape=domain["result_shape"],
                operand_kind=domain["operand_kind"],
            ),
            search_runs=search,
            holdout_runs=holdout,
            confirmation_runs=[],
            query_sizes=[],
            query_shapes=({"result": domain["result_shape"]},),
        )

    from groundupscale.backends.ascend_910b2 import (
        compose_elementwise_frontier_schedule,
    )

    result = compose_elementwise_frontier_schedule(
        inventory_path, frontiers
    )

    assert result["status"] == "qualified"
    assert result["schedule"]["kind"] == "serialized"
    assert result["schedule"]["selected_duration_ns"] == pytest.approx(
        12 * 10_012
    )
    assert len(result["leaves"]) == 12
    assert len({leaf["stable_path"] for leaf in result["leaves"]}) == 12
    assert all(leaf["status"] == "exact-anchor" for leaf in result["leaves"])
    assert all(leaf["provisional_estimate_ns"] is None for leaf in result["leaves"])
    assert any("layer_0" in leaf["stable_path"] for leaf in result["leaves"])
    assert any("layer_1" in leaf["stable_path"] for leaf in result["leaves"])

    silu = inventory["domains"]["silu-mlp-gate"]
    unstable_search = [
        _elementwise_measurement_run(
            measurements,
            run_id=f"issue45-silu-unknown-search-{session}",
            operation="SiLU",
            shape=silu["result_shape"],
            operand_kind=silu["operand_kind"],
            median_ns=median_ns,
            process_id=10 + session,
        )
        for session, median_ns in enumerate((10_000, 10_100, 13_000), start=1)
    ]
    unstable_holdout = [
        _elementwise_measurement_run(
            measurements,
            run_id=f"issue45-silu-unknown-holdout-{session}",
            operation="SiLU",
            shape=silu["result_shape"],
            operand_kind=silu["operand_kind"],
            median_ns=10_200 + session,
            process_id=13 + session,
        )
        for session in (1, 2, 3)
    ]
    frontiers["silu-mlp-gate"] = OperatorFrontierBundleWriter().run(
        tmp_path / "frontiers",
        run_id="issue45-silu-schedule-unknown-v1",
        qualification_policy=_exact_elementwise_policy(
            operation="SiLU",
            shape=silu["result_shape"],
            operand_kind=silu["operand_kind"],
        ),
        search_runs=unstable_search,
        holdout_runs=unstable_holdout,
        confirmation_runs=[],
        query_sizes=[],
        query_shapes=({"result": silu["result_shape"]},),
    )

    incomplete = compose_elementwise_frontier_schedule(inventory_path, frontiers)

    assert incomplete["status"] == "unknown"
    assert incomplete["schedule"]["selected_duration_ns"] is None
    silu_leaves = [leaf for leaf in incomplete["leaves"] if leaf["domain_id"] == "silu-mlp-gate"]
    assert {leaf["status"] for leaf in silu_leaves} == {"unknown"}
    assert {leaf["reason_code"] for leaf in silu_leaves} == {
        "exact-shape-session-repeatability-failed"
    }
    assert all(leaf["duration_ns"] is None for leaf in silu_leaves)
