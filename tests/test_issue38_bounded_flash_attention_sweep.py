from __future__ import annotations

from pathlib import Path
import os
import subprocess
from types import SimpleNamespace

import yaml


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    ROOT
    / "specs/policies/ascend-910b2-flash-attention-bounded-sequence-sweep-v1.yaml"
)
COLLECTION_SCRIPT = (
    ROOT
    / "goal_process/issue-38-ascend-flash-attention-sequence-sweep/"
    "collect_bounded_sequence_sweep.sh"
)
PUBLISH_SCRIPT = (
    ROOT
    / "goal_process/issue-38-ascend-flash-attention-sequence-sweep/"
    "publish_qualification.py"
)
UNKNOWN_EVIDENCE = (
    ROOT
    / "goal_process/issue-38-ascend-flash-attention-sequence-sweep/"
    "evidence/qualification-unknown.json"
)
PUBLISHED_BUNDLE = (
    ROOT
    / "goal_process/issue-38-ascend-flash-attention-sequence-sweep/"
    "evidence/runs/issue38-ascend-flash-attention-sequence-sweep-v2"
)


def test_issue38_collection_plan_locks_bounded_equal_length_tnd_domain() -> None:
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    scope = policy["scope"]
    plan = policy["collection_plan"]

    fixed_domain = {
        "hardware_cohort": "ascend-npu-23b93a89d5fecc79",
        "operation": "FlashAttentionForward",
        "sequence_count": 1,
        "head_count": 8,
        "head_dimension": 64,
        "dtype": "float16",
        "layout": "TND",
        "causal": False,
        "mask": "none",
        "dropout_probability": 0.0,
        "mode": "forward",
        "candidate_ids": ["torch_npu.npu_fusion_attention"],
    }
    assert {key: scope[key] for key in fixed_domain} == fixed_domain
    assert scope["anchor_sequence_lengths"] == plan[
        "main_sweep_sequence_lengths"
    ]
    assert scope["confirmation_sequence_lengths"] == plan[
        "independent_validation_sequence_lengths"
    ]
    assert {1, 128, 512, 1024, 2048, 4096, 8192}.issubset(
        set(plan["main_sweep_sequence_lengths"])
    )
    assert plan["maximum_supplemental_rounds"] == 1
    assert plan["supplemental_trigger"] == (
        "candidate-switch-residual-or-boundary-evidence-only"
    )
    assert plan["minimum_independent_process_sessions_per_lane"] == 3
    assert plan["raw_sample_retention"] == (
        "preserve-all-no-selective-exclusion"
    )
    assert plan["completion_boundary"] == (
        "end-event-synchronize-plus-device-synchronize"
    )
    assert policy["response_target"] == "latency"
    assert policy["response_kind"] == "setup-plus-throughput"


def test_measure_cli_builds_fixed_equal_length_tnd_forward_case(
    tmp_path: Path, monkeypatch
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
                """{
  "run_id": "issue38-cli-test",
  "status": "completed",
  "device": "ascend-npu",
  "hardware_cohort": "ascend-npu-23b93a89d5fecc79"
}\n""",
                encoding="utf-8",
            )
            return run

    adapter = SimpleNamespace(name="fake-ascend-adapter")
    monkeypatch.setattr(cli, "MeasurementRunBundleWriter", FakeWriter)
    monkeypatch.setattr(
        cli,
        "verify_run_bundle",
        lambda run: {"passed": True},
    )

    exit_code = cli.main(
        [
            "measure",
            "--device",
            "ascend-npu",
            "--operation",
            "FlashAttentionForward",
            "--sequence-count",
            "1",
            "--sequence-length",
            "4096",
            "--head-count",
            "8",
            "--head-dimension",
            "64",
            "--dtype",
            "float16",
            "--layout",
            "TND",
            "--candidate",
            "torch_npu.npu_fusion_attention",
            "--seed",
            "20260813",
            "--warmup",
            "20",
            "--repetitions",
            "100",
            "--inner-iterations",
            "1",
            "--artifact-store",
            str(tmp_path),
            "--run-id",
            "issue38-cli-test",
            "--json",
        ],
        measurement_adapter_factory=lambda *args, **kwargs: adapter,
    )

    assert exit_code == 0
    assert captured["adapter"] is adapter
    assert captured["case"] == {
        "schema": (
            "groundupscale.dev/exact-shape-flash-attention-tnd-case/v1alpha1"
        ),
        "operation": "FlashAttentionForward",
        "shape": {
            "sequence_count": 1,
            "sequence_lengths": [4096],
            "head_count": 8,
            "head_dimension": 64,
        },
        "dtype": "float16",
        "layout": "TND",
        "causal": False,
        "mask": "none",
        "dropout_probability": 0.0,
        "mode": "forward",
        "seed": 20260813,
        "candidate": "torch_npu.npu_fusion_attention",
        "warmup_iterations": 20,
        "repetitions": 100,
        "inner_iterations": 1,
    }


def test_collection_script_runs_only_declared_main_shapes_and_three_sessions(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "commands.log"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$ISSUE38_COMMAND_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "GROUNDUPSCALE_ISSUE38_WORKSPACE": str(workspace),
        "GROUNDUPSCALE_NPU_PYTHON": str(fake_python),
        "ISSUE38_COMMAND_LOG": str(log),
    }

    completed = subprocess.run(
        ["bash", str(COLLECTION_SCRIPT), "main"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    commands = log.read_text(encoding="utf-8").splitlines()
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    declared = policy["collection_plan"]["main_sweep_sequence_lengths"]
    assert len(commands) == len(declared) * 3
    for length in declared:
        matching = [
            command
            for command in commands
            if f"--sequence-length {length} " in f"{command} "
        ]
        assert len(matching) == 3
        assert all("--operation FlashAttentionForward" in item for item in matching)
        assert all("--sequence-count 1" in item for item in matching)
        assert all("--head-count 8 --head-dimension 64" in item for item in matching)
        assert all("--dtype float16 --layout TND" in item for item in matching)
        assert all(
            "--candidate torch_npu.npu_fusion_attention" in item
            for item in matching
        )


def test_collection_script_rejects_untriggered_supplemental_round(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        ["bash", str(COLLECTION_SCRIPT), "supplemental"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GROUNDUPSCALE_ISSUE38_WORKSPACE": str(tmp_path / "workspace"),
        },
    )

    assert completed.returncode == 2
    assert "requires a reviewed trigger" in completed.stderr


def test_qualification_publisher_uses_all_three_independent_evidence_lanes() -> None:
    source = PUBLISH_SCRIPT.read_text(encoding="utf-8")

    assert 'search_runs=_runs(evidence / "runs", "main")' in source
    assert 'holdout_runs=_runs(evidence / "runs", "holdout")' in source
    assert (
        'confirmation_runs=_runs(evidence / "runs", "validation")' in source
    )
    assert "verify_run_bundle(run)" in source
    assert "diagnose_run_bundle(run)" in source
    assert "9000" in source


def test_qualification_publisher_can_start_without_external_pythonpath() -> None:
    completed = subprocess.run(
        [str(Path(os.sys.executable)), str(PUBLISH_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
    )

    assert completed.returncode == 0, completed.stderr
    assert "--workspace" in completed.stdout


def test_qualification_publisher_replays_existing_real_bundle() -> None:
    completed = subprocess.run(
        [
            str(Path(os.sys.executable)),
            str(PUBLISH_SCRIPT),
            "--workspace",
            str(ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
    )

    assert completed.returncode == 0, completed.stderr
    summary = yaml.safe_load(completed.stdout)
    assert summary["verification_passed"] is True
    assert summary["qualification_status"] == "unknown"
    assert summary["reason_code"] == "bounded-collection-stability-failed"


def test_interrupted_real_collection_publishes_a_bounded_structured_unknown() -> None:
    evidence = yaml.safe_load(UNKNOWN_EVIDENCE.read_text(encoding="utf-8"))

    assert evidence["issue"] == 38
    assert evidence["status"] == "unknown"
    assert evidence["reason_code"] == "bounded-collection-stability-failed"
    assert evidence["hardware_cohort"] == "ascend-npu-23b93a89d5fecc79"
    assert evidence["collection"]["main"]["verified_run_bundles"] == 99
    assert evidence["collection"]["holdout"]["verified_run_bundles"] == 99
    assert evidence["collection"]["validation"]["verified_run_bundles"] == 36
    assert len(evidence["qualification_gate_failures"]) == 5
    assert evidence["stopping_decision"] == {
        "status": "stopped",
        "supplemental_rounds_executed": 0,
        "maximum_supplemental_rounds": 1,
        "additional_model_complexity_allowed": False,
        "different_hardware_cohort_allowed": False,
    }
    assert evidence["four_k_hypothesis"]["global_boundary_published"] is False
    assert all(
        query["status"] == "unknown"
        for query in evidence["representative_queries"]
    )


def test_real_unknown_qualification_bundle_is_self_contained_and_verifiable() -> None:
    from groundupscale.diagnostics import diagnose_run_bundle
    from groundupscale.run_bundle import verify_run_bundle

    verification = verify_run_bundle(PUBLISHED_BUNDLE)
    diagnosis = diagnose_run_bundle(PUBLISHED_BUNDLE)

    assert verification["passed"] is True
    assert [
        query["status"]
        for query in diagnosis["capability_surface_queries"]
    ] == ["unknown"] * 7
    assert {
        query["reason_code"]
        for query in diagnosis["capability_surface_queries"]
    } == {"bounded-collection-stability-failed"}
