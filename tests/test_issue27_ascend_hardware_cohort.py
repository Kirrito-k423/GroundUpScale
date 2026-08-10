from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from types import ModuleType

import pytest

from groundupscale.measurement_contract import (
    COHORT_IDENTITY_DIMENSIONS,
    HardwareValidityIdentity,
    MeasurementCapabilityManifest,
    ObservationFieldStatus,
    TimerEvidence,
)


EVIDENCE_PATH = (
    Path(__file__).parents[1]
    / "goal_process/issue-27-ascend-hardware-cohort/evidence"
    / "ascend-910b2-matmul-cohort-20260810-v5.json"
)
COLLECTOR_PATH = EVIDENCE_PATH.parents[1] / "collect_ascend_matmul.py"


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _load_evidence() -> dict[str, object]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _source_session_paths() -> list[Path]:
    evidence = _load_evidence()
    return [
        EVIDENCE_PATH.parent / source["path"]
        for source in evidence["source_artifacts"]
    ]


def _load_collector_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "issue27_collect_ascend_matmul", COLLECTOR_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_ascend_probe_freezes_complete_hardware_cohort() -> None:
    evidence = _load_evidence()

    assert evidence["schema"] == (
        "groundupscale.dev/ascend-exact-shape-hardware-cohort-evidence/"
        "v1alpha1"
    )
    assert evidence["ticket"] == 27
    assert evidence["status"] == "completed"
    assert evidence["scope"] == {
        "kind": "real-device-evidence-only",
        "formal_measurement_adapter": "out-of-scope-issue-28",
        "frontier_promotion": "not-evaluated",
    }

    hardware = evidence["hardware"]
    assert hardware["device_name"] == "910B2"
    assert hardware["device_version"] == "V1"
    assert hardware["logical_device"] == "npu:0"
    assert hardware["physical_device"] == {"npu_id": 0, "chip_id": 0}
    assert hardware["pcie_bus_id"]
    assert hardware["vdie_id"]
    assert hardware["hbm_capacity_bytes"] == 65_536 * 1024 * 1024
    assert hardware["fallback_detected"] is False
    assert hardware["sources"]
    assert hardware["device_selection"] == {
        "environment_value": "0",
        "environment_variable": "ASCEND_RT_VISIBLE_DEVICES",
        "host_mapping_record": {
            "chip_id": 0,
            "chip_logic_id": 0,
            "chip_name": "Ascend 910B2",
            "npu_id": 0,
        },
        "logical_device": "npu:0",
        "logical_device_index": 0,
        "mapping_source": (
            "artifact://sessions/issue27-s52.json#command_snapshots/"
            "npu_mapping"
        ),
        "physical_chip_id": 0,
        "physical_npu_id": 0,
        "runtime_current_device": 0,
        "runtime_device_name": "Ascend910B2",
        "visible_chip_logic_id": 0,
    }

    software = evidence["software"]
    for field in (
        "os",
        "kernel",
        "driver",
        "firmware",
        "cann",
        "python",
        "torch",
        "torch_npu",
    ):
        assert software[field]["value"]
        assert software[field]["source"]

    cohort_identity = evidence["cohort_identity"]
    HardwareValidityIdentity.from_document(cohort_identity)
    assert evidence["cohort_digest"] == _canonical_digest(cohort_identity)
    assert evidence["cohort_id"].endswith(evidence["cohort_digest"][:16])
    assert cohort_identity["partition"] == (
        "physical-npu-0/chip-0/chip-logic-0/ascend-rt-visible-devices=0"
    )
    assert evidence["cohort_repeatability"] == {
        "identity_rule": (
            "canonical-sha256-of-all-stable-cohort-dimensions"
        ),
        "independent_process_sessions": 3,
        "scope": "cohort-identity-only",
        "status": "identity-stable",
        "unique_cohort_digests": [evidence["cohort_digest"]],
    }
    timing_repeatability = evidence["timing_repeatability"]
    assert timing_repeatability["status"] == "passed"
    assert timing_repeatability["independent_process_sessions"] == 3
    assert timing_repeatability["excluded_samples"] == 0
    assert timing_repeatability["maximum_session_iqr_fraction_of_median"] == 0.1
    assert (
        timing_repeatability["observed_maximum_session_median_deviation_fraction"]
        <= timing_repeatability["maximum_session_median_deviation_fraction"]
        == 0.05
    )
    assert timing_repeatability["frontier_qualification"] == (
        "not-evaluated-by-issue-27"
    )

    assert cohort_identity["power_clock"]["power_policy"].startswith(
        "unsupported("
    )


def test_real_sessions_close_npu_boundary_and_reproduce_raw_timing() -> None:
    evidence = _load_evidence()
    sessions = evidence["sessions"]

    assert len(sessions) >= 3
    assert {session["cohort_digest"] for session in sessions} == {
        evidence["cohort_digest"]
    }
    assert len({session["session_id"] for session in sessions}) == len(sessions)
    assert len(
        {
            _canonical_digest(session["process_identity"])
            for session in sessions
        }
    ) == len(sessions)

    for session in sessions:
        timer = TimerEvidence.from_documents(
            session["timer"], session["completion_boundary"]
        )
        assert timer.source == "torch.npu.Event.elapsed_time"
        assert timer.resolution_ns == 20.0
        assert session["timer"]["resolution_kind"] == (
            "empirically-observed-output-step"
        )
        resolution_evidence = session["timer"]["resolution_evidence"]
        assert resolution_evidence["documented_api_output_unit"] == (
            "milliseconds-float"
        )
        assert resolution_evidence["method"] == (
            "gcd-of-distinct-integer-nanosecond-event-sample-differences"
        )
        assert resolution_evidence["observed_resolution_ns"] == 20
        assert resolution_evidence["distinct_observed_values"] >= 90
        assert resolution_evidence["difference_count"] == (
            resolution_evidence["distinct_observed_values"] - 1
        )
        source_evidence = session["timer"]["source_evidence"]
        assert source_evidence["installed_source_path"].endswith(
            "torch_npu/npu/streams.py"
        )
        assert len(source_evidence["installed_source_sha256"]) == 64
        assert source_evidence["official_api_reference"].startswith(
            "https://www.hiascend.com/"
        )
        assert session["warmup"]["iterations"] == 20
        assert session["warmup"]["synchronized"] is True

        raw_samples_ns = session["raw_samples_ns"]
        assert len(raw_samples_ns) == 100
        assert all(sample > 0 for sample in raw_samples_ns)
        assert all(sample % int(timer.resolution_ns) == 0 for sample in raw_samples_ns)
        assert session["summary_ns"]["median"] == statistics.median(
            raw_samples_ns
        )
        assert session["measurement_policy"] == {
            "frontier_qualification": "not-evaluated-by-issue-27",
            "maximum_session_iqr_fraction_of_median": 0.1,
            "maximum_session_median_deviation_fraction": 0.05,
            "policy_id": "issue27-raw-timing-repeatability-v1",
            "repetitions": 100,
            "sample_exclusion": "none-preserve-all-raw-samples",
            "warmup_iterations": 20,
        }
        assert session["timing_quality"]["status"] == "passed"
        assert session["timing_quality"]["excluded_samples"] == 0
        assert session["summary_ns"]["iqr_fraction_of_median"] <= 0.1

        correctness = session["correctness"]
        assert correctness["status"] == "passed"
        assert correctness["oracle"] == "cpu-float64-matmul"
        assert correctness["candidate_device"] == "npu:0"
        assert correctness["cpu_fallback"] is False
        assert correctness["shape_exact"] is True
        assert correctness["finite"] is True
        assert session["environment"]["power_policy_status"] == "unsupported"
        assert session["environment"]["frontier_eligibility"] == (
            "ineligible-missing-power-policy-evidence"
        )


def test_npu_mapping_snapshot_is_parsed_into_physical_selection() -> None:
    collector = _load_collector_module()
    snapshot = """
NPU ID     Chip ID     Chip Logic ID     Chip Name
0          0           0                 Ascend 910B2
0          1           -                 Mcu
1          0           1                 Ascend 910B2
"""

    assert collector.parse_npu_mapping(snapshot, npu_id=0, chip_id=0) == {
        "npu_id": 0,
        "chip_id": 0,
        "chip_logic_id": 0,
        "chip_name": "Ascend 910B2",
    }
    with pytest.raises(collector.ProbeBlocked):
        collector.parse_npu_mapping(snapshot, npu_id=7, chip_id=0)

    remapped = collector.parse_npu_mapping(
        snapshot.replace(
            "0          0           0                 Ascend 910B2",
            "0          0           7                 Ascend 910B2",
        ),
        npu_id=0,
        chip_id=0,
    )
    collector.require_visible_device_mapping(remapped, "7")
    with pytest.raises(collector.ProbeBlocked):
        collector.require_visible_device_mapping(remapped, "0")


def test_capability_manifest_preserves_field_status_and_sources() -> None:
    evidence = _load_evidence()
    manifest = MeasurementCapabilityManifest.from_document(
        evidence["measurement_capability_manifest"],
        adapter_id="issue27-one-off-ascend-probe",
        cohort_id=evidence["cohort_id"],
    )

    assert manifest.primary_timer_available is True
    fields = {field.name: field.to_document() for field in manifest.fields}
    assert {
        "timer.primary",
        "timer.host_visible_completion",
        "synchronization.device_stream",
        "memory.framework",
        "memory.hbm_device_wide",
        "power.device_wide",
        "power.policy",
        "frequency.hbm",
        "frequency.ai_core",
        "transfer.h2d",
        "transfer.d2h",
        "attribution.device_kernel",
        "profiling.operator_timeline",
        "counter.ai_core",
    } <= set(fields)

    unavailable = {
        ObservationFieldStatus.UNSUPPORTED.value,
        ObservationFieldStatus.PERMISSION_DENIED.value,
        ObservationFieldStatus.NOT_REQUESTED.value,
        ObservationFieldStatus.NOT_APPLICABLE.value,
        ObservationFieldStatus.COLLECTION_FAILED.value,
        ObservationFieldStatus.UNKNOWN.value,
    }
    for field in fields.values():
        assert field["source"]
        assert field["scope"]
        assert field["attribution"]
        assert field["intrusion"]
        if field["status"] in unavailable:
            assert "value" not in field


def test_every_stable_identity_dimension_changes_the_cohort_digest() -> None:
    identity = _load_evidence()["cohort_identity"]
    baseline_digest = _canonical_digest(identity)

    for dimension in COHORT_IDENTITY_DIMENSIONS:
        changed = deepcopy(identity)
        value = changed[dimension]
        if isinstance(value, dict):
            value["issue27_identity_probe"] = "changed"
        else:
            changed[dimension] = f"{value}-changed"
        assert _canonical_digest(changed) != baseline_digest


def test_frozen_source_sessions_are_digest_verifiable_and_secret_free() -> None:
    evidence = _load_evidence()
    expected_digest = evidence.pop("evidence_digest")
    assert expected_digest == _canonical_digest(evidence)

    forbidden_key_fragments = ("password", "private_key", "secret", "token")
    for source in evidence["source_artifacts"]:
        path = EVIDENCE_PATH.parent / source["path"]
        content = path.read_bytes()
        assert sha256(content).hexdigest() == source["sha256"]
        session = json.loads(content)
        assert session["schema"] == source["schema"]
        assert session["session_id"] == source["session_id"]

        def assert_no_secret_keys(value: object) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    assert not any(
                        fragment in key.lower()
                        for fragment in forbidden_key_fragments
                    )
                    assert_no_secret_keys(item)
            elif isinstance(value, list):
                for item in value:
                    assert_no_secret_keys(item)

        assert_no_secret_keys(session)


def test_missing_npu_runtime_fails_closed_with_structured_blocked_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "blocked.json"
    environment = dict(os.environ)
    environment["PATH"] = ""

    completed = subprocess.run(
        [
            sys.executable,
            str(COLLECTOR_PATH),
            "collect",
            "--session-id",
            "forced-unavailable",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    blocked = json.loads(output.read_text(encoding="utf-8"))
    assert blocked["status"] == "blocked"
    assert blocked["reason_codes"]
    assert "hardware" not in blocked


def _run_merge(output: Path, *sessions: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(COLLECTOR_PATH),
            "merge",
            "--output",
            str(output),
            *(str(session) for session in sessions),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_merge_rejects_duplicate_sessions_as_non_independent(
    tmp_path: Path,
) -> None:
    session = _source_session_paths()[0]
    output = tmp_path / "duplicate.json"

    completed = _run_merge(output, session, session, session)

    assert completed.returncode == 2
    assert not output.exists()


def test_merge_recomputes_cohort_digest_instead_of_trusting_session(
    tmp_path: Path,
) -> None:
    session_paths = _source_session_paths()
    tampered = json.loads(session_paths[0].read_text(encoding="utf-8"))
    tampered["cohort_identity"]["software"] += ";tampered=true"
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    output = tmp_path / "tampered-merge.json"

    completed = _run_merge(output, tampered_path, *session_paths[1:])

    assert completed.returncode == 2
    assert not output.exists()


def test_merge_never_overwrites_immutable_evidence(tmp_path: Path) -> None:
    session_paths = _source_session_paths()
    output = tmp_path / "existing.json"
    output.write_text("immutable-sentinel\n", encoding="utf-8")

    completed = _run_merge(output, *session_paths)

    assert completed.returncode == 2
    assert output.read_text(encoding="utf-8") == "immutable-sentinel\n"


def test_merge_rejects_incomplete_frozen_measurement_contract(
    tmp_path: Path,
) -> None:
    session_paths = _source_session_paths()
    original = json.loads(session_paths[0].read_text(encoding="utf-8"))
    mutations = {
        "short-warmup": lambda session: session["warmup"].update(
            {"iterations": 9}
        ),
        "short-samples": lambda session: session["raw_samples_ns"].pop(),
        "false-summary": lambda session: session["summary_ns"].update(
            {"median": session["summary_ns"]["median"] + 1}
        ),
        "incomplete-correctness": lambda session: session["correctness"].update(
            {"shape_exact": False}
        ),
    }

    for name, mutate in mutations.items():
        tampered = deepcopy(original)
        mutate(tampered)
        tampered_path = tmp_path / f"{name}.json"
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
        output = tmp_path / f"{name}-merge.json"

        completed = _run_merge(output, tampered_path, *session_paths[1:])

        assert completed.returncode == 2, name
        assert not output.exists(), name


def test_merge_rejects_high_dispersion_session(tmp_path: Path) -> None:
    collector = _load_collector_module()
    session_paths = _source_session_paths()
    noisy = json.loads(session_paths[0].read_text(encoding="utf-8"))
    noisy_samples = [80_000 + 20 * index for index in range(50)] + [
        800_000 + 20 * index for index in range(50)
    ]
    noisy["raw_samples_ns"] = noisy_samples
    noisy["raw_elapsed_ms"] = [sample / 1_000_000 for sample in noisy_samples]
    noisy["summary_ns"] = collector.timing_summary(noisy_samples)
    noisy_path = tmp_path / "noisy.json"
    noisy_path.write_text(json.dumps(noisy), encoding="utf-8")
    output = tmp_path / "noisy-merge.json"

    completed = _run_merge(output, noisy_path, *session_paths[1:])

    assert completed.returncode == 2
    assert not output.exists()


def test_merge_recomputes_power_policy_status_from_raw_snapshot(
    tmp_path: Path,
) -> None:
    session_paths = _source_session_paths()
    tampered = json.loads(session_paths[0].read_text(encoding="utf-8"))
    tampered["environment"]["power_policy_status"] = "declared"
    tampered["environment"]["frontier_eligibility"] = "eligible"
    tampered_path = tmp_path / "forged-power-policy.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    output = tmp_path / "forged-power-policy-merge.json"

    completed = _run_merge(output, tampered_path, *session_paths[1:])

    assert completed.returncode == 2
    assert not output.exists()


def test_merge_rejects_excessive_cross_session_median_deviation(
    tmp_path: Path,
) -> None:
    collector = _load_collector_module()
    session_paths = _source_session_paths()
    shifted = json.loads(session_paths[0].read_text(encoding="utf-8"))
    shifted_samples = [sample + 20_000 for sample in shifted["raw_samples_ns"]]
    shifted["raw_samples_ns"] = shifted_samples
    shifted["raw_elapsed_ms"] = [
        sample / 1_000_000 for sample in shifted_samples
    ]
    shifted["summary_ns"] = collector.timing_summary(shifted_samples)
    shifted["timing_quality"]["observed_iqr_fraction_of_median"] = shifted[
        "summary_ns"
    ]["iqr_fraction_of_median"]
    shifted_path = tmp_path / "shifted-session.json"
    shifted_path.write_text(json.dumps(shifted), encoding="utf-8")
    output = tmp_path / "shifted-session-merge.json"

    completed = _run_merge(output, shifted_path, *session_paths[1:])

    assert completed.returncode == 2
    assert not output.exists()


def test_merge_rebuilds_partition_from_raw_mapping(tmp_path: Path) -> None:
    session_paths = _source_session_paths()
    tampered = json.loads(session_paths[0].read_text(encoding="utf-8"))
    mapping_snapshot = tampered["command_snapshots"]["npu_mapping"]
    mapping_snapshot["stdout"] = """
NPU ID     Chip ID     Chip Logic ID     Chip Name
0          0           7                 Ascend 910B2
"""
    mapping_snapshot["stdout_sha256"] = sha256(
        mapping_snapshot["stdout"].encode("utf-8")
    ).hexdigest()
    selection = tampered["hardware"]["device_selection"]
    selection["host_mapping_record"]["chip_logic_id"] = 7
    selection["visible_chip_logic_id"] = 7
    selection["environment_value"] = "7"
    tampered_path = tmp_path / "partition-mismatch.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    output = tmp_path / "partition-mismatch-merge.json"

    completed = _run_merge(output, tampered_path, *session_paths[1:])

    assert completed.returncode == 2
    assert not output.exists()
