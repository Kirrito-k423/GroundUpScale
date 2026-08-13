from __future__ import annotations

from groundupscale.alias_materialization import (
    build_alias_materialization_evidence,
    verify_alias_materialization_evidence,
)


COHORT = "ascend-npu-issue46-cohort"
STABLE_PATH = "semantic/demo/layer_0/attention/q_transpose"


def _contract(*, shape: list[int], stride: list[int]) -> dict[str, object]:
    return {
        "device": "npu:0",
        "dtype": "float32",
        "shape": shape,
        "stride": stride,
        "layout": "strided",
        "is_contiguous": False,
    }


def test_verified_alias_candidate_is_the_only_evidence_backed_zero() -> None:
    evidence = build_alias_materialization_evidence(
        alias_audits=[
            {
                "stable_path": STABLE_PATH,
                "operation": "Transpose",
                "input_storage_identity": "storage:shared",
                "output_storage_identity": "storage:shared",
                "input_contract": _contract(
                    shape=[1, 512, 8, 64], stride=[262144, 512, 64, 1]
                ),
                "output_contract": _contract(
                    shape=[1, 8, 512, 64], stride=[262144, 64, 512, 1]
                ),
            }
        ],
        expected_operations=[
            {
                "stable_path": STABLE_PATH,
                "operation": "Transpose",
                "logical_read_bytes": 1_048_576,
                "logical_write_bytes": 1_048_576,
            }
        ],
        selected_candidates={STABLE_PATH: "torch.transpose.npu.eager"},
        execution_mode="pytorch-eager",
        hardware_cohort=COHORT,
    )

    assert evidence["status"] == "qualified"
    assert evidence["hardware_cohort"] == COHORT
    operation = evidence["operations"][0]
    assert operation["stable_path"] == STABLE_PATH
    assert operation["selected_candidate"] == {
        "candidate_id": "torch.transpose.npu.eager",
        "execution_mode": "pytorch-eager",
        "hardware_cohort": COHORT,
    }
    assert operation["alias_audit"]["input_storage_identity"] == "storage:shared"
    assert operation["alias_audit"]["output_storage_identity"] == "storage:shared"
    assert operation["decision"] == "alias-preserving"
    assert operation["resource_demand"] == {
        "status": "known",
        "memory_read_bytes": 0,
        "memory_write_bytes": 0,
    }
    assert operation["duration"] == {
        "status": "known",
        "value_ns": 0,
        "evidence_kind": "verified-alias-preserving-candidate",
    }
    assert operation["physical_event"] is None
    assert len(evidence["evidence_version_id"]) == 64
    assert verify_alias_materialization_evidence(evidence) == {
        "passed": True,
        "failures": [],
    }


def test_materializing_candidate_emits_scheduled_physical_event_and_demand() -> None:
    evidence = build_alias_materialization_evidence(
        alias_audits=[
            {
                "stable_path": STABLE_PATH,
                "operation": "Transpose",
                "input_storage_identity": "storage:input",
                "output_storage_identity": "storage:materialized-output",
                "input_contract": _contract(
                    shape=[1, 512, 8, 64], stride=[262144, 512, 64, 1]
                ),
                "output_contract": _contract(
                    shape=[1, 8, 512, 64], stride=[262144, 32768, 64, 1]
                ),
            }
        ],
        expected_operations=[
            {
                "stable_path": STABLE_PATH,
                "operation": "Transpose",
                "logical_read_bytes": 1_048_576,
                "logical_write_bytes": 1_048_576,
            }
        ],
        selected_candidates={
            STABLE_PATH: {
                "candidate_id": "transpose-contiguous.npu.eager",
                "duration_ns": 18_400,
                "evidence_refs": ["run://issue46-materialization/timing"],
            }
        },
        execution_mode="pytorch-eager",
        hardware_cohort=COHORT,
    )

    operation = evidence["operations"][0]
    assert operation["decision"] == "materialization"
    assert operation["resource_demand"] == {
        "status": "known",
        "memory_read_bytes": 1_048_576,
        "memory_write_bytes": 1_048_576,
    }
    assert operation["duration"] == {
        "status": "known",
        "value_ns": 18_400,
        "evidence_refs": ["run://issue46-materialization/timing"],
    }
    event = operation["physical_event"]
    assert event["event_id"].startswith("physical-event:")
    assert event["kind"] == "materialization"
    assert event["duration_ns"] == 18_400
    assert event["resource_claims"] == [
        {
            "resource_id": "memory.interface",
            "kind": "throughput",
            "read_bytes": 1_048_576,
            "write_bytes": 1_048_576,
            "lifetime": {
                "start": event["event_id"],
                "end": event["event_id"],
            },
            "provenance": {
                "evidence_refs": ["run://issue46-materialization/timing"]
            },
        }
    ]
    assert evidence["schedule"]["physical_events"] == [event]
    assert evidence["decomposition"] == {
        "materialization_duration_ns": 18_400,
        "alias_duration_ns": 0,
        "unknown_stable_paths": [],
    }
    assert verify_alias_materialization_evidence(evidence)["passed"] is True


def test_missing_alias_audit_is_structured_unknown_and_a_new_evidence_version() -> None:
    aliased = build_alias_materialization_evidence(
        alias_audits=[
            {
                "stable_path": STABLE_PATH,
                "operation": "Transpose",
                "input_storage_identity": "storage:shared",
                "output_storage_identity": "storage:shared",
                "input_contract": _contract(
                    shape=[1, 512, 8, 64], stride=[262144, 512, 64, 1]
                ),
                "output_contract": _contract(
                    shape=[1, 8, 512, 64], stride=[262144, 64, 512, 1]
                ),
            }
        ],
        expected_operations=[
            {
                "stable_path": STABLE_PATH,
                "operation": "Transpose",
                "logical_read_bytes": 1_048_576,
                "logical_write_bytes": 1_048_576,
            }
        ],
        selected_candidates={STABLE_PATH: "torch.transpose.npu.eager"},
        execution_mode="pytorch-eager",
        hardware_cohort=COHORT,
    )
    missing = build_alias_materialization_evidence(
        alias_audits=[],
        expected_operations=[
            {
                "stable_path": STABLE_PATH,
                "operation": "Transpose",
                "logical_read_bytes": 1_048_576,
                "logical_write_bytes": 1_048_576,
            }
        ],
        selected_candidates={STABLE_PATH: "torch.transpose.npu.eager"},
        execution_mode="pytorch-eager",
        hardware_cohort=COHORT,
    )

    assert missing["status"] == "unknown"
    operation = missing["operations"][0]
    assert operation["decision"] == "unknown"
    assert operation["alias_audit"] is None
    assert operation["resource_demand"] == {"status": "unknown"}
    assert operation["duration"] == {"status": "unknown", "value_ns": None}
    assert operation["physical_event"] is None
    assert missing["decomposition"]["unknown_stable_paths"] == [STABLE_PATH]
    assert missing["evidence_version_id"] != aliased["evidence_version_id"]
    assert verify_alias_materialization_evidence(missing)["passed"] is True


def test_verifier_rejects_zero_invented_without_an_alias_audit() -> None:
    evidence = build_alias_materialization_evidence(
        alias_audits=[],
        expected_operations=[
            {
                "stable_path": STABLE_PATH,
                "operation": "Transpose",
                "logical_read_bytes": 1_048_576,
                "logical_write_bytes": 1_048_576,
            }
        ],
        selected_candidates={STABLE_PATH: "torch.transpose.npu.eager"},
        execution_mode="pytorch-eager",
        hardware_cohort=COHORT,
    )
    evidence["operations"][0].update(
        {
            "decision": "alias-preserving",
            "resource_demand": {
                "status": "known",
                "memory_read_bytes": 0,
                "memory_write_bytes": 0,
            },
            "duration": {
                "status": "known",
                "value_ns": 0,
                "evidence_kind": "verified-alias-preserving-candidate",
            },
        }
    )

    verification = verify_alias_materialization_evidence(evidence)

    assert verification["passed"] is False
    assert "evidence version digest mismatch" in verification["failures"]
    assert "unverified alias zero" in verification["failures"]
