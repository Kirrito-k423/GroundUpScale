from __future__ import annotations

import json
from pathlib import Path

import pytest

from groundupscale.schedule_frontier import (
    compose_schedule_frontier,
    render_schedule_frontier_report,
)


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "issue20-synthetic-transformer-schedule.json"
)


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_synthetic_transformer_schedule_preserves_four_untrusted_axes() -> None:
    result = compose_schedule_frontier(_fixture())

    assert result["fixture"] == {
        "source_issue": "#7",
        "classification": ["aggregated", "prototype-only", "untrusted"],
        "promotion_eligible": False,
        "real_hardware_claim": None,
    }
    assert list(result["axes"]) == [
        "resource_physical_floor",
        "operator_achievable_frontier",
        "schedule_achievable_frontier",
        "observation",
    ]
    assert {
        name: axis["fixture_duration_ns"]
        for name, axis in result["axes"].items()
    } == {
        "resource_physical_floor": 5_553_976,
        "operator_achievable_frontier": 51_632_000,
        "schedule_achievable_frontier": 53_232_000,
        "observation": 92_814_479,
    }
    assert all(axis["status"] == "unknown" for axis in result["axes"].values())
    assert result["axes"]["resource_physical_floor"]["reason_code"] == (
        "prototype-only-untrusted-fixture"
    )
    assert result["axes"]["operator_achievable_frontier"]["aggregation"] == {
        "kind": "aggregate-critical-path",
        "node_count": 24,
    }


def test_schedule_uses_declared_candidate_overlap_and_explicit_edges() -> None:
    result = compose_schedule_frontier(_fixture())

    trace = result["schedule_trace"]
    assert trace["policy"] == {
        "policy_id": "fixture://issue-20/explicit-schedule-policy",
        "version": "1",
        "schedule": "dependency-only",
    }
    op_01 = next(
        candidate
        for candidate in trace["implementation_candidates"]
        if candidate["candidate_id"] == "op-01"
    )
    assert op_01["local_duration_ns"] == 2_000_000
    assert op_01["resource_composition"] == {
        "kind": "explicit-overlap",
        "claim_ids": ["op-01-compute", "op-01-memory"],
    }
    assert trace["operator_aggregate_ns"] == 51_632_000
    assert trace["composition"] == {
        "serialized_duration_ns": 53_232_000,
        "critical_path_duration_ns": 53_232_000,
        "resource_duration_ns": 51_632_000,
        "ideal_dag_duration_ns": 53_232_000,
        "selected_duration_ns": 53_232_000,
        "limiting_resource": "compute.fp32",
    }
    assert trace["critical_path_event_ids"] == [
        "dispatch-1",
        *[f"op-{index:02d}" for index in range(1, 13)],
        "dispatch-2",
        *[f"op-{index:02d}" for index in range(13, 25)],
    ]
    assert {edge["kind"] for edge in trace["dependency_path"]} == {
        "semantic-data",
        "execution-order",
        "execution-resource",
    }
    assert result["axes"]["operator_achievable_frontier"][
        "fixture_duration_ns"
    ] == trace["operator_aggregate_ns"]
    assert result["axes"]["schedule_achievable_frontier"][
        "fixture_duration_ns"
    ] == trace["composition"]["selected_duration_ns"]


def test_implicit_schedule_effects_are_rejected_with_specific_reasons() -> None:
    document = _fixture()
    document["requested_effects"] = [
        {"effect_id": "fuse", "kind": "fusion"},
        {"effect_id": "concurrent", "kind": "concurrency"},
        {"effect_id": "mask", "kind": "communication-masking"},
        {"effect_id": "contend", "kind": "contention"},
        {"effect_id": "dispatch", "kind": "dispatch"},
    ]

    result = compose_schedule_frontier(document)

    assert result["effect_rejections"] == [
        {
            "effect_id": "fuse",
            "kind": "fusion",
            "status": "rejected",
            "reason_code": "fusion-requires-explicit-group",
        },
        {
            "effect_id": "concurrent",
            "kind": "concurrency",
            "status": "rejected",
            "reason_code": "concurrency-requires-explicit-execution-semantics",
        },
        {
            "effect_id": "mask",
            "kind": "communication-masking",
            "status": "rejected",
            "reason_code": "communication-masking-requires-explicit-event-pair",
        },
        {
            "effect_id": "contend",
            "kind": "contention",
            "status": "rejected",
            "reason_code": "contention-requires-explicit-resource-claim",
        },
        {
            "effect_id": "dispatch",
            "kind": "dispatch",
            "status": "rejected",
            "reason_code": "dispatch-requires-explicit-transformation",
        },
    ]


def test_candidate_local_max_is_unknown_without_declared_overlap() -> None:
    document = _fixture()
    op_01 = next(
        candidate
        for candidate in document["implementation_candidates"]
        if candidate["candidate_id"] == "op-01"
    )
    del op_01["resource_composition"]
    selected_op_01 = next(
        event
        for event in document["execution_ir"]["physical_events"]
        if event["candidate"]["candidate_id"] == "op-01"
    )
    del selected_op_01["candidate"]["resource_composition"]

    result = compose_schedule_frontier(document)

    trace = result["schedule_trace"]
    assert {
        key: trace[key]
        for key in ("status", "reason_code", "candidate_id")
    } == {
        "status": "unknown",
        "reason_code": "candidate-resource-composition-undeclared",
        "candidate_id": "op-01",
    }
    assert trace["frontier_identity"]["frontier_id"] == (
        "fixture://issue-20/two-layer-transformer-frontier"
    )
    assert trace["frontier_identity"]["version"] == "1"
    assert len(trace["frontier_identity"]["input_digest"]) == 64
    assert len(trace["frontier_identity"]["evidence_digest"]) == 64
    assert trace["uncertainty"]["status"] == "unknown"
    assert trace["evidence_refs"] == [
        "fixture://issue-7/schedule-frontier-input"
    ]
    assert trace["frontier_identity"] == compose_schedule_frontier(document)[
        "schedule_trace"
    ]["frontier_identity"]
    assert trace["frontier_identity"] != compose_schedule_frontier(_fixture())[
        "schedule_trace"
    ]["frontier_identity"]
    assert result["axes"]["schedule_achievable_frontier"]["status"] == "unknown"
    assert result["axes"]["schedule_achievable_frontier"]["reason_code"] == (
        "candidate-resource-composition-undeclared"
    )
    assert result["axes"]["operator_achievable_frontier"][
        "fixture_duration_ns"
    ] == 51_632_000
    assert result["ledger"]["status"] == "conserved"
    assert result["ledger"]["reconciled_total_ns"] == 92_814_479
    assert result["counterfactuals"][0]["counterfactual_e2e_ns"] == 80_814_479
    report = render_schedule_frontier_report(result)
    assert (
        "Schedule Achievable Frontier: unknown; fixture-only=53.232000 ms"
    ) in report
    assert "Ledger: 51.100746 ms operation leaves" in report


def test_exclusive_leaves_and_residual_conserve_e2e_without_parent_counting() -> None:
    result = compose_schedule_frontier(_fixture())

    ledger = result["ledger"]
    assert ledger["status"] == "conserved"
    assert ledger["leaf_semantics"] == "mutually-exclusive"
    assert ledger["e2e_duration_ns"] == 92_814_479
    assert ledger["operation_leaf_total_ns"] == 51_100_746
    assert ledger["leaf_total_ns"] == 73_100_746
    assert ledger["residual"] == {
        "residual_id": "unattributed-residual",
        "kind": "unattributed",
        "duration_ns": 19_713_733,
        "evidence_refs": ["fixture://issue-7/unattributed-residual"],
    }
    assert ledger["reconciled_total_ns"] == 92_814_479
    assert ledger["leaf_identity_conservation"] == {
        "unique_leaf_count": 28,
        "duplicate_leaf_ids": [],
        "unassigned_leaf_ids": [],
    }
    assert all(parent["additive"] is False for parent in ledger["parents"])
    assert ledger["parent_span_total_included_ns"] == 0
    assert result["axes"]["observation"]["fixture_duration_ns"] == ledger[
        "reconciled_total_ns"
    ]


def test_batched_dispatch_counterfactual_only_recovers_declared_bucket() -> None:
    result = compose_schedule_frontier(_fixture())

    counterfactual = result["counterfactuals"][0]
    assert counterfactual["transformation_id"] == (
        "batched-dispatch-counterfactual"
    )
    assert counterfactual["kind"] == "batched-dispatch"
    assert counterfactual["status"] == "conserved"
    assert counterfactual["baseline_e2e_ns"] == 92_814_479
    assert counterfactual["counterfactual_e2e_ns"] == 80_814_479
    assert counterfactual["recovered_ns"] == 12_000_000
    assert counterfactual["removed_leaf_ids"] == [
        "dispatch-leaf-1",
        "dispatch-leaf-2",
    ]
    assert counterfactual["leaf_identity_conservation"] == {
        "before_count": 28,
        "preserved_count": 26,
        "removed_leaf_ids": ["dispatch-leaf-1", "dispatch-leaf-2"],
        "added_leaf_ids": [],
    }
    assert counterfactual["operation_leaf_total_ns"] == {
        "before": 51_100_746,
        "after": 51_100_746,
    }
    assert counterfactual["operator_achievable_frontier_ns"] == {
        "before": 51_632_000,
        "after": 51_632_000,
    }
    assert counterfactual["axis_mutations"] == []


def test_missing_baseline_preflight_and_anchors_remain_unknown() -> None:
    result = compose_schedule_frontier(_fixture())

    assert result["evidence_qualification"] == {
        "status": "unknown",
        "gates": {
            "baseline_timing_lane": {
                "status": "unknown",
                "reason_code": "missing-qualified-baseline-timing-lane",
            },
            "preflight": {
                "status": "unknown",
                "reason_code": "missing-passed-preflight",
            },
            "operator_frontier_anchors": {
                "status": "unknown",
                "reason_code": (
                    "missing-qualified-active-operator-frontier-anchors"
                ),
            },
        },
    }
    assert result["axes"]["resource_physical_floor"]["status"] == "unknown"
    assert result["axes"]["operator_achievable_frontier"]["status"] == "unknown"
    assert result["axes"]["operator_achievable_frontier"]["reason_code"] == (
        "missing-qualified-active-operator-frontier-anchors"
    )
    assert result["axes"]["schedule_achievable_frontier"]["status"] == "unknown"
    assert result["axes"]["schedule_achievable_frontier"]["reason_code"] == (
        "operator-frontier-unknown"
    )
    assert result["axes"]["observation"]["status"] == "unknown"
    assert result["axes"]["observation"]["reason_code"] == (
        "missing-qualified-baseline-timing-lane"
    )
    assert result["fixture"]["real_hardware_claim"] is None


def test_report_is_a_projection_of_the_same_untrusted_schedule_result() -> None:
    result = compose_schedule_frontier(_fixture())

    report = render_schedule_frontier_report(result)

    assert (
        "Evidence: aggregated, prototype-only, untrusted; "
        "promotion-eligible=false; real-hardware-claim=none"
    ) in report
    assert (
        "Resource Physical Floor: unknown; fixture-only=5.553976 ms"
    ) in report
    assert (
        "Operator Achievable Frontier: unknown; fixture-only=51.632000 ms; "
        "24-node aggregate critical path; not a single MatMul"
    ) in report
    assert (
        "Schedule Achievable Frontier: unknown; fixture-only=53.232000 ms"
    ) in report
    assert "Observation: unknown; fixture-only=92.814479 ms" in report
    assert (
        "Ledger: 51.100746 ms operation leaves + 22.000000 ms other "
        "exclusive leaves + 19.713733 ms unattributed residual = "
        "92.814479 ms E2E; parent spans are index-only"
    ) in report
    assert (
        "batched-dispatch-counterfactual: recovered=12.000000 ms; "
        "counterfactual E2E=80.814479 ms; Operator Frontier "
        "unchanged=51.632000 ms"
    ) in report
    assert "Real M4 values are not produced by this fixture." in report
    assert "missing-qualified-baseline-timing-lane" in report
    assert "missing-passed-preflight" in report
    assert "missing-qualified-active-operator-frontier-anchors" in report
    assert "Diagnostic Trigger" not in report
    assert "Performance Diagnosis Verdict" not in report


def test_frontier_only_schedules_planner_selected_execution_events() -> None:
    document = _fixture()

    result = compose_schedule_frontier(document)

    trace = result["schedule_trace"]
    assert trace["execution_ir"] == {
        "execution_ir_id": (
            "fixture://issue-20/two-layer-transformer-execution-ir"
        ),
        "schema": "groundupscale.dev/execution-ir/v1alpha1",
        "state": "unscheduled",
        "evidence_refs": ["fixture://issue-7/planner-selection"],
    }
    assert len(document["implementation_candidates"]) == 27
    assert len(trace["implementation_candidates"]) == 26
    assert len(trace["selected_events"]) == 26
    assert trace["rejected_candidates"] == [
        {
            "candidate_id": "op-01-alternative",
            "reason_code": "not-selected-by-planner",
            "evidence_refs": ["fixture://issue-7/op-01-alternative-rejection"],
        }
    ]
    assert all(
        event["candidate_id"]
        == trace["implementation_candidates"][index]["candidate_id"]
        for index, event in enumerate(trace["selected_events"])
    )
    assert all(
        event["duration_model"]
        == {"kind": "candidate-local-resource-composition"}
        for event in document["execution_ir"]["physical_events"]
    )
    assert all(
        event["candidate"]["resource_claims"]
        for event in document["execution_ir"]["physical_events"]
    )
    assert "semantic_dependencies" in document["execution_ir"]
    assert "execution_dependencies" in document["execution_ir"]

    incomplete = _fixture()
    incomplete["execution_ir"]["physical_events"].pop()
    incomplete_result = compose_schedule_frontier(incomplete)
    incomplete_trace = incomplete_result["schedule_trace"]
    assert {
        key: incomplete_trace[key]
        for key in ("status", "reason_code", "candidate_id")
    } == {
        "status": "unknown",
        "reason_code": "execution-event-selection-incomplete",
        "candidate_id": "op-24",
    }
    assert "Ledger: unknown (ledger-selection-evidence-incomplete)" in (
        render_schedule_frontier_report(incomplete_result)
    )
    assert "schedule-candidate-selection" not in {
        record["phase"]
        for record in incomplete_result["provenance_graph"]["records"]
    }

    duplicate = _fixture()
    duplicate["execution_ir"]["physical_events"][-1]["candidate"] = duplicate[
        "execution_ir"
    ]["physical_events"][-2]["candidate"]
    with pytest.raises(
        ValueError, match="duplicate-implementation-candidate-selection"
    ):
        compose_schedule_frontier(duplicate)


def test_resource_claims_are_typed_and_missing_semantics_fail_closed() -> None:
    result = compose_schedule_frontier(_fixture())

    op_01 = next(
        candidate
        for candidate in result["schedule_trace"][
            "implementation_candidates"
        ]
        if candidate["candidate_id"] == "op-01"
    )
    compute_claim = op_01["resource_claims"][0]
    assert compute_claim == {
        "claim_id": "op-01-compute",
        "resource_id": "compute.fp32",
        "kind": "throughput",
        "work_or_amount": {"value": 1, "unit": "synthetic-work-unit"},
        "duration_ns": 2_000_000,
        "allocation_bounds": {
            "minimum": 1.0,
            "maximum": 1.0,
            "unit": "normalized-share",
        },
        "sharing": "declared-shared-throughput",
        "lifetime": {"start": "event-start", "end": "event-end"},
        "affinity": {"kind": "resource-id", "value": "compute.fp32"},
        "provenance": {
            "evidence_refs": ["fixture://issue-7/op-01-compute"]
        },
    }

    missing_provenance = _fixture()
    missing_claim = missing_provenance["implementation_candidates"][1][
        "resource_claims"
    ][0]
    del missing_claim["provenance"]
    missing_result = compose_schedule_frontier(missing_provenance)
    assert {
        key: missing_result["schedule_trace"][key]
        for key in ("status", "reason_code", "claim_id")
    } == {
        "status": "unknown",
        "reason_code": "resource-claim-provenance-missing",
        "claim_id": "op-01-compute",
    }

    unsupported_kind = _fixture()
    capacity_claim = unsupported_kind["implementation_candidates"][1][
        "resource_claims"
    ][0]
    capacity_claim["kind"] = "capacity"
    selected_capacity_claim = unsupported_kind["execution_ir"][
        "physical_events"
    ][1]["candidate"]["resource_claims"][0]
    selected_capacity_claim["kind"] = "capacity"
    capacity_result = compose_schedule_frontier(unsupported_kind)
    assert {
        key: capacity_result["schedule_trace"][key]
        for key in ("status", "reason_code", "claim_id")
    } == {
        "status": "unknown",
        "reason_code": "resource-kind-not-duration-bearing",
        "claim_id": "op-01-compute",
    }


def test_ledger_rejects_unknown_operation_candidate_and_invalid_parent_graph() -> None:
    unknown_candidate = _fixture()
    unknown_candidate["ledger"]["leaves"][1]["candidate_id"] = "not-selected"
    with pytest.raises(
        ValueError, match="ledger-operation-candidate-not-selected"
    ):
        compose_schedule_frontier(unknown_candidate)

    cyclic = _fixture()
    cyclic["ledger"]["parents"][1]["child_parent_ids"] = ["e2e"]
    with pytest.raises(ValueError, match="ledger-parent-graph-cycle"):
        compose_schedule_frontier(cyclic)

    duplicate_root = _fixture()
    duplicate_root["ledger"]["parents"].append(
        {
            "span_id": "second-e2e",
            "kind": "e2e",
            "additive": False,
            "child_parent_ids": [],
            "leaf_ids": [],
        }
    )
    with pytest.raises(ValueError, match="ledger-requires-one-e2e-root"):
        compose_schedule_frontier(duplicate_root)

    unreachable = _fixture()
    unreachable["ledger"]["parents"][0]["child_parent_ids"] = ["layer-1"]
    with pytest.raises(ValueError, match="ledger-parent-not-reachable-from-e2e"):
        compose_schedule_frontier(unreachable)


def test_frontier_replay_preserves_identity_digests_uncertainty_and_all_edges() -> None:
    document = _fixture()

    result = compose_schedule_frontier(document)
    replay = compose_schedule_frontier(document)

    trace = result["schedule_trace"]
    assert trace["frontier_identity"]["frontier_id"] == (
        "fixture://issue-20/two-layer-transformer-frontier"
    )
    assert trace["frontier_identity"]["version"] == "1"
    assert len(trace["frontier_identity"]["input_digest"]) == 64
    assert len(trace["frontier_identity"]["evidence_digest"]) == 64
    assert trace["frontier_identity"] == replay["schedule_trace"][
        "frontier_identity"
    ]
    assert trace["uncertainty"] == {
        "status": "unknown",
        "reason_code": "prototype-only-untrusted-fixture",
        "evidence_refs": ["fixture://issue-7/uncertainty-unavailable"],
    }
    assert trace["evidence_refs"] == [
        "fixture://issue-7/schedule-frontier-input"
    ]
    assert trace["explicit_edges"] == [
        *document["execution_ir"]["semantic_dependencies"],
        *document["execution_ir"]["execution_dependencies"],
    ]
    assert trace["transformations"] == document["transformations"]
    assert trace["overlap_claims"] == [
        {
            "candidate_id": "op-01",
            "composition": {
                "kind": "explicit-overlap",
                "claim_ids": ["op-01-compute", "op-01-memory"],
            },
            "evidence_refs": ["fixture://issue-7/op-01"],
        }
    ]

    changed = _fixture()
    changed["execution_ir"]["execution_dependencies"][0][
        "edge_id"
    ] = "changed-edge-id"
    changed_result = compose_schedule_frontier(changed)
    assert changed_result["schedule_trace"]["frontier_identity"][
        "input_digest"
    ] != trace["frontier_identity"]["input_digest"]

    requested = _fixture()
    requested["requested_effects"] = [
        {"effect_id": "fuse", "kind": "fusion"}
    ]
    requested_result = compose_schedule_frontier(requested)
    assert requested_result["schedule_trace"]["frontier_identity"][
        "evidence_digest"
    ] != trace["frontier_identity"]["evidence_digest"]


def test_edges_and_concurrency_require_direct_evidence() -> None:
    document = _fixture()
    assert all(
        edge["evidence_refs"]
        for edge in [
            *document["execution_ir"]["semantic_dependencies"],
            *document["execution_ir"]["execution_dependencies"],
        ]
    )

    missing_edge_evidence = _fixture()
    del missing_edge_evidence["execution_ir"]["execution_dependencies"][0][
        "evidence_refs"
    ]
    missing_edge_result = compose_schedule_frontier(missing_edge_evidence)
    assert {
        key: missing_edge_result["schedule_trace"][key]
        for key in ("status", "reason_code")
    } == {
        "status": "unknown",
        "reason_code": "dependency-edge-evidence-missing",
    }

    missing_group_evidence = _fixture()
    missing_group_evidence["execution_ir"]["concurrency_groups"] = [
        {"group_id": "declared", "event_ids": ["op-01", "op-02"]}
    ]
    missing_group_result = compose_schedule_frontier(missing_group_evidence)
    assert {
        key: missing_group_result["schedule_trace"][key]
        for key in ("status", "reason_code")
    } == {
        "status": "unknown",
        "reason_code": "concurrency-group-evidence-missing",
    }


def test_batched_dispatch_emits_immutable_derivation_and_provenance_graph() -> None:
    result = compose_schedule_frontier(_fixture())

    counterfactual = result["counterfactuals"][0]
    graph = result["provenance_graph"]
    assert graph["schema"] == "groundupscale.dev/provenance-graph/v1alpha1"
    assert graph["append_only"] is True
    assert [record["phase"] for record in graph["records"]] == [
        "schedule-candidate-selection",
        "schedule-compose",
        "schedule-ledger-compose",
        "schedule-counterfactual",
    ]
    selection_record = graph["records"][0]
    assert (
        "rejected_candidate=op-01-alternative;"
        "reason=not-selected-by-planner;"
        "evidence_refs=fixture://issue-7/op-01-alternative-rejection"
    ) in selection_record["assumptions"]
    record = graph["records"][-1]
    assert counterfactual["derivation_record_id"] == record["derivation_id"]
    assert record["rule"] == "batched-dispatch@1"
    assert record["source_path"] == "ScheduleLedger"
    assert record["target_node_ids"] == [
        "counterfactual:batched-dispatch-counterfactual"
    ]
    assert "removed_leaf_ids=dispatch-leaf-1,dispatch-leaf-2" in record[
        "assumptions"
    ]
    assert record == compose_schedule_frontier(_fixture())[
        "provenance_graph"
    ]["records"][-1]
