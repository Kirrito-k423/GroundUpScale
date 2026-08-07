"""Public Schedule Frontier composition and report facade."""

from __future__ import annotations

from typing import Any, Mapping

from groundupscale.ir.common import (
    DerivationRecord,
    canonical_data,
    derivation_identity,
)
from groundupscale.ir.semantic import ProvenanceGraph
from groundupscale.schedule_evidence import (
    SCHEDULE_FRONTIER_INPUT_SCHEMA,
    SCHEDULE_FRONTIER_RESULT_SCHEMA,
    ScheduleEvidenceUnknown,
    ScheduleFrontierError,
    canonical_digest,
)
from groundupscale.schedule_input import (
    compose_schedule_trace,
    effect_rejections,
    evidence_qualification,
    prototype_fixture,
    schedule_frontier_replay_metadata,
    selected_candidate_ids,
    untrusted_axes,
)
from groundupscale.schedule_ledger import (
    compose_counterfactuals,
    compose_ledger,
)
from groundupscale.schedule_report import (
    render_schedule_frontier_report,
)


def _derivation(
    *,
    phase: str,
    rule: str,
    fingerprint: str,
    source_path: str,
    stable_path: str,
    target_node_ids: tuple[str, ...],
    assumptions: tuple[str, ...] = (),
) -> DerivationRecord:
    return DerivationRecord(
        derivation_id=derivation_identity(rule, fingerprint, stable_path),
        phase=phase,
        rule=rule,
        source_path=source_path,
        source_stable_path=stable_path,
        target_node_ids=target_node_ids,
        assumptions=assumptions,
        warnings=("prototype-only-untrusted-fixture",),
    )


def compose_schedule_frontier(
    document: Mapping[str, object],
) -> dict[str, Any]:
    """Compose one schedule diagnostic without promoting prototype evidence."""

    if document.get("schema") != SCHEDULE_FRONTIER_INPUT_SCHEMA:
        raise ScheduleFrontierError("unsupported-schedule-frontier-schema")
    fixture = prototype_fixture(document)
    axes = untrusted_axes(document)
    qualification = evidence_qualification(document, axes)
    try:
        schedule_trace = compose_schedule_trace(document, axes)
    except ScheduleEvidenceUnknown as unknown:
        axes["schedule_achievable_frontier"]["status"] = "unknown"
        axes["schedule_achievable_frontier"][
            "reason_code"
        ] = unknown.reason_code
        schedule_trace = {
            **schedule_frontier_replay_metadata(document),
            "status": "unknown",
            "reason_code": unknown.reason_code,
            **unknown.context,
        }

    planner_candidate_ids = selected_candidate_ids(document)
    try:
        ledger = compose_ledger(document, axes, planner_candidate_ids)
    except ScheduleFrontierError as error:
        if (
            schedule_trace.get("reason_code")
            != "execution-event-selection-incomplete"
            or str(error) != "ledger-operation-candidate-not-selected"
        ):
            raise
        ledger = {
            "status": "unknown",
            "reason_code": "ledger-selection-evidence-incomplete",
        }
    if ledger["status"] == "conserved":
        counterfactuals, counterfactual_records = compose_counterfactuals(
            document, axes, ledger
        )
    else:
        counterfactuals = []
        counterfactual_records = []
    frontier = document.get("schedule_frontier")
    execution_ir = document.get("execution_ir")
    graph_owner = (
        frontier.get("frontier_id") if isinstance(frontier, dict) else "unknown"
    )
    execution_ir_id = (
        execution_ir.get("execution_ir_id")
        if isinstance(execution_ir, dict)
        else "unknown"
    )
    fingerprint = (
        schedule_trace["frontier_identity"]["input_digest"]
        if schedule_trace.get("status") != "unknown"
        else canonical_digest(document)
    )
    physical_events = (
        execution_ir.get("physical_events")
        if isinstance(execution_ir, dict)
        else []
    )
    derivation_records: list[DerivationRecord] = []
    if schedule_trace.get("status") != "unknown":
        derivation_records.extend(
            (
                _derivation(
                    phase="schedule-candidate-selection",
                    rule="planner-select-candidate@1",
                    fingerprint=fingerprint,
                    source_path="ImplementationCandidates",
                    stable_path=str(execution_ir_id),
                    target_node_ids=tuple(
                        f"event:{event['event_id']}"
                        for event in physical_events
                        if isinstance(event, dict)
                        and isinstance(event.get("event_id"), str)
                    ),
                    assumptions=(
                        "selection-complete-and-exclusive",
                        *(
                            "rejected_candidate="
                            + rejection["candidate_id"]
                            + ";reason="
                            + rejection["reason_code"]
                            + ";evidence_refs="
                            + ",".join(rejection["evidence_refs"])
                            for rejection in schedule_trace[
                                "rejected_candidates"
                            ]
                        ),
                    ),
                ),
                _derivation(
                    phase="schedule-compose",
                    rule="explicit-schedule-frontier@1",
                    fingerprint=fingerprint,
                    source_path="ExecutionIR",
                    stable_path=str(graph_owner),
                    target_node_ids=(f"schedule-frontier:{graph_owner}",),
                    assumptions=("explicit-dependency-and-resource-edges",),
                ),
            )
        )
    if ledger["status"] == "conserved":
        derivation_records.append(
            _derivation(
                phase="schedule-ledger-compose",
                rule="exclusive-leaves-plus-residual@1",
                fingerprint=fingerprint,
                source_path="ObservationLedger",
                stable_path="e2e",
                target_node_ids=("schedule-ledger:e2e",),
                assumptions=("parent-spans-index-only",),
            )
        )
    derivation_records.extend(counterfactual_records)
    provenance = ProvenanceGraph(
        schema="groundupscale.dev/provenance-graph/v1alpha1",
        records=tuple(derivation_records),
    )
    provenance_document = canonical_data(provenance)
    provenance_document.update(
        {"graph_id": f"{graph_owner}/provenance", "append_only": True}
    )

    return {
        "schema": SCHEDULE_FRONTIER_RESULT_SCHEMA,
        "fixture": fixture,
        "axes": axes,
        "evidence_qualification": qualification,
        "schedule_trace": schedule_trace,
        "effect_rejections": effect_rejections(document),
        "ledger": ledger,
        "counterfactuals": counterfactuals,
        "provenance_graph": provenance_document,
    }


__all__ = [
    "ScheduleFrontierError",
    "compose_schedule_frontier",
    "render_schedule_frontier_report",
]
