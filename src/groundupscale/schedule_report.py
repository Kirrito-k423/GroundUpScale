"""Human-readable projection of a composed Schedule Frontier result."""

from __future__ import annotations

from typing import Mapping

from groundupscale.schedule_evidence import (
    SCHEDULE_FRONTIER_RESULT_SCHEMA,
    ScheduleFrontierError,
    finite_nonnegative,
)
from groundupscale.schedule_input import AXIS_NAMES, QUALIFICATION_GATES


def _format_ms(duration_ns: object) -> str:
    if not finite_nonnegative(duration_ns):
        raise ScheduleFrontierError("invalid-report-duration")
    return f"{float(duration_ns) / 1_000_000:.6f}"


def render_schedule_frontier_report(result: Mapping[str, object]) -> str:
    """Project a composed result without changing evidence qualification."""

    fixture = result.get("fixture")
    axes = result.get("axes")
    ledger = result.get("ledger")
    qualification = result.get("evidence_qualification")
    counterfactuals = result.get("counterfactuals")
    if (
        result.get("schema") != SCHEDULE_FRONTIER_RESULT_SCHEMA
        or not isinstance(fixture, dict)
        or not isinstance(axes, dict)
        or not isinstance(ledger, dict)
        or ledger.get("status") not in {"conserved", "unknown"}
        or not isinstance(qualification, dict)
        or not isinstance(counterfactuals, list)
    ):
        raise ScheduleFrontierError("invalid-schedule-frontier-result")
    classification = fixture.get("classification")
    if not isinstance(classification, list):
        raise ScheduleFrontierError("invalid-schedule-frontier-result")
    real_hardware_claim = fixture.get("real_hardware_claim")
    if real_hardware_claim is not None and not isinstance(real_hardware_claim, str):
        raise ScheduleFrontierError("invalid-schedule-frontier-result")

    lines = [
        "Schedule Frontier diagnostic",
        "Evidence: "
        + ", ".join(str(item) for item in classification)
        + "; promotion-eligible="
        + str(fixture.get("promotion_eligible")).lower()
        + "; real-hardware-claim="
        + (real_hardware_claim or "none"),
    ]
    labels = {
        "resource_physical_floor": "Resource Physical Floor",
        "operator_achievable_frontier": "Operator Achievable Frontier",
        "schedule_achievable_frontier": "Schedule Achievable Frontier",
        "observation": "Observation",
    }
    for name in AXIS_NAMES:
        axis = axes.get(name)
        if not isinstance(axis, dict):
            raise ScheduleFrontierError("invalid-schedule-frontier-result")
        line = (
            f"{labels[name]}: {axis.get('status', 'unknown')}; "
            f"fixture-only={_format_ms(axis.get('fixture_duration_ns'))} ms"
        )
        if name == "operator_achievable_frontier":
            aggregation = axis.get("aggregation")
            if not isinstance(aggregation, dict):
                raise ScheduleFrontierError("invalid-schedule-frontier-result")
            line += (
                f"; {aggregation.get('node_count')}-node aggregate critical "
                "path; not a single MatMul"
            )
        lines.append(line)
    if ledger["status"] == "unknown":
        lines.append(f"Ledger: unknown ({ledger.get('reason_code', 'unknown')})")
    else:
        other_leaf_ns = (
            ledger["leaf_total_ns"] - ledger["operation_leaf_total_ns"]
        )
        lines.append(
            "Ledger: "
            f"{_format_ms(ledger['operation_leaf_total_ns'])} ms operation leaves + "
            f"{_format_ms(other_leaf_ns)} ms other exclusive leaves + "
            f"{_format_ms(ledger['residual']['duration_ns'])} ms unattributed "
            f"residual = {_format_ms(ledger['reconciled_total_ns'])} ms E2E; "
            "parent spans are index-only"
        )
    for counterfactual in counterfactuals:
        if not isinstance(counterfactual, dict):
            raise ScheduleFrontierError("invalid-schedule-frontier-result")
        lines.append(
            f"{counterfactual.get('transformation_id')}: "
            f"recovered={_format_ms(counterfactual.get('recovered_ns'))} ms; "
            "counterfactual E2E="
            f"{_format_ms(counterfactual.get('counterfactual_e2e_ns'))} ms; "
            "Operator Frontier unchanged="
            + _format_ms(
                counterfactual["operator_achievable_frontier_ns"]["after"]
            )
            + " ms"
        )
    gates = qualification.get("gates")
    if not isinstance(gates, dict):
        raise ScheduleFrontierError("invalid-schedule-frontier-result")
    for gate_name in QUALIFICATION_GATES:
        gate = gates.get(gate_name)
        if not isinstance(gate, dict):
            raise ScheduleFrontierError("invalid-schedule-frontier-result")
        lines.append(
            f"{gate_name}: {gate.get('status', 'unknown')}; "
            f"reason={gate.get('reason_code', 'none')}"
        )
    lines.append("Real M4 values are not produced by this fixture.")
    return "\n".join(lines) + "\n"


__all__ = ["render_schedule_frontier_report"]
