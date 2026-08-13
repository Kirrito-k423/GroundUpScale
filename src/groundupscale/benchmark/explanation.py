"""Queryable explanation graph and a small standalone HTML reader."""

from __future__ import annotations

import html
import json
from typing import Any

from groundupscale.ir import CostProgram, HardwareBackendPrediction, canonical_data


def build_explanation_graph(
    cost: CostProgram,
    benchmark: dict[str, Any],
    trace: dict[str, Any],
    live_set: dict[str, Any],
    hardware_prediction: HardwareBackendPrediction | None = None,
    comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    alignment_by_node: dict[str, list[str]] = {}
    for entry in trace["alignment_map"]["entries"]:
        for node_id in entry["compiled_node_ids"]:
            alignment_by_node.setdefault(node_id, []).append(entry["span_id"])

    for case in benchmark["cases"]:
        latency_id = f"metric:latency:{case['case_id']}"
        throughput_id = f"metric:throughput:{case['case_id']}"
        scope_id = f"scope:{case['resolved_scope']}"
        nodes.extend(
            [
                {
                    "id": latency_id,
                    "kind": "metric",
                    "name": "median latency",
                    "value": case["latency"]["median_ns"],
                    "unit": "ns",
                    "source": "benchmark-observation",
                },
                {
                    "id": throughput_id,
                    "kind": "metric",
                    "name": "throughput",
                    "value": case["latency"]["throughput_per_second"],
                    "unit": "invocations/s",
                    "source": "derived-from-latency",
                },
                {
                    "id": scope_id,
                    "kind": "scope",
                    "stable_path": case["resolved_scope"],
                    "case_id": case["case_id"],
                },
            ]
        )
        edges.extend(
            [
                {"source": latency_id, "target": scope_id, "kind": "measures"},
                {"source": throughput_id, "target": latency_id, "kind": "derived_from"},
            ]
        )

    for operation in cost.walk_operations():
        cost_id = f"cost:{operation.node_id}"
        formula_id = f"formula:{operation.node_id}"
        nodes.extend(
            [
                {
                    "id": cost_id,
                    "kind": "cost-operation",
                    "stable_path": operation.stable_path,
                    "operation": operation.operation,
                    "flops": operation.metrics.flops,
                    "logical_read_bytes": operation.metrics.logical_read_bytes,
                    "logical_write_bytes": operation.metrics.logical_write_bytes,
                    "semantic_node_id": operation.semantic_node_id,
                },
                {
                    "id": formula_id,
                    "kind": "formula",
                    "rule_id": operation.formula.rule_id,
                    "expression": operation.formula.flops_expression,
                    "assumptions": list(operation.formula.assumptions),
                },
            ]
        )
        edges.append({"source": cost_id, "target": formula_id, "kind": "explained_by"})
        for span_id in alignment_by_node.get(operation.semantic_node_id, []):
            span_node_id = f"span:{span_id}"
            edges.append({"source": cost_id, "target": span_node_id, "kind": "observed_as"})

    for event in trace["events"]:
        nodes.append(
            {
                "id": f"span:{event['span_id']}",
                "kind": "observation-span",
                "stable_path": event["stable_path"],
                "duration_ns": event["host_duration_ns"],
                "clock_domain": event["clock_domain"],
                "compiled_node_id": event["compiled_node_id"],
            }
        )

    peak_id = "metric:predicted-framework-peak-memory"
    peak_scope_id = f"scope:{live_set['peak_operation_stable_path']}"
    nodes.append(
        {
            "id": peak_id,
            "kind": "metric",
            "name": "predicted framework peak memory",
            "value": live_set["predicted_framework_peak_bytes"],
            "unit": "bytes",
            "source": "semantic-live-set",
            "exclusions": live_set["exclusions"],
        }
    )
    if not any(node["id"] == peak_scope_id for node in nodes):
        nodes.append(
            {
                "id": peak_scope_id,
                "kind": "scope",
                "stable_path": live_set["peak_operation_stable_path"],
            }
        )
    edges.append({"source": peak_id, "target": peak_scope_id, "kind": "peaks_at"})
    hardware_duration_entrypoints: list[str] = []
    if hardware_prediction is not None:
        backend_id = f"backend:{hardware_prediction.backend_id}"
        hardware_floor_id = "metric:hardware-empirical-time-floor"
        missing_compute_id = "capability:missing-fp32-flops-per-second"
        nodes.extend(
            [
                {
                    "id": backend_id,
                    "kind": "hardware-backend",
                    "backend_id": hardware_prediction.backend_id,
                    "backend_version": hardware_prediction.backend_version,
                    "placement": hardware_prediction.placement,
                    "status": hardware_prediction.status,
                    "prediction_complete": hardware_prediction.prediction_complete,
                },
                {
                    "id": hardware_floor_id,
                    "kind": "metric",
                    "name": "algorithm-independent empirical hardware floor",
                    "value": hardware_prediction.program_bounds.empirical_hardware_floor_ns,
                    "provisional_estimate_ns": hardware_prediction.program_bounds.provisional_estimate_ns,
                    "provisional_evidence_tier": hardware_prediction.program_bounds.provisional_evidence_tier,
                    "provisional_reason_codes": list(
                        hardware_prediction.program_bounds.provisional_reason_codes
                    ),
                    "unit": "ns",
                    "source": "hardware-backend-prediction",
                    "status": hardware_prediction.status,
                    "is_full_duration_prediction": False,
                    "minimum_work_flops": hardware_prediction.program_bounds.flops,
                    "compulsory_bytes": hardware_prediction.program_bounds.compulsory_bytes,
                    "materialized_bytes": hardware_prediction.program_bounds.materialized_bytes,
                    "compute_time_ns": hardware_prediction.program_bounds.empirical_compute_time_ns,
                    "memory_time_ns": hardware_prediction.program_bounds.empirical_memory_time_ns,
                    "schedule": hardware_prediction.program_bounds.schedule,
                    "serialized_hardware_floor_ns": hardware_prediction.program_bounds.serialized_hardware_floor_ns,
                    "critical_path_hardware_floor_ns": hardware_prediction.program_bounds.critical_path_hardware_floor_ns,
                    "resource_hardware_floor_ns": hardware_prediction.program_bounds.resource_hardware_floor_ns,
                    "resource_physical_floor_ns": hardware_prediction.program_bounds.resource_physical_floor_ns,
                    "ideal_dag_hardware_floor_ns": hardware_prediction.program_bounds.ideal_dag_hardware_floor_ns,
                    "limiting_resource": hardware_prediction.program_bounds.limiting_resource,
                    "resource_limiting_resource": hardware_prediction.program_bounds.resource_limiting_resource,
                    "formula": hardware_prediction.program_bounds.formula,
                    "assumptions": list(
                        hardware_prediction.program_bounds.assumptions
                    ),
                },
                {
                    "id": missing_compute_id,
                    "kind": "missing-capability",
                    "name": "FP32 compute throughput",
                    "status": hardware_prediction.program_bounds.compute_time.status,
                    "reason": hardware_prediction.program_bounds.compute_time.reason,
                    "value_ns": (
                        hardware_prediction.program_bounds.compute_time.value_ns
                    ),
                    "required_capability": (
                        hardware_prediction.program_bounds.compute_time.required_capability
                    ),
                    "evidence": canonical_data(
                        hardware_prediction.program_bounds.compute_time.evidence
                    ),
                },
            ]
        )
        edges.extend(
            [
                {
                    "source": hardware_floor_id,
                    "target": backend_id,
                    "kind": "estimated_by",
                },
                {
                    "source": backend_id,
                    "target": missing_compute_id,
                    "kind": "vendor_theory_unavailable",
                },
            ]
        )
        for capability in hardware_prediction.measured_capabilities:
            capability_id = f"capability:measured:{capability.resource}"
            nodes.append(
                {
                    "id": capability_id,
                    "kind": "measured-hardware-capability",
                    "resource": capability.resource,
                    "unit": capability.unit,
                    "robust_achievable_rate": capability.robust_achievable_rate,
                    "optimistic_rate": capability.optimistic_rate,
                    "selected_probe": capability.selected_robust_probe,
                    "hardware_cohort": capability.hardware_cohort,
                    "environment_eligible": capability.environment_eligible,
                    "source_path": capability.source_path,
                    "source_sha256": capability.source_sha256,
                }
            )
            edges.append(
                {
                    "source": hardware_floor_id,
                    "target": capability_id,
                    "kind": "bounded_by",
                }
            )
        hardware_duration_entrypoints.append(hardware_floor_id)
        for candidate in hardware_prediction.candidates:
            candidate_id = f"implementation:{candidate.candidate_id}"
            nodes.append(
                {
                    "id": candidate_id,
                    "kind": "implementation-candidate",
                    "stable_path": candidate.stable_path,
                    "operation": candidate.operation,
                    "implementation": candidate.implementation,
                    "flops": candidate.flops,
                    "compulsory_bytes": candidate.compulsory_bytes,
                    "materialized_bytes": candidate.materialized_bytes,
                    "duration": canonical_data(candidate.duration),
                    "phase_schedule": canonical_data(candidate.phase_schedule),
                    "provisional_phase_schedule": canonical_data(
                        candidate.provisional_phase_schedule
                    ),
                }
            )
            edges.extend(
                [
                    {
                        "source": f"cost:{candidate.cost_node_id}",
                        "target": candidate_id,
                        "kind": "lowered_to",
                    },
                    {
                        "source": candidate_id,
                        "target": backend_id,
                        "kind": "provided_by",
                    },
                ]
            )
            if candidate.phase_schedule is not None:
                phase_node_ids = {
                    phase.phase_id: (
                        f"operator-phase:{candidate.candidate_id}:{phase.phase_id}"
                    )
                    for phase in candidate.phase_schedule.phases
                }
                for phase in candidate.phase_schedule.phases:
                    phase_node_id = phase_node_ids[phase.phase_id]
                    nodes.append(
                        {
                            "id": phase_node_id,
                            "kind": "operator-phase",
                            "candidate_id": candidate_id,
                            "phase_id": phase.phase_id,
                            "phase_name": phase.phase_name,
                            "operation_class": phase.operation_class,
                            "status": phase.status,
                            "minimum_flops": phase.minimum_flops,
                            "logical_read_bytes": (
                                phase.logical_read_bytes
                            ),
                            "logical_write_bytes": (
                                phase.logical_write_bytes
                            ),
                            "required_compute_capability": phase.required_compute_capability,
                            "required_memory_capability": phase.required_memory_capability,
                            "compute_time_ns": phase.compute_time_ns,
                            "memory_time_ns": phase.memory_time_ns,
                            "resource_composition": phase.resource_composition,
                            "overlap_evidence_refs": list(phase.overlap_evidence_refs),
                            "capability_evidence_refs": list(
                                phase.capability_evidence_refs
                            ),
                            "local_hardware_floor_ns": (
                                phase.local_hardware_floor_ns
                            ),
                            "limiting_resource": phase.limiting_resource,
                            "missing_capabilities": list(phase.missing_capabilities),
                        }
                    )
                    edges.append(
                        {
                            "source": candidate_id,
                            "target": phase_node_id,
                            "kind": "composed_of_phase",
                        }
                    )
                    edges.extend(
                        {
                            "source": phase_node_id,
                            "target": phase_node_ids[predecessor_id],
                            "kind": "phase_depends_on",
                        }
                        for predecessor_id in phase.predecessor_phase_ids
                    )
    comparison_entrypoints: list[str] = []
    if comparison is not None:
        for item in comparison["latency_cases"]:
            case_id = item["case_id"]
            predicted_id = f"metric:empirical-hardware-floor:{case_id}"
            comparison_id = f"comparison:latency:{case_id}"
            observed_id = f"metric:latency:{case_id}"
            nodes.extend(
                [
                    {
                        "id": predicted_id,
                        "kind": "metric",
                        "name": "scope empirical hardware floor",
                        "value": item["predicted"][
                            "empirical_hardware_floor_ns"
                        ],
                        "unit": "ns",
                        "scope": item["scope"],
                        "status": item["predicted"]["status"],
                        "source": "hardware-backend-prediction",
                        "is_full_duration_prediction": False,
                        "minimum_work_flops": item["predicted"]["minimum_work_flops"],
                        "compulsory_bytes": item["predicted"]["compulsory_bytes"],
                        "compute_time_ns": item["predicted"]["empirical_compute_time_ns"],
                        "memory_time_ns": item["predicted"]["empirical_memory_time_ns"],
                        "schedule": item["predicted"]["schedule"],
                        "serialized_hardware_floor_ns": item["predicted"][
                            "serialized_hardware_floor_ns"
                        ],
                        "critical_path_hardware_floor_ns": item["predicted"][
                            "critical_path_hardware_floor_ns"
                        ],
                        "resource_hardware_floor_ns": item["predicted"][
                            "resource_hardware_floor_ns"
                        ],
                        "ideal_dag_hardware_floor_ns": item["predicted"][
                            "ideal_dag_hardware_floor_ns"
                        ],
                        "limiting_resource": item["predicted"]["limiting_resource"],
                        "resource_limiting_resource": item["predicted"][
                            "resource_limiting_resource"
                        ],
                        "operator_achievable_frontier_ns": item["predicted"].get(
                            "operator_achievable_frontier_ns"
                        ),
                        "operator_frontier_standard_uncertainty_ns": item[
                            "predicted"
                        ].get("operator_frontier_standard_uncertainty_ns"),
                        "operator_frontier_match_status": item["predicted"].get(
                            "operator_frontier_match_status"
                        ),
                        "operator_frontier_anchor_ids": item["predicted"].get(
                            "operator_frontier_anchor_ids", []
                        ),
                        "operator_frontier_hardware_cohort": item["predicted"].get(
                            "operator_frontier_hardware_cohort"
                        ),
                        "operator_frontier_candidate_digest": item["predicted"].get(
                            "operator_frontier_candidate_digest"
                        ),
                        "operator_frontier_input_corpus_digest": item["predicted"].get(
                            "operator_frontier_input_corpus_digest"
                        ),
                        "operator_frontier_execution_contract_digest": item[
                            "predicted"
                        ].get("operator_frontier_execution_contract_digest"),
                    },
                    {
                        "id": comparison_id,
                        "kind": "metric-comparison",
                        "metric": "latency",
                        "case_id": case_id,
                        "scope": item["scope"],
                        "observed_minus_hardware_floor_ns": item["comparison"][
                            "observed_minus_hardware_floor_ns"
                        ],
                        "observed_to_hardware_floor_ratio": item["comparison"][
                            "observed_to_hardware_floor_ratio"
                        ],
                        "operator_frontier_efficiency": item["comparison"].get(
                            "operator_frontier_efficiency"
                        ),
                        "frontier_efficiency_status": item["comparison"].get(
                            "frontier_efficiency_status"
                        ),
                        "operator_frontier_gap_status": item["comparison"].get(
                            "operator_frontier_gap_status"
                        ),
                        "operator_frontier_combined_uncertainty_ns": item[
                            "comparison"
                        ].get("operator_frontier_combined_uncertainty_ns"),
                        "operator_frontier_uncertainty_components_ns": item[
                            "comparison"
                        ].get("operator_frontier_uncertainty_components_ns"),
                        "operator_frontier_uncertainty_policy": item[
                            "comparison"
                        ].get("operator_frontier_uncertainty_policy"),
                        "operator_frontier_comparison_reason_codes": item[
                            "comparison"
                        ].get("operator_frontier_comparison_reason_codes", []),
                        "relative_prediction_error": item["comparison"][
                            "relative_prediction_error"
                        ],
                        "error_status": item["comparison"]["error_status"],
                        "interpretation": item["comparison"]["interpretation"],
                    },
                ]
            )
            edges.extend(
                [
                    {
                        "source": comparison_id,
                        "target": predicted_id,
                        "kind": "compares_prediction",
                    },
                    {
                        "source": comparison_id,
                        "target": observed_id,
                        "kind": "compares_observation",
                    },
                ]
            )
            comparison_entrypoints.append(comparison_id)

        observed_memory_id = "metric:observed-framework-peak-memory"
        memory_comparison_id = "comparison:framework-peak-memory"
        nodes.extend(
            [
                {
                    "id": observed_memory_id,
                    "kind": "metric",
                    "name": "observed framework peak memory",
                    "value": comparison["memory"]["observed"][
                        "framework_peak_bytes"
                    ],
                    "unit": "bytes",
                    "source": "memory-observation",
                    "stable_path": comparison["memory"]["observed"][
                        "peak_stable_path"
                    ],
                },
                {
                    "id": memory_comparison_id,
                    "kind": "metric-comparison",
                    "metric": "framework-peak-memory",
                    **comparison["memory"]["comparison"],
                },
            ]
        )
        edges.extend(
            [
                {
                    "source": memory_comparison_id,
                    "target": peak_id,
                    "kind": "compares_prediction",
                },
                {
                    "source": memory_comparison_id,
                    "target": observed_memory_id,
                    "kind": "compares_observation",
                },
            ]
        )
        comparison_entrypoints.append(memory_comparison_id)

    return {
        "schema": "groundupscale.dev/explanation-graph/v1alpha2",
        "entrypoints": {
            "latency": [
                node["id"]
                for node in nodes
                if node["kind"] == "metric" and node.get("name") == "median latency"
            ],
            "throughput": [
                node["id"]
                for node in nodes
                if node["kind"] == "metric" and node.get("name") == "throughput"
            ],
            "peak_memory": [peak_id],
            "hardware_duration_bound": hardware_duration_entrypoints,
            "prediction_observation_comparison": comparison_entrypoints,
        },
        "nodes": nodes,
        "edges": edges,
        "calibration_status": "not-yet-applied",
    }


def _format_ms(value_ns: float | None) -> str:
    return f"{value_ns / 1_000_000:.3f}" if value_ns is not None else "not available"


def _format_share(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "not available"


def _format_us(value: float | None) -> str:
    return f"{value / 1_000:.3f} μs" if value is not None else "not available"


def _render_compound_phase_schedules(comparison: dict[str, Any]) -> str:
    sections: list[str] = []
    for item in comparison["latency_cases"]:
        authoritative = item["predicted"].get("compound_phase_schedule")
        provisional = item["predicted"].get(
            "compound_provisional_phase_schedule"
        )
        if authoritative is None and provisional is None:
            continue
        schedule = (
            authoritative
            if authoritative is not None
            and authoritative["selected_duration_ns"] is not None
            else provisional or authoritative
        )
        assert schedule is not None
        phase_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(phase['phase_name']))}</td>"
            f"<td><code>{html.escape(str(phase['operation_class']))}</code></td>"
            f"<td>{html.escape(', '.join(phase['predecessor_phase_ids']) or '—')}</td>"
            f"<td>{phase['minimum_flops']:,}</td>"
            f"<td>{phase['logical_read_bytes'] + phase['logical_write_bytes']:,}</td>"
            f"<td>{_format_us(phase['compute_time_ns'])}</td>"
            f"<td>{_format_us(phase['memory_time_ns'])}</td>"
            f"<td><code>{html.escape(str(phase['resource_composition']))}</code></td>"
            f"<td>{_format_us(phase['local_hardware_floor_ns'])}</td>"
            f"<td><code>{html.escape(str(phase['limiting_resource'] or '—'))}</code></td>"
            "<td>"
            + (
                "<br>".join(
                    f"<code>{html.escape(str(reference))}</code>"
                    for reference in phase["capability_evidence_refs"]
                )
                or "—"
            )
            + "</td>"
            "</tr>"
            for phase in schedule["phases"]
        )
        if authoritative is not None and authoritative["selected_duration_ns"] is not None:
            selected_summary = (
                f"<strong>{_format_us(authoritative['selected_duration_ns'])}</strong>"
            )
        elif provisional is not None and provisional["selected_duration_ns"] is not None:
            selected_summary = (
                "<strong>unknown</strong>；exploratory 降级预估 "
                f"<strong>{_format_us(provisional['selected_duration_ns'])}</strong>"
            )
        else:
            missing = authoritative or schedule
            selected_summary = (
                "<strong>unknown</strong>（缺失："
                + html.escape(", ".join(missing["missing_capabilities"]))
                + "）"
            )
        sections.append(
            "<h3>"
            f"{html.escape(item['case_id'])} · "
            f"{html.escape(str(schedule['policy']))}</h3>"
            "<p>"
            f"无 Chunk Pipeline Contract；所选串行参考为 "
            f"{selected_summary}。"
            "依赖 phase 不允许隐式重叠。</p>"
            "<table><thead><tr><th>Phase</th><th>计算类别</th>"
            "<th>前驱 Phase ID</th><th>最小 FLOPs</th><th>读写字节</th>"
            "<th>计算项</th><th>内存项</th><th>资源组合</th><th>局部地板</th>"
            "<th>限制资源</th><th>能力证据</th></tr></thead>"
            f"<tbody>{phase_rows}</tbody></table>"
        )
    if not sections:
        return ""
    return (
        "<h2>复合算子 Phase 串行构成</h2>"
        "<p>本节直接投影 Hardware Backend 的 Operator Phase Graph；"
        "报告层不重新计算。数值仍是资源地板参考，不是完整实现耗时预测。</p>"
        + "".join(sections)
    )


def _render_side_decomposition(title: str, side: dict[str, Any]) -> str:
    if not side.get("available"):
        return (
            f"<h3>{html.escape(title)}</h3>"
            '<div class="warning">无法生成：'
            f"{html.escape(str(side.get('reason', 'unknown reason')))}</div>"
        )
    rows = "".join(
        "<tr>"
        f"<td>{item['rank']}</td>"
        f"<td>{_format_ms(item['time_ns'])}</td>"
        f"<td>{_format_share(item['share_of_e2e'])}</td>"
        f"<td>{html.escape(str(item['operation']))}</td>"
        f"<td><code>{html.escape(item['stable_path'])}</code></td>"
        f"<td>{html.escape(', '.join(item['selection_reasons']))}</td>"
        f"<td>{html.escape(str(item.get('evidence', 'unknown')))}</td>"
        "</tr>"
        for item in side["selected"]
    )
    return f"""
<h3>{html.escape(title)}</h3>
<p>分母：{_format_ms(side['e2e_ns'])} ms；统计口径：<code>{html.escape(side['statistic'])}</code>。展示独立 Top 10 与所有占本侧 E2E 至少 10% 的项的并集。</p>
<table><thead><tr><th>本侧排名</th><th>时间 (ms)</th><th>占本侧 E2E</th><th>操作</th><th>Stable Path</th><th>入选原因</th><th>证据</th></tr></thead><tbody>{rows}</tbody></table>
"""


def _render_latency_decomposition(
    comparison: dict[str, Any],
) -> str:
    decomposition = comparison.get("latency_decomposition")
    if decomposition is None:
        return """
<h2>预测与实测时间 Top 10</h2>
<div class="warning">报告缺少 latency_decomposition；生成流程未满足 Top10/10% 完成门禁。</div>
"""
    predicted = decomposition["predicted"]
    observed = decomposition["observed"]
    provisional_mode = (
        decomposition.get("comparison_role") == "exploratory-planning-only"
    )
    benchmark_e2e = max(
        comparison["latency_cases"],
        key=lambda item: item["observed"]["median_ns"],
    )
    prediction_label = (
        "exploratory 降级预估（仅规划，非诊断）"
        if provisional_mode
        else "hardware floor"
    )
    contract = f"""
<h2>时间分解比较契约</h2>
<p>预测侧是 <code>{html.escape(str(predicted.get('kind')))}</code>（{prediction_label}），所选调度为 <code>{html.escape(str(predicted.get('schedule')))}</code>，E2E={_format_ms(predicted.get('e2e_ns'))} ms。实测 Top 10 来自一次同步诊断 Trace，E2E={_format_ms(observed.get('e2e_ns'))} ms；重复 Benchmark 的 E2E 中位数为 {_format_ms(benchmark_e2e['observed']['median_ns'])} ms。单次 Trace 不冒充 Benchmark 中位数；降级预估与实测的差距仅供规划，不计算 prediction error，不产生诊断 Verdict。</p>
"""
    predicted_table = _render_side_decomposition(
        "预测侧探索性规划 Top 10（非诊断）"
        if provisional_mode
        else "预测侧 Top 10",
        predicted,
    )
    observed_table = _render_side_decomposition("实测侧 Top 10", observed)

    joined_rows_parts = []
    for item in decomposition["joined"]:
        ratio = item["observed_to_predicted_ratio"]
        ratio_text = f"{ratio:.2f}×" if ratio is not None else "not comparable"
        evidence_detail = item["evidence_quality"]
        frontier_reasons = item.get("frontier_observation_reason_codes", [])
        if frontier_reasons:
            evidence_detail += " · " + ", ".join(frontier_reasons)
        joined_rows_parts.append(
            "<tr>"
            f"<td><code>{html.escape(item['stable_path'])}</code></td>"
            f"<td>{html.escape(str(item['operation']))}</td>"
            f"<td>{item['predicted_rank'] or '—'}</td>"
            f"<td>{_format_ms(item['predicted_time_ns'])}</td>"
            f"<td>{_format_share(item['predicted_share_of_e2e'])}</td>"
            f"<td>{item['observed_rank'] or '—'}</td>"
            f"<td>{_format_ms(item['observed_time_ns'])}</td>"
            f"<td>{_format_share(item['observed_share_of_e2e'])}</td>"
            f"<td>{_format_ms(item['observed_minus_predicted_ns'])}</td>"
            f"<td>{ratio_text}</td>"
            f"<td>{html.escape(evidence_detail)}</td>"
            "</tr>"
        )
    joined_rows = "".join(joined_rows_parts)
    joined_heading = (
        "Top 10 联合探索性差距（非诊断）"
        if provisional_mode
        else "Top 10 联合差异排名"
    )
    joined_table = f"""
<h3>{joined_heading}</h3>
<table><thead><tr><th>Stable Path</th><th>操作</th><th>预测排名</th><th>预测 (ms)</th><th>预测占比</th><th>实测排名</th><th>实测 (ms)</th><th>实测占比</th><th>实测−预测 (ms)</th><th>倍率</th><th>证据</th></tr></thead><tbody>{joined_rows}</tbody></table>
"""

    largest = decomposition["largest_discrepancy"]
    drilldown = (
        f"""
<h3>最大差异下钻</h3>
<p><code>{html.escape(largest['stable_path'])}</code>（{html.escape(str(largest['operation']))}）是已对齐入选项中绝对差异最大的叶子：预测 {_format_ms(largest['predicted_time_ns'])} ms，单次 Trace 实测 {_format_ms(largest['observed_time_ns'])} ms，差异 {_format_ms(largest['observed_minus_predicted_ns'])} ms。下钻边界：<code>{html.escape(largest['evidence_boundary'])}</code>；{html.escape(largest['drilldown_status'])}。</p>
"""
        if largest is not None
        else """
<h3>最大差异下钻</h3>
<div class="warning">降级预估不产生最大差异诊断；请仅使用上方排名和倍率做探索性规划。</div>
"""
        if provisional_mode
        else """
<h3>最大差异下钻</h3>
<div class="warning">预测和实测之间没有可按 Stable Path 对齐的叶子项。</div>
"""
    )
    predicted_reconciliation = predicted.get("reconciliation", {})
    observed_reconciliation = observed.get("reconciliation", {})
    reconciliation = f"""
<h3>时间回收</h3>
<table><thead><tr><th>侧</th><th>E2E (ms)</th><th>全量归因/区间并集 (ms)</th><th>Selected (ms)</th><th>Other (ms)</th><th>Unattributed (ms)</th><th>Overlap (ms)</th><th>覆盖率</th></tr></thead><tbody>
<tr><td>预测</td><td>{_format_ms(predicted.get('e2e_ns'))}</td><td>{_format_ms(predicted_reconciliation.get('all_attributed_ns'))}</td><td>{_format_ms(predicted_reconciliation.get('selected_sum_ns'))}</td><td>{_format_ms(predicted_reconciliation.get('other_ns'))}</td><td>{_format_ms(predicted_reconciliation.get('unattributed_ns'))}</td><td>{_format_ms(predicted_reconciliation.get('overlap_ns'))}</td><td>{_format_share(predicted_reconciliation.get('coverage'))}</td></tr>
<tr><td>实测 Trace</td><td>{_format_ms(observed.get('e2e_ns'))}</td><td>{_format_ms(observed_reconciliation.get('attributed_interval_union_ns'))}</td><td>{_format_ms(observed_reconciliation.get('selected_accounted_ns'))}</td><td>{_format_ms(observed_reconciliation.get('other_ns'))}</td><td>{_format_ms(observed_reconciliation.get('unattributed_ns'))}</td><td>{_format_ms(observed_reconciliation.get('overlap_ns'))}</td><td>{_format_share(observed_reconciliation.get('coverage'))}</td></tr>
</tbody></table>
<h3>{'规划说明' if provisional_mode else '结论'}</h3>
<p>{'预测侧是 exploratory 降级预估；排名和倍率只用于定位下一个采集或建模目标，不是诊断结论，不能进入 Frontier、校准或 prediction error。' if provisional_mode else '预测侧按逐候选局部硬件地板解释优化下界；实测侧按同步 operation span 和显式 unattributed host time 解释本次诊断执行。差异可能来自实现/算法效率、物化与布局、调度及框架开销，不能直接校准为硬件能力误差。'}</p>
"""
    return (
        contract
        + predicted_table
        + observed_table
        + joined_table
        + drilldown
        + reconciliation
    )


def render_report_html(
    *,
    run_id: str,
    device: str,
    benchmark: dict[str, Any],
    trace: dict[str, Any],
    live_set: dict[str, Any],
    explanation: dict[str, Any],
    comparison: dict[str, Any] | None = None,
    memory_observation: dict[str, Any] | None = None,
) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(case['case_id'])}</td>"
        f"<td><code>{html.escape(case['resolved_scope'])}</code></td>"
        f"<td>{case['latency']['median_ns'] / 1_000_000:.3f}</td>"
        f"<td>{case['latency']['iqr_over_median'] * 100:.2f}%</td>"
        f"<td>{case['latency']['throughput_per_second']:.3f}</td>"
        "</tr>"
        for case in benchmark["cases"]
    )
    graph_json = json.dumps(explanation, ensure_ascii=False).replace("</", "<\\/")
    hardware_floor = next(
        (
            node
            for node in explanation["nodes"]
            if node["id"] == "metric:hardware-empirical-time-floor"
        ),
        None,
    )
    hardware_metric = ""
    if hardware_floor is not None:
        measured_capabilities = [
            node
            for node in explanation["nodes"]
            if node["kind"] == "measured-hardware-capability"
        ]
        capability_quality = (
            "trusted"
            if measured_capabilities
            and all(node["environment_eligible"] for node in measured_capabilities)
            else "exploratory（能力测量环境门禁未通过）"
        )
        floor_value = hardware_floor.get("value")
        floor_text = (
            f"{floor_value / 1_000_000:.3f} ms"
            if isinstance(floor_value, (int, float))
            else "unknown（后端仅提供部分算子地板）"
        )
        if "resource_physical_floor_ns" in hardware_floor:
            hardware_metric = (
                '<div class="metric">M4 CPU 可采纳串行时延：'
                f"{_format_ms(hardware_floor['value'])} ms"
                "；降级预估："
                f"{_format_ms(hardware_floor.get('provisional_estimate_ns'))} ms"
                "；Resource Physical Floor："
                f"{_format_ms(hardware_floor['resource_physical_floor_ns'])} ms"
                "；理想 DAG 资源参考："
                f"{_format_ms(hardware_floor['ideal_dag_hardware_floor_ns'])} ms"
                "（复合 phase 能力不完整时前者保持 unknown；"
                "后两者不可作为点预测；"
                f"evidence={capability_quality}）</div>"
            )
        else:
            hardware_metric = (
                f'<div class="metric">{html.escape(device)} 经验硬件地板：'
                f"{floor_text}"
                "（跨算子 P80 能力包络，不是当前实现耗时预测；"
                f"evidence={capability_quality}）</div>"
            )
    comparison_section = ""
    latency_decomposition_section = ""
    compound_phase_section = ""
    if comparison is not None:
        has_exact_frontier = any(
            item["predicted"].get("operator_frontier_match_status")
            == "exact-anchor"
            for item in comparison["latency_cases"]
        )
        has_provisional = any(
            item["predicted"].get("provisional_estimate_ns") is not None
            and item["predicted"].get("provisional_evidence_tier")
            in {"exploratory", "provisional"}
            for item in comparison["latency_cases"]
        )
        prediction_reason_codes = sorted(
            {
                str(reason)
                for item in comparison["latency_cases"]
                for reason in item["predicted"].get(
                    "provisional_reason_codes", []
                )
            }
        )
        observation_reason_codes = sorted(
            {
                str(reason)
                for item in comparison["latency_cases"]
                for reason in item["observed"].get("reason_codes", [])
            }
        )
        comparison_rows_parts: list[str] = []
        for item in comparison["latency_cases"]:
            authoritative = item["predicted"]["full_duration_ns"]
            provisional = item["predicted"]["provisional_estimate_ns"]
            physical_floor = item["predicted"]["resource_physical_floor_ns"]
            ideal_dag = item["predicted"]["ideal_dag_hardware_floor_ns"]
            authoritative_ratio = (
                item["observed"]["median_ns"] / authoritative
                if authoritative is not None and authoritative > 0
                else None
            )
            provisional_ratio = item["comparison"]["observed_to_provisional_ratio"]
            operator_frontier = item["predicted"].get(
                "operator_achievable_frontier_ns"
            )
            frontier_efficiency = item["comparison"].get(
                "operator_frontier_efficiency"
            )
            frontier_status = item["comparison"].get(
                "frontier_efficiency_status", "not-evaluable"
            )
            frontier_gap_status = item["comparison"].get(
                "operator_frontier_gap_status", "not-evaluable"
            )
            authoritative_text = (
                f"{authoritative / 1_000_000:.3f}"
                if authoritative is not None
                else "N/A"
            )
            provisional_text = (
                f"{provisional / 1_000_000:.3f}<br><small>"
                "预测证据："
                f"{html.escape(str(item['predicted'].get('provisional_evidence_tier') or 'unverified'))}"
                "</small>"
                if provisional is not None
                else "N/A"
            )
            physical_floor_text = (
                f"{physical_floor / 1_000_000:.3f}"
                if physical_floor is not None
                else "N/A"
            )
            ideal_dag_text = (
                f"{ideal_dag / 1_000_000:.3f}"
                if ideal_dag is not None
                else "N/A"
            )
            authoritative_ratio_text = (
                f"{authoritative_ratio:.2f}×"
                if authoritative_ratio is not None
                else "N/A"
            )
            provisional_ratio_text = (
                f"{provisional_ratio:.2f}×"
                if provisional_ratio is not None
                else "N/A"
            )
            anchor_ids = item["predicted"].get(
                "operator_frontier_anchor_ids", []
            )
            frontier_uncertainty = item["predicted"].get(
                "operator_frontier_standard_uncertainty_ns"
            )
            frontier_cohort = item["predicted"].get(
                "operator_frontier_hardware_cohort"
            )
            frontier_candidate = item["predicted"].get(
                "operator_frontier_candidate_digest"
            )
            frontier_text = (
                f"{operator_frontier / 1_000_000:.3f}<br><small>"
                "Exact-Shape Operator Frontier；"
                f"±{float(frontier_uncertainty or 0) / 1_000_000:.6f} ms；Anchor: "
                f"{html.escape(', '.join(str(value) for value in anchor_ids))}"
                f"；cohort: {html.escape(str(frontier_cohort)[:20])}…"
                f"；candidate: {html.escape(str(frontier_candidate)[:12])}…"
                "</small>"
                if operator_frontier is not None
                else "N/A<br><small>"
                + html.escape(
                    ", ".join(
                        str(value)
                        for value in item["predicted"].get(
                            "operator_frontier_reason_codes", []
                        )
                    )
                    or "no exact Anchor"
                )
                + "</small>"
            )
            frontier_efficiency_text = (
                f"{frontier_efficiency:.2%}<br><small>"
                f"{html.escape(str(frontier_status))}; "
                f"{html.escape(str(frontier_gap_status))}; "
                "combined uncertainty ±"
                f"{float(item['comparison'].get('operator_frontier_combined_uncertainty_ns') or 0) / 1_000_000:.6f} ms"
                "</small>"
                if frontier_efficiency is not None
                else "N/A<br><small>"
                + html.escape(
                    ", ".join(
                        str(value)
                        for value in item["comparison"].get(
                            "operator_frontier_comparison_reason_codes", []
                        )
                    )
                    or "not evaluable"
                )
                + "</small>"
            )
            validation_text = (
                "Exact-Shape Frontier 对照已资格化；不是 prediction error"
                if frontier_status == "qualified"
                else "Frontier 已资格化，但本次 Observation 降级；不是 prediction error"
                if operator_frontier is not None
                else "权威值仍 unknown；降级预估仅供规划"
            )
            comparison_rows_parts.append(
                "<tr>"
                f"<td>{html.escape(item['case_id'])}</td>"
                f"<td><code>{html.escape(item['scope'])}</code></td>"
                f"<td>{html.escape(str(item['predicted']['schedule']))}</td>"
                f"<td>{authoritative_text}</td>"
                f"<td>{provisional_text}</td>"
                f"<td>{frontier_text}</td>"
                f"<td>{physical_floor_text}</td>"
                f"<td>{ideal_dag_text}</td>"
                f"<td>{item['observed']['median_ns'] / 1_000_000:.3f}"
                f"<br><small>实测证据：{html.escape(str(item['observed'].get('evidence_tier', 'unverified')))}"
                f"；{html.escape(', '.join(str(value) for value in item['observed'].get('reason_codes', [])) or 'none')}</small></td>"
                f"<td>{authoritative_ratio_text}</td>"
                f"<td>{provisional_ratio_text}</td>"
                f"<td>{frontier_efficiency_text}</td>"
                f"<td>{html.escape(str(item['predicted']['limiting_resource']))}</td>"
                f"<td>{validation_text}</td>"
                "</tr>"
            )
        comparison_rows = "".join(comparison_rows_parts)
        memory = comparison["memory"]
        prediction_reasons = html.escape(
            ", ".join(prediction_reason_codes) or "none"
        )
        observation_reasons = html.escape(
            ", ".join(observation_reason_codes) or "none"
        )
        comparison_section = f"""
<h2>预测—实测对照</h2>
<div class="warning">{'表中 Exact-Shape Operator Frontier 只对 Shape、dtype、layout/stride/alignment、threads、完整 Hardware Validity Cohort、candidate binary、input corpus、execution contract、correctness 与 timing boundary 全等的行生效；Frontier Efficiency=Frontier/Observation，表示当前实现接近已验证前沿的程度，不是 prediction error。任一 identity 不同都会显示 N/A 和 reason code。' if has_exact_frontier else ''}{' 存在 Exact-Shape Anchor 的行不改变其余行的 exploratory/unknown 边界；降级预估有数值但不可采纳、不可用于校准或 Frontier 晋级。权威 selected 时延仍只在完整证据满足时给出。' if has_provisional else ''}</div>
<p><strong>预测证据原因：</strong><code>{prediction_reasons}</code>；<strong>实测证据原因：</strong><code>{observation_reasons}</code>。两侧独立降级，任一侧 exploratory 都不得用于 prediction error 或证据晋级。</p>
<table><thead><tr><th>Case</th><th>Stable Path</th><th>调度</th><th>权威 selected 时延 (ms)</th><th>降级预估 (ms)</th><th>Exact-Shape Frontier (ms)</th><th>资源 Physical Floor (ms)</th><th>理想 DAG (ms)</th><th>实测中位数 (ms)</th><th>实测/权威 selected</th><th>实测/降级预估</th><th>Frontier Efficiency</th><th>组合限制</th><th>校验状态</th></tr></thead><tbody>{comparison_rows}</tbody></table>
<h3>峰值内存</h3>
<div class="metric">预测 framework peak：{memory['predicted']['framework_peak_bytes']:,} B</div>
<div class="metric">实测 framework peak：{memory['observed']['framework_peak_bytes']:,} B</div>
<div class="metric">绝对相对误差：{memory['comparison']['absolute_relative_error']:.2%}</div>
"""
        latency_decomposition_section = _render_latency_decomposition(comparison)
        compound_phase_section = _render_compound_phase_schedules(comparison)
    memory_section = ""
    if memory_observation is not None and all(
        key in memory_observation
        for key in (
            "logical_tensor_live_set",
            "framework_device_memory",
            "process_memory",
        )
    ):
        logical_bytes = memory_observation["logical_tensor_live_set"].get(
            "peak_framework_tensor_bytes"
        )
        framework_bytes = memory_observation["framework_device_memory"].get(
            "peak_allocated_bytes"
        )
        process_bytes = memory_observation["process_memory"].get("peak_rss_bytes")

        def byte_text(value: Any) -> str:
            return f"{value:,} B" if isinstance(value, int) else "N/A"

        memory_section = f"""
<h2>内存归因</h2>
<p>以下三项分别报告逻辑张量存储、框架设备分配器峰值和进程级主机 RSS，口径不混算。</p>
<div class="metric">逻辑张量 live set：{byte_text(logical_bytes)}</div>
<div class="metric">框架设备内存：{byte_text(framework_bytes)}</div>
<div class="metric">进程 RSS：{byte_text(process_bytes)}</div>
"""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GroundUpScale Run {html.escape(run_id)}</title>
<style>
body{{font:15px/1.55 system-ui,sans-serif;max-width:1180px;margin:2rem auto;padding:0 1rem;color:#17202a}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d8dee4;padding:.5rem;text-align:left}}
th{{background:#f6f8fa}}code{{font-size:.85em}}.metric{{display:inline-block;margin:.5rem 1rem .5rem 0;padding:1rem;background:#f6f8fa;border-radius:8px}}.warning{{padding:1rem;background:#fff4ce;border-left:4px solid #d29922}}
</style></head><body>
<h1>GroundUpScale 可解释运行报告</h1>
<p>Run <code>{html.escape(run_id)}</code> · device <code>{html.escape(device)}</code></p>
<div class="metric">预测 framework peak：{live_set['predicted_framework_peak_bytes']:,} B</div>
{hardware_metric}
<div class="metric">Trace 对齐覆盖率：{trace['alignment_map']['coverage']:.1%}</div>
<div class="metric">未归因 host 时间：{trace['error_attribution']['unattributed_host_ns']:,} ns</div>
<h2>Benchmark Cases</h2>
<table><thead><tr><th>Case</th><th>Stable Path</th><th>Median (ms)</th><th>IQR/median</th><th>吞吐 (/s)</th></tr></thead><tbody>{rows}</tbody></table>
{comparison_section}
{memory_section}
{compound_phase_section}
{latency_decomposition_section}
<h2>下钻入口</h2>
<p>完整节点、公式、Stable Path、span 和连边保存在 <code>prediction/explanation.graph.json</code>；本页内嵌同一份图，供 Web 读取。</p>
<script id="explanation-graph" type="application/json">{graph_json}</script>
</body></html>"""


__all__ = ["build_explanation_graph", "render_report_html"]
