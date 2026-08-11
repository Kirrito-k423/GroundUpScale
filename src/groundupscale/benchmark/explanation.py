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
                    "unit": "ns",
                    "source": "hardware-backend-prediction",
                    "status": hardware_prediction.status,
                    "is_full_duration_prediction": False,
                    "minimum_work_flops": hardware_prediction.program_bounds.flops,
                    "compulsory_bytes": hardware_prediction.program_bounds.compulsory_bytes,
                    "compute_time_ns": hardware_prediction.program_bounds.empirical_compute_time_ns,
                    "memory_time_ns": hardware_prediction.program_bounds.empirical_memory_time_ns,
                    "limiting_resource": hardware_prediction.program_bounds.limiting_resource,
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
                        "limiting_resource": item["predicted"]["limiting_resource"],
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
        "schema": "groundupscale.dev/explanation-graph/v1alpha1",
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
        hardware_metric = (
            f'<div class="metric">{html.escape(device)} 经验硬件地板：'
            f"{floor_text}"
            "（跨算子 P80 能力包络，不是当前实现耗时预测；"
            f"evidence={capability_quality}）</div>"
        )
    comparison_section = ""
    if comparison is not None:
        comparison_rows_parts: list[str] = []
        for item in comparison["latency_cases"]:
            floor = item["predicted"]["empirical_hardware_floor_ns"]
            ratio = item["comparison"]["observed_to_hardware_floor_ratio"]
            floor_text = f"{floor / 1_000_000:.3f}" if floor is not None else "N/A"
            ratio_text = f"{ratio:.2f}×" if ratio is not None else "N/A"
            comparison_rows_parts.append(
                "<tr>"
                f"<td>{html.escape(item['case_id'])}</td>"
                f"<td><code>{html.escape(item['scope'])}</code></td>"
                f"<td>{floor_text}</td>"
                f"<td>{item['observed']['median_ns'] / 1_000_000:.3f}</td>"
                f"<td>{ratio_text}</td>"
                f"<td>{html.escape(str(item['predicted']['limiting_resource']))}</td>"
                "<td>不可计算预测误差（这是算法无关硬件地板）</td>"
                "</tr>"
            )
        comparison_rows = "".join(comparison_rows_parts)
        memory = comparison["memory"]
        comparison_section = f"""
<h2>预测—实测对照</h2>
<p>时延列把实测中位数与算法无关经验硬件地板并列展示。二者距离表示实现、调度和系统开销形成的优化空间，不是点预测误差。</p>
<table><thead><tr><th>Case</th><th>Stable Path</th><th>硬件地板 (ms)</th><th>实测中位数 (ms)</th><th>实测/地板</th><th>限制资源</th><th>校验状态</th></tr></thead><tbody>{comparison_rows}</tbody></table>
<h3>峰值内存</h3>
<div class="metric">预测 framework peak：{memory['predicted']['framework_peak_bytes']:,} B</div>
<div class="metric">实测 framework peak：{memory['observed']['framework_peak_bytes']:,} B</div>
<div class="metric">绝对相对误差：{memory['comparison']['absolute_relative_error']:.2%}</div>
"""
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
th{{background:#f6f8fa}}code{{font-size:.85em}}.metric{{display:inline-block;margin:.5rem 1rem .5rem 0;padding:1rem;background:#f6f8fa;border-radius:8px}}
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
<h2>下钻入口</h2>
<p>完整节点、公式、Stable Path、span 和连边保存在 <code>prediction/explanation.graph.json</code>；本页内嵌同一份图，供 Web 读取。</p>
<script id="explanation-graph" type="application/json">{graph_json}</script>
</body></html>"""


__all__ = ["build_explanation_graph", "render_report_html"]
