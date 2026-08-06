"""Queryable explanation graph and a small standalone HTML reader."""

from __future__ import annotations

import html
import json
from typing import Any

from groundupscale.ir import CostProgram


def build_explanation_graph(
    cost: CostProgram,
    benchmark: dict[str, Any],
    trace: dict[str, Any],
    live_set: dict[str, Any],
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
<div class="metric">Trace 对齐覆盖率：{trace['alignment_map']['coverage']:.1%}</div>
<div class="metric">未归因 host 时间：{trace['error_attribution']['unattributed_host_ns']:,} ns</div>
<h2>Benchmark Cases</h2>
<table><thead><tr><th>Case</th><th>Stable Path</th><th>Median (ms)</th><th>IQR/median</th><th>吞吐 (/s)</th></tr></thead><tbody>{rows}</tbody></table>
<h2>下钻入口</h2>
<p>完整节点、公式、Stable Path、span 和连边保存在 <code>prediction/explanation.graph.json</code>；本页内嵌同一份图，供 Web 读取。</p>
<script id="explanation-graph" type="application/json">{graph_json}</script>
</body></html>"""


__all__ = ["build_explanation_graph", "render_report_html"]
