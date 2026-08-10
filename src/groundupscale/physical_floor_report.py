"""Deterministic HTML projection of a Physical Floor comparison."""

from __future__ import annotations

from html import escape
from typing import Any


def render_physical_floor_report(comparison: dict[str, Any]) -> str:
    floor = comparison["physical_floor"]
    observation = comparison["observation"]
    result = comparison["comparison"]
    quality = floor["quality"]
    reasons = ", ".join(quality["reason_codes"]) or "none"
    assumptions = "".join(
        f"<li>{escape(assumption)}</li>" for assumption in floor["assumptions"]
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Ascend MatMul Physical Floor 与 Observation</title>
<style>body{{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;line-height:1.5}}code{{background:#f4f4f4;padding:.15rem .3rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:.5rem;text-align:left}}.unknown{{color:#a33}}.quality{{color:#875b00}}</style>
</head><body>
<h1>Ascend MatMul：Resource Physical Floor 与 Observation</h1>
<p>Stable Path：<code>{escape(comparison['stable_path'])}</code></p>
<p>Hardware Cohort：<code>{escape(comparison['hardware_cohort'])}</code></p>
<table><thead><tr><th>层</th><th>值</th><th>解释</th></tr></thead><tbody>
<tr><td>Vendor theory</td><td>unknown</td><td>未发布可比的单卡 910B2 FP32/HBM 理论值</td></tr>
<tr><td>compute Physical Floor</td><td>{floor['compute_time_ns'] / 1_000:.3f} μs</td><td>minimum work / measured FP32 P80</td></tr>
<tr><td>memory Physical Floor</td><td>{floor['memory_time_ns'] / 1_000:.3f} μs</td><td>compulsory bytes / measured HBM-copy P80</td></tr>
<tr><td>Resource Physical Floor</td><td>{floor['resource_physical_floor_ns'] / 1_000:.3f} μs</td><td>局部资源 envelope；不是当前实现耗时预测</td></tr>
<tr><td>Operator Frontier</td><td>unknown</td><td>Issue #29 不执行 Frontier qualification</td></tr>
<tr><td>Observation</td><td>{observation['median_ns'] / 1_000:.3f} μs</td><td>声明 Completion Boundary 的真实 Baseline Timing median</td></tr>
<tr><td>Observation / Floor</td><td>{result['observed_to_physical_floor_ratio']:.3f}×</td><td>优化 headroom，不是 prediction error</td></tr>
</tbody></table>
<p class="unknown">完整实现 duration：unknown</p>
<p class="quality">Capability quality：{escape(quality['status'])}；reasons={escape(reasons)}</p>
<h2>Physical Floor assumptions</h2><ul>{assumptions}</ul>
<p>不支持的 Cost IR 区域：{comparison['unsupported_regions']['count']}；保持 partial/unknown。</p>
</body></html>"""


__all__ = ["render_physical_floor_report"]
