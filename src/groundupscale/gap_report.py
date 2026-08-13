"""Replayable E2E prediction-observation gap and reconciliation report."""

from __future__ import annotations

from hashlib import sha256
from html import escape
import json
from math import isfinite, sqrt
from pathlib import Path
import tempfile
from typing import Any, Mapping

from groundupscale.ir import canonical_data
from groundupscale.run_bundle import RUN_ID_PATTERN, RunBundleExistsError


INPUT_SCHEMA = "groundupscale.dev/e2e-gap-report-input/v1alpha1"
RESULT_SCHEMA = "groundupscale.dev/e2e-gap-report/v1alpha1"
REPORT_SCHEMA = "groundupscale.dev/e2e-gap-report-html/v1alpha1"
PRODUCER = "groundupscale@0.1.0"


class GapReportError(ValueError):
    """The report input cannot be interpreted without inventing evidence."""


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(canonical_data(value), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def _mapping(value: object, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GapReportError(reason)
    return value


def _number(value: object, reason: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(float(value))
        or float(value) < 0
    ):
        raise GapReportError(reason)
    return float(value)


def _refs(value: object) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise GapReportError("invalid-evidence-refs")
    return list(value)


def _items(side: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = side.get("items", [])
    if not isinstance(raw, list):
        raise GapReportError("invalid-side-items")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_item in raw:
        item = _mapping(raw_item, "invalid-side-item")
        path = item.get("stable_path")
        if not isinstance(path, str) or not path or path in seen:
            raise GapReportError("invalid-or-duplicate-stable-path")
        seen.add(path)
        if item.get("inclusive") is True:
            raise GapReportError("inclusive-parent-is-navigation-only")
        status = item.get("status", "known")
        duration = item.get("duration_ns")
        if status == "known":
            duration = _number(duration, "invalid-item-duration")
        elif status == "unknown" and duration is None:
            duration = None
        else:
            raise GapReportError("invalid-item-status")
        uncertainty = item.get("standard_uncertainty_ns")
        if uncertainty is not None:
            uncertainty = _number(uncertainty, "invalid-item-uncertainty")
        items.append(
            {
                "stable_path": path,
                "operation_class": item.get("operation_class"),
                "status": status,
                "duration_ns": duration,
                "standard_uncertainty_ns": uncertainty,
                "evidence_quality": item.get(
                    "evidence_quality",
                    "direct-qualified" if status == "known" else "structured-unknown",
                ),
                "evidence_refs": _refs(item.get("evidence_refs", [])),
                **(
                    {"evidence_boundaries": list(item["evidence_boundaries"])}
                    if isinstance(item.get("evidence_boundaries"), list)
                    else {}
                ),
            }
        )
    return items


def _select(
    items: list[dict[str, Any]], e2e_ns: float | None, policy: Mapping[str, Any]
) -> dict[str, Any]:
    known = [item for item in items if item["duration_ns"] is not None]
    ranked = sorted(known, key=lambda item: (-item["duration_ns"], item["stable_path"]))
    enriched = []
    for rank, item in enumerate(ranked, start=1):
        enriched.append(
            {
                **item,
                "rank": rank,
                "share_of_e2e": item["duration_ns"] / e2e_ns if e2e_ns else None,
            }
        )
    top_k = policy.get("top_k")
    threshold = policy.get("mandatory_share_of_e2e")
    if not isinstance(top_k, int) or top_k <= 0 or top_k != 10:
        raise GapReportError("invalid-top-k-policy")
    threshold = _number(threshold, "invalid-share-policy")
    if threshold != 0.10:
        raise GapReportError("invalid-share-policy")
    top10 = enriched[:top_k]
    top_paths = {item["stable_path"] for item in top10}
    mandatory = [
        item
        for item in enriched
        if e2e_ns is not None and item["duration_ns"] >= e2e_ns * threshold
    ]
    mandatory_paths = {item["stable_path"] for item in mandatory}
    selected = [
        {
            **item,
            "selection_reasons": [
                reason
                for reason, applies in (
                    ("top10", item["stable_path"] in top_paths),
                    ("at-least-10%-of-e2e", item["stable_path"] in mandatory_paths),
                )
                if applies
            ],
        }
        for item in enriched
        if item["stable_path"] in top_paths | mandatory_paths
    ]
    return {
        "all_items": enriched,
        "unknown_items": [item for item in items if item["duration_ns"] is None],
        "top10": top10,
        "mandatory": mandatory,
        "selected": selected,
    }


def _side(value: object, policy: Mapping[str, Any], *, predicted: bool) -> dict[str, Any]:
    side = _mapping(value, "invalid-report-side")
    status = side.get("status")
    available = status in {"known", "available"}
    e2e = _number(side.get("e2e_duration_ns"), "invalid-side-e2e") if available else None
    items = _items(side)
    selected = _select(items, e2e, policy)
    all_attributed = sum(item["duration_ns"] for item in selected["all_items"])
    selected_ns = sum(item["duration_ns"] for item in selected["selected"])
    unattributed = _number(side.get("unattributed_ns", 0), "invalid-unattributed") if available else None
    overlap = _number(side.get("overlap_ns", 0), "invalid-overlap") if available else None
    accounted = (
        all_attributed + unattributed - overlap
        if available and unattributed is not None and overlap is not None
        else None
    )
    reconciliation = {
        "e2e_ns": e2e,
        "selected_ns": selected_ns if available else None,
        "all_attributed_ns": all_attributed if available else None,
        "other_ns": max(0.0, all_attributed - selected_ns) if available else None,
        "unattributed_ns": unattributed,
        "overlap_ns": overlap,
        "accounted_e2e_ns": accounted,
        "residual_ns": e2e - accounted if e2e is not None and accounted is not None else None,
    }
    uncertainty = side.get("standard_uncertainty_ns")
    return {
        "status": status,
        "available": available,
        "e2e_duration_ns": e2e,
        "standard_uncertainty_ns": (
            _number(uncertainty, "invalid-side-uncertainty")
            if uncertainty is not None
            else None
        ),
        "bound_kind": side.get("bound_kind") if predicted else None,
        "accounting": side.get("accounting"),
        "evidence_refs": _refs(side.get("evidence_refs", [])),
        **selected,
        "reconciliation": reconciliation,
        **(
            {
                "reason_code": side.get("reason_code"),
                "evidence_boundaries": list(side.get("evidence_boundaries", [])),
                "required_next_measurement": side.get("required_next_measurement"),
            }
            if not available
            else {}
        ),
    }


def _classification(row: Mapping[str, Any]) -> str:
    operation = str(row.get("operation_class") or "").lower()
    if operation in {"view", "transpose", "reshape", "copy"}:
        return "materialization-layout"
    return "scheduling-integration"


def compose_gap_report(document: Mapping[str, object]) -> dict[str, Any]:
    """Compose one report without converting missing evidence into zero."""

    if document.get("schema") != INPUT_SCHEMA:
        raise GapReportError("unsupported-gap-report-input")
    identity = dict(_mapping(document.get("identity"), "invalid-report-identity"))
    policy = _mapping(document.get("policy"), "invalid-report-policy")
    predicted = _side(document.get("predicted"), policy, predicted=True)
    observed = _side(document.get("observed"), policy, predicted=False)

    predicted_by_path = {item["stable_path"]: item for item in predicted["all_items"]}
    observed_by_path = {item["stable_path"]: item for item in observed["all_items"]}
    selected_paths = {
        item["stable_path"] for item in predicted["selected"]
    } | {item["stable_path"] for item in observed["selected"]}
    rows = []
    for path in sorted(selected_paths):
        p = predicted_by_path.get(path)
        o = observed_by_path.get(path)
        p_ns = p["duration_ns"] if p else None
        o_ns = o["duration_ns"] if o else None
        gap = abs(o_ns - p_ns) if p_ns is not None and o_ns is not None else None
        combined = (
            sqrt((p.get("standard_uncertainty_ns") or 0) ** 2 + (o.get("standard_uncertainty_ns") or 0) ** 2)
            if p is not None and o is not None
            else None
        )
        rows.append(
            {
                "stable_path": path,
                "operation_class": (o or p or {}).get("operation_class"),
                "predicted_time_ns": p_ns,
                "observed_time_ns": o_ns,
                "absolute_gap_ns": gap,
                "ratio": o_ns / p_ns if p_ns not in {None, 0} and o_ns is not None else None,
                "predicted_rank": p.get("rank") if p else None,
                "observed_rank": o.get("rank") if o else None,
                "predicted_share_of_e2e": p.get("share_of_e2e") if p else None,
                "observed_share_of_e2e": o.get("share_of_e2e") if o else None,
                "combined_uncertainty_ns": combined,
                "predicted_evidence_quality": p.get("evidence_quality") if p else "unavailable",
                "observed_evidence_quality": o.get("evidence_quality") if o else "unavailable",
                "predicted_evidence_refs": p.get("evidence_refs", []) if p else [],
                "observed_evidence_refs": o.get("evidence_refs", []) if o else [],
            }
        )
    rows.sort(key=lambda row: (row["absolute_gap_ns"] is None, -(row["absolute_gap_ns"] or 0), row["stable_path"]))

    p_e2e, o_e2e = predicted["e2e_duration_ns"], observed["e2e_duration_ns"]
    p_u, o_u = predicted["standard_uncertainty_ns"], observed["standard_uncertainty_ns"]
    point_prediction = predicted.get("bound_kind") == "point-prediction"
    comparable = p_e2e is not None and o_e2e is not None and point_prediction
    e2e_gap = abs(o_e2e - p_e2e) if comparable else None
    combined_e2e = sqrt((p_u or 0) ** 2 + (o_u or 0) ** 2) if comparable else None
    metrics = {
        "e2e_absolute_gap_ns": e2e_gap,
        "e2e_ratio": o_e2e / p_e2e if comparable and p_e2e else None,
        "combined_uncertainty_ns": combined_e2e,
        "frontier_efficiency": p_e2e / o_e2e if comparable and o_e2e else None,
        "relative_prediction_error": e2e_gap / o_e2e if comparable and o_e2e else None,
        "applicability": "comparable-point-prediction" if comparable else "unavailable-non-point-or-missing-side",
    }

    diagnosis_policy = _mapping(policy.get("deep_diagnosis"), "invalid-diagnosis-policy")
    minimum_gap = _number(diagnosis_policy.get("minimum_absolute_gap_ns"), "invalid-diagnosis-policy")
    minimum_relative = _number(diagnosis_policy.get("minimum_relative_gap"), "invalid-diagnosis-policy")
    triggered = []
    if comparable:
        for row in rows:
            gap = row["absolute_gap_ns"]
            uncertainty = row["combined_uncertainty_ns"]
            observed_ns = row["observed_time_ns"]
            relative = gap / observed_ns if gap is not None and observed_ns else None
            if (
                gap is not None
                and uncertainty is not None
                and gap > uncertainty
                and gap > minimum_gap
                and relative is not None
                and relative > minimum_relative
            ):
                triggered.append({**row, "classification": _classification(row)})
    diagnosis = {
        "status": "evaluated" if comparable else "unavailable",
        "policy": dict(policy),
        "triggered": triggered,
        "reason_code": None if comparable else "comparison-not-applicable",
    }
    if triggered:
        largest = max(triggered, key=lambda row: row["absolute_gap_ns"])
        drilldown = {
            "kind": "actionable-operation",
            "stable_path": largest["stable_path"],
            "classification": largest["classification"],
            "non_overlapping_children": [],
            "evidence_refs": sorted(set(largest["predicted_evidence_refs"] + largest["observed_evidence_refs"])),
        }
    elif not observed["available"]:
        drilldown = {
            "kind": "evidence-boundary",
            "stable_path": None,
            "evidence_boundaries": observed.get("evidence_boundaries", []),
            "required_next_measurement": observed.get("required_next_measurement"),
        }
    elif not predicted["available"]:
        drilldown = {
            "kind": "evidence-boundary",
            "stable_path": None,
            "evidence_boundaries": predicted.get("evidence_boundaries", []),
            "required_next_measurement": predicted.get("required_next_measurement"),
        }
    else:
        drilldown = {"kind": "none", "stable_path": None, "evidence_boundaries": []}
    return {
        "schema": RESULT_SCHEMA,
        "status": "complete" if comparable else "structured-unknown",
        "identity": identity,
        "visibility_rule": {
            "top_k": 10,
            "mandatory_share_of_e2e": 0.10,
            "selection": "independent-per-side-then-exact-stable-path-union",
        },
        "predicted": predicted,
        "observed": observed,
        "gap_table": rows,
        "metrics": metrics,
        "diagnosis": diagnosis,
        "drilldown": drilldown,
        "derivation": {"input_sha256": _digest(_json_bytes(document))},
    }


def _fmt(value: object) -> str:
    return "unavailable" if value is None else f"{float(value):.3f}"


def render_gap_report_html(report: Mapping[str, Any]) -> str:
    """Project the machine report verbatim into a human-readable report."""

    rows = "".join(
        "<tr>"
        f"<td><code>{escape(str(row['stable_path']))}</code></td>"
        f"<td>{_fmt(row['predicted_time_ns'])}</td><td>{_fmt(row['observed_time_ns'])}</td>"
        f"<td>{_fmt(row['absolute_gap_ns'])}</td><td>{_fmt(row['ratio'])}</td>"
        f"<td>{escape(str(row['predicted_evidence_quality']))}</td>"
        f"<td>{escape(str(row['observed_evidence_quality']))}</td>"
        "</tr>"
        for row in report["gap_table"]
    )
    boundaries = ", ".join(report["drilldown"].get("evidence_boundaries", [])) or "none"
    payload = json.dumps(canonical_data(report), ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>E2E prediction-observation gap report</title></head>
<body><h1>E2E prediction-observation gap report</h1>
<p>Status: <strong>{escape(str(report['status']))}</strong>. Evidence boundary: {escape(boundaries)}.</p>
<p>E2E absolute gap (ns): {_fmt(report['metrics']['e2e_absolute_gap_ns'])}; ratio: {_fmt(report['metrics']['e2e_ratio'])}; combined uncertainty (ns): {_fmt(report['metrics']['combined_uncertainty_ns'])}; Frontier efficiency: {_fmt(report['metrics']['frontier_efficiency'])}; relative prediction error: {_fmt(report['metrics']['relative_prediction_error'])}.</p>
<table><thead><tr><th>Stable Path</th><th>Predicted ns</th><th>Observed ns</th><th>Absolute gap ns</th><th>Ratio</th><th>Predicted evidence</th><th>Observed evidence</th></tr></thead><tbody>{rows}</tbody></table>
<script type=\"application/json\" id=\"groundupscale-gap-report\">{payload}</script></body></html>\n"""


def write_gap_report_bundle(
    artifact_store: str | Path, *, run_id: str, document: Mapping[str, object]
) -> Path:
    """Write an immutable report bundle whose two projections share one input."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise GapReportError("unsafe-run-id")
    report = compose_gap_report(document)
    root = Path(artifact_store).resolve() / "runs"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_id
    if destination.exists():
        raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=root))
    artifacts = []

    def write(role: str, relative: str, payload: bytes, media_type: str, schema: str, inputs: list[str]) -> None:
        path = temporary / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts.append({
            "role": role, "path": relative, "media_type": media_type,
            "schema": schema, "sha256": _digest(payload), "produced_by": PRODUCER,
            "inputs": inputs,
        })

    try:
        write("e2e-gap-report-input", "resolved/e2e-gap-report-input.json", _json_bytes(document), "application/json", INPUT_SCHEMA, [])
        write("e2e-gap-report", "comparison/e2e-gap-report.json", _json_bytes(report), "application/json", RESULT_SCHEMA, ["resolved/e2e-gap-report-input.json"])
        write("html-report", "reports/report.html", render_gap_report_html(report).encode(), "text/html", REPORT_SCHEMA, ["comparison/e2e-gap-report.json"])
        manifest = {
            "schema": "groundupscale.dev/run-manifest/v1alpha1",
            "run_id": run_id,
            "bundle_kind": "e2e-gap-report",
            "status": "completed",
            "hardware_cohort": report["identity"].get("hardware_cohort"),
            "producer": PRODUCER,
            "artifacts": artifacts,
            **(
                {"source_bundles": list(document["source_bundles"])}
                if isinstance(document.get("source_bundles"), list)
                else {}
            ),
        }
        (temporary / "run.manifest.json").write_bytes(_json_bytes(manifest))
        temporary.rename(destination)
    except BaseException:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def main() -> None:
    """Build a bundle from one locked JSON input without touching hardware."""

    import argparse

    parser = argparse.ArgumentParser(prog="python -m groundupscale.gap_report")
    parser.add_argument("input")
    parser.add_argument("--artifact-store", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    document = json.loads(Path(args.input).read_text(encoding="utf-8"))
    destination = write_gap_report_bundle(
        args.artifact_store, run_id=args.run_id, document=document
    )
    print(destination)


if __name__ == "__main__":
    main()


__all__ = [
    "GapReportError", "compose_gap_report", "render_gap_report_html",
    "write_gap_report_bundle",
]
