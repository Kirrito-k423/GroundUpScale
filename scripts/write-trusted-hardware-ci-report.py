#!/usr/bin/env python3
"""Write the machine-readable result of the local trusted hardware lane."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


SCHEMA = "groundupscale.dev/trusted-hardware-ci-report/v1alpha1"
STATUSES = frozenset(
    {"evidence_collected", "quarantined", "hardware_unavailable"}
)


def _manifest_identity(run_bundle: str) -> dict[str, object]:
    manifest_path = Path(run_bundle) / "run.manifest.json"
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Run Manifest has no run_id")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Run Manifest has no artifact list")
    return {
        "run_id": run_id,
        "manifest_sha256": sha256(payload).hexdigest(),
        "artifact_count": len(artifacts),
        "artifact_roles": sorted(
            artifact["role"]
            for artifact in artifacts
            if isinstance(artifact, dict) and isinstance(artifact.get("role"), str)
        ),
    }


def _bundle_ref(manifest: dict[str, object]) -> dict[str, object]:
    return {
        "run_id": manifest["run_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "artifact_ref": "artifact://run-bundle/run.manifest.json",
    }


def _evidence_ref(path: str, report_path: str) -> str:
    candidate = Path(path)
    try:
        relative = candidate.relative_to(Path(report_path).parent)
    except ValueError:
        return f"artifact://attempt-evidence/{candidate.name}"
    return f"artifact://attempt-evidence/{relative.as_posix()}"


def _resolve_previous_bundle(report_path: Path, value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict) or not isinstance(value.get("run_id"), str):
        raise ValueError("previous report has no Run ID")
    artifact_store = report_path.parents[2]
    return str(artifact_store / "runs" / value["run_id"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--status", required=True, choices=sorted(STATUSES))
    parser.add_argument("--run-bundle")
    parser.add_argument("--reason-code", action="append", default=[])
    parser.add_argument("--failure-evidence", action="append", default=[])
    parser.add_argument("--previous-qualified-root")
    parser.add_argument("--reason-codes-from-json")
    args = parser.parse_args()

    reason_codes = list(args.reason_code)
    if args.reason_codes_from_json:
        try:
            failure = json.loads(
                Path(args.reason_codes_from_json).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            failure = {}
        source_reason_codes = failure.get("reason_codes")
        if isinstance(source_reason_codes, list):
            for reason_code in source_reason_codes:
                if isinstance(reason_code, str) and reason_code not in reason_codes:
                    reason_codes.append(reason_code)

    current_manifest: dict[str, str] | None = None
    if args.run_bundle:
        try:
            current_manifest = _manifest_identity(args.run_bundle)
        except (OSError, json.JSONDecodeError, ValueError):
            current_manifest = None

    previous_qualified_evidence: list[dict[str, str]] = []
    previous_evidence_errors: list[str] = []
    if args.previous_qualified_root:
        root = Path(args.previous_qualified_root)
        for report_path in sorted(
            root.glob("*/trusted-hardware-ci-report.json")
        ):
            try:
                prior = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous_evidence_errors.append(str(report_path))
                continue
            run_bundle = prior.get("run_bundle")
            if prior.get("schema") != SCHEMA or prior.get("status") not in STATUSES:
                previous_evidence_errors.append(str(report_path))
                continue
            if prior.get("status") == "evidence_collected":
                prior_manifest = prior.get("run_manifest")
                try:
                    observed_manifest = _manifest_identity(
                        _resolve_previous_bundle(report_path, run_bundle)
                    )
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    previous_evidence_errors.append(str(report_path))
                    continue
                if prior_manifest != observed_manifest:
                    previous_evidence_errors.append(str(report_path))
                    continue
                identity = _bundle_ref(observed_manifest)
                if identity not in previous_qualified_evidence:
                    previous_qualified_evidence.append(identity)

    status = args.status
    failure_evidence = list(args.failure_evidence)
    if previous_evidence_errors:
        if status == "evidence_collected":
            status = "quarantined"
        if "previous-evidence-unreadable" not in reason_codes:
            reason_codes.append("previous-evidence-unreadable")
        for report_path in previous_evidence_errors:
            if report_path not in failure_evidence:
                failure_evidence.append(report_path)

    if status == "evidence_collected" and (
        not isinstance(args.run_bundle, str)
        or current_manifest is None
        or reason_codes
        or failure_evidence
    ):
        parser.error("evidence_collected requires one Bundle and no failures")
    if status == "quarantined" and (not reason_codes or not failure_evidence):
        parser.error("quarantined requires reasons and failure evidence")
    if status == "hardware_unavailable" and (
        not reason_codes or args.run_bundle is not None
    ):
        parser.error("hardware_unavailable requires reasons and no Bundle")

    report = {
        "schema": SCHEMA,
        "run_tag": args.run_tag,
        "device": args.device,
        "status": status,
        "promotion_allowed": False,
        "reason_codes": reason_codes,
        "run_bundle": (
            _bundle_ref(current_manifest)
            if current_manifest is not None
            else None
        ),
        "run_manifest": current_manifest,
        "failure_evidence": [
            _evidence_ref(path, args.output) for path in failure_evidence
        ],
        "previous_qualified_evidence": previous_qualified_evidence,
        "policies": {
            "collection": "groundupscale.dev/trusted-hardware-ci-policy/v1alpha1",
            "noise": "groundupscale.dev/trusted-hardware-noise-check/v1alpha1",
            "promotion": "groundupscale.dev/trusted-hardware-promotion/v1alpha1",
            "retention": "groundupscale.dev/trusted-hardware-retention/v1alpha1",
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    return 1 if previous_evidence_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
