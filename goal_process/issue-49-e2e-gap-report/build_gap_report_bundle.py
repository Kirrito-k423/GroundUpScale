"""Publish issue #49 from the reviewed #47/#48 authority bundles."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path

from groundupscale.gap_report import (
    derive_tiered_iteration_report,
    write_gap_report_bundle,
)
from groundupscale.run_bundle import verify_run_bundle


ROOT = Path(__file__).resolve().parents[2]
PREDICTED_RUN = (
    ROOT
    / "goal_process/issue-48-schedule-achievable-frontier/evidence/runs"
    / "issue48-20260814T0002Z-schedule-frontier-unknown-v2"
)
OBSERVED_RUN = (
    ROOT
    / "goal_process/issue-47-observed-decomposition/evidence/runs"
    / "issue47-ascend-observed-decomposition-20260813-v1"
)
MODEL_RUN = (
    ROOT
    / "goal_process/issue-30-ascend-transformer-demo/evidence/runs"
    / "ascend-910b2-transformer-demo-20260811-v1"
)
LEGACY_REPORT_RUN = (
    ROOT
    / "goal_process/issue-49-e2e-gap-report/evidence/runs"
    / "issue49-20260814T0730Z-e2e-gap-report-v12"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_source(run: Path) -> dict[str, object]:
    manifest_path = run / "run.manifest.json"
    manifest = _load(manifest_path)
    verification = verify_run_bundle(run)
    if verification.get("passed") is not True:
        raise RuntimeError(f"source bundle failed verification: {run}")
    return {
        "run_id": manifest["run_id"],
        "bundle_kind": manifest["bundle_kind"],
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "verification_passed": True,
        "evidence_ref": (
            f"run://{manifest['run_id']}@sha256:"
            f"{sha256(manifest_path.read_bytes()).hexdigest()}"
        ),
        "path": str(run.relative_to(ROOT)),
    }


def _artifact_contract(run: Path, relative: str) -> dict[str, object]:
    payload = run / relative
    return {
        "run_id": _load(run / "run.manifest.json")["run_id"],
        "artifact_path": relative,
        "artifact_sha256": sha256(payload.read_bytes()).hexdigest(),
    }


def build_document() -> dict[str, object]:
    predicted = _load(PREDICTED_RUN / "comparison/model-e2e-frontier.json")
    observed = _load(OBSERVED_RUN / "observation/observed-decomposition.json")
    identity = observed["identity"]
    issue30_run = MODEL_RUN
    issue30_contract = _load(issue30_run / "resolved/execution-contract.json")
    issue30_benchmark = _load(issue30_run / "observation/raw/benchmark.json")
    issue30_case = next(
        case for case in issue30_benchmark["cases"]
        if case["case_id"] == identity["benchmark_case"]
    )
    issue30_identity = _manifest_source(issue30_run)
    predicted_issue30 = next(
        source
        for source in predicted["evidence"]["source_bundles"]
        if source["issue"] == 30
    )
    observed_issue30 = next(
        source
        for source in _load(OBSERVED_RUN / "run.manifest.json")["source_runs"]
        if source["run_id"] == issue30_identity["run_id"]
    )
    report_identity = {
        "case": identity["benchmark_case"],
        "shape": identity["shape"],
        "dtype": issue30_contract["dtype"],
        "candidate_id": identity["candidate_id"],
        "hardware_cohort": identity["hardware_cohort"],
        "completion_boundary": identity["completion_boundary"],
    }
    if (
        predicted["model_id"] != "two-layer-transformer-prefill"
        or predicted["hardware_cohort"] != report_identity["hardware_cohort"]
        or predicted["axes"]["observation"]["value_ns"]
        != observed["baseline_e2e_observation"]["median_ns"]
        or predicted["axes"]["observation"]["value_ns"]
        != issue30_case["latency"]["median_ns"]
        or predicted_issue30["run_id"] != issue30_identity["run_id"]
        or predicted_issue30["manifest_sha256"]
        != issue30_identity["manifest_sha256"]
        or observed_issue30["manifest_sha256"]
        != issue30_identity["manifest_sha256"]
        or issue30_contract["shape"]["bindings"] != {
            "B": 1, "D": 64, "H": 512, "I": 2048, "NH": 8, "S": 512
        }
        or report_identity["shape"] != [1, 512, 512]
        or report_identity["candidate_id"]
        != "ascend-two-layer-transformer-pytorch-eager-v1"
        or issue30_contract["baseline_timing"]["completion_protocol"]
        != report_identity["completion_boundary"].replace(
            "end-npu-event", "end-event", 1
        )
    ):
        raise RuntimeError("#47/#48 same-boundary identity mismatch")
    predicted_leaves = predicted["coverage"]["predicted_leaves"]
    predicted_items = [
        {
            "stable_path": leaf["stable_path"],
            "operation_class": leaf["operation_class"],
            "status": leaf["status"],
            "duration_ns": leaf["duration_ns"],
            "standard_uncertainty_ns": None,
            "evidence_quality": (
                "evidence-qualified-candidate"
                if leaf["status"] == "known"
                else "structured-unknown"
            ),
            "evidence_refs": leaf["evidence_refs"],
            "accounting_interval": [index, index + 1],
            "evidence_boundaries": leaf["missing_operation_classes"],
        }
        for index, leaf in enumerate(predicted_leaves)
    ]
    observed_decomposition = observed["observed_decomposition"]
    observed_items = [
        {
            "stable_path": leaf["stable_path"],
            "operation_class": leaf.get("operation_class"),
            "status": "known",
            "duration_ns": leaf["duration_ns"],
            "standard_uncertainty_ns": leaf.get("standard_uncertainty_ns"),
            "evidence_quality": leaf.get("evidence_quality", "direct-qualified"),
            "evidence_refs": leaf.get("evidence_refs", []),
            "accounting_interval": [index, index + 1],
        }
        for index, leaf in enumerate(observed_decomposition.get("leaves", []))
    ]
    document: dict[str, object] = {
        "schema": "groundupscale.dev/e2e-gap-report-input/v1alpha3",
        "identity": report_identity,
        "policy": {
            "policy_id": "issue49-e2e-gap-materiality-v1",
            "version": "1",
            "top_k": 10,
            "mandatory_share_of_e2e": 0.10,
            "deep_diagnosis": {
                "minimum_absolute_gap_ns": 50_000,
                "minimum_relative_gap": 0.10,
            },
            "classifications": [
                "capability-model",
                "implementation-headroom",
                "materialization-layout",
                "scheduling-integration",
                "instrumentation",
                "noise",
            ],
        },
        "predicted": {
            "identity": report_identity,
            "status": "known" if predicted["status"] == "complete" else "unknown",
            "e2e_duration_ns": predicted["schedule"]["selected_feasible_duration_ns"],
            "standard_uncertainty_ns": predicted["uncertainty"]["combined_ns"],
            "bound_kind": "point-prediction",
            "items": predicted_items,
            "reason_code": "incomplete-schedule-frontier",
            "evidence_boundaries": [
                item["required_evidence"] for item in predicted["missing_evidence"]
            ],
            "required_next_measurement": (
                "qualify every mandatory leaf and schedule effect in the same Hardware Cohort"
            ),
            "evidence_refs": [_manifest_source(PREDICTED_RUN)["evidence_ref"]],
        },
        "observed": {
            "identity": report_identity,
            "status": observed_decomposition["status"],
            "e2e_duration_ns": observed_decomposition["e2e_duration_ns"],
            "standard_uncertainty_ns": None,
            "items": observed_items,
            "reason_code": observed_decomposition.get("reason_code"),
            "evidence_boundaries": observed_decomposition.get(
                "evidence_boundaries", []
            ),
            "required_next_measurement": observed_decomposition.get(
                "required_next_measurement"
            ),
            "accounting": "interval-union-or-critical-path",
            "evidence_refs": [_manifest_source(OBSERVED_RUN)["evidence_ref"]],
        },
        "source_bundles": [
            _manifest_source(PREDICTED_RUN),
            _manifest_source(OBSERVED_RUN),
            _manifest_source(MODEL_RUN),
        ],
        "iteration_report_derivation": {
            "schema": "groundupscale.dev/e2e-gap-report-value-derivation/v1alpha2",
            "report_policy": {
                "policy_id": "direct-measurement-observation-v2",
                "version": "2",
                "grade_minimum_intervals": {
                    "C": [0.70, 1.30],
                    "D": [0.50, 2.00],
                },
                "measured_uncertainty": "recorded-sample-statistics-only",
            },
            "prediction_model": {
                "policy_id": "serialized-resource-model-with-dispatch-floor-v1",
                "version": "1",
                "compute_efficiency": 0.50,
                "memory_efficiency": 0.50,
                "dispatch_floor_ns": 15_000.0,
                "schedule": "serialized-unfused",
                "purpose": "iteration-prior-only",
            },
            "observation_component_model": {
                "policy_id": "direct-measurements-only-v2",
                "version": "2",
                "purpose": "measured-column-only",
            },
            "artifacts": {
                "cost_ir": _artifact_contract(MODEL_RUN, "ir/cost.ir.json"),
                "hardware_backend": _artifact_contract(
                    MODEL_RUN, "prediction/hardware-backend.json"
                ),
                "baseline_observation": _artifact_contract(
                    OBSERVED_RUN, "observation/observed-decomposition.json"
                ),
                "execution_contract": _artifact_contract(
                    MODEL_RUN, "resolved/execution-contract.json"
                ),
                "schedule_authority": _artifact_contract(
                    PREDICTED_RUN, "comparison/model-e2e-frontier.json"
                ),
            },
        },
        "supersedes": [
            {
                "run_id": "issue49-20260814T0730Z-e2e-gap-report-v12",
                "path": "../issue49-20260814T0730Z-e2e-gap-report-v12",
                "manifest_sha256": sha256(
                    (LEGACY_REPORT_RUN / "run.manifest.json").read_bytes()
                ).hexdigest(),
            }
        ],
    }
    document["iteration_report"] = derive_tiered_iteration_report(document, ROOT)
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="一键生成两层 Transformer 中文预测—实测报告"
    )
    parser.add_argument(
        "--run-id",
        default=(
            "issue49-"
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-"
            "e2e-gap-report"
        ),
        help="唯一 Run Bundle ID；默认使用当前 UTC 时间",
    )
    parser.add_argument(
        "--artifact-store",
        type=Path,
        default=Path(__file__).parent / "evidence",
    )
    args = parser.parse_args()
    destination = write_gap_report_bundle(
        args.artifact_store,
        run_id=args.run_id,
        document=build_document(),
    )
    verification = verify_run_bundle(destination)
    if verification.get("passed") is not True:
        raise RuntimeError(
            f"generated report failed public verification: {verification}"
        )
    print(destination)
