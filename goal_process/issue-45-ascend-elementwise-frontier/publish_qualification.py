from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from groundupscale.diagnostics import diagnose_run_bundle
from groundupscale.operator_frontier import OperatorFrontierBundleWriter
from groundupscale.run_bundle import verify_run_bundle


def _domain_policy(
    template: dict[str, object], domain: dict[str, object]
) -> dict[str, object]:
    common = dict(template["common_scope"])
    return {
        key: value
        for key, value in template.items()
        if key not in {"hardware_cohort", "domains", "common_scope"}
    } | {
        "policy_id": f"{template['policy_id']}-{domain['domain_id']}",
        "scope": {
            "hardware_cohort": template["hardware_cohort"],
            "operation": domain["operation"],
            "dtype": common["dtype"],
            "layout": common["layout"],
            "operand_kind": domain["operand_kind"],
            "sequence_distribution_mode": common[
                "sequence_distribution_mode"
            ],
            "result_shapes": [domain["result_shape"]],
            "candidate_ids": [domain["candidate_id"]],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--version", default="v2")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    root = workspace / "goal_process/issue-45-ascend-elementwise-frontier"
    evidence = root / "evidence"
    template = yaml.safe_load(
        (
            workspace
            / "specs/policies/ascend-910b2-elementwise-exact-frontier-v1.yaml"
        ).read_text(encoding="utf-8")
    )
    summaries: list[dict[str, object]] = []
    for domain in template["domains"]:
        domain_id = domain["domain_id"]
        search = sorted(
            (evidence / "runs").glob(
                f"issue45-{domain_id}-search-{args.session_id}-*"
            )
        )
        holdout = sorted(
            (evidence / "runs").glob(
                f"issue45-{domain_id}-holdout-{args.session_id}-*"
            )
        )
        if len(search) != 3 or len(holdout) != 3:
            raise SystemExit(f"{domain_id}: expected three search and holdout runs")
        run_id = f"issue45-{domain_id}-frontier-{args.session_id}-{args.version}"
        run = evidence / "runs" / run_id
        if not run.exists():
            run = OperatorFrontierBundleWriter().run(
                evidence,
                run_id=run_id,
                qualification_policy=_domain_policy(template, domain),
                search_runs=search,
                holdout_runs=holdout,
                confirmation_runs=[],
                query_sizes=[],
                query_shapes=(
                    {"result": domain["result_shape"]},
                    {"result": [1]},
                ),
            )
        verification = verify_run_bundle(run)
        diagnosis = diagnose_run_bundle(run)
        qualification = json.loads(
            (run / "frontier/qualification.json").read_text(encoding="utf-8")
        )
        summaries.append(
            {
                "domain_id": domain_id,
                "run_bundle": str(run),
                "verification_passed": verification["passed"],
                "qualification_status": qualification["status"],
                "anchor_latency_ns": (
                    qualification["anchors"][0]["latency_ns"]
                    if qualification["anchors"]
                    else None
                ),
                "standard_uncertainty_latency_ns": (
                    qualification["anchors"][0][
                        "standard_uncertainty_latency_ns"
                    ]
                    if qualification["anchors"]
                    else None
                ),
                "reason_code": qualification.get("reason_code"),
                "minimum_next_evidence_boundary": qualification.get(
                    "minimum_next_evidence_boundary"
                ),
                "queries": [
                    {
                        "shape": query["query_shape"],
                        "status": query["status"],
                        "reason_code": query.get("reason_code"),
                    }
                    for query in diagnosis["capability_surface_queries"]
                ],
            }
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(item["verification_passed"] for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
