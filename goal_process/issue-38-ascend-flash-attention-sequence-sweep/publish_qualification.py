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


def _runs(root: Path, lane: str) -> list[Path]:
    return sorted(root.glob(f"issue38-{lane}-s*-npu-fusion-attention-*"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--run-id", default="issue38-ascend-flash-attention-sequence-sweep-v1"
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    policy_path = workspace / (
        "specs/policies/"
        "ascend-910b2-flash-attention-bounded-sequence-sweep-v1.yaml"
    )
    evidence = workspace / (
        "goal_process/issue-38-ascend-flash-attention-sequence-sweep/evidence"
    )
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    run = evidence / "runs" / args.run_id
    if not run.exists():
        run = OperatorFrontierBundleWriter().run(
            evidence,
            run_id=args.run_id,
            qualification_policy=policy,
            search_runs=_runs(evidence / "runs", "main"),
            holdout_runs=_runs(evidence / "runs", "holdout"),
            confirmation_runs=_runs(evidence / "runs", "validation"),
            query_sizes=(1, 128, 1024, 4096, 6144, 8192, 9000),
        )
    verification = verify_run_bundle(run)
    qualification = json.loads(
        (run / "frontier/qualification.json").read_text(encoding="utf-8")
    )
    diagnosis = diagnose_run_bundle(run)
    summary = {
        "run_bundle": str(run),
        "verification_passed": verification["passed"],
        "qualification_status": qualification["status"],
        "reason_code": qualification.get("reason_code"),
        "stopping_decision": qualification.get("stopping_decision"),
        "response_attempt": qualification["surface"].get("response_attempt"),
        "queries": [
            {
                "shape": item["query_shape"],
                "status": item["status"],
                "reason_code": item.get("reason_code"),
                "shape_regime": item.get("shape_regime"),
            }
            for item in diagnosis["capability_surface_queries"]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
