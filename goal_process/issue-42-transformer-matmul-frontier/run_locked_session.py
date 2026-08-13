#!/usr/bin/env python3
"""Pre-registered issue #42 Ascend measurement/qualification session.

The outer invocation must be covered by the host-wide with-ascend-lock wrapper.
Each measurement is intentionally a fresh Python process so search and holdout
session identities are disjoint.  No supplemental round is permitted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from groundupscale.measurement_adapters.ascend_npu import AscendNpuMeasurementAdapter
from groundupscale.measurement_run import MeasurementRunBundleWriter
from groundupscale.run_bundle import verify_run_bundle
from groundupscale.transformer_matmul_frontier import (
    TransformerMatmulExactAnchorBundleWriter,
    TransformerMatmulFrontierBundleWriter,
    transformer_matmul_measurement_case,
)


DOMAINS = (
    "attention-context",
    "attention-qk",
    "mlp-contract",
    "mlp-expand",
    "projection",
)
SEARCH_SESSIONS = 3
HOLDOUT_SESSIONS = 3
WARMUP_ITERATIONS = 100
REPETITIONS = 100
INNER_ITERATIONS = 100
SEED = 20260813


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _collect_one(args: argparse.Namespace) -> int:
    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    run = MeasurementRunBundleWriter(
        AscendNpuMeasurementAdapter(logical_device_index=0)
    ).run(args.artifact_store, case=case, run_id=args.run_id)
    verification = verify_run_bundle(run)
    print(json.dumps(verification, ensure_ascii=False, sort_keys=True))
    return 0 if verification.get("passed") is True else 1


def _session(args: argparse.Namespace) -> int:
    if os.environ.get("GROUNDUPSCALE_ISSUE") != "42":
        raise SystemExit("GROUNDUPSCALE_ISSUE=42 is required")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0":
        raise SystemExit("ASCEND_RT_VISIBLE_DEVICES=0 is required")
    lock_owner = Path(
        "/home/t00906153/.groundupscale/locks/ascend-910b2-host.owner"
    )
    if not lock_owner.is_file():
        raise SystemExit("host lock owner metadata is missing; use with-ascend-lock")

    repository = Path(args.repository).resolve()
    artifact_store = Path(args.artifact_store).resolve()
    transformer_run = (repository / args.transformer_run).resolve()
    evidence_root = Path(args.evidence_root).resolve()
    tmp_root = Path(args.tmp_root).resolve()
    metadata_root = evidence_root / "session-metadata"
    cases_root = evidence_root / "preregistered-cases"
    for root in (artifact_store, evidence_root, tmp_root):
        root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()
    (metadata_root / "lock-owner-start.txt").parent.mkdir(
        parents=True, exist_ok=True
    )
    (metadata_root / "lock-owner-start.txt").write_text(
        lock_owner.read_text(encoding="utf-8"), encoding="utf-8"
    )

    preregistration = {
        "schema": "groundupscale.dev/issue42-preregistration/v1alpha1",
        "issue": 42,
        "session_id": args.session_id,
        "hardware_cohort": "ascend-npu-23b93a89d5fecc79",
        "device_visibility": "ASCEND_RT_VISIBLE_DEVICES=0",
        "candidate_set": [
            "torch.matmul",
            "torch.matmul.transpose-1-2-contiguous",
        ],
        "domain_order": list(DOMAINS),
        "lanes": {
            "search_sessions_per_domain": SEARCH_SESSIONS,
            "independent_holdout_sessions_per_domain": HOLDOUT_SESSIONS,
        },
        "timing": {
            "warmup_iterations": WARMUP_ITERATIONS,
            "repetitions": REPETITIONS,
            "inner_iterations": INNER_ITERATIONS,
            "primary_response": "latency_ns",
            "rate": "derived-only-from-declared-work-and-latency",
        },
        "qualification": {
            "correctness": "passed",
            "timing_quality": "passed",
            "maximum_lane_median_relative_range": 0.10,
            "maximum_search_holdout_relative_gap": 0.10,
            "session_identity": "run-id-and-process-identity-disjoint",
        },
        "stopping_condition": (
            "exactly one 3-search + 3-holdout round per domain; publish "
            "structured unknown for any explicit capability/evidence failure; "
            "no supplemental rounds"
        ),
    }
    _write_json(metadata_root / "preregistration.json", preregistration)

    results: dict[str, object] = {}
    anchor_runs: list[Path] = []
    for domain in DOMAINS:
        case = transformer_matmul_measurement_case(
            transformer_run,
            domain_class=domain,
            seed=SEED,
            warmup_iterations=WARMUP_ITERATIONS,
            repetitions=REPETITIONS,
            inner_iterations=INNER_ITERATIONS,
        )
        case_path = cases_root / f"{domain}.json"
        _write_json(case_path, case)
        lanes: dict[str, list[Path]] = {"search": [], "holdout": []}
        try:
            for lane, count in (
                ("search", SEARCH_SESSIONS),
                ("holdout", HOLDOUT_SESSIONS),
            ):
                for index in range(1, count + 1):
                    run_id = (
                        f"issue42-{args.session_id}-{domain}-{lane}-{index:02d}"
                    )
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--collect-one",
                            "--case",
                            str(case_path),
                            "--artifact-store",
                            str(artifact_store),
                            "--run-id",
                            run_id,
                        ],
                        check=True,
                        cwd=repository,
                        env={**os.environ, "TMPDIR": str(tmp_root)},
                    )
                    lanes[lane].append(artifact_store / "runs" / run_id)
            anchor_id = f"issue42-{args.session_id}-{domain}-exact-anchor"
            anchor_run = TransformerMatmulExactAnchorBundleWriter().run(
                artifact_store,
                run_id=anchor_id,
                search_runs=lanes["search"],
                holdout_runs=lanes["holdout"],
                candidate_runs=lanes["search"],
            )
            anchor_verification = verify_run_bundle(anchor_run)
            if anchor_verification.get("passed") is not True:
                raise RuntimeError(
                    f"exact Anchor verification failed: {anchor_verification}"
                )
            anchor_runs.append(anchor_run)
            anchor = json.loads(
                (anchor_run / "frontier/exact-anchor.json").read_text(
                    encoding="utf-8"
                )
            )
            results[domain] = {
                "status": anchor["status"],
                "reason_codes": anchor["reason_codes"],
                "anchor_run": str(anchor_run),
                "search_runs": [str(path) for path in lanes["search"]],
                "holdout_runs": [str(path) for path in lanes["holdout"]],
            }
        except Exception as error:  # bounded session publishes exact boundary
            results[domain] = {
                "status": "unknown",
                "reason_code": "bounded-domain-collection-or-qualification-failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_search_runs": [str(path) for path in lanes["search"]],
                "completed_holdout_runs": [str(path) for path in lanes["holdout"]],
                "additional_rounds_allowed": False,
            }

    frontier_run = TransformerMatmulFrontierBundleWriter().run(
        artifact_store,
        run_id=f"issue42-{args.session_id}-transformer-matmul-frontier",
        transformer_run=transformer_run,
        frontier_runs=anchor_runs,
    )
    frontier_verification = verify_run_bundle(frontier_run)
    ended_at = datetime.now(UTC).isoformat()
    metadata = {
        "schema": "groundupscale.dev/issue42-locked-session/v1alpha1",
        "issue": 42,
        "session_id": args.session_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "hardware_cohort": "ascend-npu-23b93a89d5fecc79",
        "device_visibility": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        "lock_owner_start": "lock-owner-start.txt",
        "domain_results": results,
        "frontier_run": str(frontier_run),
        "frontier_verification": frontier_verification,
    }
    _write_json(metadata_root / "session-result.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if frontier_verification.get("passed") is True else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-one", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--run-id")
    parser.add_argument("--repository", default=".")
    parser.add_argument("--transformer-run")
    parser.add_argument("--artifact-store")
    parser.add_argument("--evidence-root")
    parser.add_argument("--tmp-root")
    parser.add_argument("--session-id")
    args = parser.parse_args()
    if args.collect_one:
        return _collect_one(args)
    required = (
        "transformer_run",
        "artifact_store",
        "evidence_root",
        "tmp_root",
        "session_id",
    )
    if any(not getattr(args, name) for name in required):
        parser.error("full session requires transformer/artifact/evidence/tmp/session args")
    return _session(args)


if __name__ == "__main__":
    raise SystemExit(main())
