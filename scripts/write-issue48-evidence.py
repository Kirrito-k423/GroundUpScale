#!/usr/bin/env python3
"""Publish the immutable issue #48 structured-unknown authority bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from groundupscale.issue48_composition import compose_issue48_input
from groundupscale.model_e2e_frontier import write_model_e2e_frontier_bundle
from groundupscale.run_bundle import verify_run_bundle


DEFAULT_RUN_ID = "issue48-20260814T0001Z-schedule-frontier-unknown-v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    repository = args.repository.resolve()
    artifact_store = (
        repository
        / "goal_process/issue-48-schedule-achievable-frontier/evidence"
    )
    run = write_model_e2e_frontier_bundle(
        compose_issue48_input(repository),
        artifact_store,
        run_id=args.run_id,
    )
    verification = verify_run_bundle(run)
    if verification.get("passed") is not True:
        raise SystemExit(f"published bundle failed verification: {verification}")
    print(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
