#!/usr/bin/env python3
"""Apply the documented local M4 per-case noise gate to one CPU Bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MAXIMUM_IQR_OVER_MEDIAN = 0.03
SCHEMA = "groundupscale.dev/trusted-hardware-noise-check/v1alpha1"


def _write_result(path: str, result: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _invalid_result(reason_code: str) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "policy_id": "local-m4-benchmark-noise-v1",
        "maximum_iqr_over_median": MAXIMUM_IQR_OVER_MEDIAN,
        "passed": False,
        "failures": [],
        "reason_codes": [reason_code],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_bundle")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.run_bundle).resolve()
    try:
        manifest = json.loads(
            (root / "run.manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        _write_result(args.output, _invalid_result("run-manifest-invalid"))
        return 2
    matches = [
        artifact
        for artifact in manifest.get("artifacts", [])
        if artifact.get("role") == "benchmark-observation"
    ]
    if len(matches) != 1:
        _write_result(
            args.output,
            _invalid_result("benchmark-observation-role-invalid"),
        )
        return 2
    benchmark_path = (root / str(matches[0].get("path", ""))).resolve()
    if root not in benchmark_path.parents:
        _write_result(
            args.output,
            _invalid_result("benchmark-observation-path-invalid"),
        )
        return 2
    try:
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        cases = benchmark["cases"]
        if not isinstance(cases, list):
            raise ValueError("cases must be a list")
        failures = []
        for case in cases:
            case_id = case["case_id"]
            iqr_over_median = case["latency"]["iqr_over_median"]
            if not isinstance(case_id, str) or not isinstance(
                iqr_over_median, (int, float)
            ):
                raise ValueError("invalid benchmark case")
            if iqr_over_median > MAXIMUM_IQR_OVER_MEDIAN:
                failures.append(
                    {
                        "case_id": case_id,
                        "iqr_over_median": iqr_over_median,
                    }
                )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        _write_result(args.output, _invalid_result("benchmark-observation-invalid"))
        return 2
    result = {
        "schema": SCHEMA,
        "policy_id": "local-m4-benchmark-noise-v1",
        "maximum_iqr_over_median": MAXIMUM_IQR_OVER_MEDIAN,
        "passed": not failures,
        "failures": failures,
    }
    _write_result(args.output, result)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
