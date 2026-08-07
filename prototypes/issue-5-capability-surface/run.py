#!/usr/bin/env python3.11
"""PROTOTYPE ONLY: batch replay and tiny TUI for issue 5."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Callable

from capability_surface_prototype import (
    aligned_only_surface,
    contradicted_cliff_surface,
    inspect_m4_observation,
    matmul_2d_surface,
    query_surface,
    smooth_1d_surface,
    smooth_1d_surface_v2,
    stable_digest,
)


PROTOTYPE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROTOTYPE_DIR.parents[1]
M4_OBSERVATION = REPOSITORY_ROOT / "goal_process/mac-transformer-ir-calibration-slice/evidence/apple-m4-cpu-microbenchmark-observation-v2.json"
RAW_RESULTS = PROTOTYPE_DIR / "results/raw-results.json"
RUN_COMMAND = "python3.11 prototypes/issue-5-capability-surface/run.py --batch"


def smooth_1d_scenario() -> dict[str, Any]:
    surface = smooth_1d_surface()
    aligned = {"alignment": "aligned", "dtype": "fp32"}
    non_aligned = {"alignment": "non_aligned", "dtype": "fp32"}
    queries = {
        "shape_128": query_surface(surface, (128.0,), aligned),
        "shape_201": query_surface(surface, (201.0,), non_aligned),
        "shape_512": query_surface(surface, (512.0,), aligned),
    }
    scan = [query_surface(surface, (float(shape),), non_aligned) for shape in range(128, 513)]
    scan_rates = [item["rate"] for item in scan]
    adjacent_deltas = [right - left for left, right in zip(scan_rates, scan_rates[1:])]
    return {
        "scenario": "smooth_1d_exact_and_interpolated",
        "input": surface.source_record(),
        "queries": queries,
        "continuity_scan": {
            "coordinates": list(range(128, 513)),
            "rates": scan_rates,
            "maximum_adjacent_delta": max(adjacent_deltas),
            "minimum_adjacent_delta": min(adjacent_deltas),
        },
    }


def matmul_2d_scenario() -> dict[str, Any]:
    surface = matmul_2d_surface()
    domain = {"alignment": "aligned", "dtype": "fp32", "K": "256"}
    return {
        "scenario": "matmul_2d_retained_triangle",
        "input": surface.source_record(),
        "inside_query": query_surface(surface, (256.0, 320.0), domain),
        "bounding_box_but_outside_query": query_surface(surface, (400.0, 400.0), domain),
    }


def alignment_rejection_scenario() -> dict[str, Any]:
    surface = aligned_only_surface()
    return {
        "scenario": "alignment_regime_rejection",
        "input": surface.source_record(),
        "query": query_surface(
            surface,
            (201.0,),
            {"alignment": "non_aligned", "dtype": "fp32"},
        ),
    }


def cliff_counterexample_scenario() -> dict[str, Any]:
    surface = contradicted_cliff_surface()
    return {
        "scenario": "confirmation_probe_rejects_smoothed_cliff",
        "input": surface.source_record(),
        "query": query_surface(
            surface,
            (201.0,),
            {"alignment": "mixed_confirmed", "dtype": "fp32"},
        ),
    }


def m4_counterexample_scenario() -> dict[str, Any]:
    return {
        "scenario": "ineligible_m4_bundle_cannot_supply_frontier_anchor",
        "observation": inspect_m4_observation(
            M4_OBSERVATION,
            str(M4_OBSERVATION.relative_to(REPOSITORY_ROOT)),
        ),
    }


def versioning_scenario() -> dict[str, Any]:
    domain = {"alignment": "non_aligned", "dtype": "fp32"}
    version_one = smooth_1d_surface()
    version_two = smooth_1d_surface_v2()
    return {
        "scenario": "new_anchor_creates_new_immutable_surface_version",
        "version_one_query": query_surface(version_one, (201.0,), domain),
        "version_two_query": query_surface(version_two, (201.0,), domain),
        "inputs": {
            "version_one": version_one.source_record(),
            "version_two": version_two.source_record(),
        },
    }


def build_scenarios() -> list[dict[str, Any]]:
    return [
        smooth_1d_scenario(),
        matmul_2d_scenario(),
        alignment_rejection_scenario(),
        cliff_counterexample_scenario(),
        m4_counterexample_scenario(),
        versioning_scenario(),
    ]


def _assertion(assertion_id: str, expected: Any, observed: Any, passed: bool) -> dict[str, Any]:
    return {
        "assertion_id": assertion_id,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def evaluate(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    smooth, two_dimensional, alignment, cliff, m4, versioning = scenarios
    q128 = smooth["queries"]["shape_128"]
    q201 = smooth["queries"]["shape_201"]
    q512 = smooth["queries"]["shape_512"]
    inside = two_dimensional["inside_query"]
    outside = two_dimensional["bounding_box_but_outside_query"]
    alignment_query = alignment["query"]
    cliff_query = cliff["query"]
    m4_observation = m4["observation"]
    version_one_query = versioning["version_one_query"]
    version_two_query = versioning["version_two_query"]
    required_provenance = {
        "surface_id",
        "surface_version",
        "hardware_cohort",
        "surface_input_digest",
        "coordinate_transform",
        "raw_coordinates",
        "transformed_coordinates",
        "cell_id",
        "anchors",
        "weights",
        "uncertainty",
        "status",
        "reason",
    }
    expected_step = 0.6e12 / 384
    assertions = [
        _assertion(
            "exact-knots-share-query-path",
            {"statuses": ["exact_anchor", "exact_anchor"], "cell": "smooth-line-128-512"},
            {"statuses": [q128["status"], q512["status"]], "cells": [q128["cell_id"], q512["cell_id"]]},
            q128["status"] == q512["status"] == "exact_anchor"
            and q128["cell_id"] == q201["cell_id"] == q512["cell_id"] == "smooth-line-128-512",
        ),
        _assertion(
            "shape-201-continuous-rate",
            1.3140625e12,
            q201["rate"],
            q201["status"] == "interpolated" and math.isclose(q201["rate"], 1.3140625e12, rel_tol=1e-12),
        ),
        _assertion(
            "retained-cell-scan-has-no-step",
            expected_step,
            smooth["continuity_scan"]["maximum_adjacent_delta"],
            math.isclose(smooth["continuity_scan"]["maximum_adjacent_delta"], expected_step, rel_tol=1e-12)
            and math.isclose(smooth["continuity_scan"]["minimum_adjacent_delta"], expected_step, rel_tol=1e-12),
        ),
        _assertion(
            "two-dimensional-inside-query",
            {"rate": 1.55e12, "weights": [1 / 6, 1 / 3, 1 / 2]},
            {"rate": inside["rate"], "weights": inside["weights"]},
            inside["status"] == "interpolated"
            and math.isclose(inside["rate"], 1.55e12, rel_tol=1e-12)
            and all(
                math.isclose(observed, expected, rel_tol=1e-12)
                for observed, expected in zip(inside["weights"], [1 / 6, 1 / 3, 1 / 2], strict=True)
            ),
        ),
        _assertion(
            "two-dimensional-box-is-not-domain",
            "unknown(outside_validated_domain)",
            f"{outside['status']}({outside['reason']})",
            outside["status"] == "unknown" and outside["reason"] == "outside_validated_domain" and outside["rate"] is None,
        ),
        _assertion(
            "unvalidated-alignment-is-rejected",
            "unknown(alignment_regime_unvalidated)",
            f"{alignment_query['status']}({alignment_query['reason']})",
            alignment_query["status"] == "unknown"
            and alignment_query["reason"] == "alignment_regime_unvalidated"
            and alignment_query["rate"] is None,
        ),
        _assertion(
            "cliff-confirmation-rejects-cell",
            "unknown(interpolation_error_exceeds_budget)",
            {
                "status": cliff_query["status"],
                "reason": cliff_query["reason"],
                "authoritative_rate": cliff_query["rate"],
                "residual_fraction": cliff_query["cell_validation"]["maximum_observed_residual_fraction"],
            },
            cliff_query["status"] == "unknown"
            and cliff_query["reason"] == "interpolation_error_exceeds_budget"
            and cliff_query["rate"] is None
            and cliff_query["cell_validation"]["passed"] is False,
        ),
        _assertion(
            "ineligible-m4-evidence-is-not-an-anchor",
            {"frontier_status": "unknown", "authoritative_rate": None},
            {
                "case_eligible": m4_observation["selected_case"]["case_eligible"],
                "environment_eligible": m4_observation["environment"]["eligible"],
                "frontier_status": m4_observation["frontier_status"],
                "authoritative_rate": m4_observation["authoritative_rate"],
            },
            m4_observation["selected_case"]["case_eligible"] is True
            and m4_observation["environment"]["eligible"] is False
            and m4_observation["frontier_status"] == "unknown"
            and m4_observation["authoritative_rate"] is None,
        ),
        _assertion(
            "query-provenance-is-complete",
            sorted(required_provenance),
            sorted(required_provenance.intersection(q201)),
            required_provenance.issubset(q201)
            and q201["uncertainty"] is not None
            and all(key in q201["uncertainty"] for key in ("anchor_standard_uncertainty", "interpolation_standard_uncertainty", "combined_standard_uncertainty", "rate_interval")),
        ),
        _assertion(
            "new-anchor-creates-new-immutable-surface-version",
            {
                "v1": {"status": "interpolated", "rate": 1.3140625e12},
                "v2": {"status": "exact_anchor", "rate": 1.28e12},
                "different_input_digests": True,
            },
            {
                "v1": {"version": version_one_query["surface_version"], "status": version_one_query["status"], "rate": version_one_query["rate"]},
                "v2": {"version": version_two_query["surface_version"], "status": version_two_query["status"], "rate": version_two_query["rate"]},
                "different_input_digests": version_one_query["surface_input_digest"] != version_two_query["surface_input_digest"],
            },
            version_one_query["surface_version"] == "v1"
            and version_one_query["status"] == "interpolated"
            and math.isclose(version_one_query["rate"], 1.3140625e12, rel_tol=1e-12)
            and version_two_query["surface_version"] == "v2"
            and version_two_query["status"] == "exact_anchor"
            and math.isclose(version_two_query["rate"], 1.28e12, rel_tol=1e-12)
            and version_one_query["surface_input_digest"] != version_two_query["surface_input_digest"],
        ),
    ]
    return assertions


def decision_payload() -> dict[str, Any]:
    scenarios = build_scenarios()
    assertions = evaluate(scenarios)
    return {
        "schema": "groundupscale.dev/throwaway-capability-surface-prototype/v1",
        "prototype_only": True,
        "question": "Can retained local simplicial cells provide continuous explainable versioned queries and refuse unsupported authority?",
        "run_command": RUN_COMMAND,
        "scenarios": scenarios,
        "counterexamples": [
            scenarios[2],
            scenarios[3],
            scenarios[4],
        ],
        "assertions": assertions,
        "hypothesis_verdict": "supported_for_method_contract" if all(item["passed"] for item in assertions) else "falsified",
        "production_disposition": "decision_only_do_not_evolve_or_copy_prototype_code",
    }


def batch() -> int:
    first = decision_payload()
    first_digest = stable_digest(first)
    second_digest = stable_digest(decision_payload())
    deterministic = _assertion(
        "identical-input-replay-is-deterministic",
        first_digest,
        second_digest,
        first_digest == second_digest,
    )
    first["assertions"].append(deterministic)
    first["decision_digest_before_replay_assertion"] = first_digest
    first["replay_digest"] = second_digest
    first["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "repository_root": str(REPOSITORY_ROOT),
    }
    first["hypothesis_verdict"] = (
        "supported_for_method_contract"
        if all(item["passed"] for item in first["assertions"])
        else "falsified"
    )

    for scenario in first["scenarios"]:
        print(json.dumps(scenario, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps({"assertions": first["assertions"], "verdict": first["hypothesis_verdict"]}, ensure_ascii=False, indent=2, sort_keys=True))

    RAW_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RAW_RESULTS.write_text(
        json.dumps(first, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"raw results: {RAW_RESULTS}")
    return 0 if first["hypothesis_verdict"] == "supported_for_method_contract" else 1


def interactive() -> int:
    actions: dict[str, tuple[str, Callable[[], dict[str, Any]]]] = {
        "1": ("smooth 1D exact/interpolated", smooth_1d_scenario),
        "2": ("2D retained triangle", matmul_2d_scenario),
        "3": ("alignment rejection", alignment_rejection_scenario),
        "4": ("cliff counterexample", cliff_counterexample_scenario),
        "5": ("M4 evidence rejection", m4_counterexample_scenario),
        "6": ("immutable surface version update", versioning_scenario),
    }
    current: dict[str, Any] = {
        "prototype_only": True,
        "question": "Does the local validated-cell query contract behave as intended?",
        "state": "choose a scenario",
    }
    while True:
        print("\033[2J\033[H", end="")
        print("\033[1mPROTOTYPE ONLY — current full state\033[0m")
        print(json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True))
        print()
        print("  ".join(f"\033[1m[{key}]\033[0m \033[2m{label}\033[0m" for key, (label, _) in actions.items()))
        print("\033[1m[a]\033[0m \033[2mrun and record all\033[0m  \033[1m[q]\033[0m \033[2mquit\033[0m")
        choice = input("> ").strip().lower()
        if choice == "q":
            return 0
        if choice == "a":
            return batch()
        if choice in actions:
            current = actions[choice][1]()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", action="store_true", help="run all pre-registered scenarios and record raw JSON")
    args = parser.parse_args()
    return batch() if args.batch else interactive()


if __name__ == "__main__":
    raise SystemExit(main())
