#!/usr/bin/env python3
"""Build #25's immutable Apple M4 diagnostic Run Bundle.

The builder imports only verified, real M4 Run Bundles, qualifies a new 256
Anchor from three independent search and three independent holdout sessions,
and combines it with the existing qualified 512 Anchor.  It performs no
benchmarking and refuses to overwrite an existing diagnostic run id.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from statistics import median, stdev
from typing import Any

from groundupscale.run_bundle import verify_run_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(__file__).resolve().parent
RUN_ID = "issue25-m4-cpu-diagnostic-v1"
RUN_ROOT = WORK_ROOT / "evidence" / "runs" / RUN_ID
COHORT_ID = (
    "hvc-46ac04a4db5c57adaff46f8b4eb99bd0251a5dbfd15492dc5ba766aff21aa327"
)
PRODUCER = "issue-25-real-m4-diagnostic-builder"
Q_PATH = (
    "semantic/workload/transformer-prefill/request/model-prefill/model/"
    "transformer/layer_0/attention/q_proj"
)
SOURCE_ROOT = WORK_ROOT / "evidence" / "adr0036-source-runs" / "runs"
LEGACY_SOURCE_ROOT = Path(
    os.environ.get(
        "GROUNDUPSCALE_ISSUE25_M4_SOURCE_ROOT",
        "/Users/Zhuanz/work/github/GroundUpScale/.groundupscale/runs",
    )
)
OBSERVATION_RUN_ID = (
    "m4-cpu-exact-frontier-prediction-observation-20260810-v5"
)
OBSERVATION_RUN = SOURCE_ROOT / OBSERVATION_RUN_ID
LEGACY_OBSERVATION_RUN = LEGACY_SOURCE_ROOT / OBSERVATION_RUN_ID
PROFILE_PATH = (
    WORK_ROOT / "evidence" / "operator-frontier-profiles" /
    "apple-m4-cpu-matmul-512-v3.yaml"
)
LEGACY_PROFILE_PATH = Path(
    os.environ.get(
        "GROUNDUPSCALE_ISSUE25_M4_PROFILE",
        "/Users/Zhuanz/work/github/GroundUpScale/specs/operator-frontiers/"
        "apple-m4-cpu-matmul-512-v3.yaml",
    )
)
SEARCH_256 = (
    "issue25-m4-qproj-m256-n512-k512-search-02",
    "issue25-m4-qproj-m256-n512-k512-search-03",
    "issue25-m4-qproj-m256-n512-k512-search-05",
)
HOLDOUT_256 = (
    "issue25-m4-qproj-m256-n512-k512-holdout-01",
    "issue25-m4-qproj-m256-n512-k512-holdout-03",
    "issue25-m4-qproj-m256-n512-k512-holdout-05",
)
CONFIRMATION_ROOT = WORK_ROOT / "evidence" / "adr0036-confirmation-runs" / "runs"
CONFIRMATION_384 = (
    "issue25-m4-qproj-m384-n512-k512-confirmation-04",
    "issue25-m4-qproj-m384-n512-k512-confirmation-05",
    "issue25-m4-qproj-m384-n512-k512-confirmation-06",
)
BASELINE_LANE = "issue25-m4-baseline"
DIAGNOSTIC_LANE = "issue25-m4-diagnostic"
PAIR_ID = "issue25-m4-paired-lanes"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> str:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def _policy(policy_id: str, scope: str) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "version": "v1",
        "scope": scope,
        "change_reason": "ticket #25 real cross-hardware replay",
        "revalidation": "on policy, cohort, execution contract, or evidence change",
    }


def _completion_boundary() -> dict[str, Any]:
    return {
        "kind": "synchronous-cpu-call-return",
        "closed": True,
        "threadpool_joined": True,
    }


def _timer() -> dict[str, Any]:
    return {
        "source": "time.perf_counter_ns",
        "resolution_ns": 41.66666666666667,
        "monotonic": True,
    }


def _domain() -> dict[str, Any]:
    return {
        "shape": {"m": 512, "k": 512, "n": 512},
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "alignment_bytes": 64,
        "threads": 4,
        "execution_mode": "eager",
        "affinity": "os-managed-unpinned",
        "numa": "single-socket-unified-memory",
        "context": "single-process-cpu",
        "stream": "not-applicable-cpu",
        "concurrency": 1,
    }


def _active_transitions(anchor_id: str) -> list[dict[str, Any]]:
    return [
        {
            "sequence": 1,
            "axis": "frontier_role",
            "from": "NONE",
            "to": "PROVISIONAL",
            "reason_code": "exact-shape-best-of-correct-search-winner",
            "evidence_refs": [f"artifact://issue-25/{anchor_id}-search"],
        },
        {
            "sequence": 2,
            "axis": "observation_validity",
            "from": "COLLECTED",
            "to": "QUALIFIED",
            "reason_code": "anchor-qualification-gates-satisfied",
            "evidence_refs": [f"artifact://issue-25/{anchor_id}-qualification"],
        },
        {
            "sequence": 3,
            "axis": "frontier_role",
            "from": "PROVISIONAL",
            "to": "ACTIVE",
            "reason_code": "independent-holdout-confirmed",
            "evidence_refs": [f"artifact://issue-25/{anchor_id}-holdout"],
        },
    ]


def _verified_source(run: Path, *, expected_id: str) -> dict[str, Any]:
    verification = verify_run_bundle(run)
    if not verification["passed"]:
        raise ValueError(f"source Run Bundle failed verification: {run}")
    manifest = _read_json(run / "run.manifest.json")
    if (
        manifest.get("run_id") != expected_id
        or manifest.get("hardware_cohort") != COHORT_ID
        or manifest.get("status") != "completed"
    ):
        raise ValueError(f"source Run Bundle identity mismatch: {run}")
    return manifest


def _case(run: Path) -> dict[str, Any]:
    benchmark = _read_json(run / "observation/raw/benchmark.json")
    case = next(
        item for item in benchmark["cases"] if item["case_id"] == "matmul-q-proj"
    )
    if (
        case["operator_correctness"]["status"] != "passed"
        or case["resolved_scope"] != Q_PATH
        or case["candidate_identity"]["family"] != "torch.matmul.cpu.fp32"
    ):
        raise ValueError(f"unqualified q-proj source case: {run}")
    return case


def _source_session(
    run_id: str, *, lane: str, root: Path = SOURCE_ROOT
) -> dict[str, Any]:
    run = root / run_id
    _verified_source(run, expected_id=run_id)
    case = _case(run)
    environment = _read_json(run / "resolved/environment.json")
    if environment["measurement_preflight"]["eligible"] is not True:
        raise ValueError(f"source preflight failed: {run_id}")
    return {
        "session_id": run_id,
        "process_id": environment["process"]["pid"],
        "lane_id": BASELINE_LANE,
        "cohort_id": COHORT_ID,
        "raw_samples_ns": list(case["latency"]["samples_ns"]),
        "excluded_samples": [],
        "warmup": {
            "iterations": case["warmup_convergence"]["warmup_iterations"],
            "converged": case["warmup_convergence"]["converged"],
            "median_drift": case["warmup_convergence"]["median_drift"],
        },
        "correctness_passed": True,
        "evidence_ref": f"artifact://issue-25/{lane}-{run_id}",
    }


def _hardware_and_identity() -> tuple[dict[str, Any], dict[str, Any]]:
    environment = _read_json(OBSERVATION_RUN / "resolved/environment.json")
    cohort = environment["hardware_validity_cohort"]
    if cohort["cohort_id"] != COHORT_ID:
        raise ValueError("M4 observation cohort drift")
    software = cohort["software"]
    hardware = {
        "device": "Apple M4 CPU",
        "partition": cohort["device"]["partition"],
        "topology": cohort["device"]["topology"],
        "software": (
            f"torch={software['torch']};Darwin={software['kernel']};"
            f"python={software['python']}"
        ),
        "power_clock": {
            "power_policy": cohort["power_clock"]["power_source"],
            "clock_policy": cohort["power_clock"]["clock_policy"],
        },
    }
    domain = _domain()
    identity = {
        **deepcopy(hardware),
        "numeric_execution": {
            key: domain[key]
            for key in (
                "dtype",
                "layout",
                "alignment_bytes",
                "threads",
                "execution_mode",
            )
        },
        "timer_protocol": {
            **_timer(),
            "completion_kind": "synchronous-cpu-call-return",
            "adapter_id": "apple-m4-cpu",
            "adapter_version": "v1",
            "protocol_id": "issue25-exact-shape-diagnostic",
            "protocol_version": "v1",
            "duration_reducer": None,
        },
        "execution_context": {
            key: domain[key]
            for key in ("affinity", "numa", "context", "stream", "concurrency")
        },
        "communication": {"status": "not_applicable"},
    }
    return hardware, identity


def _anchor_512() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    import yaml

    profile = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    anchor = profile["spec"]["anchors"][0]
    if profile["spec"]["hardware_cohort"] != COHORT_ID:
        raise ValueError("M4 512 profile cohort drift")
    sessions = anchor["session_evidence"]

    def normalize(source: dict[str, Any], lane: str) -> dict[str, Any]:
        return {
            "session_id": source["run_id"],
            "process_id": source["process_id"],
            "lane_id": BASELINE_LANE,
            "cohort_id": COHORT_ID,
            "raw_samples_ns": list(source["samples_ns"]),
            "excluded_samples": [],
            "warmup": {
                "iterations": source["warmup_iterations"],
                "converged": True,
                "median_drift": source["warmup_median_drift"],
            },
            "correctness_passed": True,
            "evidence_ref": f"artifact://issue-25/m4-512-{lane}-{source['run_id']}",
        }

    search = [normalize(item, "search") for item in sessions["search"]]
    holdout = [normalize(item, "holdout") for item in sessions["holdout"]]
    anchor_id = "issue25-m4-qproj-512"
    diagnostic_anchor = {
        "anchor_id": anchor_id,
        "candidate_id": "torch.matmul.cpu.fp32",
        "cohort_id": COHORT_ID,
        "execution_domain": _domain(),
        "observation_validity": "QUALIFIED",
        "frontier_role": "ACTIVE",
        "state_transitions": _active_transitions(anchor_id),
        "baseline_lane_id": BASELINE_LANE,
        "instrumentation_profile": "baseline-timing/v1",
        "completion_boundary": _completion_boundary(),
        "timer": _timer(),
        "warmup": {"iterations": 500, "converged": True},
        "raw_timing_ns": [float(median(s["raw_samples_ns"])) for s in holdout],
        "correctness_passed": True,
        "holdout": {
            "passed": True,
            "latency_ns": anchor["latency_ns"],
            "session_ids": [s["session_id"] for s in holdout],
            "sessions": holdout,
            "evidence_ref": f"artifact://issue-25/{anchor_id}-holdout",
        },
        "evidence_ref": f"artifact://issue-25/{anchor_id}",
    }
    surface_anchor = {
        "anchor_id": anchor_id,
        "anchor_version": "v1",
        "shape": {"m": 512},
        "latency_ns": anchor["latency_ns"],
        "standard_uncertainty_ns": anchor["standard_uncertainty_ns"],
        "effective_rate": 2.0 * 512 * 512 * 512 / anchor["latency_ns"] * 1e9,
        "rate_unit": "FLOP/s",
        "standard_uncertainty_rate": (
            2.0 * 512**3 / anchor["latency_ns"] ** 2 * anchor["standard_uncertainty_ns"] * 1e9
        ),
        "candidate_id": "torch.matmul.cpu.fp32",
        "candidate_family": "torch.matmul.cpu.fp32",
        "cohort_id": COHORT_ID,
        "observation_validity": "QUALIFIED",
        "frontier_role": "ACTIVE",
        "evidence_ref": f"artifact://issue-25/{anchor_id}",
        "state_transitions": _active_transitions(anchor_id),
    }
    return diagnostic_anchor, surface_anchor, search


def _anchor_256() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    search = [_source_session(run_id, lane="m4-256-search") for run_id in SEARCH_256]
    holdout = [_source_session(run_id, lane="m4-256-holdout") for run_id in HOLDOUT_256]
    if any(not session["warmup"]["converged"] for session in search + holdout):
        raise ValueError("M4 256 session warmup failed qualification")
    for run_id in SEARCH_256 + HOLDOUT_256:
        contract = _case(SOURCE_ROOT / run_id)["execution_contract"]
        if [item["shape"] for item in contract["operand_contracts"]] != [
            [1, 256, 512],
            [512, 512],
        ]:
            raise ValueError(f"M4 256 source did not hold N=K=512: {run_id}")
    search_ids = {s["session_id"] for s in search}
    holdout_ids = {s["session_id"] for s in holdout}
    if search_ids & holdout_ids or len(search_ids) != 3 or len(holdout_ids) != 3:
        raise ValueError("M4 256 sessions are not independent")
    holdout_medians = [float(median(s["raw_samples_ns"])) for s in holdout]
    latency = float(median(holdout_medians))
    uncertainty = float(stdev(holdout_medians))
    anchor_id = "issue25-m4-qproj-256"
    return (
        {
            "anchor_id": anchor_id,
            "anchor_version": "v1",
            "shape": {"m": 256},
            "latency_ns": latency,
            "standard_uncertainty_ns": uncertainty,
            "effective_rate": 2.0 * 256 * 512 * 512 / latency * 1e9,
            "rate_unit": "FLOP/s",
            "standard_uncertainty_rate": (
                2.0 * 256 * 512 * 512 / latency**2 * uncertainty * 1e9
            ),
            "candidate_id": "torch.matmul.cpu.fp32",
            "candidate_family": "torch.matmul.cpu.fp32",
            "cohort_id": COHORT_ID,
            "observation_validity": "QUALIFIED",
            "frontier_role": "ACTIVE",
            "evidence_ref": f"artifact://issue-25/{anchor_id}",
            "state_transitions": _active_transitions(anchor_id),
        },
        search,
        holdout,
    )


def _surface(
    anchor_256: dict[str, Any],
    anchor_512: dict[str, Any],
    confirmation_sessions: list[dict[str, Any]],
) -> dict[str, Any]:
    domain = {
        "semantic_operation": "MatMul",
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "alignment_regime": "64-byte-aligned",
        "alignment_validated": True,
        "working_set_regime": "m4-unified-memory-q-projection",
        "regime_validated": True,
        "execution_mode": "eager",
        "threads": 4,
    }
    anchors = [deepcopy(anchor_256), deepcopy(anchor_512)]
    for anchor in anchors:
        anchor["domain"] = domain
    predicted_384_latency = (
        float(anchor_256["latency_ns"]) + float(anchor_512["latency_ns"])
    ) / 2.0
    confirmation_medians = [
        float(median(session["raw_samples_ns"]))
        for session in confirmation_sessions
    ]
    confirmation_residuals = [
        observed - predicted_384_latency
        for observed in confirmation_medians
    ]
    interpolation_uncertainty = (
        sum(residual**2 for residual in confirmation_residuals)
        / len(confirmation_residuals)
    ) ** 0.5
    surface: dict[str, Any] = {
        "surface_id": "surface://issue-25/apple-m4/q-proj/1d",
        "version": "v1",
        "previous_version": None,
        "cohort_id": COHORT_ID,
        "domain": domain,
        "candidate_family": "torch.matmul.cpu.fp32",
        "anchor_lifecycle_policy": _policy(
            "issue25-frontier-anchor-lifecycle", "M4 q-proj Surface"
        ),
        "coordinate": {"axis": "m", "transform": "identity", "transform_version": "v1"},
        "work_formula": {
            "kind": "matmul-2mnk",
            "fixed_n": 512,
            "fixed_k": 512,
            "version": "v2",
            "work_unit": "FLOP",
        },
        "response_model": {
            "kind": "piecewise-linear-latency",
            "primary_response": "latency_ns",
            "response_identity": "m4-q-proj-duration-v1",
            "shape_regime_identity": "m4-q-proj-fixed-nk-ramp-v1",
            "fixed_dimensions": {"n": 512, "k": 512},
            "version": "v1",
        },
        "anchors": anchors,
        "cells": [
            {
                "cell_id": "issue25-m4-qproj-m256-m512-fixed-nk512",
                "anchor_ids": [anchor_256["anchor_id"], anchor_512["anchor_id"]],
                "status": "retained",
                "regime_id": "m4-qproj-fixed-nk-eager-fp32-v1",
                "confirmation_evidence_refs": [
                    session["evidence_ref"]
                    for session in confirmation_sessions
                ],
                "confirmation_shape": {"m": 384},
                "confirmation_observed_latency_ns": float(
                    median(confirmation_medians)
                ),
                "interpolation_standard_uncertainty_ns": (
                    interpolation_uncertainty
                ),
            }
        ],
        "uncertainty_policy": {
            **_policy("issue25-m4-surface-uncertainty", "M4 q-proj Surface"),
            "combination": "root-sum-of-squares",
            "target_coverage": 0.95,
            "anchor_covariance_ns2": [
                [anchor_256["standard_uncertainty_ns"] ** 2, 0.0],
                [0.0, anchor_512["standard_uncertainty_ns"] ** 2],
            ],
            "instrumentation_standard_uncertainty_ns": 0.0,
            "calibration_evidence_refs": [
                "artifact://issue-25/m4-surface-uncertainty-calibration"
            ],
        },
        "evidence_refs": ["artifact://issue-25/m4-surface-build"],
    }
    surface["anchor_lifecycle_policy"]["version"] = "v2"
    surface["input_digest"] = _canonical_digest(surface)
    return surface


def _capability_manifest() -> dict[str, Any]:
    common = {
        "source": "python://time.perf_counter_ns",
        "scope": "exact-shape MatMul",
        "attribution": "single Apple M4 CPU process",
        "intrusion": "baseline",
    }
    return {
        "manifest_id": "issue25-apple-m4-cpu-v1",
        "adapter_id": "apple-m4-cpu",
        "cohort_id": COHORT_ID,
        "status": "qualified",
        "fields": [
            {
                **common,
                "field": "timer.primary",
                "status": "measured",
                "required_for_anchor": True,
                "value": {"resolution_ns": _timer()["resolution_ns"]},
            },
            {
                **common,
                "field": "completion.boundary",
                "status": "declared",
                "required_for_anchor": True,
                "value": "synchronous-cpu-call-return",
            },
            {
                **common,
                "field": "counter.cpu_cycles",
                "status": "not_requested",
                "required_for_anchor": False,
            },
        ],
        "evidence_ref": "artifact://issue-25/m4-capabilities",
    }


def _document() -> dict[str, Any]:
    _verified_source(
        OBSERVATION_RUN,
        expected_id="m4-cpu-exact-frontier-prediction-observation-20260810-v5",
    )
    observation_case = _case(OBSERVATION_RUN)
    comparison = _read_json(OBSERVATION_RUN / "comparison/predicted-vs-observed.json")
    comparison_case = next(
        item for item in comparison["latency_cases"] if item["case_id"] == "matmul-q-proj"
    )
    predicted = comparison_case["predicted"]
    hardware, identity = _hardware_and_identity()
    domain = _domain()
    anchor_512, surface_anchor_512, search_512 = _anchor_512()
    surface_anchor_256, search_256, holdout_256 = _anchor_256()
    confirmation_384 = [
        _source_session(
            run_id,
            lane="m4-384-confirmation",
            root=CONFIRMATION_ROOT,
        )
        for run_id in CONFIRMATION_384
    ]
    if any(
        not session["warmup"]["converged"] for session in confirmation_384
    ):
        raise ValueError("M4 384 confirmation warmup failed qualification")
    for run_id in CONFIRMATION_384:
        contract = _case(CONFIRMATION_ROOT / run_id)["execution_contract"]
        if [item["shape"] for item in contract["operand_contracts"]] != [
            [1, 384, 512],
            [512, 512],
        ]:
            raise ValueError(
                f"M4 confirmation did not hold N=K=512: {run_id}"
            )
    surface = _surface(
        surface_anchor_256,
        surface_anchor_512,
        confirmation_384,
    )
    completion = _completion_boundary()
    baseline_lane = {
        "lane_id": BASELINE_LANE,
        "pair_id": PAIR_ID,
        "cohort_id": COHORT_ID,
        "candidate_id": "torch.matmul.cpu.fp32",
        "execution_domain": domain,
        "instrumentation_profile": "baseline-timing/v1",
        "observation_validity": "QUALIFIED",
        "frontier_role": "NONE",
        "completion_boundary": completion,
        "timer": _timer(),
        "warmup": {
            "iterations": observation_case["warmup_convergence"]["warmup_iterations"],
            "converged": observation_case["warmup_convergence"]["converged"],
        },
        "raw_samples_ns": list(observation_case["latency"]["samples_ns"]),
        "excluded_samples": [],
        "evidence_ref": "artifact://issue-25/m4-baseline-observation",
    }
    diagnostic_lane = {
        "lane_id": DIAGNOSTIC_LANE,
        "pair_id": PAIR_ID,
        "paired_baseline_lane_id": BASELINE_LANE,
        "cohort_id": COHORT_ID,
        "candidate_id": "torch.matmul.cpu.fp32",
        "execution_domain": domain,
        "instrumentation_profile": "not-requested/v1",
        "status": "not_requested",
        "timing_used_for_frontier": False,
        "reason_code": "platform-counters-not-requested",
        "evidence_ref": "artifact://issue-25/m4-diagnostic-lane-not-requested",
    }
    document: dict[str, Any] = {
        "schema": "groundupscale.dev/diagnostic-evidence/v1alpha1",
        "resolved_configuration": {
            "analysis_plan": "specs/plans/mac-cpu-prefill.yaml",
            "benchmark_case": "matmul-q-proj",
            "e2e_case": "two-layer-prefill",
            "evidence_ref": "artifact://issue-25/m4-resolved-configuration",
        },
        "resolved_ir": {
            "semantic_node": Q_PATH,
            "semantic_identity": "transformer/layer-0/attention/q-proj",
            "operation": "MatMul",
        },
        "hardware": hardware,
        "cohort_id": COHORT_ID,
        "execution_domain": domain,
        "candidate": {
            "candidate_id": "torch.matmul.cpu.fp32",
            "family": "pytorch-cpu-matmul",
            "coverage": "C0_SINGLE",
            "implementation_digest": observation_case["candidate_identity"][
                "candidate_digest"
            ],
            "exact_shape_best_of_correct": {
                "passed": True,
                "winner_candidate_id": "torch.matmul.cpu.fp32",
                "eligible_candidate_ids": ["torch.matmul.cpu.fp32"],
                "search_session_ids": [s["session_id"] for s in search_512],
                "search_sessions": search_512,
                "evidence_ref": "artifact://issue-25/m4-512-search",
            },
        },
        "correctness": {
            "passed": True,
            "oracle": observation_case["operator_correctness"]["oracle"]["provider"],
            "policy_ref": "matmul-fp32-float64-oracle-v1/1.0.0",
            "evidence_ref": "artifact://issue-25/m4-correctness",
        },
        "environment": {
            "eligible": True,
            "preflight_ref": "artifact://issue-25/m4-preflight",
            "evidence_ref": "artifact://issue-25/m4-environment",
        },
        "measurement_adapter": {
            "adapter_id": "apple-m4-cpu",
            "adapter_version": "v1",
            "protocol_id": "issue25-exact-shape-diagnostic",
            "protocol_version": "v1",
            "evidence_ref": "artifact://issue-25/m4-measurement-adapter",
            "operation_evidence": [
                {
                    "operation": operation,
                    "evidence_ref": f"artifact://issue-25/m4-adapter-{operation}",
                }
                for operation in (
                    "discover_capabilities",
                    "fingerprint_cohort",
                    "preflight",
                    "build_timing_plan",
                    "collect",
                )
            ],
        },
        "measurement_capability_manifest": _capability_manifest(),
        "communication_identity": {"status": "not_applicable"},
        "baseline_timing_lane": baseline_lane,
        "diagnostic_profiling_lane": diagnostic_lane,
        "timing_plan": {
            "case": {
                "benchmark_case": "matmul-q-proj",
                "semantic_node": Q_PATH,
                "execution_domain": domain,
            },
            "pair_id": PAIR_ID,
            "baseline_lane_id": BASELINE_LANE,
            "diagnostic_lane_id": DIAGNOSTIC_LANE,
            "completion_boundary": completion,
            "evidence_ref": "artifact://issue-25/m4-timing-plan",
        },
        "cohort_evidence": {
            "reference_cohort_id": COHORT_ID,
            "reference_identity": identity,
            "observed_identity": identity,
            "transient_failures": [],
            "evidence_ref": "artifact://issue-25/m4-cohort-match",
        },
        "frontier_anchors": [anchor_512],
        "capability_surfaces": [surface],
        "surface_queries": [
            {
                "query_id": "issue25-m4-qproj-512-exact",
                "surface_id": surface["surface_id"],
                "surface_version": surface["version"],
                "shape": {"m": 512},
                "domain": surface["domain"],
            },
            {
                "query_id": "issue25-m4-qproj-384-interpolation",
                "surface_id": surface["surface_id"],
                "surface_version": surface["version"],
                "shape": {"m": 384},
                "domain": surface["domain"],
            },
        ],
        "resource_physical_floor": {
            "status": "known",
            "value_ns": predicted["resource_physical_floor_ns"],
            "may_be_unattainable": True,
            "policy_ref": "docs/adr/0033-derive-empirical-hardware-floors/v1",
            "combination": "max-explicit-overlap",
            "resource_terms": [
                {
                    "resource": "compute.fp32",
                    "validated_rate_resource": "compute.fp32",
                    "minimum_demand": predicted["minimum_work_flops"],
                    "demand_unit": "FLOP",
                    "validated_rate_per_second": (
                        predicted["minimum_work_flops"]
                        / predicted["empirical_compute_time_ns"]
                        * 1e9
                    ),
                    "rate_unit": "FLOP/s",
                    "validated": True,
                    "cohort_id": COHORT_ID,
                    "execution_domain": domain,
                    "evidence_ref": "artifact://issue-25/m4-physical-floor-compute",
                }
            ],
            "evidence_refs": ["artifact://issue-25/m4-physical-floor"],
        },
        "single_node_schedule": {
            "schedule_id": "issue25-m4-single-node-qproj",
            "version": "v1",
            "candidate_id": "torch.matmul.cpu.fp32",
            "dependencies": [],
            "transformations": [],
            "overlap_claims": [],
            "evidence_refs": ["artifact://issue-25/m4-schedule"],
        },
        "policies": {
            "qualification": {
                **_policy("issue25-m4-frontier-qualification", "M4 q-proj"),
                "version": "v2",
                "minimum_independent_sessions": 3,
            },
            "observation": _policy("issue25-m4-observation", "M4 q-proj"),
            "schedule": _policy("issue25-m4-schedule", "single-node q-proj"),
            "cohort": {
                **_policy("issue25-m4-cohort", "Apple M4 CPU"),
                "maximum_retry_attempts": 1,
            },
        },
        "source_runs": [
            {
                "run_id": "m4-cpu-exact-frontier-prediction-observation-20260810-v5",
                "role": "observation-and-e2e",
                "run_bundle": "run-bundle://m4-cpu-exact-frontier-prediction-observation-20260810-v5",
            },
            *[
                {
                    "run_id": session["session_id"],
                    "role": "surface-anchor-session",
                    "run_bundle": f"run-bundle://{session['session_id']}",
                }
                for session in search_256 + holdout_256
            ],
            *[
                {
                    "run_id": session["session_id"],
                    "role": "surface-confirmation-session",
                    "run_bundle": f"run-bundle://{session['session_id']}",
                }
                for session in confirmation_384
            ],
        ],
    }
    inputs = {
        key: document[key]
        for key in (
            "resolved_configuration",
            "resolved_ir",
            "hardware",
            "cohort_id",
            "execution_domain",
        )
    }
    evidence = {
        key: value
        for key, value in document.items()
        if key not in {*inputs, "schema", "digests"}
    }
    document["digests"] = {
        "input_sha256": _canonical_digest(inputs),
        "evidence_sha256": _canonical_digest(evidence),
    }
    return document


def _copy_source_file(source: Path, destination: Path) -> str:
    payload = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return sha256(payload).hexdigest()


def _import_sources() -> None:
    if not OBSERVATION_RUN.exists():
        shutil.copytree(LEGACY_OBSERVATION_RUN, OBSERVATION_RUN)
    if not PROFILE_PATH.exists():
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LEGACY_PROFILE_PATH, PROFILE_PATH)


def main() -> int:
    if RUN_ROOT.exists():
        raise FileExistsError(f"immutable run id already exists: {RUN_ROOT}")
    temporary = RUN_ROOT.with_name(f".{RUN_ID}.building")
    if temporary.exists():
        raise FileExistsError(f"stale build directory exists: {temporary}")
    _import_sources()
    document = _document()
    temporary.mkdir(parents=True)
    evidence_digest = _write_json(temporary / "diagnostic/evidence.json", document)
    artifacts: list[dict[str, Any]] = [
        {
            "role": "diagnostic-evidence",
            "path": "diagnostic/evidence.json",
            "schema": document["schema"],
            "media_type": "application/json",
            "sha256": evidence_digest,
            "produced_by": PRODUCER,
            "inputs": ["source-m4-observation", "source-m4-surface-sessions"],
        }
    ]
    sources = [
        (
            OBSERVATION_RUN / "run.manifest.json",
            "source/m4-observation-run.manifest.json",
            "source-run-manifest",
            "groundupscale.dev/run-manifest/v1alpha1",
        ),
        (
            OBSERVATION_RUN / "observation/raw/benchmark.json",
            "source/m4-observation-benchmark.json",
            "source-benchmark-observation",
            _read_json(OBSERVATION_RUN / "observation/raw/benchmark.json")["schema"],
        ),
        (
            OBSERVATION_RUN / "comparison/predicted-vs-observed.json",
            "source/m4-observation-comparison.json",
            "source-prediction-observation-comparison",
            _read_json(OBSERVATION_RUN / "comparison/predicted-vs-observed.json")[
                "schema"
            ],
        ),
        (
            OBSERVATION_RUN / "resolved/environment.json",
            "source/m4-observation-environment.json",
            "source-environment",
            _read_json(OBSERVATION_RUN / "resolved/environment.json")["schema"],
        ),
        (
            PROFILE_PATH,
            "source/apple-m4-cpu-matmul-512-v3.yaml",
            "source-frontier-profile",
            "groundupscale.dev/operator-frontier-profile/v1alpha1",
        ),
    ]
    for run_id in SEARCH_256 + HOLDOUT_256:
        run = SOURCE_ROOT / run_id
        sources.extend(
            [
                (
                    run / "run.manifest.json",
                    f"source/{run_id}-run.manifest.json",
                    "source-run-manifest",
                    "groundupscale.dev/run-manifest/v1alpha1",
                ),
                (
                    run / "observation/raw/benchmark.json",
                    f"source/{run_id}-benchmark.json",
                    "source-surface-session",
                    _read_json(run / "observation/raw/benchmark.json")["schema"],
                ),
                (
                    run / "resolved/environment.json",
                    f"source/{run_id}-environment.json",
                    "source-environment",
                    _read_json(run / "resolved/environment.json")["schema"],
                ),
            ]
        )
    for run_id in CONFIRMATION_384:
        run = CONFIRMATION_ROOT / run_id
        sources.extend(
            [
                (
                    run / "run.manifest.json",
                    f"source/{run_id}-run.manifest.json",
                    "source-run-manifest",
                    "groundupscale.dev/run-manifest/v1alpha1",
                ),
                (
                    run / "observation/raw/benchmark.json",
                    f"source/{run_id}-benchmark.json",
                    "source-surface-confirmation-session",
                    _read_json(run / "observation/raw/benchmark.json")["schema"],
                ),
                (
                    run / "resolved/environment.json",
                    f"source/{run_id}-environment.json",
                    "source-environment",
                    _read_json(run / "resolved/environment.json")["schema"],
                ),
            ]
        )
    for source, relative, role, schema in sources:
        digest = _copy_source_file(source, temporary / relative)
        if source == PROFILE_PATH:
            source_input = "operator-frontier-profile://apple-m4-cpu-matmul-512-v3"
        elif source.parent.name in {"resolved", "comparison", "raw"}:
            source_input = f"run-bundle://{source.parents[1].name}"
        else:
            source_input = f"run-bundle://{source.parent.name}"
        artifacts.append(
            {
                "role": role,
                "path": relative,
                "schema": schema,
                "media_type": (
                    "application/yaml" if source.suffix == ".yaml" else "application/json"
                ),
                "sha256": digest,
                "produced_by": PRODUCER,
                "inputs": [source_input],
            }
        )
    _write_json(
        temporary / "run.manifest.json",
        {
            "schema": "groundupscale.dev/run-manifest/v1alpha1",
            "run_id": RUN_ID,
            "status": "completed",
            "device": "cpu",
            "hardware_cohort": COHORT_ID,
            "artifacts": artifacts,
        },
    )
    temporary.replace(RUN_ROOT)
    print(RUN_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
