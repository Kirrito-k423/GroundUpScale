#!/usr/bin/env python3
"""Build #25's immutable ADR-0036 Ascend diagnostic Run Bundle.

The #32 diagnosis stays immutable.  This builder verifies and snapshots it,
then replaces only its rejected square-Shape Effective-Rate Surface with a
latency-primary, fixed-N/K M sweep backed by fifteen real NPU sessions.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from math import hypot
import os
from pathlib import Path
import shutil
from statistics import median, stdev
from typing import Any

from groundupscale.run_bundle import verify_run_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(__file__).resolve().parent
RUN_ID = "issue25-ascend-910b2-diagnostic-v1"
RUN_ROOT = WORK_ROOT / "evidence" / "runs" / RUN_ID
SOURCE_RUN = (
    REPOSITORY_ROOT
    / "goal_process/issue-32-ascend-diagnostic-bundle/evidence/runs"
    / "issue32-ascend-910b2-diagnostic-v1"
)
SESSION_ROOT = WORK_ROOT / "evidence" / "ascend-sessions"
COHORT_ID = "ascend-npu-23b93a89d5fecc79"
PRODUCER = "issue-25-ascend-latency-surface-builder"
SEARCH_256 = tuple(
    f"issue25-ascend-qproj-m256-n512-k512-search-0{index}"
    for index in range(1, 4)
)
HOLDOUT_256 = tuple(
    f"issue25-ascend-qproj-m256-n512-k512-holdout-0{index}"
    for index in range(1, 4)
)
SEARCH_512 = tuple(
    f"issue25-ascend-qproj-m512-n512-k512-search-0{index}"
    for index in range(1, 4)
)
HOLDOUT_512 = tuple(
    f"issue25-ascend-qproj-m512-n512-k512-holdout-0{index}"
    for index in range(1, 4)
)
CONFIRMATION_384 = tuple(
    f"issue25-ascend-qproj-m384-n512-k512-confirmation-0{index}"
    for index in range(1, 4)
)


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


def _session(session_id: str, lane: str, m: int) -> dict[str, Any]:
    value = _read_json(SESSION_ROOT / f"{session_id}.json")
    contract = value.get("execution_contract")
    if not (
        value.get("schema")
        == "groundupscale.dev/ascend-latency-surface-session/v1alpha1"
        and value.get("session_id") == session_id
        and value.get("lane") == lane
        and value.get("cohort_id") == COHORT_ID
        and value.get("device") == {"logical": "npu:0", "name": "Ascend910B2"}
        and isinstance(value.get("process_id"), int)
        and isinstance(contract, dict)
        and contract.get("shape") == {"m": m, "n": 512, "k": 512}
        and contract.get("candidate_id") == "torch.matmul"
        and contract.get("candidate_family") == "pytorch-ascend-matmul"
        and contract.get("completion_boundary")
        == "device-event-end-synchronize-plus-device-synchronize"
        and contract.get("response_identity")
        == "ascend-q-proj-device-event-duration-v1"
        and contract.get("shape_regime_identity")
        == "ascend-q-proj-fixed-nk-ramp-v1"
        and value.get("correctness", {}).get("passed") is True
        and value.get("warmup", {}).get("converged") is True
        and len(value.get("raw_samples_ns", [])) == 100
        and value.get("excluded_samples") == []
    ):
        raise ValueError(f"unqualified Ascend session: {session_id}")
    return value


def _transitions(anchor_id: str) -> list[dict[str, Any]]:
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


def _domain(source_surface: dict[str, Any]) -> dict[str, Any]:
    domain = deepcopy(source_surface["domain"])
    domain["working_set_regime"] = "fixed-n512-k512-m-ramp-256-512"
    domain["working_set_validated"] = True
    domain["regime_validated"] = True
    return domain


def _surface(
    source_document: dict[str, Any],
    sessions: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_surface = source_document["capability_surfaces"][0]
    domain = _domain(source_surface)
    holdout_256 = [float(sessions[item]["median_ns"]) for item in HOLDOUT_256]
    search_256 = [float(sessions[item]["median_ns"]) for item in SEARCH_256]
    latency_256 = float(median(holdout_256))
    uncertainty_256 = max(
        float(stdev(holdout_256)),
        (
            sum((item - latency_256) ** 2 for item in search_256)
            / len(search_256)
        ) ** 0.5,
    )
    holdout_512 = [float(sessions[item]["median_ns"]) for item in HOLDOUT_512]
    search_512 = [float(sessions[item]["median_ns"]) for item in SEARCH_512]
    latency_512 = float(median(holdout_512))
    uncertainty_512 = max(
        float(stdev(holdout_512)),
        (
            sum((item - latency_512) ** 2 for item in search_512)
            / len(search_512)
        ) ** 0.5,
    )
    predicted_384 = (latency_256 + latency_512) / 2.0
    confirmation_384 = [
        float(sessions[item]["median_ns"]) for item in CONFIRMATION_384
    ]
    interpolation_uncertainty = (
        sum((item - predicted_384) ** 2 for item in confirmation_384)
        / len(confirmation_384)
    ) ** 0.5

    def anchor(
        anchor_id: str,
        m: int,
        latency: float,
        uncertainty: float,
        evidence_ref: str,
    ) -> dict[str, Any]:
        work = 2 * m * 512 * 512
        rate = work / latency * 1_000_000_000.0
        rate_uncertainty = work * 1_000_000_000.0 * uncertainty / latency**2
        return {
            "anchor_id": anchor_id,
            "anchor_version": "v1",
            "candidate_id": "torch.matmul",
            "candidate_family": "pytorch-ascend-matmul",
            "cohort_id": COHORT_ID,
            "domain": domain,
            "shape": {"m": m},
            "latency_ns": latency,
            "standard_uncertainty_ns": uncertainty,
            "effective_rate": rate,
            "standard_uncertainty_rate": rate_uncertainty,
            "rate_unit": "FLOP/s",
            "observation_validity": "QUALIFIED",
            "frontier_role": "ACTIVE",
            "state_transitions": _transitions(anchor_id),
            "evidence_ref": evidence_ref,
        }

    anchors = [
        anchor(
            "issue25-ascend-qproj-256",
            256,
            latency_256,
            uncertainty_256,
            "artifact://issue-25/ascend-qproj-256-anchor",
        ),
        anchor(
            "issue25-ascend-qproj-512",
            512,
            latency_512,
            uncertainty_512,
            "artifact://issue-25/ascend-qproj-512-anchor",
        ),
    ]
    surface = {
        "surface_id": "surface://issue-25/ascend-910b2/q-proj/1d",
        "version": "v1",
        "previous_version": None,
        "cohort_id": COHORT_ID,
        "domain": domain,
        "candidate_family": "pytorch-ascend-matmul",
        "coordinate": {
            "axis": "m",
            "transform": "identity",
            "transform_version": "v1",
        },
        "response_model": {
            "kind": "piecewise-linear-latency",
            "primary_response": "latency_ns",
            "response_identity": "ascend-q-proj-device-event-duration-v1",
            "shape_regime_identity": "ascend-q-proj-fixed-nk-ramp-v1",
            "fixed_dimensions": {"n": 512, "k": 512},
            "version": "v1",
        },
        "work_formula": {
            "kind": "matmul-2mnk",
            "fixed_n": 512,
            "fixed_k": 512,
            "version": "v2",
            "work_unit": "FLOP",
        },
        "anchors": anchors,
        "cells": [
            {
                "cell_id": "issue25-ascend-qproj-m256-m512-fixed-nk512",
                "anchor_ids": [
                    "issue25-ascend-qproj-256",
                    "issue25-ascend-qproj-512",
                ],
                "regime_id": "ascend-qproj-fixed-nk-ramp-v1",
                "status": "retained",
                "confirmation_shape": {"m": 384},
                "confirmation_observed_latency_ns": float(
                    median(confirmation_384)
                ),
                "confirmation_evidence_refs": [
                    f"artifact://issue-25/ascend-surface-session-{item}"
                    for item in CONFIRMATION_384
                ],
                "interpolation_standard_uncertainty_ns": (
                    interpolation_uncertainty
                ),
            }
        ],
        "anchor_lifecycle_policy": {
            "policy_id": "issue25-ascend-frontier-anchor-lifecycle",
            "version": "v2",
            "scope": "Ascend q-proj latency Surface",
            "change_reason": "ADR 0036 fixed-N/K M-sweep replay for ticket #25",
            "revalidation": "on cohort, contract, response, regime, or evidence change",
        },
        "uncertainty_policy": {
            "policy_id": "issue25-ascend-surface-uncertainty",
            "version": "v1",
            "scope": "Ascend q-proj latency Surface",
            "change_reason": "independent holdout and boundary confirmation",
            "revalidation": "on cohort, contract, anchor, or confirmation change",
            "combination": "root-sum-of-squares",
            "target_coverage": 0.68,
            "anchor_covariance_ns2": [
                [uncertainty_256**2, 0.0],
                [0.0, uncertainty_512**2],
            ],
            "instrumentation_standard_uncertainty_ns": 20.0,
            "calibration_evidence_refs": [
                f"artifact://issue-25/ascend-surface-session-{item}"
                for item in CONFIRMATION_384
            ],
        },
        "evidence_refs": [
            "artifact://issue-25/ascend-qproj-256-anchor",
            "artifact://issue-25/ascend-qproj-512-anchor",
            "artifact://issue-25/ascend-remote-execution",
        ],
    }
    surface["input_digest"] = _canonical_digest(surface)
    queries = [
        {
            "query_id": "issue25-ascend-qproj-512-exact",
            "surface_id": surface["surface_id"],
            "surface_version": surface["version"],
            "shape": {"m": 512},
            "domain": domain,
        },
        {
            "query_id": "issue25-ascend-qproj-384-interpolation",
            "surface_id": surface["surface_id"],
            "surface_version": surface["version"],
            "shape": {"m": 384},
            "domain": domain,
        },
    ]
    return surface, queries


def _document(
    source: dict[str, Any], sessions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    document = deepcopy(source)
    document["resolved_configuration"] = {
        "analysis_plan": "issue-25-cross-hardware-replay",
        "benchmark_case": "ascend-910b2-q-projection-512",
    }
    document["resolved_ir"]["semantic_identity"] = (
        "transformer/layer-0/attention/q-proj"
    )
    surface, queries = _surface(document, sessions)
    document["capability_surfaces"] = [surface]
    document["surface_queries"] = queries
    old_anchor = document["frontier_anchors"][0]

    def normalize(session_id: str, lane: str) -> dict[str, Any]:
        session = sessions[session_id]
        return {
            "session_id": session_id,
            "process_id": session["process_id"],
            "lane_id": old_anchor["baseline_lane_id"],
            "cohort_id": COHORT_ID,
            "latency_ns": float(session["median_ns"]),
            "raw_samples_ns": list(session["raw_samples_ns"]),
            "excluded_samples": [],
            "evidence_ref": (
                f"artifact://issue-25/ascend-surface-session-{session_id}"
            ),
        }

    search_512 = [normalize(item, "search") for item in SEARCH_512]
    holdout_512 = [normalize(item, "holdout") for item in HOLDOUT_512]
    active_anchor = deepcopy(old_anchor)
    active_anchor["anchor_id"] = "issue25-ascend-qproj-512"
    active_anchor.pop("source_anchor_id", None)
    active_anchor["state_transitions"] = _transitions(
        "issue25-ascend-qproj-512"
    )
    active_anchor["raw_timing_ns"] = [
        item["latency_ns"] for item in holdout_512
    ]
    active_anchor["holdout"] = {
        "passed": True,
        "session_ids": list(HOLDOUT_512),
        "sessions": holdout_512,
        "latency_ns": float(
            median(item["latency_ns"] for item in holdout_512)
        ),
        "evidence_ref": "artifact://issue-25/issue25-ascend-qproj-512-holdout",
    }
    active_anchor["search"] = {
        "session_ids": list(SEARCH_512),
        "sessions": search_512,
        "evidence_ref": "artifact://issue-25/issue25-ascend-qproj-512-search",
    }
    active_anchor["evidence_ref"] = (
        "artifact://issue-25/ascend-qproj-512-anchor"
    )
    document["frontier_anchors"] = [active_anchor]
    exact_uncertainty = hypot(
        float(surface["anchors"][1]["standard_uncertainty_ns"]),
        float(
            surface["uncertainty_policy"][
                "instrumentation_standard_uncertainty_ns"
            ]
        ),
    )
    for item in document["diagnostic_trigger_input"]["items"]:
        item["predicted_ns"] = float(surface["anchors"][1]["latency_ns"])
        item["combined_uncertainty_ns"] = exact_uncertainty
    integration = document["shape_disambiguation_probes"][0][
        "integration_overhead_evidence"
    ]
    integration_frontier_ref = (
        "artifact://issue-25/ascend-integration-operator-frontier"
    )
    integration_contract_ref = (
        "artifact://issue-25/ascend-integration-contract"
    )
    qualification_ref = (
        "artifact://issue-25/ascend-frontier-qualification"
    )
    surface_ref = {
        "surface_id": surface["surface_id"],
        "version": surface["version"],
        "input_digest": surface["input_digest"],
    }
    exact_query = deepcopy(queries[0])
    integration["operator_frontier"] = {
        "schema": "groundupscale.dev/operator-frontier-evidence/v1alpha1",
        "stable_path": document["shape_disambiguation_probes"][0][
            "stable_path"
        ],
        "cohort_id": COHORT_ID,
        "execution_domain": deepcopy(document["execution_domain"]),
        "observation_validity": "QUALIFIED",
        "frontier_role": "ACTIVE",
        "latency_ns": float(surface["anchors"][1]["latency_ns"]),
        "combined_uncertainty_ns": exact_uncertainty,
        "surface": surface_ref,
        "uncertainty_basis": {
            "kind": "verified-capability-surface-query",
            "qualification_evidence_ref": qualification_ref,
            "query": exact_query,
            "source_policy": {
                key: surface["uncertainty_policy"][key]
                for key in (
                    "policy_id",
                    "version",
                    "combination",
                    "target_coverage",
                )
            },
            "latency_interval": {
                "lower_ns": float(surface["anchors"][1]["latency_ns"])
                - exact_uncertainty,
                "upper_ns": float(surface["anchors"][1]["latency_ns"])
                + exact_uncertainty,
            },
            "surface_uncertainty_ns": exact_uncertainty,
        },
        "uncertainty_policy": {
            "policy_id": surface["uncertainty_policy"]["policy_id"],
            "version": surface["uncertainty_policy"]["version"],
            "combination": surface["uncertainty_policy"]["combination"],
            "target_coverage": surface["uncertainty_policy"][
                "target_coverage"
            ],
        },
        "evidence_ref": integration_frontier_ref,
        "evidence_refs": [integration_frontier_ref],
    }
    integration["evidence_refs"] = [integration_contract_ref]
    document["source_runs"] = [
        {
            "run_id": "issue32-ascend-910b2-diagnostic-v1",
            "role": "diagnostic-and-exact-512-anchor",
            "run_bundle": (
                "run-bundle://issue32-ascend-910b2-diagnostic-v1"
            ),
        },
        *[
            {
                "run_id": session_id,
                "role": (
                    "surface-confirmation-session"
                    if sessions[session_id]["lane"] == "confirmation"
                    else "surface-anchor-session"
                ),
                "run_bundle": (
                    f"artifact://issue-25/ascend-surface-session-{session_id}"
                ),
            }
            for session_id in sorted(sessions)
        ],
    ]
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


def main() -> int:
    if RUN_ROOT.exists():
        raise FileExistsError(f"immutable run id already exists: {RUN_ROOT}")
    temporary = RUN_ROOT.with_name(f".{RUN_ID}.building")
    if temporary.exists():
        raise FileExistsError(f"stale build directory exists: {temporary}")
    verification = verify_run_bundle(SOURCE_RUN)
    if not verification["passed"]:
        raise ValueError(f"#32 source Bundle failed verification: {verification}")

    sessions = {
        session_id: _session(
            session_id,
            "confirmation" if session_id in CONFIRMATION_384 else (
                "search"
                if session_id in (*SEARCH_256, *SEARCH_512)
                else "holdout"
            ),
            384
            if session_id in CONFIRMATION_384
            else 512
            if session_id in (*SEARCH_512, *HOLDOUT_512)
            else 256,
        )
        for session_id in (
            *SEARCH_256,
            *HOLDOUT_256,
            *SEARCH_512,
            *HOLDOUT_512,
            *CONFIRMATION_384,
        )
    }
    if len({value["process_id"] for value in sessions.values()}) != 15:
        raise ValueError("Ascend sessions must come from fifteen independent processes")
    remote_execution_path = WORK_ROOT / "evidence/ascend-remote-execution.json"
    remote_execution = _read_json(remote_execution_path)
    if (
        remote_execution.get("final_device_health") != "OK"
        or remote_execution.get("process_ids")
        != sorted(value["process_id"] for value in sessions.values())
        or remote_execution.get("session_sha256")
        != {
            session_id: sha256(
                (SESSION_ROOT / f"{session_id}.json").read_bytes()
            ).hexdigest()
            for session_id in sorted(sessions)
        }
    ):
        raise ValueError("Ascend remote execution record does not match sessions")

    shutil.copytree(SOURCE_RUN, temporary)
    source_document = _read_json(SOURCE_RUN / "diagnostic/evidence.json")
    document = _document(source_document, sessions)
    evidence_digest = _write_json(
        temporary / "diagnostic/evidence.json", document
    )
    manifest = _read_json(SOURCE_RUN / "run.manifest.json")
    manifest["run_id"] = RUN_ID
    source_manifest = SOURCE_RUN / "run.manifest.json"
    manifest["source_manifest_integrity"] = "required"
    manifest["source_runs"] = [
        {
            "run_id": "issue32-ascend-910b2-diagnostic-v1",
            "role": "diagnostic-and-exact-512-anchor",
            "path": os.path.relpath(SOURCE_RUN, RUN_ROOT),
            "manifest_sha256": sha256(source_manifest.read_bytes()).hexdigest(),
        }
    ]
    for artifact in manifest["artifacts"]:
        if artifact.get("role") == "diagnostic-evidence":
            artifact["sha256"] = evidence_digest
            artifact["produced_by"] = PRODUCER
            artifact["inputs"] = [
                "run-bundle://issue32-ascend-910b2-diagnostic-v1",
                "source-latency-surface-session",
            ]
    for session_id, session in sorted(sessions.items()):
        source_path = SESSION_ROOT / f"{session_id}.json"
        relative = f"source/ascend-latency-surface/{session_id}.json"
        destination = temporary / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        manifest["artifacts"].append(
            {
                "role": "source-latency-surface-session",
                "uri": f"artifact://issue-25/ascend-surface-session-{session_id}",
                "path": relative,
                "schema": session["schema"],
                "media_type": "application/json",
                "sha256": sha256(source_path.read_bytes()).hexdigest(),
                "produced_by": PRODUCER,
                "inputs": [f"remote://A2-AK-225/{session_id}"],
            }
        )
    support_documents = {
        "artifact://issue-25/ascend-integration-contract": {
            "schema": (
                "groundupscale.dev/diagnostic-supporting-evidence/v1alpha1"
            ),
            "payload": {
                key: deepcopy(value)
                for key, value in document["shape_disambiguation_probes"][0][
                    "integration_overhead_evidence"
                ].items()
                if key != "evidence_refs"
            },
        },
        "artifact://issue-25/ascend-frontier-qualification": {
            "schema": (
                "groundupscale.dev/operator-frontier-qualification/v1alpha1"
            ),
            "status": "qualified",
            "hardware_cohort": COHORT_ID,
            "surface": deepcopy(document["capability_surfaces"][0]),
        },
        "artifact://issue-25/ascend-integration-operator-frontier": {
            key: deepcopy(value)
            for key, value in document["shape_disambiguation_probes"][0][
                "integration_overhead_evidence"
            ]["operator_frontier"].items()
            if key != "evidence_ref"
        },
        "artifact://issue-25/issue25-ascend-qproj-256-search": {
            "schema": "groundupscale.dev/diagnostic-supporting-evidence/v1alpha1",
            "lane": "search",
            "session_refs": [
                f"artifact://issue-25/ascend-surface-session-{item}"
                for item in SEARCH_256
            ],
        },
        "artifact://issue-25/issue25-ascend-qproj-256-qualification": {
            "schema": "groundupscale.dev/diagnostic-supporting-evidence/v1alpha1",
            "gate": "correctness-environment-timer-completion-independent-sessions",
            "status": "qualified",
            "session_refs": [
                f"artifact://issue-25/ascend-surface-session-{item}"
                for item in (*SEARCH_256, *HOLDOUT_256)
            ],
        },
        "artifact://issue-25/issue25-ascend-qproj-256-holdout": {
            "schema": "groundupscale.dev/diagnostic-supporting-evidence/v1alpha1",
            "lane": "holdout",
            "session_refs": [
                f"artifact://issue-25/ascend-surface-session-{item}"
                for item in HOLDOUT_256
            ],
        },
        "artifact://issue-25/ascend-qproj-256-anchor": {
            "schema": "groundupscale.dev/operator-frontier-evidence/v1alpha1",
            "anchor_id": "issue25-ascend-qproj-256",
            "search_session_refs": [
                f"artifact://issue-25/ascend-surface-session-{item}"
                for item in SEARCH_256
            ],
            "holdout_session_refs": [
                f"artifact://issue-25/ascend-surface-session-{item}"
                for item in HOLDOUT_256
            ],
            "latency_ns": next(
                anchor["latency_ns"]
                for anchor in document["capability_surfaces"][0]["anchors"]
                if anchor["anchor_id"] == "issue25-ascend-qproj-256"
            ),
            "status": "QUALIFIED",
            "frontier_role": "ACTIVE",
        },
        "artifact://issue-25/issue25-ascend-qproj-512-search": {
            "schema": "groundupscale.dev/diagnostic-supporting-evidence/v1alpha1",
            "lane": "search",
            "session_refs": [
                f"artifact://issue-25/ascend-surface-session-{item}"
                for item in SEARCH_512
            ],
        },
        "artifact://issue-25/issue25-ascend-qproj-512-qualification": {
            "schema": "groundupscale.dev/diagnostic-supporting-evidence/v1alpha1",
            "gate": "correctness-environment-timer-completion-independent-sessions",
            "status": "qualified",
            "session_refs": [
                f"artifact://issue-25/ascend-surface-session-{item}"
                for item in (*SEARCH_512, *HOLDOUT_512)
            ],
        },
        "artifact://issue-25/issue25-ascend-qproj-512-holdout": {
            "schema": "groundupscale.dev/diagnostic-supporting-evidence/v1alpha1",
            "lane": "holdout",
            "session_refs": [
                f"artifact://issue-25/ascend-surface-session-{item}"
                for item in HOLDOUT_512
            ],
        },
        "artifact://issue-25/ascend-qproj-512-anchor": {
            "schema": "groundupscale.dev/operator-frontier-evidence/v1alpha1",
            "anchor_id": "issue25-ascend-qproj-512",
            "search_session_refs": [
                f"artifact://issue-25/ascend-surface-session-{item}"
                for item in SEARCH_512
            ],
            "holdout_session_refs": [
                f"artifact://issue-25/ascend-surface-session-{item}"
                for item in HOLDOUT_512
            ],
            "latency_ns": next(
                anchor["latency_ns"]
                for anchor in document["capability_surfaces"][0]["anchors"]
                if anchor["anchor_id"] == "issue25-ascend-qproj-512"
            ),
            "status": "QUALIFIED",
            "frontier_role": "ACTIVE",
        },
    }
    for index, (uri, content) in enumerate(
        sorted(support_documents.items()), start=1
    ):
        relative = f"diagnostic/issue25-supporting/{index:02d}.json"
        digest = _write_json(temporary / relative, content)
        manifest["artifacts"].append(
            {
                "role": (
                    "operator-frontier-evidence"
                    if content["schema"].endswith(
                        "operator-frontier-evidence/v1alpha1"
                    )
                    else "source-frontier-qualification"
                    if content["schema"].endswith(
                        "operator-frontier-qualification/v1alpha1"
                    )
                    else "diagnostic-supporting-evidence"
                ),
                "uri": uri,
                "path": relative,
                "schema": content["schema"],
                "media_type": "application/json",
                "sha256": digest,
                "produced_by": PRODUCER,
                "inputs": ["source-latency-surface-session"],
            }
        )
    remote_relative = "source/ascend-latency-surface/remote-execution.json"
    remote_destination = temporary / remote_relative
    shutil.copyfile(remote_execution_path, remote_destination)
    manifest["artifacts"].append(
        {
            "role": "source-remote-execution",
            "uri": "artifact://issue-25/ascend-remote-execution",
            "path": remote_relative,
            "schema": remote_execution["schema"],
            "media_type": "application/json",
            "sha256": sha256(remote_execution_path.read_bytes()).hexdigest(),
            "produced_by": PRODUCER,
            "inputs": ["remote://A2-AK-225/issue25-adr0036-collection"],
        }
    )
    _write_json(temporary / "run.manifest.json", manifest)
    temporary.replace(RUN_ROOT)
    print(RUN_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
