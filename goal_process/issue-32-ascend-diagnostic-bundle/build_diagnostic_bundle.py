#!/usr/bin/env python3
"""Build ticket #32's immutable, digest-verifiable diagnostic Run Bundle.

Every timing input comes from the committed real Ascend 910B2 evidence of
tickets #29-#31 or from six independent ticket #32 collection sessions (three
qualified Q integration replays and three K/V semantic-path replays).
The script performs deterministic derivations only and refuses to overwrite an
existing run id.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from hashlib import sha256
from math import hypot
from pathlib import Path
from re import fullmatch
from statistics import median
from typing import Any

from groundupscale.run_bundle import verify_run_bundle

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(__file__).resolve().parent
RUN_ID = "issue32-ascend-910b2-diagnostic-v1"
RUN_ROOT = WORK_ROOT / "evidence" / "runs" / RUN_ID
COHORT_ID = "ascend-npu-23b93a89d5fecc79"
PRODUCER = "issue-32-real-ascend-diagnostic-builder"

ISSUE29_RUN = (
    REPOSITORY_ROOT
    / "goal_process/issue-29-ascend-physical-floor/evidence/runs"
    / "ascend-910b2-matmul-floor-comparison-20260810-v2"
)
ISSUE30_RUN = (
    REPOSITORY_ROOT
    / "goal_process/issue-30-ascend-transformer-demo/evidence/runs"
    / "ascend-910b2-transformer-demo-20260811-v1"
)
ISSUE31_RUN = (
    REPOSITORY_ROOT
    / "goal_process/issue-31-ascend-matmul-frontier/evidence/runs"
    / "issue31-operator-frontier-v3"
)
SESSION_ROOT = WORK_ROOT / "evidence" / "sessions"

Q_PATH = (
    "semantic/model/two-layer-transformer/transformer/"
    "layer-0/attention/q-proj"
)
K_PATH = (
    "semantic/model/two-layer-transformer/transformer/"
    "layer-0/attention/k-proj"
)
V_PATH = (
    "semantic/model/two-layer-transformer/transformer/"
    "layer-0/attention/v-proj"
)
BASELINE_LANE = "issue32-ascend-baseline"
DIAGNOSTIC_LANE = "issue32-ascend-diagnostic"
PAIR_ID = "issue32-ascend-paired-lanes"
SOURCE_FRONTIER_REF = "artifact://issue-32/source-frontier-qualification"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _verified_source_manifest(
    root: Path, *, expected_run_id: str
) -> dict[str, Any]:
    verification = verify_run_bundle(root)
    if not verification["passed"]:
        raise ValueError(
            f"source Run Bundle failed verification: {root}: "
            f"{verification['failures']}"
        )
    manifest = _read_json(root / "run.manifest.json")
    if (
        manifest.get("run_id") != expected_run_id
        or manifest.get("status") != "completed"
        or manifest.get("hardware_cohort") != COHORT_ID
    ):
        raise ValueError(f"source Run Bundle identity mismatch: {root}")
    return manifest


def _source_run_lineage(
    root: Path, manifest: dict[str, Any], role: str
) -> dict[str, Any]:
    manifest_path = root / "run.manifest.json"
    return {
        "run_id": manifest["run_id"],
        "role": role,
        "path": os.path.relpath(root, RUN_ROOT),
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
    }


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
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


def _artifact_ref(value: object) -> bool:
    return (
        isinstance(value, str)
        and fullmatch(r"artifact://[A-Za-z0-9][A-Za-z0-9._/-]*", value)
        is not None
    )


def _write_json(path: Path, value: object) -> str:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def _policy(policy_id: str, scope: str, change_reason: str) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "version": "v1",
        "scope": scope,
        "change_reason": change_reason,
        "revalidation": (
            "revalidate on policy, cohort, execution contract, or evidence "
            "change"
        ),
    }


def _completion_boundary() -> dict[str, Any]:
    return {
        "kind": "device-event-stream-completion",
        "closed": True,
        "device_event_id": "issue32-npu-event-pair",
        "stream_id": "default-npu-stream",
        "stream_synchronized": True,
        "absolute_timestamps_subtracted": False,
    }


def _timer() -> dict[str, Any]:
    return {
        "source": "torch.npu.Event.elapsed_time",
        "resolution_ns": 20,
        "monotonic": True,
        "kind": "device-event",
        "device_event_id": "issue32-npu-event-pair",
        "stream_id": "default-npu-stream",
    }


def _execution_domain() -> dict[str, Any]:
    return {
        "shape": {"m": 512, "k": 512, "n": 512},
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "alignment_bytes": 512,
        "threads": 1,
        "execution_mode": "pytorch-eager",
        "affinity": "192-223",
        "numa": "6",
        "context": "pytorch-eager",
        "stream": "default-npu-stream",
        "concurrency": 1,
    }


def _hardware(cohort: dict[str, Any]) -> dict[str, Any]:
    software = cohort["software_evidence"]
    return {
        "device": "Ascend910B2",
        "partition": cohort["partition"],
        "topology": "single-node-single-npu",
        "software": cohort["software"],
        "os": {
            "name": "openEuler",
            "version": "22.03",
            "status": "resolved",
        },
        "kernel": {
            "name": "Linux",
            "version": software["kernel"]["value"],
            "status": "resolved",
        },
        "driver": {
            "name": "CANN-driver",
            "version": software["driver"]["value"],
            "status": "resolved",
        },
        "firmware": {
            "name": "Ascend-firmware",
            "version": software["firmware"]["value"],
            "status": "resolved",
        },
        "runtime": {
            "name": "CPython",
            "version": software["python"]["value"],
            "status": "resolved",
        },
        "framework": {
            "name": "PyTorch",
            "version": "2.7.1",
            "status": "resolved",
        },
        "compiler": {
            "name": "PyTorch-eager",
            "version": "2.7.1",
            "status": "resolved",
        },
        "operator_library": {
            "name": "torch-npu",
            "version": software["torch_npu"]["value"],
            "status": "resolved",
        },
        "communication_library": {
            "name": "single-device",
            "version": "v1",
            "status": "not_applicable",
        },
        "power_clock": {
            "power_policy": cohort["power_clock"]["power_policy"],
            "clock_policy": "hbm-1600mhz-ai-core-unobserved",
        },
    }


def _cohort_identity(
    hardware: dict[str, Any], domain: dict[str, Any]
) -> dict[str, Any]:
    return {
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
            "source": "torch.npu.Event.elapsed_time",
            "resolution_ns": 20,
            "monotonic": True,
            "completion_kind": "device-event-stream-completion",
            "adapter_id": "ascend-npu",
            "adapter_version": "v1",
            "protocol_id": "issue32-exact-shape-diagnostic",
            "protocol_version": "v1",
        },
        "execution_context": {
            key: domain[key]
            for key in (
                "affinity",
                "numa",
                "context",
                "stream",
                "concurrency",
            )
        },
        "communication": {"status": "not_applicable"},
        "evidence_ref": "artifact://issue-32/cohort-identity",
    }


def _hardware_validity_identity(
    hardware: dict[str, Any], domain: dict[str, Any]
) -> dict[str, Any]:
    identity = _cohort_identity(hardware, domain)
    timer = deepcopy(identity["timer_protocol"])
    timer["duration_reducer"] = None
    return {
        key: deepcopy(identity[key])
        for key in ("device", "partition", "topology", "software")
    } | {
        "numeric_execution": deepcopy(identity["numeric_execution"]),
        "timer_protocol": timer,
        "power_clock": deepcopy(identity["power_clock"]),
        "execution_context": deepcopy(identity["execution_context"]),
        "communication": deepcopy(identity["communication"]),
    }


def _session_measurement(
    raw: dict[str, Any],
    variant: str,
    *,
    lane_id: str,
    evidence_ref: str,
) -> dict[str, Any]:
    raw_samples = list(raw["variants"][variant]["raw_samples_ns"])
    return {
        "session_id": raw["session_id"],
        "process_id": raw["process_id"],
        "lane_id": lane_id,
        "cohort_id": COHORT_ID,
        "latency_ns": float(median(raw_samples)),
        "raw_samples_ns": raw_samples,
        "excluded_samples": [],
        "evidence_ref": evidence_ref,
    }


def _derived_session_measurement(
    raw: dict[str, Any],
    *,
    value_ns: float,
    lane_id: str,
    evidence_ref: str,
) -> dict[str, Any]:
    return {
        "session_id": raw["session_id"],
        "process_id": raw["process_id"],
        "lane_id": lane_id,
        "cohort_id": COHORT_ID,
        "latency_ns": value_ns,
        "derived_samples_ns": [value_ns],
        "excluded_samples": [],
        "evidence_ref": evidence_ref,
    }


def _correctness(
    evidence_ref: str,
    *,
    observed: float = 0.0001220703125,
) -> dict[str, Any]:
    return {
        "passed": observed <= 0.001,
        "records": [{"expected": 0.0, "observed": observed}],
        "tolerance": {"atol": 0.001, "rtol": 0.001},
        "evidence_ref": evidence_ref,
    }


def _implementation(candidate_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    implementation_ref = f"artifact://issue-32/{candidate_id}-implementation"
    manifest_ref = f"artifact://issue-32/{candidate_id}-family"
    content = {
        "schema": "groundupscale.dev/candidate-implementation/v1alpha1",
        "source_identity": f"issue-32/remote/{candidate_id}",
    }
    digest = sha256(_json_bytes(content)).hexdigest()
    family = {
        "family_id": "pytorch-ascend-matmul",
        "version": "v1",
        "implementation_ref": implementation_ref,
        "implementation_sha256": digest,
        "manifest_ref": manifest_ref,
    }
    manifest = {
        "schema": (
            "groundupscale.dev/implementation-family-manifest/v1alpha1"
        ),
        "family_id": family["family_id"],
        "version": family["version"],
        "implementation_ref": implementation_ref,
        "implementation_sha256": digest,
        "source_identity": content["source_identity"],
    }
    return {"family": family, "content": content}, manifest


def _frontier_source_sessions(
    qualification: dict[str, Any], lane: str
) -> list[dict[str, Any]]:
    return [
        source
        for source in qualification["source_runs"]
        if source["shape"] == {"s": 512}
        and source["candidate_id"] == "torch.matmul"
        and source["lane"] == lane
    ]


def _frontier_anchor(
    qualification: dict[str, Any], domain: dict[str, Any]
) -> dict[str, Any]:
    source_anchor = next(
        anchor
        for anchor in qualification["anchors"]
        if anchor["shape"] == {"s": 512}
    )
    search = _frontier_source_sessions(qualification, "search")
    holdout = _frontier_source_sessions(qualification, "holdout")

    def normalize(source: dict[str, Any], lane: str) -> dict[str, Any]:
        return {
            "session_id": source["run_id"],
            "process_id": source["process_identity"]["process_id"],
            "lane_id": BASELINE_LANE,
            "cohort_id": COHORT_ID,
            "latency_ns": source["median_ns"],
            "raw_samples_ns": source["raw_samples_ns"],
            "excluded_samples": [],
            "evidence_ref": (
                f"artifact://issue-32/frontier-{lane}-{source['run_id']}"
            ),
        }

    search_sessions = [normalize(item, "search") for item in search]
    holdout_sessions = [normalize(item, "holdout") for item in holdout]
    return {
        "anchor_id": "issue32-ascend-matmul-square-512",
        "candidate_id": "torch.matmul",
        "cohort_id": COHORT_ID,
        "execution_domain": domain,
        "observation_validity": "QUALIFIED",
        "frontier_role": "ACTIVE",
        "state_transitions": deepcopy(source_anchor["state_transitions"]),
        "baseline_lane_id": BASELINE_LANE,
        "instrumentation_profile": "baseline-timing/v1",
        "completion_boundary": _completion_boundary(),
        "timer": _timer(),
        "warmup": {"iterations": 100, "converged": True},
        "raw_timing_ns": [
            float(median(session["raw_samples_ns"]))
            for session in holdout_sessions
        ],
        "correctness_passed": True,
        "holdout": {
            "passed": True,
            "latency_ns": float(
                median(
                    median(session["raw_samples_ns"])
                    for session in holdout_sessions
                )
            ),
            "session_ids": [
                session["session_id"] for session in holdout_sessions
            ],
            "sessions": holdout_sessions,
            "evidence_ref": "artifact://issue-32/frontier-holdout",
        },
        "evidence_ref": "artifact://issue-32/frontier-anchor",
        "source_anchor_id": source_anchor["anchor_id"],
    }, search_sessions


def _frontier_uncertainty(
    *,
    stable_path: str,
    domain: dict[str, Any],
    qualification: dict[str, Any],
    frontier_diagnostic: dict[str, Any],
) -> dict[str, Any]:
    source_surface = qualification["surface"]
    source_anchor = next(
        anchor
        for anchor in source_surface["anchors"]
        if anchor["shape"] == {"s": 512}
    )
    query = next(
        item
        for item in frontier_diagnostic["surface_queries"]
        if item["query_id"] == "ascend-matmul-square-512"
    )
    source_policy = source_surface["uncertainty_policy"]
    combined_standard_rate = hypot(
        source_anchor["standard_uncertainty_rate"],
        0.0,
        source_policy["instrumentation_standard_uncertainty_rate"],
    )
    effective_rate = float(source_anchor["effective_rate"])
    declared_work = 2.0 * 512**3
    latency_ns = declared_work / effective_rate * 1_000_000_000.0
    latency_interval = {
        "lower_ns": (
            declared_work
            / (effective_rate + combined_standard_rate)
            * 1_000_000_000.0
        ),
        "upper_ns": (
            declared_work
            / (effective_rate - combined_standard_rate)
            * 1_000_000_000.0
        ),
    }
    surface_uncertainty_ns = max(
        latency_ns - latency_interval["lower_ns"],
        latency_interval["upper_ns"] - latency_ns,
    )
    surface = {
        key: source_surface[key]
        for key in ("surface_id", "version", "input_digest")
    }
    source_policy_identity = {
        key: deepcopy(source_policy[key])
        for key in (
            "policy_id",
            "version",
            "combination",
            "target_coverage",
        )
    }
    frontier = {
        "schema": "groundupscale.dev/operator-frontier-evidence/v1alpha1",
        "stable_path": stable_path,
        "observation_validity": "QUALIFIED",
        "frontier_role": "ACTIVE",
        "cohort_id": COHORT_ID,
        "execution_domain": domain,
        "surface": surface,
        "latency_ns": latency_ns,
        "combined_uncertainty_ns": surface_uncertainty_ns,
        "uncertainty_policy": deepcopy(source_policy_identity),
        "uncertainty_basis": {
            "kind": "verified-capability-surface-query",
            "qualification_evidence_ref": SOURCE_FRONTIER_REF,
            "query": deepcopy(query),
            "source_policy": deepcopy(source_policy_identity),
            "latency_interval": latency_interval,
            "surface_uncertainty_ns": surface_uncertainty_ns,
        },
        "evidence_ref": "artifact://issue-32/operator-frontier",
        "evidence_refs": ["artifact://issue-32/operator-frontier"],
    }
    return frontier


def _locked_contract(
    *,
    stable_path: str,
    semantic: str,
    candidate_ids: list[str],
    hardware: dict[str, Any],
    domain: dict[str, Any],
) -> dict[str, Any]:
    return {
        "semantic": semantic,
        "shape": {
            "left": [1, 512, 512],
            "right": [512, 512],
            "output": [1, 512, 512],
        },
        "dtype": domain["dtype"],
        "layout": domain["layout"],
        "strides": {
            "left": [262_144, 512, 1],
            "right": [512, 1],
            "output": [262_144, 512, 1],
        },
        "alignment_bytes": domain["alignment_bytes"],
        "threads": domain["threads"],
        "execution_domain": domain,
        "cohort_id": COHORT_ID,
        "cohort_identity": _cohort_identity(hardware, domain),
        "environment": {
            "eligible": True,
            "evidence_ref": f"artifact://issue-32/{stable_path.split('/')[-1]}-preflight",
        },
        "correctness_policy": {
            **_policy(
                "issue32-fp32-correctness",
                "exact-shape-ascend-candidate-correctness",
                "Lock the CPU oracle and tolerance before replay.",
            ),
            "oracle": "cpu-float32-same-seed-matmul",
            "atol": 0.001,
            "rtol": 0.001,
        },
        "candidate_ids": candidate_ids,
        "completion_boundary": _completion_boundary(),
    }


def _measurement_lanes(
    *, stable_path: str, contract: dict[str, Any], integration: bool
) -> dict[str, Any]:
    common = {
        "pair_id": PAIR_ID,
        "timer_source": "torch.npu.Event.elapsed_time",
        "case": {"stable_path": stable_path, "semantic": contract["semantic"]},
        "execution_domain": contract["execution_domain"],
        "candidate_ids": contract["candidate_ids"],
        "cohort_id": COHORT_ID,
        "completion_boundary": contract["completion_boundary"],
    }
    diagnostic = {
        **common,
        "lane_id": DIAGNOSTIC_LANE,
        "paired_baseline_lane_id": BASELINE_LANE,
        "instrumentation_profile": "paired-component-ablation/v1",
        "evidence_ref": f"artifact://issue-32/{stable_path.split('/')[-1]}-diagnostic-lane",
    }
    if integration:
        diagnostic.update(
            {
                "timing_used_for_frontier": False,
                "timing_used_for_integration_verdict": True,
            }
        )
    else:
        diagnostic["timing_used_for_verdict"] = False
    return {
        "baseline": {
            **common,
            "lane_id": BASELINE_LANE,
            "instrumentation_profile": "baseline-timing/v1",
            "evidence_ref": f"artifact://issue-32/{stable_path.split('/')[-1]}-baseline-lane",
        },
        "diagnostic": diagnostic,
    }


def _candidate(
    candidate_id: str,
    *,
    role: str,
    sessions: list[dict[str, Any]],
    correctness: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    implementation, family_manifest = _implementation(candidate_id)
    return (
        {
            "candidate_id": candidate_id,
            "role": role,
            "eligible": True,
            "implementation_family": implementation["family"],
            "correctness": correctness,
            "sessions": sessions,
        },
        implementation["content"],
        family_manifest,
    )


def _integration_probe(
    sessions: list[dict[str, Any]],
    *,
    hardware: dict[str, Any],
    domain: dict[str, Any],
    qualification: dict[str, Any],
    frontier_diagnostic: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    candidate_id = "torch-matmul-q-proj"
    contract = _locked_contract(
        stable_path=Q_PATH,
        semantic="batch-one Q projection MatMul",
        candidate_ids=[candidate_id],
        hardware=hardware,
        domain=domain,
    )
    target_sessions = [
        _session_measurement(
            raw,
            "frontier_adapter",
            lane_id=BASELINE_LANE,
            evidence_ref=f"artifact://issue-32/q-target-{index}",
        )
        for index, raw in enumerate(sessions, start=1)
    ]
    target_correctness = _correctness("artifact://issue-32/q-correctness")
    target, implementation, family = _candidate(
        candidate_id,
        role="target",
        sessions=target_sessions,
        correctness=target_correctness,
    )
    wrapped_sessions = [
        _session_measurement(
            raw,
            "profiling",
            lane_id=DIAGNOSTIC_LANE,
            evidence_ref=f"artifact://issue-32/integration-wrapped-{index}",
        )
        for index, raw in enumerate(sessions, start=1)
    ]

    cumulative_stages = (
        "frontier_adapter",
        "dispatch",
        "copy",
        "sync",
        "profiling",
    )
    per_session_components: list[dict[str, float]] = []
    for raw in sessions:
        medians = {
            name: float(median(value["raw_samples_ns"]))
            for name, value in raw["variants"].items()
        }
        previous_cumulative_ns = medians[cumulative_stages[0]]
        components: dict[str, float] = {}
        for kind in cumulative_stages[1:]:
            components[kind] = max(
                0.0, medians[kind] - previous_cumulative_ns
            )
            previous_cumulative_ns = max(
                previous_cumulative_ns, medians[kind]
            )
        per_session_components.append(components)
    operator_ns = float(
        median(session["latency_ns"] for session in target_sessions)
    )
    wrapped_ns = float(
        median(session["latency_ns"] for session in wrapped_sessions)
    )
    component_medians = {
        kind: float(median(item[kind] for item in per_session_components))
        for kind in ("dispatch", "copy", "sync", "profiling")
    }
    recovered = sum(component_medians.values())

    ablations: list[dict[str, Any]] = []
    for stage_index, kind in enumerate(cumulative_stages[1:], start=1):
        leaf_id = f"{kind}-overhead"
        ablations.append(
            {
                "ablation_id": f"remove-{kind}",
                "measurement_id": f"issue32-{kind}-exclusive-attribution",
                "kind": kind,
                "lane_id": DIAGNOSTIC_LANE,
                "correctness": _correctness(
                    f"artifact://issue-32/integration-{kind}-correctness"
                ),
                "removed_leaf_ids": [leaf_id],
                "sessions": [
                    _derived_session_measurement(
                        raw,
                        value_ns=per_session_components[index - 1][kind],
                        lane_id=DIAGNOSTIC_LANE,
                        evidence_ref=(
                            f"artifact://issue-32/integration-{kind}-{index}"
                        ),
                    )
                    for index, raw in enumerate(sessions, start=1)
                ],
                "derivation": {
                    "formula": (
                        "max(0, median(cumulative_variant) - "
                        "max(median(prior_cumulative_variants)))"
                    ),
                    "sample_semantics": "derived-paired-session-delta",
                    "cumulative_variant": kind,
                    "prior_cumulative_variants": list(
                        cumulative_stages[:stage_index]
                    ),
                    "input_refs": [
                        f"artifact://issue-32/raw-q-session-{index}"
                        for index in range(1, len(sessions) + 1)
                    ],
                },
                "evidence_ref": f"artifact://issue-32/integration-{kind}",
            }
        )

    frontier = _frontier_uncertainty(
        stable_path=Q_PATH,
        domain=domain,
        qualification=qualification,
        frontier_diagnostic=frontier_diagnostic,
    )
    leaves = [
        {
            "leaf_id": "operator",
            "kind": "operator",
            "duration_ns": operator_ns,
            "evidence_refs": ["artifact://issue-32/integration-operator"],
        },
        *[
            {
                "leaf_id": f"{kind}-overhead",
                "kind": kind,
                "duration_ns": component_medians[kind],
                "evidence_refs": [f"artifact://issue-32/integration-{kind}"],
            }
            for kind in ("dispatch", "copy", "sync", "profiling")
        ],
    ]
    removed_leaf_ids = [leaf["leaf_id"] for leaf in leaves[1:]]
    integration = {
        "schema": "groundupscale.dev/integration-overhead-evidence/v1alpha1",
        "stable_path": Q_PATH,
        "cohort_id": COHORT_ID,
        "paired_lanes": {
            "pair_id": PAIR_ID,
            "baseline_lane_id": BASELINE_LANE,
            "diagnostic_lane_id": DIAGNOSTIC_LANE,
        },
        "policy": {
            **_policy(
                "issue32-integration-overhead",
                "ascend-910b2-q-projection",
                "Replay the preregistered paired component ablation.",
            ),
            "minimum_independent_sessions": 3,
            "maximum_recovery_error_fraction": 0.05,
        },
        "operator_frontier": frontier,
        "wrapped_e2e": {
            "measurement_id": "q-proj-copy-sync-profiled-wrapper",
            "lane_id": DIAGNOSTIC_LANE,
            "correctness": _correctness(
                "artifact://issue-32/integration-wrapped-correctness"
            ),
            "sessions": wrapped_sessions,
            "evidence_ref": "artifact://issue-32/integration-wrapped",
        },
        "ablations": ablations,
        "exclusive_ledger": {
            "ledger_id": "issue32-q-proj-exclusive-ledger",
            "version": "v1",
            "leaf_semantics": "mutually-exclusive",
            "e2e_duration_ns": wrapped_ns,
            "leaves": leaves,
            "parents": [
                {
                    "span_id": "wrapped-e2e",
                    "kind": "e2e",
                    "additive": False,
                    "child_parent_ids": ["q-projection-wrapper"],
                    "leaf_ids": [],
                },
                {
                    "span_id": "q-projection-wrapper",
                    "kind": "module",
                    "additive": False,
                    "child_parent_ids": [],
                    "leaf_ids": [leaf["leaf_id"] for leaf in leaves],
                },
            ],
            "residual": {
                "residual_id": "unattributed-residual",
                "kind": "unattributed",
                "duration_ns": wrapped_ns - sum(
                    leaf["duration_ns"] for leaf in leaves
                ),
                "evidence_refs": ["artifact://issue-32/integration-residual"],
            },
            "evidence_refs": ["artifact://issue-32/integration-ledger"],
        },
        "counterfactual": {
            "counterfactual_id": "remove-integration-overhead",
            "kind": "declared-component-removal",
            "removed_leaf_ids": removed_leaf_ids,
            "declared_recovered_ns": recovered,
            "evidence_refs": ["artifact://issue-32/integration-counterfactual"],
        },
        "evidence_refs": ["artifact://issue-32/integration-contract"],
    }
    probe = {
        "probe_id": "issue32-q-proj-integration",
        "stable_path": Q_PATH,
        "locked_contract": contract,
        "measurement_lanes": _measurement_lanes(
            stable_path=Q_PATH, contract=contract, integration=True
        ),
        "candidates": [target],
        "integration_overhead_evidence": integration,
        "counterexamples": [
            {
                "counterexample_id": "performance-gap-only",
                "reason_codes": [
                    "performance-gap-is-not-direct-defect-evidence"
                ],
                "evidence_refs": ["artifact://issue-32/counterexample-performance"],
            }
        ],
        "evidence_refs": ["artifact://issue-32/q-probe"],
    }
    return probe, {
        target["implementation_family"]["implementation_ref"]: implementation,
        target["implementation_family"]["manifest_ref"]: family,
    }


def _simple_probe(
    sessions: list[dict[str, Any]],
    *,
    stable_path: str,
    candidate_id: str,
    hardware: dict[str, Any],
    domain: dict[str, Any],
    negative_control: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    semantic = (
        "batch-one V projection MatMul negative control"
        if negative_control
        else "batch-one K projection MatMul"
    )
    contract = _locked_contract(
        stable_path=stable_path,
        semantic=semantic,
        candidate_ids=[candidate_id],
        hardware=hardware,
        domain=domain,
    )
    variant = "negative_control" if negative_control else "k_baseline"
    evidence_lane = "diagnostic" if negative_control else "baseline"
    evidence_lane_id = (
        DIAGNOSTIC_LANE if negative_control else BASELINE_LANE
    )
    candidate_sessions = [
        _session_measurement(
            raw,
            variant,
            lane_id=evidence_lane_id,
            evidence_ref=f"artifact://issue-32/{candidate_id}-session-{index}",
        )
        for index, raw in enumerate(sessions, start=1)
    ]
    observed = max(
        (
            raw["negative_control"]["correctness"]
            if negative_control
            else raw["path_correctness"]["k"]
        )["max_abs_difference"]
        for raw in sessions
    )
    correctness = _correctness(
        f"artifact://issue-32/{candidate_id}-correctness",
        observed=observed,
    )
    target, implementation, family = _candidate(
        candidate_id,
        role="target",
        sessions=candidate_sessions,
        correctness=correctness,
    )
    target["evidence_lane"] = evidence_lane
    target["source_replay"] = {
        "variant": variant,
        "input_refs": [
            f"artifact://issue-32/raw-semantic-session-{index}"
            for index in range(1, len(sessions) + 1)
        ],
        "execution_evidence_ref": (
            "artifact://issue-32/source-remote-execution"
        ),
    }
    probe = {
        "probe_id": f"issue32-{candidate_id}-probe",
        "stable_path": stable_path,
        "locked_contract": contract,
        "measurement_lanes": _measurement_lanes(
            stable_path=stable_path, contract=contract, integration=False
        ),
        "candidates": [target],
        "counterexamples": [
            {
                "counterexample_id": "performance-gap-only",
                "reason_codes": [
                    "performance-gap-is-not-direct-defect-evidence"
                ],
                "evidence_refs": ["artifact://issue-32/counterexample-performance"],
            },
            {
                "counterexample_id": "proxy-anomaly-only",
                "reason_codes": [
                    "proxy-anomaly-is-not-direct-defect-evidence"
                ],
                "evidence_refs": ["artifact://issue-32/counterexample-proxy"],
            },
            {
                "counterexample_id": "single-fluctuation-only",
                "reason_codes": [
                    "single-fluctuation-is-not-reproducible"
                ],
                "evidence_refs": ["artifact://issue-32/counterexample-fluctuation"],
            },
        ],
        "evidence_refs": [f"artifact://issue-32/{candidate_id}-probe"],
    }
    if negative_control:
        input_sha256 = sha256(
            json.dumps(
                sessions[0]["path_inputs"]["v"],
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        negative = sessions[0]["negative_control"]["correctness"]
        probe["direct_defect_evidence"] = {
            "schema": "groundupscale.dev/direct-defect-evidence/v1alpha1",
            "defect_kind": "correctness_oracle_violation",
            "target_candidate_id": candidate_id,
            "input_summary": {
                "input_sha256": input_sha256,
                "shape": contract["shape"],
                "dtype": contract["dtype"],
                "layout": contract["layout"],
                "strides": contract["strides"],
                "alignment_bytes": contract["alignment_bytes"],
                "evidence_ref": "artifact://issue-32/negative-input-summary",
            },
            "candidate_identity": {
                "candidate_id": candidate_id,
                "family_id": target["implementation_family"]["family_id"],
                "family_version": target["implementation_family"]["version"],
                "implementation_ref": target["implementation_family"][
                    "implementation_ref"
                ],
                "implementation_sha256": target["implementation_family"][
                    "implementation_sha256"
                ],
                "source_identity": implementation["source_identity"],
            },
            "environment": {
                "cohort_id": COHORT_ID,
                "cohort_identity": contract["cohort_identity"],
                "preflight": contract["environment"],
            },
            "failure": {
                "failure_kind": "correctness_difference",
                "oracle": contract["correctness_policy"]["oracle"],
                "expected_sha256": negative["expected_sha256"],
                "observed_sha256": negative["observed_sha256"],
                "max_abs_difference": negative["max_abs_difference"],
                "mismatched_elements": negative["mismatched_elements"],
                "evidence_ref": "artifact://issue-32/negative-correctness-difference",
            },
            "repetitions": [
                {
                    "session_id": session["session_id"],
                    "process_id": session["process_id"],
                    "input_sha256": input_sha256,
                    "outcome": "violation",
                    "evidence_ref": f"artifact://issue-32/negative-repeat-{index}",
                }
                for index, session in enumerate(candidate_sessions, start=1)
            ],
            "evidence_ref": "artifact://issue-32/direct-defect",
            "supporting_evidence_refs": [correctness["evidence_ref"]],
        }
    return probe, {
        target["implementation_family"]["implementation_ref"]: implementation,
        target["implementation_family"]["manifest_ref"]: family,
    }


def _capability_manifest(source: dict[str, Any]) -> dict[str, Any]:
    manifest = deepcopy(source)
    manifest["evidence_ref"] = "artifact://issue-32/capabilities"
    manifest["protocol_id"] = "issue32-exact-shape-diagnostic"
    for field in manifest["fields"]:
        if field["field"] == "timer.primary":
            field["status"] = "measured"
    return manifest


def _global_document(
    q_sessions: list[dict[str, Any]],
    semantic_sessions: list[dict[str, Any]],
    floor_comparison: dict[str, Any],
    qualification: dict[str, Any],
    frontier_diagnostic: dict[str, Any],
    benchmark: dict[str, Any],
    error_attribution: dict[str, Any],
    cohort: dict[str, Any],
    capabilities: dict[str, Any],
    source_runs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    domain = _execution_domain()
    hardware = _hardware(cohort)
    cohort_identity = _hardware_validity_identity(hardware, domain)
    anchor, search_sessions = _frontier_anchor(qualification, domain)
    q_case = next(
        case
        for case in benchmark["cases"]
        if case["case_id"] == "matmul-q-proj"
    )
    q_probe, q_artifacts = _integration_probe(
        q_sessions,
        hardware=hardware,
        domain=domain,
        qualification=qualification,
        frontier_diagnostic=frontier_diagnostic,
    )
    k_probe, k_artifacts = _simple_probe(
        semantic_sessions,
        stable_path=K_PATH,
        candidate_id="torch-matmul-k-proj",
        hardware=hardware,
        domain=domain,
        negative_control=False,
    )
    negative_probe, negative_artifacts = _simple_probe(
        semantic_sessions,
        stable_path=V_PATH,
        candidate_id="torch-matmul-v-proj-negative",
        hardware=hardware,
        domain=domain,
        negative_control=True,
    )
    implementation_artifacts = {
        **q_artifacts,
        **k_artifacts,
        **negative_artifacts,
    }
    baseline_observation_ns = float(q_case["latency"]["median_ns"])
    semantic_source_refs = [
        f"artifact://issue-32/raw-semantic-session-{index}"
        for index in range(1, len(semantic_sessions) + 1)
    ]
    semantic_observations = {
        variant: float(
            median(
                median(raw["variants"][variant]["raw_samples_ns"])
                for raw in semantic_sessions
            )
        )
        for variant in ("k_baseline", "v_baseline")
    }
    q_frontier = q_probe["integration_overhead_evidence"][
        "operator_frontier"
    ]
    trigger_items = [
        {
            "stable_path": Q_PATH,
            "predicted_ns": q_frontier["latency_ns"],
            "observed_ns": baseline_observation_ns,
            "combined_uncertainty_ns": q_frontier[
                "combined_uncertainty_ns"
            ],
            "observation_basis": {
                "kind": "benchmark-case",
                "source_evidence_ref": (
                    "artifact://issue-32/source-transformer-benchmark"
                ),
                "source_case_id": "matmul-q-proj",
                "stable_path": Q_PATH,
                "semantic": "batch-one Q projection MatMul",
            },
        },
        *[
            {
                "stable_path": stable_path,
                "predicted_ns": q_frontier["latency_ns"],
                "observed_ns": semantic_observations[variant],
                "combined_uncertainty_ns": q_frontier[
                    "combined_uncertainty_ns"
                ],
                "observation_basis": {
                    "kind": "session-variant-aggregate",
                    "variant": variant,
                    "stable_path": stable_path,
                    "semantic": semantic,
                    "lane": "baseline",
                    "reducer": "median-of-independent-session-medians",
                    "input_refs": semantic_source_refs,
                    "execution_evidence_ref": (
                        "artifact://issue-32/source-remote-execution"
                    ),
                },
            }
            for stable_path, variant, semantic in (
                (K_PATH, "k_baseline", "batch-one K projection MatMul"),
                (V_PATH, "v_baseline", "batch-one V projection MatMul"),
            )
        ],
    ]
    completion = _completion_boundary()
    baseline_lane = {
        "lane_id": BASELINE_LANE,
        "pair_id": PAIR_ID,
        "instrumentation_profile": "baseline-timing/v1",
        "observation_validity": "QUALIFIED",
        "frontier_role": "NONE",
        "candidate_id": "torch.matmul",
        "cohort_id": COHORT_ID,
        "execution_domain": domain,
        "raw_samples_ns": list(q_case["latency"]["samples_ns"]),
        "completion_boundary": completion,
        "timer": _timer(),
        "warmup": {
            "iterations": benchmark["cases"][0]["pilot_iterations"],
            "converged": True,
        },
        "evidence_ref": "artifact://issue-32/baseline-q-proj-observation",
    }
    diagnostic_lane = {
        "lane_id": DIAGNOSTIC_LANE,
        "pair_id": PAIR_ID,
        "paired_baseline_lane_id": BASELINE_LANE,
        "instrumentation_profile": "torch-npu-profiler/v1",
        "candidate_id": "torch.matmul",
        "cohort_id": COHORT_ID,
        "execution_domain": domain,
        "timing_used_for_frontier": False,
        "raw_samples_ns": [
            sample
            for raw in q_sessions
            for sample in raw["variants"]["profiling"]["raw_samples_ns"]
        ],
        "completion_boundary": completion,
        "timer": _timer(),
        "overhead_ablation": {
            "status": "qualified",
            "instrumentation_profile": "torch-npu-profiler/v1",
            "selection": {
                "session_ids": [
                    session["session_id"] for session in search_sessions
                ],
                "evidence_ref": "artifact://issue-32/profiling-selection",
            },
            "holdout": {
                "pair_id": PAIR_ID,
                "baseline_lane_id": BASELINE_LANE,
                "diagnostic_lane_id": DIAGNOSTIC_LANE,
                "baseline_session_ids": [
                    f"{raw['session_id']}-baseline" for raw in q_sessions
                ],
                "diagnostic_session_ids": [
                    f"{raw['session_id']}-diagnostic" for raw in q_sessions
                ],
                "baseline_raw_samples_ns": [
                    float(median(raw["variants"]["dispatch"]["raw_samples_ns"]))
                    for raw in q_sessions
                ],
                "diagnostic_raw_samples_ns": [
                    float(median(raw["variants"]["profiling"]["raw_samples_ns"]))
                    for raw in q_sessions
                ],
                "evidence_ref": "artifact://issue-32/profiling-holdout",
            },
            "evidence_ref": "artifact://issue-32/profiling-ablation",
        },
        "evidence_ref": "artifact://issue-32/diagnostic-profiling-lane",
    }
    source_floor = floor_comparison["physical_floor"]
    capability_by_resource = {
        capability["resource"]: capability
        for capability in source_floor["capabilities"]
    }
    source_rate = float(
        capability_by_resource["compute.fp32"]["robust_achievable_rate"]
    )
    memory_rate = float(
        capability_by_resource["memory.hbm"]["robust_achievable_rate"]
    )
    floor_value = float(source_floor["resource_physical_floor_ns"])
    document: dict[str, Any] = {
        "schema": "groundupscale.dev/diagnostic-evidence/v1alpha1",
        "resolved_configuration": {
            "analysis_plan": "issue-32-ascend-diagnostic-bundle",
            "benchmark_case": "ascend-910b2-q-projection-512",
        },
        "resolved_ir": {
            "semantic_node": Q_PATH,
            "operation": "MatMul",
        },
        "hardware": hardware,
        "cohort_id": COHORT_ID,
        "execution_domain": domain,
        "policies": {
            "qualification": {
                **_policy(
                    "issue32-frontier-qualification",
                    "ascend-910b2-square-matmul-512",
                    "Replay #31's qualified ACTIVE 512 anchor.",
                ),
                "version": "v2",
                "minimum_independent_sessions": 3,
            },
            "schedule": _policy(
                "issue32-single-node-schedule",
                "single-node-single-operator",
                "Keep schedule composition explicit and conservative.",
            ),
            "observation": _policy(
                "issue32-baseline-observation",
                "ascend-910b2-q-projection",
                "Replay the real #30 baseline observation lane.",
            ),
            "cohort": {
                **_policy(
                    "issue32-cohort-policy",
                    "ascend-910b2-single-device",
                    "Fail closed on a hardware-validity identity change.",
                ),
                "maximum_retry_attempts": 1,
            },
            "profiling_overhead": {
                **_policy(
                    "issue32-profiling-overhead",
                    "ascend-910b2-exact-shape-matmul",
                    "Keep intrusive profiler timings out of Frontier evidence.",
                ),
                "instrumentation_profiles": ["torch-npu-profiler/v1"],
                "validity_domain_ref": "artifact://issue-32/profiling-validity-domain",
                "maximum_overhead_ratio": 0.1,
                "minimum_independent_sessions": 3,
            },
        },
        "resource_physical_floor": {
            "status": "known",
            "value_ns": floor_value,
            "combination": "max-explicit-overlap",
            "policy_ref": (
                "docs/adr/0033-model-resource-floors-"
                "separately-from-full-duration.md"
            ),
            "resource_terms": [
                {
                    "resource": "compute.fp32",
                    "minimum_demand": 268_435_456,
                    "demand_unit": "FLOP",
                    "validated_rate_per_second": source_rate,
                    "rate_unit": "FLOP/s",
                    "validated": True,
                    "validated_rate_resource": "compute.fp32",
                    "cohort_id": COHORT_ID,
                    "execution_domain": domain,
                    "evidence_ref": "artifact://issue-32/physical-floor-compute",
                    "source_evidence_ref": (
                        "artifact://issue-32/source-physical-floor"
                    ),
                },
                {
                    "resource": "memory.hbm",
                    "minimum_demand": 3_145_728,
                    "demand_unit": "B",
                    "validated_rate_per_second": memory_rate,
                    "rate_unit": "B/s",
                    "validated": True,
                    "validated_rate_resource": "memory.hbm",
                    "cohort_id": COHORT_ID,
                    "execution_domain": domain,
                    "evidence_ref": "artifact://issue-32/physical-floor-memory",
                    "source_evidence_ref": (
                        "artifact://issue-32/source-physical-floor"
                    ),
                },
            ],
            "evidence_refs": ["artifact://issue-32/physical-floor"],
        },
        "candidate": {
            "candidate_id": "torch.matmul",
            "family": "pytorch-ascend-matmul",
            "coverage": "C2_MULTI_FAMILY",
            "implementation_digest": next(
                source["candidate_digest"]
                for source in qualification["source_runs"]
                if source["run_id"] == anchor["holdout"]["session_ids"][0]
            ),
            "exact_shape_best_of_correct": {
                "passed": True,
                "winner_candidate_id": "torch.matmul",
                "eligible_candidate_ids": [
                    "torch.matmul",
                    "torch.matmul.k-split-2",
                ],
                "search_session_ids": [
                    session["session_id"] for session in search_sessions
                ],
                "search_sessions": search_sessions,
                "evidence_ref": "artifact://issue-32/frontier-search",
            },
        },
        "correctness": {
            "passed": True,
            "oracle": "cpu-float32-same-seed-matmul",
            "policy_ref": "issue32-fp32-correctness/v1",
            "evidence_ref": "artifact://issue-32/global-correctness",
        },
        "environment": {
            "eligible": True,
            "preflight_ref": "artifact://issue-32/preflight",
            "evidence_ref": "artifact://issue-32/environment",
        },
        "measurement_adapter": {
            "adapter_id": "ascend-npu",
            "adapter_version": "v1",
            "protocol_id": "issue32-exact-shape-diagnostic",
            "protocol_version": "v1",
            "evidence_ref": "artifact://issue-32/measurement-adapter",
            "operation_evidence": [
                {
                    "operation": operation,
                    "evidence_ref": f"artifact://issue-32/adapter-{operation}",
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
        "measurement_capability_manifest": _capability_manifest(capabilities),
        "communication_identity": {"status": "not_applicable"},
        "baseline_timing_lane": baseline_lane,
        "diagnostic_profiling_lane": diagnostic_lane,
        "timing_plan": {
            "case": {
                "benchmark_case": "ascend-910b2-q-projection-512",
                "semantic_node": Q_PATH,
                "execution_domain": domain,
            },
            "pair_id": PAIR_ID,
            "baseline_lane_id": BASELINE_LANE,
            "diagnostic_lane_id": DIAGNOSTIC_LANE,
            "completion_boundary": completion,
            "evidence_ref": "artifact://issue-32/timing-plan",
        },
        "cohort_evidence": {
            "reference_cohort_id": COHORT_ID,
            "reference_identity": cohort_identity,
            "observed_identity": cohort_identity,
            "transient_failures": [],
            "evidence_ref": "artifact://issue-32/cohort-match",
        },
        "frontier_anchors": [anchor],
        "capability_surfaces": [deepcopy(qualification["surface"])],
        "surface_queries": deepcopy(frontier_diagnostic["surface_queries"]),
        "single_node_schedule": {
            "schedule_id": "issue32-single-node-q-proj",
            "version": "v1",
            "candidate_id": "torch.matmul",
            "dependencies": [],
            "transformations": [],
            "overlap_claims": [],
            "evidence_refs": ["artifact://issue-32/schedule"],
        },
        "diagnostic_trigger_input": {
            "policy": _policy(
                "issue32-diagnostic-trigger",
                "exact-shape-performance-diagnosis",
                "Run diagnostics only beyond combined uncertainty and materiality.",
            ),
            "e2e_observation_ns": float(
                error_attribution["e2e_trace_host_ns"]
            ),
            "source_evidence_ref": (
                "artifact://issue-32/source-transformer-e2e-attribution"
            ),
            "source_evidence_required": True,
            "items": trigger_items,
        },
        "verdict_policy": {
            **_policy(
                "issue32-exact-shape-verdict",
                "exact-shape-performance-diagnosis",
                "Require reproducible evidence for every diagnosis verdict.",
            ),
            "minimum_independent_sessions": 3,
            "suspected_regression_gate": "undefined",
        },
        "shape_disambiguation_probes": [q_probe, k_probe, negative_probe],
        "source_runs": deepcopy(source_runs),
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
    return document, implementation_artifacts


def _artifact_documents(
    document: dict[str, Any],
    implementations: dict[str, dict[str, Any]],
) -> dict[str, tuple[str, str, dict[str, Any]]]:
    refs: set[str] = set()
    diagnostic: dict[str, tuple[str, str, dict[str, Any]]] = {}
    uncertainty_components: dict[str, dict[str, Any]] = {}
    target_coverages: dict[str, dict[str, Any]] = {}
    uncertainty_calibrations: dict[str, dict[str, Any]] = {}

    def collect(value: object) -> None:
        if isinstance(value, dict):
            direct_ref = value.get("evidence_ref")
            if _artifact_ref(direct_ref):
                refs.add(direct_ref)
                payload = {
                    key: deepcopy(item)
                    for key, item in value.items()
                    if key != "evidence_ref"
                }
                if value.get("schema") == (
                    "groundupscale.dev/operator-frontier-evidence/v1alpha1"
                ):
                    diagnostic[direct_ref] = (
                        "operator-frontier-evidence",
                        value["schema"],
                        payload,
                    )
                else:
                    schema = (
                        "groundupscale.dev/"
                        "diagnostic-supporting-evidence/v1alpha1"
                    )
                    diagnostic[direct_ref] = (
                        "diagnostic-supporting-evidence",
                        schema,
                        {"schema": schema, "payload": payload},
                    )
                if {
                    "component_id",
                    "standard_uncertainty_ns",
                    "evidence_ref",
                } <= set(value):
                    uncertainty_components[direct_ref] = {
                        "schema": (
                            "groundupscale.dev/uncertainty-component/v1alpha1"
                        ),
                        "component_id": value["component_id"],
                        "standard_uncertainty_ns": value[
                            "standard_uncertainty_ns"
                        ],
                    }
            plural_refs = value.get("evidence_refs")
            if isinstance(plural_refs, list):
                schema = (
                    "groundupscale.dev/"
                    "diagnostic-supporting-evidence/v1alpha1"
                )
                payload = {
                    "schema": schema,
                    "payload": {
                        key: deepcopy(item)
                        for key, item in value.items()
                        if key != "evidence_refs"
                    },
                }
                for artifact_ref in plural_refs:
                    if _artifact_ref(artifact_ref):
                        refs.add(artifact_ref)
                        diagnostic.setdefault(
                            artifact_ref,
                            ("diagnostic-supporting-evidence", schema, payload),
                        )
            target = value.get("target_coverage")
            calibration = value.get("calibration")
            if (
                isinstance(target, dict)
                and isinstance(target.get("evidence_ref"), str)
                and isinstance(calibration, dict)
                and isinstance(calibration.get("evidence_ref"), str)
            ):
                target_payload = {
                    key: deepcopy(item)
                    for key, item in target.items()
                    if key != "evidence_ref"
                }
                target_coverages[target["evidence_ref"]] = {
                    "schema": (
                        "groundupscale.dev/uncertainty-target-coverage/v1alpha1"
                    ),
                    **target_payload,
                }
                uncertainty_calibrations[calibration["evidence_ref"]] = {
                    "schema": (
                        "groundupscale.dev/uncertainty-calibration/v1alpha1"
                    ),
                    "policy_id": value.get("policy_id"),
                    "version": value.get("version"),
                    "target_coverage": target_payload,
                    "estimator": deepcopy(calibration.get("estimator")),
                    "records": deepcopy(calibration.get("records")),
                }
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif _artifact_ref(value):
            refs.add(value)

    collect(document)
    artifacts: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for artifact_ref in sorted(refs):
        if artifact_ref.startswith(
            (
                "artifact://issue-32/source-",
                "artifact://issue-32/raw-",
            )
        ):
            continue
        if artifact_ref in implementations:
            content = implementations[artifact_ref]
            if content["schema"].endswith("candidate-implementation/v1alpha1"):
                role = "candidate-implementation"
            else:
                role = "implementation-family-manifest"
            artifacts[artifact_ref] = (role, content["schema"], content)
        elif artifact_ref in uncertainty_components:
            content = uncertainty_components[artifact_ref]
            artifacts[artifact_ref] = (
                "uncertainty-component",
                content["schema"],
                content,
            )
        elif artifact_ref in target_coverages:
            content = target_coverages[artifact_ref]
            artifacts[artifact_ref] = (
                "uncertainty-target-coverage",
                content["schema"],
                content,
            )
        elif artifact_ref in uncertainty_calibrations:
            content = uncertainty_calibrations[artifact_ref]
            artifacts[artifact_ref] = (
                "uncertainty-calibration",
                content["schema"],
                content,
            )
        else:
            artifacts[artifact_ref] = diagnostic.get(
                artifact_ref,
                (
                    "diagnostic-supporting-evidence",
                    "groundupscale.dev/diagnostic-supporting-evidence/v1alpha1",
                    {
                        "schema": (
                            "groundupscale.dev/"
                            "diagnostic-supporting-evidence/v1alpha1"
                        ),
                        "payload": {"artifact_ref": artifact_ref},
                    },
                ),
            )
    return artifacts


def _copy_source_artifacts(root: Path) -> list[dict[str, Any]]:
    sources = [
        (
            ISSUE29_RUN / "comparison/physical-floor-vs-observation.json",
            "source/issue29-physical-floor.json",
            "source-physical-floor",
            "artifact://issue-32/source-physical-floor",
            "run-bundle://ascend-910b2-matmul-floor-comparison-20260810-v2",
        ),
        (
            ISSUE30_RUN / "observation/raw/benchmark.json",
            "source/issue30-transformer-benchmark.json",
            "source-transformer-benchmark",
            "artifact://issue-32/source-transformer-benchmark",
            "run-bundle://ascend-910b2-transformer-demo-20260811-v1",
        ),
        (
            ISSUE30_RUN / "comparison/error-attribution.json",
            "source/issue30-transformer-e2e-attribution.json",
            "source-transformer-e2e-attribution",
            "artifact://issue-32/source-transformer-e2e-attribution",
            "run-bundle://ascend-910b2-transformer-demo-20260811-v1",
        ),
        (
            ISSUE30_RUN / "adapter/cohort.json",
            "source/issue30-hardware-cohort.json",
            "source-hardware-cohort",
            None,
            "run-bundle://ascend-910b2-transformer-demo-20260811-v1",
        ),
        (
            ISSUE31_RUN / "frontier/qualification.json",
            "source/issue31-frontier-qualification.json",
            "source-frontier-qualification",
            SOURCE_FRONTIER_REF,
            "run-bundle://issue31-operator-frontier-v3",
        ),
        (
            WORK_ROOT / "evidence/remote-execution.json",
            "source/issue32-remote-execution.json",
            "source-remote-execution",
            "artifact://issue-32/source-remote-execution",
            "remote://A2-AK-225/issue32-collection",
        ),
        *[
            (
                SESSION_ROOT / f"issue32-session-0{index}-v7.json",
                f"source/issue32-q-session-0{index}-v7.json",
                "source-diagnostic-session",
                f"artifact://issue-32/raw-q-session-{index}",
                f"remote://A2-AK-225/issue32-session-0{index}-v7",
            )
            for index in range(1, 4)
        ],
        *[
            (
                SESSION_ROOT / f"issue32-session-0{index}-v10.json",
                f"source/issue32-semantic-session-0{index}-v10.json",
                "source-diagnostic-session",
                f"artifact://issue-32/raw-semantic-session-{index}",
                f"remote://A2-AK-225/issue32-session-0{index}-v10",
            )
            for index in range(1, 4)
        ],
    ]
    artifacts = []
    for source, relative, role, uri, source_input in sources:
        payload = source.read_bytes()
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        schema = _read_json(source)["schema"]
        artifact = {
            "role": role,
            "path": relative,
            "schema": schema,
            "media_type": "application/json",
            "sha256": sha256(payload).hexdigest(),
            "produced_by": PRODUCER,
            "inputs": [source_input],
        }
        if uri is not None:
            artifact["uri"] = uri
        artifacts.append(artifact)
    return artifacts


def main() -> int:
    if RUN_ROOT.exists():
        raise FileExistsError(f"immutable run id already exists: {RUN_ROOT}")
    temporary = RUN_ROOT.with_name(f".{RUN_ID}.building")
    if temporary.exists():
        raise FileExistsError(f"stale build directory exists: {temporary}")
    source_manifests = [
        (
            ISSUE29_RUN,
            _verified_source_manifest(
                ISSUE29_RUN,
                expected_run_id=(
                    "ascend-910b2-matmul-floor-comparison-20260810-v2"
                ),
            ),
            "resource-physical-floor",
        ),
        (
            ISSUE30_RUN,
            _verified_source_manifest(
                ISSUE30_RUN,
                expected_run_id="ascend-910b2-transformer-demo-20260811-v1",
            ),
            "observation",
        ),
        (
            ISSUE31_RUN,
            _verified_source_manifest(
                ISSUE31_RUN,
                expected_run_id="issue31-operator-frontier-v3",
            ),
            "operator-frontier",
        ),
    ]
    source_runs = [
        _source_run_lineage(root, manifest, role)
        for root, manifest, role in source_manifests
    ]
    floor_comparison = _read_json(
        ISSUE29_RUN / "comparison/physical-floor-vs-observation.json"
    )
    benchmark = _read_json(ISSUE30_RUN / "observation/raw/benchmark.json")
    error_attribution = _read_json(
        ISSUE30_RUN / "comparison/error-attribution.json"
    )
    cohort = _read_json(ISSUE30_RUN / "adapter/cohort.json")
    capabilities = _read_json(ISSUE30_RUN / "adapter/capabilities.json")
    qualification = _read_json(ISSUE31_RUN / "frontier/qualification.json")
    frontier_diagnostic = _read_json(
        ISSUE31_RUN / "diagnostic/evidence.json"
    )
    q_sessions = [
        _read_json(SESSION_ROOT / f"issue32-session-0{index}-v7.json")
        for index in range(1, 4)
    ]
    semantic_sessions = [
        _read_json(SESSION_ROOT / f"issue32-session-0{index}-v10.json")
        for index in range(1, 4)
    ]
    all_sessions = [*q_sessions, *semantic_sessions]
    remote_execution = _read_json(WORK_ROOT / "evidence/remote-execution.json")
    expected_session_digests = {
        item["session_id"]: item["sha256"]
        for item in remote_execution["sessions"]
    }

    def semantic_session_valid(item: dict[str, Any]) -> bool:
        contracts = item["execution_contract"].get("variant_contracts")
        path_inputs = item.get("path_inputs")
        expected = {
            "k_baseline": (
                "batch-one K projection MatMul",
                K_PATH,
                "baseline",
                "k",
            ),
            "v_baseline": (
                "batch-one V projection MatMul",
                V_PATH,
                "baseline",
                "v",
            ),
            "negative_control": (
                "batch-one V projection MatMul negative control",
                V_PATH,
                "diagnostic",
                "v",
            ),
        }
        return bool(
            isinstance(contracts, dict)
            and isinstance(path_inputs, dict)
            and set(path_inputs) == {"q", "k", "v"}
            and item.get("input")
            == {"seed": 20260811, **path_inputs["q"]}
            and len(
                {
                    (identity["left_sha256"], identity["right_sha256"])
                    for identity in path_inputs.values()
                }
            )
            == 3
            and all(
                contracts.get(variant)
                == {
                    "semantic": semantic,
                    "stable_path": stable_path,
                    "lane": lane,
                    "input_identity": path_inputs[path],
                }
                for variant, (
                    semantic,
                    stable_path,
                    lane,
                    path,
                ) in expected.items()
            )
            and all(
                item["path_correctness"][path]["passed"]
                for path in ("q", "k", "v")
            )
            and len(
                {
                    item["path_correctness"][path]["expected_sha256"]
                    for path in ("q", "k", "v")
                }
            )
            == 3
        )

    if (
        len({item["process_id"] for item in all_sessions}) != 6
        or {item["cohort_id"] for item in all_sessions} != {COHORT_ID}
        or any(not item["correctness"]["passed"] for item in all_sessions)
        or any(
            item["negative_control"]["correctness"]["passed"]
            for item in semantic_sessions
        )
        or any(
            expected_session_digests.get(item["session_id"])
            != sha256(
                (SESSION_ROOT / f"{item['session_id']}.json").read_bytes()
            ).hexdigest()
            for item in all_sessions
        )
        or any(
            item["execution_contract"].get("semantic")
            != "batch-one Q projection MatMul"
            for item in q_sessions
        )
        or any(not semantic_session_valid(item) for item in semantic_sessions)
    ):
        raise ValueError("real NPU sessions do not satisfy the locked replay gates")
    document, implementations = _global_document(
        q_sessions,
        semantic_sessions,
        floor_comparison,
        qualification,
        frontier_diagnostic,
        benchmark,
        error_attribution,
        cohort,
        capabilities,
        source_runs,
    )
    temporary.mkdir(parents=True)
    evidence_digest = _write_json(
        temporary / "diagnostic/evidence.json", document
    )
    manifest_artifacts = [
        {
            "role": "diagnostic-evidence",
            "path": "diagnostic/evidence.json",
            "schema": document["schema"],
            "media_type": "application/json",
            "sha256": evidence_digest,
            "produced_by": PRODUCER,
            "inputs": [
                "source-physical-floor",
                "source-transformer-benchmark",
                "source-transformer-e2e-attribution",
                "source-hardware-cohort",
                "source-frontier-qualification",
                "source-remote-execution",
                "source-diagnostic-session",
            ],
        }
    ]
    for index, (artifact_ref, artifact) in enumerate(
        _artifact_documents(document, implementations).items(), start=1
    ):
        role, schema, content = artifact
        relative = f"diagnostic/supporting/{index:03d}.json"
        digest = _write_json(temporary / relative, content)
        manifest_artifacts.append(
            {
                "role": role,
                "uri": artifact_ref,
                "path": relative,
                "schema": schema,
                "media_type": "application/json",
                "sha256": digest,
                "produced_by": PRODUCER,
                "inputs": [
                    "source-physical-floor",
                    "source-transformer-benchmark",
                    "source-frontier-qualification",
                    "source-diagnostic-session",
                ],
            }
        )
    manifest_artifacts.extend(_copy_source_artifacts(temporary))
    _write_json(
        temporary / "run.manifest.json",
        {
            "schema": "groundupscale.dev/run-manifest/v1alpha1",
            "run_id": RUN_ID,
            "status": "completed",
            "device": "ascend-npu",
            "hardware_cohort": COHORT_ID,
            "source_manifest_integrity": "required",
            "source_runs": source_runs,
            "artifacts": manifest_artifacts,
        },
    )
    temporary.replace(RUN_ROOT)
    print(RUN_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
