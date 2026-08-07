"""Pure throwaway ledger model for issue 7; do not promote to production."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


EPSILON_MS = 1e-9


@dataclass(frozen=True)
class PathResult:
    duration_ms: float
    node_ids: tuple[str, ...]


def load_input() -> tuple[dict[str, Any], str]:
    path = Path(__file__).with_name("input.json")
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def _node_id(layer: int, key: str) -> str:
    return f"layer_{layer}/{key}"


def expand_nodes(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    previous_layer_final: str | None = None
    for layer_index, layer in enumerate(fixture["layers"]):
        by_key: dict[str, str] = {}
        for template in fixture["operator_templates"]:
            node_id = _node_id(layer, template["key"])
            semantic_after: list[str] = []
            for predecessor in template["semantic_after"]:
                if predecessor == "$previous_layer":
                    if previous_layer_final is not None:
                        semantic_after.append(previous_layer_final)
                else:
                    semantic_after.append(by_key[predecessor])
            execution_after = [by_key[item] for item in template["execution_after"]]
            physical_floor = _apply_local_rule(
                template["physical_components_ms"], "max"
            )
            operator_frontier = _apply_local_rule(
                template["candidate_components_ms"],
                template["candidate_overlap_rule"],
            )
            observed_ms = template["observed_ms"][layer_index]
            source_kind = "synthetic_for_prototype"
            if layer == 0 and template["key"] == "q_projection":
                source_kind = "published_functional_m4_trace"
            if layer == 0 and template["key"] == "softmax":
                source_kind = "published_functional_m4_trace"
            nodes.append(
                {
                    "id": node_id,
                    "stable_path": f"semantic/model/layer_{layer}/{template['module']}/{template['key']}",
                    "layer": layer,
                    "module": template["module"],
                    "stage": template["stage"],
                    "semantic_after": semantic_after,
                    "execution_after": execution_after,
                    "resource_claim": template["resource_claim"],
                    "physical_components_ms": deepcopy(template["physical_components_ms"]),
                    "physical_floor_ms": physical_floor,
                    "candidate_components_ms": deepcopy(template["candidate_components_ms"]),
                    "candidate_overlap_rule": template["candidate_overlap_rule"],
                    "operator_frontier_ms": operator_frontier,
                    "frontier_uncertainty_ms": template["frontier_uncertainty_ms"],
                    "observed_ms": observed_ms,
                    "observation_source_kind": source_kind,
                }
            )
            by_key[template["key"]] = node_id
        previous_layer_final = by_key["mlp_down_residual"]
    return nodes


def _apply_local_rule(components: dict[str, float], rule: str) -> float:
    if not components:
        raise ValueError("candidate has no resource components")
    if rule == "max":
        return max(components.values())
    if rule == "sum":
        return sum(components.values())
    raise ValueError(f"unsupported candidate-local rule: {rule}")


def longest_path(nodes: list[dict[str, Any]], duration_key: str) -> PathResult:
    by_id = {node["id"]: node for node in nodes}
    remaining = set(by_id)
    distances: dict[str, float] = {}
    paths: dict[str, tuple[str, ...]] = {}
    while remaining:
        progressed = False
        for node_id in sorted(remaining):
            node = by_id[node_id]
            predecessors = list(
                dict.fromkeys(node["semantic_after"] + node["execution_after"])
            )
            if not all(predecessor in distances for predecessor in predecessors):
                continue
            if predecessors:
                winner = max(predecessors, key=lambda item: distances[item])
                prior_duration = distances[winner]
                prior_path = paths[winner]
            else:
                prior_duration = 0.0
                prior_path = ()
            distances[node_id] = prior_duration + node[duration_key]
            paths[node_id] = prior_path + (node_id,)
            remaining.remove(node_id)
            progressed = True
        if not progressed:
            raise ValueError(f"dependency cycle or missing predecessor: {sorted(remaining)}")
    terminal = max(distances, key=distances.get)
    return PathResult(distances[terminal], paths[terminal])


def _validate_explicit_semantics(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    reasons = {
        "fusion": "fusion_requires_explicit_group",
        "concurrency": "concurrency_requires_explicit_execution_semantics",
        "communication_masking": "communication_masking_requires_explicit_event_pair",
        "resource_contention": "contention_requires_explicit_resource_claim",
    }
    results = []
    for counterexample in fixture["counterexamples"]:
        accepted = bool(counterexample["explicit"])
        reason = None if accepted else reasons[counterexample["kind"]]
        results.append(
            {
                **counterexample,
                "accepted": accepted,
                "actual_reason": reason,
                "rejected_as_expected": (
                    not accepted and reason == counterexample["expected_reason"]
                ),
            }
        )
    return results


def _schedule_candidates(
    fixture: dict[str, Any], operator_path: PathResult
) -> list[dict[str, Any]]:
    candidates = []
    for candidate in fixture["schedule_candidates"]:
        transformations_valid = all(
            transformation.get("explicit", False)
            for transformation in candidate["explicit_transformations"]
        )
        operator_frontier_preserved = all(
            transformation.get("preserves_operator_frontier", False)
            for transformation in candidate["explicit_transformations"]
        ) if candidate["explicit_transformations"] else True
        overhead_ms = sum(
            item["duration_ms"] for item in candidate["required_overheads_ms"]
        )
        candidates.append(
            {
                "id": candidate["id"],
                "valid": bool(candidate["validated_for_prototype"])
                and transformations_valid,
                "operator_frontier_preserved": operator_frontier_preserved,
                "critical_path_operator_ms": operator_path.duration_ms,
                "required_overhead_ms": overhead_ms,
                "schedule_duration_ms": operator_path.duration_ms + overhead_ms,
                "critical_path": list(operator_path.node_ids),
                "explicit_transformations": deepcopy(
                    candidate["explicit_transformations"]
                ),
            }
        )
    return candidates


def _observed_ledger(
    fixture: dict[str, Any], nodes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = [
        {
            "id": f"trace/{node['id']}",
            "duration_ms": node["observed_ms"],
            "kind": "operator_execution",
            "scope": f"layer_{node['layer']}/{node['module']}",
            "operator_id": node["id"],
            "source_kind": node["observation_source_kind"],
        }
        for node in nodes
    ]
    rows.extend(
        {
            **deepcopy(item),
            "id": f"trace/{item['id']}",
            "source_kind": "synthetic_for_prototype",
        }
        for item in fixture["observed_overheads_ms"]
    )
    return rows


def _aggregate_ledger(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: defaultdict[str, float] = defaultdict(float)
    by_scope: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        by_kind[row["kind"]] += row["duration_ms"]
        by_scope[row["scope"]] += row["duration_ms"]
    return {
        "by_kind_ms": dict(sorted(by_kind.items())),
        "by_scope_ms": dict(sorted(by_scope.items())),
        "total_ms": sum(item["duration_ms"] for item in rows),
    }


def _drilldown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_modules: defaultdict[str, defaultdict[str, list[dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    e2e_direct: list[dict[str, Any]] = []
    for row in rows:
        scope_parts = row["scope"].split("/", 1)
        if scope_parts[0] == "e2e":
            e2e_direct.append(row)
            continue
        stage, module = scope_parts
        stage_modules[stage][module].append(row)

    stages = []
    for stage_id in sorted(stage_modules):
        modules = []
        for module_id in sorted(stage_modules[stage_id]):
            leaves = stage_modules[stage_id][module_id]
            modules.append(
                {
                    "id": module_id,
                    "duration_ms": sum(item["duration_ms"] for item in leaves),
                    "leaves": deepcopy(leaves),
                }
            )
        stages.append(
            {
                "id": stage_id,
                "duration_ms": sum(item["duration_ms"] for item in modules),
                "modules": modules,
            }
        )
    all_leaf_ids = [
        leaf["id"]
        for stage in stages
        for module in stage["modules"]
        for leaf in module["leaves"]
    ] + [item["id"] for item in e2e_direct]
    return {
        "id": "e2e",
        "duration_ms": sum(item["duration_ms"] for item in rows),
        "stages": stages,
        "direct_leaves": deepcopy(e2e_direct),
        "leaf_ids": all_leaf_ids,
    }


def _counterfactual(
    fixture: dict[str, Any], rows: list[dict[str, Any]], operator_total_ms: float
) -> dict[str, Any]:
    candidate = next(
        item
        for item in fixture["schedule_candidates"]
        if item["id"] == "batched_dispatch_counterfactual"
    )
    removed_ids = {
        f"trace/{bucket_id}"
        for transformation in candidate["explicit_transformations"]
        for bucket_id in transformation["removes_observed_buckets"]
    }
    kept = [row for row in rows if row["id"] not in removed_ids]
    removed = [row for row in rows if row["id"] in removed_ids]
    before_total = sum(row["duration_ms"] for row in rows)
    after_total = sum(row["duration_ms"] for row in kept)
    removed_total = sum(row["duration_ms"] for row in removed)
    after_operator_total = sum(
        row["duration_ms"] for row in kept if row["kind"] == "operator_execution"
    )
    return {
        "id": candidate["id"],
        "explicit_transformations": deepcopy(candidate["explicit_transformations"]),
        "removed_rows": removed,
        "before_e2e_ms": before_total,
        "after_e2e_ms": after_total,
        "observed_delta_ms": before_total - after_total,
        "removed_bucket_total_ms": removed_total,
        "operator_observed_before_ms": operator_total_ms,
        "operator_observed_after_ms": after_operator_total,
        "operator_frontier_changed": False,
    }


def _top_ten(nodes: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [
        {"rank": rank, "operator_id": node["id"], "duration_ms": node[key]}
        for rank, node in enumerate(
            sorted(nodes, key=lambda item: item[key], reverse=True)[:10], start=1
        )
    ]


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def evaluate() -> dict[str, Any]:
    fixture, input_digest = load_input()
    nodes = expand_nodes(fixture)
    physical_path = longest_path(nodes, "physical_floor_ms")
    operator_path = longest_path(nodes, "operator_frontier_ms")
    schedules = _schedule_candidates(fixture, operator_path)
    valid_schedules = [item for item in schedules if item["valid"]]
    schedule_frontier = min(valid_schedules, key=lambda item: item["schedule_duration_ms"])
    rows = _observed_ledger(fixture, nodes)
    ledger = _aggregate_ledger(rows)
    drilldown = _drilldown(rows)
    operator_total_ms = ledger["by_kind_ms"]["operator_execution"]
    counterfactual = _counterfactual(fixture, rows, operator_total_ms)
    semantic_counterexamples = _validate_explicit_semantics(fixture)

    expected_e2e_ms = fixture["reference_observations_ms"]["two_layer_prefill"]["value"]
    row_ids = [row["id"] for row in rows]
    unique_ids = len(set(row_ids)) == len(row_ids)
    drilldown_leaf_ids = drilldown["leaf_ids"]
    drilldown_conserved = (
        sorted(drilldown_leaf_ids) == sorted(row_ids)
        and len(drilldown_leaf_ids) == len(set(drilldown_leaf_ids))
        and abs(drilldown["duration_ms"] - ledger["total_ms"]) <= EPSILON_MS
    )
    all_operator_observations_in_band = all(
        abs(node["observed_ms"] - node["operator_frontier_ms"])
        <= node["frontier_uncertainty_ms"] + EPSILON_MS
        for node in nodes
    )
    verdict = (
        "integration_overhead"
        if all_operator_observations_in_band
        and expected_e2e_ms
        > schedule_frontier["schedule_duration_ms"] + 2.0
        else "insufficient_evidence"
    )
    declared_physical_floor_ms = 5.553976
    declared_operator_critical_path_ms = 51.632
    explicit_semantics_enforced = all(
        item["rejected_as_expected"] for item in semantic_counterexamples
    )
    counterfactual_valid = (
        not counterfactual["operator_frontier_changed"]
        and abs(
            counterfactual["operator_observed_before_ms"]
            - counterfactual["operator_observed_after_ms"]
        )
        <= EPSILON_MS
        and abs(
            counterfactual["observed_delta_ms"]
            - counterfactual["removed_bucket_total_ms"]
        )
        <= EPSILON_MS
    )
    checks = {
        "ledger_conserved": unique_ids
        and drilldown_conserved
        and abs(ledger["total_ms"] - expected_e2e_ms) <= EPSILON_MS,
        "critical_path_valid": abs(
            physical_path.duration_ms - declared_physical_floor_ms
        )
        <= EPSILON_MS
        and abs(operator_path.duration_ms - declared_operator_critical_path_ms)
        <= EPSILON_MS
        and all(item["operator_frontier_preserved"] for item in valid_schedules),
        "explicit_semantics_enforced": explicit_semantics_enforced,
        "integration_overhead_detected": verdict == "integration_overhead",
        "counterfactual_explained": counterfactual_valid,
    }

    result = {
        "schema": "groundupscale.dev/throwaway-schedule-frontier-result/v0",
        "prototype_only": True,
        "question_answered": all(checks.values()),
        "input": {
            "sha256": input_digest,
            "fixture": fixture,
            "expanded_nodes": nodes,
        },
        "environment": {
            "git_revision": _git_revision(),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "boundaries_ms": {
            "resource_physical_floor": physical_path.duration_ms,
            "operator_achievable_frontier": operator_path.duration_ms,
            "schedule_achievable_frontier": schedule_frontier["schedule_duration_ms"],
            "observed_trace_e2e": expected_e2e_ms,
        },
        "critical_paths": {
            "resource_physical_floor": list(physical_path.node_ids),
            "operator_achievable_frontier": list(operator_path.node_ids),
            "schedule_achievable_frontier": schedule_frontier["critical_path"],
        },
        "schedule_candidates": schedules,
        "observed_time_ledger": {
            "rows": rows,
            **ledger,
            "expected_e2e_ms": expected_e2e_ms,
            "unique_leaf_ids": unique_ids,
            "drilldown_conserved": drilldown_conserved,
            "drilldown": drilldown,
        },
        "top_10": {
            "predicted_operator_frontier": _top_ten(
                nodes, "operator_frontier_ms"
            ),
            "observed_operator_trace": _top_ten(nodes, "observed_ms"),
        },
        "diagnosis": {
            "all_operator_observations_in_frontier_band": all_operator_observations_in_band,
            "schedule_frontier_uncertainty_ms": 2.0,
            "unexplained_above_schedule_frontier_ms": expected_e2e_ms
            - schedule_frontier["schedule_duration_ms"],
            "verdict": verdict,
        },
        "counterfactual": counterfactual,
        "counterexamples": semantic_counterexamples,
        "checks": checks,
    }
    return result
