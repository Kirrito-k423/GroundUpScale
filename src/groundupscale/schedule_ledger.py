"""Exclusive E2E ledger and counterfactual provenance composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from groundupscale.ir.common import DerivationRecord, derivation_identity
from groundupscale.schedule_evidence import (
    ScheduleFrontierError,
    finite_nonnegative,
    valid_evidence_refs,
)


@dataclass(frozen=True)
class LedgerLeaf:
    leaf_id: str
    kind: str
    duration_ns: float
    evidence_refs: tuple[str, ...]
    candidate_id: str | None = None

    @classmethod
    def from_document(cls, value: object) -> LedgerLeaf:
        leaf_id = value.get("leaf_id") if isinstance(value, dict) else None
        kind = value.get("kind") if isinstance(value, dict) else None
        candidate_id = value.get("candidate_id") if isinstance(value, dict) else None
        if (
            not isinstance(leaf_id, str)
            or not leaf_id
            or kind not in {"operation", "dispatch", "schedule-wait"}
            or not finite_nonnegative(value.get("duration_ns"))
            or not valid_evidence_refs(value.get("evidence_refs"))
            or (
                kind == "operation"
                and (not isinstance(candidate_id, str) or not candidate_id)
            )
            or (kind != "operation" and candidate_id is not None)
        ):
            raise ScheduleFrontierError("invalid-exclusive-leaf")
        return cls(
            leaf_id=leaf_id,
            kind=kind,
            duration_ns=float(value["duration_ns"]),
            evidence_refs=tuple(value["evidence_refs"]),
            candidate_id=candidate_id,
        )

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "leaf_id": self.leaf_id,
            "kind": self.kind,
            "duration_ns": self.duration_ns,
            "evidence_refs": list(self.evidence_refs),
        }
        if self.candidate_id is not None:
            document["candidate_id"] = self.candidate_id
        return document


@dataclass(frozen=True)
class LedgerParent:
    span_id: str
    kind: str
    child_parent_ids: tuple[str, ...]
    leaf_ids: tuple[str, ...]

    @classmethod
    def from_document(cls, value: object) -> LedgerParent:
        children = value.get("child_parent_ids") if isinstance(value, dict) else None
        leaves = value.get("leaf_ids") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("span_id"), str)
            or not value["span_id"]
            or value.get("kind") not in {"e2e", "module"}
            or value.get("additive") is not False
            or not isinstance(children, list)
            or not all(isinstance(item, str) and item for item in children)
            or not isinstance(leaves, list)
            or not all(isinstance(item, str) and item for item in leaves)
        ):
            raise ScheduleFrontierError("invalid-ledger-parent-index")
        return cls(
            span_id=value["span_id"],
            kind=value["kind"],
            child_parent_ids=tuple(children),
            leaf_ids=tuple(leaves),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "kind": self.kind,
            "additive": False,
            "child_parent_ids": list(self.child_parent_ids),
            "leaf_ids": list(self.leaf_ids),
        }


@dataclass(frozen=True)
class LedgerResidual:
    residual_id: str
    duration_ns: float
    evidence_refs: tuple[str, ...]

    @classmethod
    def from_document(cls, value: object) -> LedgerResidual:
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("residual_id"), str)
            or not value["residual_id"]
            or value.get("kind") != "unattributed"
            or not finite_nonnegative(value.get("duration_ns"))
            or not valid_evidence_refs(value.get("evidence_refs"))
        ):
            raise ScheduleFrontierError("invalid-unattributed-residual")
        return cls(
            residual_id=value["residual_id"],
            duration_ns=float(value["duration_ns"]),
            evidence_refs=tuple(value["evidence_refs"]),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "residual_id": self.residual_id,
            "kind": "unattributed",
            "duration_ns": self.duration_ns,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class BatchedDispatchTransformation:
    transformation_id: str
    version: str
    removed_leaf_ids: tuple[str, ...]
    declared_recovered_ns: float
    evidence_refs: tuple[str, ...]

    @classmethod
    def from_document(cls, value: object) -> BatchedDispatchTransformation:
        removed = value.get("removed_leaf_ids") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("transformation_id"), str)
            or not value["transformation_id"]
            or value.get("kind") != "batched-dispatch"
            or not isinstance(value.get("version"), str)
            or not value["version"]
            or not isinstance(removed, list)
            or not removed
            or len(set(removed)) != len(removed)
            or not all(isinstance(item, str) and item for item in removed)
            or not finite_nonnegative(value.get("declared_recovered_ns"))
            or not valid_evidence_refs(value.get("evidence_refs"))
        ):
            raise ScheduleFrontierError(
                "invalid-batched-dispatch-transformation"
            )
        return cls(
            transformation_id=value["transformation_id"],
            version=value["version"],
            removed_leaf_ids=tuple(removed),
            declared_recovered_ns=float(value["declared_recovered_ns"]),
            evidence_refs=tuple(value["evidence_refs"]),
        )


def _validate_parent_graph(parents: tuple[LedgerParent, ...]) -> None:
    parent_by_id: dict[str, LedgerParent] = {}
    for parent in parents:
        if parent.span_id in parent_by_id:
            raise ScheduleFrontierError("invalid-ledger-parent-index")
        parent_by_id[parent.span_id] = parent
    if any(
        child_id not in parent_by_id
        for parent in parents
        for child_id in parent.child_parent_ids
    ):
        raise ScheduleFrontierError("unknown-ledger-child-parent")
    roots = [parent for parent in parents if parent.kind == "e2e"]
    if len(roots) != 1:
        raise ScheduleFrontierError("ledger-requires-one-e2e-root")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(parent_id: str) -> None:
        if parent_id in visiting:
            raise ScheduleFrontierError("ledger-parent-graph-cycle")
        if parent_id in visited:
            return
        visiting.add(parent_id)
        for child_id in parent_by_id[parent_id].child_parent_ids:
            visit(child_id)
        visiting.remove(parent_id)
        visited.add(parent_id)

    visit(roots[0].span_id)
    if visited != set(parent_by_id):
        raise ScheduleFrontierError("ledger-parent-not-reachable-from-e2e")


def compose_ledger(
    document: Mapping[str, object],
    axes: Mapping[str, Mapping[str, Any]],
    selected_candidate_ids: set[str],
) -> dict[str, Any]:
    value = document.get("ledger")
    leaves_value = value.get("leaves") if isinstance(value, dict) else None
    parents_value = value.get("parents") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("leaf_semantics") != "mutually-exclusive"
        or not finite_nonnegative(value.get("e2e_duration_ns"))
        or not isinstance(leaves_value, list)
        or not leaves_value
        or not isinstance(parents_value, list)
        or not parents_value
    ):
        raise ScheduleFrontierError("invalid-exclusive-ledger")
    if value["e2e_duration_ns"] != axes["observation"]["fixture_duration_ns"]:
        raise ScheduleFrontierError("ledger-observation-does-not-reconcile")
    leaves = tuple(LedgerLeaf.from_document(item) for item in leaves_value)
    leaf_ids = [leaf.leaf_id for leaf in leaves]
    if len(set(leaf_ids)) != len(leaf_ids):
        raise ScheduleFrontierError("invalid-exclusive-leaf")
    operation_candidate_ids: set[str] = set()
    for leaf in leaves:
        if leaf.kind != "operation":
            continue
        if leaf.candidate_id in operation_candidate_ids:
            raise ScheduleFrontierError("duplicate-operation-leaf-candidate")
        if leaf.candidate_id not in selected_candidate_ids:
            raise ScheduleFrontierError(
                "ledger-operation-candidate-not-selected"
            )
        operation_candidate_ids.add(leaf.candidate_id or "")
    parents = tuple(LedgerParent.from_document(item) for item in parents_value)
    _validate_parent_graph(parents)
    assigned_leaf_ids = [
        leaf_id for parent in parents for leaf_id in parent.leaf_ids
    ]
    duplicates = sorted(
        {
            leaf_id
            for leaf_id in assigned_leaf_ids
            if assigned_leaf_ids.count(leaf_id) > 1
        }
    )
    unassigned = sorted(set(leaf_ids) - set(assigned_leaf_ids))
    unknown_assignments = sorted(set(assigned_leaf_ids) - set(leaf_ids))
    if duplicates or unassigned or unknown_assignments:
        raise ScheduleFrontierError("ledger-leaf-identity-not-conserved")
    residual = LedgerResidual.from_document(value.get("residual"))
    leaf_total_ns = sum(leaf.duration_ns for leaf in leaves)
    reconciled_total_ns = leaf_total_ns + residual.duration_ns
    if reconciled_total_ns != value["e2e_duration_ns"]:
        raise ScheduleFrontierError("exclusive-ledger-does-not-reconcile")
    return {
        "status": "conserved",
        "leaf_semantics": "mutually-exclusive",
        "e2e_duration_ns": value["e2e_duration_ns"],
        "parents": [parent.to_document() for parent in parents],
        "leaves": [leaf.to_document() for leaf in leaves],
        "operation_leaf_total_ns": sum(
            leaf.duration_ns for leaf in leaves if leaf.kind == "operation"
        ),
        "leaf_total_ns": leaf_total_ns,
        "residual": residual.to_document(),
        "reconciled_total_ns": reconciled_total_ns,
        "leaf_identity_conservation": {
            "unique_leaf_count": len(leaf_ids),
            "duplicate_leaf_ids": duplicates,
            "unassigned_leaf_ids": unassigned,
        },
        "parent_span_total_included_ns": 0,
    }


def compose_counterfactuals(
    document: Mapping[str, object],
    axes: Mapping[str, Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[DerivationRecord]]:
    values = document.get("transformations")
    if not isinstance(values, list):
        raise ScheduleFrontierError("invalid-schedule-transformations")
    leaf_by_id = {leaf["leaf_id"]: leaf for leaf in ledger["leaves"]}
    transformation_ids: set[str] = set()
    results: list[dict[str, Any]] = []
    records: list[DerivationRecord] = []
    for value in values:
        transformation = BatchedDispatchTransformation.from_document(value)
        if transformation.transformation_id in transformation_ids:
            raise ScheduleFrontierError("invalid-schedule-transformation")
        if any(leaf_id not in leaf_by_id for leaf_id in transformation.removed_leaf_ids):
            raise ScheduleFrontierError(
                "invalid-batched-dispatch-transformation"
            )
        removed = [leaf_by_id[leaf_id] for leaf_id in transformation.removed_leaf_ids]
        if any(leaf["kind"] != "dispatch" for leaf in removed):
            raise ScheduleFrontierError(
                "batched-dispatch-may-only-remove-dispatch-leaves"
            )
        recovered_ns = sum(leaf["duration_ns"] for leaf in removed)
        if recovered_ns != transformation.declared_recovered_ns:
            raise ScheduleFrontierError(
                "batched-dispatch-recovered-time-does-not-reconcile"
            )
        preserved = [
            leaf
            for leaf in ledger["leaves"]
            if leaf["leaf_id"] not in transformation.removed_leaf_ids
        ]
        counterfactual_e2e_ns = (
            sum(leaf["duration_ns"] for leaf in preserved)
            + ledger["residual"]["duration_ns"]
        )
        if ledger["e2e_duration_ns"] - counterfactual_e2e_ns != recovered_ns:
            raise ScheduleFrontierError(
                "batched-dispatch-ledger-difference-does-not-reconcile"
            )
        operation_total_ns = sum(
            leaf["duration_ns"]
            for leaf in ledger["leaves"]
            if leaf["kind"] == "operation"
        )
        operator_frontier_ns = axes["operator_achievable_frontier"][
            "fixture_duration_ns"
        ]
        transformation_ids.add(transformation.transformation_id)
        rule = f"batched-dispatch@{transformation.version}"
        stable_path = transformation.transformation_id
        record_id = derivation_identity(
            rule,
            str(ledger["e2e_duration_ns"]),
            stable_path,
        )
        record = DerivationRecord(
            derivation_id=record_id,
            phase="schedule-counterfactual",
            rule=rule,
            source_path="ScheduleLedger",
            source_stable_path=stable_path,
            target_node_ids=(
                f"counterfactual:{transformation.transformation_id}",
            ),
            assumptions=(
                "mutually-exclusive-ledger-leaves",
                "removed_leaf_ids="
                + ",".join(transformation.removed_leaf_ids),
                f"recovered_ns={recovered_ns:g}",
                "evidence_refs=" + ",".join(transformation.evidence_refs),
            ),
            warnings=("prototype-only-untrusted-fixture",),
        )
        records.append(record)
        results.append(
            {
                "transformation_id": transformation.transformation_id,
                "kind": "batched-dispatch",
                "version": transformation.version,
                "status": "conserved",
                "derivation_record_id": record_id,
                "evidence_refs": list(transformation.evidence_refs),
                "baseline_e2e_ns": ledger["e2e_duration_ns"],
                "counterfactual_e2e_ns": counterfactual_e2e_ns,
                "recovered_ns": recovered_ns,
                "removed_leaf_ids": list(transformation.removed_leaf_ids),
                "leaf_identity_conservation": {
                    "before_count": len(ledger["leaves"]),
                    "preserved_count": len(preserved),
                    "removed_leaf_ids": list(transformation.removed_leaf_ids),
                    "added_leaf_ids": [],
                },
                "operation_leaf_total_ns": {
                    "before": operation_total_ns,
                    "after": operation_total_ns,
                },
                "operator_achievable_frontier_ns": {
                    "before": operator_frontier_ns,
                    "after": operator_frontier_ns,
                },
                "axis_mutations": [],
            }
        )
    return results, records


__all__ = ["compose_counterfactuals", "compose_ledger"]
