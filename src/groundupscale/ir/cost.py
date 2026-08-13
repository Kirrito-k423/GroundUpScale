"""Hardware-independent algorithmic cost representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TypeAlias

from groundupscale.ir.semantic import (
    ProvenanceGraph,
    SemanticTensorType,
    ValidationResult,
)


@dataclass(frozen=True)
class CostMetrics:
    flops: int = 0
    logical_read_bytes: int = 0
    logical_write_bytes: int = 0
    materialized_read_bytes: int = 0
    materialized_write_bytes: int = 0
    parameter_read_bytes: int = 0
    buffer_read_bytes: int = 0
    activation_read_bytes: int = 0
    explicit_activation_bytes: int = 0
    alias_result_bytes: int = 0

    def __add__(self, other: CostMetrics) -> CostMetrics:
        return CostMetrics(
            flops=self.flops + other.flops,
            logical_read_bytes=self.logical_read_bytes + other.logical_read_bytes,
            logical_write_bytes=self.logical_write_bytes + other.logical_write_bytes,
            materialized_read_bytes=(
                self.materialized_read_bytes + other.materialized_read_bytes
            ),
            materialized_write_bytes=(
                self.materialized_write_bytes + other.materialized_write_bytes
            ),
            parameter_read_bytes=self.parameter_read_bytes + other.parameter_read_bytes,
            buffer_read_bytes=self.buffer_read_bytes + other.buffer_read_bytes,
            activation_read_bytes=self.activation_read_bytes + other.activation_read_bytes,
            explicit_activation_bytes=(
                self.explicit_activation_bytes + other.explicit_activation_bytes
            ),
            alias_result_bytes=self.alias_result_bytes + other.alias_result_bytes,
        )


@dataclass(frozen=True)
class CostFormula:
    rule_id: str
    flops_expression: str
    logical_read_expression: str
    logical_write_expression: str
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class OperatorPhase:
    phase_id: str
    phase_name: str
    operation_class: str
    compute_capability_resource: str
    memory_capability_resource: str
    predecessor_phase_ids: tuple[str, ...]
    input_roles: tuple[str, ...]
    output_roles: tuple[str, ...]
    minimum_flops: int
    logical_read_bytes: int
    logical_write_bytes: int
    assumptions: tuple[str, ...]
    derivation_ids: tuple[str, ...]


@dataclass(frozen=True)
class OperatorPhaseGraph:
    graph_id: str
    phases: tuple[OperatorPhase, ...]
    output_phase_ids: tuple[str, ...]


@dataclass(frozen=True)
class CostOperation:
    local_id: str
    operation: str
    semantic_node_id: str
    definition_id: str
    stable_path: str
    node_id: str
    operands: tuple[str, ...]
    results: tuple[str, ...]
    operand_types: tuple[SemanticTensorType, ...]
    result_types: tuple[SemanticTensorType, ...]
    state_artifact_ids: tuple[str, ...]
    dependency_cost_node_ids: tuple[str, ...]
    metrics: CostMetrics
    formula: CostFormula
    phase_graph: OperatorPhaseGraph | None
    derivation_ids: tuple[str, ...]


@dataclass(frozen=True)
class CostRegion:
    local_id: str
    kind: str
    semantic_node_id: str
    definition_id: str
    stable_path: str
    node_id: str
    items: tuple[CostItem, ...]
    metrics: CostMetrics
    derivation_ids: tuple[str, ...]

    def walk_items(self) -> Iterator[CostItem]:
        for item in self.items:
            yield item
            if isinstance(item, CostRegion):
                yield from item.walk_items()


CostItem: TypeAlias = CostOperation | CostRegion


@dataclass(frozen=True)
class CostSummary:
    metrics: CostMetrics
    parameter_bytes: int
    buffer_bytes: int
    workload_artifact_bytes: int
    serial_flops: int
    ideal_parallel_critical_path_flops: int


@dataclass(frozen=True)
class CostProgram:
    schema: str
    name: str
    compilation_fingerprint: str
    semantic_compilation_fingerprint: str
    root: CostRegion
    summary: CostSummary

    def walk_operations(self) -> Iterator[CostOperation]:
        for item in self.root.walk_items():
            if isinstance(item, CostOperation):
                yield item


@dataclass(frozen=True)
class CostLoweringResult:
    cost_ir: CostProgram
    provenance: ProvenanceGraph
    validation_results: tuple[ValidationResult, ...]
    compilation_fingerprint: str
