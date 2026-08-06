"""Hierarchical, hardware-independent semantic program representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TypeAlias

from groundupscale.ir.common import DerivationRecord


@dataclass(frozen=True)
class SemanticTensorType:
    dtype: str
    shape: tuple[int, ...]
    layout: str


@dataclass(frozen=True)
class SemanticValue:
    value_id: str
    node_id: str
    stable_path: str
    kind: str
    tensor: SemanticTensorType
    producer_id: str | None
    consumer_ids: tuple[str, ...]
    alias_of: str | None
    derivation_ids: tuple[str, ...]


@dataclass(frozen=True)
class SemanticStateArtifact:
    artifact_id: str
    node_id: str
    stable_path: str
    role: str
    tensor: SemanticTensorType
    source_kind: str
    initial_version: int | None
    derivation_ids: tuple[str, ...]


@dataclass(frozen=True)
class SemanticStateEffect:
    effect_id: str
    node_id: str
    stable_path: str
    kind: str
    artifact_id: str
    input_value_id: str | None
    output_value_id: str | None
    owner_node_id: str | None
    version_before: int | None
    version_after: int | None
    derivation_ids: tuple[str, ...]


@dataclass(frozen=True)
class SemanticOperation:
    local_id: str
    operation: str
    definition_id: str
    stable_path: str
    node_id: str
    operands: tuple[str, ...]
    results: tuple[str, ...]
    attributes: tuple[tuple[str, str | int | float | bool], ...]
    state_effect_ids: tuple[str, ...]
    derivation_ids: tuple[str, ...]


@dataclass(frozen=True)
class SemanticRegion:
    local_id: str
    kind: str
    definition_id: str
    stable_path: str
    node_id: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    items: tuple[SemanticItem, ...]
    state_effect_ids: tuple[str, ...]
    attributes: tuple[tuple[str, str | int | float | bool], ...]
    derivation_ids: tuple[str, ...]

    def walk_items(self) -> Iterator[SemanticItem]:
        for item in self.items:
            yield item
            if isinstance(item, SemanticRegion):
                yield from item.walk_items()


SemanticItem: TypeAlias = SemanticOperation | SemanticRegion


@dataclass(frozen=True)
class SemanticProgram:
    schema: str
    name: str
    compilation_fingerprint: str
    symbols: tuple[tuple[str, int], ...]
    root: SemanticRegion
    values: tuple[SemanticValue, ...]
    state_artifacts: tuple[SemanticStateArtifact, ...]
    state_effects: tuple[SemanticStateEffect, ...]

    def walk_operations(self) -> Iterator[SemanticOperation]:
        for item in self.root.walk_items():
            if isinstance(item, SemanticOperation):
                yield item


@dataclass(frozen=True)
class ProvenanceGraph:
    schema: str
    records: tuple[DerivationRecord, ...]


@dataclass(frozen=True)
class CompilerDiagnostic:
    severity: str
    code: str
    message: str
    stable_path: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    check: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SemanticCompilationResult:
    semantic_ir: SemanticProgram
    provenance: ProvenanceGraph
    diagnostics: tuple[CompilerDiagnostic, ...]
    validation_results: tuple[ValidationResult, ...]
    compilation_fingerprint: str
