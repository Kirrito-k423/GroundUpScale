"""Logical workload control IR; ModelCall remains an expandable action leaf."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TypeAlias

from groundupscale.ir.common import DerivationRecord
from groundupscale.ir.model import IRTensorType


@dataclass(frozen=True)
class IRArtifact:
    name: str
    tensor: IRTensorType
    role: str


@dataclass(frozen=True)
class IRModelCall:
    local_id: str
    definition_id: str
    stable_path: str
    node_id: str
    derivation_ids: tuple[str, ...]
    model_name: str
    model_version: str
    model_reference: str
    entrypoint: str
    inputs: tuple[tuple[str, str], ...]
    outputs: tuple[tuple[str, str], ...]

    def walk(self) -> Iterator[WorkloadNode]:
        yield self


@dataclass(frozen=True)
class IRSequence:
    local_id: str
    definition_id: str
    stable_path: str
    node_id: str
    derivation_ids: tuple[str, ...]
    children: tuple[WorkloadNode, ...]

    def walk(self) -> Iterator[WorkloadNode]:
        yield self
        for child in self.children:
            yield from child.walk()


WorkloadNode: TypeAlias = IRModelCall | IRSequence


@dataclass(frozen=True)
class WorkloadIR:
    schema: str
    name: str
    version: str
    compilation_fingerprint: str
    source_sha256: str
    artifacts: tuple[IRArtifact, ...]
    root: IRSequence
    provenance: tuple[DerivationRecord, ...]

    def walk_nodes(self) -> Iterator[WorkloadNode]:
        return self.root.walk()
