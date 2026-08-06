"""Hardware-independent, hierarchical model structure IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from groundupscale.ir.common import DerivationRecord


ShapeDimension = int | str


@dataclass(frozen=True)
class IRTensorType:
    dtype: str
    shape: tuple[ShapeDimension, ...]
    layout: str


@dataclass(frozen=True)
class IRPort:
    name: str
    tensor: IRTensorType


@dataclass(frozen=True)
class IRState:
    name: str
    role: str
    tensor: IRTensorType
    trainable: bool


@dataclass(frozen=True)
class IRCallStep:
    local_id: str
    target: str
    entrypoint: str
    inputs: tuple[tuple[str, str], ...]
    outputs: tuple[tuple[str, str], ...]
    source_kind: str = "call"


@dataclass(frozen=True)
class IREntrypoint:
    name: str
    inputs: tuple[IRPort, ...]
    outputs: tuple[IRPort, ...]
    steps: tuple[IRCallStep, ...]


@dataclass(frozen=True)
class IRModule:
    local_id: str
    module_kind: str
    operation: str | None
    definition_id: str
    stable_path: str
    node_id: str
    derivation_ids: tuple[str, ...]
    inputs: tuple[IRPort, ...]
    outputs: tuple[IRPort, ...]
    state: tuple[IRState, ...]
    attributes: tuple[tuple[str, str | int | float | bool], ...]
    entrypoints: tuple[IREntrypoint, ...]
    children: tuple[IRModule, ...]
    repeat_group: str | None = None
    repeat_index: int | None = None

    def walk(self) -> Iterator[IRModule]:
        yield self
        for child in self.children:
            yield from child.walk()

    def entrypoint(self, name: str) -> IREntrypoint:
        for entrypoint in self.entrypoints:
            if entrypoint.name == name:
                return entrypoint
        raise KeyError(f"module {self.stable_path!r} has no entrypoint {name!r}")


@dataclass(frozen=True)
class ModelIR:
    schema: str
    name: str
    version: str
    compilation_fingerprint: str
    source_sha256: str
    symbols: tuple[tuple[str, tuple[tuple[str, int | str | None], ...]], ...]
    constraints: tuple[str, ...]
    root: IRModule
    provenance: tuple[DerivationRecord, ...]

    def walk_modules(self) -> Iterator[IRModule]:
        return self.root.walk()

    def entrypoint(self, name: str) -> IREntrypoint:
        return self.root.entrypoint(name)
