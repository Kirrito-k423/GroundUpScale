"""Immutable IR identities, provenance, and canonical serialization."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


def canonical_data(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: canonical_data(getattr(value, field.name))
            for field in fields(value)
        }
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return canonical_data(model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(key): canonical_data(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [canonical_data(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonical_data(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def content_fingerprint(*values: Any) -> str:
    digest = sha256()
    for value in values:
        digest.update(canonical_json(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def node_identity(ir_kind: str, compilation_fingerprint: str, stable_path: str) -> str:
    return f"{ir_kind}:{content_fingerprint(ir_kind, compilation_fingerprint, stable_path)}"


def derivation_identity(
    phase: str, compilation_fingerprint: str, stable_path: str
) -> str:
    return f"derivation:{content_fingerprint(phase, compilation_fingerprint, stable_path)}"


@dataclass(frozen=True)
class DerivationRecord:
    derivation_id: str
    phase: str
    rule: str
    source_path: str
    source_stable_path: str
    target_node_ids: tuple[str, ...]
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
