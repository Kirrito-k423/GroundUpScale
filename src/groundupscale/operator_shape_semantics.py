"""Operator-specific Shape identity and declared-work semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Any


class UnsupportedOperatorShape(ValueError):
    """An operation has no complete, supported Shape semantics contract."""


@dataclass(frozen=True)
class OperatorShapeSemantics:
    operation: str
    normalized_shape: dict[str, object]
    shape_identity: str
    coordinate_axis: str
    coordinate_value: int | None
    domain_facets: dict[str, object]
    work_formula: dict[str, object]
    declared_work: float


def _identity(operation: str, shape: dict[str, object]) -> str:
    payload = json.dumps(
        {"operation": operation, "shape": shape},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"operator-shape://{operation}/{sha256(payload).hexdigest()}"


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _matmul_from_case(case: dict[str, object]) -> OperatorShapeSemantics:
    shape = case.get("shape")
    left = shape.get("left") if isinstance(shape, dict) else None
    right = shape.get("right") if isinstance(shape, dict) else None
    if (
        not isinstance(left, list)
        or not isinstance(right, list)
        or len(left) != 2
        or len(right) != 2
        or not all(_positive_integer(value) for value in (*left, *right))
        or left[1] != right[0]
    ):
        raise UnsupportedOperatorShape(
            "MatMul requires positive integer M/N/K and compatible operands"
        )
    m, k, n = int(left[0]), int(left[1]), int(right[1])
    normalized = {"m": m, "n": n, "k": k}
    return OperatorShapeSemantics(
        operation="MatMul",
        normalized_shape=normalized,
        shape_identity=_identity("MatMul", normalized),
        coordinate_axis="m",
        coordinate_value=m,
        domain_facets={
            "semantic_operation": "MatMul",
            "dtype": case.get("dtype"),
            "layout": case.get("layout"),
        },
        work_formula={
            "kind": "matmul-2mnk-fixed-nk",
            "version": "v1",
            "fixed_n": n,
            "fixed_k": k,
            "work_unit": "FLOP",
        },
        declared_work=float(2 * m * n * k),
    )


def _flash_attention_from_parts(
    *,
    sequence_lengths: object,
    sequence_count: object,
    head_count: object,
    head_dimension: object,
    dtype: object,
    layout: object,
    causal: object,
    mask: object,
    dropout_probability: object,
    mode: object,
) -> OperatorShapeSemantics:
    if (
        not isinstance(sequence_lengths, list)
        or not sequence_lengths
        or not all(_positive_integer(value) for value in sequence_lengths)
        or not _positive_integer(sequence_count)
        or int(sequence_count) != len(sequence_lengths)
        or not _positive_integer(head_count)
        or not _positive_integer(head_dimension)
        or dtype != "float16"
        or layout != "TND"
        or not isinstance(causal, bool)
        or mask != "none"
        or not isinstance(dropout_probability, (int, float))
        or isinstance(dropout_probability, bool)
        or not isfinite(float(dropout_probability))
        or float(dropout_probability) != 0.0
        or mode != "forward"
    ):
        raise UnsupportedOperatorShape(
            "FlashAttentionForward requires a positive TND forward Shape, "
            "float16, no mask, and zero dropout"
        )
    lengths = [int(value) for value in sequence_lengths]
    heads = int(head_count)
    dimension = int(head_dimension)
    normalized = {
        "sequence_count": len(lengths),
        "sequence_lengths": lengths,
        "total_tokens": sum(lengths),
        "head_count": heads,
        "head_dimension": dimension,
        "causal": causal,
    }
    attention_pairs = sum(
        length * (length + 1) // 2 if causal else length**2
        for length in lengths
    )
    declared_work = float(4 * heads * dimension * attention_pairs)
    equal_length = len(set(lengths)) == 1
    return OperatorShapeSemantics(
        operation="FlashAttentionForward",
        normalized_shape=normalized,
        shape_identity=_identity("FlashAttentionForward", normalized),
        coordinate_axis="sequence_length",
        coordinate_value=lengths[0] if equal_length else None,
        domain_facets={
            "semantic_operation": "FlashAttentionForward",
            "dtype": dtype,
            "layout": layout,
            "sequence_count": len(lengths),
            "head_count": heads,
            "head_dimension": dimension,
            "causal": causal,
            "mask": mask,
            "dropout_probability": float(dropout_probability),
            "mode": mode,
        },
        work_formula={
            "kind": "flash-attention-tnd-forward-qk-pv",
            "version": "v1",
            "causal": causal,
            "work_unit": "FLOP",
        },
        declared_work=declared_work,
    )


def semantics_from_case(case: dict[str, object]) -> OperatorShapeSemantics:
    operation = case.get("operation")
    if operation == "MatMul":
        return _matmul_from_case(case)
    if operation == "FlashAttentionForward":
        shape = case.get("shape")
        if not isinstance(shape, dict):
            raise UnsupportedOperatorShape("FlashAttentionForward requires Shape")
        return _flash_attention_from_parts(
            sequence_lengths=shape.get("sequence_lengths"),
            sequence_count=shape.get("sequence_count"),
            head_count=shape.get("head_count"),
            head_dimension=shape.get("head_dimension"),
            dtype=case.get("dtype"),
            layout=case.get("layout"),
            causal=case.get("causal"),
            mask=case.get("mask"),
            dropout_probability=case.get("dropout_probability"),
            mode=case.get("mode"),
        )
    raise UnsupportedOperatorShape(f"unsupported operation: {operation!r}")

def semantics_for_coordinate(
    reference: OperatorShapeSemantics, coordinate: int
) -> OperatorShapeSemantics:
    if not _positive_integer(coordinate):
        raise UnsupportedOperatorShape("operator coordinate must be positive")
    if reference.operation == "MatMul":
        shape = reference.normalized_shape
        return _matmul_from_case(
            {
                "operation": "MatMul",
                "shape": {
                    "left": [coordinate, shape["k"]],
                    "right": [shape["k"], shape["n"]],
                },
                "dtype": reference.domain_facets["dtype"],
                "layout": reference.domain_facets["layout"],
            }
        )
    domain = reference.domain_facets
    return _flash_attention_from_parts(
        sequence_lengths=[coordinate] * int(domain["sequence_count"]),
        sequence_count=domain["sequence_count"],
        head_count=domain["head_count"],
        head_dimension=domain["head_dimension"],
        dtype=domain["dtype"],
        layout=domain["layout"],
        causal=domain["causal"],
        mask=domain["mask"],
        dropout_probability=domain["dropout_probability"],
        mode=domain["mode"],
    )


def semantics_from_surface_query(
    surface: dict[str, Any], query_shape: object
) -> OperatorShapeSemantics:
    domain = surface.get("domain")
    if not isinstance(domain, dict) or not isinstance(query_shape, dict):
        raise UnsupportedOperatorShape("query requires Shape and domain")
    operation = domain.get("semantic_operation")
    if operation == "MatMul":
        work_formula = surface.get("work_formula")
        if not isinstance(work_formula, dict):
            raise UnsupportedOperatorShape("MatMul query requires work formula")
        if set(query_shape) == {"m"}:
            return _matmul_from_case(
                {
                    "operation": "MatMul",
                    "shape": {
                        "left": [query_shape["m"], work_formula.get("fixed_k")],
                        "right": [work_formula.get("fixed_k"), work_formula.get("fixed_n")],
                    },
                    "dtype": domain.get("dtype"),
                    "layout": domain.get("layout"),
                }
            )
        raise UnsupportedOperatorShape("unsupported MatMul query Shape")
    if operation == "FlashAttentionForward":
        lengths = (
            [query_shape["sequence_length"]] * int(domain.get("sequence_count", 0))
            if set(query_shape) == {"sequence_length"}
            else query_shape.get("sequence_lengths")
        )
        return _flash_attention_from_parts(
            sequence_lengths=lengths,
            sequence_count=domain.get("sequence_count"),
            head_count=domain.get("head_count"),
            head_dimension=domain.get("head_dimension"),
            dtype=domain.get("dtype"),
            layout=domain.get("layout"),
            causal=domain.get("causal"),
            mask=domain.get("mask"),
            dropout_probability=domain.get("dropout_probability"),
            mode=domain.get("mode"),
        )
    raise UnsupportedOperatorShape(f"unsupported operation: {operation!r}")
