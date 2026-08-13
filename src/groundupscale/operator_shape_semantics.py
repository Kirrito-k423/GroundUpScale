"""Operator-specific Shape identity and declared-work semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, prod
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
        or len(left) < 2
        or len(right) < 2
        or not all(_positive_integer(value) for value in (*left, *right))
        or left[-1] != right[-2]
    ):
        raise UnsupportedOperatorShape(
            "MatMul requires positive integer M/N/K and compatible operands"
        )
    m, k, n = int(left[-2]), int(left[-1]), int(right[-1])
    left_batch = [int(value) for value in left[:-2]]
    right_batch = [int(value) for value in right[:-2]]
    try:
        reversed_batch = []
        for left_dimension, right_dimension in zip(
            reversed([1] * (len(right_batch) - len(left_batch)) + left_batch),
            reversed([1] * (len(left_batch) - len(right_batch)) + right_batch),
            strict=True,
        ):
            if left_dimension != right_dimension and 1 not in {
                left_dimension,
                right_dimension,
            }:
                raise ValueError
            reversed_batch.append(max(left_dimension, right_dimension))
        batch = list(reversed(reversed_batch))
    except ValueError as error:
        raise UnsupportedOperatorShape(
            "MatMul requires broadcast-compatible batch dimensions"
        ) from error
    normalized = {"m": m, "n": n, "k": k}
    if batch:
        normalized.update(
            {
                "left_batch_shape": left_batch,
                "right_batch_shape": right_batch,
                "batch_shape": batch,
            }
        )
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
        declared_work=float(2 * prod(batch) * m * n * k),
    )


def _softmax_phase_from_case(
    case: dict[str, object],
) -> OperatorShapeSemantics:
    shape = case.get("shape")
    axis = case.get("axis")
    phase = case.get("phase")
    if (
        not isinstance(shape, list)
        or not shape
        or not all(_positive_integer(value) for value in shape)
        or not isinstance(axis, int)
        or isinstance(axis, bool)
        or not -len(shape) <= axis < len(shape)
        or phase
        not in {"max_reduce", "subtract", "exp", "sum_reduce", "normalize"}
        or case.get("dtype") != "float32"
        or case.get("layout") != "contiguous"
    ):
        raise UnsupportedOperatorShape(
            "SoftmaxPhase requires a positive float32 contiguous Shape, valid "
            "axis, and one mandatory Softmax phase"
        )
    normalized = {
        "shape": [int(value) for value in shape],
        "axis": axis,
        "phase": phase,
    }
    element_count = 1
    for value in shape:
        element_count *= int(value)
    reduction_count = element_count // int(shape[axis])
    declared_work = (
        element_count - reduction_count
        if phase in {"max_reduce", "sum_reduce"}
        else element_count
    )
    return OperatorShapeSemantics(
        operation="SoftmaxPhase",
        normalized_shape=normalized,
        shape_identity=_identity("SoftmaxPhase", normalized),
        coordinate_axis="exact-shape",
        coordinate_value=None,
        domain_facets={
            "semantic_operation": "SoftmaxPhase",
            "phase": phase,
            "shape": normalized["shape"],
            "axis": axis,
            "dtype": "float32",
            "layout": "contiguous",
        },
        work_formula={
            "kind": "softmax-phase-exact-invocation",
            "version": "v1",
            "work_unit": "elementary-operation",
        },
        declared_work=float(declared_work),
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


_ELEMENTWISE_OPERAND_KINDS = {
    "Add": {"tensor-tensor", "tensor-broadcast"},
    "Mul": {"tensor-tensor", "tensor-scalar"},
    "SiLU": {"tensor"},
}


def _elementwise_from_case(case: dict[str, object]) -> OperatorShapeSemantics:
    operation = str(case.get("operation"))
    shape = case.get("shape")
    result = shape.get("result") if isinstance(shape, dict) else None
    operand_kind = case.get("operand_kind")
    if (
        operation not in _ELEMENTWISE_OPERAND_KINDS
        or case.get("dtype") != "float32"
        or case.get("layout") != "contiguous"
        or not isinstance(result, list)
        or not result
        or not all(_positive_integer(value) for value in result)
        or operand_kind not in _ELEMENTWISE_OPERAND_KINDS[operation]
    ):
        raise UnsupportedOperatorShape(
            "elementwise operations require a positive contiguous float32 "
            "result Shape and an operation-specific operand kind"
        )
    normalized_result = [int(value) for value in result]
    elements = prod(normalized_result)
    normalized = {"result": normalized_result, "elements": elements}
    operations_per_element = 5 if operation == "SiLU" else 1
    return OperatorShapeSemantics(
        operation=operation,
        normalized_shape=normalized,
        shape_identity=_identity(operation, normalized),
        coordinate_axis="elements",
        coordinate_value=elements,
        domain_facets={
            "semantic_operation": operation,
            "dtype": case.get("dtype"),
            "layout": case.get("layout"),
            "operand_kind": operand_kind,
        },
        work_formula={
            "kind": "elementwise-result-elements",
            "version": "v1",
            "operations_per_element": operations_per_element,
            "work_unit": "FLOP",
        },
        declared_work=float(elements * operations_per_element),
    )


def semantics_from_case(case: dict[str, object]) -> OperatorShapeSemantics:
    operation = case.get("operation")
    if operation == "MatMul":
        return _matmul_from_case(case)
    if operation == "SoftmaxPhase":
        return _softmax_phase_from_case(case)
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
    if operation in _ELEMENTWISE_OPERAND_KINDS:
        return _elementwise_from_case(case)
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
                        "left": [
                            query_shape["m"],
                            work_formula.get("fixed_k"),
                        ],
                        "right": [
                            work_formula.get("fixed_k"),
                            work_formula.get("fixed_n"),
                        ],
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
    if operation in _ELEMENTWISE_OPERAND_KINDS:
        result = query_shape.get("result")
        return _elementwise_from_case(
            {
                "operation": operation,
                "shape": {"result": result},
                "operand_kind": domain.get("operand_kind"),
                "dtype": domain.get("dtype"),
                "layout": domain.get("layout"),
            }
        )
    raise UnsupportedOperatorShape(f"unsupported operation: {operation!r}")
