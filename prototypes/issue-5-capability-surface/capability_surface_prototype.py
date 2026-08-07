"""PROTOTYPE ONLY: scenario-bound simplicial capability queries for issue 5.

This code is a disposable experiment and MUST NOT evolve into production code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    coordinates: tuple[float, ...]
    rate: float
    standard_uncertainty: float


@dataclass(frozen=True)
class Confirmation:
    confirmation_id: str
    coordinates: tuple[float, ...]
    observed_rate: float


@dataclass(frozen=True)
class Cell:
    cell_id: str
    anchor_ids: tuple[str, ...]
    confirmations: tuple[Confirmation, ...] = ()
    maximum_confirmation_residual_fraction: float = 0.15
    interpolation_standard_uncertainty: float = 0.0


@dataclass(frozen=True)
class Surface:
    surface_id: str
    version: str
    cohort: str
    coordinate_names: tuple[str, ...]
    coordinate_transform: str
    validated_domain: dict[str, tuple[str, ...]]
    anchors: tuple[Anchor, ...]
    cells: tuple[Cell, ...]

    def source_record(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def input_digest(self) -> str:
        return stable_digest(self.source_record())


def _weights_1d(point: tuple[float, ...], anchors: list[Anchor]) -> list[float] | None:
    if len(anchors) != 2:
        raise ValueError("a one-dimensional cell requires two anchors")
    left, right = anchors
    x = point[0]
    x0 = left.coordinates[0]
    x1 = right.coordinates[0]
    if math.isclose(x0, x1):
        return None
    right_weight = (x - x0) / (x1 - x0)
    left_weight = 1.0 - right_weight
    if min(left_weight, right_weight) < -1e-12:
        return None
    return [left_weight, right_weight]


def _weights_2d(point: tuple[float, ...], anchors: list[Anchor]) -> list[float] | None:
    if len(anchors) != 3:
        raise ValueError("a two-dimensional cell requires three anchors")
    (x, y) = point
    (x1, y1), (x2, y2), (x3, y3) = [anchor.coordinates for anchor in anchors]
    denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if math.isclose(denominator, 0.0):
        return None
    first = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
    second = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
    third = 1.0 - first - second
    weights = [first, second, third]
    if min(weights) < -1e-12:
        return None
    return weights


def _cell_weights(
    surface: Surface,
    cell: Cell,
    coordinates: tuple[float, ...],
) -> tuple[list[Anchor], list[float]] | None:
    anchors_by_id = {anchor.anchor_id: anchor for anchor in surface.anchors}
    anchors = [anchors_by_id[anchor_id] for anchor_id in cell.anchor_ids]
    if len(surface.coordinate_names) == 1:
        weights = _weights_1d(coordinates, anchors)
    elif len(surface.coordinate_names) == 2:
        weights = _weights_2d(coordinates, anchors)
    else:
        raise ValueError("this throwaway prototype only supports one or two dimensions")
    if weights is None:
        return None
    return anchors, weights


def _rate(anchors: list[Anchor], weights: list[float]) -> float:
    return sum(weight * anchor.rate for weight, anchor in zip(weights, anchors, strict=True))


def _cell_validation(surface: Surface, cell: Cell) -> dict[str, Any]:
    residuals: list[dict[str, Any]] = []
    for confirmation in cell.confirmations:
        located = _cell_weights(surface, cell, confirmation.coordinates)
        if located is None:
            residuals.append(
                {
                    "confirmation_id": confirmation.confirmation_id,
                    "status": "not_inside_cell",
                }
            )
            continue
        anchors, weights = located
        predicted_rate = _rate(anchors, weights)
        residual_fraction = abs(predicted_rate - confirmation.observed_rate) / confirmation.observed_rate
        residuals.append(
            {
                "confirmation_id": confirmation.confirmation_id,
                "coordinates": list(confirmation.coordinates),
                "observed_rate": confirmation.observed_rate,
                "predicted_rate": predicted_rate,
                "residual_fraction": residual_fraction,
                "status": "evaluated",
            }
        )
    evaluated = [item["residual_fraction"] for item in residuals if item["status"] == "evaluated"]
    maximum_residual = max(evaluated, default=0.0)
    return {
        "confirmations": residuals,
        "maximum_observed_residual_fraction": maximum_residual,
        "maximum_allowed_residual_fraction": cell.maximum_confirmation_residual_fraction,
        "passed": maximum_residual <= cell.maximum_confirmation_residual_fraction,
    }


def _base_result(
    surface: Surface,
    coordinates: tuple[float, ...],
    query_domain: dict[str, str],
) -> dict[str, Any]:
    return {
        "surface_id": surface.surface_id,
        "surface_version": surface.version,
        "hardware_cohort": surface.cohort,
        "surface_input_digest": surface.input_digest,
        "coordinate_transform": surface.coordinate_transform,
        "raw_coordinates": dict(zip(surface.coordinate_names, coordinates, strict=True)),
        "transformed_coordinates": list(coordinates),
        "query_domain": query_domain,
        "cell_id": None,
        "anchors": [],
        "weights": [],
        "rate": None,
        "unit": "FLOP/s",
        "uncertainty": None,
        "status": "unknown",
        "reason": None,
    }


def query_surface(
    surface: Surface,
    coordinates: tuple[float, ...],
    query_domain: dict[str, str],
) -> dict[str, Any]:
    """Return an authoritative point only inside a retained, validated cell."""

    result = _base_result(surface, coordinates, query_domain)
    for key, allowed_values in surface.validated_domain.items():
        observed = query_domain.get(key)
        if observed not in allowed_values:
            result["reason"] = (
                "alignment_regime_unvalidated"
                if key == "alignment"
                else f"{key}_domain_unvalidated"
            )
            return result

    for cell in surface.cells:
        located = _cell_weights(surface, cell, coordinates)
        if located is None:
            continue
        anchors, weights = located
        point_rate = _rate(anchors, weights)
        validation = _cell_validation(surface, cell)
        result.update(
            {
                "cell_id": cell.cell_id,
                "anchors": [
                    {
                        "anchor_id": anchor.anchor_id,
                        "coordinates": list(anchor.coordinates),
                        "rate": anchor.rate,
                        "standard_uncertainty": anchor.standard_uncertainty,
                    }
                    for anchor in anchors
                ],
                "weights": weights,
                "cell_validation": validation,
            }
        )
        exact_index = next(
            (index for index, weight in enumerate(weights) if math.isclose(weight, 1.0, abs_tol=1e-12)),
            None,
        )
        if not validation["passed"] and exact_index is None:
            result["reason"] = "interpolation_error_exceeds_budget"
            result["provisional_rate"] = point_rate
            return result

        anchor_uncertainty = math.sqrt(
            sum(
                (weight * anchor.standard_uncertainty) ** 2
                for weight, anchor in zip(weights, anchors, strict=True)
            )
        )
        interpolation_uncertainty = (
            0.0 if exact_index is not None else cell.interpolation_standard_uncertainty
        )
        total_uncertainty = math.hypot(
            anchor_uncertainty,
            interpolation_uncertainty,
        )
        result.update(
            {
                "rate": point_rate,
                "uncertainty": {
                    "anchor_standard_uncertainty": anchor_uncertainty,
                    "interpolation_standard_uncertainty": interpolation_uncertainty,
                    "combined_standard_uncertainty": total_uncertainty,
                    "rate_interval": [point_rate - total_uncertainty, point_rate + total_uncertainty],
                },
                "status": "exact_anchor" if exact_index is not None else "interpolated",
                "reason": None,
            }
        )
        return result

    result["reason"] = "outside_validated_domain"
    return result


def inspect_m4_observation(path: Path, source_path: str) -> dict[str, Any]:
    raw_bytes = path.read_bytes()
    bundle = json.loads(raw_bytes)
    selected_probe = next(probe for probe in bundle["probes"] if probe["probe_id"] == "matrix-fp32-cube")
    selected_case = next(
        case
        for case in selected_probe["cases"]
        if case["shape"] == [128, 128, 128] and case["threads"] == 4
    )
    environment_eligible = bundle["environment"]["eligible"]
    return {
        "source_path": source_path,
        "source_sha256": f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}",
        "captured_at": bundle["captured_at"],
        "hardware_cohort": bundle["hardware_cohort"],
        "observation_kind": "resource_microbenchmark_not_operator_frontier_anchor",
        "environment": {
            "eligible": environment_eligible,
            "reason_codes": bundle["environment"]["reason_codes"],
            "policy_id": bundle["environment"]["policy"]["policy_id"],
        },
        "selected_case": {
            "probe_id": selected_probe["probe_id"],
            "shape": selected_case["shape"],
            "threads": selected_case["threads"],
            "implementation": selected_case["implementation"],
            "case_eligible": selected_case["eligible"],
            "achieved_rate": selected_case["achieved_rate"],
            "unit": selected_case["unit"],
        },
        "frontier_status": "unknown",
        "reason": "environment_ineligible" if not environment_eligible else "not_frontier_qualified",
        "authoritative_rate": None,
    }


def smooth_1d_surface() -> Surface:
    return Surface(
        surface_id="synthetic-square-matmul-scale",
        version="v1",
        cohort="synthetic-cohort",
        coordinate_names=("S",),
        coordinate_transform="identity/v1",
        validated_domain={"alignment": ("aligned", "non_aligned"), "dtype": ("fp32",)},
        anchors=(
            Anchor("smooth-a128", (128.0,), 1.2e12, 0.02e12),
            Anchor("smooth-a512", (512.0,), 1.8e12, 0.03e12),
        ),
        cells=(
            Cell(
                "smooth-line-128-512",
                ("smooth-a128", "smooth-a512"),
                interpolation_standard_uncertainty=0.04e12,
            ),
        ),
    )


def smooth_1d_surface_v2() -> Surface:
    return Surface(
        surface_id="synthetic-square-matmul-scale",
        version="v2",
        cohort="synthetic-cohort",
        coordinate_names=("S",),
        coordinate_transform="identity/v1",
        validated_domain={"alignment": ("aligned", "non_aligned"), "dtype": ("fp32",)},
        anchors=(
            Anchor("smooth-a128", (128.0,), 1.2e12, 0.02e12),
            Anchor("smooth-a201", (201.0,), 1.28e12, 0.018e12),
            Anchor("smooth-a512", (512.0,), 1.8e12, 0.03e12),
        ),
        cells=(
            Cell(
                "smooth-line-128-201",
                ("smooth-a128", "smooth-a201"),
                interpolation_standard_uncertainty=0.025e12,
            ),
            Cell(
                "smooth-line-201-512",
                ("smooth-a201", "smooth-a512"),
                interpolation_standard_uncertainty=0.035e12,
            ),
        ),
    )


def aligned_only_surface() -> Surface:
    return Surface(
        surface_id="synthetic-aligned-only",
        version="v1",
        cohort="synthetic-cohort",
        coordinate_names=("S",),
        coordinate_transform="identity/v1",
        validated_domain={"alignment": ("aligned",), "dtype": ("fp32",)},
        anchors=(
            Anchor("aligned-a128", (128.0,), 1.2e12, 0.02e12),
            Anchor("aligned-a512", (512.0,), 1.8e12, 0.03e12),
        ),
        cells=(Cell("aligned-line-128-512", ("aligned-a128", "aligned-a512")),),
    )


def matmul_2d_surface() -> Surface:
    return Surface(
        surface_id="synthetic-matmul-mn-k256",
        version="v1",
        cohort="synthetic-cohort",
        coordinate_names=("M", "N"),
        coordinate_transform="identity/v1",
        validated_domain={"alignment": ("aligned",), "dtype": ("fp32",), "K": ("256",)},
        anchors=(
            Anchor("matmul-A", (128.0, 128.0), 1.2e12, 0.02e12),
            Anchor("matmul-B", (512.0, 128.0), 1.8e12, 0.03e12),
            Anchor("matmul-C", (128.0, 512.0), 1.5e12, 0.025e12),
        ),
        cells=(
            Cell(
                "matmul-triangle-ABC",
                ("matmul-A", "matmul-B", "matmul-C"),
                interpolation_standard_uncertainty=0.05e12,
            ),
        ),
    )


def contradicted_cliff_surface() -> Surface:
    return Surface(
        surface_id="synthetic-contradicted-sparse-cell",
        version="v1",
        cohort="synthetic-cohort",
        coordinate_names=("S",),
        coordinate_transform="identity/v1",
        validated_domain={"alignment": ("mixed_confirmed",), "dtype": ("fp32",)},
        anchors=(
            Anchor("cliff-a128", (128.0,), 1.2e12, 0.02e12),
            Anchor("cliff-a512", (512.0,), 1.8e12, 0.03e12),
        ),
        cells=(
            Cell(
                "cliff-line-128-512",
                ("cliff-a128", "cliff-a512"),
                confirmations=(Confirmation("cliff-confirmation-256", (256.0,), 0.65e12),),
                maximum_confirmation_residual_fraction=0.15,
            ),
        ),
    )
