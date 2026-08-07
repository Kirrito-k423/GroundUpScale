from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from groundupscale.diagnostics import (
    DiagnosticBundleIntegrityError,
    diagnose_run_bundle,
)
from _diagnostic_test_support import (
    canonical_digest as _canonical_digest,
    write_json as _write_json,
)


def _surface_version(
    version: str,
    *,
    previous_version: str | None,
    include_201_anchor: bool = False,
) -> dict[str, object]:
    domain = {
        "semantic_operation": "MatMul",
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "alignment_regime": "all-query-shapes-validated",
        "alignment_validated": True,
        "working_set_regime": "l2-resident",
        "regime_validated": True,
        "execution_mode": "eager",
        "threads": 10,
    }
    anchors: list[dict[str, object]] = [
        {
            "anchor_id": "anchor-128",
            "anchor_version": "v1",
            "shape": {"s": 128},
            "effective_rate": 1_200_000_000_000.0,
            "rate_unit": "FLOP/s",
            "candidate_id": "fixture.matmul.128",
            "candidate_family": "fixture-matmul-family",
            "cohort_id": "fixture-m4-cohort-v1",
            "domain": domain,
            "observation_validity": "QUALIFIED",
            "frontier_role": "ACTIVE",
            "evidence_ref": "artifact://frontier/anchor-128.json",
        },
        {
            "anchor_id": "anchor-512",
            "anchor_version": "v1",
            "shape": {"s": 512},
            "effective_rate": 1_800_000_000_000.0,
            "rate_unit": "FLOP/s",
            "candidate_id": "fixture.matmul.512",
            "candidate_family": "fixture-matmul-family",
            "cohort_id": "fixture-m4-cohort-v1",
            "domain": domain,
            "observation_validity": "QUALIFIED",
            "frontier_role": "ACTIVE",
            "evidence_ref": "artifact://frontier/anchor-512.json",
        },
    ]
    cells: list[dict[str, object]] = [
        {
            "cell_id": "cell-128-512",
            "anchor_ids": ["anchor-128", "anchor-512"],
            "status": "retained",
            "regime_id": "aligned-l2-v1",
            "confirmation_evidence_refs": [
                "artifact://frontier/confirmation-v1.json"
            ],
            "interpolation_standard_uncertainty_rate": 40_000_000_000.0,
        }
    ]
    covariance = [
        [400_000_000_000_000_000_000.0, 0.0],
        [0.0, 900_000_000_000_000_000_000.0],
    ]
    if include_201_anchor:
        anchors.insert(
            1,
            {
                "anchor_id": "anchor-201",
                "anchor_version": "v1",
                "shape": {"s": 201},
                "effective_rate": 1_280_000_000_000.0,
                "rate_unit": "FLOP/s",
                "candidate_id": "fixture.matmul.201",
                "candidate_family": "fixture-matmul-family",
                "cohort_id": "fixture-m4-cohort-v1",
                "domain": domain,
                "observation_validity": "QUALIFIED",
                "frontier_role": "ACTIVE",
                "evidence_ref": "artifact://frontier/anchor-201.json",
            },
        )
        cells = [
            {
                "cell_id": "cell-128-201",
                "anchor_ids": ["anchor-128", "anchor-201"],
                "status": "retained",
                "regime_id": "aligned-l2-v1",
                "confirmation_evidence_refs": [
                    "artifact://frontier/confirmation-v2-left.json"
                ],
                "interpolation_standard_uncertainty_rate": 30_000_000_000.0,
            },
            {
                "cell_id": "cell-201-512",
                "anchor_ids": ["anchor-201", "anchor-512"],
                "status": "retained",
                "regime_id": "aligned-l2-v1",
                "confirmation_evidence_refs": [
                    "artifact://frontier/confirmation-v2-right.json"
                ],
                "interpolation_standard_uncertainty_rate": 30_000_000_000.0,
            },
        ]
        covariance = [
            [400_000_000_000_000_000_000.0, 0.0, 0.0],
            [0.0, 625_000_000_000_000_000_000.0, 0.0],
            [0.0, 0.0, 900_000_000_000_000_000_000.0],
        ]
    surface: dict[str, object] = {
        "surface_id": "surface://fixture/matmul/1d",
        "version": version,
        "previous_version": previous_version,
        "cohort_id": "fixture-m4-cohort-v1",
        "domain": domain,
        "candidate_family": "fixture-matmul-family",
        "coordinate": {
            "axis": "s",
            "transform": "identity",
            "transform_version": "v1",
        },
        "work_formula": {
            "kind": "square-matmul-2s3",
            "version": "v1",
            "work_unit": "FLOP",
        },
        "anchors": anchors,
        "cells": cells,
        "uncertainty_policy": {
            "policy_id": "fixture-surface-uncertainty",
            "version": "v1",
            "scope": "fixture MatMul/float32/M4",
            "change_reason": "calibrated fixture policy",
            "revalidation": "on cohort, anchor, or cell change",
            "combination": "root-sum-of-squares",
            "target_coverage": 0.95,
            "anchor_covariance": covariance,
            "instrumentation_standard_uncertainty_rate": 10_000_000_000.0,
            "calibration_evidence_refs": [
                "artifact://frontier/uncertainty-calibration.json"
            ],
        },
        "evidence_refs": ["artifact://frontier/surface-build.json"],
    }
    surface["input_digest"] = _canonical_digest(surface)
    return surface


def _two_dimensional_surface(
    *,
    surface_id: str = "surface://fixture/matmul/2d/family-a",
    candidate_family: str = "fixture-matmul-family-a",
    algorithm_family: str = "blocked-matmul",
    version: str = "v1",
    previous_version: str | None = None,
) -> dict[str, object]:
    domain = {
        "semantic_operation": "MatMul",
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "alignment_regime": "all-query-shapes-validated",
        "alignment_validated": True,
        "working_set_regime": "l2-resident",
        "working_set_validated": True,
        "kernel_dispatch_regime": "fixture-kernel-v1",
        "kernel_dispatch_validated": True,
        "regime_validated": True,
        "execution_mode": "eager",
        "threads": 10,
    }
    anchors: list[dict[str, object]] = []
    for anchor_id, shape, rate in (
        ("anchor-a", {"m": 128, "n": 128}, 1_200_000_000_000.0),
        ("anchor-b", {"m": 512, "n": 128}, 1_800_000_000_000.0),
        ("anchor-c", {"m": 128, "n": 512}, 1_500_000_000_000.0),
    ):
        anchors.append(
            {
                "anchor_id": f"{candidate_family}-{anchor_id}",
                "anchor_version": "v1",
                "shape": shape,
                "effective_rate": rate,
                "rate_unit": "FLOP/s",
                "candidate_id": f"{candidate_family}.{anchor_id}",
                "candidate_family": candidate_family,
                "algorithm_family": algorithm_family,
                "cohort_id": "fixture-m4-cohort-v1",
                "domain": domain,
                "observation_validity": "QUALIFIED",
                "frontier_role": "ACTIVE",
                "evidence_ref": f"artifact://frontier/{candidate_family}-{anchor_id}.json",
            }
        )
    surface: dict[str, object] = {
        "surface_id": surface_id,
        "version": version,
        "previous_version": previous_version,
        "cohort_id": "fixture-m4-cohort-v1",
        "domain": domain,
        "candidate_family": candidate_family,
        "algorithm_family": algorithm_family,
        "coordinate": {
            "axes": ["m", "n"],
            "transform": "identity",
            "transform_version": "v1",
        },
        "domain_policy": {
            "policy_id": "fixture-2d-domain",
            "version": "v1",
            "scope": "fixture MatMul/float32/M4",
            "change_reason": "validated fixture simplex limits",
            "revalidation": "on cohort, anchor, cell, or regime change",
            "cell_kind": "2d-simplex",
            "max_edge_span": 600.0,
            "minimum_twice_area": 1.0,
            "barycentric_tolerance": 1e-12,
        },
        "work_formula": {
            "kind": "matmul-2mnk",
            "fixed_k": 256,
            "version": "v1",
            "work_unit": "FLOP",
        },
        "anchors": anchors,
        "cells": [
            {
                "cell_id": f"{candidate_family}-cell-abc",
                "anchor_ids": [anchor["anchor_id"] for anchor in anchors],
                "status": "retained",
                "regime_id": "aligned-l2-kernel-v1",
                "confirmation_evidence_refs": [
                    f"artifact://frontier/{candidate_family}-confirmation.json"
                ],
                "interpolation_standard_uncertainty_rate": 40_000_000_000.0,
            }
        ],
        "uncertainty_policy": {
            "policy_id": "fixture-surface-uncertainty",
            "version": "v1",
            "scope": "fixture MatMul/float32/M4",
            "change_reason": "calibrated fixture policy",
            "revalidation": "on cohort, anchor, or cell change",
            "combination": "root-sum-of-squares",
            "target_coverage": 0.95,
            "anchor_covariance": [
                [400_000_000_000_000_000_000.0, 0.0, 0.0],
                [0.0, 900_000_000_000_000_000_000.0, 0.0],
                [0.0, 0.0, 625_000_000_000_000_000_000.0],
            ],
            "instrumentation_standard_uncertainty_rate": 10_000_000_000.0,
            "calibration_evidence_refs": [
                "artifact://frontier/uncertainty-calibration.json"
            ],
        },
        "evidence_refs": ["artifact://frontier/surface-2d-build.json"],
    }
    surface["input_digest"] = _canonical_digest(surface)
    return surface


def _two_dimensional_grid_surface(
    *,
    version: str = "v1",
    previous_version: str | None = None,
) -> dict[str, object]:
    surface = _two_dimensional_surface(
        version=version,
        previous_version=previous_version,
    )
    domain = surface["domain"]
    candidate_family = surface["candidate_family"]
    surface["anchors"].append(
        {
            "anchor_id": f"{candidate_family}-anchor-d",
            "anchor_version": "v1",
            "shape": {"m": 512, "n": 512},
            "effective_rate": 1_900_000_000_000.0,
            "rate_unit": "FLOP/s",
            "candidate_id": f"{candidate_family}.anchor-d",
            "candidate_family": candidate_family,
            "algorithm_family": surface["algorithm_family"],
            "cohort_id": "fixture-m4-cohort-v1",
            "domain": domain,
            "observation_validity": "QUALIFIED",
            "frontier_role": "ACTIVE",
            "evidence_ref": (
                f"artifact://frontier/{candidate_family}-anchor-d.json"
            ),
        }
    )
    surface["cells"].append(
        {
            "cell_id": f"{candidate_family}-cell-bdc",
            "anchor_ids": [
                f"{candidate_family}-anchor-b",
                f"{candidate_family}-anchor-d",
                f"{candidate_family}-anchor-c",
            ],
            "status": "retained",
            "regime_id": "aligned-l2-kernel-v1",
            "confirmation_evidence_refs": [
                f"artifact://frontier/{candidate_family}-confirmation-bdc.json"
            ],
            "interpolation_standard_uncertainty_rate": 40_000_000_000.0,
        }
    )
    covariance = surface["uncertainty_policy"]["anchor_covariance"]
    for row in covariance:
        row.append(0.0)
    covariance.append(
        [0.0, 0.0, 0.0, 784_000_000_000_000_000_000.0]
    )
    _refresh_surface_digest(surface)
    return surface


def _candidate_envelope(
    surfaces: list[dict[str, object]],
    *,
    validated_seams: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    envelope: dict[str, object] = {
        "envelope_id": "envelope://fixture/matmul/2d",
        "version": "v1",
        "cohort_id": "fixture-m4-cohort-v1",
        "domain": surfaces[0]["domain"],
        "domain_policy": surfaces[0]["domain_policy"],
        "surface_refs": [
            {
                "surface_id": surface["surface_id"],
                "version": surface["version"],
            }
            for surface in surfaces
        ],
        "support_policy": {
            "policy_id": "fixture-candidate-support",
            "version": "v1",
            "scope": "fixture MatMul/float32/M4",
            "change_reason": "validated candidate envelope fixture",
            "revalidation": "on facet support or seam evidence change",
            "rule": "common-stable-support-or-validated-seam",
            "validated_seams": validated_seams or [],
        },
        "evidence_refs": ["artifact://frontier/candidate-envelope.json"],
    }
    envelope["input_digest"] = _canonical_digest(envelope)
    return envelope


def _write_surface_bundle(
    tmp_path: Path,
    *,
    surfaces: list[dict[str, object]] | None = None,
    queries: list[dict[str, object]] | None = None,
    surface_updates: list[dict[str, object]] | None = None,
    candidate_envelopes: list[dict[str, object]] | None = None,
) -> Path:
    run = tmp_path / "surface-bundle"
    domain = _surface_version("v1", previous_version=None)["domain"]
    inputs = {
        "resolved_configuration": {
            "analysis_plan": "fixture-capability-surface",
            "benchmark_case": "matmul-square-shape-query",
        },
        "resolved_ir": {
            "semantic_node": "semantic/model/layers/0/matmul",
            "operation": "MatMul",
        },
        "hardware": {
            "device": "Fixture M4 CPU",
            "partition": "host",
            "topology": "single-socket",
            "software": "fixture-runtime-v1",
        },
        "cohort_id": "fixture-m4-cohort-v1",
        "execution_domain": {
            "shape": {"s": 201},
            "dtype": "float32",
            "layout": "row-major-contiguous",
            "alignment_bytes": 1,
            "threads": 10,
            "execution_mode": "eager",
        },
    }
    evidence = {
        "capability_surfaces": (
            surfaces
            if surfaces is not None
            else [_surface_version("v1", previous_version=None)]
        ),
        "surface_queries": (
            queries
            if queries is not None
            else [
                {
                    "query_id": "q128-v1",
                    "surface_id": "surface://fixture/matmul/1d",
                    "surface_version": "v1",
                    "shape": {"s": 128},
                    "domain": domain,
                },
                {
                    "query_id": "q201-v1",
                    "surface_id": "surface://fixture/matmul/1d",
                    "surface_version": "v1",
                    "shape": {"s": 201},
                    "domain": domain,
                },
                {
                    "query_id": "q512-v1",
                    "surface_id": "surface://fixture/matmul/1d",
                    "surface_version": "v1",
                    "shape": {"s": 512},
                    "domain": domain,
                },
            ]
        ),
    }
    if surface_updates is not None:
        evidence["surface_updates"] = surface_updates
    if candidate_envelopes is not None:
        evidence["candidate_envelopes"] = candidate_envelopes
    document = {
        "schema": "groundupscale.dev/diagnostic-evidence/v1alpha1",
        **inputs,
        **evidence,
        "digests": {
            "input_sha256": _canonical_digest(inputs),
            "evidence_sha256": _canonical_digest(evidence),
        },
    }
    evidence_path = run / "diagnostic/evidence.json"
    evidence_digest = _write_json(evidence_path, document)
    manifest = {
        "schema": "groundupscale.dev/run-manifest/v1alpha1",
        "run_id": "fixture-surface-bundle",
        "status": "completed",
        "device": "cpu",
        "hardware_cohort": "fixture-m4-cohort-v1",
        "artifacts": [
            {
                "role": "diagnostic-evidence",
                "path": "diagnostic/evidence.json",
                "schema": document["schema"],
                "media_type": "application/json",
                "sha256": evidence_digest,
                "produced_by": "groundupscale-test-fixture",
                "inputs": [],
            }
        ],
    }
    _write_json(run / "run.manifest.json", manifest)
    return run


def _refresh_surface_digest(surface: dict[str, object]) -> None:
    surface.pop("input_digest", None)
    surface["input_digest"] = _canonical_digest(surface)


def _queries_by_id(result: dict[str, object]) -> dict[str, dict[str, object]]:
    queries = result["capability_surface_queries"]
    assert isinstance(queries, list)
    return {query["query_id"]: query for query in queries}


def test_2d_retained_simplex_interpolates_and_rejects_false_bounding_box(
    tmp_path: Path,
) -> None:
    surface = _two_dimensional_surface()
    domain = surface["domain"]
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[surface],
        queries=[
            {
                "query_id": "q-256-320",
                "surface_id": surface["surface_id"],
                "surface_version": "v1",
                "shape": {"m": 256, "n": 320},
                "domain": domain,
            },
            {
                "query_id": "q-400-400",
                "surface_id": surface["surface_id"],
                "surface_version": "v1",
                "shape": {"m": 400, "n": 400},
                "domain": domain,
            },
        ],
    )

    queries = _queries_by_id(diagnose_run_bundle(run))

    inside = queries["q-256-320"]
    assert inside["status"] == "interpolated"
    assert inside["reason_code"] is None
    assert inside["cell_id"] == "fixture-matmul-family-a-cell-abc"
    assert inside["weights"] == pytest.approx([1 / 6, 1 / 3, 1 / 2])
    assert inside["effective_rate"] == pytest.approx(
        {"value": 1_550_000_000_000.0, "unit": "FLOP/s"}
    )
    assert inside["work_rate_latency"] == pytest.approx(
        {
            "declared_work": 41_943_040.0,
            "work_unit": "FLOP",
            "value_ns": 27_060.025806451613,
        }
    )
    assert [anchor["anchor_id"] for anchor in inside["anchors"]] == [
        "fixture-matmul-family-a-anchor-a",
        "fixture-matmul-family-a-anchor-b",
        "fixture-matmul-family-a-anchor-c",
    ]
    assert inside["selected_candidate_family"] == "fixture-matmul-family-a"
    assert inside["domain_policy"] == surface["domain_policy"]
    assert set(inside["uncertainty"]["components"]) == {
        "anchor_standard_rate",
        "interpolation_standard_rate",
        "instrumentation_standard_rate",
    }

    false_box = queries["q-400-400"]
    assert false_box["status"] == "unknown"
    assert false_box["reason_code"] == "outside_validated_domain"
    assert false_box["effective_rate"] is None
    assert false_box["anchors"] == []
    assert false_box["weights"] == []


def test_2d_property_exact_knots_and_outside_points_never_guess(
    tmp_path: Path,
) -> None:
    surface = _two_dimensional_surface()
    exact_shapes = [(128, 128), (512, 128), (128, 512)]
    outside_shapes = [(400, 400), (127, 320), (320, 513)]
    queries = [
        {
            "query_id": f"q-exact-{m}-{n}",
            "surface_id": surface["surface_id"],
            "surface_version": "v1",
            "shape": {"m": m, "n": n},
            "domain": surface["domain"],
        }
        for m, n in exact_shapes
    ] + [
        {
            "query_id": f"q-outside-{m}-{n}",
            "surface_id": surface["surface_id"],
            "surface_version": "v1",
            "shape": {"m": m, "n": n},
            "domain": surface["domain"],
        }
        for m, n in outside_shapes
    ]

    results = _queries_by_id(
        diagnose_run_bundle(
            _write_surface_bundle(
                tmp_path, surfaces=[surface], queries=queries
            )
        )
    )

    for m, n in exact_shapes:
        query = results[f"q-exact-{m}-{n}"]
        assert query["status"] == "exact_anchor"
        assert sum(query["weights"]) == pytest.approx(1.0)
        assert sorted(query["weights"]) == pytest.approx([0.0, 0.0, 1.0])
    for m, n in outside_shapes:
        query = results[f"q-outside-{m}-{n}"]
        assert query["status"] == "unknown"
        assert query["reason_code"] == "outside_validated_domain"
        assert query["effective_rate"] is None


def test_2d_property_adjacent_retained_cells_are_c0_on_shared_edge(
    tmp_path: Path,
) -> None:
    surface = _two_dimensional_grid_surface()
    points = {
        "left": (319.999999, 319.999999),
        "seam": (320.0, 320.0),
        "right": (320.000001, 320.000001),
    }
    queries = [
        {
            "query_id": f"q-{name}",
            "surface_id": surface["surface_id"],
            "surface_version": "v1",
            "shape": {"m": m, "n": n},
            "domain": surface["domain"],
        }
        for name, (m, n) in points.items()
    ]

    results = _queries_by_id(
        diagnose_run_bundle(
            _write_surface_bundle(
                tmp_path, surfaces=[surface], queries=queries
            )
        )
    )

    assert results["q-left"]["cell_id"].endswith("cell-abc")
    assert results["q-right"]["cell_id"].endswith("cell-bdc")
    seam_rate = results["q-seam"]["effective_rate"]["value"]
    assert seam_rate == pytest.approx(1_650_000_000_000.0)
    assert abs(results["q-left"]["effective_rate"]["value"] - seam_rate) < 10_000
    assert abs(results["q-right"]["effective_rate"]["value"] - seam_rate) < 10_000


def test_2d_property_surface_versions_replay_deterministically(
    tmp_path: Path,
) -> None:
    v1 = _two_dimensional_surface()
    v2 = _two_dimensional_grid_surface(version="v2", previous_version="v1")
    queries = [
        {
            "query_id": f"q-400-400-{version}",
            "surface_id": v1["surface_id"],
            "surface_version": version,
            "shape": {"m": 400, "n": 400},
            "domain": v1["domain"],
        }
        for version in ("v1", "v2")
    ]
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[v1, v2],
        queries=queries,
    )

    first = diagnose_run_bundle(run)
    replay = diagnose_run_bundle(run)

    assert replay == first
    results = _queries_by_id(first)
    assert results["q-400-400-v1"]["reason_code"] == (
        "outside_validated_domain"
    )
    assert results["q-400-400-v2"]["status"] == "interpolated"
    summaries = {
        surface["version"]: surface for surface in first["capability_surfaces"]
    }
    assert summaries["v2"]["transition"]["added_anchor_ids"] == [
        "fixture-matmul-family-a-anchor-d"
    ]
    assert summaries["v1"]["input_digest"] != summaries["v2"]["input_digest"]


@pytest.mark.parametrize(
    ("failure", "query_shape", "reason_code"),
    [
        ("degenerate", {"m": 256, "n": 128}, "degenerate_simplex"),
        (
            "long-cell",
            {"m": 256, "n": 320},
            "cell_span_exceeds_policy",
        ),
        ("explicit-hole", {"m": 256, "n": 320}, "explicit_domain_hole"),
    ],
)
def test_2d_invalid_or_removed_cell_preserves_stable_rejection(
    tmp_path: Path,
    failure: str,
    query_shape: dict[str, int],
    reason_code: str,
) -> None:
    surface = _two_dimensional_surface()
    cell = surface["cells"][0]
    if failure == "degenerate":
        surface["anchors"][2]["shape"] = {"m": 320, "n": 128}
    elif failure == "long-cell":
        surface["domain_policy"]["max_edge_span"] = 500.0
    else:
        cell["status"] = "hole"
        cell["rejection_evidence_refs"] = [
            "artifact://frontier/explicit-hole-evidence.json"
        ]
    _refresh_surface_digest(surface)
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[surface],
        queries=[
            {
                "query_id": f"q-{failure}",
                "surface_id": surface["surface_id"],
                "surface_version": "v1",
                "shape": query_shape,
                "domain": surface["domain"],
            }
        ],
    )

    rejected = _queries_by_id(diagnose_run_bundle(run))[f"q-{failure}"]

    assert rejected["status"] == "unknown"
    assert rejected["reason_code"] == reason_code
    assert rejected["cell_id"] == "fixture-matmul-family-a-cell-abc"
    assert rejected["domain_policy"] == surface["domain_policy"]
    assert rejected["effective_rate"] is None


def test_explicit_hole_cannot_be_masked_by_an_overlapping_retained_cell(
    tmp_path: Path,
) -> None:
    surface = _two_dimensional_surface()
    retained = surface["cells"][0]
    surface["cells"].append(
        {
            **retained,
            "cell_id": "fixture-matmul-family-a-explicit-hole",
            "status": "hole",
            "rejection_evidence_refs": [
                "artifact://frontier/overlapping-hole.json"
            ],
        }
    )
    _refresh_surface_digest(surface)
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[surface],
        queries=[
            {
                "query_id": "q-overlapping-hole",
                "surface_id": surface["surface_id"],
                "surface_version": "v1",
                "shape": {"m": 256, "n": 320},
                "domain": surface["domain"],
            }
        ],
    )

    rejected = _queries_by_id(diagnose_run_bundle(run))["q-overlapping-hole"]

    assert rejected["status"] == "unknown"
    assert rejected["reason_code"] == "explicit_domain_hole"
    assert rejected["cell_id"] == "fixture-matmul-family-a-explicit-hole"
    assert rejected["effective_rate"] is None

    human = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(run),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert human.returncode == 0, human.stderr
    assert "unknown (explicit_domain_hole)" in human.stdout
    assert "domain-policy=fixture-2d-domain/v1" in human.stdout
    assert "cell=fixture-matmul-family-a-explicit-hole" in human.stdout
    assert "artifact://frontier/overlapping-hole.json" in human.stdout


@pytest.mark.parametrize(
    ("field", "reason_code"),
    [
        ("alignment_validated", "alignment_regime_unvalidated"),
        ("working_set_validated", "working_set_regime_unvalidated"),
        (
            "kernel_dispatch_validated",
            "kernel_dispatch_regime_unvalidated",
        ),
    ],
)
def test_2d_missing_regime_validation_never_implies_true(
    tmp_path: Path,
    field: str,
    reason_code: str,
) -> None:
    surface = _two_dimensional_surface()
    domain = dict(surface["domain"])
    domain.pop(field)
    surface["domain"] = domain
    for anchor in surface["anchors"]:
        anchor["domain"] = domain
    _refresh_surface_digest(surface)
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[surface],
        queries=[
            {
                "query_id": f"q-missing-{field}",
                "surface_id": surface["surface_id"],
                "surface_version": "v1",
                "shape": {"m": 256, "n": 320},
                "domain": domain,
            }
        ],
    )

    rejected = _queries_by_id(diagnose_run_bundle(run))[f"q-missing-{field}"]

    assert rejected["status"] == "unknown"
    assert rejected["reason_code"] == reason_code
    assert rejected["effective_rate"] is None


@pytest.mark.parametrize(
    "failure",
    ["anchor-shape-inf", "anchor-rate-inf", "covariance-nan"],
)
def test_2d_nonfinite_numeric_evidence_fails_closed(
    tmp_path: Path,
    failure: str,
) -> None:
    surface = _two_dimensional_surface()
    if failure == "anchor-shape-inf":
        surface["anchors"][2]["shape"] = {"m": 128, "n": float("inf")}
    elif failure == "anchor-rate-inf":
        surface["anchors"][1]["effective_rate"] = float("inf")
    else:
        surface["uncertainty_policy"]["anchor_covariance"][0][0] = float(
            "nan"
        )
    _refresh_surface_digest(surface)
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[surface],
        queries=[
            {
                "query_id": f"q-{failure}",
                "surface_id": surface["surface_id"],
                "surface_version": "v1",
                "shape": {"m": 256, "n": 320},
                "domain": surface["domain"],
            }
        ],
    )

    rejected = _queries_by_id(diagnose_run_bundle(run))[f"q-{failure}"]

    assert rejected["status"] == "unknown"
    assert rejected["effective_rate"] is None


def test_2d_rate_interval_overflow_fails_closed(
    tmp_path: Path,
) -> None:
    surface = _two_dimensional_surface()
    for anchor in surface["anchors"]:
        anchor["effective_rate"] = 1.7e308
    surface["uncertainty_policy"]["anchor_covariance"] = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    surface["uncertainty_policy"][
        "instrumentation_standard_uncertainty_rate"
    ] = 1.0e308
    surface["cells"][0]["interpolation_standard_uncertainty_rate"] = 0.0
    _refresh_surface_digest(surface)
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[surface],
        queries=[
            {
                "query_id": "q-overflowing-rate-interval",
                "surface_id": surface["surface_id"],
                "surface_version": "v1",
                "shape": {"m": 256, "n": 320},
                "domain": surface["domain"],
            }
        ],
    )

    rejected = _queries_by_id(diagnose_run_bundle(run))[
        "q-overflowing-rate-interval"
    ]

    assert rejected["status"] == "unknown"
    assert rejected["reason_code"] == "invalid_nonfinite_rate_interval"
    assert rejected["effective_rate"] is None


def test_2d_oversized_integer_shape_fails_closed(
    tmp_path: Path,
) -> None:
    surface = _two_dimensional_surface()
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[surface],
        queries=[
            {
                "query_id": "q-oversized-integer-shape",
                "surface_id": surface["surface_id"],
                "surface_version": "v1",
                "shape": {"m": 10**1000, "n": 320},
                "domain": surface["domain"],
            }
        ],
    )

    rejected = _queries_by_id(diagnose_run_bundle(run))[
        "q-oversized-integer-shape"
    ]

    assert rejected["status"] == "unknown"
    assert rejected["reason_code"] == "invalid_query_shape"
    assert rejected["effective_rate"] is None


@pytest.mark.parametrize(
    ("validated_field", "reason_code"),
    [
        ("alignment_validated", "alignment_regime_unvalidated"),
        ("working_set_validated", "working_set_regime_unvalidated"),
        (
            "kernel_dispatch_validated",
            "kernel_dispatch_regime_unvalidated",
        ),
    ],
)
def test_2d_unvalidated_regime_seam_fails_closed_with_specific_reason(
    tmp_path: Path,
    validated_field: str,
    reason_code: str,
) -> None:
    surface = _two_dimensional_surface()
    domain = {**surface["domain"], validated_field: False}
    surface["domain"] = domain
    for anchor in surface["anchors"]:
        anchor["domain"] = domain
    _refresh_surface_digest(surface)
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[surface],
        queries=[
            {
                "query_id": f"q-{validated_field}",
                "surface_id": surface["surface_id"],
                "surface_version": "v1",
                "shape": {"m": 256, "n": 320},
                "domain": domain,
            }
        ],
    )

    rejected = _queries_by_id(diagnose_run_bundle(run))[
        f"q-{validated_field}"
    ]

    assert rejected["status"] == "unknown"
    assert rejected["reason_code"] == reason_code
    assert rejected["effective_rate"] is None


@pytest.mark.parametrize("failure", ["missing-surface-family", "mixed-anchor"])
def test_2d_algorithm_family_is_a_required_facet_identity(
    tmp_path: Path,
    failure: str,
) -> None:
    surface = _two_dimensional_surface()
    if failure == "missing-surface-family":
        surface.pop("algorithm_family")
    else:
        surface["anchors"][1]["algorithm_family"] = "unrelated-algorithm"
    _refresh_surface_digest(surface)
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[surface],
        queries=[
            {
                "query_id": f"q-{failure}",
                "surface_id": surface["surface_id"],
                "surface_version": "v1",
                "shape": {"m": 256, "n": 320},
                "domain": surface["domain"],
            }
        ],
    )

    rejected = _queries_by_id(diagnose_run_bundle(run))[f"q-{failure}"]

    assert rejected["status"] == "unknown"
    assert rejected["reason_code"] == "incomplete_candidate_family_facet"
    assert rejected["effective_rate"] is None


def test_2d_surface_lineage_cannot_change_algorithm_family(
    tmp_path: Path,
) -> None:
    v1 = _two_dimensional_surface()
    v2 = _two_dimensional_grid_surface(version="v2", previous_version="v1")
    v2["algorithm_family"] = "silent-replacement-algorithm"
    _refresh_surface_digest(v2)
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[v1, v2],
        queries=[
            {
                "query_id": "q-algorithm-lineage",
                "surface_id": v1["surface_id"],
                "surface_version": "v2",
                "shape": {"m": 400, "n": 400},
                "domain": v1["domain"],
            }
        ],
    )

    result = diagnose_run_bundle(run)

    assert [surface["version"] for surface in result["capability_surfaces"]] == [
        "v1"
    ]
    rejected = _queries_by_id(result)["q-algorithm-lineage"]
    assert rejected["reason_code"] == "surface_version_not_found"


def test_candidate_envelope_selects_winner_only_on_common_retained_support(
    tmp_path: Path,
) -> None:
    family_a = _two_dimensional_surface()
    family_b = _two_dimensional_surface(
        surface_id="surface://fixture/matmul/2d/family-b",
        candidate_family="fixture-matmul-family-b",
        algorithm_family="streamed-matmul",
    )
    for anchor, rate in zip(
        family_b["anchors"],
        (1_100_000_000_000.0, 2_000_000_000_000.0, 1_600_000_000_000.0),
        strict=True,
    ):
        anchor["effective_rate"] = rate
    _refresh_surface_digest(family_b)
    envelope = _candidate_envelope([family_a, family_b])
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[family_a, family_b],
        candidate_envelopes=[envelope],
        queries=[
            {
                "query_id": "q-common-support-envelope",
                "envelope_id": envelope["envelope_id"],
                "envelope_version": "v1",
                "shape": {"m": 256, "n": 320},
                "domain": family_a["domain"],
            }
        ],
    )

    result = diagnose_run_bundle(run)
    query = _queries_by_id(result)["q-common-support-envelope"]

    assert [
        (surface["candidate_family"], surface["algorithm_family"])
        for surface in result["capability_surfaces"]
    ] == [
        ("fixture-matmul-family-a", "blocked-matmul"),
        ("fixture-matmul-family-b", "streamed-matmul"),
    ]
    assert result["candidate_envelopes"] == [
        {
            "envelope_id": "envelope://fixture/matmul/2d",
            "version": "v1",
            "input_digest": envelope["input_digest"],
            "cohort_id": "fixture-m4-cohort-v1",
            "domain": family_a["domain"],
            "domain_policy": family_a["domain_policy"],
            "candidate_families": [
                "fixture-matmul-family-a",
                "fixture-matmul-family-b",
            ],
            "algorithm_families": ["blocked-matmul", "streamed-matmul"],
            "support_policy": envelope["support_policy"],
        }
    ]
    assert query["status"] == "interpolated"
    assert query["reason_code"] is None
    assert query["envelope"] == {
        "envelope_id": "envelope://fixture/matmul/2d",
        "version": "v1",
        "input_digest": envelope["input_digest"],
    }
    assert query["candidate_families"] == [
        "fixture-matmul-family-a",
        "fixture-matmul-family-b",
    ]
    assert query["selected_candidate_family"] == "fixture-matmul-family-b"
    assert query["selected_algorithm_family"] == "streamed-matmul"
    assert query["effective_rate"] == pytest.approx(
        {"value": 1_650_000_000_000.0, "unit": "FLOP/s"}
    )
    assert query["cell_id"] == "fixture-matmul-family-b-cell-abc"
    assert query["weights"] == pytest.approx([1 / 6, 1 / 3, 1 / 2])
    assert [facet["status"] for facet in query["candidate_facets"]] == [
        "interpolated",
        "interpolated",
    ]
    assert query["support_policy"] == envelope["support_policy"]

    human = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(run),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert human.returncode == 0, human.stderr
    assert (
        "Capability Envelope q-common-support-envelope "
        "[envelope://fixture/matmul/2d@v1]: interpolated"
    ) in human.stdout
    assert (
        "candidates=fixture-matmul-family-a/blocked-matmul,"
        "fixture-matmul-family-b/streamed-matmul"
    ) in human.stdout
    assert "winner=fixture-matmul-family-b/streamed-matmul" in human.stdout
    assert "domain-policy=fixture-2d-domain/v1" in human.stdout
    assert "cell=fixture-matmul-family-b-cell-abc" in human.stdout
    assert (
        "anchors=fixture-matmul-family-b-anchor-a,"
        "fixture-matmul-family-b-anchor-b,"
        "fixture-matmul-family-b-anchor-c"
    ) in human.stdout
    assert "weights=[0.16666666666666666,0.3333333333333333,0.5]" in (
        human.stdout
    )
    assert '"interpolation_standard_rate"' in human.stdout


@pytest.mark.parametrize("mismatch", ["coordinate", "work-formula"])
def test_candidate_envelope_requires_comparable_facet_semantics(
    tmp_path: Path,
    mismatch: str,
) -> None:
    family_a = _two_dimensional_surface()
    family_b = _two_dimensional_surface(
        surface_id="surface://fixture/matmul/2d/family-b",
        candidate_family="fixture-matmul-family-b",
        algorithm_family="streamed-matmul",
    )
    if mismatch == "coordinate":
        family_b["coordinate"] = {
            **family_b["coordinate"],
            "transform_version": "incomparable-v2",
        }
    else:
        family_b["work_formula"] = {
            **family_b["work_formula"],
            "fixed_k": 512,
        }
    _refresh_surface_digest(family_b)
    envelope = _candidate_envelope([family_a, family_b])
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[family_a, family_b],
        candidate_envelopes=[envelope],
        queries=[
            {
                "query_id": f"q-incomparable-{mismatch}",
                "envelope_id": envelope["envelope_id"],
                "envelope_version": "v1",
                "shape": {"m": 256, "n": 320},
                "domain": family_a["domain"],
            }
        ],
    )

    result = diagnose_run_bundle(run)

    assert result["candidate_envelopes"] == []
    rejected = _queries_by_id(result)[f"q-incomparable-{mismatch}"]
    assert rejected["status"] == "unknown"
    assert rejected["reason_code"] == "candidate_envelope_version_not_found"


def test_candidate_support_disappearance_blocks_unvalidated_envelope_switch(
    tmp_path: Path,
) -> None:
    family_a = _two_dimensional_surface()
    family_b = _two_dimensional_surface(
        surface_id="surface://fixture/matmul/2d/family-b",
        candidate_family="fixture-matmul-family-b",
        algorithm_family="streamed-matmul",
    )
    family_b["cells"][0]["status"] = "candidate_support_boundary"
    family_b["cells"][0]["support_seam_id"] = "seam-family-b-to-a"
    family_b["cells"][0]["rejection_evidence_refs"] = [
        "artifact://frontier/candidate-support-boundary.json"
    ]
    _refresh_surface_digest(family_b)
    envelope = _candidate_envelope([family_a, family_b])
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[family_a, family_b],
        candidate_envelopes=[envelope],
        queries=[
            {
                "query_id": "q-unvalidated-support-boundary",
                "envelope_id": envelope["envelope_id"],
                "envelope_version": "v1",
                "shape": {"m": 256, "n": 320},
                "domain": family_a["domain"],
            }
        ],
    )

    query = _queries_by_id(diagnose_run_bundle(run))[
        "q-unvalidated-support-boundary"
    ]

    assert query["status"] == "unknown"
    assert query["reason_code"] == "candidate_domain_boundary_unvalidated"
    assert query["selected_candidate_family"] is None
    assert query["effective_rate"] is None
    assert query["candidate_facets"][1]["status"] == "unknown"
    assert query["candidate_facets"][1]["reason_code"] == (
        "candidate_domain_boundary_unvalidated"
    )
    assert query["candidate_facets"][1]["cell_id"] == (
        "fixture-matmul-family-b-cell-abc"
    )
    assert query["candidate_facets"][1]["support_seam_id"] == (
        "seam-family-b-to-a"
    )

    human = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(run),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert human.returncode == 0, human.stderr
    assert "unknown (candidate_domain_boundary_unvalidated)" in human.stdout
    assert "fixture-matmul-family-a/blocked-matmul:interpolated" in human.stdout
    assert (
        "fixture-matmul-family-b/streamed-matmul:unknown"
        "(candidate_domain_boundary_unvalidated)"
    ) in human.stdout
    assert "seam=seam-family-b-to-a" in human.stdout
    assert "artifact://frontier/candidate-support-boundary.json" in human.stdout


def test_independently_validated_candidate_seam_allows_explicit_switch(
    tmp_path: Path,
) -> None:
    family_a = _two_dimensional_surface()
    family_b = _two_dimensional_surface(
        surface_id="surface://fixture/matmul/2d/family-b",
        candidate_family="fixture-matmul-family-b",
        algorithm_family="streamed-matmul",
    )
    family_b["cells"][0]["status"] = "candidate_support_boundary"
    family_b["cells"][0]["support_seam_id"] = "seam-family-b-to-a"
    family_b["cells"][0]["rejection_evidence_refs"] = [
        "artifact://frontier/candidate-support-boundary.json"
    ]
    _refresh_surface_digest(family_b)
    validated_seam = {
        "seam_id": "seam-family-b-to-a",
        "unsupported_candidate_family": "fixture-matmul-family-b",
        "validation_version": "v1",
        "evidence_ref": "artifact://frontier/validated-family-b-to-a.json",
    }
    envelope = _candidate_envelope(
        [family_a, family_b], validated_seams=[validated_seam]
    )
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[family_a, family_b],
        candidate_envelopes=[envelope],
        queries=[
            {
                "query_id": "q-validated-support-boundary",
                "envelope_id": envelope["envelope_id"],
                "envelope_version": "v1",
                "shape": {"m": 256, "n": 320},
                "domain": family_a["domain"],
            }
        ],
    )

    query = _queries_by_id(diagnose_run_bundle(run))[
        "q-validated-support-boundary"
    ]

    assert query["status"] == "interpolated"
    assert query["reason_code"] is None
    assert query["selected_candidate_family"] == "fixture-matmul-family-a"
    assert query["effective_rate"]["value"] == pytest.approx(
        1_550_000_000_000.0
    )
    assert query["candidate_facets"][1]["reason_code"] == (
        "candidate_domain_boundary_unvalidated"
    )
    assert query["support_transitions"] == [validated_seam]
    assert validated_seam["evidence_ref"] in query["evidence_refs"]


def test_candidate_support_boundary_property_holds_across_removed_cell(
    tmp_path: Path,
) -> None:
    family_a = _two_dimensional_surface()
    family_b = _two_dimensional_surface(
        surface_id="surface://fixture/matmul/2d/family-b",
        candidate_family="fixture-matmul-family-b",
        algorithm_family="streamed-matmul",
    )
    family_b["cells"][0]["status"] = "candidate_support_boundary"
    family_b["cells"][0]["support_seam_id"] = "seam-family-b-to-a"
    _refresh_surface_digest(family_b)
    envelope = _candidate_envelope([family_a, family_b])
    shapes = [(200, 200), (256, 320), (320, 256)]
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[family_a, family_b],
        candidate_envelopes=[envelope],
        queries=[
            {
                "query_id": f"q-boundary-{m}-{n}",
                "envelope_id": envelope["envelope_id"],
                "envelope_version": "v1",
                "shape": {"m": m, "n": n},
                "domain": family_a["domain"],
            }
            for m, n in shapes
        ],
    )

    results = _queries_by_id(diagnose_run_bundle(run))

    for m, n in shapes:
        query = results[f"q-boundary-{m}-{n}"]
        assert query["status"] == "unknown"
        assert query["reason_code"] == (
            "candidate_domain_boundary_unvalidated"
        )
        assert query["selected_candidate_family"] is None
        assert query["effective_rate"] is None


def test_v1_exact_knots_and_interpolation_share_one_provenance_path(
    tmp_path: Path,
) -> None:
    result = diagnose_run_bundle(_write_surface_bundle(tmp_path))

    queries = _queries_by_id(result)
    assert queries["q128-v1"]["status"] == "exact_anchor"
    assert queries["q201-v1"]["status"] == "interpolated"
    assert queries["q512-v1"]["status"] == "exact_anchor"
    assert queries["q201-v1"]["effective_rate"] == pytest.approx(
        {"value": 1_314_062_500_000.0, "unit": "FLOP/s"}
    )
    assert queries["q201-v1"]["work_rate_latency"] == pytest.approx(
        {
            "declared_work": 16_241_202.0,
            "work_unit": "FLOP",
            "value_ns": 12_359.535410225923,
        }
    )
    assert queries["q201-v1"]["weights"] == pytest.approx(
        [0.8098958333333334, 0.19010416666666666]
    )
    assert queries["q128-v1"]["weights"] == pytest.approx([1.0, 0.0])
    assert queries["q512-v1"]["weights"] == pytest.approx([0.0, 1.0])
    for query in queries.values():
        assert query["surface"] == {
            "surface_id": "surface://fixture/matmul/1d",
            "version": "v1",
            "input_digest": result["capability_surfaces"][0]["input_digest"],
        }
        assert query["cohort_id"] == "fixture-m4-cohort-v1"
        assert query["domain"]["alignment_regime"] == (
            "all-query-shapes-validated"
        )
        assert query["candidate_families"] == ["fixture-matmul-family"]
        assert query["selected_candidate_family"] == "fixture-matmul-family"
        assert query["cell_id"] == "cell-128-512"
        assert [anchor["anchor_id"] for anchor in query["anchors"]] == [
            "anchor-128",
            "anchor-512",
        ]
        assert query["reason_code"] is None
        assert query["uncertainty"]["components"] == {
            "anchor_standard_rate": query["uncertainty"]["components"][
                "anchor_standard_rate"
            ],
            "interpolation_standard_rate": (
                0.0 if query["status"] == "exact_anchor" else 40_000_000_000.0
            ),
            "instrumentation_standard_rate": 10_000_000_000.0,
        }


def test_qualified_201_anchor_creates_v2_without_rewriting_v1(
    tmp_path: Path,
) -> None:
    v1 = _surface_version("v1", previous_version=None)
    v2_template = _surface_version(
        "v2", previous_version="v1", include_201_anchor=True
    )
    update = {
        "update_id": "surface-update-add-anchor-201",
        "surface_id": "surface://fixture/matmul/1d",
        "base_version": "v1",
        "new_version": "v2",
        "anchor": v2_template["anchors"][1],
        "cells": v2_template["cells"],
        "uncertainty_policy": v2_template["uncertainty_policy"],
        "evidence_refs": ["artifact://frontier/add-anchor-201.json"],
    }
    domain = v1["domain"]
    run = _write_surface_bundle(
        tmp_path,
        surfaces=[v1],
        surface_updates=[update],
        queries=[
            {
                "query_id": "q201-v1",
                "surface_id": "surface://fixture/matmul/1d",
                "surface_version": "v1",
                "shape": {"s": 201},
                "domain": domain,
            },
            {
                "query_id": "q201-v2",
                "surface_id": "surface://fixture/matmul/1d",
                "surface_version": "v2",
                "shape": {"s": 201},
                "domain": domain,
            },
        ],
    )

    first = diagnose_run_bundle(run)
    replay = diagnose_run_bundle(run)

    assert replay == first
    queries = _queries_by_id(first)
    assert queries["q201-v1"]["status"] == "interpolated"
    assert queries["q201-v1"]["effective_rate"]["value"] == pytest.approx(
        1_314_062_500_000.0
    )
    assert queries["q201-v2"]["status"] == "exact_anchor"
    assert queries["q201-v2"]["effective_rate"]["value"] == pytest.approx(
        1_280_000_000_000.0
    )
    assert queries["q201-v1"]["surface"]["input_digest"] != queries[
        "q201-v2"
    ]["surface"]["input_digest"]
    surfaces = {
        surface["version"]: surface for surface in first["capability_surfaces"]
    }
    assert surfaces["v1"]["anchor_ids"] == ["anchor-128", "anchor-512"]
    assert surfaces["v2"]["anchor_ids"] == [
        "anchor-128",
        "anchor-201",
        "anchor-512",
    ]
    assert surfaces["v2"]["transition"] == {
        "previous_version": "v1",
        "previous_input_digest": surfaces["v1"]["input_digest"],
        "added_anchor_ids": ["anchor-201"],
        "removed_anchor_ids": [],
    }


def test_duplicate_surface_identity_is_rejected_deterministically(
    tmp_path: Path,
) -> None:
    first = _surface_version("v1", previous_version=None)
    conflict = _surface_version("v1", previous_version=None)
    conflict["anchors"][0]["effective_rate"] = 1_100_000_000_000.0
    _refresh_surface_digest(conflict)

    with pytest.raises(
        DiagnosticBundleIntegrityError,
        match="duplicate capability surface version",
    ):
        diagnose_run_bundle(
            _write_surface_bundle(
                tmp_path,
                surfaces=[first, conflict],
                queries=[],
            )
        )


def test_orphan_surface_cannot_create_a_replayable_descendant(
    tmp_path: Path,
) -> None:
    orphan = _surface_version("v1", previous_version="missing-v0")
    _refresh_surface_digest(orphan)
    v2_template = _surface_version(
        "v2", previous_version="v1", include_201_anchor=True
    )
    update = {
        "update_id": "orphan-update-add-anchor-201",
        "surface_id": "surface://fixture/matmul/1d",
        "base_version": "v1",
        "new_version": "v2",
        "anchor": v2_template["anchors"][1],
        "cells": v2_template["cells"],
        "uncertainty_policy": v2_template["uncertainty_policy"],
        "evidence_refs": ["artifact://frontier/orphan-update.json"],
    }
    domain = orphan["domain"]
    queries = [
        {
            "query_id": f"q201-{version}",
            "surface_id": "surface://fixture/matmul/1d",
            "surface_version": version,
            "shape": {"s": 201},
            "domain": domain,
        }
        for version in ("v1", "v2")
    ]

    result = diagnose_run_bundle(
        _write_surface_bundle(
            tmp_path,
            surfaces=[orphan],
            surface_updates=[update],
            queries=queries,
        )
    )

    assert result["capability_surfaces"] == []
    rejected = _queries_by_id(result)
    assert rejected["q201-v1"]["reason_code"] == "surface_version_not_found"
    assert rejected["q201-v2"]["reason_code"] == "surface_version_not_found"


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        ("outside-domain", "outside_validated_domain"),
        ("alignment-mismatch", "alignment_regime_unvalidated"),
        ("alignment-unvalidated", "alignment_regime_unvalidated"),
        ("regime-unvalidated", "shape_regime_unvalidated"),
        (
            "missing-combination-policy",
            "missing_uncertainty_combination_policy",
        ),
        (
            "missing-target-coverage",
            "missing_uncertainty_combination_policy",
        ),
        (
            "missing-calibration-evidence",
            "insufficient_uncertainty_evidence",
        ),
        (
            "missing-cell-calibration",
            "insufficient_uncertainty_evidence",
        ),
    ],
)
def test_surface_query_fails_closed_without_fallback(
    tmp_path: Path, failure: str, reason_code: str
) -> None:
    surface = _surface_version("v1", previous_version=None)
    domain = dict(surface["domain"])
    query = {
        "query_id": f"q-{failure}",
        "surface_id": "surface://fixture/matmul/1d",
        "surface_version": "v1",
        "shape": {"s": 201},
        "domain": domain,
    }
    if failure == "outside-domain":
        query["shape"] = {"s": 513}
        surface["fallbacks"] = {
            "global_p80_rate": 1_500_000_000_000.0,
            "nearest_neighbor": True,
        }
    elif failure == "alignment-mismatch":
        query["domain"] = {**domain, "alignment_regime": "unvalidated"}
    elif failure == "alignment-unvalidated":
        surface["domain"] = {**domain, "alignment_validated": False}
        query["domain"] = surface["domain"]
        for anchor in surface["anchors"]:
            anchor["domain"] = surface["domain"]
    elif failure == "regime-unvalidated":
        surface["domain"] = {**domain, "regime_validated": False}
        query["domain"] = surface["domain"]
        for anchor in surface["anchors"]:
            anchor["domain"] = surface["domain"]
    elif failure == "missing-combination-policy":
        surface.pop("uncertainty_policy")
    elif failure == "missing-target-coverage":
        surface["uncertainty_policy"].pop("target_coverage")
    elif failure == "missing-calibration-evidence":
        surface["uncertainty_policy"]["calibration_evidence_refs"] = []
    elif failure == "missing-cell-calibration":
        surface["cells"][0]["confirmation_evidence_refs"] = []
    _refresh_surface_digest(surface)

    result = diagnose_run_bundle(
        _write_surface_bundle(tmp_path, surfaces=[surface], queries=[query])
    )

    rejected = _queries_by_id(result)[f"q-{failure}"]
    assert rejected["status"] == "unknown"
    assert rejected["reason_code"] == reason_code
    assert rejected["effective_rate"] is None
    assert rejected["work_rate_latency"] is None
    assert rejected["selected_candidate_family"] is None


def test_surface_input_digest_rejects_summary_tampering(tmp_path: Path) -> None:
    run = _write_surface_bundle(tmp_path)
    evidence_path = run / "diagnostic/evidence.json"
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    document["capability_surfaces"][0]["input_digest"] = "0" * 64
    document["digests"]["evidence_sha256"] = _canonical_digest(
        {
            "capability_surfaces": document["capability_surfaces"],
            "surface_queries": document["surface_queries"],
        }
    )
    evidence_digest = _write_json(evidence_path, document)
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["sha256"] = evidence_digest
    _write_json(manifest_path, manifest)

    with pytest.raises(
        DiagnosticBundleIntegrityError,
        match="capability surface input digest mismatch",
    ):
        diagnose_run_bundle(run)

    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(run),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "capability surface input digest mismatch" in rejected.stderr


def test_cli_and_report_project_the_same_surface_query(tmp_path: Path) -> None:
    run = _write_surface_bundle(tmp_path)
    direct = diagnose_run_bundle(run)

    machine = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(run),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    human = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(run),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert machine.returncode == 0, machine.stderr
    assert json.loads(machine.stdout) == direct
    assert human.returncode == 0, human.stderr
    assert (
        "Capability Surface q201-v1 "
        "[surface://fixture/matmul/1d@v1]: interpolated"
    ) in human.stdout
    assert "cohort=fixture-m4-cohort-v1" in human.stdout
    assert 'domain={"alignment_regime":"all-query-shapes-validated"' in (
        human.stdout
    )
    assert "candidate-family=fixture-matmul-family" in human.stdout
    assert "rate=1.314062500 TFLOP/s" in human.stdout
    assert "anchors=anchor-128,anchor-512" in human.stdout
    assert "weights=[0.8098958333333334,0.19010416666666666]" in human.stdout
    assert '"anchor_standard_rate"' in human.stdout
    assert '"interpolation_standard_rate"' in human.stdout
    assert '"instrumentation_standard_rate"' in human.stdout


def test_retained_v1_cell_is_continuous_between_exact_knots(
    tmp_path: Path,
) -> None:
    surface = _surface_version("v1", previous_version=None)
    domain = surface["domain"]
    shapes = [128, 200, 201, 202, 512]
    queries = [
        {
            "query_id": f"q{shape}-v1",
            "surface_id": "surface://fixture/matmul/1d",
            "surface_version": "v1",
            "shape": {"s": shape},
            "domain": domain,
        }
        for shape in shapes
    ]

    result = diagnose_run_bundle(
        _write_surface_bundle(
            tmp_path, surfaces=[surface], queries=queries
        )
    )

    by_id = _queries_by_id(result)
    rates = {
        shape: by_id[f"q{shape}-v1"]["effective_rate"]["value"]
        for shape in shapes
    }
    assert rates[128] == pytest.approx(1_200_000_000_000.0)
    assert rates[512] == pytest.approx(1_800_000_000_000.0)
    assert rates[201] == pytest.approx(1_314_062_500_000.0)
    assert rates[201] - rates[200] == pytest.approx(1_562_500_000.0)
    assert rates[202] - rates[201] == pytest.approx(1_562_500_000.0)
