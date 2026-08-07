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


def _write_surface_bundle(
    tmp_path: Path,
    *,
    surfaces: list[dict[str, object]] | None = None,
    queries: list[dict[str, object]] | None = None,
    surface_updates: list[dict[str, object]] | None = None,
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
