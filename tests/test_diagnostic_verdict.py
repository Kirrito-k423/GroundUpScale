from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from groundupscale.diagnostics import (
    diagnose_run_bundle,
    render_diagnostic_report,
)
from _diagnostic_test_support import (
    canonical_digest,
    write_json,
)


def _write_bundle(
    tmp_path: Path,
    *,
    diagnostic_items: list[dict[str, object]],
    probes: list[dict[str, object]] | None = None,
    verdict_policy: dict[str, object] | None = None,
    execution_domain_override: dict[str, object] | None = None,
    implementation_source_identities: dict[str, str] | None = None,
    trigger_policy: dict[str, object] | None = None,
) -> Path:
    run = tmp_path / "issue-21-diagnostic"
    probe_contract = (
        probes[0].get("locked_contract")
        if isinstance(probes, list)
        and probes
        and isinstance(probes[0], dict)
        else None
    )
    cohort_identity = (
        probe_contract.get("cohort_identity")
        if isinstance(probe_contract, dict)
        and isinstance(probe_contract.get("cohort_identity"), dict)
        else {}
    )
    execution_domain = (
        probe_contract.get("execution_domain")
        if isinstance(probe_contract, dict)
        and isinstance(probe_contract.get("execution_domain"), dict)
        else {
            "shape": {"m": 257, "k": 257, "n": 257},
            "dtype": "float32",
            "layout": "row-major-contiguous",
            "alignment_bytes": 16,
            "threads": 1,
            "execution_mode": "eager",
        }
    )
    if execution_domain_override is not None:
        execution_domain = execution_domain_override
    inputs = {
        "resolved_configuration": {
            "analysis_plan": "issue-21-diagnostic",
            "benchmark_case": "independent-top10",
        },
        "resolved_ir": {
            "semantic_node": "semantic/workload/main",
            "operation": "MatMul",
        },
        "hardware": {
            "device": cohort_identity.get("device", "Apple M4 CPU"),
            "partition": cohort_identity.get("partition", "host"),
            "topology": cohort_identity.get("topology", "single-socket"),
            "software": cohort_identity.get("software", "torch-2.13-cpu"),
            **{
                key: cohort_identity[key]
                for key in (
                    "os",
                    "kernel",
                    "driver",
                    "firmware",
                    "runtime",
                    "framework",
                    "compiler",
                    "operator_library",
                    "communication_library",
                    "power_clock",
                )
                if key in cohort_identity
            },
        },
        "cohort_id": "apple-m4-cpu-darwin-arm64-torch2.13-baseline-v1",
        "execution_domain": execution_domain,
    }
    evidence = {
        "diagnostic_trigger_input": {
            "policy": trigger_policy
            or {
                "policy_id": "diagnostic-trigger",
                "version": "v1",
                "scope": "exact-shape-performance-diagnosis",
                "change_reason": "issue-21 normative trigger",
                "revalidation": "on uncertainty or materiality policy change",
            },
            "e2e_observation_ns": 1_000,
            "items": diagnostic_items,
        }
    }
    if probes is not None:
        evidence["shape_disambiguation_probes"] = probes
    if verdict_policy is not None:
        evidence["verdict_policy"] = verdict_policy
    document = {
        "schema": "groundupscale.dev/diagnostic-evidence/v1alpha1",
        **inputs,
        **evidence,
        "digests": {
            "input_sha256": canonical_digest(inputs),
            "evidence_sha256": canonical_digest(evidence),
        },
    }
    evidence_path = run / "diagnostic/evidence.json"
    artifact_digest = write_json(evidence_path, document)
    supporting_refs: set[str] = set()

    def collect_artifact_refs(value: object) -> None:
        if isinstance(value, dict):
            for item in value.values():
                collect_artifact_refs(item)
        elif isinstance(value, list):
            for item in value:
                collect_artifact_refs(item)
        elif isinstance(value, str) and value.startswith("artifact://"):
            supporting_refs.add(value)

    collect_artifact_refs(evidence)
    def implementation_source_identity(
        candidate: dict[str, object],
    ) -> str:
        return (implementation_source_identities or {}).get(
            candidate["candidate_id"],
            f"issue-6/a5a04f9c/{candidate['candidate_id']}",
        )

    family_manifests = {
        candidate["implementation_family"]["manifest_ref"]: {
            "schema": "groundupscale.dev/implementation-family-manifest/v1alpha1",
            "family_id": candidate["implementation_family"]["family_id"],
            "version": candidate["implementation_family"]["version"],
            "implementation_ref": candidate["implementation_family"][
                "implementation_ref"
            ],
            "implementation_sha256": candidate["implementation_family"][
                "implementation_sha256"
            ],
            "source_identity": implementation_source_identity(candidate),
        }
        for probe in probes or []
        for candidate in probe.get("candidates", [])
        if isinstance(candidate, dict)
        and isinstance(candidate.get("implementation_family"), dict)
    }
    implementation_artifacts = {
        candidate["implementation_family"]["implementation_ref"]: {
            "schema": "groundupscale.dev/candidate-implementation/v1alpha1",
            "source_identity": implementation_source_identity(candidate),
        }
        for probe in probes or []
        for candidate in probe.get("candidates", [])
        if isinstance(candidate, dict)
        and isinstance(candidate.get("implementation_family"), dict)
    }
    uncertainty_components: dict[str, dict[str, object]] = {}
    target_coverages: dict[str, dict[str, object]] = {}
    uncertainty_calibrations: dict[str, dict[str, object]] = {}

    def collect_uncertainty_artifacts(value: object) -> None:
        if isinstance(value, dict):
            if (
                set(("component_id", "standard_uncertainty_ns", "evidence_ref"))
                <= set(value)
                and isinstance(value["evidence_ref"], str)
            ):
                uncertainty_components[value["evidence_ref"]] = {
                    "schema": "groundupscale.dev/uncertainty-component/v1alpha1",
                    "component_id": value["component_id"],
                    "standard_uncertainty_ns": value[
                        "standard_uncertainty_ns"
                    ],
                }
            target_coverage = value.get("target_coverage")
            calibration = value.get("calibration")
            if (
                isinstance(target_coverage, dict)
                and isinstance(target_coverage.get("evidence_ref"), str)
                and isinstance(calibration, dict)
                and isinstance(calibration.get("evidence_ref"), str)
            ):
                coverage_payload = {
                    key: item
                    for key, item in target_coverage.items()
                    if key != "evidence_ref"
                }
                target_coverages[target_coverage["evidence_ref"]] = {
                    "schema": "groundupscale.dev/uncertainty-target-coverage/v1alpha1",
                    **coverage_payload,
                }
                uncertainty_calibrations[calibration["evidence_ref"]] = {
                    "schema": "groundupscale.dev/uncertainty-calibration/v1alpha1",
                    "policy_id": value.get("policy_id"),
                    "version": value.get("version"),
                    "target_coverage": coverage_payload,
                    "estimator": calibration.get("estimator"),
                    "records": calibration.get("records"),
                }
            for item in value.values():
                collect_uncertainty_artifacts(item)
        elif isinstance(value, list):
            for item in value:
                collect_uncertainty_artifacts(item)

    collect_uncertainty_artifacts(evidence)
    supporting_artifacts = []
    for index, artifact_ref in enumerate(sorted(supporting_refs), start=1):
        path = f"diagnostic/supporting/{index:03d}.json"
        is_family_manifest = artifact_ref in family_manifests
        if is_family_manifest:
            artifact_role = "implementation-family-manifest"
            artifact_schema = (
                "groundupscale.dev/implementation-family-manifest/v1alpha1"
            )
            artifact_payload = family_manifests[artifact_ref]
        elif artifact_ref in uncertainty_components:
            artifact_role = "uncertainty-component"
            artifact_schema = (
                "groundupscale.dev/uncertainty-component/v1alpha1"
            )
            artifact_payload = uncertainty_components[artifact_ref]
        elif artifact_ref in implementation_artifacts:
            artifact_role = "candidate-implementation"
            artifact_schema = (
                "groundupscale.dev/candidate-implementation/v1alpha1"
            )
            artifact_payload = implementation_artifacts[artifact_ref]
        elif artifact_ref in target_coverages:
            artifact_role = "uncertainty-target-coverage"
            artifact_schema = (
                "groundupscale.dev/uncertainty-target-coverage/v1alpha1"
            )
            artifact_payload = target_coverages[artifact_ref]
        elif artifact_ref in uncertainty_calibrations:
            artifact_role = "uncertainty-calibration"
            artifact_schema = (
                "groundupscale.dev/uncertainty-calibration/v1alpha1"
            )
            artifact_payload = uncertainty_calibrations[artifact_ref]
        else:
            artifact_role = "diagnostic-supporting-evidence"
            artifact_schema = "groundupscale.dev/test-evidence/v1alpha1"
            artifact_payload = {
                "schema": artifact_schema,
                "artifact_ref": artifact_ref,
            }
        digest = write_json(
            run / path,
            artifact_payload,
        )
        supporting_artifacts.append(
            {
                "role": artifact_role,
                "uri": artifact_ref,
                "path": path,
                "schema": artifact_schema,
                "media_type": "application/json",
                "sha256": digest,
                "produced_by": "groundupscale-test-fixture",
                "inputs": [],
            }
        )
    write_json(
        run / "run.manifest.json",
        {
            "schema": "groundupscale.dev/run-manifest/v1alpha1",
            "run_id": "issue-21-diagnostic",
            "status": "completed",
            "device": "cpu",
            "hardware_cohort": inputs["cohort_id"],
            "artifacts": [
                {
                    "role": "diagnostic-evidence",
                    "path": "diagnostic/evidence.json",
                    "schema": document["schema"],
                    "media_type": "application/json",
                    "sha256": artifact_digest,
                    "produced_by": "groundupscale-test-fixture",
                    "inputs": [],
                },
                *supporting_artifacts,
            ],
        },
    )
    return run


def test_trigger_uses_independent_top10_union_uncertainty_and_materiality(
    tmp_path: Path,
) -> None:
    shared = [
        {
            "stable_path": f"semantic/shared-{index:02d}",
            "predicted_ns": 112 - index,
            "observed_ns": 112 - index,
            "combined_uncertainty_ns": 1,
        }
        for index in range(1, 10)
    ]
    run = _write_bundle(
        tmp_path,
        diagnostic_items=[
            {
                "stable_path": "semantic/predicted-only",
                "predicted_ns": 120,
                "observed_ns": 30,
                "combined_uncertainty_ns": 10,
            },
            {
                "stable_path": "semantic/observed-only",
                "predicted_ns": 30,
                "observed_ns": 120,
                "combined_uncertainty_ns": 10,
            },
            *shared,
            {
                "stable_path": "semantic/absolute-gap",
                "predicted_ns": 0,
                "observed_ns": 101,
                "combined_uncertainty_ns": 10,
            },
            {
                "stable_path": "semantic/equal-e2e-threshold",
                "predicted_ns": 0,
                "observed_ns": 100,
                "combined_uncertainty_ns": 1,
            },
        ],
    )

    trigger = diagnose_run_bundle(run)["diagnostic_trigger"]

    assert [item["stable_path"] for item in trigger["predicted_top10"]] == [
        "semantic/predicted-only",
        *[f"semantic/shared-{index:02d}" for index in range(1, 10)],
    ]
    assert [item["stable_path"] for item in trigger["observed_top10"]] == [
        "semantic/observed-only",
        *[f"semantic/shared-{index:02d}" for index in range(1, 10)],
    ]
    assert {item["stable_path"] for item in trigger["top10_union"]} == {
        "semantic/predicted-only",
        "semantic/observed-only",
        *{f"semantic/shared-{index:02d}" for index in range(1, 10)},
    }
    assert [item["stable_path"] for item in trigger["triggered"]] == [
        "semantic/predicted-only",
        "semantic/observed-only",
        "semantic/absolute-gap",
    ]
    by_path = {item["stable_path"]: item for item in trigger["evaluated"]}
    assert by_path["semantic/absolute-gap"]["materiality"] == {
        "predicted_top10": False,
        "observed_top10": False,
        "gap_exceeds_e2e_tenth": True,
    }
    assert by_path["semantic/equal-e2e-threshold"]["triggered"] is False
    assert by_path["semantic/equal-e2e-threshold"]["reason_code"] == (
        "gap-not-material"
    )


def test_exact_shape_probe_locks_contract_and_excludes_incorrect_fast_candidate(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=fixture["shape_disambiguation_probes"],
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "complete"
    assert probe["locked_contract"] == fixture[
        "shape_disambiguation_probes"
    ][0]["locked_contract"]
    assert probe["evaluation_order"] == [
        "lock-exact-shape-contract",
        "validate-correctness",
        "select-best-of-correct",
    ]
    by_candidate = {
        item["candidate_id"]: item for item in probe["candidate_evaluations"]
    }
    assert by_candidate["truncated-negative-control"][
        "eligible_for_best_of_correct"
    ] is False
    assert by_candidate["truncated-negative-control"]["exclusion_reason"] == (
        "correctness-failed"
    )
    assert probe["best_of_correct"] == {
        "candidate_id": "batched-matmul",
        "aggregate_latency_ns": 330_418.89285714284,
        "session_ids": ["session-1", "session-2", "session-3"],
    }


def test_search_correctness_is_recomputed_instead_of_trusting_passed_flag(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    probes[0]["candidates"][2]["correctness"]["passed"] = True
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]
    negative = probe["candidate_evaluations"][2]

    assert negative["correctness"]["passed"] is False
    assert negative["eligible_for_best_of_correct"] is False


@pytest.mark.parametrize("policy_failure", ["missing", "candidate-override"])
def test_search_correctness_uses_locked_versioned_policy(
    tmp_path: Path,
    policy_failure: str,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    if policy_failure == "missing":
        del probes[0]["locked_contract"]["correctness_policy"]
        expected_reason = "invalid-shape-probe"
    else:
        probes[0]["candidates"][2]["correctness"]["tolerance"] = {
            "atol": 2,
            "rtol": 2,
        }
        expected_reason = "invalid-candidate-evidence"
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == expected_reason


def test_issue6_context_matmul_fixture_produces_implementation_headroom(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=fixture["shape_disambiguation_probes"],
        verdict_policy=fixture["verdict_policy"],
    )

    result = diagnose_run_bundle(run)
    verdict = result["performance_diagnosis_verdicts"][0]

    assert verdict["verdict"] == "implementation_headroom"
    assert verdict["status"] == "decided"
    assert verdict["metrics"]["speedup_fraction"] == pytest.approx(
        fixture["fixture_expectation"]["speedup_fraction"]
    )
    assert verdict["metrics"]["faster_in_every_session"] is True
    assert verdict["metrics"]["independent_session_count"] == 3
    assert verdict["metrics"]["independent_process_count"] == 3
    probe = result["shape_disambiguation_probes"][0]
    assert probe["measurement_lanes"]["diagnostic"][
        "timing_used_for_verdict"
    ] is False
    assert all(
        len(samples) == 12
        for candidate in probe["candidate_evaluations"]
        if candidate["role"] in {"target", "alternative"}
        for samples in candidate["raw_samples_ns"].values()
    )
    assert "minimum_speedup_fraction" not in result["verdict_policy"]
    assert {gate["gate_id"] for gate in verdict["gates"]["satisfied"]} >= {
        "diagnostic-trigger-met",
        "exact-shape-contract-locked",
        "correctness-before-best-of-correct",
        "same-hardware-validity-cohort",
        "eligible-probe-environment",
        "paired-baseline-diagnostic-lanes",
        "reproducible-faster-alternative",
    }
    assert verdict["gates"]["failed"] == []
    assert verdict["bundle_refs"] == [
        "run-bundle://issue-21-diagnostic",
        "artifact://issue-6/context-matmul-probe",
    ]
    assert verdict["counterexamples"] == [
        {
            "candidate_id": "truncated-negative-control",
            "reason_code": "correctness-failed",
            "evidence_refs": [
                "artifact://issue-6/truncated-correctness",
                "artifact://issue-6/truncated-session-1",
            ],
        }
    ]


def test_implementation_headroom_requires_distinct_alternative_source(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    target = probes[0]["candidates"][0]
    alternative = probes[0]["candidates"][1]
    alternative["implementation_family"] = deepcopy(
        target["implementation_family"]
    )
    shared_source = f"issue-6/a5a04f9c/{target['candidate_id']}"
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
        implementation_source_identities={
            target["candidate_id"]: shared_source,
            alternative["candidate_id"]: shared_source,
        },
    )

    result = diagnose_run_bundle(run)
    verdict = result["performance_diagnosis_verdicts"][0]

    assert result["shape_disambiguation_probes"][0]["status"] == "complete"
    assert verdict["verdict"] == "insufficient_evidence"
    assert {
        gate["gate_id"] for gate in verdict["gates"]["failed"]
    } == {"distinct-target-alternative-implementation"}


def test_issue6_257_cube_c1_and_neighbourhood_gap_is_insufficient_evidence(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=fixture["shape_disambiguation_probes"],
        verdict_policy=fixture["verdict_policy"],
    )

    result = diagnose_run_bundle(run)
    verdict = result["performance_diagnosis_verdicts"][0]

    assert verdict["verdict"] == "insufficient_evidence"
    assert verdict["surface_action"] == {
        "action": "preserve",
        "surface": {
            "surface_id": "m4-matmul-fp32",
            "version": "v1",
        },
        "reason_code": "insufficient-evidence-cannot-lower-surface",
    }
    assert "frontier_shift" not in {
        item["verdict"] for item in result["performance_diagnosis_verdicts"]
    }
    failed_gate_ids = {
        gate["gate_id"] for gate in verdict["gates"]["failed"]
    }
    assert failed_gate_ids >= {
        "reproducible-faster-alternative",
        "frontier-shift-independent-candidate-coverage",
        "frontier-shift-validated-neighbourhood",
        "frontier-shift-independent-holdout",
        "frontier-shift-same-hardware-validity-cohort",
        "frontier-shift-all-eligible-candidates-below-surface-band",
    }
    satisfied_gate_ids = {
        gate["gate_id"] for gate in verdict["gates"]["satisfied"]
    }
    assert satisfied_gate_ids >= {
        "frontier-shift-minimum-independent-sessions",
    }
    assert verdict["counterexamples"] == [
        {
            "counterexample_id": "257-cube-c1-neighbourhood-insufficient",
            "reason_codes": [
                "c2-or-c3-independent-candidate-families-missing",
                "neighbourhood-regime-not-validated",
            ],
            "evidence_refs": ["artifact://issue-6/257-cube-counterexample"],
        }
    ]
    assert verdict["gates"]["not_evaluated"] == [
        {
            "gate_id": "frontier-shift",
            "reason_code": "prerequisites-failed",
                "evidence_refs": [
                    "artifact://issue-6/257-frontier-gates",
                    "artifact://issue-6/257-holdout-missing",
                    "artifact://issue-6/257-neighbourhood-calibration",
                    "artifact://issue-6/257-neighbourhood-target-coverage",
                    "artifact://issue-6/257-surface-target-coverage",
                    "artifact://issue-6/257-surface-uncertainty-calibration",
                    "artifact://issue-6/257-surface-v1",
                    "artifact://issue-6/257-uncertainty-anchor",
                    "artifact://issue-6/257-uncertainty-instrumentation",
                    "artifact://issue-6/257-uncertainty-interpolation",
                    "artifact://issue-6/numpy-direct-manifest",
                    "artifact://issue-6/torch-direct-manifest",
                    "artifact://issue-6/numpy-direct-implementation",
                    "artifact://issue-6/torch-direct-implementation",
                ],
        },
        {
            "gate_id": "suspected-regression",
            "reason_code": "policy-undefined",
            "evidence_refs": [
                "artifact://issue-6/257-cube-probe",
                "artifact://issue-6/torch-correctness",
                "artifact://issue-6/torch-session-1",
                "artifact://issue-6/torch-session-2",
                "artifact://issue-6/torch-session-3",
                "artifact://issue-6/numpy-correctness",
                "artifact://issue-6/numpy-session-1",
                "artifact://issue-6/numpy-session-2",
                "artifact://issue-6/numpy-session-3",
            ],
        },
    ]


def test_report_projects_trigger_probe_verdict_gates_refs_and_counterexamples(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=fixture["shape_disambiguation_probes"],
        verdict_policy=fixture["verdict_policy"],
    )
    result = diagnose_run_bundle(run)

    assert result["verdict_vocabulary"] == [
        "frontier_shift",
        "implementation_headroom",
        "integration_overhead",
        "suspected_regression",
        "insufficient_evidence",
        "confirmed_bug",
    ]
    report = render_diagnostic_report(result)

    assert "Diagnostic Trigger [diagnostic-trigger/v1]" in report
    assert "predicted Top 10: semantic/context-matmul" in report
    assert "observed Top 10: semantic/context-matmul" in report
    assert "triggered: semantic/context-matmul" in report
    assert "Shape Disambiguation Probe probe-context-matmul: complete" in report
    assert 'semantic=bhqk,bkhd->bqhd Context MatMul' in report
    assert "correctness -> best-of-correct: batched-matmul" in report
    assert (
        "Performance Diagnosis Verdict semantic/context-matmul: "
        "implementation_headroom"
    ) in report
    assert "satisfied gates:" in report
    assert "failed gates: none" in report
    assert "not_evaluated gates: suspected-regression(policy-undefined)" in report
    assert (
        "bundle refs: run-bundle://issue-21-diagnostic, "
        "artifact://issue-6/context-matmul-probe"
    ) in report
    assert "counterexamples: truncated-negative-control(correctness-failed)" in (
        report
    )


def test_trigger_without_probe_evidence_requests_probe_and_fails_closed(
    tmp_path: Path,
) -> None:
    run = _write_bundle(
        tmp_path,
        diagnostic_items=[
            {
                "stable_path": "semantic/missing-probe",
                "predicted_ns": 100,
                "observed_ns": 200,
                "combined_uncertainty_ns": 1,
            }
        ],
        verdict_policy={
            "policy_id": "exact-shape-verdict",
            "version": "v1",
            "scope": "exact-shape-performance-diagnosis",
            "change_reason": "issue-21 normative verdict gates",
            "revalidation": "on verdict gate or evidence policy change",
            "minimum_independent_sessions": 3,
            "suspected_regression_gate": "undefined",
        },
    )

    result = diagnose_run_bundle(run)

    assert result["shape_disambiguation_probes"] == [
        {
            "probe_id": "probe-request:semantic/missing-probe",
            "stable_path": "semantic/missing-probe",
            "status": "requested",
            "reason_code": "exact-shape-probe-evidence-not-provided",
            "required_lock_fields": [
                "semantic",
                "shape",
                "dtype",
                "layout",
                "strides",
                "alignment_bytes",
                "threads",
                "execution_domain",
                "cohort_id",
                "cohort_identity",
                "environment",
                "correctness_policy",
                "candidate_ids",
                "completion_boundary",
                "measurement_lanes",
            ],
            "evidence_refs": [],
        }
    ]
    verdict = result["performance_diagnosis_verdicts"][0]
    assert verdict["verdict"] == "insufficient_evidence"
    assert verdict["gates"]["satisfied"] == [
        {
            "gate_id": "diagnostic-trigger-met",
            "evidence_refs": ["run-bundle://issue-21-diagnostic"],
        }
    ]
    assert verdict["gates"]["failed"] == [
        {
            "gate_id": "exact-shape-probe-complete",
            "reason_code": "exact-shape-probe-evidence-not-provided",
            "evidence_refs": [],
        }
    ]
    assert verdict["gates"]["not_evaluated"][-1] == {
        "gate_id": "suspected-regression",
        "reason_code": "policy-undefined",
        "evidence_refs": ["run-bundle://issue-21-diagnostic"],
    }


def test_headroom_requires_manifest_resolvable_digest_verified_artifacts(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=fixture["shape_disambiguation_probes"],
        verdict_policy=fixture["verdict_policy"],
    )
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        artifact
        for artifact in manifest["artifacts"]
        if artifact.get("uri") != "artifact://issue-6/context-matmul-probe"
    ]
    write_json(manifest_path, manifest)

    result = diagnose_run_bundle(run)

    assert result["shape_disambiguation_probes"][0]["status"] == (
        "insufficient_evidence"
    )
    assert result["shape_disambiguation_probes"][0]["reason_code"] == (
        "unresolved-probe-evidence-ref"
    )
    assert result["performance_diagnosis_verdicts"][0]["verdict"] == (
        "insufficient_evidence"
    )


@pytest.mark.parametrize("ref_kind", ["family-manifest", "correctness"])
def test_probe_rejects_non_artifact_evidence_references(
    tmp_path: Path,
    ref_kind: str,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    candidate = probes[0]["candidates"][0]
    if ref_kind == "family-manifest":
        candidate["implementation_family"]["manifest_ref"] = "fake-ref"
    else:
        candidate["correctness"]["evidence_ref"] = "fake-ref"
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-shape-probe"


def test_candidate_family_must_match_digest_verified_manifest_content(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=fixture["shape_disambiguation_probes"],
        verdict_policy=fixture["verdict_policy"],
    )
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    family_artifact = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact.get("uri")
        == "artifact://issue-6/legacy-einsum-manifest"
    )
    family_artifact["sha256"] = write_json(
        run / family_artifact["path"],
        {
            "schema": "groundupscale.dev/implementation-family-manifest/v1alpha1",
            "family_id": "forged-family",
            "version": "v9",
        },
    )
    write_json(manifest_path, manifest)

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-candidate-evidence"


def test_headroom_requires_locked_eligible_environment(tmp_path: Path) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    probes[0]["locked_contract"]["environment"]["eligible"] = False
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    result = diagnose_run_bundle(run)

    assert result["shape_disambiguation_probes"][0]["reason_code"] == (
        "ineligible-probe-environment"
    )
    assert result["performance_diagnosis_verdicts"][0]["verdict"] == (
        "insufficient_evidence"
    )


@pytest.mark.parametrize(
    ("verdict_policy", "reason_code"),
    [
        (None, "verdict-policy-missing"),
        (
            {
                "policy_id": "exact-shape-verdict",
                "version": "v1",
                "scope": "exact-shape-performance-diagnosis",
                "change_reason": "issue-21 normative verdict gates",
                "revalidation": "on verdict gate or evidence policy change",
                "minimum_independent_sessions": 1,
                "suspected_regression_gate": "undefined",
            },
            "verdict-policy-invalid",
        ),
        (
            {
                "policy_id": "exact-shape-verdict",
                "version": "v1",
                "minimum_independent_sessions": 0,
                "suspected_regression_gate": "invented-default",
            },
            "verdict-policy-invalid",
        ),
    ],
)
def test_missing_or_invalid_verdict_policy_fails_closed(
    tmp_path: Path,
    verdict_policy: dict[str, object] | None,
    reason_code: str,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=fixture["shape_disambiguation_probes"],
        verdict_policy=verdict_policy,
    )

    result = diagnose_run_bundle(run)

    assert result["verdict_policy"] == {
        "status": "unknown",
        "reason_code": reason_code,
    }
    verdict = result["performance_diagnosis_verdicts"][0]
    assert verdict["verdict"] == "insufficient_evidence"
    assert verdict["gates"]["failed"] == [
        {
            "gate_id": "verdict-policy-valid",
            "reason_code": reason_code,
            "evidence_refs": ["run-bundle://issue-21-diagnostic"],
        }
    ]


def test_target_only_probe_fails_closed_instead_of_dropping_verdict(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    probes[0]["candidates"] = [probes[0]["candidates"][0]]
    probes[0]["locked_contract"]["candidate_ids"] = ["torch-direct"]
    probes[0]["measurement_lanes"]["baseline"]["candidate_ids"] = [
        "torch-direct"
    ]
    probes[0]["measurement_lanes"]["diagnostic"]["candidate_ids"] = [
        "torch-direct"
    ]
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    verdict = diagnose_run_bundle(run)["performance_diagnosis_verdicts"][0]

    assert verdict["verdict"] == "insufficient_evidence"
    assert verdict["gates"]["failed"] == [
        {
            "gate_id": "correct-eligible-alternative-present",
            "reason_code": "no-correct-eligible-alternative",
            "evidence_refs": ["artifact://issue-6/257-cube-probe"],
        }
    ]


def test_probe_rejects_bundle_execution_domain_mismatch(tmp_path: Path) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    mismatched_domain = deepcopy(
        fixture["shape_disambiguation_probes"][0]["locked_contract"][
            "execution_domain"
        ]
    )
    mismatched_domain["threads"] = 1
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=fixture["shape_disambiguation_probes"],
        verdict_policy=fixture["verdict_policy"],
        execution_domain_override=mismatched_domain,
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-shape-probe"


def test_probe_rejects_incomplete_hardware_validity_cohort(tmp_path: Path) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    del probes[0]["locked_contract"]["cohort_identity"]["kernel"]
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-shape-probe"


@pytest.mark.parametrize(
    "invalid_identity",
    [
        "unknown",
        "system-firmware-unknown-version-recorded",
        "Accelerate",
        {"name": "Accelerate", "version": "latest1", "status": "resolved"},
        {"name": "unknown", "version": "1.0.0", "status": "resolved"},
    ],
)
def test_probe_rejects_unknown_or_unversioned_cohort_identity(
    tmp_path: Path,
    invalid_identity: object,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    probes[0]["locked_contract"]["cohort_identity"][
        "operator_library"
    ] = invalid_identity
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-shape-probe"


@pytest.mark.parametrize(
    "failure",
    ["os-not-applicable", "runtime-none", "dtype-not-applicable", "timer-na"],
)
def test_probe_requires_resolved_required_execution_identity(
    tmp_path: Path,
    failure: str,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    contract = probes[0]["locked_contract"]
    cohort = contract["cohort_identity"]
    if failure == "os-not-applicable":
        cohort["os"]["status"] = "not_applicable"
    elif failure == "runtime-none":
        cohort["runtime"]["name"] = "none"
    elif failure == "dtype-not-applicable":
        contract["dtype"] = "not_applicable"
        contract["execution_domain"]["dtype"] = "not_applicable"
        cohort["numeric_execution"]["dtype"] = "not_applicable"
        for lane in probes[0]["measurement_lanes"].values():
            lane["execution_domain"]["dtype"] = "not_applicable"
    else:
        cohort["timer_protocol"]["source"] = "n/a"
        for lane in probes[0]["measurement_lanes"].values():
            lane["timer_source"] = "n/a"
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-shape-probe"


@pytest.mark.parametrize(
    "invalid_path",
    ["unknown", "semantic/../matmul", "semantic/not_applicable"],
)
def test_stable_path_requires_canonical_resolved_hierarchy(
    tmp_path: Path,
    invalid_path: str,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    original_path = probes[0]["stable_path"]
    probes[0]["stable_path"] = invalid_path
    for lane in probes[0]["measurement_lanes"].values():
        lane["case"]["stable_path"] = invalid_path
    diagnostic_items = deepcopy(fixture["diagnostic_items"])
    for item in diagnostic_items:
        if item["stable_path"] == original_path:
            item["stable_path"] = invalid_path
    run = _write_bundle(
        tmp_path,
        diagnostic_items=diagnostic_items,
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    result = diagnose_run_bundle(run)

    assert not any(
        verdict.get("verdict") == "implementation_headroom"
        and verdict.get("status") == "decided"
        for verdict in result["performance_diagnosis_verdicts"]
    )


@pytest.mark.parametrize(
    ("identity_group", "identity_key", "invalid_value"),
    [
        (None, "device", "unknown"),
        (None, "device", "not_applicable"),
        (None, "device", "Apple/./M4"),
        (None, "partition", "unknown"),
        (None, "topology", "unknown"),
        (None, "software", "unknown"),
        ("power_clock", "power_policy", "latest"),
        ("power_clock", "clock_policy", "unknown"),
        ("execution_context", "affinity", "   "),
        ("communication", "status", "unknown"),
    ],
)
def test_probe_rejects_unknown_cohort_settings(
    tmp_path: Path,
    identity_group: str | None,
    identity_key: str,
    invalid_value: str,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    cohort = probes[0]["locked_contract"]["cohort_identity"]
    if identity_group is None:
        cohort[identity_key] = invalid_value
    else:
        cohort[identity_group][identity_key] = invalid_value
    if identity_group == "execution_context":
        probes[0]["locked_contract"]["execution_domain"][identity_key] = (
            invalid_value
        )
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-shape-probe"


@pytest.mark.parametrize(
    "identity_field",
    [
        "semantic",
        "dtype",
        "layout",
        "correctness-policy-id",
        "correctness-scope",
        "correctness-oracle",
    ],
)
def test_probe_rejects_unknown_exact_contract_identity(
    tmp_path: Path,
    identity_field: str,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    contract = probes[0]["locked_contract"]
    if identity_field.startswith("correctness-"):
        correctness_key = identity_field.removeprefix("correctness-").replace(
            "-", "_"
        )
        contract["correctness_policy"][correctness_key] = "unknown"
    else:
        contract[identity_field] = "unknown"
        if identity_field in {"dtype", "layout"}:
            contract["execution_domain"][identity_field] = "unknown"
            contract["cohort_identity"]["numeric_execution"][
                identity_field
            ] = "unknown"
        for lane in probes[0]["measurement_lanes"].values():
            if identity_field in lane["case"]:
                lane["case"][identity_field] = "unknown"
            if identity_field in lane["execution_domain"]:
                lane["execution_domain"][identity_field] = "unknown"
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-shape-probe"


@pytest.mark.parametrize("mismatch", ["session-lane", "lane-pair"])
def test_probe_requires_case_bound_paired_lanes_and_baseline_samples(
    tmp_path: Path,
    mismatch: str,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-context-matmul-headroom.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    if mismatch == "session-lane":
        probes[0]["candidates"][0]["sessions"][0]["lane_id"] = (
            "context-diagnostic"
        )
        expected_reason = "invalid-candidate-evidence"
    else:
        probes[0]["measurement_lanes"]["diagnostic"]["pair_id"] = (
            "other-pair"
        )
        expected_reason = "invalid-shape-probe"
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == expected_reason


def test_frontier_gates_are_derived_instead_of_trusting_authored_booleans(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    for candidate in probes[0]["candidates"]:
        candidate["sessions"] = candidate["sessions"][:2]
    frontier = probes[0]["frontier_shift_evidence"]
    frontier["minimum_sessions_met"] = True
    frontier["independent_holdout"] = True
    frontier["stable_neighbouring_anchors"] = True
    frontier["dense_local_shape_probe"] = True
    frontier["same_regime_refit"] = True
    frontier["candidate_coverage"] = "C3_ENUMERATED_POOL"
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    verdict = diagnose_run_bundle(run)["performance_diagnosis_verdicts"][0]
    failed = {gate["gate_id"] for gate in verdict["gates"]["failed"]}

    assert "frontier-shift-minimum-independent-sessions" in failed
    assert "frontier-shift-independent-holdout" in failed
    assert "frontier-shift-validated-neighbourhood" in failed
    assert "frontier-shift-independent-candidate-coverage" in failed


def test_frontier_candidate_families_require_distinct_implementation_digests(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    target_family = probes[0]["candidates"][0]["implementation_family"]
    alternative_family = probes[0]["candidates"][1]["implementation_family"]
    alternative_family["family_id"] = "forged-independent-family"
    alternative_family["manifest_ref"] = (
        "artifact://issue-21/forged-independent-family-manifest"
    )
    alternative_family["implementation_sha256"] = target_family[
        "implementation_sha256"
    ]
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    result = diagnose_run_bundle(run)
    probe = result["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-candidate-evidence"


def test_candidate_implementation_digest_is_recomputed_from_verified_artifact(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    probes[0]["candidates"][0]["implementation_family"][
        "implementation_sha256"
    ] = "1" * 64
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-candidate-evidence"


@pytest.mark.parametrize(
    "source_alias",
    ["same", "trailing-whitespace", "dot-segment"],
)
def test_frontier_candidate_coverage_rejects_same_source_identity(
    tmp_path: Path,
    source_alias: str,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    candidates = probes[0]["candidates"]
    first_id = candidates[0]["candidate_id"]
    second_id = candidates[1]["candidate_id"]
    shared_source = f"issue-6/a5a04f9c/{first_id}"
    second_source = {
        "same": shared_source,
        "trailing-whitespace": f"{shared_source} ",
        "dot-segment": f"issue-6/a5a04f9c/./{first_id}",
    }[source_alias]
    candidates[1]["implementation_family"]["family_id"] = (
        "forged-independent-family"
    )
    candidates[1]["implementation_family"]["manifest_ref"] = (
        "artifact://issue-21/forged-independent-family-manifest"
    )
    forged_payload = {
        "schema": "groundupscale.dev/candidate-implementation/v1alpha1",
        "source_identity": second_source,
    }
    candidates[1]["implementation_family"]["implementation_sha256"] = (
        write_json(tmp_path / "forged-implementation.json", forged_payload)
    )
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
        implementation_source_identities={second_id: second_source},
    )

    result = diagnose_run_bundle(run)
    if source_alias == "same":
        verdict = result["performance_diagnosis_verdicts"][0]
        failed = {gate["gate_id"] for gate in verdict["gates"]["failed"]}
        assert "distinct-target-alternative-implementation" in failed
    else:
        probe = result["shape_disambiguation_probes"][0]
        assert probe["status"] == "insufficient_evidence"
        assert probe["reason_code"] == "invalid-candidate-evidence"


def test_frontier_neighbourhood_thresholds_require_versioned_policy(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    del probes[0]["frontier_shift_evidence"]["neighbourhood"][
        "qualification_policy"
    ]["minimum_refit_records"]
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-shape-probe"


def _populate_frontier_holdout(
    probes: list[dict[str, object]],
    *,
    process_ids: list[int],
) -> None:
    probe = probes[0]
    contract = probe["locked_contract"]
    frontier = probe["frontier_shift_evidence"]
    holdout = frontier["holdout"]
    holdout_ids = ["holdout-1", "holdout-2", "holdout-3"]
    cohort_id = contract["cohort_id"]
    lane_id = probe["measurement_lanes"]["baseline"]["lane_id"]
    holdout["sessions"] = [
        {
            "session_id": session_id,
            "process_id": process_id,
            "lane_id": lane_id,
            "cohort_id": cohort_id,
            "evidence_ref": f"artifact://issue-21/{session_id}",
        }
        for session_id, process_id in zip(
            holdout_ids, process_ids, strict=True
        )
    ]
    holdout["candidate_results"] = [
        {
            "candidate_id": candidate["candidate_id"],
            "correctness_records": [
                {"expected": 1.0, "observed": 1.0},
                {"expected": -2.0, "observed": -2.0},
            ],
            "correctness_tolerance": {"atol": 0.0001, "rtol": 0.0001},
            "correctness_evidence_ref": candidate["correctness"][
                "evidence_ref"
            ],
            "sessions": [
                {
                    "session_id": session_id,
                    "process_id": process_id,
                    "lane_id": lane_id,
                    "cohort_id": cohort_id,
                    "raw_samples_ns": [30_000, 31_000, 32_000],
                    "excluded_samples": [],
                    "evidence_ref": (
                        "artifact://issue-21/"
                        f"{candidate['candidate_id']}-{session_id}"
                    ),
                }
                for session_id, process_id in zip(
                    holdout_ids, process_ids, strict=True
                )
            ],
            "correctness_passed": False,
            "holdout_latency_ns": 1,
        }
        for candidate in probe["candidates"]
    ]


def test_frontier_holdout_recomputes_correctness_and_latency_from_raw_records(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    _populate_frontier_holdout(probes, process_ids=[47_175, 47_179, 47_188])
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    verdict = diagnose_run_bundle(run)["performance_diagnosis_verdicts"][0]
    satisfied = {gate["gate_id"] for gate in verdict["gates"]["satisfied"]}

    assert "frontier-shift-independent-holdout" in satisfied
    assert "frontier-shift-same-hardware-validity-cohort" in satisfied
    assert (
        "frontier-shift-all-eligible-candidates-below-surface-band"
        in satisfied
    )


def test_frontier_band_is_recomputed_from_versioned_surface_uncertainty_records(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    _populate_frontier_holdout(probes, process_ids=[47_175, 47_179, 47_188])
    frontier = probes[0]["frontier_shift_evidence"]
    frontier["old_surface_uncertainty_band_ns"] = {"lower": 0, "upper": 1}
    frontier["old_surface_reference"]["predicted_ns"] = 50_000
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    verdict = diagnose_run_bundle(run)["performance_diagnosis_verdicts"][0]
    failed = {gate["gate_id"] for gate in verdict["gates"]["failed"]}

    assert (
        "frontier-shift-all-eligible-candidates-below-surface-band" in failed
    )


def test_frontier_surface_uncertainty_requires_versioned_combination_policy(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    del probes[0]["frontier_shift_evidence"][
        "old_surface_uncertainty_policy"
    ]["scope"]
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-shape-probe"


def test_frontier_uncertainty_requires_target_coverage_and_calibration(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    del probes[0]["frontier_shift_evidence"][
        "old_surface_uncertainty_policy"
    ]["target_coverage"]
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-shape-probe"


def test_frontier_uncertainty_calibration_must_cover_numeric_target(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    policy = probes[0]["frontier_shift_evidence"][
        "old_surface_uncertainty_policy"
    ]
    policy["calibration"]["records"] = policy["calibration"]["records"][:2]
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-shape-probe"


@pytest.mark.parametrize(
    "failure",
    [
        "missing-estimator",
        "shared-process",
        "validation-miss",
        "target-alias",
    ],
)
def test_frontier_uncertainty_requires_versioned_independent_validation(
    tmp_path: Path,
    failure: str,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    policy = probes[0]["frontier_shift_evidence"][
        "old_surface_uncertainty_policy"
    ]
    records = policy["calibration"]["records"]
    if failure == "missing-estimator":
        del policy["calibration"]["estimator"]
    elif failure == "shared-process":
        records[3]["process_id"] = records[0]["process_id"]
    elif failure == "validation-miss":
        records[3]["observed_samples_ns"] = [20_000, 20_000]
    else:
        aliased_target = f"{records[0]['target_id']} "
        records[3]["target_id"] = aliased_target
        policy["target_coverage"]["required_validation_target_ids"][0] = (
            aliased_target
        )
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-shape-probe"


@pytest.mark.parametrize(
    "policy_location",
    ["trigger", "surface", "neighbourhood", "verdict"],
)
@pytest.mark.parametrize("unresolved_value", ["unknown", "not_applicable"])
def test_unresolved_governance_policy_fails_closed(
    tmp_path: Path,
    policy_location: str,
    unresolved_value: str,
) -> None:
    fixture_name = (
        "issue21-context-matmul-headroom.json"
        if policy_location in {"trigger", "verdict"}
        else "issue21-257-cube-insufficient.json"
    )
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / fixture_name).read_text(
            encoding="utf-8"
        )
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    verdict_policy = deepcopy(fixture["verdict_policy"])
    trigger_policy = None
    if policy_location == "trigger":
        trigger_policy = {
            "policy_id": unresolved_value,
            "version": "v1",
            "scope": "exact-shape-performance-diagnosis",
            "change_reason": "issue-21 normative trigger",
            "revalidation": "on uncertainty or materiality policy change",
        }
    elif policy_location == "surface":
        probes[0]["frontier_shift_evidence"][
            "old_surface_uncertainty_policy"
        ]["policy_id"] = unresolved_value
    elif policy_location == "neighbourhood":
        probes[0]["frontier_shift_evidence"]["neighbourhood"][
            "qualification_policy"
        ]["scope"] = unresolved_value
    else:
        verdict_policy["policy_id"] = unresolved_value
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=verdict_policy,
        trigger_policy=trigger_policy,
    )

    result = diagnose_run_bundle(run)

    assert not any(
        verdict.get("verdict") == "implementation_headroom"
        and verdict.get("status") == "decided"
        for verdict in result["performance_diagnosis_verdicts"]
    )
    if policy_location in {"surface", "neighbourhood"}:
        assert result["shape_disambiguation_probes"][0]["status"] == (
            "insufficient_evidence"
        )


def test_uncertainty_component_value_must_match_verified_artifact_content(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=fixture["shape_disambiguation_probes"],
        verdict_policy=fixture["verdict_policy"],
    )
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(
        item
        for item in manifest["artifacts"]
        if item.get("uri") == "artifact://issue-6/257-uncertainty-anchor"
    )
    artifact["sha256"] = write_json(
        run / artifact["path"],
        {
            "schema": "groundupscale.dev/uncertainty-component/v1alpha1",
            "component_id": "anchor",
            "standard_uncertainty_ns": 599,
        },
    )
    write_json(manifest_path, manifest)

    probe = diagnose_run_bundle(run)["shape_disambiguation_probes"][0]

    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-shape-probe"


@pytest.mark.parametrize("boundary_mismatch", ["prediction", "uncertainty"])
def test_frontier_surface_band_binds_the_trigger_boundary(
    tmp_path: Path,
    boundary_mismatch: str,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    _populate_frontier_holdout(probes, process_ids=[47_175, 47_179, 47_188])
    frontier = probes[0]["frontier_shift_evidence"]
    if boundary_mismatch == "prediction":
        frontier["old_surface_reference"]["predicted_ns"] = 24_500
    else:
        frontier["old_surface_uncertainty_records"][0][
            "standard_uncertainty_ns"
        ] = 601
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    result = diagnose_run_bundle(run)
    if boundary_mismatch == "uncertainty":
        probe = result["shape_disambiguation_probes"][0]
        assert probe["status"] == "insufficient_evidence"
        assert probe["reason_code"] == "invalid-shape-probe"
        return
    verdict = result["performance_diagnosis_verdicts"][0]
    failed = {gate["gate_id"] for gate in verdict["gates"]["failed"]}

    assert (
        "frontier-shift-all-eligible-candidates-below-surface-band" in failed
    )


def test_frontier_holdout_processes_must_be_disjoint_from_search_processes(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    _populate_frontier_holdout(probes, process_ids=[37_175, 37_179, 37_188])
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    verdict = diagnose_run_bundle(run)["performance_diagnosis_verdicts"][0]
    failed = {gate["gate_id"] for gate in verdict["gates"]["failed"]}

    assert "frontier-shift-independent-holdout" in failed


@pytest.mark.parametrize("raw_failure", ["correctness", "latency"])
def test_frontier_holdout_rejects_authored_summary_when_raw_records_fail(
    tmp_path: Path,
    raw_failure: str,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    _populate_frontier_holdout(probes, process_ids=[47_175, 47_179, 47_188])
    result = probes[0]["frontier_shift_evidence"]["holdout"][
        "candidate_results"
    ][0]
    result["correctness_passed"] = True
    result["holdout_latency_ns"] = 99_999
    if raw_failure == "correctness":
        result["correctness_records"][0]["observed"] = 2.0
    else:
        result["sessions"][0]["raw_samples_ns"] = [1_000, 1_100, 1_200]
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    verdict = diagnose_run_bundle(run)["performance_diagnosis_verdicts"][0]
    failed = {gate["gate_id"] for gate in verdict["gates"]["failed"]}

    assert (
        "frontier-shift-all-eligible-candidates-below-surface-band" in failed
    )


def test_frontier_holdout_must_remain_on_locked_baseline_lane(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    _populate_frontier_holdout(probes, process_ids=[47_175, 47_179, 47_188])
    holdout = probes[0]["frontier_shift_evidence"]["holdout"]
    for session in holdout["sessions"]:
        session["lane_id"] = "matmul-257-diagnostic"
    for result in holdout["candidate_results"]:
        for session in result["sessions"]:
            session["lane_id"] = "matmul-257-diagnostic"
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    verdict = diagnose_run_bundle(run)["performance_diagnosis_verdicts"][0]
    failed = {gate["gate_id"] for gate in verdict["gates"]["failed"]}

    assert "frontier-shift-independent-holdout" in failed


def test_frontier_neighbourhood_is_derived_from_anchor_probe_and_refit_records(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    probe = probes[0]
    neighbourhood = probe["frontier_shift_evidence"]["neighbourhood"]
    cohort_id = probe["locked_contract"]["cohort_id"]
    regime_id = neighbourhood["regime_id"]

    def observation(shape: dict[str, int], suffix: str) -> dict[str, object]:
        return {
            "shape": shape,
            "cohort_id": cohort_id,
            "regime_id": regime_id,
            "observation_validity": "QUALIFIED",
            "frontier_role": "ACTIVE",
            "surface": {"surface_id": "m4-matmul-fp32", "version": "v1"},
            "correctness_records": [{"expected": 1.0, "observed": 1.0}],
            "correctness_tolerance": {"atol": 0.0001, "rtol": 0.0001},
            "holdout_sessions": [
                {
                    "session_id": f"{suffix}-session-{index}",
                    "process_id": 57_000 + index,
                    "lane_id": "matmul-257-baseline",
                    "cohort_id": cohort_id,
                    "raw_samples_ns": [25_000, 25_100, 25_200],
                    "excluded_samples": [],
                    "evidence_ref": (
                        f"artifact://issue-21/neighbourhood-{suffix}-session-{index}"
                    ),
                }
                for index in range(1, 4)
            ],
            "predicted_ns": 25_000,
            "observed_ns": 25_100,
            "combined_uncertainty_ns": 200,
            "uncertainty_records": [
                {
                    "component_id": "anchor",
                    "standard_uncertainty_ns": 120,
                    "evidence_ref": (
                        f"artifact://issue-21/{suffix}-uncertainty-anchor"
                    ),
                },
                {
                    "component_id": "interpolation",
                    "standard_uncertainty_ns": 160,
                    "evidence_ref": (
                        f"artifact://issue-21/{suffix}-uncertainty-interpolation"
                    ),
                },
                {
                    "component_id": "instrumentation",
                    "standard_uncertainty_ns": 0,
                    "evidence_ref": (
                        f"artifact://issue-21/{suffix}-uncertainty-instrumentation"
                    ),
                },
            ],
            "evidence_ref": f"artifact://issue-21/neighbourhood-{suffix}",
        }

    local_shapes = []
    target_shape = {"m": 257, "k": 257, "n": 257}
    for name in sorted(target_shape):
        for delta in (-1, 1):
            shape = dict(target_shape)
            shape[name] += delta
            local_shapes.append(shape)
    neighbourhood["anchor_records"] = [
        observation(local_shapes[0], "anchor-1"),
        observation(local_shapes[1], "anchor-2"),
    ]
    neighbourhood["local_probe_records"] = [
        observation(shape, f"local-{index}")
        for index, shape in enumerate(local_shapes, start=1)
    ]
    neighbourhood["refit_records"] = [
        observation(local_shapes[index], f"refit-{index + 1}")
        for index in range(3)
    ]
    neighbourhood["stable_neighbouring_anchors"] = False
    neighbourhood["dense_local_shape_probe"] = False
    neighbourhood["same_regime_refit"] = False
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    verdict = diagnose_run_bundle(run)["performance_diagnosis_verdicts"][0]
    satisfied = {gate["gate_id"] for gate in verdict["gates"]["satisfied"]}

    assert "frontier-shift-validated-neighbourhood" in satisfied


@pytest.mark.parametrize(
    "record_failure",
    ["unstable-local", "duplicate-refit", "inflated-uncertainty"],
)
def test_frontier_neighbourhood_rejects_unstable_or_duplicate_records(
    tmp_path: Path,
    record_failure: str,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "issue21-257-cube-insufficient.json"
        ).read_text(encoding="utf-8")
    )
    probes = deepcopy(fixture["shape_disambiguation_probes"])
    probe = probes[0]
    neighbourhood = probe["frontier_shift_evidence"]["neighbourhood"]
    cohort_id = probe["locked_contract"]["cohort_id"]
    regime_id = neighbourhood["regime_id"]

    def observation(shape: dict[str, int], suffix: str) -> dict[str, object]:
        return {
            "shape": shape,
            "cohort_id": cohort_id,
            "regime_id": regime_id,
            "observation_validity": "QUALIFIED",
            "frontier_role": "ACTIVE",
            "surface": {"surface_id": "m4-matmul-fp32", "version": "v1"},
            "correctness_records": [{"expected": 1.0, "observed": 1.0}],
            "correctness_tolerance": {"atol": 0.0001, "rtol": 0.0001},
            "holdout_sessions": [
                {
                    "session_id": f"{suffix}-session-{index}",
                    "process_id": 57_000 + index,
                    "lane_id": "matmul-257-baseline",
                    "cohort_id": cohort_id,
                    "raw_samples_ns": [25_000, 25_100, 25_200],
                    "excluded_samples": [],
                    "evidence_ref": f"artifact://issue-21/{suffix}-{index}",
                }
                for index in range(1, 4)
            ],
            "predicted_ns": 25_000,
            "observed_ns": 25_100,
            "combined_uncertainty_ns": 200,
            "uncertainty_records": [
                {
                    "component_id": "anchor",
                    "standard_uncertainty_ns": 120,
                    "evidence_ref": (
                        f"artifact://issue-21/{suffix}-uncertainty-anchor"
                    ),
                },
                {
                    "component_id": "interpolation",
                    "standard_uncertainty_ns": 160,
                    "evidence_ref": (
                        f"artifact://issue-21/{suffix}-uncertainty-interpolation"
                    ),
                },
                {
                    "component_id": "instrumentation",
                    "standard_uncertainty_ns": 0,
                    "evidence_ref": (
                        f"artifact://issue-21/{suffix}-uncertainty-instrumentation"
                    ),
                },
            ],
            "evidence_ref": f"artifact://issue-21/{suffix}",
        }

    target_shape = {"m": 257, "k": 257, "n": 257}
    local_shapes = []
    for name in sorted(target_shape):
        for delta in (-1, 1):
            shape = dict(target_shape)
            shape[name] += delta
            local_shapes.append(shape)
    neighbourhood["anchor_records"] = [
        observation(local_shapes[0], "anchor-1"),
        observation(local_shapes[1], "anchor-2"),
    ]
    neighbourhood["local_probe_records"] = [
        observation(shape, f"local-{index}")
        for index, shape in enumerate(local_shapes, start=1)
    ]
    neighbourhood["refit_records"] = [
        observation(local_shapes[index], f"refit-{index + 1}")
        for index in range(3)
    ]
    if record_failure == "unstable-local":
        neighbourhood["local_probe_records"][0]["observed_ns"] = 99_999
        neighbourhood["local_probe_records"][0][
            "combined_uncertainty_ns"
        ] = 100_000
    elif record_failure == "duplicate-refit":
        duplicated = neighbourhood["refit_records"][0]
        neighbourhood["refit_records"] = [deepcopy(duplicated) for _ in range(3)]
    else:
        neighbourhood["local_probe_records"][0]["observed_ns"] = 99_999
        neighbourhood["local_probe_records"][0]["uncertainty_records"][0][
            "standard_uncertainty_ns"
        ] = 100_000
        neighbourhood["local_probe_records"][0][
            "combined_uncertainty_ns"
        ] = 100_000
    run = _write_bundle(
        tmp_path,
        diagnostic_items=fixture["diagnostic_items"],
        probes=probes,
        verdict_policy=fixture["verdict_policy"],
    )

    result = diagnose_run_bundle(run)
    if record_failure == "inflated-uncertainty":
        probe_result = result["shape_disambiguation_probes"][0]
        assert probe_result["status"] == "insufficient_evidence"
        assert probe_result["reason_code"] == "invalid-shape-probe"
        return
    verdict = result["performance_diagnosis_verdicts"][0]
    failed = {gate["gate_id"] for gate in verdict["gates"]["failed"]}

    assert "frontier-shift-validated-neighbourhood" in failed
