from __future__ import annotations

import json
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from statistics import median

import pytest

from groundupscale.diagnostics import (
    DiagnosticBundleIntegrityError,
    diagnose_run_bundle,
    render_diagnostic_report,
)
from groundupscale.run_bundle import verify_run_bundle

REPOSITORY_ROOT = Path(__file__).parents[1]
AUTHORITATIVE_BUNDLE = (
    REPOSITORY_ROOT
    / "goal_process"
    / "issue-32-ascend-diagnostic-bundle"
    / "evidence"
    / "runs"
    / "issue32-ascend-910b2-diagnostic-v1"
)


def _canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def _resign_diagnostic_bundle(
    tampered: Path,
    evidence: dict[str, object],
    manifest: dict[str, object],
) -> None:
    inputs = {
        key: evidence[key]
        for key in (
            "resolved_configuration",
            "resolved_ir",
            "hardware",
            "cohort_id",
            "execution_domain",
        )
    }
    evidence_body = {
        key: value
        for key, value in evidence.items()
        if key not in {*inputs, "schema", "digests"}
    }
    evidence["digests"] = {
        "input_sha256": _canonical_digest(inputs),
        "evidence_sha256": _canonical_digest(evidence_body),
    }
    evidence_digest = _write_json(
        tampered / "diagnostic/evidence.json", evidence
    )
    next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["path"] == "diagnostic/evidence.json"
    )["sha256"] = evidence_digest
    for source in manifest["source_runs"]:
        original_source_manifest = (
            AUTHORITATIVE_BUNDLE / source["path"] / "run.manifest.json"
        ).resolve()
        copied_source_manifest = (
            tampered / source["path"] / "run.manifest.json"
        ).resolve()
        copied_source_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_source_manifest, copied_source_manifest)
    _write_json(tampered / "run.manifest.json", manifest)


def test_real_ascend_bundle_replays_four_axes_and_evidence_qualified_verdicts(
) -> None:
    verification = verify_run_bundle(AUTHORITATIVE_BUNDLE)
    assert verification["passed"] is True
    assert verification["failures"] == []
    assert verification["artifact_count"] > 20

    result = diagnose_run_bundle(AUTHORITATIVE_BUNDLE)
    axes = result["axes"]
    assert set(axes) == {
        "resource_physical_floor",
        "operator_achievable_frontier",
        "schedule_achievable_frontier",
        "observation",
    }
    assert {axis["status"] for axis in axes.values()} == {"known"}
    assert axes["resource_physical_floor"]["may_be_unattainable"] is True
    assert axes["resource_physical_floor"]["value_ns"] == pytest.approx(
        13_998.515,
        rel=1e-6,
    )
    assert axes["operator_achievable_frontier"]["value_ns"] == pytest.approx(
        16_331.5,
    )
    assert axes["schedule_achievable_frontier"]["value_ns"] >= axes[
        "operator_achievable_frontier"
    ]["value_ns"]
    assert axes["observation"]["value_ns"] > axes[
        "schedule_achievable_frontier"
    ]["value_ns"]
    assert result["comparisons"]["physical_floor_to_observation"][
        "prediction_error_ns"
    ] is None
    assert result["adapter_contract"]["lanes"] == {
        "pair_id": "issue32-ascend-paired-lanes",
        "baseline_lane_id": "issue32-ascend-baseline",
        "diagnostic_lane_id": "issue32-ascend-diagnostic",
        "diagnostic_frontier_eligible": False,
        "reason_code": "profiling-overhead-error-budget-exceeded",
    }
    surface_query = next(
        query
        for query in result["capability_surface_queries"]
        if query["query_id"] == "ascend-matmul-square-512"
    )
    assert surface_query["status"] == "exact_anchor"
    assert surface_query["surface"]["surface_id"].startswith(
        "surface://ascend-npu-23b93a89d5fecc79/"
    )
    assert result["adapter_contract"]["surface_refs"] == [
        surface_query["surface"]
    ]

    verdicts = {
        verdict["stable_path"]: verdict
        for verdict in result["performance_diagnosis_verdicts"]
    }
    integration = verdicts[
        "semantic/model/two-layer-transformer/transformer/"
        "layer-0/attention/q-proj"
    ]
    assert integration["verdict"] == "integration_overhead"
    assert integration["ledger"]["status"] == "conserved"
    assert integration["ledger"]["parent_span_total_included_ns"] == 0
    assert {
        leaf["kind"] for leaf in integration["ledger"]["leaves"]
    } >= {"operator", "copy", "dispatch", "sync", "profiling"}
    assert integration["ledger"]["residual"]["kind"] == "unattributed"
    assert integration["ledger"]["residual"]["duration_ns"] >= 0
    assert integration["metrics"][
        "operator_frontier_combined_uncertainty_ns"
    ] == pytest.approx(107.21727757774897, rel=1e-9)
    assert integration["surface_action"]["action"] == "preserve"
    preserved_frontier = integration["surface_action"][
        "operator_achievable_frontier_ns"
    ]
    assert preserved_frontier["before"] == pytest.approx(
        axes["operator_achievable_frontier"]["value_ns"]
    )
    assert preserved_frontier["after"] == pytest.approx(
        axes["operator_achievable_frontier"]["value_ns"]
    )

    insufficient = verdicts[
        "semantic/model/two-layer-transformer/transformer/"
        "layer-0/attention/k-proj"
    ]
    assert insufficient["verdict"] == "insufficient_evidence"
    assert all(
        gate["gate_id"] != "direct-correctness-violation"
        for gate in insufficient["gates"]["satisfied"]
    )

    confirmed = verdicts[
        "semantic/model/two-layer-transformer/transformer/"
        "layer-0/attention/v-proj"
    ]
    assert confirmed["verdict"] == "confirmed_bug"
    assert any(
        gate["gate_id"] == "direct-correctness-violation"
        for gate in confirmed["gates"]["satisfied"]
    )
    assert {
        reason_code
        for counterexample in confirmed["counterexamples"]
        for reason_code in counterexample.get(
            "reason_codes", [counterexample.get("reason_code")]
        )
    } >= {
        "performance-gap-is-not-direct-defect-evidence",
        "proxy-anomaly-is-not-direct-defect-evidence",
        "single-fluctuation-is-not-reproducible",
    }


def test_real_bundle_replays_source_values_and_derived_ablation_contracts() -> None:
    document = json.loads(
        (AUTHORITATIVE_BUNDLE / "diagnostic/evidence.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (AUTHORITATIVE_BUNDLE / "run.manifest.json").read_text(
            encoding="utf-8"
        )
    )

    def artifact(uri: str) -> dict[str, object]:
        record = next(
            item for item in manifest["artifacts"] if item.get("uri") == uri
        )
        return json.loads(
            (AUTHORITATIVE_BUNDLE / record["path"]).read_text(
                encoding="utf-8"
            )
        )

    source_floor = artifact("artifact://issue-32/source-physical-floor")
    source_benchmark = artifact(
        "artifact://issue-32/source-transformer-benchmark"
    )
    source_e2e = artifact(
        "artifact://issue-32/source-transformer-e2e-attribution"
    )
    source_remote = artifact(
        "artifact://issue-32/source-remote-execution"
    )
    remote_by_session = {
        item["session_id"]: item for item in source_remote["sessions"]
    }
    source_capabilities = {
        item["resource"]: item
        for item in source_floor["physical_floor"]["capabilities"]
    }
    terms = {
        item["resource"]: item
        for item in document["resource_physical_floor"]["resource_terms"]
    }
    assert terms["compute.fp32"]["validated_rate_per_second"] == (
        source_capabilities["compute.fp32"]["robust_achievable_rate"]
    )
    assert terms["memory.hbm"]["validated_rate_per_second"] == (
        source_capabilities["memory.hbm"]["robust_achievable_rate"]
    )
    assert all(
        term["source_evidence_ref"]
        == "artifact://issue-32/source-physical-floor"
        for term in terms.values()
    )

    q_case = next(
        case
        for case in source_benchmark["cases"]
        if case["case_id"] == "matmul-q-proj"
    )
    trigger_q = next(
        item
        for item in document["diagnostic_trigger_input"]["items"]
        if item["stable_path"].endswith("/q-proj")
    )
    assert trigger_q["observed_ns"] == q_case["latency"]["median_ns"]
    assert document["diagnostic_trigger_input"]["e2e_observation_ns"] == (
        source_e2e["e2e_trace_host_ns"]
    )
    assert document["diagnostic_trigger_input"]["source_evidence_ref"] == (
        "artifact://issue-32/source-transformer-e2e-attribution"
    )
    q_basis = trigger_q["observation_basis"]
    assert q_basis == {
        "kind": "benchmark-case",
        "semantic": "batch-one Q projection MatMul",
        "source_case_id": "matmul-q-proj",
        "source_evidence_ref": (
            "artifact://issue-32/source-transformer-benchmark"
        ),
        "stable_path": trigger_q["stable_path"],
    }

    for suffix, variant, semantic in (
        ("/k-proj", "k_baseline", "batch-one K projection MatMul"),
        ("/v-proj", "v_baseline", "batch-one V projection MatMul"),
    ):
        trigger = next(
            item
            for item in document["diagnostic_trigger_input"]["items"]
            if item["stable_path"].endswith(suffix)
        )
        basis = trigger["observation_basis"]
        assert basis["variant"] == variant
        assert basis["semantic"] == semantic
        assert basis["stable_path"] == trigger["stable_path"]
        assert basis["lane"] == "baseline"
        assert basis["reducer"] == (
            "median-of-independent-session-medians"
        )
        source_sessions = [artifact(ref) for ref in basis["input_refs"]]
        assert len({raw["process_id"] for raw in source_sessions}) == 3
        for ref, raw in zip(
            basis["input_refs"], source_sessions, strict=True
        ):
            path_inputs = raw["path_inputs"]
            assert len(
                {
                    (identity["left_sha256"], identity["right_sha256"])
                    for identity in path_inputs.values()
                }
            ) == 3
            variant_contract = raw["execution_contract"][
                "variant_contracts"
            ][variant]
            assert variant_contract == {
                "lane": "baseline",
                "semantic": semantic,
                "stable_path": trigger["stable_path"],
                "input_identity": path_inputs[variant[0]],
            }
            source_manifest_entry = next(
                item for item in manifest["artifacts"] if item.get("uri") == ref
            )
            remote_record = remote_by_session[raw["session_id"]]
            assert source_manifest_entry["sha256"] == remote_record["sha256"]
            assert raw["process_id"] == remote_record["process_id"]
            assert raw["process_started_at"] == remote_record["started_at"]
        replayed = median(
            median(raw["variants"][variant]["raw_samples_ns"])
            for raw in source_sessions
        )
        assert trigger["observed_ns"] == replayed

    probes = {
        probe["stable_path"]: probe
        for probe in document["shape_disambiguation_probes"]
    }
    assert probes[next(path for path in probes if path.endswith("/q-proj"))][
        "locked_contract"
    ]["semantic"] == "batch-one Q projection MatMul"
    assert probes[next(path for path in probes if path.endswith("/k-proj"))][
        "locked_contract"
    ]["semantic"] == "batch-one K projection MatMul"
    assert probes[
        next(path for path in probes if path.endswith("/v-proj"))
    ]["locked_contract"]["semantic"] == (
        "batch-one V projection MatMul negative control"
    )
    negative_probe = probes[
        next(path for path in probes if path.endswith("/v-proj"))
    ]
    negative_target = negative_probe["candidates"][0]
    assert negative_target["evidence_lane"] == "diagnostic"
    assert {
        session["lane_id"] for session in negative_target["sessions"]
    } == {negative_probe["measurement_lanes"]["diagnostic"]["lane_id"]}

    q_probe = probes[next(path for path in probes if path.endswith("/q-proj"))]
    integration = q_probe["integration_overhead_evidence"]
    frontier = integration["operator_frontier"]
    assert "replay_measurement_uncertainty" not in frontier["uncertainty_basis"]
    assert frontier["combined_uncertainty_ns"] == pytest.approx(
        frontier["uncertainty_basis"]["surface_uncertainty_ns"]
    )
    assert integration["exclusive_ledger"]["residual"]["kind"] == (
        "unattributed"
    )
    assert integration["exclusive_ledger"]["residual"]["duration_ns"] >= 0
    for ablation in integration["ablations"]:
        assert ablation["derivation"]["input_refs"]
        for session in ablation["sessions"]:
            assert session["derived_samples_ns"]
            assert "raw_samples_ns" not in session


def test_cli_json_and_human_report_drill_down_to_raw_bundle() -> None:
    expected = diagnose_run_bundle(AUTHORITATIVE_BUNDLE)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(AUTHORITATIVE_BUNDLE),
            "--json",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == expected

    report = render_diagnostic_report(expected)
    for expected_text in (
        f"derivation: {expected['derivation']['derivation_id']}",
        "qualification policy: issue32-frontier-qualification/v2",
        "Frontier Anchor issue32-ascend-matmul-square-512",
        "Capability Surface ascend-matmul-square-512",
        "candidate search: winner=torch.matmul",
        "raw bundle: run-bundle://issue32-ascend-910b2-diagnostic-v1",
        "source run issue31-operator-frontier-v3: operator-frontier",
        "Shape Disambiguation Probe issue32-q-proj-integration: complete",
        (
            "trigger observation basis: "
            "semantic/model/two-layer-transformer/transformer/"
            "layer-0/attention/k-proj"
        ),
        "ablation remove-profiling(profiling)",
        "satisfied gates: diagnostic-trigger-met",
        "Operator Achievable Frontier preserved",
        "Performance Diagnosis Verdict",
    ):
        assert expected_text in report

    manifest = json.loads(
        (AUTHORITATIVE_BUNDLE / "run.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        source.get("path") and source.get("manifest_sha256")
        for source in manifest["source_runs"]
    )


def test_missing_ablation_artifact_fails_closed(tmp_path: Path) -> None:
    tampered = tmp_path / "tampered-diagnostic-bundle"
    shutil.copytree(AUTHORITATIVE_BUNDLE, tampered)
    manifest = json.loads(
        (tampered / "run.manifest.json").read_text(encoding="utf-8")
    )
    artifact = next(
        item
        for item in manifest["artifacts"]
        if item.get("uri") == "artifact://issue-32/integration-copy-1"
    )
    (tampered / artifact["path"]).unlink()

    verification = verify_run_bundle(tampered)
    assert verification["passed"] is False
    assert verification["failures"]
    with pytest.raises(DiagnosticBundleIntegrityError):
        diagnose_run_bundle(tampered)


def test_source_run_manifest_lineage_fails_closed(tmp_path: Path) -> None:
    tampered = (
        tmp_path
        / "goal_process"
        / "issue-32-ascend-diagnostic-bundle"
        / "evidence"
        / "runs"
        / AUTHORITATIVE_BUNDLE.name
    )
    shutil.copytree(AUTHORITATIVE_BUNDLE, tampered)
    manifest_path = tampered / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for source in manifest["source_runs"]:
        original_source_manifest = (
            AUTHORITATIVE_BUNDLE / source["path"] / "run.manifest.json"
        ).resolve()
        copied_source_manifest = (
            tampered / source["path"] / "run.manifest.json"
        ).resolve()
        copied_source_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_source_manifest, copied_source_manifest)
    manifest["source_runs"][0]["manifest_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(tampered)
    assert "source Run manifest digest mismatch: " in "\n".join(
        verification["failures"]
    )
    with pytest.raises(DiagnosticBundleIntegrityError):
        diagnose_run_bundle(tampered)


def test_derived_ablation_is_replayed_from_raw_session_sources(
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "tampered-derived-ablation"
    shutil.copytree(AUTHORITATIVE_BUNDLE, tampered)
    manifest_path = tampered / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_session = "issue32-session-01-v7"

    def rewrite(value: object) -> bool:
        changed = False
        if isinstance(value, dict):
            if (
                value.get("session_id") == target_session
                and isinstance(value.get("derived_samples_ns"), list)
                and value["derived_samples_ns"]
            ):
                value["derived_samples_ns"][0] += 1.0
                value["latency_ns"] += 1.0
                changed = True
            for item in value.values():
                changed = rewrite(item) or changed
        elif isinstance(value, list):
            for item in value:
                changed = rewrite(item) or changed
        return changed

    evidence_path = tampered / "diagnostic/evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert rewrite(evidence)
    inputs = {
        key: evidence[key]
        for key in (
            "resolved_configuration",
            "resolved_ir",
            "hardware",
            "cohort_id",
            "execution_domain",
        )
    }
    evidence_body = {
        key: value
        for key, value in evidence.items()
        if key not in {*inputs, "schema", "digests"}
    }

    def canonical(value: object) -> str:
        return sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    evidence["digests"] = {
        "input_sha256": canonical(inputs),
        "evidence_sha256": canonical(evidence_body),
    }

    def write_json(path: Path, value: object) -> str:
        payload = (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        path.write_bytes(payload)
        return sha256(payload).hexdigest()

    for artifact in manifest["artifacts"]:
        path = tampered / artifact["path"]
        if artifact["path"] == "diagnostic/evidence.json":
            artifact["sha256"] = write_json(path, evidence)
            continue
        if not artifact["path"].startswith("diagnostic/supporting/"):
            continue
        content = json.loads(path.read_text(encoding="utf-8"))
        if rewrite(content):
            artifact["sha256"] = write_json(path, content)
    for source in manifest["source_runs"]:
        original_source_manifest = (
            AUTHORITATIVE_BUNDLE / source["path"] / "run.manifest.json"
        ).resolve()
        copied_source_manifest = (
            tampered / source["path"] / "run.manifest.json"
        ).resolve()
        copied_source_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_source_manifest, copied_source_manifest)
    write_json(manifest_path, manifest)

    assert verify_run_bundle(tampered)["passed"] is True
    result = diagnose_run_bundle(tampered)
    q_verdict = next(
        verdict
        for verdict in result["performance_diagnosis_verdicts"]
        if verdict["stable_path"].endswith("/q-proj")
    )
    assert q_verdict["verdict"] == "insufficient_evidence"
    assert q_verdict["gates"]["failed"][0]["reason_code"] == (
        "integration-overhead-evidence-invalid"
    )


def test_trigger_rejects_relabelled_stable_path_source_evidence(
    tmp_path: Path,
) -> None:
    tampered = (
        tmp_path
        / "goal_process"
        / "issue-32-ascend-diagnostic-bundle"
        / "evidence"
        / "runs"
        / AUTHORITATIVE_BUNDLE.name
    )
    shutil.copytree(AUTHORITATIVE_BUNDLE, tampered)
    evidence_path = tampered / "diagnostic/evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    k_item = next(
        item
        for item in evidence["diagnostic_trigger_input"]["items"]
        if item["stable_path"].endswith("/k-proj")
    )
    k_item["observation_basis"]["semantic"] = (
        "batch-one Q projection MatMul"
    )

    inputs = {
        key: evidence[key]
        for key in (
            "resolved_configuration",
            "resolved_ir",
            "hardware",
            "cohort_id",
            "execution_domain",
        )
    }
    evidence_body = {
        key: value
        for key, value in evidence.items()
        if key not in {*inputs, "schema", "digests"}
    }

    def canonical(value: object) -> str:
        return sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def write_json(path: Path, value: object) -> str:
        payload = (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        path.write_bytes(payload)
        return sha256(payload).hexdigest()

    evidence["digests"] = {
        "input_sha256": canonical(inputs),
        "evidence_sha256": canonical(evidence_body),
    }
    evidence_digest = write_json(evidence_path, evidence)
    manifest_path = tampered / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["path"] == "diagnostic/evidence.json"
    )["sha256"] = evidence_digest
    for source in manifest["source_runs"]:
        original_source_manifest = (
            AUTHORITATIVE_BUNDLE / source["path"] / "run.manifest.json"
        ).resolve()
        copied_source_manifest = (
            tampered / source["path"] / "run.manifest.json"
        ).resolve()
        copied_source_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_source_manifest, copied_source_manifest)
    write_json(manifest_path, manifest)

    assert verify_run_bundle(tampered)["passed"] is True
    trigger = diagnose_run_bundle(tampered)["diagnostic_trigger"]
    assert trigger == {
        "status": "unknown",
        "reason_code": "diagnostic-trigger-source-evidence-mismatch",
        "evidence_refs": [],
    }


def test_trigger_requires_source_and_remote_session_digest(
    tmp_path: Path,
) -> None:
    for scenario in ("missing-source", "remote-digest-mismatch"):
        tampered = (
            tmp_path
            / scenario
            / "goal_process"
            / "issue-32-ascend-diagnostic-bundle"
            / "evidence"
            / "runs"
            / AUTHORITATIVE_BUNDLE.name
        )
        shutil.copytree(AUTHORITATIVE_BUNDLE, tampered)
        evidence = json.loads(
            (tampered / "diagnostic/evidence.json").read_text(
                encoding="utf-8"
            )
        )
        manifest = json.loads(
            (tampered / "run.manifest.json").read_text(encoding="utf-8")
        )
        trigger_input = evidence["diagnostic_trigger_input"]
        if scenario == "missing-source":
            trigger_input.pop("source_evidence_ref")
            expected_reason = "diagnostic-trigger-source-evidence-invalid"
        else:
            k_item = next(
                item
                for item in trigger_input["items"]
                if item["stable_path"].endswith("/k-proj")
            )
            delta_ns = 100_000.0
            for ref in k_item["observation_basis"]["input_refs"]:
                artifact = next(
                    item
                    for item in manifest["artifacts"]
                    if item.get("uri") == ref
                )
                path = tampered / artifact["path"]
                source = json.loads(path.read_text(encoding="utf-8"))
                samples = source["variants"]["k_baseline"][
                    "raw_samples_ns"
                ]
                source["variants"]["k_baseline"]["raw_samples_ns"] = [
                    sample + delta_ns for sample in samples
                ]
                source["variants"]["k_baseline"]["median_ns"] += delta_ns
                artifact["sha256"] = _write_json(path, source)
            k_item["observed_ns"] += delta_ns
            expected_reason = "diagnostic-trigger-source-evidence-mismatch"
        _resign_diagnostic_bundle(tampered, evidence, manifest)

        assert verify_run_bundle(tampered)["passed"] is True
        trigger = diagnose_run_bundle(tampered)["diagnostic_trigger"]
        assert trigger["status"] == "unknown"
        assert trigger["reason_code"] == expected_reason


def test_baseline_compatibility_ref_cannot_skip_item_replay(
    tmp_path: Path,
) -> None:
    tampered = (
        tmp_path
        / "baseline-compatibility-bypass"
        / "goal_process"
        / "issue-32-ascend-diagnostic-bundle"
        / "evidence"
        / "runs"
        / AUTHORITATIVE_BUNDLE.name
    )
    shutil.copytree(AUTHORITATIVE_BUNDLE, tampered)
    evidence = json.loads(
        (tampered / "diagnostic/evidence.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (tampered / "run.manifest.json").read_text(encoding="utf-8")
    )
    trigger_input = evidence["diagnostic_trigger_input"]
    trigger_input["baseline_observation_evidence_ref"] = (
        "artifact://issue-32/source-transformer-benchmark"
    )
    q_observed = next(
        item["observed_ns"]
        for item in trigger_input["items"]
        if item["stable_path"].endswith("/q-proj")
    )
    for item in trigger_input["items"]:
        item["observed_ns"] = q_observed
    k_item = next(
        item
        for item in trigger_input["items"]
        if item["stable_path"].endswith("/k-proj")
    )
    k_item["observation_basis"]["semantic"] = "bogus semantic"
    _resign_diagnostic_bundle(tampered, evidence, manifest)

    assert verify_run_bundle(tampered)["passed"] is True
    trigger = diagnose_run_bundle(tampered)["diagnostic_trigger"]
    assert trigger == {
        "status": "unknown",
        "reason_code": "diagnostic-trigger-source-evidence-mismatch",
        "evidence_refs": [],
    }


def test_negative_correctness_is_replayed_from_remote_bound_sources(
    tmp_path: Path,
) -> None:
    tampered = (
        tmp_path
        / "negative-correctness"
        / "goal_process"
        / "issue-32-ascend-diagnostic-bundle"
        / "evidence"
        / "runs"
        / AUTHORITATIVE_BUNDLE.name
    )
    shutil.copytree(AUTHORITATIVE_BUNDLE, tampered)
    manifest = json.loads(
        (tampered / "run.manifest.json").read_text(encoding="utf-8")
    )
    remote_artifact = next(
        item
        for item in manifest["artifacts"]
        if item.get("uri") == "artifact://issue-32/source-remote-execution"
    )
    remote_path = tampered / remote_artifact["path"]
    remote = json.loads(remote_path.read_text(encoding="utf-8"))
    source_records = {
        item["session_id"]: item for item in remote["sessions"]
    }
    for artifact in manifest["artifacts"]:
        uri = artifact.get("uri")
        if not (
            isinstance(uri, str)
            and uri.startswith("artifact://issue-32/raw-semantic-session-")
        ):
            continue
        source_path = tampered / artifact["path"]
        source = json.loads(source_path.read_text(encoding="utf-8"))
        negative = source["negative_control"]["correctness"]
        negative.update(
            {
                "passed": True,
                "observed_sha256": negative["expected_sha256"],
                "max_abs_difference": 0.0,
                "mismatched_elements": 0,
            }
        )
        artifact["sha256"] = _write_json(source_path, source)
        source_records[source["session_id"]]["sha256"] = artifact["sha256"]
    remote_artifact["sha256"] = _write_json(remote_path, remote)
    evidence = json.loads(
        (tampered / "diagnostic/evidence.json").read_text(encoding="utf-8")
    )
    _resign_diagnostic_bundle(tampered, evidence, manifest)

    assert verify_run_bundle(tampered)["passed"] is True
    result = diagnose_run_bundle(tampered)
    v_probe = next(
        probe
        for probe in result["shape_disambiguation_probes"]
        if probe["stable_path"].endswith("/v-proj")
    )
    assert v_probe["status"] == "insufficient_evidence"
    assert v_probe["reason_code"] == "invalid-candidate-evidence"
    v_verdict = next(
        verdict
        for verdict in result["performance_diagnosis_verdicts"]
        if verdict["stable_path"].endswith("/v-proj")
    )
    assert v_verdict["verdict"] != "confirmed_bug"


def test_negative_control_candidate_must_remain_on_diagnostic_lane(
    tmp_path: Path,
) -> None:
    tampered = (
        tmp_path
        / "lane-mismatch"
        / "goal_process"
        / "issue-32-ascend-diagnostic-bundle"
        / "evidence"
        / "runs"
        / AUTHORITATIVE_BUNDLE.name
    )
    shutil.copytree(AUTHORITATIVE_BUNDLE, tampered)
    evidence = json.loads(
        (tampered / "diagnostic/evidence.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (tampered / "run.manifest.json").read_text(encoding="utf-8")
    )
    negative_probe = next(
        probe
        for probe in evidence["shape_disambiguation_probes"]
        if probe["stable_path"].endswith("/v-proj")
    )
    for session in negative_probe["candidates"][0]["sessions"]:
        session["lane_id"] = negative_probe["measurement_lanes"]["baseline"][
            "lane_id"
        ]
    _resign_diagnostic_bundle(tampered, evidence, manifest)

    assert verify_run_bundle(tampered)["passed"] is True
    result = diagnose_run_bundle(tampered)
    probe = next(
        probe
        for probe in result["shape_disambiguation_probes"]
        if probe["stable_path"].endswith("/v-proj")
    )
    assert probe["status"] == "insufficient_evidence"
    assert probe["reason_code"] == "invalid-candidate-evidence"
