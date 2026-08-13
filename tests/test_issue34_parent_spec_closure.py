from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import yaml

from groundupscale.diagnostics import diagnose_run_bundle
from groundupscale.run_bundle import verify_run_bundle


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "evidence/datasets"
QUALIFICATIONS = ROOT / "evidence/qualifications"


def _dataset(issue: int) -> dict[str, object]:
    return yaml.safe_load(
        (DATASETS / f"issue{issue}-ascend-operator-frontier-corpus-v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_large_hardware_corpora_are_content_addressed_not_committed_run_trees() -> None:
    issue36 = _dataset(36)
    issue38 = _dataset(38)

    for document, issue, count in ((issue36, 36, 72), (issue38, 38, 236)):
        assert document["schema"] == "groundupscale.dev/evidence-dataset/v1alpha1"
        assert document["issue"] == issue
        archive = document["archive"]
        assert re.fullmatch(r"[0-9a-f]{64}", archive["sha256"])
        assert archive["uri"].startswith("https://github.com/")
        assert archive["sha256"] in archive["uri"]
        assert len(document["members"]) == count
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", member["manifest_sha256"])
            and member["artifact_uri"].startswith("https://github.com/")
            for member in document["members"]
        )

    committed_runs = ROOT / (
        "goal_process/issue-38-ascend-flash-attention-sequence-sweep/"
        "evidence/runs"
    )
    assert not committed_runs.exists()


def test_issue36_committed_unknown_is_verifiable_and_replayable_offline() -> None:
    run = QUALIFICATIONS / "issue36-bounded-collection-corpus-incomplete-v1"

    verification = verify_run_bundle(run)
    diagnosis = diagnose_run_bundle(run)

    assert verification["passed"] is True
    assert verification["failures"] == []
    assert [
        query["status"] for query in diagnosis["capability_surface_queries"]
    ] == ["unknown"] * 4
    assert {
        query["reason_code"]
        for query in diagnosis["capability_surface_queries"]
    } == {"bounded-collection-corpus-incomplete"}


def test_issue38_minimal_committed_unknown_replays_without_raw_corpus() -> None:
    run = QUALIFICATIONS / "issue38-bounded-collection-stability-failed-v1"

    verification = verify_run_bundle(run)
    diagnosis = diagnose_run_bundle(run)

    assert verification["passed"] is True
    assert [
        query["status"] for query in diagnosis["capability_surface_queries"]
    ] == ["unknown"] * 7
    assert {
        query["reason_code"]
        for query in diagnosis["capability_surface_queries"]
    } == {"bounded-collection-stability-failed"}


def test_content_addressed_lineage_tampering_fails_bundle_verification(
    tmp_path: Path,
) -> None:
    source = QUALIFICATIONS / "issue36-bounded-collection-corpus-incomplete-v1"
    run = tmp_path / source.name
    shutil.copytree(source, run)
    dataset_path = run / "source/dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["members"][0]["manifest_sha256"] = "0" * 64
    dataset_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_artifact = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "source-dataset-manifest"
    )
    from hashlib import sha256

    dataset_artifact["sha256"] = sha256(dataset_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(run)
    assert verification["passed"] is False
    assert "invalid content-addressed evidence dataset" in verification["failures"]


def test_content_addressed_dataset_identity_cannot_be_silently_rebound(
    tmp_path: Path,
) -> None:
    source = QUALIFICATIONS / "issue36-bounded-collection-corpus-incomplete-v1"
    run = tmp_path / source.name
    shutil.copytree(source, run)
    qualification_path = run / "frontier/qualification.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["source_dataset"]["dataset_digest"] = "0" * 64
    qualification_path.write_text(
        json.dumps(qualification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qualification_artifact = next(
        artifact
        for artifact in manifest["artifacts"]
        if artifact["role"] == "operator-frontier-qualification"
    )
    from hashlib import sha256

    qualification_artifact["sha256"] = sha256(
        qualification_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    verification = verify_run_bundle(run)
    assert verification["passed"] is False
    assert (
        "operator Frontier source dataset identity mismatch"
        in verification["failures"]
    )
