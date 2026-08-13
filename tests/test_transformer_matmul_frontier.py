from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

from groundupscale.run_bundle import verify_run_bundle
from groundupscale.transformer_matmul_frontier import (
    TransformerMatmulFrontierBundleWriter,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
FROZEN_DEMO = (
    REPOSITORY_ROOT
    / "goal_process/issue-30-ascend-transformer-demo/evidence/runs"
    / "ascend-910b2-transformer-demo-20260811-v1"
)
Q_PROJ_FRONTIER = (
    REPOSITORY_ROOT
    / "goal_process/issue-31-ascend-matmul-frontier/evidence/runs"
    / "issue31-operator-frontier-v3"
)


def _artifact(run: Path, role: str) -> dict[str, object]:
    manifest = json.loads(
        (run / "run.manifest.json").read_text(encoding="utf-8")
    )
    entry = next(
        item for item in manifest["artifacts"] if item["role"] == role
    )
    return json.loads((run / entry["path"]).read_text(encoding="utf-8"))


def _rewrite_artifact(
    run: Path, role: str, mutate: object
) -> None:
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item for item in manifest["artifacts"] if item["role"] == role
    )
    artifact_path = run / entry["path"]
    document = json.loads(artifact_path.read_text(encoding="utf-8"))
    mutate(document)  # type: ignore[operator]
    payload = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    artifact_path.write_bytes(payload)
    entry["sha256"] = sha256(payload).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_public_bundle_lists_every_distinct_demo_matmul_domain(
    tmp_path: Path,
) -> None:
    run = TransformerMatmulFrontierBundleWriter().run(
        tmp_path,
        run_id="issue42-domain-inventory-v1",
        transformer_run=FROZEN_DEMO,
        frontier_runs=(),
    )

    assert verify_run_bundle(run)["passed"] is True
    inventory = _artifact(run, "matmul-domain-inventory")
    assert inventory["source_run_id"] == (
        "ascend-910b2-transformer-demo-20260811-v1"
    )
    assert inventory["source_hardware_cohort"] == (
        "ascend-npu-23b93a89d5fecc79"
    )
    assert inventory["matmul_leaf_count"] == 18
    assert inventory["distinct_domain_count"] == 5
    assert [domain["domain_class"] for domain in inventory["domains"]] == [
        "attention-context",
        "attention-qk",
        "mlp-contract",
        "mlp-expand",
        "projection",
    ]
    assert sum(len(domain["stable_paths"]) for domain in inventory["domains"]) == 18
    assert all(
        domain["identity"]["semantic_operation"] == "MatMul"
        and domain["identity"]["dtype"] == "float32"
        and domain["identity"]["execution_mode"] == "pytorch-eager"
        and domain["identity"]["hardware_cohort"]
        == "ascend-npu-23b93a89d5fecc79"
        and domain["identity"]["candidate_family"]
        and domain["identity"]["operand_contracts"]
        and domain["identity"]["result_contract"]
        for domain in inventory["domains"]
    )


def test_q_projection_surface_cannot_cross_complete_domain_contracts(
    tmp_path: Path,
) -> None:
    run = TransformerMatmulFrontierBundleWriter().run(
        tmp_path,
        run_id="issue42-q-proj-does-not-cover-demo-v1",
        transformer_run=FROZEN_DEMO,
        frontier_runs=(Q_PROJ_FRONTIER,),
    )

    assert verify_run_bundle(run)["passed"] is True
    qualification = _artifact(run, "transformer-matmul-frontier-qualification")
    assert qualification["status"] == "unknown"
    assert qualification["qualified_domain_count"] == 0
    assert qualification["required_domain_count"] == 5
    assert qualification["coverage_fraction"] == 0.0
    assert qualification["effective_rate_derivation"] == (
        "declared_work_flop / latency_seconds; latency is primary"
    )
    assert len(qualification["domain_queries"]) == 5
    assert all(
        query["status"] == "unknown"
        and query["latency_ns"] is None
        and query["effective_rate"] is None
        and query["reason_codes"]
        and query["minimum_next_measurement"]["domain_id"]
        == query["domain_id"]
        and query["minimum_next_measurement"]["search_holdout_identity"]
        == "disjoint"
        for query in qualification["domain_queries"]
    )
    projection = next(
        query
        for query in qualification["domain_queries"]
        if query["domain_class"] == "projection"
    )
    assert "batch-transpose-contract-mismatch" in projection["reason_codes"]
    assert projection["considered_evidence"] == [
        {
            "run_id": "issue31-operator-frontier-v3",
            "qualification_status": "qualified",
            "surface_id": (
                "surface://ascend-npu-23b93a89d5fecc79/matmul/square/"
                "2897f33121678193"
            ),
            "surface_version": "v-fab9fb985107087d",
        }
    ]


def test_public_verifier_rederives_domain_coverage_after_local_rehash(
    tmp_path: Path,
) -> None:
    run = TransformerMatmulFrontierBundleWriter().run(
        tmp_path,
        run_id="issue42-verifier-rederives-v1",
        transformer_run=FROZEN_DEMO,
        frontier_runs=(Q_PROJ_FRONTIER,),
    )

    def claim_projection_known(document: dict[str, object]) -> None:
        query = document["domain_queries"][-1]  # type: ignore[index]
        query["status"] = "known"
        query["latency_ns"] = 16_331.5
        query["effective_rate"] = 16_436_668_768_943.451
        query["reason_codes"] = []
        document["qualified_domain_count"] = 1
        document["coverage_fraction"] = 0.2

    _rewrite_artifact(
        run,
        "transformer-matmul-frontier-qualification",
        claim_projection_known,
    )

    verification = verify_run_bundle(run)
    assert verification["passed"] is False
    assert "Transformer MatMul Frontier derivation mismatch" in verification[
        "failures"
    ]


def test_public_cli_publishes_replayable_domain_qualification(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "qualify-transformer-matmul",
            str(FROZEN_DEMO),
            "--frontier-run",
            str(Q_PROJ_FRONTIER),
            "--artifact-store",
            str(tmp_path),
            "--run-id",
            "issue42-cli-v1",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary == {
        "coverage": "0/5",
        "hardware_cohort": "ascend-npu-23b93a89d5fecc79",
        "run_bundle": str(tmp_path / "runs/issue42-cli-v1"),
        "run_id": "issue42-cli-v1",
        "status": "unknown",
        "verification_passed": True,
    }
