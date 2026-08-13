from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import pytest

from groundupscale.run_bundle import verify_run_bundle
from groundupscale.transformer_matmul_frontier import (
    TransformerMatmulExactAnchorBundleWriter,
    TransformerMatmulFrontierBundleWriter,
    transformer_matmul_measurement_case,
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
ISSUE36_INCOMPLETE = (
    REPOSITORY_ROOT
    / "goal_process/issue-36-ascend-matmul-m-sweep/evidence"
    / "qualification-unknown.json"
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


def _copy_and_rehash_source_artifact(
    source: Path, role: str, mutate: object
) -> None:
    manifest_path = source / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item for item in manifest["artifacts"] if item["role"] == role
    )
    artifact_path = source / entry["path"]
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


def test_verifier_rejects_source_tamper_and_manifest_identity_forgery(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-demo"
    shutil.copytree(FROZEN_DEMO, source)
    run = TransformerMatmulFrontierBundleWriter().run(
        tmp_path / "derived",
        run_id="issue42-source-integrity-v1",
        transformer_run=source,
        frontier_runs=(),
    )

    cost_ir = source / "ir/cost.ir.json"
    document = json.loads(cost_ir.read_text(encoding="utf-8"))
    document["summary"]["metrics"]["flops"] += 1
    cost_ir.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert verify_run_bundle(run)["passed"] is False

    source = tmp_path / "source-demo-clean"
    shutil.copytree(FROZEN_DEMO, source)
    forged = TransformerMatmulFrontierBundleWriter().run(
        tmp_path / "forged",
        run_id="issue42-forged-identity-v1",
        transformer_run=source,
        frontier_runs=(),
    )
    manifest_path = forged / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "qualified"
    manifest["hardware_cohort"] = "ascend-npu-forged"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verification = verify_run_bundle(forged)
    assert verification["passed"] is False
    assert "Transformer MatMul Frontier identity mismatch" in verification[
        "failures"
    ]


def test_verifier_resolves_source_paths_from_bundle_not_process_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run = TransformerMatmulFrontierBundleWriter().run(
        tmp_path,
        run_id="issue42-relocatable-path-v1",
        transformer_run=FROZEN_DEMO,
        frontier_runs=(Q_PROJ_FRONTIER,),
    )

    monkeypatch.chdir(tmp_path)

    assert verify_run_bundle(run)["passed"] is True


def test_inventory_fails_closed_when_model_workload_or_semantic_sources_diverge(
    tmp_path: Path,
) -> None:
    semantic_source = tmp_path / "semantic-divergence"
    shutil.copytree(FROZEN_DEMO, semantic_source)

    def remove_one_semantic_matmul(document: dict[str, object]) -> None:
        def remove(items: list[dict[str, object]]) -> bool:
            for index, item in enumerate(items):
                if item.get("local_id") == "q_proj":
                    items.pop(index)
                    return True
                children = item.get("items")
                if isinstance(children, list) and remove(children):
                    return True
            return False

        assert remove(document["root"]["items"])  # type: ignore[index]

    _copy_and_rehash_source_artifact(
        semantic_source, "semantic-ir", remove_one_semantic_matmul
    )
    with pytest.raises(ValueError, match="four-source MatMul inventory mismatch"):
        TransformerMatmulFrontierBundleWriter().run(
            tmp_path / "semantic-output",
            run_id="issue42-semantic-divergence-v1",
            transformer_run=semantic_source,
            frontier_runs=(),
        )

    workload_source = tmp_path / "workload-divergence"
    shutil.copytree(FROZEN_DEMO, workload_source)

    def remove_model_call(document: dict[str, object]) -> None:
        document["documents"]["workload"]["spec"]["root"]["children"] = []  # type: ignore[index]

    _copy_and_rehash_source_artifact(
        workload_source, "resolved-input-lock", remove_model_call
    )
    with pytest.raises(ValueError, match="Workload Spec does not select the Model Spec"):
        TransformerMatmulFrontierBundleWriter().run(
            tmp_path / "workload-output",
            run_id="issue42-workload-divergence-v1",
            transformer_run=workload_source,
            frontier_runs=(),
        )


def _write_exact_anchor(
    path: Path, *, domain: dict[str, object], latency_ns: float
) -> Path:
    identity = domain["identity"]
    document = {
        "schema": (
            "groundupscale.dev/transformer-matmul-exact-anchor/v1alpha1"
        ),
        "evidence_id": "issue42-projection-anchor-test-v1",
        "status": "qualified",
        "response_target": "latency",
        "hardware_cohort": identity["hardware_cohort"],  # type: ignore[index]
        "domain_identity": identity,
        "domain_identity_digest": domain["domain_id"].split(":", 1)[1],  # type: ignore[union-attr]
        "latency_ns": latency_ns,
        "search_run_ids": ["issue42-search-1", "issue42-search-2", "issue42-search-3"],
        "holdout_run_ids": ["issue42-holdout-1", "issue42-holdout-2", "issue42-holdout-3"],
    }
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_exact_anchor_yields_known_latency_and_rate_is_only_derived(
    tmp_path: Path,
) -> None:
    inventory_run = TransformerMatmulFrontierBundleWriter().run(
        tmp_path / "inventory",
        run_id="issue42-exact-anchor-inventory-v1",
        transformer_run=FROZEN_DEMO,
        frontier_runs=(),
    )
    inventory = _artifact(inventory_run, "matmul-domain-inventory")
    projection = next(
        domain
        for domain in inventory["domains"]
        if domain["domain_class"] == "projection"
    )
    anchor = _write_exact_anchor(
        tmp_path / "projection-anchor.json",
        domain=projection,
        latency_ns=20_000.0,
    )

    run = TransformerMatmulFrontierBundleWriter().run(
        tmp_path / "qualified",
        run_id="issue42-exact-anchor-known-v1",
        transformer_run=FROZEN_DEMO,
        frontier_runs=(anchor,),
    )

    assert verify_run_bundle(run)["passed"] is True
    qualification = _artifact(run, "transformer-matmul-frontier-qualification")
    query = next(
        item
        for item in qualification["domain_queries"]
        if item["domain_class"] == "projection"
    )
    assert query["status"] == "known"
    assert query["latency_ns"] == 20_000.0
    assert query["effective_rate"] == 13_421_772_800_000.0
    assert query["effective_rate_derivation"] == (
        "declared_work_flop / latency_seconds"
    )
    assert query["reason_codes"] == []
    assert qualification["qualified_domain_count"] == 1


def test_issue36_incomplete_surface_is_unqualified_but_does_not_mask_anchor(
    tmp_path: Path,
) -> None:
    inventory_run = TransformerMatmulFrontierBundleWriter().run(
        tmp_path / "inventory",
        run_id="issue42-issue36-inventory-v1",
        transformer_run=FROZEN_DEMO,
        frontier_runs=(),
    )
    inventory = _artifact(inventory_run, "matmul-domain-inventory")
    projection = next(
        domain
        for domain in inventory["domains"]
        if domain["domain_class"] == "projection"
    )
    anchor = _write_exact_anchor(
        tmp_path / "projection-anchor.json",
        domain=projection,
        latency_ns=20_000.0,
    )

    run = TransformerMatmulFrontierBundleWriter().run(
        tmp_path / "qualified",
        run_id="issue42-issue36-does-not-mask-anchor-v1",
        transformer_run=FROZEN_DEMO,
        frontier_runs=(ISSUE36_INCOMPLETE, anchor),
    )

    assert verify_run_bundle(run)["passed"] is True
    qualification = _artifact(run, "transformer-matmul-frontier-qualification")
    projection_query = next(
        item
        for item in qualification["domain_queries"]
        if item["domain_class"] == "projection"
    )
    assert projection_query["status"] == "known"
    assert projection_query["considered_evidence"][0] == {
        "evidence_id": "evidence/qualifications/issue36-bounded-collection-corpus-incomplete-v1",
        "evidence_kind": "incomplete-surface",
        "qualification_status": "unknown",
        "surface_id": None,
        "surface_version": None,
    }


@pytest.mark.parametrize(
    ("domain_class", "left", "right", "result", "candidate"),
    [
        (
            "attention-qk",
            [1, 8, 512, 64],
            [1, 8, 64, 512],
            [1, 8, 512, 512],
            "torch.matmul",
        ),
        (
            "attention-context",
            [1, 8, 512, 512],
            [1, 8, 512, 64],
            [1, 512, 8, 64],
            "torch.matmul.transpose-1-2-contiguous",
        ),
    ],
)
def test_inventory_builds_exact_batched_transposed_measurement_contract(
    domain_class: str,
    left: list[int],
    right: list[int],
    result: list[int],
    candidate: str,
) -> None:
    case = transformer_matmul_measurement_case(
        FROZEN_DEMO,
        domain_class=domain_class,
        seed=20260813,
        warmup_iterations=100,
        repetitions=100,
        inner_iterations=100,
    )

    assert case["shape"] == {"left": left, "right": right, "result": result}
    assert case["candidate"] == candidate
    assert case["domain_identity_digest"]
    assert len(case["operand_storage_contracts"]) == 2
    if domain_class == "attention-qk":
        assert [
            item["storage_shape"] for item in case["operand_storage_contracts"]
        ] == [[1, 512, 8, 64], [1, 512, 8, 64]]
        assert [
            item["permutation"] for item in case["operand_storage_contracts"]
        ] == [[0, 2, 1, 3], [0, 2, 3, 1]]
    elif domain_class == "attention-context":
        assert case["operand_storage_contracts"][1]["storage_shape"] == [
            1,
            512,
            8,
            64,
        ]
        assert case["operand_storage_contracts"][1]["permutation"] == [
            0,
            2,
            1,
            3,
        ]
    assert case["result_transform"] == (
        {"permutation": [0, 2, 1, 3], "materialize_contiguous": True}
        if domain_class == "attention-context"
        else {"permutation": [0, 1, 2, 3], "materialize_contiguous": False}
    )


def _copy_domain_measurement(
    destination: Path,
    *,
    run_id: str,
    process_id: int,
    domain: dict[str, object],
    median_ns: int,
) -> Path:
    source = (
        REPOSITORY_ROOT
        / "goal_process/issue-31-ascend-matmul-frontier/evidence/runs"
        / "issue31-search-v4-s512-torch-matmul-01"
    )
    shutil.copytree(source, destination)
    manifest_path = destination / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = run_id
    identity = domain["identity"]
    domain_record = {
        "identity": identity,
        "identity_digest": domain["domain_id"].split(":", 1)[1],  # type: ignore[union-attr]
        "declared_work_flop": domain["declared_work_flop"],
    }

    def rewrite(role: str, mutate) -> None:
        entry = next(item for item in manifest["artifacts"] if item["role"] == role)
        path = destination / entry["path"]
        document = json.loads(path.read_text(encoding="utf-8"))
        mutate(document)
        payload = (
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        path.write_bytes(payload)
        entry["sha256"] = sha256(payload).hexdigest()

    rewrite(
        "benchmark-case",
        lambda value: value.update(
            {
                "domain_identity": identity,
                "domain_identity_digest": domain_record["identity_digest"],
                "declared_work_flop": domain["declared_work_flop"],
            }
        ),
    )

    def candidate(value: dict[str, object]) -> None:
        value["candidate_family"] = identity["candidate_family"]  # type: ignore[index]
        value["execution_mode"] = identity["execution_mode"]  # type: ignore[index]
        value["transformer_matmul_domain"] = domain_record
        value.pop("candidate_digest", None)
        from groundupscale.ir import content_fingerprint

        value["candidate_digest"] = content_fingerprint(value)

    rewrite("candidate-identity", candidate)
    rewrite(
        "execution-contract",
        lambda value: value.update({"transformer_matmul_domain": domain_record}),
    )
    rewrite(
        "environment",
        lambda value: value["measurement_session"].update(
            {
                "session_id": run_id,
                "process_id": process_id,
                "process_started_at": f"2026-08-13T00:00:{process_id % 60:02d}+00:00",
            }
        ),
    )

    def timing(value: dict[str, object]) -> None:
        samples = [median_ns] * 100
        value["samples"] = samples
        value["summary"] = {
            "count": 100,
            "minimum": median_ns,
            "p10": float(median_ns),
            "q1": float(median_ns),
            "median": float(median_ns),
            "q3": float(median_ns),
            "p90": float(median_ns),
            "maximum": median_ns,
            "iqr": 0.0,
            "iqr_fraction_of_median": 0.0,
            "median_absolute_deviation": 0.0,
            "mad_fraction_of_median": 0.0,
        }

    rewrite("raw-timing-observation", timing)
    rewrite(
        "measurement-collection",
        lambda value: value.update(
            {
                "timing_quality": {
                    **value["timing_quality"],
                    "observed_iqr_fraction_of_median": 0.0,
                    "timer_resolution_fraction_of_median": 0.2 / median_ns,
                }
            }
        ),
    )
    # Keep the aggregate collection artifact and validity manifest replayable.
    for component_role, component_key in (
        ("candidate-identity", "candidate_identity"),
        ("execution-contract", "execution_contract"),
        ("raw-timing-observation", "raw_timing"),
    ):
        component_entry = next(
            item for item in manifest["artifacts"] if item["role"] == component_role
        )
        component = json.loads(
            (destination / component_entry["path"]).read_text(encoding="utf-8")
        )
        rewrite(
            "measurement-collection",
            lambda value, key=component_key, item=component: value.update({key: item}),
        )
    # measurement-collection was rewritten after earlier component snapshots;
    # refresh its digest last.
    collection_entry = next(
        item
        for item in manifest["artifacts"]
        if item["role"] == "measurement-collection"
    )
    collection_path = destination / collection_entry["path"]
    collection_entry["sha256"] = sha256(collection_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def test_exact_anchor_bundle_rederives_disjoint_search_holdout_sessions(
    tmp_path: Path,
) -> None:
    inventory_run = TransformerMatmulFrontierBundleWriter().run(
        tmp_path / "inventory",
        run_id="issue42-anchor-source-inventory-v1",
        transformer_run=FROZEN_DEMO,
        frontier_runs=(),
    )
    inventory = _artifact(inventory_run, "matmul-domain-inventory")
    projection = next(
        item for item in inventory["domains"] if item["domain_class"] == "projection"
    )
    search = [
        _copy_domain_measurement(
            tmp_path / f"search-{index}",
            run_id=f"issue42-search-{index}",
            process_id=100 + index,
            domain=projection,
            median_ns=20_000 + index,
        )
        for index in range(1, 4)
    ]
    holdout = [
        _copy_domain_measurement(
            tmp_path / f"holdout-{index}",
            run_id=f"issue42-holdout-{index}",
            process_id=200 + index,
            domain=projection,
            median_ns=20_002 + index,
        )
        for index in range(1, 4)
    ]

    anchor_run = TransformerMatmulExactAnchorBundleWriter().run(
        tmp_path / "anchors",
        run_id="issue42-projection-exact-anchor-v1",
        search_runs=search,
        holdout_runs=holdout,
    )

    assert verify_run_bundle(anchor_run)["passed"] is True
    anchor = _artifact(anchor_run, "transformer-matmul-exact-anchor")
    assert anchor["latency_ns"] == 20_004.0
    assert anchor["search_run_ids"] == [f"issue42-search-{index}" for index in range(1, 4)]
    assert anchor["holdout_run_ids"] == [f"issue42-holdout-{index}" for index in range(1, 4)]

    frontier_run = TransformerMatmulFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="issue42-projection-known-from-run-bundle-v1",
        transformer_run=FROZEN_DEMO,
        frontier_runs=(anchor_run,),
    )
    qualification = _artifact(frontier_run, "transformer-matmul-frontier-qualification")
    projection_query = next(
        item for item in qualification["domain_queries"] if item["domain_class"] == "projection"
    )
    assert projection_query["status"] == "known"
    assert projection_query["latency_ns"] == 20_004.0


def test_unrepeatable_holdout_publishes_replayable_structured_unknown(
    tmp_path: Path,
) -> None:
    inventory_run = TransformerMatmulFrontierBundleWriter().run(
        tmp_path / "inventory",
        run_id="issue42-unknown-anchor-inventory-v1",
        transformer_run=FROZEN_DEMO,
        frontier_runs=(),
    )
    inventory = _artifact(inventory_run, "matmul-domain-inventory")
    projection = next(
        item for item in inventory["domains"] if item["domain_class"] == "projection"
    )
    search = [
        _copy_domain_measurement(
            tmp_path / f"search-unknown-{index}",
            run_id=f"issue42-search-unknown-{index}",
            process_id=300 + index,
            domain=projection,
            median_ns=20_000,
        )
        for index in range(1, 4)
    ]
    holdout = [
        _copy_domain_measurement(
            tmp_path / f"holdout-unknown-{index}",
            run_id=f"issue42-holdout-unknown-{index}",
            process_id=400 + index,
            domain=projection,
            median_ns=value,
        )
        for index, value in enumerate((20_000, 20_000, 30_000), 1)
    ]

    anchor_run = TransformerMatmulExactAnchorBundleWriter().run(
        tmp_path / "anchors",
        run_id="issue42-projection-unknown-anchor-v1",
        search_runs=search,
        holdout_runs=holdout,
    )

    assert verify_run_bundle(anchor_run)["passed"] is True
    anchor = _artifact(anchor_run, "transformer-matmul-exact-anchor")
    assert anchor["status"] == "unknown"
    assert anchor["latency_ns"] is None
    assert anchor["reason_codes"] == ["independent-holdout-repeatability-failed"]
    assert anchor["repeatability"]["holdout_relative_range"] == 0.5
    assert anchor["repeatability"]["maximum_relative_range"] == 0.1

    frontier = TransformerMatmulFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="issue42-projection-structured-unknown-v1",
        transformer_run=FROZEN_DEMO,
        frontier_runs=(anchor_run,),
    )
    qualification = _artifact(frontier, "transformer-matmul-frontier-qualification")
    projection_query = next(
        item for item in qualification["domain_queries"] if item["domain_class"] == "projection"
    )
    assert projection_query["status"] == "unknown"
    assert "exact-anchor-not-qualified" in projection_query["reason_codes"]
    assert projection_query["considered_evidence"][0]["qualification_status"] == "unknown"
