from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from statistics import median, quantiles

import yaml
import pytest
from pydantic import ValidationError

from groundupscale.cli import main
from groundupscale.ir import content_fingerprint
from groundupscale.schemas.v1alpha1 import ExactShapeOperatorFrontierAnchor


Q_PROJ_STABLE_PATH = "semantic/demo/layer_0/attention/q_proj"


def test_exact_frontier_anchor_rejects_incomplete_or_negative_strides() -> None:
    profile = yaml.safe_load(
        (
            Path(__file__).parents[1]
            / "specs/operator-frontiers/apple-m4-cpu-layer1-qk-matmul-v1.yaml"
        ).read_text(encoding="utf-8")
    )
    anchor = profile["spec"]["anchors"][0]

    incomplete = json.loads(json.dumps(anchor))
    incomplete["operand_strides"][0] = [1]
    with pytest.raises(ValidationError, match="stride rank"):
        ExactShapeOperatorFrontierAnchor.model_validate(incomplete)

    negative = json.loads(json.dumps(anchor))
    negative["result_stride"][-1] = -1
    with pytest.raises(ValidationError, match="stride values"):
        ExactShapeOperatorFrontierAnchor.model_validate(negative)

    contradictory = json.loads(json.dumps(anchor))
    contradictory["result_stride"] = [999999, 262144, 512, 1]
    with pytest.raises(ValidationError, match="contiguous stride"):
        ExactShapeOperatorFrontierAnchor.model_validate(contradictory)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_session_bundle(
    root: Path,
    *,
    run_id: str,
    process_id: int,
    median_ns: float,
    mixed_operand_layouts: bool = False,
) -> Path:
    bundle = root / run_id
    artifacts: list[dict[str, object]] = []

    def write(role: str, relative: str, value: object, schema: str) -> None:
        payload = _json_bytes(value)
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts.append(
            {
                "role": role,
                "path": relative,
                "sha256": sha256(payload).hexdigest(),
                "schema": schema,
            }
        )

    samples = [median_ns + float((index % 5) - 2) * 20.0 for index in range(20)]
    quartiles = quantiles(samples, n=4, method="inclusive")
    sample_median = float(median(samples))
    candidate = {
        "schema": "groundupscale.dev/operator-candidate-identity/v1alpha1",
        "status": "resolved",
        "family": "torch.matmul.cpu.fp32",
        "provider": "pytorch",
        "dispatch_provider_binaries": [
            {
                "role": "cpu-dispatch-provider",
                "name": "libtorch_cpu.dylib",
                "sha256": "a" * 64,
            }
        ],
    }
    candidate["candidate_digest"] = content_fingerprint(candidate)
    input_corpus = {
        "schema": "groundupscale.dev/input-corpus-identity/v1alpha1",
        "status": "resolved",
        "tensor_sha256": ["d" * 64, "f" * 64],
    }
    input_corpus["input_corpus_digest"] = content_fingerprint(input_corpus)
    execution_contract = {
        "schema": "groundupscale.dev/operator-execution-contract/v1alpha1",
        "status": "resolved",
        "operand_contracts": [
            {
                "shape": [1, 512, 512],
                "stride": [262144, 512, 1],
                "dtype": "float32",
                "layout": "row-major-contiguous",
                "minimum_alignment_bytes": 64,
            },
            {
                "shape": [512, 512],
                "stride": [1, 512] if mixed_operand_layouts else [512, 1],
                "dtype": "float32",
                "layout": (
                    "strided"
                    if mixed_operand_layouts
                    else "row-major-contiguous"
                ),
                "minimum_alignment_bytes": 64,
            },
        ],
        "result_contract": {
            "shape": [1, 512, 512],
            "stride": [262144, 512, 1],
            "dtype": "float32",
            "layout": "row-major-contiguous",
            "minimum_alignment_bytes": 64,
        },
        "execution_mode": "eager",
        "cache_state": "warm-reused-inputs-and-weights",
        "working_set_bytes": 3_145_728,
        "concurrency": "single-operator-no-overlap",
    }
    execution_contract["execution_contract_digest"] = content_fingerprint(
        execution_contract
    )
    observed_hardware = {
        "status": "resolved",
        "source": "sysctl",
        "model": "Mac16,12",
        "cpu_brand": "Apple M4",
        "physical_cpu": 10,
        "logical_cpu": 10,
        "performance_levels": {"performance_cores": 4, "efficiency_cores": 6},
    }
    cohort = {
        "schema": "groundupscale.dev/hardware-validity-cohort/v1alpha1",
        "device": {
            "hardware": ["apple-m4"],
            "device": "cpu",
            "observed_identity": observed_hardware,
            "identity_sha256": content_fingerprint(observed_hardware),
        },
        "software": {"torch": "test"},
        "numeric_execution": {"threads": 10, "interop_threads": 1},
        "power_clock": {"policy_id": "local-apple-silicon-v2"},
        "timing": {"measurement_protocol": {"version": "1.0.0"}},
    }
    cohort["cohort_id"] = f"hvc-{content_fingerprint(cohort)}"
    correctness_record = {
        "schema": "groundupscale.dev/operator-correctness-evidence/v1alpha1",
        "status": "passed",
        "candidate_family": "torch.matmul.cpu.fp32",
        "candidate_digest": candidate["candidate_digest"],
        "input_corpus_digest": input_corpus["input_corpus_digest"],
        "oracle": {
            "policy_id": "matmul-fp32-float64-oracle-v1",
            "version": "1.0.0",
            "provider": "torch.float64.matmul",
            "atol": 1e-5,
            "rtol": 1e-4,
            "accumulation_dtype": "float64",
            "invariants": ["shape-exact", "finite-output"],
        },
        "max_absolute_error": 0.0,
        "max_relative_error": 0.0,
        "shape_matches": True,
        "finite": True,
        "actual_output_sha256": "a" * 64,
        "reference_output_sha256": "f" * 64,
    }
    write(
        "environment",
        "resolved/environment.json",
        {
            "schema": "groundupscale.dev/environment/v1alpha1",
            "device": "cpu",
            "torch": {"num_threads": 10},
            "hardware_validity_cohort": cohort,
            "process": {"pid": process_id, "session_id": run_id},
            "measurement_preflight": {
                "schema": "groundupscale.dev/environment-validity/v1alpha1",
                "eligible": True,
                "reason_codes": [],
                "policy": {"policy_id": "local-apple-silicon-v2"},
            },
        },
        "groundupscale.dev/environment/v1alpha1",
    )
    write(
        "resolved-input-lock",
        "resolved/inputs.lock.json",
        {
            "schema": "groundupscale.dev/resolved-input-lock/v1alpha1",
            "documents": {
                "hardware_capability_profiles": [
                    {
                        "spec": {
                            "target": {"hardware": "apple-m4", "device": "cpu"},
                            "hardware_cohort": "apple-m4-exact-cohort",
                        }
                    }
                ]
            },
        },
        "groundupscale.dev/resolved-input-lock/v1alpha1",
    )
    write(
        "cost-ir",
        "ir/cost.ir.json",
        {
            "schema": "groundupscale.dev/cost-ir/v1alpha2",
            "root": {
                "items": [
                    {
                        "stable_path": "cost/semantic/demo/layer_0/attention/q_proj",
                        "operation": "MatMul",
                        "operand_types": [
                            {
                                "dtype": "float32",
                                "shape": [1, 512, 512],
                                "layout": "contiguous",
                            },
                            {
                                "dtype": "float32",
                                "shape": [512, 512],
                                "layout": (
                                    "transposed-strided"
                                    if mixed_operand_layouts
                                    else "contiguous"
                                ),
                            },
                        ],
                        "result_types": [
                            {
                                "dtype": "float32",
                                "shape": [1, 512, 512],
                                "layout": "contiguous",
                            }
                        ],
                    }
                ]
            },
        },
        "groundupscale.dev/cost-ir/v1alpha2",
    )
    write(
        "benchmark-observation",
        "observation/raw/benchmark.json",
        {
            "schema": "groundupscale.dev/benchmark-observation/v1alpha1",
            "device": "cpu",
            "torch_num_threads": 10,
            "cases": [
                {
                    "case_id": "matmul-q-proj",
                    "resolved_scope": "semantic/demo/layer_0/attention/q_proj",
                    "samples": len(samples),
                    "candidate_identity": candidate,
                    "input_corpus": input_corpus,
                    "execution_contract": execution_contract,
                    "operator_correctness": correctness_record,
                    "warmup_convergence": {
                        "policy": {
                            "policy_id": "local-m4-exact-shape-timing-v1",
                            "version": "1.0.0",
                            "minimum_warmup_iterations": 500,
                            "convergence_window_count": 7,
                            "convergence_iterations_per_window": 20,
                            "maximum_warmup_median_drift": 0.05,
                            "minimum_samples": 20,
                            "minimum_windows_per_sample": 5,
                            "minimum_timed_duration_ns": 100_000_000,
                        },
                        "warmup_iterations": 500,
                        "window_samples_ns": [median_ns] * 7,
                        "median_drift": 0.0,
                        "converged": True,
                    },
                    "timing_contract": {
                        "policy": {
                            "policy_id": "local-m4-exact-shape-timing-v1",
                            "version": "1.0.0",
                            "minimum_warmup_iterations": 500,
                            "convergence_window_count": 7,
                            "convergence_iterations_per_window": 20,
                            "maximum_warmup_median_drift": 0.05,
                            "minimum_samples": 20,
                            "minimum_windows_per_sample": 5,
                            "minimum_timed_duration_ns": 100_000_000,
                        },
                        "timing_scope": "host_visible_completion",
                        "completion_boundary": "synchronous-cpu-call-return",
                        "timer": {
                            "source": "time.perf_counter_ns",
                            "monotonic": True,
                            "resolution_ns": 1.0,
                        },
                        "instrumentation_profile": "benchmark",
                        "exclusions": [],
                    },
                    "latency": {
                        "samples_ns": samples,
                        "window_samples_ns": [
                            [sample * 1000.0] * 5 for sample in samples
                        ],
                        "normalized_window_samples_ns": [
                            [sample] * 5
                            for sample in samples
                        ],
                        "inner_iterations": 1000,
                        "windows_per_sample": 5,
                        "median_ns": sample_median,
                        "q1_ns": float(quartiles[0]),
                        "q3_ns": float(quartiles[2]),
                        "iqr_over_median": float(
                            (quartiles[2] - quartiles[0]) / sample_median
                        ),
                    },
                }
            ],
        },
        "groundupscale.dev/benchmark-observation/v1alpha1",
    )
    write(
        "correctness-observation",
        "observation/correctness.json",
        {
            "schema": "groundupscale.dev/correctness-observation/v1alpha2",
            "passed": True,
            "operator_cases": [
                {
                    "case_id": "matmul-q-proj",
                    "stable_path": "semantic/demo/layer_0/attention/q_proj",
                    "candidate_identity": candidate,
                    "input_corpus": input_corpus,
                    "execution_contract": execution_contract,
                    "correctness": correctness_record,
                }
            ],
        },
        "groundupscale.dev/correctness-observation/v1alpha2",
    )
    manifest = {
        "schema": "groundupscale.dev/run-manifest/v1alpha1",
        "run_id": run_id,
        "status": "completed",
        "device": "cpu",
        "hardware_cohort": cohort["cohort_id"],
        "environment_validity": "passed",
        "artifacts": artifacts,
    }
    (bundle / "run.manifest.json").write_bytes(_json_bytes(manifest))
    return bundle


def _rewrite_artifact(bundle: Path, role: str, mutate) -> None:
    manifest_path = bundle / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(item for item in manifest["artifacts"] if item["role"] == role)
    path = bundle / artifact["path"]
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    payload = _json_bytes(value)
    path.write_bytes(payload)
    artifact["sha256"] = sha256(payload).hexdigest()
    manifest_path.write_bytes(_json_bytes(manifest))


def test_qualify_frontier_rederives_exact_shape_anchor_from_independent_holdouts(
    tmp_path: Path, capsys
) -> None:
    search = [
        _write_session_bundle(
            tmp_path, run_id=f"search-{index}", process_id=100 + index, median_ns=value
        )
        for index, value in enumerate((158_000.0, 154_000.0, 156_000.0), 1)
    ]
    holdout = [
        _write_session_bundle(
            tmp_path, run_id=f"holdout-{index}", process_id=200 + index, median_ns=value
        )
        for index, value in enumerate((156_000.0, 154_000.0, 158_000.0), 1)
    ]
    observation = tmp_path / "evidence/operator-frontier.json"
    profile = tmp_path / "specs/operator-frontiers/matmul-512.yaml"
    argv = [
        "qualify-frontier",
        "--case-id",
        "matmul-q-proj",
        "--stable-path-pattern",
        Q_PROJ_STABLE_PATH,
        "--candidate-family",
        "torch.matmul.cpu.fp32",
        "--profile-name",
        "apple-m4-cpu-matmul-512",
        "--profile-version",
        "1.0.0",
        "--observation-output",
        str(observation),
        "--profile-output",
        str(profile),
        "--json",
    ]
    for path in search:
        argv.extend(("--search-run", str(path)))
    for path in holdout:
        argv.extend(("--holdout-run", str(path)))

    exit_code = main(argv)

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "QUALIFIED"
    source = json.loads(observation.read_text(encoding="utf-8"))
    anchor = source["anchors"][0]
    assert anchor["latency_ns"] == 156_000.0
    assert anchor["standard_uncertainty_ns"] == 2_000.0
    assert anchor["search_run_ids"] == ["search-1", "search-2", "search-3"]
    assert anchor["holdout_run_ids"] == ["holdout-1", "holdout-2", "holdout-3"]
    assert anchor["operand_shapes"] == [[1, 512, 512], [512, 512]]
    assert anchor["result_shape"] == [1, 512, 512]
    assert anchor["session_evidence"]["holdout"][0]["median_ns"] == 156_000.0
    document = yaml.safe_load(profile.read_text(encoding="utf-8"))
    assert document["kind"] == "OperatorFrontierProfile"
    assert document["spec"]["source"]["sha256"] == sha256(
        observation.read_bytes()
    ).hexdigest()
    assert document["spec"]["anchors"] == source["anchors"]


def test_qualify_frontier_preserves_mixed_operand_layouts(
    tmp_path: Path, capsys
) -> None:
    search = [
        _write_session_bundle(
            tmp_path,
            run_id=f"mixed-search-{index}",
            process_id=300 + index,
            median_ns=580_000.0 + index * 100.0,
            mixed_operand_layouts=True,
        )
        for index in range(1, 4)
    ]
    holdout = [
        _write_session_bundle(
            tmp_path,
            run_id=f"mixed-holdout-{index}",
            process_id=400 + index,
            median_ns=580_000.0 + index * 100.0,
            mixed_operand_layouts=True,
        )
        for index in range(1, 4)
    ]
    observation = tmp_path / "evidence/mixed-layout-frontier.json"
    profile = tmp_path / "specs/operator-frontiers/mixed-layout-frontier.yaml"
    argv = [
        "qualify-frontier",
        "--case-id",
        "matmul-q-proj",
        "--stable-path-pattern",
        Q_PROJ_STABLE_PATH,
        "--candidate-family",
        "torch.matmul.cpu.fp32",
        "--profile-name",
        "apple-m4-cpu-mixed-layout-matmul",
        "--profile-version",
        "1.0.0",
        "--observation-output",
        str(observation),
        "--profile-output",
        str(profile),
        "--json",
    ]
    for path in search:
        argv.extend(("--search-run", str(path)))
    for path in holdout:
        argv.extend(("--holdout-run", str(path)))

    assert main(argv) == 0
    capsys.readouterr()
    anchor = json.loads(observation.read_text(encoding="utf-8"))["anchors"][0]
    assert anchor["layout"] == "mixed-explicit"
    assert anchor["operand_layouts"] == [
        "row-major-contiguous",
        "strided",
    ]
    assert anchor["result_layout"] == "row-major-contiguous"
    assert anchor["operand_strides"] == [[262144, 512, 1], [1, 512]]


def test_qualify_frontier_fails_closed_when_holdout_reuses_a_search_process(
    tmp_path: Path, capsys
) -> None:
    search = [
        _write_session_bundle(
            tmp_path, run_id=f"search-{index}", process_id=100 + index, median_ns=155_000.0
        )
        for index in range(1, 4)
    ]
    holdout = [
        _write_session_bundle(
            tmp_path,
            run_id=f"holdout-{index}",
            process_id=101 if index == 1 else 200 + index,
            median_ns=155_000.0,
        )
        for index in range(1, 4)
    ]
    observation = tmp_path / "frontier.json"
    profile = tmp_path / "frontier.yaml"
    argv = [
        "qualify-frontier",
        "--case-id",
        "matmul-q-proj",
        "--stable-path-pattern",
        Q_PROJ_STABLE_PATH,
        "--candidate-family",
        "torch.matmul.cpu.fp32",
        "--profile-name",
        "apple-m4-cpu-matmul-512",
        "--profile-version",
        "1.0.0",
        "--observation-output",
        str(observation),
        "--profile-output",
        str(profile),
        "--json",
    ]
    for path in search:
        argv.extend(("--search-run", str(path)))
    for path in holdout:
        argv.extend(("--holdout-run", str(path)))

    exit_code = main(argv)

    assert exit_code == 2
    rejection = json.loads(capsys.readouterr().out)
    assert rejection["status"] == "insufficient_evidence"
    assert rejection["reason_code"] == "frontier-sessions-not-independent"
    assert not observation.exists()
    assert not profile.exists()


def test_qualify_frontier_rejects_correctness_without_exact_candidate_oracle(
    tmp_path: Path, capsys
) -> None:
    search = [
        _write_session_bundle(
            tmp_path, run_id=f"search-{index}", process_id=100 + index, median_ns=155_000.0
        )
        for index in range(1, 4)
    ]
    holdout = [
        _write_session_bundle(
            tmp_path, run_id=f"holdout-{index}", process_id=200 + index, median_ns=155_000.0
        )
        for index in range(1, 4)
    ]
    for bundle in (*search, *holdout):
        _rewrite_artifact(
            bundle,
            "correctness-observation",
            lambda value: value.pop("operator_cases"),
        )
    argv = _qualification_argv(tmp_path, search=search, holdout=holdout)

    exit_code = main(argv)

    assert exit_code == 2
    rejection = json.loads(capsys.readouterr().out)
    assert rejection["reason_code"] == "frontier-correctness-evidence-invalid"


def test_qualify_frontier_rejects_caller_claimed_candidate_family(
    tmp_path: Path, capsys
) -> None:
    search = [
        _write_session_bundle(
            tmp_path, run_id=f"search-{index}", process_id=100 + index, median_ns=155_000.0
        )
        for index in range(1, 4)
    ]
    holdout = [
        _write_session_bundle(
            tmp_path, run_id=f"holdout-{index}", process_id=200 + index, median_ns=155_000.0
        )
        for index in range(1, 4)
    ]
    argv = _qualification_argv(
        tmp_path,
        search=search,
        holdout=holdout,
        candidate_family="caller.fabricated.family",
    )

    exit_code = main(argv)

    assert exit_code == 2
    rejection = json.loads(capsys.readouterr().out)
    assert rejection["reason_code"] == "frontier-candidate-identity-mismatch"


def test_qualify_frontier_rejects_missing_timing_and_warmup_contract(
    tmp_path: Path, capsys
) -> None:
    search = [
        _write_session_bundle(
            tmp_path, run_id=f"search-{index}", process_id=100 + index, median_ns=155_000.0
        )
        for index in range(1, 4)
    ]
    holdout = [
        _write_session_bundle(
            tmp_path, run_id=f"holdout-{index}", process_id=200 + index, median_ns=155_000.0
        )
        for index in range(1, 4)
    ]
    for bundle in (*search, *holdout):
        _rewrite_artifact(
            bundle,
            "benchmark-observation",
            lambda value: value["cases"][0].pop("timing_contract"),
        )
    argv = _qualification_argv(tmp_path, search=search, holdout=holdout)

    exit_code = main(argv)

    assert exit_code == 2
    rejection = json.loads(capsys.readouterr().out)
    assert rejection["reason_code"] == "frontier-timing-evidence-invalid"


def test_qualify_frontier_rejects_unrepeatable_holdout_sessions(
    tmp_path: Path, capsys
) -> None:
    search = [
        _write_session_bundle(
            tmp_path, run_id=f"search-{index}", process_id=100 + index, median_ns=155_000.0
        )
        for index in range(1, 4)
    ]
    holdout = [
        _write_session_bundle(
            tmp_path,
            run_id=f"holdout-{index}",
            process_id=200 + index,
            median_ns=value,
        )
        for index, value in enumerate((120_000.0, 155_000.0, 200_000.0), 1)
    ]
    argv = _qualification_argv(tmp_path, search=search, holdout=holdout)

    exit_code = main(argv)

    assert exit_code == 2
    rejection = json.loads(capsys.readouterr().out)
    assert rejection["reason_code"] == "frontier-session-repeatability-failed"


def test_qualify_frontier_rejects_wildcard_path_for_exact_input_corpus(
    tmp_path: Path, capsys
) -> None:
    search = [
        _write_session_bundle(
            tmp_path, run_id=f"search-{index}", process_id=100 + index, median_ns=155_000.0
        )
        for index in range(1, 4)
    ]
    holdout = [
        _write_session_bundle(
            tmp_path, run_id=f"holdout-{index}", process_id=200 + index, median_ns=155_000.0
        )
        for index in range(1, 4)
    ]
    argv = _qualification_argv(tmp_path, search=search, holdout=holdout)
    pattern_index = argv.index("--stable-path-pattern") + 1
    argv[pattern_index] = "*/attention/q_proj"

    exit_code = main(argv)

    assert exit_code == 2
    rejection = json.loads(capsys.readouterr().out)
    assert rejection["reason_code"] == "frontier-input-corpus-scope-mismatch"


def test_qualify_frontier_recomputes_embedded_candidate_domain_digests(
    tmp_path: Path, capsys
) -> None:
    search = [
        _write_session_bundle(
            tmp_path, run_id=f"search-{index}", process_id=100 + index, median_ns=155_000.0
        )
        for index in range(1, 4)
    ]
    holdout = [
        _write_session_bundle(
            tmp_path, run_id=f"holdout-{index}", process_id=200 + index, median_ns=155_000.0
        )
        for index in range(1, 4)
    ]

    def tamper_candidate(value: dict) -> None:
        case = value["cases"][0]
        case["candidate_identity"]["provider"] = "tampered-provider"

    def tamper_correctness(value: dict) -> None:
        record = value["operator_cases"][0]
        record["candidate_identity"]["provider"] = "tampered-provider"

    for bundle in (*search, *holdout):
        _rewrite_artifact(bundle, "benchmark-observation", tamper_candidate)
        _rewrite_artifact(bundle, "correctness-observation", tamper_correctness)

    exit_code = main(_qualification_argv(tmp_path, search=search, holdout=holdout))

    assert exit_code == 2
    rejection = json.loads(capsys.readouterr().out)
    assert rejection["reason_code"] == "frontier-candidate-identity-mismatch"


def test_qualify_frontier_requires_replayable_observed_m4_identity(
    tmp_path: Path, capsys
) -> None:
    search = [
        _write_session_bundle(
            tmp_path, run_id=f"search-{index}", process_id=100 + index, median_ns=155_000.0
        )
        for index in range(1, 4)
    ]
    holdout = [
        _write_session_bundle(
            tmp_path, run_id=f"holdout-{index}", process_id=200 + index, median_ns=155_000.0
        )
        for index in range(1, 4)
    ]
    for bundle in (*search, *holdout):
        manifest_path = bundle / "run.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = next(
            item for item in manifest["artifacts"] if item["role"] == "environment"
        )
        path = bundle / artifact["path"]
        environment = json.loads(path.read_text(encoding="utf-8"))
        cohort = environment["hardware_validity_cohort"]
        cohort["device"].pop("observed_identity")
        cohort["device"].pop("identity_sha256")
        cohort_body = {key: value for key, value in cohort.items() if key != "cohort_id"}
        cohort["cohort_id"] = f"hvc-{content_fingerprint(cohort_body)}"
        payload = _json_bytes(environment)
        path.write_bytes(payload)
        artifact["sha256"] = sha256(payload).hexdigest()
        manifest["hardware_cohort"] = cohort["cohort_id"]
        manifest_path.write_bytes(_json_bytes(manifest))

    exit_code = main(_qualification_argv(tmp_path, search=search, holdout=holdout))

    assert exit_code == 2
    rejection = json.loads(capsys.readouterr().out)
    assert rejection["reason_code"] == "frontier-hardware-cohort-invalid"


def _qualification_argv(
    tmp_path: Path,
    *,
    search: list[Path],
    holdout: list[Path],
    candidate_family: str = "torch.matmul.cpu.fp32",
) -> list[str]:
    argv = [
        "qualify-frontier",
        "--case-id",
        "matmul-q-proj",
        "--stable-path-pattern",
        Q_PROJ_STABLE_PATH,
        "--candidate-family",
        candidate_family,
        "--profile-name",
        "apple-m4-cpu-matmul-512",
        "--profile-version",
        "2.0.0",
        "--observation-output",
        str(tmp_path / "frontier.json"),
        "--profile-output",
        str(tmp_path / "frontier.yaml"),
        "--json",
    ]
    for path in search:
        argv.extend(("--search-run", str(path)))
    for path in holdout:
        argv.extend(("--holdout-run", str(path)))
    return argv
