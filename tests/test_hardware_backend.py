from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest
import yaml

from groundupscale.benchmark.comparison import build_prediction_observation_comparison
from groundupscale.ir import content_fingerprint
from groundupscale.pipeline import compile_analysis_bundle, compile_analysis_plan
from groundupscale.run_bundle import RunBundleWriter
from groundupscale.specs import SpecRepository


REPOSITORY_ROOT = Path(__file__).parents[1]
Q_PROJ_STABLE_PATH = (
    "semantic/workload/transformer-prefill/request/model-prefill/model/"
    "transformer/layer_0/attention/q_proj"
)


def _resolved_test_candidate() -> dict[str, object]:
    candidate: dict[str, object] = {
        "schema": "groundupscale.dev/operator-candidate-identity/v1alpha1",
        "status": "resolved",
        "family": "torch.matmul.cpu.fp32",
        "provider": "pytorch-test",
    }
    candidate["candidate_digest"] = content_fingerprint(candidate)
    return candidate


def _resolved_test_input_corpus() -> dict[str, object]:
    corpus: dict[str, object] = {
        "schema": "groundupscale.dev/input-corpus-identity/v1alpha1",
        "status": "resolved",
        "seed": 42,
        "tensor_sha256": ["d" * 64, "f" * 64],
    }
    corpus["input_corpus_digest"] = content_fingerprint(corpus)
    return corpus


def _repository_with_exact_matmul_frontier(tmp_path: Path) -> Path:
    shutil.copytree(REPOSITORY_ROOT / "specs", tmp_path / "specs")
    capability_profile = yaml.safe_load(
        (tmp_path / "specs/hardware-capabilities/apple-m4-cpu-local.yaml").read_text(
            encoding="utf-8"
        )
    )
    capability_source = capability_profile["spec"]["source"]["path"]
    source_target = tmp_path / capability_source
    source_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPOSITORY_ROOT / capability_source, source_target)

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
                "stride": [512, 1],
                "dtype": "float32",
                "layout": "row-major-contiguous",
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
        "working_set_bytes": 3145728,
        "concurrency": "single-operator-no-overlap",
    }
    execution_contract["execution_contract_digest"] = content_fingerprint(
        execution_contract
    )
    execution_digest = execution_contract["execution_contract_digest"]
    candidate_digest = _resolved_test_candidate()["candidate_digest"]
    input_corpus_digest = _resolved_test_input_corpus()["input_corpus_digest"]
    observation = {
        "schema": "groundupscale.dev/operator-frontier-observation/v1alpha1",
        "target": {"hardware": "apple-m4", "device": "cpu"},
        "hardware_cohort": "hvc-test-exact-matmul",
        "anchors": [
            {
                "anchor_id": "m4-cpu-matmul-512-v1",
                "case_id": "matmul-q-proj",
                "stable_path_pattern": Q_PROJ_STABLE_PATH,
                "semantic_operation": "MatMul",
                "operand_shapes": [[1, 512, 512], [512, 512]],
                "result_shape": [1, 512, 512],
                "dtype": "float32",
                "layout": "row-major-contiguous",
                "operand_strides": [[262144, 512, 1], [512, 1]],
                "result_stride": [262144, 512, 1],
                "minimum_alignment_bytes": 64,
                "working_set_bytes": 3145728,
                "threads": 10,
                "interop_threads": 1,
                "execution_mode": "eager",
                "candidate_family": "torch.matmul.cpu.fp32",
                "candidate_digest": candidate_digest,
                "input_corpus_digest": input_corpus_digest,
                "execution_contract_digest": execution_digest,
                "timing_scope": "host_visible_completion",
                "completion_boundary": "synchronous-cpu-call-return",
                "instrumentation_profile": "benchmark",
                "observation_validity": "QUALIFIED",
                "frontier_role": "ACTIVE",
                "latency_ns": 154_000.0,
                "standard_uncertainty_ns": 500.0,
                "search_run_ids": ["search-1", "search-2", "search-3"],
                "holdout_run_ids": ["holdout-1", "holdout-2", "holdout-3"],
                "measurement_hardware_cohort": "hvc-test-exact-matmul",
                "qualification_policy": {
                    "policy_id": "exact-shape-operator-frontier-qualification",
                    "version": "2.0.0",
                    "minimum_search_sessions": 3,
                    "minimum_holdout_sessions": 3,
                    "maximum_session_iqr_over_median": 0.03,
                    "maximum_session_median_relative_range": 0.05,
                    "maximum_search_holdout_relative_gap": 0.05,
                    "minimum_warmup_iterations": 500,
                    "maximum_warmup_median_drift": 0.05,
                    "minimum_samples": 20,
                    "minimum_windows_per_sample": 5,
                    "minimum_timed_duration_ns": 100000000,
                    "estimator": "median(independent_holdout_session_medians)",
                    "uncertainty": "sample-standard-deviation(independent_holdout_session_medians)",
                },
                "candidate_coverage": {
                    "level": "C0_SINGLE",
                    "scope": "declared-runtime-candidate-family",
                    "evaluated_candidate_families": ["torch.matmul.cpu.fp32"],
                    "selected_candidate_family": "torch.matmul.cpu.fp32",
                    "evaluated_candidate_digests": [candidate_digest],
                    "selected_candidate_digest": candidate_digest,
                    "limitation": "does-not-establish-global-optimum-or-support-frontier-shift",
                },
                "session_evidence": {
                    "search": [
                        {
                            "run_id": f"search-{index}",
                            "process_id": 100 + index,
                            "run_bundle": f"/evidence/search-{index}",
                            "run_manifest_sha256": "a" * 64,
                            "benchmark_sha256": "b" * 64,
                            "correctness_sha256": "c" * 64,
                            "hardware_cohort": "hvc-test-exact-matmul",
                            "candidate_digest": candidate_digest,
                            "input_corpus_digest": input_corpus_digest,
                            "execution_contract_digest": execution_digest,
                            "correctness_oracle_policy_id": "matmul-fp32-float64-oracle-v1",
                            "timing_scope": "host_visible_completion",
                            "completion_boundary": "synchronous-cpu-call-return",
                            "timer_source": "time.perf_counter_ns",
                            "timer_resolution_ns": 1.0,
                            "instrumentation_profile": "benchmark",
                            "warmup_iterations": 500,
                            "warmup_window_samples_ns": [value] * 7,
                            "warmup_median_drift": 0.0,
                            "timed_duration_ns": 2000000000.0,
                            "median_ns": value,
                            "iqr_over_median": 0.0,
                            "sample_count": 20,
                            "samples_ns": [
                                value - 100.0,
                                *([value] * 18),
                                value + 100.0,
                            ],
                        }
                        for index, value in enumerate(
                            (155_000.0, 154_000.0, 153_000.0), 1
                        )
                    ],
                    "holdout": [
                        {
                            "run_id": f"holdout-{index}",
                            "process_id": 200 + index,
                            "run_bundle": f"/evidence/holdout-{index}",
                            "run_manifest_sha256": "d" * 64,
                            "benchmark_sha256": "e" * 64,
                            "correctness_sha256": "f" * 64,
                            "hardware_cohort": "hvc-test-exact-matmul",
                            "candidate_digest": candidate_digest,
                            "input_corpus_digest": input_corpus_digest,
                            "execution_contract_digest": execution_digest,
                            "correctness_oracle_policy_id": "matmul-fp32-float64-oracle-v1",
                            "timing_scope": "host_visible_completion",
                            "completion_boundary": "synchronous-cpu-call-return",
                            "timer_source": "time.perf_counter_ns",
                            "timer_resolution_ns": 1.0,
                            "instrumentation_profile": "benchmark",
                            "warmup_iterations": 500,
                            "warmup_window_samples_ns": [value] * 7,
                            "warmup_median_drift": 0.0,
                            "timed_duration_ns": 2000000000.0,
                            "median_ns": value,
                            "iqr_over_median": 0.0,
                            "sample_count": 20,
                            "samples_ns": [
                                value - 100.0,
                                *([value] * 18),
                                value + 100.0,
                            ],
                        }
                        for index, value in enumerate(
                            (153_500.0, 154_000.0, 154_500.0), 1
                        )
                    ],
                },
            }
        ],
    }
    source_bytes = json.dumps(
        observation, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    source_path = tmp_path / "specs/operator-frontiers/evidence/matmul-512.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source_bytes)
    profile = {
        "apiVersion": "groundupscale.dev/v1alpha1",
        "kind": "OperatorFrontierProfile",
        "metadata": {"name": "apple-m4-cpu-matmul-512", "version": "0.1.0"},
        "spec": {
            "target": observation["target"],
            "hardware_cohort": observation["hardware_cohort"],
            "source": {
                "path": "specs/operator-frontiers/evidence/matmul-512.json",
                "sha256": sha256(source_bytes).hexdigest(),
                "schema": observation["schema"],
            },
            "anchors": observation["anchors"],
        },
    }
    profile_path = tmp_path / "specs/operator-frontiers/apple-m4-cpu-matmul-512.yaml"
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    plan_path = tmp_path / "specs/plans/mac-cpu-prefill.yaml"
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    plan["spec"]["operator_frontier_profiles"] = [
        {
            "path": "specs/operator-frontiers/apple-m4-cpu-matmul-512.yaml",
            "version": "0.1.0",
        }
    ]
    plan["spec"]["operator_frontier_execution_domain"] = {
        "hardware_cohort": "hvc-test-exact-matmul",
        "threads": 10,
        "interop_threads": 1,
        "execution_mode": "eager",
        "timing_scope": "host_visible_completion",
        "completion_boundary": "synchronous-cpu-call-return",
        "instrumentation_profile": "benchmark",
        "candidate_families": {
            Q_PROJ_STABLE_PATH: "torch.matmul.cpu.fp32"
        },
        "candidate_digests": {
            Q_PROJ_STABLE_PATH: candidate_digest
        },
        "execution_contract_digests": {
            Q_PROJ_STABLE_PATH: execution_digest
        },
        "execution_contracts": {
            Q_PROJ_STABLE_PATH: execution_contract
        },
        "input_corpus_digests": {
            Q_PROJ_STABLE_PATH: input_corpus_digest
        },
    }
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return tmp_path


def _rewrite_frontier_cohort(repository: Path, cohort: str) -> None:
    source_path = repository / "specs/operator-frontiers/evidence/matmul-512.json"
    observation = json.loads(source_path.read_text(encoding="utf-8"))
    observation["hardware_cohort"] = cohort
    source_bytes = json.dumps(
        observation, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    source_path.write_bytes(source_bytes)
    profile_path = repository / "specs/operator-frontiers/apple-m4-cpu-matmul-512.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["spec"]["hardware_cohort"] = cohort
    profile["spec"]["source"]["sha256"] = sha256(source_bytes).hexdigest()
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")


def test_exact_shape_operator_frontier_reaches_the_m4_prediction_backend(
    tmp_path: Path,
) -> None:
    repository = _repository_with_exact_matmul_frontier(tmp_path)

    compiled = compile_analysis_plan(
        repository, repository / "specs/plans/mac-cpu-prefill.yaml"
    )
    prediction = compiled.hardware_prediction
    assert prediction is not None
    q_proj = next(
        candidate
        for candidate in prediction.candidates
        if candidate.stable_path.endswith("/layer_0/attention/q_proj")
    )

    assert q_proj.duration.operator_achievable_frontier_ns == pytest.approx(154_000.0)
    assert q_proj.duration.operator_frontier_standard_uncertainty_ns == pytest.approx(
        500.0
    )
    assert q_proj.duration.operator_frontier_match_status == "exact-anchor"
    assert q_proj.duration.operator_frontier_anchor_id == "m4-cpu-matmul-512-v1"
    assert q_proj.duration.empirical_hardware_floor_ns == pytest.approx(
        153_527.65853810357
    )
    scope = next(
        bound for bound in prediction.scope_bounds if bound.case_id == "matmul-q-proj"
    )
    assert scope.operator_achievable_frontier_ns == pytest.approx(154_000.0)
    layer_1 = next(
        candidate
        for candidate in prediction.candidates
        if candidate.stable_path.endswith("/layer_1/attention/q_proj")
    )
    assert layer_1.duration.operator_achievable_frontier_ns is None
    assert layer_1.duration.operator_frontier_match_status == "unknown"


def test_operator_frontier_from_another_hardware_cohort_fails_closed(
    tmp_path: Path,
) -> None:
    repository = _repository_with_exact_matmul_frontier(tmp_path)
    _rewrite_frontier_cohort(repository, "apple-m4-cpu-foreign-cohort")

    compiled = compile_analysis_plan(
        repository, repository / "specs/plans/mac-cpu-prefill.yaml"
    )
    prediction = compiled.hardware_prediction
    assert prediction is not None
    q_proj = next(
        candidate
        for candidate in prediction.candidates
        if candidate.stable_path.endswith("/layer_0/attention/q_proj")
    )

    assert q_proj.duration.operator_achievable_frontier_ns is None
    assert q_proj.duration.operator_frontier_match_status == "unknown"
    assert q_proj.duration.operator_frontier_reason_codes == (
        "operator-frontier-hardware-cohort-mismatch",
    )


def test_run_bundle_reports_exact_shape_frontier_efficiency_without_calling_it_prediction_error(
    tmp_path: Path,
) -> None:
    repository = _repository_with_exact_matmul_frontier(tmp_path / "repository")
    compiled = compile_analysis_plan(
        repository, repository / "specs/plans/mac-cpu-prefill.yaml"
    )
    run = RunBundleWriter(compiled).run(
        tmp_path / "artifacts",
        run_id="exact-frontier-demo",
        samples_override=4,
        warmup_override=0,
        windows_per_sample=1,
        target_window_ns=1,
        environment_validity={
            "schema": "groundupscale.dev/environment-validity/v1alpha1",
            "eligible": True,
            "reason_codes": [],
            "policy": {"policy_id": "local-apple-silicon-v2"},
        },
        require_valid_environment=True,
    )

    comparison = json.loads(
        (run / "comparison/predicted-vs-observed.json").read_text(encoding="utf-8")
    )
    q_proj = next(
        item for item in comparison["latency_cases"] if item["case_id"] == "matmul-q-proj"
    )
    assert q_proj["predicted"]["operator_achievable_frontier_ns"] == 154_000.0
    assert q_proj["predicted"]["operator_frontier_match_status"] == "exact-anchor"
    assert q_proj["comparison"]["operator_frontier_efficiency"] is None
    assert (
        q_proj["comparison"]["frontier_efficiency_status"]
        == "not-evaluable-observation-domain"
    )
    assert "operator-frontier-observation-cohort-mismatch" in q_proj[
        "comparison"
    ]["operator_frontier_comparison_reason_codes"]
    assert q_proj["comparison"]["relative_prediction_error"] is None
    report = (run / "reports/report.html").read_text(encoding="utf-8")
    assert "Exact-Shape Operator Frontier" in report
    assert "m4-cpu-matmul-512-v1" in report
    assert "Frontier Efficiency" in report
    assert "不是 prediction error" in report


def test_exact_frontier_rejects_a_rehashed_but_different_stride_contract(
    tmp_path: Path,
) -> None:
    repository = _repository_with_exact_matmul_frontier(tmp_path)
    plan_path = repository / "specs/plans/mac-cpu-prefill.yaml"
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    domain = plan["spec"]["operator_frontier_execution_domain"]
    contract = domain["execution_contracts"][Q_PROJ_STABLE_PATH]
    contract["operand_contracts"][1]["stride"] = [1, 512]
    contract["operand_contracts"][1]["layout"] = "strided"
    digest_body = {
        key: value
        for key, value in contract.items()
        if key != "execution_contract_digest"
    }
    contract["execution_contract_digest"] = content_fingerprint(digest_body)
    domain["execution_contract_digests"][Q_PROJ_STABLE_PATH] = contract[
        "execution_contract_digest"
    ]
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")

    compiled = compile_analysis_plan(repository, plan_path)
    q_proj = next(
        bound
        for bound in compiled.hardware_prediction.scope_bounds
        if bound.case_id == "matmul-q-proj"
    )

    assert q_proj.operator_achievable_frontier_ns is None
    assert q_proj.operator_frontier_match_status == "unknown"
    assert "exact-shape-operator-frontier-not-found" in (
        q_proj.operator_frontier_reason_codes
    )


def _exact_frontier_observation_comparison(tmp_path: Path) -> tuple[dict, dict, dict]:
    repository = _repository_with_exact_matmul_frontier(tmp_path)
    compiled = compile_analysis_plan(
        repository, repository / "specs/plans/mac-cpu-prefill.yaml"
    )
    prediction = compiled.hardware_prediction
    assert prediction is not None
    q_proj_bound = next(
        bound for bound in prediction.scope_bounds if bound.case_id == "matmul-q-proj"
    )
    candidate = _resolved_test_candidate()
    input_corpus = _resolved_test_input_corpus()
    execution = yaml.safe_load(
        (repository / "specs/plans/mac-cpu-prefill.yaml").read_text(
            encoding="utf-8"
        )
    )["spec"]["operator_frontier_execution_domain"]["execution_contracts"][
        Q_PROJ_STABLE_PATH
    ]
    assert execution["execution_contract_digest"] == (
        q_proj_bound.operator_frontier_execution_contract_digest
    )
    timing_policy = {
        "policy_id": "local-m4-exact-shape-timing-v1",
        "version": "1.0.0",
        "minimum_warmup_iterations": 500,
        "convergence_window_count": 7,
        "convergence_iterations_per_window": 20,
        "maximum_warmup_median_drift": 0.05,
        "minimum_samples": 20,
        "minimum_windows_per_sample": 5,
        "minimum_timed_duration_ns": 100_000_000,
    }
    case = {
        "case_id": "matmul-q-proj",
        "resolved_scope": Q_PROJ_STABLE_PATH,
        "mode": "operator",
        "samples": 20,
        "candidate_identity": candidate,
        "input_corpus": input_corpus,
        "execution_contract": execution,
        "timing_contract": {
            "policy": timing_policy,
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
        "warmup_convergence": {
            "policy": timing_policy,
            "warmup_iterations": 500,
            "window_samples_ns": [154_000.0] * 7,
            "median_drift": 0.0,
            "converged": True,
        },
        "latency": {
            "samples_ns": [155_000.0] * 20,
            "window_samples_ns": [[1_085_000] * 5 for _ in range(20)],
            "normalized_window_samples_ns": [[155_000.0] * 5 for _ in range(20)],
            "inner_iterations": 7,
            "windows_per_sample": 5,
            "median_ns": 155_000.0,
            "q1_ns": 155_000.0,
            "q3_ns": 155_000.0,
            "iqr_over_median": 0.0,
            "throughput_per_second": 1_000_000_000 / 155_000.0,
        },
    }
    operator_record = {
        "case_id": case["case_id"],
        "stable_path": case["resolved_scope"],
        "candidate_identity": deepcopy(candidate),
        "input_corpus": deepcopy(input_corpus),
        "execution_contract": deepcopy(execution),
        "correctness": {
            "schema": "groundupscale.dev/operator-correctness-evidence/v1alpha1",
            "status": "passed",
            "candidate_family": candidate["family"],
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
        },
    }
    common = {
        "hardware_prediction": prediction,
        "benchmark": {"cases": [case]},
        "trace": None,
        "live_set": {
            "predicted_framework_peak_bytes": 1,
            "peak_operation_stable_path": case["resolved_scope"],
            "exclusions": [],
        },
        "tensor_storage_observation": {
            "peak_framework_tensor_bytes": 1,
            "observer": "test",
            "peak_stable_path": case["resolved_scope"],
            "excludes": [],
        },
        "observation_evidence_tier": "qualified",
        "observation_hardware_cohort": "hvc-test-exact-matmul",
        "observation_operator_cases": (operator_record,),
    }
    return common, case, operator_record


def test_frontier_comparison_rejects_under_sampled_current_observation(
    tmp_path: Path,
) -> None:
    common, case, _ = _exact_frontier_observation_comparison(tmp_path)
    case["samples"] = 4
    case["latency"]["samples_ns"] = [155_000.0] * 4
    case["latency"]["window_samples_ns"] = [[1_000_000]] * 4
    case["latency"]["normalized_window_samples_ns"] = [[155_000.0]] * 4
    case["latency"]["windows_per_sample"] = 1

    result = build_prediction_observation_comparison(**common)

    comparison = result["latency_cases"][0]["comparison"]
    assert comparison["operator_frontier_efficiency"] is None
    assert "operator-frontier-observation-timing-unqualified" in comparison[
        "operator_frontier_comparison_reason_codes"
    ]
    observed = result["latency_cases"][0]["observed"]
    assert observed["environment_evidence_tier"] == "qualified"
    assert observed["evidence_tier"] == "unqualified"
    assert observed["frontier_observation_gate"] == {
        "status": "unqualified",
        "reason_codes": ["timing-policy-or-shape-invalid"],
    }


def test_frontier_comparison_rejects_unbound_current_correctness_identity(
    tmp_path: Path,
) -> None:
    common, _, operator_record = _exact_frontier_observation_comparison(tmp_path)
    operator_record["correctness"]["candidate_digest"] = "f" * 64

    result = build_prediction_observation_comparison(**common)

    comparison = result["latency_cases"][0]["comparison"]
    assert comparison["operator_frontier_efficiency"] is None
    assert "operator-frontier-observation-correctness-unqualified" in comparison[
        "operator_frontier_comparison_reason_codes"
    ]


def test_frontier_comparison_rejects_invalid_current_stride_contract(
    tmp_path: Path,
) -> None:
    common, case, operator_record = _exact_frontier_observation_comparison(tmp_path)
    case["execution_contract"]["operand_contracts"][0]["stride"] = [1]
    operator_record["execution_contract"] = deepcopy(case["execution_contract"])

    result = build_prediction_observation_comparison(**common)

    item = result["latency_cases"][0]
    assert item["observed"]["evidence_tier"] == "unqualified"
    assert "operator-execution-contract-invalid" in item["observed"][
        "reason_codes"
    ]
    assert item["comparison"]["operator_frontier_efficiency"] is None
    assert "operator-frontier-observation-execution-domain-invalid" in item[
        "comparison"
    ]["operator_frontier_comparison_reason_codes"]


@pytest.mark.parametrize(
    ("document_key", "tampered_field", "tampered_value", "reason_code"),
    [
        (
            "candidate_identity",
            "provider",
            "tampered-provider",
            "operator-candidate-identity-digest-invalid",
        ),
        (
            "input_corpus",
            "seed",
            999,
            "operator-input-corpus-digest-invalid",
        ),
    ],
)
def test_frontier_comparison_recomputes_current_identity_digests(
    tmp_path: Path,
    document_key: str,
    tampered_field: str,
    tampered_value: object,
    reason_code: str,
) -> None:
    common, case, operator_record = _exact_frontier_observation_comparison(tmp_path)
    case[document_key][tampered_field] = tampered_value
    operator_record[document_key] = deepcopy(case[document_key])

    result = build_prediction_observation_comparison(**common)

    item = result["latency_cases"][0]
    assert item["observed"]["evidence_tier"] == "unqualified"
    assert reason_code in item["observed"]["reason_codes"]
    assert item["comparison"]["operator_frontier_efficiency"] is None


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("missing-oracle", "operator-correctness-evidence-unqualified"),
        ("impossible-error", "operator-correctness-evidence-unqualified"),
    ],
)
def test_frontier_comparison_validates_the_exact_correctness_oracle(
    tmp_path: Path,
    mutation: str,
    reason_code: str,
) -> None:
    common, _, operator_record = _exact_frontier_observation_comparison(tmp_path)
    if mutation == "missing-oracle":
        operator_record["correctness"].pop("oracle")
    else:
        operator_record["correctness"]["max_absolute_error"] = 1e30

    result = build_prediction_observation_comparison(**common)

    item = result["latency_cases"][0]
    assert item["observed"]["evidence_tier"] == "unqualified"
    assert reason_code in item["observed"]["reason_codes"]
    assert item["comparison"]["operator_frontier_efficiency"] is None


def test_frontier_gap_uses_combined_anchor_observation_and_repeatability_uncertainty(
    tmp_path: Path,
) -> None:
    common, _, _ = _exact_frontier_observation_comparison(tmp_path)

    result = build_prediction_observation_comparison(**common)

    comparison = result["latency_cases"][0]["comparison"]
    assert comparison["operator_frontier_gap_status"] == (
        "within-combined-uncertainty"
    )
    assert comparison["operator_frontier_combined_uncertainty_ns"] > 7_750.0
    assert comparison["operator_frontier_uncertainty_policy"] == {
        "policy_id": "exact-frontier-observation-combined-uncertainty",
        "version": "1.0.0",
        "composition": "root-sum-square",
        "maximum_session_repeatability_fraction": 0.05,
        "coverage_basis": "six-session-qualification-bound-and-current-IQR",
    }


def test_m4_cpu_backend_emits_empirical_algorithm_independent_hardware_floors() -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )

    prediction = compiled.hardware_prediction

    assert prediction is not None
    assert prediction.schema == (
        "groundupscale.dev/hardware-backend-prediction/v1alpha2"
    )
    assert prediction.backend_id == "apple.m4.cpu.resource-envelope"
    assert prediction.placement == "local-m4/cpu"
    assert prediction.status == "phase-capabilities-incomplete"
    assert prediction.prediction_complete is False
    assert prediction.program_bounds.materialized_bytes == 289_415_168
    assert prediction.program_bounds.compulsory_bytes == 37_756_928
    assert prediction.program_bounds.vendor_memory_time_floor_ns == pytest.approx(
        314_641.06666666665
    )
    assert prediction.program_bounds.empirical_compute_time_ns == pytest.approx(
        5_553_975.963160658
    )

    layer0_qk = next(
        bound
        for bound in prediction.scope_bounds
        if bound.case_id == "matmul-layer0-qk"
    )
    assert layer0_qk.operator_achievable_frontier_ns is None
    assert "frontier-session-repeatability-exceeds-policy" in (
        layer0_qk.operator_frontier_reason_codes
    )
    assert prediction.program_bounds.empirical_memory_time_ns == pytest.approx(
        2_281_867.569778439
    )
    assert prediction.program_bounds.schedule == "serialized-unfused"
    assert prediction.program_bounds.ideal_dag_hardware_floor_ns == pytest.approx(
        5_553_975.963160659
    )
    assert prediction.program_bounds.serialized_hardware_floor_ns is None
    assert prediction.program_bounds.empirical_hardware_floor_ns is None
    assert prediction.program_bounds.resource_physical_floor_ns == pytest.approx(
        6_833_309.828880091
    )
    assert prediction.program_bounds.limiting_resource is None
    assert prediction.program_bounds.resource_limiting_resource == "compute.fp32"
    assert prediction.program_bounds.full_duration_ns is None
    assert prediction.program_bounds.compute_time.status == "unknown"
    assert prediction.program_bounds.compute_time.reason == (
        "vendor_does_not_publish_frequency_or_fma_issue_rate"
    )

    candidates = tuple(prediction.candidates)
    assert len(candidates) == 52
    q_proj = next(
        candidate
        for candidate in candidates
        if candidate.stable_path.endswith("/attention/q_proj")
    )
    assert q_proj.operation == "MatMul"
    assert q_proj.flops == 268_435_456
    assert q_proj.compulsory_bytes == 3_145_728
    assert q_proj.materialized_bytes == 3_145_728
    assert q_proj.duration.empirical_compute_time_ns == pytest.approx(
        153_527.65853810357
    )
    assert q_proj.duration.empirical_memory_time_ns == pytest.approx(
        24_802.206311951108
    )
    assert q_proj.duration.empirical_hardware_floor_ns == pytest.approx(
        153_527.65853810357
    )
    assert q_proj.duration.full_duration_ns is None
    assert q_proj.duration.status == "empirical-hardware-lower-bound"

    aliases = tuple(
        candidate
        for candidate in candidates
        if candidate.operation in {"View", "Transpose"}
    )
    assert aliases
    assert all(candidate.compulsory_bytes == 0 for candidate in aliases)
    assert all(
        candidate.duration.empirical_hardware_floor_ns == 0 for candidate in aliases
    )

    resources = {item.resource: item for item in prediction.measured_capabilities}
    assert resources["compute.fp32"].robust_achievable_rate == pytest.approx(
        1_748_450_139_577.8
    )
    assert resources["memory.shared"].robust_achievable_rate == pytest.approx(
        126_832_587_409.13748
    )
    assert resources["compute.fp32"].environment_eligible is False

    e2e = next(
        bound for bound in prediction.scope_bounds if bound.case_id == "two-layer-prefill"
    )
    assert e2e.compulsory_bytes == 37_756_928
    assert e2e.materialized_bytes == 289_415_168
    assert e2e.schedule == "serialized-unfused"
    assert e2e.critical_path_hardware_floor_ns is None
    assert e2e.resource_hardware_floor_ns == pytest.approx(5_553_975.963160659)
    assert e2e.ideal_dag_hardware_floor_ns == pytest.approx(5_553_975.963160659)
    assert e2e.serialized_hardware_floor_ns is None
    assert e2e.empirical_hardware_floor_ns is None
    assert e2e.resource_physical_floor_ns == pytest.approx(6_833_309.828880091)


def test_m4_gpu_plan_does_not_silently_reuse_the_cpu_backend() -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-mps-prefill.yaml"
    )

    assert compiled.hardware_prediction is None


def test_softmax_candidate_serializes_dependent_phases_without_chunk_contract() -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )

    prediction = compiled.hardware_prediction
    assert prediction is not None
    softmax = next(
        candidate
        for candidate in prediction.candidates
        if candidate.stable_path.endswith("/layer_0/attention/softmax")
    )

    schedule = softmax.phase_schedule
    assert schedule is not None
    assert schedule.policy == "serialized-no-chunk"
    assert schedule.chunk_pipeline_contract_id is None
    assert [phase.phase_name for phase in schedule.phases] == [
        "max_reduce",
        "subtract",
        "exp",
        "sum_reduce",
        "normalize",
    ]
    assert [phase.predecessor_phase_ids for phase in schedule.phases] == [
        (),
        (schedule.phases[0].phase_id,),
        (schedule.phases[1].phase_id,),
        (schedule.phases[2].phase_id,),
        (schedule.phases[3].phase_id,),
    ]
    assert schedule.status == "unknown"
    assert schedule.serialized_duration_ns is None
    assert schedule.critical_path_duration_ns is None
    assert schedule.selected_duration_ns is None
    assert softmax.duration.empirical_hardware_floor_ns is None
    assert softmax.duration.resource_physical_floor_ns == pytest.approx(
        132_278.43366373924
    )
    assert {
        "compute.reduction.max.fp32",
        "compute.transcendental.exp.fp32",
        "compute.reduction.sum.fp32",
        "memory.row-reduction.fp32",
    } <= set(schedule.missing_capabilities)
    assert all(phase.status == "unknown" for phase in schedule.phases)
    assert all(phase.resource_composition == "max" for phase in schedule.phases)
    assert all(not phase.overlap_evidence_refs for phase in schedule.phases)
    assert all(not phase.capability_evidence_refs for phase in schedule.phases)
    assert schedule.formula == "sum(phase.local_hardware_floor_ns)"


def test_rmsnorm_candidate_serializes_every_dependent_computation_phase() -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT, REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )

    prediction = compiled.hardware_prediction
    assert prediction is not None
    rmsnorm = next(
        candidate
        for candidate in prediction.candidates
        if candidate.stable_path.endswith("/layer_0/input_norm")
    )

    schedule = rmsnorm.phase_schedule
    assert schedule is not None
    assert [phase.phase_name for phase in schedule.phases] == [
        "square",
        "reduce_sum",
        "mean_scale",
        "epsilon_add",
        "rsqrt",
        "input_scale",
        "weight_scale",
    ]
    assert [phase.minimum_flops for phase in schedule.phases] == [
        262_144,
        261_632,
        512,
        512,
        512,
        262_144,
        262_144,
    ]
    assert [phase.logical_read_bytes + phase.logical_write_bytes for phase in schedule.phases] == [
        2_097_152,
        1_050_624,
        4_096,
        4_096,
        4_096,
        2_099_200,
        2_099_200,
    ]
    assert schedule.policy == "serialized-no-chunk"
    assert schedule.chunk_pipeline_contract_id is None
    assert schedule.status == "unknown"
    assert schedule.selected_duration_ns is None
    assert rmsnorm.duration.empirical_hardware_floor_ns is None


def test_matching_phase_capabilities_enable_only_the_explicit_serial_sum() -> None:
    bundle = SpecRepository(REPOSITORY_ROOT).load_analysis_plan(
        REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )
    profile = bundle.hardware_capability_profiles[0]
    compute_template = next(
        resource for resource in profile.spec.resources if resource.unit == "FLOP/s"
    )
    memory_template = next(
        resource for resource in profile.spec.resources if resource.unit == "B/s"
    )
    compute_resources = {
        "compute.reduction.max.fp32",
        "compute.elementwise.subtract.fp32",
        "compute.transcendental.exp.fp32",
        "compute.reduction.sum.fp32",
        "compute.elementwise.divide.fp32",
    }
    memory_resources = {
        "memory.row-reduction.fp32",
        "memory.broadcast-read-write.fp32",
        "memory.elementwise-read-write.fp32",
    }
    exact_resources = tuple(
        compute_template.model_copy(
            update={
                "resource": resource,
                "robust_achievable_rate": 1_000_000_000.0,
                "optimistic_rate": 1_000_000_000.0,
            }
        )
        for resource in sorted(compute_resources)
    ) + tuple(
        memory_template.model_copy(
            update={
                "resource": resource,
                "robust_achievable_rate": 1_000_000_000.0,
                "optimistic_rate": 1_000_000_000.0,
            }
        )
        for resource in sorted(memory_resources)
    )
    qualified_profile = profile.model_copy(
        update={
            "spec": profile.spec.model_copy(
                update={
                    "resources": profile.spec.resources + exact_resources,
                    "environment": {**profile.spec.environment, "eligible": True},
                }
            )
        }
    )
    compiled = compile_analysis_bundle(
        replace(bundle, hardware_capability_profiles=(qualified_profile,))
    )
    prediction = compiled.hardware_prediction
    assert prediction is not None
    softmax = next(
        candidate
        for candidate in prediction.candidates
        if candidate.stable_path.endswith("/layer_0/attention/softmax")
    )

    schedule = softmax.phase_schedule
    assert schedule is not None
    assert schedule.status == "known"
    assert schedule.missing_capabilities == ()
    assert all(phase.resource_composition == "max" for phase in schedule.phases)
    assert all(not phase.overlap_evidence_refs for phase in schedule.phases)
    assert all(len(phase.capability_evidence_refs) == 2 for phase in schedule.phases)
    assert schedule.serialized_duration_ns == pytest.approx(67_174_400.0)
    assert schedule.critical_path_duration_ns == pytest.approx(67_174_400.0)
    assert schedule.selected_duration_ns == pytest.approx(67_174_400.0)
    assert softmax.duration.empirical_hardware_floor_ns == pytest.approx(
        67_174_400.0
    )


def test_exploratory_phase_profile_emits_numbers_without_promoting_authority() -> None:
    compiled = compile_analysis_plan(
        REPOSITORY_ROOT,
        REPOSITORY_ROOT
        / "goal_process/mac-transformer-ir-calibration-slice/"
        "mac-cpu-prefill-phase-exploratory.yaml",
    )
    prediction = compiled.hardware_prediction
    assert prediction is not None
    softmax = next(
        candidate
        for candidate in prediction.candidates
        if candidate.stable_path.endswith("/layer_0/attention/softmax")
    )
    rmsnorm = next(
        candidate
        for candidate in prediction.candidates
        if candidate.stable_path.endswith("/layer_0/input_norm")
    )

    assert softmax.phase_schedule is not None
    assert softmax.phase_schedule.status == "unknown"
    assert softmax.phase_schedule.selected_duration_ns is None
    assert softmax.duration.empirical_hardware_floor_ns is None
    assert softmax.duration.provisional_estimate_ns == pytest.approx(
        691_693.8713503513
    )
    assert rmsnorm.duration.provisional_estimate_ns == pytest.approx(
        57_279.93574268858
    )
    assert softmax.duration.provisional_evidence_tier == "exploratory"
    assert set(softmax.duration.provisional_reason_codes) == {
        "load-above-policy",
        "total-competing-cpu-above-policy",
    }
    assert prediction.program_bounds.empirical_hardware_floor_ns is None
    assert prediction.program_bounds.provisional_estimate_ns == pytest.approx(
        8_160_245.494435462
    )


@pytest.mark.parametrize("invalid_kind", ["wrong-unit", "duplicate-resource"])
def test_phase_capability_identity_includes_unit_and_uniqueness(
    invalid_kind: str,
) -> None:
    bundle = SpecRepository(REPOSITORY_ROOT).load_analysis_plan(
        REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"
    )
    profile = bundle.hardware_capability_profiles[0]
    compute_template = next(
        resource for resource in profile.spec.resources if resource.unit == "FLOP/s"
    )
    memory_template = next(
        resource for resource in profile.spec.resources if resource.unit == "B/s"
    )
    compute_resources = (
        "compute.reduction.max.fp32",
        "compute.elementwise.subtract.fp32",
        "compute.transcendental.exp.fp32",
        "compute.reduction.sum.fp32",
        "compute.elementwise.divide.fp32",
    )
    memory_resources = (
        "memory.row-reduction.fp32",
        "memory.broadcast-read-write.fp32",
        "memory.elementwise-read-write.fp32",
    )
    exact_resources = tuple(
        compute_template.model_copy(update={"resource": resource})
        for resource in compute_resources
    ) + tuple(
        memory_template.model_copy(update={"resource": resource})
        for resource in memory_resources
    )
    if invalid_kind == "wrong-unit":
        exact_resources = tuple(
            memory_template.model_copy(update={"resource": resource.resource})
            if resource.resource == "compute.transcendental.exp.fp32"
            else resource
            for resource in exact_resources
        )
        expected_error = "compute.transcendental.exp.fp32:expected-FLOP/s-got-B/s"
    else:
        exact_resources += (exact_resources[0],)
        expected_error = "compute.reduction.max.fp32:duplicate-envelope"
    invalid_profile = profile.model_copy(
        update={
            "spec": profile.spec.model_copy(
                update={
                    "resources": profile.spec.resources + exact_resources,
                    "environment": {**profile.spec.environment, "eligible": True},
                }
            )
        }
    )

    compiled = compile_analysis_bundle(
        replace(bundle, hardware_capability_profiles=(invalid_profile,))
    )
    prediction = compiled.hardware_prediction
    assert prediction is not None
    softmax = next(
        candidate
        for candidate in prediction.candidates
        if candidate.stable_path.endswith("/layer_0/attention/softmax")
    )

    assert softmax.phase_schedule is not None
    assert softmax.phase_schedule.status == "unknown"
    assert expected_error in softmax.phase_schedule.missing_capabilities
