from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from test_ascend_npu_measurement_adapter import (
    _available_runtime,
    _complete_system_probe,
)
from test_ascend_npu_operator_frontier import _rewrite_environment_session

from groundupscale.diagnostics import diagnose_run_bundle, render_diagnostic_report
from groundupscale.measurement_adapters.ascend_npu import (
    AscendNpuMeasurementAdapter,
)
from groundupscale.measurement_run import MeasurementRunBundleWriter
from groundupscale.operator_frontier import OperatorFrontierBundleWriter
from groundupscale.run_bundle import verify_run_bundle


SEQUENCE_COUNT = 2
HEAD_COUNT = 8
HEAD_DIMENSION = 64
SETUP_LATENCY_NS = 7_000_000
ASYMPTOTIC_RATE_FLOP_S = 1_000_000_000_000


def _declared_work(sequence_lengths: list[int]) -> int:
    return 4 * HEAD_COUNT * HEAD_DIMENSION * sum(
        length**2 for length in sequence_lengths
    )


def _expected_latency_ns(sequence_length: int) -> int:
    work = _declared_work([sequence_length] * SEQUENCE_COUNT)
    return round(
        SETUP_LATENCY_NS
        + work / ASYMPTOTIC_RATE_FLOP_S * 1_000_000_000
    )


def _flash_attention_measurement_run(
    root: Path,
    *,
    run_id: str,
    sequence_lengths: list[int],
    median_ns: int,
    process_id: int,
) -> Path:
    shape = {
        "sequence_count": len(sequence_lengths),
        "sequence_lengths": sequence_lengths,
        "head_count": HEAD_COUNT,
        "head_dimension": HEAD_DIMENSION,
    }
    case = {
        "schema": (
            "groundupscale.dev/exact-shape-flash-attention-tnd-case/v1alpha1"
        ),
        "operation": "FlashAttentionForward",
        "shape": shape,
        "dtype": "float16",
        "layout": "TND",
        "causal": False,
        "mask": "none",
        "dropout_probability": 0.0,
        "mode": "forward",
        "seed": 20260813,
        "candidate": "torch_npu.npu_fusion_attention",
        "warmup_iterations": 20,
        "repetitions": 5,
        "inner_iterations": 1,
    }
    vector_identity = ",".join(str(length) for length in sequence_lengths)
    raw = {
        "runtime_device_name": "Ascend910B2",
        "candidate_device": "npu:0",
        "cpu_fallback": False,
        "minimum_alignment_bytes": 64,
        "q_sha256": sha256(f"q-{vector_identity}".encode()).hexdigest(),
        "k_sha256": sha256(f"k-{vector_identity}".encode()).hexdigest(),
        "v_sha256": sha256(f"v-{vector_identity}".encode()).hexdigest(),
        "target_output_sha256": sha256(
            f"output-{vector_identity}".encode()
        ).hexdigest(),
        "correctness": {
            "status": "passed",
            "oracle": "deterministic-float64-attention",
            "atol": 0.001,
            "rtol": 0.001,
            "max_absolute_error": 0.0002,
            "max_relative_error": 0.0004,
            "finite": True,
            "shape_exact": True,
        },
        "raw_samples_ns": [
            median_ns - 2,
            median_ns - 1,
            median_ns,
            median_ns + 1,
            median_ns + 2,
        ],
        "memory": {
            "allocated_bytes_before": 0,
            "allocated_bytes_after": 0,
            "reserved_bytes_after": 0,
            "maximum_allocated_bytes": 0,
        },
        "device_event_id": "per-sample-torch-npu-event-pair",
        "stream_id": "default-npu-stream",
    }
    adapter = AscendNpuMeasurementAdapter(
        runtime_loader=_available_runtime,
        collection_executor=lambda *args: raw,
        system_probe=_complete_system_probe,
    )
    run = MeasurementRunBundleWriter(adapter).run(root, case=case, run_id=run_id)
    _rewrite_environment_session(run, process_id=process_id)
    assert verify_run_bundle(run)["passed"] is True
    return run


def _flash_attention_inputs(
    tmp_path: Path,
) -> tuple[list[Path], list[Path], list[Path]]:
    measurements = tmp_path / "flash-attention-measurements"
    search: list[Path] = []
    holdout: list[Path] = []
    validation: list[Path] = []
    process_id = 10_000
    main_lengths = (128, 1024, 3072, 4097, 6144, 8192)
    validation_lengths = (512, 2048, 3584, 4096, 5120, 7168)
    for sequence_length in main_lengths:
        for lane, target in (("search", search), ("holdout", holdout)):
            for session in range(3):
                target.append(
                    _flash_attention_measurement_run(
                        measurements,
                        run_id=(
                            f"flash-{lane}-s{sequence_length}-{session}"
                        ),
                        sequence_lengths=[sequence_length] * SEQUENCE_COUNT,
                        median_ns=_expected_latency_ns(sequence_length) + session,
                        process_id=process_id,
                    )
                )
                process_id += 1
    for sequence_length in validation_lengths:
        for session in range(3):
            validation.append(
                _flash_attention_measurement_run(
                    measurements,
                    run_id=f"flash-validation-s{sequence_length}-{session}",
                    sequence_lengths=[sequence_length] * SEQUENCE_COUNT,
                    median_ns=_expected_latency_ns(sequence_length) + session,
                    process_id=process_id,
                )
            )
            process_id += 1
    return search, holdout, validation


def _flash_attention_policy() -> dict[str, object]:
    main_lengths = [128, 1024, 3072, 4097, 6144, 8192]
    validation_lengths = [512, 2048, 3584, 4096, 5120, 7168]
    return {
        "schema": (
            "groundupscale.dev/operator-frontier-qualification-policy/v1alpha1"
        ),
        "policy_id": "test-ascend-flash-attention-tnd-forward",
        "version": "v1",
        "scope": {
            "hardware_cohort": "ascend-npu-febd831c8d07e06f",
            "operation": "FlashAttentionForward",
            "dtype": "float16",
            "layout": "TND",
            "sequence_count": SEQUENCE_COUNT,
            "head_count": HEAD_COUNT,
            "head_dimension": HEAD_DIMENSION,
            "causal": False,
            "mask": "none",
            "dropout_probability": 0.0,
            "mode": "forward",
            "anchor_sequence_lengths": main_lengths,
            "confirmation_sequence_lengths": validation_lengths,
            "candidate_ids": ["torch_npu.npu_fusion_attention"],
        },
        "collection_plan": {
            "plan_id": "test-ascend-flash-attention-tnd-forward-v1",
            "hardware_cohort": "ascend-npu-febd831c8d07e06f",
            "operation": "FlashAttentionForward",
            "dtype": "float16",
            "layout": "TND",
            "sequence_count": SEQUENCE_COUNT,
            "head_count": HEAD_COUNT,
            "head_dimension": HEAD_DIMENSION,
            "causal": False,
            "mask": "none",
            "dropout_probability": 0.0,
            "mode": "forward",
            "candidate_ids": ["torch_npu.npu_fusion_attention"],
            "execution_mode": "pytorch-eager",
            "warmup_iterations": 20,
            "repetitions": 5,
            "inner_iterations": 1,
            "completion_boundary": (
                "end-event-synchronize-plus-device-synchronize"
            ),
            "instrumentation_profile": "ascend-npu-baseline-timing-v1",
            "random_seed": 20260813,
            "raw_sample_retention": "preserve-all-no-selective-exclusion",
            "main_sweep_sequence_lengths": main_lengths,
            "independent_holdout_sequence_lengths": main_lengths,
            "independent_validation_sequence_lengths": validation_lengths,
            "supplemental_sequence_lengths": [],
            "maximum_supplemental_rounds": 0,
            "executed_supplemental_rounds": 0,
        },
        "minimum_search_sessions": 3,
        "minimum_holdout_sessions": 3,
        "minimum_confirmation_sessions": 3,
        "minimum_warmup_iterations": 20,
        "maximum_session_median_relative_range": 0.10,
        "minimum_candidate_coverage": "C0_SINGLE",
        "holdout_candidate_scope": "all-eligible-candidates",
        "response_target": "latency",
        "response_kind": "setup-plus-throughput",
        "response_version": "v1",
        "maximum_relative_error": 0.01,
        "maximum_setup_fraction_for_steady": 0.10,
        "uncertainty_combination": "root-sum-of-squares",
        "target_coverage": 0.68,
        "sample_exclusion": "none-preserve-all-raw-samples",
        "estimator": "median(independent-holdout-session-medians)",
        "change_reason": "deterministic ticket-37 synthetic fixture",
        "revalidation": "on cohort, domain, candidate, evidence, or policy change",
        "evidence_ref": "test://ticket-37/flash-attention-policy-v1",
    }


def test_equal_length_tnd_flash_attention_qualifies_and_queries_public_surface(
    tmp_path: Path,
) -> None:
    search, holdout, validation = _flash_attention_inputs(tmp_path)
    source_manifest = json.loads(
        (search[0] / "run.manifest.json").read_text(encoding="utf-8")
    )

    def source_artifact(role: str) -> dict[str, object]:
        artifact = next(
            item for item in source_manifest["artifacts"] if item["role"] == role
        )
        return json.loads(
            (search[0] / artifact["path"]).read_text(encoding="utf-8")
        )

    candidate = source_artifact("candidate-identity")
    assert candidate["operation"] == "FlashAttentionForward"
    assert candidate["semantic_domain"]["causal"] is False
    inputs = source_artifact("input-corpus")
    assert inputs["sequence_lengths"] == [128, 128]
    execution = source_artifact("execution-contract")
    assert execution["mask"] == "none"
    assert execution["dropout_probability"] == 0.0
    assert execution["mode"] == "forward"

    run = OperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="ascend-flash-attention-tnd-forward-v1",
        qualification_policy=_flash_attention_policy(),
        search_runs=search,
        holdout_runs=holdout,
        confirmation_runs=validation,
        query_sizes=(1024, 1536, 3800, 7000, 9000),
    )

    assert verify_run_bundle(run)["passed"] is True
    qualification = json.loads(
        (run / "frontier/qualification.json").read_text(encoding="utf-8")
    )
    assert qualification["status"] == "qualified"
    assert qualification["surface"]["work_formula"] == {
        "kind": "flash-attention-tnd-forward-qk-pv",
        "version": "v1",
        "causal": False,
        "work_unit": "FLOP",
    }

    result = diagnose_run_bundle(run)
    queries = {
        query["query_shape"]["sequence_length"]: query
        for query in result["capability_surface_queries"]
    }
    assert queries[1024]["status"] == "exact_anchor"
    assert queries[1024]["latency"]["declared_work"] == 4_294_967_296
    assert queries[1536]["status"] == "modeled"
    assert queries[1536]["shape_regime"]["classification"] == "ramp"
    assert queries[3800]["status"] == "unknown"
    assert queries[3800]["reason_code"] == "shape_regime_unvalidated"
    assert queries[7000]["status"] == "modeled"
    assert queries[7000]["shape_regime"]["classification"] == "steady"
    assert queries[9000]["status"] == "unknown"
    assert queries[9000]["reason_code"] == "outside_validated_domain"
    assert queries[7000]["effective_rate"]["value"] == (
        queries[7000]["latency"]["declared_work"]
        / (queries[7000]["latency"]["value_ns"] * 1e-9)
    )

    report = render_diagnostic_report(result)
    assert "FlashAttentionForward" in report
    assert "Shape Regime=" in report
    assert "declared-work=" in report
    assert "evidence=" in report


def test_ragged_tnd_distributions_keep_distinct_exact_shape_identities(
    tmp_path: Path,
) -> None:
    measurements = tmp_path / "ragged-measurements"
    search: list[Path] = []
    holdout: list[Path] = []
    process_id = 20_000
    distributions = ([256, 768], [512, 512])
    for sequence_lengths in distributions:
        latency = round(
            SETUP_LATENCY_NS
            + _declared_work(sequence_lengths)
            / ASYMPTOTIC_RATE_FLOP_S
            * 1_000_000_000
        )
        vector = "-".join(str(value) for value in sequence_lengths)
        for lane, target in (("search", search), ("holdout", holdout)):
            for session in range(3):
                target.append(
                    _flash_attention_measurement_run(
                        measurements,
                        run_id=f"ragged-{lane}-{vector}-{session}",
                        sequence_lengths=list(sequence_lengths),
                        median_ns=latency + session,
                        process_id=process_id,
                    )
                )
                process_id += 1
    policy = _flash_attention_policy()
    policy.pop("collection_plan")
    policy["policy_id"] = "test-ragged-flash-attention-exact-only"
    policy["scope"] = {
        **policy["scope"],
        "sequence_distribution_mode": "exact-only",
        "sequence_vectors": [list(item) for item in distributions],
    }

    run = OperatorFrontierBundleWriter().run(
        tmp_path / "frontier",
        run_id="ascend-ragged-flash-attention-exact-v1",
        qualification_policy=policy,
        search_runs=search,
        holdout_runs=holdout,
        confirmation_runs=(),
        query_sizes=(),
        query_shapes=(
            {"sequence_lengths": [256, 768]},
            {"sequence_lengths": [512, 512]},
            {"sequence_lengths": [128, 896]},
        ),
    )

    assert verify_run_bundle(run)["passed"] is True
    result = diagnose_run_bundle(run)
    first, second, unsupported = result["capability_surface_queries"]
    assert first["status"] == "exact_anchor"
    assert second["status"] == "exact_anchor"
    assert first["operator_shape_identity"] != second["operator_shape_identity"]
    assert first["latency"]["declared_work"] == 1_342_177_280
    assert second["latency"]["declared_work"] == 1_073_741_824
    assert unsupported["status"] == "unknown"
    assert unsupported["reason_code"] == (
        "unsupported_sequence_distribution_interpolation"
    )
