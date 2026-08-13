"""Evidence-qualified exact-Shape Operator Frontier construction."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from statistics import median, stdev
from typing import Any

import yaml

from groundupscale.benchmark.frontier_evidence import (
    embedded_digest_is_valid,
    validate_exact_matmul_correctness,
    validate_exact_timing_evidence,
)
from groundupscale.ir import canonical_data, content_fingerprint
from groundupscale.run_bundle import verify_run_bundle
from groundupscale.schemas.v1alpha1 import (
    ExactOperatorExecutionContract,
    OperatorFrontierProfileDocument,
)


class FrontierQualificationError(ValueError):
    """Raised when immutable Run Bundles cannot qualify an exact Anchor."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "frontier-qualification-evidence-invalid",
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FrontierQualificationError(f"{path}: expected a JSON object")
    return value


def _artifact(root: Path, manifest: dict[str, Any], role: str) -> tuple[Path, str]:
    matches = [item for item in manifest.get("artifacts", ()) if item.get("role") == role]
    if len(matches) != 1:
        raise FrontierQualificationError(
            f"{root}: expected exactly one {role} artifact"
        )
    path = (root / str(matches[0].get("path", ""))).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise FrontierQualificationError(f"{root}: invalid {role} artifact path")
    return path, str(matches[0].get("sha256", ""))


def _objects(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _objects(child)


def _canonical_layout(layout: str) -> str:
    if layout == "contiguous":
        return "row-major-contiguous"
    if layout in {"transposed", "transposed-strided"}:
        return "strided"
    return layout


def _observed_m4_identity_is_valid(cohort: dict[str, Any]) -> bool:
    device = cohort.get("device")
    observed = device.get("observed_identity") if isinstance(device, dict) else None
    if not isinstance(device, dict) or not isinstance(observed, dict):
        return False
    levels = observed.get("performance_levels")
    physical = observed.get("physical_cpu")
    logical = observed.get("logical_cpu")
    return bool(
        observed.get("status") == "resolved"
        and observed.get("source") == "sysctl"
        and isinstance(observed.get("model"), str)
        and str(observed["model"]).startswith("Mac")
        and isinstance(observed.get("cpu_brand"), str)
        and str(observed["cpu_brand"]).startswith("Apple M")
        and isinstance(physical, int)
        and physical > 0
        and isinstance(logical, int)
        and logical >= physical
        and isinstance(levels, dict)
        and isinstance(levels.get("performance_cores"), int)
        and isinstance(levels.get("efficiency_cores"), int)
        and levels["performance_cores"] + levels["efficiency_cores"] == physical
        and device.get("identity_sha256") == content_fingerprint(observed)
    )


def _session(root: str | Path, *, case_id: str) -> dict[str, Any]:
    bundle = Path(root).resolve()
    verification = verify_run_bundle(bundle)
    if verification.get("passed") is not True:
        raise FrontierQualificationError(
            f"{bundle}: Run Bundle digest verification failed"
        )
    manifest_path = bundle / "run.manifest.json"
    manifest = _load_json(manifest_path)
    if (
        manifest.get("status") != "completed"
        or manifest.get("device") != "cpu"
        or manifest.get("environment_validity") != "passed"
    ):
        raise FrontierQualificationError(f"{bundle}: Run Bundle is not trusted CPU evidence")

    environment_path, _ = _artifact(bundle, manifest, "environment")
    environment = _load_json(environment_path)
    preflight = environment.get("measurement_preflight")
    process = environment.get("process")
    if not isinstance(preflight, dict) or preflight.get("eligible") is not True:
        raise FrontierQualificationError(f"{bundle}: environment preflight is not eligible")
    if not isinstance(process, dict) or not isinstance(process.get("pid"), int):
        raise FrontierQualificationError(f"{bundle}: measurement process identity is missing")
    cohort = environment.get("hardware_validity_cohort")
    if not isinstance(cohort, dict):
        raise FrontierQualificationError(
            f"{bundle}: complete hardware cohort is missing",
            reason_code="frontier-hardware-cohort-invalid",
        )
    cohort_id = cohort.get("cohort_id")
    cohort_body = {key: value for key, value in cohort.items() if key != "cohort_id"}
    if (
        not isinstance(cohort_id, str)
        or cohort_id != f"hvc-{content_fingerprint(cohort_body)}"
        or manifest.get("hardware_cohort") != cohort_id
        or not _observed_m4_identity_is_valid(cohort)
    ):
        raise FrontierQualificationError(
            f"{bundle}: hardware cohort identity is inconsistent",
            reason_code="frontier-hardware-cohort-invalid",
        )

    resolved_path, _ = _artifact(bundle, manifest, "resolved-input-lock")
    resolved = _load_json(resolved_path)
    documents = resolved.get("documents")
    profiles = (
        documents.get("hardware_capability_profiles")
        if isinstance(documents, dict)
        else None
    )
    if not isinstance(profiles, list) or len(profiles) != 1:
        raise FrontierQualificationError(f"{bundle}: expected one capability profile")
    capability_spec = profiles[0].get("spec") if isinstance(profiles[0], dict) else None
    if not isinstance(capability_spec, dict):
        raise FrontierQualificationError(f"{bundle}: invalid capability profile")
    capability_cohort = capability_spec.get("hardware_cohort")
    target = capability_spec.get("target")
    if not isinstance(capability_cohort, str) or not capability_cohort:
        raise FrontierQualificationError(f"{bundle}: capability cohort is missing")
    if not isinstance(target, dict):
        raise FrontierQualificationError(f"{bundle}: capability target is missing")
    if target.get("device") != "cpu" or not isinstance(target.get("hardware"), str):
        raise FrontierQualificationError(f"{bundle}: capability target is invalid")

    benchmark_path, benchmark_digest = _artifact(
        bundle, manifest, "benchmark-observation"
    )
    benchmark = _load_json(benchmark_path)
    cases = [item for item in benchmark.get("cases", ()) if item.get("case_id") == case_id]
    if len(cases) != 1:
        raise FrontierQualificationError(f"{bundle}: expected one benchmark case {case_id}")
    case = cases[0]
    candidate = case.get("candidate_identity")
    input_corpus = case.get("input_corpus")
    execution_contract = case.get("execution_contract")
    if not all(
        isinstance(value, dict) and value.get("status") == "resolved"
        for value in (candidate, input_corpus, execution_contract)
    ):
        raise FrontierQualificationError(
            f"{bundle}: exact candidate/domain identity is missing",
            reason_code="frontier-candidate-identity-mismatch",
        )
    try:
        typed_execution_contract = ExactOperatorExecutionContract.model_validate(
            execution_contract
        )
    except ValueError as error:
        raise FrontierQualificationError(
            f"{bundle}: exact execution contract is invalid: {error}",
            reason_code="frontier-execution-contract-invalid",
        ) from error
    execution_contract = typed_execution_contract.model_dump(
        mode="json", by_alias=True
    )
    candidate_family = candidate.get("family")
    candidate_digest = candidate.get("candidate_digest")
    input_corpus_digest = input_corpus.get("input_corpus_digest")
    execution_contract_digest = execution_contract.get("execution_contract_digest")
    dispatch_binaries = candidate.get("dispatch_provider_binaries")
    if (
        not embedded_digest_is_valid(candidate, "candidate_digest")
        or not embedded_digest_is_valid(input_corpus, "input_corpus_digest")
        or not embedded_digest_is_valid(
            execution_contract, "execution_contract_digest"
        )
        or not isinstance(candidate_family, str)
        or not isinstance(dispatch_binaries, list)
        or not dispatch_binaries
        or any(
            not isinstance(item, dict)
            or item.get("role") != "cpu-dispatch-provider"
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
            for item in dispatch_binaries
        )
    ):
        raise FrontierQualificationError(
            f"{bundle}: exact candidate/domain digest is missing",
            reason_code="frontier-candidate-identity-mismatch",
        )

    correctness_path, correctness_digest = _artifact(
        bundle, manifest, "correctness-observation"
    )
    correctness = _load_json(correctness_path)
    operator_records = correctness.get("operator_cases")
    matching_correctness = [
        item
        for item in operator_records
        if isinstance(item, dict)
        and item.get("case_id") == case_id
        and item.get("stable_path") == case.get("resolved_scope")
    ] if isinstance(operator_records, list) else []
    if len(matching_correctness) != 1:
        raise FrontierQualificationError(
            f"{bundle}: exact operator correctness evidence is missing",
            reason_code="frontier-correctness-evidence-invalid",
        )
    correctness_record = matching_correctness[0]
    correctness_evidence = validate_exact_matmul_correctness(
        correctness_record,
        candidate=candidate,
        input_corpus=input_corpus,
        execution=execution_contract,
    )
    if correctness_evidence is None:
        raise FrontierQualificationError(
            f"{bundle}: exact operator correctness evidence is invalid",
            reason_code="frontier-correctness-evidence-invalid",
        )
    oracle = correctness_evidence.oracle
    timing_validity = validate_exact_timing_evidence(case)
    if not timing_validity.qualified:
        raise FrontierQualificationError(
            f"{bundle}: timing/warmup evidence is invalid: "
            + ", ".join(timing_validity.reason_codes),
            reason_code="frontier-timing-evidence-invalid",
        )
    timing = case.get("timing_contract")
    warmup = case.get("warmup_convergence")
    timer = timing.get("timer") if isinstance(timing, dict) else None
    assert isinstance(timing, dict)
    assert isinstance(warmup, dict)
    assert isinstance(timer, dict)
    threads = benchmark.get("torch_num_threads")
    if threads != environment.get("torch", {}).get("num_threads"):
        raise FrontierQualificationError(f"{bundle}: benchmark thread identity mismatch")

    cost_path, _ = _artifact(bundle, manifest, "cost-ir")
    cost = _load_json(cost_path)
    resolved_scope = case.get("resolved_scope")
    operations = [
        item
        for item in _objects(cost)
        if item.get("stable_path") == f"cost/{resolved_scope}"
        and isinstance(item.get("operation"), str)
    ]
    if len(operations) != 1:
        raise FrontierQualificationError(f"{bundle}: exact Cost IR operation is ambiguous")
    operation = operations[0]
    operand_types = operation.get("operand_types")
    result_types = operation.get("result_types")
    if (
        not isinstance(operand_types, list)
        or not operand_types
        or not isinstance(result_types, list)
        or len(result_types) != 1
    ):
        raise FrontierQualificationError(f"{bundle}: invalid exact-Shape Cost IR types")
    tensors = operand_types + result_types
    dtypes = {item.get("dtype") for item in tensors}
    if len(dtypes) != 1:
        raise FrontierQualificationError(f"{bundle}: mixed dtype exact Shape")
    operand_contracts = execution_contract.get("operand_contracts")
    result_contract = execution_contract.get("result_contract")
    if (
        not isinstance(operand_contracts, list)
        or len(operand_contracts) != len(operand_types)
        or not isinstance(result_contract, dict)
    ):
        raise FrontierQualificationError(f"{bundle}: exact tensor contracts are invalid")
    for authored, observed in zip(operand_types, operand_contracts, strict=True):
        if (
            observed.get("shape") != authored.get("shape")
            or observed.get("dtype") != authored.get("dtype")
            or _canonical_layout(str(observed.get("layout")))
            != _canonical_layout(str(authored.get("layout")))
        ):
            raise FrontierQualificationError(
                f"{bundle}: operand execution contract does not match exact Cost IR"
            )
    if (
        result_contract.get("shape") != result_types[0].get("shape")
        or result_contract.get("dtype") != result_types[0].get("dtype")
        or _canonical_layout(str(result_contract.get("layout")))
        != _canonical_layout(str(result_types[0].get("layout")))
    ):
        raise FrontierQualificationError(
            f"{bundle}: result execution contract does not match exact Cost IR"
        )
    operand_layouts = tuple(
        _canonical_layout(str(item.get("layout"))) for item in operand_contracts
    )
    result_layout = _canonical_layout(str(result_contract.get("layout")))
    layouts = {*operand_layouts, result_layout}
    if min(
        int(item["minimum_alignment_bytes"])
        for item in (*operand_contracts, result_contract)
    ) <= 0:
        raise FrontierQualificationError(
            f"{bundle}: exact tensor alignment is unresolved",
            reason_code="frontier-execution-contract-invalid",
        )

    return {
        "run_id": manifest["run_id"],
        "process_id": process["pid"],
        "run_bundle": str(bundle),
        "run_manifest_sha256": _sha256(manifest_path),
        "benchmark_sha256": benchmark_digest,
        "correctness_sha256": correctness_digest,
        "median_ns": timing_validity.median_ns,
        "iqr_over_median": timing_validity.iqr_over_median,
        "sample_count": len(timing_validity.samples_ns),
        "samples_ns": list(timing_validity.samples_ns),
        "measurement_hardware_cohort": cohort_id,
        "capability_hardware_cohort": capability_cohort,
        "target": target,
        "stable_path": resolved_scope,
        "semantic_operation": operation["operation"],
        "operand_shapes": [item["shape"] for item in operand_types],
        "result_shape": result_types[0]["shape"],
        "dtype": next(iter(dtypes)),
        "layout": next(iter(layouts)) if len(layouts) == 1 else "mixed-explicit",
        "operand_layouts": list(operand_layouts),
        "result_layout": result_layout,
        "threads": threads,
        "interop_threads": cohort.get("numeric_execution", {}).get(
            "interop_threads"
        ),
        "candidate_family": candidate_family,
        "candidate_digest": candidate_digest,
        "input_corpus_digest": input_corpus_digest,
        "execution_contract_digest": execution_contract_digest,
        "execution_contract": execution_contract,
        "correctness_oracle_policy_id": oracle.policy_id,
        "timing_scope": timing["timing_scope"],
        "completion_boundary": timing["completion_boundary"],
        "timer_source": timer["source"],
        "timer_resolution_ns": timing_validity.timer_resolution_ns,
        "instrumentation_profile": timing["instrumentation_profile"],
        "warmup_iterations": timing_validity.warmup_iterations,
        "warmup_window_samples_ns": list(
            timing_validity.warmup_window_samples_ns
        ),
        "warmup_median_drift": timing_validity.warmup_median_drift,
        "timed_duration_ns": timing_validity.timed_duration_ns,
    }


def qualify_exact_shape_frontier(
    *,
    search_runs: list[str],
    holdout_runs: list[str],
    case_id: str,
    stable_path_pattern: str,
    candidate_family: str,
    profile_name: str,
    profile_version: str,
    observation_output: str | Path,
    profile_output: str | Path,
    repository_root: str | Path = ".",
) -> OperatorFrontierProfileDocument:
    if len(search_runs) < 3 or len(holdout_runs) < 3:
        raise FrontierQualificationError("at least three search and holdout runs are required")
    search = [_session(path, case_id=case_id) for path in search_runs]
    holdout = [_session(path, case_id=case_id) for path in holdout_runs]
    sessions = search + holdout
    identity_fields = (
        "measurement_hardware_cohort",
        "target",
        "stable_path",
        "semantic_operation",
        "operand_shapes",
        "result_shape",
        "dtype",
        "layout",
        "operand_layouts",
        "result_layout",
        "threads",
        "interop_threads",
        "candidate_family",
        "candidate_digest",
        "input_corpus_digest",
        "execution_contract_digest",
        "correctness_oracle_policy_id",
        "timing_scope",
        "completion_boundary",
        "timer_source",
        "instrumentation_profile",
    )
    if any(
        session[field] != sessions[0][field]
        for session in sessions[1:]
        for field in identity_fields
    ):
        raise FrontierQualificationError("search and holdout exact-Shape identities differ")
    if str(sessions[0]["stable_path"]) != stable_path_pattern:
        raise FrontierQualificationError(
            "exact input corpus requires one exact stable path",
            reason_code="frontier-input-corpus-scope-mismatch",
        )
    if sessions[0]["candidate_family"] != candidate_family:
        raise FrontierQualificationError(
            "declared candidate family does not match measured candidate",
            reason_code="frontier-candidate-identity-mismatch",
        )
    run_ids = [str(item["run_id"]) for item in sessions]
    process_ids = [int(item["process_id"]) for item in sessions]
    if len(set(run_ids)) != len(run_ids) or len(set(process_ids)) != len(process_ids):
        raise FrontierQualificationError(
            "search and holdout sessions are not independent",
            reason_code="frontier-sessions-not-independent",
        )

    def relative_range(items: list[dict[str, Any]]) -> float:
        values = [float(item["median_ns"]) for item in items]
        center = float(median(values))
        return (max(values) - min(values)) / center

    search_medians = [float(item["median_ns"]) for item in search]
    holdout_medians = [float(item["median_ns"]) for item in holdout]
    search_center = float(median(search_medians))
    holdout_center = float(median(holdout_medians))
    if (
        relative_range(search) > 0.05
        or relative_range(holdout) > 0.05
        or abs(search_center - holdout_center) / holdout_center > 0.05
    ):
        raise FrontierQualificationError(
            "search/holdout sessions did not satisfy repeatability policy",
            reason_code="frontier-session-repeatability-failed",
        )

    root = Path(repository_root).resolve()
    def evidence(item: dict[str, Any]) -> dict[str, Any]:
        result = {
            key: item[key]
            for key in (
                "run_id",
                "process_id",
                "run_bundle",
                "run_manifest_sha256",
                "benchmark_sha256",
                "correctness_sha256",
                "candidate_digest",
                "input_corpus_digest",
                "execution_contract_digest",
                "correctness_oracle_policy_id",
                "timing_scope",
                "completion_boundary",
                "timer_source",
                "timer_resolution_ns",
                "instrumentation_profile",
                "warmup_iterations",
                "warmup_window_samples_ns",
                "warmup_median_drift",
                "timed_duration_ns",
                "median_ns",
                "iqr_over_median",
                "sample_count",
                "samples_ns",
            )
        }
        try:
            result["run_bundle"] = Path(str(item["run_bundle"])).relative_to(
                root
            ).as_posix()
        except ValueError:
            pass
        result["hardware_cohort"] = item["measurement_hardware_cohort"]
        return result

    anchor = {
        "anchor_id": f"{profile_name}-{case_id}-{profile_version}",
        "case_id": case_id,
        "stable_path_pattern": stable_path_pattern,
        "semantic_operation": sessions[0]["semantic_operation"],
        "operand_shapes": sessions[0]["operand_shapes"],
        "result_shape": sessions[0]["result_shape"],
        "dtype": sessions[0]["dtype"],
        "layout": sessions[0]["layout"],
        "operand_layouts": sessions[0]["operand_layouts"],
        "result_layout": sessions[0]["result_layout"],
        "operand_strides": [
            item["stride"]
            for item in sessions[0]["execution_contract"]["operand_contracts"]
        ],
        "result_stride": sessions[0]["execution_contract"]["result_contract"][
            "stride"
        ],
        "minimum_alignment_bytes": min(
            item["minimum_alignment_bytes"]
            for item in (
                *sessions[0]["execution_contract"]["operand_contracts"],
                sessions[0]["execution_contract"]["result_contract"],
            )
        ),
        "working_set_bytes": sessions[0]["execution_contract"][
            "working_set_bytes"
        ],
        "threads": sessions[0]["threads"],
        "interop_threads": sessions[0]["interop_threads"],
        "execution_mode": sessions[0]["execution_contract"]["execution_mode"],
        "candidate_family": sessions[0]["candidate_family"],
        "candidate_digest": sessions[0]["candidate_digest"],
        "input_corpus_digest": sessions[0]["input_corpus_digest"],
        "execution_contract_digest": sessions[0]["execution_contract_digest"],
        "timing_scope": sessions[0]["timing_scope"],
        "completion_boundary": sessions[0]["completion_boundary"],
        "instrumentation_profile": sessions[0]["instrumentation_profile"],
        "observation_validity": "QUALIFIED",
        "frontier_role": "ACTIVE",
        "latency_ns": float(median(holdout_medians)),
        "standard_uncertainty_ns": float(stdev(holdout_medians)),
        "search_run_ids": [item["run_id"] for item in search],
        "holdout_run_ids": [item["run_id"] for item in holdout],
        "measurement_hardware_cohort": sessions[0]["measurement_hardware_cohort"],
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
            "minimum_timed_duration_ns": 100_000_000,
            "estimator": "median(independent_holdout_session_medians)",
            "uncertainty": "sample-standard-deviation(independent_holdout_session_medians)",
        },
        "candidate_coverage": {
            "level": "C0_SINGLE",
            "scope": "declared-runtime-candidate-family",
            "evaluated_candidate_families": [sessions[0]["candidate_family"]],
            "selected_candidate_family": sessions[0]["candidate_family"],
            "evaluated_candidate_digests": [sessions[0]["candidate_digest"]],
            "selected_candidate_digest": sessions[0]["candidate_digest"],
            "limitation": "does-not-establish-global-optimum-or-support-frontier-shift",
        },
        "session_evidence": {
            "search": [evidence(item) for item in search],
            "holdout": [evidence(item) for item in holdout],
        },
    }
    observation = {
        "schema": "groundupscale.dev/operator-frontier-observation/v1alpha1",
        "target": sessions[0]["target"],
        "hardware_cohort": sessions[0]["measurement_hardware_cohort"],
        "anchors": [anchor],
    }
    observation_path = Path(observation_output).resolve()
    profile_path = Path(profile_output).resolve()
    if observation_path.exists() or profile_path.exists():
        raise FrontierQualificationError(
            "frontier evidence/profile outputs are immutable and must not exist"
        )
    observation_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    observation_bytes = (
        json.dumps(canonical_data(observation), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    try:
        source_path = observation_path.relative_to(root).as_posix()
    except ValueError:
        source_path = str(observation_path)
    document = OperatorFrontierProfileDocument.model_validate(
        {
            "apiVersion": "groundupscale.dev/v1alpha1",
            "kind": "OperatorFrontierProfile",
            "metadata": {"name": profile_name, "version": profile_version},
            "spec": {
                "target": observation["target"],
                "hardware_cohort": observation["hardware_cohort"],
                "source": {
                    "path": source_path,
                    "sha256": sha256(observation_bytes).hexdigest(),
                    "schema": observation["schema"],
                },
                "anchors": observation["anchors"],
            },
        }
    )
    profile_bytes = yaml.safe_dump(
        document.model_dump(mode="json", by_alias=True),
        sort_keys=False,
        allow_unicode=True,
    ).encode("utf-8")
    with observation_path.open("xb") as stream:
        stream.write(observation_bytes)
    with profile_path.open("xb") as stream:
        stream.write(profile_bytes)
    return document


__all__ = ["FrontierQualificationError", "qualify_exact_shape_frontier"]
