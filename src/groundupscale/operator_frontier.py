"""Evidence-qualified Ascend NPU MatMul Frontier construction."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from math import hypot, isfinite
from pathlib import Path
from statistics import median, stdev
from typing import Any, cast

from groundupscale.ir import content_fingerprint
from groundupscale.operator_shape_semantics import (
    OperatorShapeSemantics,
    UnsupportedOperatorShape,
    semantics_for_coordinate,
    semantics_from_case,
)
from groundupscale.run_bundle import (
    RUN_ID_PATTERN,
    RunBundleExistsError,
    verify_run_bundle,
)

QUALIFICATION_SCHEMA = (
    "groundupscale.dev/operator-frontier-qualification/v1alpha1"
)
QUALIFICATION_POLICY_SCHEMA = (
    "groundupscale.dev/operator-frontier-qualification-policy/v1alpha1"
)
DIAGNOSTIC_EVIDENCE_SCHEMA = (
    "groundupscale.dev/diagnostic-evidence/v1alpha1"
)


class OperatorFrontierQualificationError(ValueError):
    """Source measurement evidence cannot qualify an authoritative Frontier."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class _QualificationPolicy:
    document: dict[str, Any]
    digest: str
    minimum_search_sessions: int
    minimum_holdout_sessions: int
    minimum_confirmation_sessions: int
    minimum_warmup_iterations: int
    maximum_session_median_relative_range: float
    minimum_candidate_coverage: str
    target_coverage: float

    @classmethod
    def from_document(
        cls, value: dict[str, object] | None
    ) -> _QualificationPolicy:
        if value is None:
            raise OperatorFrontierQualificationError(
                "an explicit versioned qualification policy is required",
                reason_code="missing-qualification-policy",
            )
        document = dict(value)
        scope = document.get("scope")
        required_strings = (
            "policy_id",
            "version",
            "change_reason",
            "revalidation",
            "evidence_ref",
        )
        integer_fields = (
            "minimum_search_sessions",
            "minimum_holdout_sessions",
            "minimum_confirmation_sessions",
            "minimum_warmup_iterations",
        )
        if (
            document.get("schema") != QUALIFICATION_POLICY_SCHEMA
            or not isinstance(scope, dict)
            or any(
                not isinstance(document.get(field), str)
                or not str(document[field]).strip()
                for field in required_strings
            )
            or any(
                not isinstance(document.get(field), int)
                or isinstance(document.get(field), bool)
                or cast(int, document[field]) < 1
                for field in integer_fields
            )
            or document.get("holdout_candidate_scope")
            != "all-eligible-candidates"
            or document.get("sample_exclusion")
            != "none-preserve-all-raw-samples"
            or document.get("estimator")
            != "median(independent-holdout-session-medians)"
            or isinstance(document.get("collection_plan"), dict)
            and any(cast(int, document[field]) < 3 for field in integer_fields[:3])
        ):
            raise OperatorFrontierQualificationError(
                "qualification policy is incomplete or unsupported",
                reason_code="invalid-qualification-policy",
            )
        maximum_range = document.get(
            "maximum_session_median_relative_range"
        )
        minimum_coverage = document.get("minimum_candidate_coverage")
        target_coverage = document.get("target_coverage")
        if (
            not isinstance(maximum_range, (int, float))
            or isinstance(maximum_range, bool)
            or not 0 < float(maximum_range) <= 1
            or minimum_coverage
            not in {"C0_SINGLE", "C1_SAME_FAMILY", "C2_MULTI_FAMILY"}
            or document.get("uncertainty_combination")
            != "root-sum-of-squares"
            or not isinstance(target_coverage, (int, float))
            or isinstance(target_coverage, bool)
            or not 0 < float(target_coverage) <= 1
        ):
            raise OperatorFrontierQualificationError(
                "qualification policy has invalid thresholds",
                reason_code="invalid-qualification-policy",
            )
        return cls(
            document=document,
            digest=_canonical_digest(document),
            minimum_search_sessions=cast(
                int, document["minimum_search_sessions"]
            ),
            minimum_holdout_sessions=cast(
                int, document["minimum_holdout_sessions"]
            ),
            minimum_confirmation_sessions=cast(
                int, document["minimum_confirmation_sessions"]
            ),
            minimum_warmup_iterations=cast(
                int, document["minimum_warmup_iterations"]
            ),
            maximum_session_median_relative_range=float(maximum_range),
            minimum_candidate_coverage=str(minimum_coverage),
            target_coverage=float(target_coverage),
        )


@dataclass(frozen=True)
class _Observation:
    root: Path
    manifest_sha256: str
    run_id: str
    cohort_id: str
    operator_shape: OperatorShapeSemantics
    m: int
    n: int
    k: int
    candidate_id: str
    candidate_family: str
    candidate_digest: str
    candidate_protocol_digest: str
    candidate_identity: dict[str, Any]
    candidate_evidence_sha256: str
    correctness: str
    timing_quality: str
    median_ns: float
    samples_ns: tuple[int, ...]
    timer_source: str
    timer_resolution_ns: float
    completion_kind: str
    completion_protocol: str
    instrumentation_profile: str
    dtype: str
    layout: str
    alignment_bytes: int
    seed: int
    input_identity: tuple[str, ...]
    execution_contract_digest: str
    execution_protocol_digest: str
    execution_mode: str
    runtime_device_name: str
    logical_device: str
    warmup_iterations: int
    repetitions: int
    inner_iterations: int
    process_identity: tuple[int, str]

    @property
    def evidence_ref(self) -> str:
        return f"artifact://frontier/qualification.json#source-run-{self.run_id}"


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OperatorFrontierQualificationError(
            f"{path}: expected a JSON object",
            reason_code="invalid-source-artifact",
        )
    return value


def _artifact(
    root: Path,
    manifest: dict[str, Any],
    role: str,
) -> tuple[dict[str, Any], str]:
    artifacts = manifest.get("artifacts")
    matches = [
        item
        for item in artifacts
        if isinstance(item, dict) and item.get("role") == role
    ] if isinstance(artifacts, list) else []
    if len(matches) != 1:
        raise OperatorFrontierQualificationError(
            f"{root}: expected exactly one {role} artifact",
            reason_code="invalid-source-artifact",
        )
    entry = matches[0]
    path = (root / str(entry.get("path", ""))).resolve()
    if root not in path.parents or not path.is_file():
        raise OperatorFrontierQualificationError(
            f"{root}: invalid {role} artifact path",
            reason_code="invalid-source-artifact",
        )
    return _load_json(path), str(entry.get("sha256", ""))


def _observation(path: str | Path) -> _Observation:
    root = Path(path).resolve()
    verification = verify_run_bundle(root)
    if verification.get("passed") is not True:
        raise OperatorFrontierQualificationError(
            f"{root}: source Run Bundle failed verification",
            reason_code="source-run-verification-failed",
        )
    manifest_path = root / "run.manifest.json"
    manifest = _load_json(manifest_path)
    if (
        manifest.get("bundle_kind") != "exact-shape-measurement"
        or manifest.get("status") != "completed"
        or manifest.get("device") != "ascend-npu"
        or not isinstance(manifest.get("hardware_cohort"), str)
    ):
        raise OperatorFrontierQualificationError(
            f"{root}: not a completed Ascend exact-Shape measurement",
            reason_code="invalid-source-run-kind",
        )

    case, _ = _artifact(root, manifest, "benchmark-case")
    candidate, candidate_evidence_sha256 = _artifact(
        root, manifest, "candidate-identity"
    )
    correctness, _ = _artifact(root, manifest, "correctness-observation")
    raw_timing, _ = _artifact(root, manifest, "raw-timing-observation")
    completion, _ = _artifact(root, manifest, "completion-boundary")
    instrumentation, _ = _artifact(root, manifest, "instrumentation-profile")
    input_corpus, _ = _artifact(root, manifest, "input-corpus")
    execution, _ = _artifact(root, manifest, "execution-contract")
    environment, _ = _artifact(root, manifest, "environment")
    preflight, _ = _artifact(root, manifest, "measurement-preflight")
    timing_plan, _ = _artifact(root, manifest, "timing-plan")

    try:
        operator_shape = semantics_from_case(case)
    except UnsupportedOperatorShape as error:
        raise OperatorFrontierQualificationError(
            f"{root}: {error}",
            reason_code="unsupported-shape-regime",
        ) from error
    candidate_body = dict(candidate)
    candidate_digest = candidate_body.pop("candidate_digest", None)
    candidate_protocol_body = {
        key: value
        for key, value in candidate_body.items()
        if key
        not in {
            "shape",
            "operator_shape_identity",
            "minimum_alignment_bytes",
        }
    }
    session = environment.get("measurement_session")
    summary = raw_timing.get("summary")
    samples = raw_timing.get("samples")
    timer_resolution_ns = raw_timing.get("timer_resolution_ns")
    seed = case.get("seed")
    warmup_iterations = execution.get("warmup_iterations")
    repetitions = execution.get("repetitions")
    inner_iterations = execution.get("inner_iterations")
    logical_device = execution.get("logical_device")
    observation_validity = manifest.get("observation_validity")
    if (
        not isinstance(candidate_digest, str)
        or candidate_digest != content_fingerprint(candidate_body)
        or not isinstance(summary, dict)
        or not isinstance(samples, list)
        or not samples
        or not all(
            isinstance(sample, int)
            and not isinstance(sample, bool)
            and sample > 0
            for sample in samples
        )
        or not isinstance(session, dict)
        or not isinstance(session.get("process_id"), int)
        or not isinstance(session.get("process_started_at"), str)
        or not isinstance(timer_resolution_ns, (int, float))
        or isinstance(timer_resolution_ns, bool)
        or timer_resolution_ns <= 0
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or not isinstance(warmup_iterations, int)
        or isinstance(warmup_iterations, bool)
        or warmup_iterations < 0
        or not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or repetitions < 1
        or not isinstance(inner_iterations, int)
        or isinstance(inner_iterations, bool)
        or inner_iterations < 1
        or timing_plan.get("warmup_iterations") != warmup_iterations
        or timing_plan.get("repetitions") != repetitions
        or timing_plan.get("inner_iterations") != inner_iterations
        or not isinstance(timing_plan.get("case"), dict)
        or timing_plan["case"].get("seed") != seed
        or timing_plan["case"].get("warmup_iterations") != warmup_iterations
        or timing_plan["case"].get("repetitions") != repetitions
        or input_corpus.get("seed") != seed
        or not isinstance(logical_device, str)
        or not logical_device
        or preflight.get("eligible") is not True
        or completion.get("closed") is not True
        or instrumentation.get("lane") != "baseline-timing"
        or not isinstance(observation_validity, dict)
    ):
        raise OperatorFrontierQualificationError(
            f"{root}: incomplete qualification identity",
            reason_code="incomplete-qualification-identity",
        )
    if operator_shape.operation == "MatMul":
        identities = (
            input_corpus.get("left_sha256"),
            input_corpus.get("right_sha256"),
        )
    else:
        identities = (
            input_corpus.get("q_sha256"),
            input_corpus.get("k_sha256"),
            input_corpus.get("v_sha256"),
        )
    if any(
        not isinstance(identity, str) or len(identity) != 64
        for identity in identities
    ) or (
        operator_shape.operation == "FlashAttentionForward"
        and input_corpus.get("sequence_shape_identity")
        != operator_shape.shape_identity
    ):
        raise OperatorFrontierQualificationError(
            f"{root}: incomplete input identity",
            reason_code="incomplete-qualification-identity",
        )
    input_identity = cast(tuple[str, ...], identities)
    execution_body = {
        key: value
        for key, value in execution.items()
        if key != "candidate"
    }
    execution_protocol_body = {
        key: value
        for key, value in execution_body.items()
        if key != "shape"
    }
    normalized_shape = operator_shape.normalized_shape
    return _Observation(
        root=root,
        manifest_sha256=_sha256(manifest_path),
        run_id=str(manifest["run_id"]),
        cohort_id=str(manifest["hardware_cohort"]),
        operator_shape=operator_shape,
        m=int(normalized_shape.get("m", operator_shape.coordinate_value or 0)),
        n=int(normalized_shape.get("n", 0)),
        k=int(normalized_shape.get("k", 0)),
        candidate_id=str(candidate["candidate_id"]),
        candidate_family=str(candidate["candidate_family"]),
        candidate_digest=candidate_digest,
        candidate_protocol_digest=_canonical_digest(candidate_protocol_body),
        candidate_identity=candidate,
        candidate_evidence_sha256=candidate_evidence_sha256,
        correctness=str(correctness.get("status")),
        timing_quality=str(observation_validity.get("timing_quality")),
        median_ns=float(summary["median"]),
        samples_ns=tuple(samples),
        timer_source=str(raw_timing.get("timer_source")),
        # Legacy averaged integer samples may report the divided device-event
        # resolution.  Qualification remains conservative at the serialization
        # boundary, where the stored samples cannot resolve below one ns.
        timer_resolution_ns=max(1.0, float(timer_resolution_ns)),
        completion_kind=str(completion.get("kind")),
        completion_protocol=str(completion.get("protocol")),
        instrumentation_profile=str(instrumentation.get("profile_id")),
        dtype=str(case.get("dtype")),
        layout=str(case.get("layout")),
        alignment_bytes=int(candidate.get("minimum_alignment_bytes", 0)),
        seed=seed,
        input_identity=input_identity,
        execution_contract_digest=_canonical_digest(execution_body),
        execution_protocol_digest=_canonical_digest(execution_protocol_body),
        execution_mode=str(candidate.get("execution_mode")),
        runtime_device_name=str(candidate.get("runtime_device_name")),
        logical_device=logical_device,
        warmup_iterations=warmup_iterations,
        repetitions=repetitions,
        inner_iterations=inner_iterations,
        process_identity=(
            int(session["process_id"]),
            str(session["process_started_at"]),
        ),
    )


def _relative_range(values: Sequence[float]) -> float:
    center = float(median(values))
    return (max(values) - min(values)) / center


def _coverage_level(
    candidate_ids: set[str], candidate_families: set[str]
) -> str:
    if len(candidate_families) >= 2:
        return "C2_MULTI_FAMILY"
    if len(candidate_ids) >= 2:
        return "C1_SAME_FAMILY"
    return "C0_SINGLE"


def _coverage_satisfies(observed: str, required: str) -> bool:
    ranks = {
        "C0_SINGLE": 0,
        "C1_SAME_FAMILY": 1,
        "C2_MULTI_FAMILY": 2,
    }
    return ranks[observed] >= ranks[required]


def _require_policy_scope(
    policy: _QualificationPolicy,
    observations: Sequence[_Observation],
    *,
    cohort_id: str,
    anchor_shapes: Sequence[int],
    confirmation_shape: int,
    candidate_ids: Sequence[str],
) -> None:
    scope = policy.document["scope"]
    assert isinstance(scope, dict)
    if policy.document.get("response_target") == "latency":
        expected = {
            "hardware_cohort": cohort_id,
            "operation": "MatMul",
            "dtype": observations[0].dtype,
            "layout": observations[0].layout,
            "fixed_n": observations[0].n,
            "fixed_k": observations[0].k,
            "anchor_m": list(anchor_shapes),
            "confirmation_m": confirmation_shape,
            "candidate_ids": list(candidate_ids),
        }
    else:
        expected = {
            "hardware_cohort": cohort_id,
            "operation": "MatMul",
            "dtype": observations[0].dtype,
            "layout": observations[0].layout,
            "anchor_shapes": list(anchor_shapes),
            "confirmation_shape": confirmation_shape,
            "candidate_ids": list(candidate_ids),
        }
    if any(scope.get(key) != value for key, value in expected.items()):
        raise OperatorFrontierQualificationError(
            "qualification policy does not cover this evidence cohort",
            reason_code="qualification-policy-scope-mismatch",
        )
    if any(
        item.warmup_iterations < policy.minimum_warmup_iterations
        for item in observations
    ):
        raise OperatorFrontierQualificationError(
            "source evidence does not meet the versioned warmup policy",
            reason_code="warmup-policy-failed",
        )


def _require_bounded_collection_rounds(policy: _QualificationPolicy) -> None:
    collection_plan = policy.document.get("collection_plan")
    if collection_plan is None:
        return
    maximum_rounds = (
        collection_plan.get("maximum_supplemental_rounds")
        if isinstance(collection_plan, dict)
        else None
    )
    executed_rounds = (
        collection_plan.get("executed_supplemental_rounds")
        if isinstance(collection_plan, dict)
        else None
    )
    if (
        not isinstance(maximum_rounds, int)
        or isinstance(maximum_rounds, bool)
        or maximum_rounds < 0
        or not isinstance(executed_rounds, int)
        or isinstance(executed_rounds, bool)
        or executed_rounds < 0
        or executed_rounds > maximum_rounds
    ):
        raise OperatorFrontierQualificationError(
            "collection evidence exceeds the declared supplemental-round limit",
            reason_code="bounded-collection-plan-violated",
        )


def _require_collection_plan_identity(
    policy: _QualificationPolicy,
    observations: Sequence[_Observation],
) -> None:
    collection_plan = policy.document.get("collection_plan")
    if collection_plan is None:
        return
    if not isinstance(collection_plan, dict) or not observations:
        raise OperatorFrontierQualificationError(
            "collection plan is incomplete",
            reason_code="collection-plan-evidence-mismatch",
        )
    first = observations[0]
    domain_expected = (
        {"fixed_n": first.n, "fixed_k": first.k}
        if first.operator_shape.operation == "MatMul"
        else {
            key: first.operator_shape.domain_facets[key]
            for key in (
                "operation",
                "sequence_count",
                "head_count",
                "head_dimension",
                "causal",
                "mask",
                "dropout_probability",
                "mode",
            )
            if key in first.operator_shape.domain_facets
        }
    )
    if first.operator_shape.operation == "FlashAttentionForward":
        domain_expected["operation"] = first.operator_shape.operation
    expected = {
        "hardware_cohort": first.cohort_id,
        "dtype": first.dtype,
        "layout": first.layout,
        "candidate_ids": sorted({item.candidate_id for item in observations}),
        "execution_mode": first.execution_mode,
        "warmup_iterations": first.warmup_iterations,
        "repetitions": first.repetitions,
        "inner_iterations": first.inner_iterations,
        "completion_boundary": first.completion_protocol,
        "instrumentation_profile": first.instrumentation_profile,
        "random_seed": first.seed,
        "raw_sample_retention": "preserve-all-no-selective-exclusion",
        **domain_expected,
    }
    if any(collection_plan.get(key) != value for key, value in expected.items()) or any(
        item.cohort_id != first.cohort_id
        or item.operator_shape.operation != first.operator_shape.operation
        or item.operator_shape.domain_facets
        != first.operator_shape.domain_facets
        or item.dtype != first.dtype
        or item.layout != first.layout
        or item.execution_mode != first.execution_mode
        or item.warmup_iterations != first.warmup_iterations
        or item.repetitions != first.repetitions
        or item.inner_iterations != first.inner_iterations
        or item.completion_protocol != first.completion_protocol
        or item.instrumentation_profile != first.instrumentation_profile
        or item.seed != first.seed
        for item in observations
    ):
        raise OperatorFrontierQualificationError(
            "collection plan identity does not match source evidence",
            reason_code="collection-plan-evidence-mismatch",
        )


def _require_independent_sessions(observations: Sequence[_Observation]) -> None:
    run_ids = [item.run_id for item in observations]
    process_ids = [item.process_identity for item in observations]
    roots = [item.root for item in observations]
    if (
        len(run_ids) != len(set(run_ids))
        or len(process_ids) != len(set(process_ids))
        or len(roots) != len(set(roots))
    ):
        raise OperatorFrontierQualificationError(
            "search, holdout, and confirmation sessions must be disjoint",
            reason_code="sessions-not-independent",
        )


def _require_common_identity(observations: Sequence[_Observation]) -> str:
    cohorts = {item.cohort_id for item in observations}
    if len(cohorts) != 1:
        raise OperatorFrontierQualificationError(
            "source runs span multiple Hardware Cohorts",
            reason_code="hardware-cohort-mismatch",
        )
    common = {
        (
            item.dtype,
            item.layout,
            item.timer_source,
            item.timer_resolution_ns,
            item.completion_kind,
            item.completion_protocol,
            item.instrumentation_profile,
            item.execution_mode,
            item.runtime_device_name,
            item.logical_device,
            item.operator_shape.operation,
            _canonical_digest(item.operator_shape.domain_facets),
        )
        for item in observations
    }
    if len(common) != 1 or any(item.alignment_bytes < 64 for item in observations):
        raise OperatorFrontierQualificationError(
            "source runs do not share one execution and timing contract",
            reason_code="execution-contract-mismatch",
        )
    if len({item.execution_protocol_digest for item in observations}) != 1:
        raise OperatorFrontierQualificationError(
            "source runs changed the cross-Shape execution protocol",
            reason_code="execution-contract-mismatch",
        )
    candidate_protocols: dict[str, set[str]] = {}
    for item in observations:
        candidate_protocols.setdefault(item.candidate_id, set()).add(
            item.candidate_protocol_digest
        )
    if any(len(digests) != 1 for digests in candidate_protocols.values()):
        raise OperatorFrontierQualificationError(
            "source runs changed candidate build or runtime identity",
            reason_code="candidate-identity-changed",
        )
    return next(iter(cohorts))


def _same_shape_input_and_contract(observations: Sequence[_Observation]) -> bool:
    return len(
        {
            (
                item.operator_shape.shape_identity,
                item.seed,
                item.input_identity,
                item.execution_contract_digest,
            )
            for item in observations
        }
    ) == 1


def _active_transitions(anchor_id: str) -> list[dict[str, object]]:
    evidence_ref = "artifact://frontier/qualification.json"
    return [
        {
            "sequence": 1,
            "axis": "frontier_role",
            "from": "NONE",
            "to": "PROVISIONAL",
            "reason_code": "exact-shape-best-of-correct-holdout-winner",
            "evidence_refs": [evidence_ref],
        },
        {
            "sequence": 2,
            "axis": "observation_validity",
            "from": "COLLECTED",
            "to": "QUALIFIED",
            "reason_code": "qualification-gates-satisfied",
            "evidence_refs": [evidence_ref],
        },
        {
            "sequence": 3,
            "axis": "frontier_role",
            "from": "PROVISIONAL",
            "to": "ACTIVE",
            "reason_code": "independent-holdout-confirmed",
            "evidence_refs": [evidence_ref],
        },
    ]


def _source_record(
    observation: _Observation,
    *,
    lane: str,
    bundle_root: Path,
) -> dict[str, object]:
    return {
        "run_id": observation.run_id,
        "lane": lane,
        "path": os.path.relpath(observation.root, bundle_root),
        "manifest_sha256": observation.manifest_sha256,
        "hardware_cohort": observation.cohort_id,
        "semantic_operation": observation.operator_shape.operation,
        "shape": observation.operator_shape.normalized_shape,
        "shape_identity": observation.operator_shape.shape_identity,
        "declared_work": observation.operator_shape.declared_work,
        "work_formula": observation.operator_shape.work_formula,
        "candidate_id": observation.candidate_id,
        "candidate_family": observation.candidate_family,
        "candidate_digest": observation.candidate_digest,
        "candidate_protocol_digest": observation.candidate_protocol_digest,
        "candidate_identity": observation.candidate_identity,
        "candidate_evidence_sha256": observation.candidate_evidence_sha256,
        "correctness": observation.correctness,
        "timing_quality": observation.timing_quality,
        "raw_samples_ns": list(observation.samples_ns),
        "median_ns": observation.median_ns,
        "warmup_iterations": observation.warmup_iterations,
        "repetitions": observation.repetitions,
        "inner_iterations": observation.inner_iterations,
        "timer": {
            "source": observation.timer_source,
            "resolution_ns": observation.timer_resolution_ns,
        },
        "completion_boundary": {
            "kind": observation.completion_kind,
            "protocol": observation.completion_protocol,
        },
        "execution_protocol_digest": observation.execution_protocol_digest,
        "process_identity": {
            "process_id": observation.process_identity[0],
            "process_started_at": observation.process_identity[1],
        },
        "evidence_ref": observation.evidence_ref,
    }


def _write_operator_frontier_documents(
    runs_root: Path,
    destination: Path,
    *,
    run_id: str,
    cohort_id: str,
    source_records: list[dict[str, object]],
    surface: dict[str, object],
    qualification: dict[str, object],
    queries: list[dict[str, object]],
    analysis_plan: str,
    runtime_device_name: str,
    logical_device: str,
) -> Path:
    policy_document = qualification["policy"]
    semantic_operation = cast(dict[str, object], surface["domain"])[
        "semantic_operation"
    ]
    coordinate = cast(dict[str, object], surface["coordinate"])["axis"]
    inputs = {
        "resolved_configuration": {
            "analysis_plan": analysis_plan,
            "qualification_policy": policy_document,
        },
        "resolved_ir": {
            "semantic_operation": semantic_operation,
            "shape_coordinate": coordinate,
        },
        "hardware": {
            "device": runtime_device_name,
            "partition": logical_device,
            "cohort_id": cohort_id,
        },
        "cohort_id": cohort_id,
        "execution_domain": surface["domain"],
    }
    evidence = {
        "capability_surfaces": [surface],
        "surface_queries": queries,
        "operator_frontier_qualification_ref": "artifact://frontier/qualification.json",
    }
    diagnostic = {
        "schema": DIAGNOSTIC_EVIDENCE_SCHEMA,
        **inputs,
        **evidence,
        "digests": {
            "input_sha256": _canonical_digest(inputs),
            "evidence_sha256": _canonical_digest(evidence),
        },
    }
    runs_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
    artifacts = []
    for role, relative_path, document, inputs_list in (
        ("operator-frontier-qualification", "frontier/qualification.json", qualification, []),
        ("diagnostic-evidence", "diagnostic/evidence.json", diagnostic, ["operator-frontier-qualification"]),
    ):
        path = temporary / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_json_bytes(document))
        artifacts.append(
            {
                "role": role,
                "path": relative_path,
                "media_type": "application/json",
                "schema": document["schema"],
                "sha256": _sha256(path),
                "produced_by": "groundupscale-operator-frontier-v1",
                "inputs": inputs_list,
            }
        )
    manifest = {
        "schema": "groundupscale.dev/run-manifest/v1alpha1",
        "bundle_kind": "operator-frontier",
        "run_id": run_id,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "device": "ascend-npu",
        "hardware_cohort": cohort_id,
        "surface": {
            "surface_id": surface["surface_id"],
            "version": surface["version"],
            "input_digest": surface["input_digest"],
        },
        "source_runs": source_records,
        "artifacts": artifacts,
        "immutability": "writer refuses an existing run_id; source and artifact digests are authoritative",
    }
    (temporary / "run.manifest.json").write_bytes(_json_bytes(manifest))
    os.replace(temporary, destination)
    return destination


def _write_exact_distribution_bundle(
    artifact_store: str | Path,
    *,
    run_id: str,
    policy: _QualificationPolicy,
    searches: Sequence[_Observation],
    holdouts: Sequence[_Observation],
    query_shapes: Sequence[dict[str, object]],
) -> Path:
    observations = (*searches, *holdouts)
    _require_independent_sessions(observations)
    cohort_id = _require_common_identity(observations)
    first = observations[0]
    if first.operator_shape.operation != "FlashAttentionForward":
        raise OperatorFrontierQualificationError(
            "exact distribution qualification requires ragged TND evidence",
            reason_code="unsupported-shape-regime",
        )
    scope = policy.document["scope"]
    assert isinstance(scope, dict)
    vectors = scope.get("sequence_vectors")
    if (
        scope.get("sequence_distribution_mode") != "exact-only"
        or not isinstance(vectors, list)
        or sorted(tuple(item) for item in vectors if isinstance(item, list))
        != sorted(
            {
                tuple(cast(list[int], item.operator_shape.normalized_shape["sequence_lengths"]))
                for item in observations
            }
        )
    ):
        raise OperatorFrontierQualificationError(
            "qualification policy does not cover ragged sequence identities",
            reason_code="qualification-policy-scope-mismatch",
        )
    expected_scope = {
        "hardware_cohort": cohort_id,
        "operation": first.operator_shape.operation,
        **{
            key: value
            for key, value in first.operator_shape.domain_facets.items()
            if key != "semantic_operation"
        },
        "candidate_ids": sorted({item.candidate_id for item in observations}),
    }
    if any(scope.get(key) != value for key, value in expected_scope.items()):
        raise OperatorFrontierQualificationError(
            "qualification policy does not cover ragged sequence domain",
            reason_code="qualification-policy-scope-mismatch",
        )
    if any(
        item.correctness != "passed"
        or item.timing_quality != "passed"
        or item.warmup_iterations < policy.minimum_warmup_iterations
        for item in observations
    ):
        raise OperatorFrontierQualificationError(
            "ragged evidence failed qualification",
            reason_code="independent-holdout-failed",
        )
    search_by_identity = {
        identity: [
            item for item in searches if item.operator_shape.shape_identity == identity
        ]
        for identity in {item.operator_shape.shape_identity for item in searches}
    }
    holdout_by_identity = {
        identity: [
            item for item in holdouts if item.operator_shape.shape_identity == identity
        ]
        for identity in search_by_identity
    }
    if any(
        len(search_by_identity[identity]) < policy.minimum_search_sessions
        or len(holdout_by_identity[identity]) < policy.minimum_holdout_sessions
        or not _same_shape_input_and_contract(search_by_identity[identity])
        or not _same_shape_input_and_contract(holdout_by_identity[identity])
        for identity in search_by_identity
    ):
        raise OperatorFrontierQualificationError(
            "ragged evidence lacks independent exact-Shape sessions",
            reason_code="independent-holdout-failed",
        )
    runs_root = Path(artifact_store).resolve() / "runs"
    destination = runs_root / run_id
    if destination.exists():
        raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
    source_records = [
        *(
            _source_record(item, lane="search", bundle_root=destination)
            for item in searches
        ),
        *(
            _source_record(item, lane="holdout", bundle_root=destination)
            for item in holdouts
        ),
    ]
    evidence_digest = _canonical_digest(
        {
            "policy": policy.digest,
            "sources": sorted(item.manifest_sha256 for item in observations),
        }
    )
    domain = {
        **first.operator_shape.domain_facets,
        "sequence_distribution": "exact-only",
        "alignment_regime": "minimum-64-byte",
        "alignment_validated": True,
        "working_set_regime": "ragged-sequence-vector-exact-anchor",
        "working_set_validated": True,
        "kernel_dispatch_regime": first.candidate_family,
        "kernel_dispatch_validated": True,
        "regime_validated": True,
        "execution_mode": first.execution_mode,
        "threads": 1,
        "candidate_protocol_digest": first.candidate_protocol_digest,
        "candidate_set_digest": _canonical_digest(
            sorted({item.candidate_protocol_digest for item in observations})
        ),
        "execution_protocol_digest": first.execution_protocol_digest,
    }
    anchors: list[dict[str, object]] = []
    anchor_latency_variances: list[float] = []
    for identity, search_records in sorted(search_by_identity.items()):
        holdout_records = holdout_by_identity[identity]
        semantics = search_records[0].operator_shape
        medians = [item.median_ns for item in holdout_records]
        latency_ns = float(median(medians))
        rates = [semantics.declared_work / (value * 1e-9) for value in medians]
        standard_latency = float(stdev(medians))
        anchor_latency_variances.append(standard_latency**2)
        anchor_id = f"ascend-flash-attention-ragged-{identity.rsplit('/', 1)[-1][:12]}"
        anchors.append(
            {
                "anchor_id": anchor_id,
                "anchor_version": f"v-{evidence_digest[:16]}",
                "shape": {
                    "sequence_distribution_index": len(anchors) + 1
                },
                "operator_shape_identity": identity,
                "normalized_operator_shape": semantics.normalized_shape,
                "effective_rate": float(median(rates)),
                "rate_unit": "FLOP/s",
                "candidate_id": search_records[0].candidate_id,
                "candidate_family": search_records[0].candidate_family,
                "candidate_digest": search_records[0].candidate_digest,
                "candidate_protocol_digest": search_records[0].candidate_protocol_digest,
                "execution_protocol_digest": search_records[0].execution_protocol_digest,
                "cohort_id": cohort_id,
                "domain": domain,
                "observation_validity": "QUALIFIED",
                "frontier_role": "ACTIVE",
                "evidence_ref": "artifact://frontier/qualification.json",
                "state_transitions": _active_transitions(anchor_id),
                "latency_ns": latency_ns,
                "declared_work": semantics.declared_work,
                "standard_uncertainty_rate": float(stdev(rates)),
                "standard_uncertainty_latency_ns": standard_latency,
            }
        )
    surface: dict[str, object] = {
        "surface_id": f"surface://{cohort_id}/flash-attention/tnd-forward/ragged/{evidence_digest[:16]}",
        "version": f"v-{evidence_digest[:16]}",
        "previous_version": None,
        "qualification_status": "qualified",
        "cohort_id": cohort_id,
        "domain": domain,
        "candidate_family": first.candidate_family,
        "anchor_lifecycle_policy": {
            "policy_id": "frontier-anchor-lifecycle",
            "version": "v2",
            "scope": f"{cohort_id}-flash-attention-ragged-exact",
            "change_reason": "issue-37 exact-only ragged TND identity",
            "revalidation": "on cohort, domain, candidate, vector, evidence, or policy change",
        },
        "coordinate": {
            "axis": "sequence_distribution_index",
            "transform": "identity",
            "transform_version": "v1",
        },
        "work_formula": first.operator_shape.work_formula,
        "anchors": anchors,
        "cells": [
            {
                "cell_id": f"flash-attention-ragged-exact-{index}",
                "anchor_ids": [anchor["anchor_id"], anchor["anchor_id"]],
                "status": "retained",
                "regime_id": "ragged-exact-anchor",
                "confirmation_shape": anchor["shape"],
                "confirmation_observed_rate": anchor["effective_rate"],
                "confirmation_evidence_refs": [anchor["evidence_ref"]],
                "interpolation_standard_uncertainty_rate": 0.0,
            }
            for index, anchor in enumerate(anchors, start=1)
        ],
        "uncertainty_policy": {
            "policy_id": "ascend-flash-attention-ragged-exact-uncertainty",
            "version": "v1",
            "scope": "qualified exact sequence vectors only",
            "change_reason": "independent exact-Shape holdout",
            "revalidation": "on cohort, vector, evidence, or policy change",
            "combination": policy.document["uncertainty_combination"],
            "target_coverage": policy.target_coverage,
            "anchor_covariance": [
                [0.0 for _ in anchors] for _ in anchors
            ],
            "anchor_latency_covariance": [
                [variance if row == column else 0.0 for column, variance in enumerate(anchor_latency_variances)]
                for row in range(len(anchor_latency_variances))
            ],
            "response_model_standard_uncertainty_latency_ns": 0.0,
            "instrumentation_standard_uncertainty_latency_ns": first.timer_resolution_ns,
            "boundary_uncertainty": {
                "status": "not_applicable",
                "reason_code": "exact-sequence-distribution-anchor-only",
            },
            "calibration_evidence_refs": [item.evidence_ref for item in holdouts],
        },
        "evidence_refs": ["artifact://frontier/qualification.json"],
    }
    surface["input_digest"] = _canonical_digest(surface)
    policy_document = {**policy.document, "input_digest": policy.digest}
    qualification: dict[str, object] = {
        "schema": QUALIFICATION_SCHEMA,
        "status": "qualified",
        "policy": policy_document,
        "hardware_cohort": cohort_id,
        "anchors": anchors,
        "surface": surface,
        "source_runs": source_records,
    }
    queries = [
        {
            "query_id": f"ascend-flash-attention-ragged-{index}",
            "surface_id": surface["surface_id"],
            "surface_version": surface["version"],
            "shape": shape,
            "domain": domain,
        }
        for index, shape in enumerate(query_shapes, start=1)
    ]
    return _write_operator_frontier_documents(
        runs_root,
        destination,
        run_id=run_id,
        cohort_id=cohort_id,
        source_records=source_records,
        surface=surface,
        qualification=qualification,
        queries=queries,
        analysis_plan="issue-37-ascend-flash-attention-ragged-exact",
        runtime_device_name=first.runtime_device_name,
        logical_device=first.logical_device,
    )


def _write_unknown_bounded_collection_bundle(
    artifact_store: str | Path,
    *,
    run_id: str,
    policy: _QualificationPolicy,
    searches: Sequence[_Observation],
    holdouts: Sequence[_Observation],
    confirmations: Sequence[_Observation],
    query_sizes: Sequence[int],
) -> Path:
    if not searches:
        raise OperatorFrontierQualificationError(
            "bounded collection has no source evidence",
            reason_code="bounded-collection-corpus-incomplete",
        )
    observations = [*searches, *holdouts, *confirmations]
    _require_independent_sessions(observations)
    cohort_id = _require_common_identity(observations)
    _require_collection_plan_identity(policy, observations)
    plan = cast(dict[str, Any], policy.document["collection_plan"])
    main_shapes = sorted(cast(list[int], plan["main_sweep_m"]))
    holdout_shapes = sorted(
        cast(list[int], plan.get("independent_holdout_m", []))
    )
    validation_shapes = sorted(
        cast(list[int], plan["independent_validation_m"])
    )
    scope = policy.document["scope"]
    assert isinstance(scope, dict)
    expected_scope = {
        "hardware_cohort": cohort_id,
        "operation": "MatMul",
        "dtype": observations[0].dtype,
        "layout": observations[0].layout,
        "fixed_n": observations[0].n,
        "fixed_k": observations[0].k,
        "anchor_m": main_shapes,
        "confirmation_m": validation_shapes,
        "candidate_ids": sorted({item.candidate_id for item in observations}),
    }
    if (
        holdout_shapes != main_shapes
        or not {item.m for item in searches} <= set(main_shapes)
        or not {item.m for item in holdouts} <= set(holdout_shapes)
        or not {item.m for item in confirmations} <= set(validation_shapes)
        or any(scope.get(key) != value for key, value in expected_scope.items())
    ):
        raise OperatorFrontierQualificationError(
            "partial bounded corpus falls outside the declared policy scope",
            reason_code="qualification-policy-scope-mismatch",
        )
    if any(
        item.warmup_iterations < policy.minimum_warmup_iterations
        for item in observations
    ):
        raise OperatorFrontierQualificationError(
            "source evidence does not meet the versioned warmup policy",
            reason_code="warmup-policy-failed",
        )
    runs_root = Path(artifact_store).resolve() / "runs"
    destination = runs_root / run_id
    if destination.exists():
        raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
    source_records = [
        *(
            _source_record(item, lane="main-sweep", bundle_root=destination)
            for item in searches
        ),
        *(
            _source_record(item, lane="holdout", bundle_root=destination)
            for item in holdouts
        ),
        *(
            _source_record(item, lane="independent-validation", bundle_root=destination)
            for item in confirmations
        ),
    ]
    evidence_digest = _canonical_digest(
        {
            "policy": policy.digest,
            "sources": sorted(item.manifest_sha256 for item in observations),
            "status": "unknown",
        }
    )
    surface: dict[str, object] = {
        "surface_id": f"surface://{cohort_id}/matmul/fixed-nk/{evidence_digest[:16]}",
        "version": f"v-{evidence_digest[:16]}",
        "previous_version": None,
        "qualification_status": "unknown",
        "qualification_reason_code": "bounded-collection-corpus-incomplete",
        "cohort_id": cohort_id,
        "domain": {
            "semantic_operation": "MatMul",
            "dtype": searches[0].dtype,
            "layout": searches[0].layout,
            "fixed_n": searches[0].n,
            "fixed_k": searches[0].k,
            "varying_axis": "m",
            "execution_mode": searches[0].execution_mode,
        },
        "candidate_family": searches[0].candidate_family,
        "coordinate": {"axis": "m", "transform": "identity", "transform_version": "v1"},
        "work_formula": {
            "kind": "matmul-2mnk-fixed-nk",
            "version": "v1",
            "fixed_n": searches[0].n,
            "fixed_k": searches[0].k,
            "work_unit": "FLOP",
        },
        "anchors": [],
        "cells": [],
        "evidence_refs": ["artifact://frontier/qualification.json"],
    }
    surface["input_digest"] = _canonical_digest(surface)
    policy_document = {**policy.document, "input_digest": policy.digest}
    qualification: dict[str, object] = {
        "schema": QUALIFICATION_SCHEMA,
        "status": "unknown",
        "reason_code": "bounded-collection-corpus-incomplete",
        "policy": policy_document,
        "hardware_cohort": cohort_id,
        "collection_plan": plan,
        "stopping_decision": {
            "status": "stopped",
            "main_sweep_completed": sorted({item.m for item in searches})
            == main_shapes,
            "supplemental_rounds_executed": plan["executed_supplemental_rounds"],
            "maximum_supplemental_rounds": plan["maximum_supplemental_rounds"],
            "additional_model_complexity_allowed": False,
        },
        "anchors": [],
        "surface": surface,
        "source_runs": source_records,
    }
    queries = [
        {
            "query_id": f"ascend-matmul-fixed-nk-m{size}",
            "surface_id": surface["surface_id"],
            "surface_version": surface["version"],
            "shape": {"m": size},
            "domain": surface["domain"],
        }
        for size in query_sizes
    ]
    return _write_operator_frontier_documents(
        runs_root,
        destination,
        run_id=run_id,
        cohort_id=cohort_id,
        source_records=source_records,
        surface=surface,
        qualification=qualification,
        queries=queries,
        analysis_plan="issue-36-ascend-matmul-bounded-m-sweep",
        runtime_device_name=searches[0].runtime_device_name,
        logical_device=searches[0].logical_device,
    )


@dataclass(frozen=True)
class _BoundedCorpus:
    observations: tuple[_Observation, ...]
    cohort_id: str
    plan: dict[str, Any]
    main_shapes: tuple[int, ...]
    by_main_shape: dict[int, list[_Observation]]
    by_holdout_shape: dict[int, list[_Observation]]
    by_validation_shape: dict[int, list[_Observation]]


@dataclass(frozen=True)
class _BoundedResponseAnalysis:
    reference_shape: OperatorShapeSemantics
    fixed_n: int
    fixed_k: int
    slope_ns_per_work: float
    setup_latency_ns: float
    asymptotic_rate: float | None
    validation_records: tuple[dict[str, object], ...]
    ramp_validation: tuple[dict[str, object], ...]
    steady_validation: tuple[dict[str, object], ...]
    ramp_max_m: int | None
    steady_min_m: int | None
    boundary_confirmed: bool
    error_budget_passed: bool
    qualified: bool
    reason_code: str | None


def _bounded_corpus(
    policy: _QualificationPolicy,
    searches: Sequence[_Observation],
    holdouts: Sequence[_Observation],
    confirmations: Sequence[_Observation],
) -> _BoundedCorpus:
    observations = (*searches, *holdouts, *confirmations)
    _require_independent_sessions(observations)
    cohort_id = _require_common_identity(observations)
    _require_collection_plan_identity(policy, observations)
    plan = cast(dict[str, Any], policy.document["collection_plan"])
    first = observations[0]
    main_key = (
        "main_sweep_m"
        if first.operator_shape.operation == "MatMul"
        else "main_sweep_sequence_lengths"
    )
    holdout_key = (
        "independent_holdout_m"
        if first.operator_shape.operation == "MatMul"
        else "independent_holdout_sequence_lengths"
    )
    validation_key = (
        "independent_validation_m"
        if first.operator_shape.operation == "MatMul"
        else "independent_validation_sequence_lengths"
    )
    main_shapes = tuple(sorted(cast(list[int], plan[main_key])))
    holdout_shapes = tuple(
        sorted(cast(list[int], plan.get(holdout_key, [])))
    )
    validation_shapes = tuple(
        sorted(cast(list[int], plan[validation_key]))
    )
    by_main_shape = {
        shape: [
            item
            for item in searches
            if item.operator_shape.coordinate_value == shape
        ]
        for shape in main_shapes
    }
    by_holdout_shape = {
        shape: [
            item
            for item in holdouts
            if item.operator_shape.coordinate_value == shape
        ]
        for shape in main_shapes
    }
    by_validation_shape = {
        shape: [
            item
            for item in confirmations
            if item.operator_shape.coordinate_value == shape
        ]
        for shape in validation_shapes
    }
    minimum_counts = (
        policy.minimum_search_sessions,
        policy.minimum_holdout_sessions,
        policy.minimum_confirmation_sessions,
    )
    if (
        holdout_shapes != main_shapes
        or sorted(
            {
                cast(int, item.operator_shape.coordinate_value)
                for item in searches
            }
        )
        != list(main_shapes)
        or sorted(
            {
                cast(int, item.operator_shape.coordinate_value)
                for item in holdouts
            }
        )
        != list(holdout_shapes)
        or sorted(
            {
                cast(int, item.operator_shape.coordinate_value)
                for item in confirmations
            }
        )
        != list(validation_shapes)
        or any(
            len(records) < minimum
            for groups, minimum in zip(
                (
                    by_main_shape.values(),
                    by_holdout_shape.values(),
                    by_validation_shape.values(),
                ),
                minimum_counts,
                strict=True,
            )
            for records in groups
        )
        or any(
            item.correctness != "passed" or item.timing_quality != "passed"
            for item in observations
        )
        or any(
            _relative_range([item.median_ns for item in records])
            > policy.maximum_session_median_relative_range
            for records in (
                *by_main_shape.values(),
                *by_holdout_shape.values(),
                *by_validation_shape.values(),
            )
        )
    ):
        raise OperatorFrontierQualificationError(
            "bounded collection corpus is incomplete or unstable",
            reason_code="bounded-collection-corpus-incomplete",
        )
    scope = policy.document["scope"]
    assert isinstance(scope, dict)
    if first.operator_shape.operation == "MatMul":
        expected_scope = {
            "hardware_cohort": cohort_id,
            "operation": "MatMul",
            "dtype": first.dtype,
            "layout": first.layout,
            "fixed_n": first.n,
            "fixed_k": first.k,
            "anchor_m": list(main_shapes),
            "confirmation_m": list(validation_shapes),
            "candidate_ids": sorted({item.candidate_id for item in observations}),
        }
    else:
        expected_scope = {
            "hardware_cohort": cohort_id,
            **{
                key: value
                for key, value in first.operator_shape.domain_facets.items()
                if key != "semantic_operation"
            },
            "operation": first.operator_shape.operation,
            "anchor_sequence_lengths": list(main_shapes),
            "confirmation_sequence_lengths": list(validation_shapes),
            "candidate_ids": sorted({item.candidate_id for item in observations}),
        }
    if any(scope.get(key) != value for key, value in expected_scope.items()):
        raise OperatorFrontierQualificationError(
            "qualification policy does not cover this bounded evidence corpus",
            reason_code="qualification-policy-scope-mismatch",
        )
    if any(
        item.warmup_iterations < policy.minimum_warmup_iterations
        for item in observations
    ):
        raise OperatorFrontierQualificationError(
            "source evidence does not meet the versioned warmup policy",
            reason_code="warmup-policy-failed",
        )
    return _BoundedCorpus(
        observations=observations,
        cohort_id=cohort_id,
        plan=plan,
        main_shapes=main_shapes,
        by_main_shape=by_main_shape,
        by_holdout_shape=by_holdout_shape,
        by_validation_shape=by_validation_shape,
    )


def _analyze_bounded_response(
    policy: _QualificationPolicy,
    corpus: _BoundedCorpus,
) -> _BoundedResponseAnalysis:
    first = corpus.observations[0]
    fixed_n = first.n
    fixed_k = first.k
    points = [
        (
            semantics_for_coordinate(first.operator_shape, shape).declared_work,
            float(median([item.median_ns for item in records])),
        )
        for shape, records in sorted(corpus.by_main_shape.items())
    ]
    mean_work = sum(work for work, _ in points) / len(points)
    mean_latency = sum(latency for _, latency in points) / len(points)
    denominator = sum((work - mean_work) ** 2 for work, _ in points)
    slope = sum(
        (work - mean_work) * (latency - mean_latency)
        for work, latency in points
    ) / denominator
    setup_latency_ns = mean_latency - slope * mean_work
    asymptotic_rate = 1_000_000_000.0 / slope if slope > 0 else None
    validation_records: list[dict[str, object]] = []
    relative_errors: list[float] = []
    for lane, grouped in (
        ("independent-holdout", corpus.by_holdout_shape),
        ("independent-validation", corpus.by_validation_shape),
    ):
        for shape, records in sorted(grouped.items()):
            observed = float(median([item.median_ns for item in records]))
            modeled = setup_latency_ns + (
                semantics_for_coordinate(first.operator_shape, shape).declared_work
                * slope
            )
            error = abs(observed - modeled) / observed
            relative_errors.append(error)
            validation_records.append(
                {
                    "lane": lane,
                    "shape": {first.operator_shape.coordinate_axis: shape},
                    "observed_latency_ns": observed,
                    "modeled_latency_ns": modeled,
                    "relative_error": error,
                    "evidence_refs": [item.evidence_ref for item in records],
                }
            )
    independent_validation = [
        record
        for record in validation_records
        if record["lane"] == "independent-validation"
    ]
    maximum_relative_error = policy.document.get("maximum_relative_error")
    maximum_setup_fraction = policy.document.get(
        "maximum_setup_fraction_for_steady"
    )
    valid_threshold = (
        isinstance(maximum_setup_fraction, (int, float))
        and not isinstance(maximum_setup_fraction, bool)
        and 0 <= float(maximum_setup_fraction) <= 1
    )
    ramp_validation = tuple(
        record
        for record in independent_validation
        if valid_threshold
        and setup_latency_ns / float(record["observed_latency_ns"])
        > float(maximum_setup_fraction)
    )
    steady_validation = tuple(
        record
        for record in independent_validation
        if valid_threshold
        and setup_latency_ns / float(record["observed_latency_ns"])
        <= float(maximum_setup_fraction)
    )
    ramp_max_m = max(
        (
            cast(dict[str, int], record["shape"])[
                first.operator_shape.coordinate_axis
            ]
            for record in ramp_validation
        ),
        default=None,
    )
    steady_min_m = min(
        (
            cast(dict[str, int], record["shape"])[
                first.operator_shape.coordinate_axis
            ]
            for record in steady_validation
        ),
        default=None,
    )
    boundary_confirmed = (
        ramp_max_m is not None
        and steady_min_m is not None
        and ramp_max_m < steady_min_m
        and len(steady_validation) >= 3
    )
    error_budget_passed = (
        isinstance(maximum_relative_error, (int, float))
        and not isinstance(maximum_relative_error, bool)
        and 0 <= float(maximum_relative_error) <= 1
        and slope > 0
        and setup_latency_ns >= 0
        and bool(relative_errors)
        and max(relative_errors) <= float(maximum_relative_error)
    )
    qualified = error_budget_passed and boundary_confirmed
    return _BoundedResponseAnalysis(
        reference_shape=first.operator_shape,
        fixed_n=fixed_n,
        fixed_k=fixed_k,
        slope_ns_per_work=slope,
        setup_latency_ns=setup_latency_ns,
        asymptotic_rate=asymptotic_rate,
        validation_records=tuple(validation_records),
        ramp_validation=ramp_validation,
        steady_validation=steady_validation,
        ramp_max_m=ramp_max_m,
        steady_min_m=steady_min_m,
        boundary_confirmed=boundary_confirmed,
        error_budget_passed=error_budget_passed,
        qualified=qualified,
        reason_code=(
            None
            if qualified
            else "latency-response-error-budget-failed"
            if not error_budget_passed
            else "shape-regime-boundary-not-qualified"
        ),
    )


def _write_bounded_collection_bundle(
    artifact_store: str | Path,
    *,
    run_id: str,
    policy: _QualificationPolicy,
    searches: Sequence[_Observation],
    holdouts: Sequence[_Observation],
    confirmations: Sequence[_Observation],
    query_sizes: Sequence[int],
) -> Path:
    corpus = _bounded_corpus(policy, searches, holdouts, confirmations)
    analysis = _analyze_bounded_response(policy, corpus)
    observations = corpus.observations
    cohort_id = corpus.cohort_id
    plan = corpus.plan
    main_shapes = corpus.main_shapes
    by_main_shape = corpus.by_main_shape
    by_holdout_shape = corpus.by_holdout_shape
    fixed_n = analysis.fixed_n
    fixed_k = analysis.fixed_k
    reference_shape = analysis.reference_shape
    operation = reference_shape.operation
    coordinate_axis = reference_shape.coordinate_axis
    slope = analysis.slope_ns_per_work
    setup_latency_ns = analysis.setup_latency_ns
    asymptotic_rate = analysis.asymptotic_rate
    validation_records = analysis.validation_records
    ramp_validation = analysis.ramp_validation
    steady_validation = analysis.steady_validation
    ramp_max_m = analysis.ramp_max_m
    steady_min_m = analysis.steady_min_m
    boundary_confirmed = analysis.boundary_confirmed
    error_budget_passed = analysis.error_budget_passed
    qualified = analysis.qualified
    reason_code = analysis.reason_code
    maximum_relative_error = policy.document.get("maximum_relative_error")
    maximum_setup_fraction = policy.document.get(
        "maximum_setup_fraction_for_steady"
    )
    runs_root = Path(artifact_store).resolve() / "runs"
    destination = runs_root / run_id
    if destination.exists():
        raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
    source_records = [
        *(
            _source_record(item, lane="main-sweep", bundle_root=destination)
            for item in searches
        ),
        *(
            _source_record(item, lane="holdout", bundle_root=destination)
            for item in holdouts
        ),
        *(
            _source_record(item, lane="independent-validation", bundle_root=destination)
            for item in confirmations
        ),
    ]
    evidence_digest = _canonical_digest(
        {
            "policy": policy.digest,
            "sources": sorted(item.manifest_sha256 for item in observations),
        }
    )
    operation_slug = (
        "matmul/fixed-nk"
        if operation == "MatMul"
        else "flash-attention/tnd-forward/equal-length"
    )
    surface_id = f"surface://{cohort_id}/{operation_slug}/{evidence_digest[:16]}"
    domain = {
        **reference_shape.domain_facets,
        "alignment_regime": "minimum-64-byte",
        "alignment_validated": True,
        "working_set_regime": (
            f"fixed-n{fixed_n}-k{fixed_k}-m{main_shapes[0]}-{main_shapes[-1]}"
            if operation == "MatMul"
            else (
                f"equal-length-sequence-{main_shapes[0]}-{main_shapes[-1]}"
            )
        ),
        "working_set_validated": True,
        "kernel_dispatch_regime": searches[0].candidate_family,
        "kernel_dispatch_validated": True,
        "regime_validated": qualified,
        "execution_mode": searches[0].execution_mode,
        "threads": 1,
        "candidate_protocol_digest": searches[0].candidate_protocol_digest,
        "candidate_set_digest": _canonical_digest(
            sorted({item.candidate_protocol_digest for item in observations})
        ),
        "execution_protocol_digest": searches[0].execution_protocol_digest,
        **(
            {"fixed_n": fixed_n, "fixed_k": fixed_k}
            if operation == "MatMul"
            else {"sequence_distribution": "equal-length"}
        ),
        "varying_axis": coordinate_axis,
    }
    response_attempt: dict[str, object] = {
        "target": "latency",
        "kind": "setup-plus-throughput",
        "version": "v1",
        "setup_latency_ns": setup_latency_ns,
        "asymptotic_rate": asymptotic_rate,
        "rate_unit": "FLOP/s",
        "validation_records": validation_records,
        "error_budget_passed": error_budget_passed,
        "shape_regime_boundary_confirmed": boundary_confirmed,
        "boundary_evidence": {
            "last_ramp_validation_m": ramp_max_m,
            "first_steady_validation_m": steady_min_m,
            "ramp_validation_count": len(ramp_validation),
            "steady_validation_count": len(steady_validation),
        },
    }
    anchors: list[dict[str, object]] = []
    cells: list[dict[str, object]] = []
    anchor_latency_variances: list[float] = []
    if qualified:
        retained_shapes = sorted(
            {
                shape
                for shape in main_shapes
                if shape <= cast(int, ramp_max_m)
                or shape >= cast(int, steady_min_m)
            }
        )
        for shape in retained_shapes:
            search_records = by_main_shape[shape]
            holdout_records = by_holdout_shape[shape]
            holdout_medians = [item.median_ns for item in holdout_records]
            latency_ns = float(median(holdout_medians))
            shape_semantics = semantics_for_coordinate(reference_shape, shape)
            work = shape_semantics.declared_work
            rates = [work / (value * 1e-9) for value in holdout_medians]
            anchor_prefix = (
                f"ascend-matmul-fixed-n{fixed_n}-k{fixed_k}-m"
                if operation == "MatMul"
                else "ascend-flash-attention-tnd-forward-s"
            )
            anchor_id = f"{anchor_prefix}{shape}-{evidence_digest[:12]}"
            standard_latency = float(stdev(holdout_medians))
            anchor_latency_variances.append(standard_latency**2)
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    "anchor_version": f"v-{evidence_digest[:16]}",
                    "shape": {coordinate_axis: shape},
                    "operator_shape_identity": shape_semantics.shape_identity,
                    "sequence_lengths": shape_semantics.normalized_shape.get(
                        "sequence_lengths"
                    ),
                    "declared_work": work,
                    "work_formula": reference_shape.work_formula,
                    "effective_rate": float(median(rates)),
                    "rate_unit": "FLOP/s",
                    "candidate_id": search_records[0].candidate_id,
                    "candidate_family": search_records[0].candidate_family,
                    "candidate_digest": search_records[0].candidate_digest,
                    "candidate_protocol_digest": search_records[0].candidate_protocol_digest,
                    "execution_protocol_digest": search_records[0].execution_protocol_digest,
                    "cohort_id": cohort_id,
                    "domain": domain,
                    "observation_validity": "QUALIFIED",
                    "frontier_role": "ACTIVE",
                    "evidence_ref": "artifact://frontier/qualification.json",
                    "state_transitions": _active_transitions(anchor_id),
                    "latency_ns": latency_ns,
                    "standard_uncertainty_rate": float(stdev(rates)),
                    "standard_uncertainty_latency_ns": standard_latency,
                    "search_run_ids": [item.run_id for item in search_records],
                    "holdout_run_ids": [item.run_id for item in holdout_records],
                    "holdout_session_medians_ns": holdout_medians,
                }
            )
        anchor_by_m = {
            cast(dict[str, int], item["shape"])[coordinate_axis]: item
            for item in anchors
        }
        regime_shapes = {
            "ramp": [
                shape for shape in retained_shapes if shape <= cast(int, ramp_max_m)
            ],
            "steady": [
                shape for shape in retained_shapes if shape >= cast(int, steady_min_m)
            ],
        }
        validation_refs_by_regime = {
            "ramp": [
                ref
                for record in ramp_validation
                for ref in cast(list[str], record["evidence_refs"])
            ],
            "steady": [
                ref
                for record in steady_validation
                for ref in cast(list[str], record["evidence_refs"])
            ],
        }
        for classification, shapes in regime_shapes.items():
            for left, right in zip(shapes, shapes[1:], strict=False):
                regime_scope = (
                    f"fixed-n{fixed_n}-k{fixed_k}"
                    if operation == "MatMul"
                    else "flash-attention-tnd-equal-length"
                )
                regime_id = f"{cohort_id}-{regime_scope}-{classification}"
                cells.append(
                    {
                        "cell_id": (
                            f"ascend-{operation_slug.replace('/', '-')}-{classification}-"
                            f"{coordinate_axis}{left}-{right}-{evidence_digest[:12]}"
                        ),
                        "anchor_ids": [
                            anchor_by_m[left]["anchor_id"],
                            anchor_by_m[right]["anchor_id"],
                        ],
                        "status": "retained",
                        "regime_id": regime_id,
                        "confirmation_shape": {
                            coordinate_axis: (
                                ramp_max_m
                                if classification == "ramp"
                                else steady_min_m
                            )
                        },
                        "confirmation_observed_rate": asymptotic_rate,
                        "confirmation_evidence_refs": validation_refs_by_regime[classification],
                        "interpolation_standard_uncertainty_rate": 0.0,
                        "response": {
                            **response_attempt,
                            "shape_regime": {
                                "identity": regime_id,
                                "classification": classification,
                            },
                            "fit_evidence_refs": [item.evidence_ref for item in searches],
                            "holdout_evidence_refs": [item.evidence_ref for item in holdouts],
                            "confirmation_evidence_refs": validation_refs_by_regime[classification],
                            "maximum_relative_error": float(cast(float, maximum_relative_error)),
                            "maximum_setup_fraction_for_steady": float(cast(float, maximum_setup_fraction)),
                        },
                    }
                )
        if all(regime_shapes.values()):
            boundary_left = regime_shapes["ramp"][-1]
            boundary_right = regime_shapes["steady"][0]
            cells.append(
                {
                    "cell_id": (
                        f"ascend-{operation_slug.replace('/', '-')}-boundary-"
                        f"{coordinate_axis}{boundary_left}-{boundary_right}-"
                        f"{evidence_digest[:12]}"
                    ),
                    "anchor_ids": [
                        anchor_by_m[boundary_left]["anchor_id"],
                        anchor_by_m[boundary_right]["anchor_id"],
                    ],
                    "status": "regime_boundary",
                    "regime_id": f"{cohort_id}-{operation_slug.replace('/', '-')}-boundary",
                    "confirmation_shape": {
                        coordinate_axis: [ramp_max_m, steady_min_m]
                    },
                    "confirmation_observed_rate": asymptotic_rate,
                    "confirmation_evidence_refs": [
                        *validation_refs_by_regime["ramp"],
                        *validation_refs_by_regime["steady"],
                    ],
                    "interpolation_standard_uncertainty_rate": 0.0,
                }
            )
        if any(len(shapes) < 3 for shapes in regime_shapes.values()):
            qualified = False
            reason_code = "shape-regime-boundary-not-qualified"
            anchors = []
            cells = []
            domain["regime_validated"] = False
    status = "qualified" if qualified else "rejected"
    surface: dict[str, object] = {
        "surface_id": surface_id,
        "version": f"v-{evidence_digest[:16]}",
        "previous_version": None,
        "qualification_status": status,
        **({"rejection_reason_code": reason_code} if reason_code is not None else {}),
        "cohort_id": cohort_id,
        "domain": domain,
        "candidate_family": searches[0].candidate_family,
        "coordinate": {
            "axis": coordinate_axis,
            "transform": "identity",
            "transform_version": "v1",
        },
        "work_formula": reference_shape.work_formula,
        "anchors": anchors,
        "cells": cells,
        "response_attempt": response_attempt,
        **(
            {
                "anchor_lifecycle_policy": {
                    "policy_id": "frontier-anchor-lifecycle",
                    "version": "v2",
                    "scope": f"{cohort_id}-{operation_slug}-bounded-sweep",
                    "change_reason": (
                        "issue-36 qualified bounded M sweep"
                        if operation == "MatMul"
                        else "issue-37 qualified equal-length TND sweep"
                    ),
                    "revalidation": "on cohort, corpus, policy, response, or boundary change",
                },
                "uncertainty_policy": {
                    "policy_id": (
                        "ascend-matmul-bounded-m-sweep-uncertainty"
                        if operation == "MatMul"
                        else "ascend-flash-attention-tnd-sweep-uncertainty"
                    ),
                    "version": "v1",
                    "scope": (
                        "two independently validated fixed-N/K Shape regimes"
                        if operation == "MatMul"
                        else "two independently validated equal-length TND Shape regimes"
                    ),
                    "change_reason": "independent holdout and boundary validation",
                    "revalidation": "on cohort, corpus, policy, response, or boundary change",
                    "combination": policy.document["uncertainty_combination"],
                    "target_coverage": policy.target_coverage,
                    "anchor_covariance": [
                        [0.0 for _ in anchors] for _ in anchors
                    ],
                    "anchor_latency_covariance": [
                        [
                            variance if row == column else 0.0
                            for column, variance in enumerate(anchor_latency_variances)
                        ]
                        for row in range(len(anchor_latency_variances))
                    ],
                    "response_model_standard_uncertainty_latency_ns": max(
                        abs(float(record["observed_latency_ns"]) - float(record["modeled_latency_ns"]))
                        for record in validation_records
                    ),
                    "instrumentation_standard_uncertainty_rate": 0.0,
                    "instrumentation_standard_uncertainty_latency_ns": searches[0].timer_resolution_ns,
                    "boundary_standard_uncertainty_latency_ns": slope
                    * abs(
                        semantics_for_coordinate(
                            reference_shape, cast(int, steady_min_m)
                        ).declared_work
                        - semantics_for_coordinate(
                            reference_shape, cast(int, ramp_max_m)
                        ).declared_work
                    )
                    / 2.0,
                    "calibration_evidence_refs": [item.evidence_ref for item in confirmations],
                },
            }
            if qualified
            else {}
        ),
        "evidence_refs": ["artifact://frontier/qualification.json"],
    }
    surface["input_digest"] = _canonical_digest(surface)
    policy_document = {**policy.document, "input_digest": policy.digest}
    stopping_decision = {
        "status": "stopped",
        "main_sweep_completed": True,
        "supplemental_rounds_executed": plan["executed_supplemental_rounds"],
        "maximum_supplemental_rounds": plan["maximum_supplemental_rounds"],
        "additional_model_complexity_allowed": False,
    }
    qualification: dict[str, object] = {
        "schema": QUALIFICATION_SCHEMA,
        "status": status,
        **({"reason_code": reason_code} if reason_code is not None else {}),
        "policy": policy_document,
        "hardware_cohort": cohort_id,
        "collection_plan": plan,
        "stopping_decision": stopping_decision,
        "response_attempt": surface["response_attempt"],
        "anchors": anchors,
        "surface": surface,
        "source_runs": source_records,
    }
    queries = [
        {
            "query_id": (
                f"ascend-matmul-fixed-nk-m{size}"
                if operation == "MatMul"
                else f"ascend-flash-attention-tnd-s{size}"
            ),
            "surface_id": surface["surface_id"],
            "surface_version": surface["version"],
            "shape": {coordinate_axis: size},
            "domain": surface["domain"],
        }
        for size in query_sizes
    ]
    return _write_operator_frontier_documents(
        runs_root,
        destination,
        run_id=run_id,
        cohort_id=cohort_id,
        source_records=source_records,
        surface=surface,
        qualification=qualification,
        queries=queries,
        analysis_plan=(
            "issue-36-ascend-matmul-bounded-m-sweep"
            if operation == "MatMul"
            else "issue-37-ascend-flash-attention-tnd-forward"
        ),
        runtime_device_name=searches[0].runtime_device_name,
        logical_device=searches[0].logical_device,
    )


class OperatorFrontierBundleWriter:
    """Qualify exact-Shape Anchors and publish one replayable Surface Bundle."""

    def run(
        self,
        artifact_store: str | Path,
        *,
        run_id: str,
        qualification_policy: dict[str, object] | None,
        search_runs: Iterable[str | Path],
        holdout_runs: Iterable[str | Path],
        confirmation_runs: Iterable[str | Path],
        query_sizes: Sequence[int],
        query_shapes: Sequence[dict[str, object]] = (),
    ) -> Path:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(f"unsafe run_id: {run_id!r}")
        policy = _QualificationPolicy.from_document(qualification_policy)
        _require_bounded_collection_rounds(policy)
        searches = [_observation(path) for path in search_runs]
        holdouts = [_observation(path) for path in holdout_runs]
        confirmations = [_observation(path) for path in confirmation_runs]
        all_observations = [*searches, *holdouts, *confirmations]
        scope = policy.document.get("scope")
        if (
            isinstance(scope, dict)
            and scope.get("sequence_distribution_mode") == "exact-only"
        ):
            if confirmations:
                raise OperatorFrontierQualificationError(
                    "ragged exact-only qualification does not interpolate",
                    reason_code="unsupported-sequence-distribution-interpolation",
                )
            return _write_exact_distribution_bundle(
                artifact_store,
                run_id=run_id,
                policy=policy,
                searches=searches,
                holdouts=holdouts,
                query_shapes=query_shapes,
            )
        bounded_plan = policy.document.get("collection_plan")
        bounded_shapes = (
            bounded_plan.get("main_sweep_m", [])
            if isinstance(bounded_plan, dict)
            else []
        )
        if isinstance(bounded_plan, dict) and not bounded_shapes:
            bounded_shapes = bounded_plan.get(
                "main_sweep_sequence_lengths", []
            )
        if isinstance(bounded_shapes, list) and len(bounded_shapes) > 2:
            if not holdouts or not confirmations:
                return _write_unknown_bounded_collection_bundle(
                    artifact_store,
                    run_id=run_id,
                    policy=policy,
                    searches=searches,
                    holdouts=holdouts,
                    confirmations=confirmations,
                    query_sizes=query_sizes,
                )
            return _write_bounded_collection_bundle(
                artifact_store,
                run_id=run_id,
                policy=policy,
                searches=searches,
                holdouts=holdouts,
                confirmations=confirmations,
                query_sizes=query_sizes,
            )
        if not searches or not holdouts or not confirmations:
            raise OperatorFrontierQualificationError(
                "search, holdout, and confirmation evidence are required",
                reason_code="missing-qualification-lane",
            )
        _require_independent_sessions(all_observations)
        cohort_id = _require_common_identity(all_observations)
        _require_collection_plan_identity(policy, all_observations)
        latency_response = policy.document.get("response_target") == "latency"
        if latency_response:
            maximum_setup_fraction_for_steady = policy.document.get(
                "maximum_setup_fraction_for_steady"
            )
            if (
                policy.document.get("response_kind") != "setup-plus-throughput"
                or policy.document.get("response_version") != "v1"
                or len({item.n for item in all_observations}) != 1
                or len({item.k for item in all_observations}) != 1
                or not isinstance(
                    maximum_setup_fraction_for_steady, (int, float)
                )
                or isinstance(maximum_setup_fraction_for_steady, bool)
                or not 0
                <= float(maximum_setup_fraction_for_steady)
                <= 1
            ):
                raise OperatorFrontierQualificationError(
                    "latency response requires fixed N/K and a valid regime policy",
                    reason_code="fixed-nk-domain-mismatch",
                )
        elif any(
            item.m != item.n or item.m != item.k
            for item in all_observations
        ):
            raise OperatorFrontierQualificationError(
                "legacy Effective Rate Surfaces require square MatMul evidence",
                reason_code="unsupported-shape-regime",
            )

        runs_root = Path(artifact_store).resolve() / "runs"
        destination = runs_root / run_id
        if destination.exists():
            raise RunBundleExistsError(f"Run Bundle already exists: {destination}")

        search_by_shape_candidate: dict[
            tuple[int, str], list[_Observation]
        ] = {}
        for item in searches:
            search_by_shape_candidate.setdefault(
                (item.m, item.candidate_id), []
            ).append(item)
        shapes = sorted({item.m for item in searches})
        if len(shapes) != 2:
            raise OperatorFrontierQualificationError(
                "the minimal one-dimensional Surface requires exactly two anchor Shapes",
                reason_code="invalid-minimal-surface-anchor-count",
            )

        confirmation_shapes = sorted({item.m for item in confirmations})
        if (
            len(confirmation_shapes) != 1
            or not shapes[0] < confirmation_shapes[0] < shapes[1]
        ):
            raise OperatorFrontierQualificationError(
                "one independent interior confirmation Shape is required",
                reason_code="invalid-regime-confirmation-shape",
            )
        confirmation_size = confirmation_shapes[0]
        attempted_ids = sorted({item.candidate_id for item in searches})
        attempted_families = {item.candidate_family for item in searches}
        _require_policy_scope(
            policy,
            all_observations,
            cohort_id=cohort_id,
            anchor_shapes=shapes,
            confirmation_shape=confirmation_size,
            candidate_ids=attempted_ids,
        )

        candidate_records: list[dict[str, object]] = []
        candidate_records_by_key: dict[
            tuple[int, str], dict[str, object]
        ] = {}
        eligible_ids_by_shape: dict[int, set[str]] = {}
        for size in shapes:
            shape_eligible_ids: set[str] = set()
            shape_candidates = sorted(
                candidate_id
                for candidate_size, candidate_id in search_by_shape_candidate
                if candidate_size == size
            )
            if shape_candidates != attempted_ids:
                raise OperatorFrontierQualificationError(
                    f"Shape {size} does not have the complete candidate set",
                    reason_code="candidate-coverage-incomplete",
                )
            for candidate_id in shape_candidates:
                records = search_by_shape_candidate[(size, candidate_id)]
                families = {item.candidate_family for item in records}
                digests = {item.candidate_digest for item in records}
                reasons: list[str] = []
                if len(records) < policy.minimum_search_sessions:
                    reasons.append("insufficient-independent-search-sessions")
                if not _same_shape_input_and_contract(records):
                    raise OperatorFrontierQualificationError(
                        f"Shape {size} search execution contract changed",
                        reason_code="execution-contract-mismatch",
                    )
                if len(families) != 1 or len(digests) != 1:
                    raise OperatorFrontierQualificationError(
                        f"Shape {size} candidate identity changed across sessions",
                        reason_code="candidate-identity-changed",
                    )
                if any(item.correctness != "passed" for item in records):
                    reasons.append("candidate-correctness-failed")
                if any(item.timing_quality != "passed" for item in records):
                    reasons.append("candidate-timing-quality-failed")
                medians = [item.median_ns for item in records]
                if (
                    len(medians) >= policy.minimum_search_sessions
                    and _relative_range(medians)
                    > policy.maximum_session_median_relative_range
                ):
                    reasons.append("candidate-repeatability-failed")
                status = "eligible" if not reasons else "excluded"
                family = next(iter(families)) if len(families) == 1 else "mixed"
                aggregate = float(median(medians))
                record: dict[str, object] = {
                    "shape": {"m" if latency_response else "s": size},
                    "candidate_id": candidate_id,
                    "candidate_family": family,
                    "status": status,
                    "reason_codes": reasons,
                    "search_run_ids": [item.run_id for item in records],
                    "search_session_medians_ns": medians,
                    "search_aggregate_median_ns": aggregate,
                    "candidate_digests": sorted(digests),
                    "candidate_identity": records[0].candidate_identity,
                    "candidate_evidence_digests": sorted(
                        {item.candidate_evidence_sha256 for item in records}
                    ),
                    "holdout_run_ids": [],
                    "holdout_session_medians_ns": [],
                    "holdout_aggregate_median_ns": None,
                }
                candidate_records.append(record)
                candidate_records_by_key[(size, candidate_id)] = record
                if not reasons:
                    shape_eligible_ids.add(candidate_id)
            if not shape_eligible_ids:
                raise OperatorFrontierQualificationError(
                    f"Shape {size} has no correct repeatable candidate",
                    reason_code="no-best-of-correct-candidate",
                )
            eligible_ids_by_shape[size] = shape_eligible_ids

        regime_eligible_ids = set.intersection(
            *(eligible_ids_by_shape[size] for size in shapes)
        )
        candidate_family_by_id = {
            item.candidate_id: item.candidate_family for item in searches
        }
        regime_eligible_families = {
            candidate_family_by_id[candidate_id]
            for candidate_id in regime_eligible_ids
        }
        eligible_level = _coverage_level(
            regime_eligible_ids, regime_eligible_families
        )
        if not regime_eligible_ids or not _coverage_satisfies(
            eligible_level, policy.minimum_candidate_coverage
        ):
            raise OperatorFrontierQualificationError(
                "candidate coverage does not meet the versioned policy",
                reason_code="candidate-coverage-policy-failed",
            )
        for record in candidate_records:
            candidate_id = str(record["candidate_id"])
            if record["status"] == "eligible" and (
                candidate_id not in regime_eligible_ids
            ):
                record["status"] = "excluded"
                reasons = cast(list[str], record["reason_codes"])
                reasons.append("candidate-not-eligible-across-all-anchor-shapes")

        expected_holdout_keys = {
            (size, candidate_id)
            for size in shapes
            for candidate_id in regime_eligible_ids
        }
        actual_holdout_keys = {
            (item.m, item.candidate_id) for item in holdouts
        }
        if actual_holdout_keys != expected_holdout_keys:
            raise OperatorFrontierQualificationError(
                "holdout must cover every search-eligible candidate",
                reason_code="holdout-candidate-coverage-incomplete",
            )

        qualified_holdouts: dict[tuple[int, str], list[_Observation]] = {}
        for size, candidate_id in sorted(expected_holdout_keys):
            search_records = search_by_shape_candidate[(size, candidate_id)]
            shape_holdouts = [
                item
                for item in holdouts
                if item.m == size and item.candidate_id == candidate_id
            ]
            if not _same_shape_input_and_contract(shape_holdouts):
                raise OperatorFrontierQualificationError(
                    f"Shape {size} holdout execution contract changed",
                    reason_code="execution-contract-mismatch",
                )
            if (
                len(shape_holdouts) < policy.minimum_holdout_sessions
                or any(
                    item.correctness != "passed"
                    or item.timing_quality != "passed"
                    or item.candidate_family
                    != candidate_family_by_id[candidate_id]
                    for item in shape_holdouts
                )
                or _relative_range(
                    [item.median_ns for item in shape_holdouts]
                )
                > policy.maximum_session_median_relative_range
            ):
                raise OperatorFrontierQualificationError(
                    f"Shape {size} candidate {candidate_id} holdout failed",
                    reason_code="independent-holdout-failed",
                )
            if (
                search_records[0].input_identity
                != shape_holdouts[0].input_identity
                or search_records[0].execution_contract_digest
                != shape_holdouts[0].execution_contract_digest
            ):
                raise OperatorFrontierQualificationError(
                    f"Shape {size} search/holdout contract changed",
                    reason_code="execution-contract-mismatch",
                )
            qualified_holdouts[(size, candidate_id)] = shape_holdouts
            holdout_medians = [item.median_ns for item in shape_holdouts]
            record = candidate_records_by_key[(size, candidate_id)]
            record["holdout_run_ids"] = [
                item.run_id for item in shape_holdouts
            ]
            record["holdout_session_medians_ns"] = holdout_medians
            record["holdout_aggregate_median_ns"] = float(
                median(holdout_medians)
            )

        winners: dict[
            int, tuple[str, str, list[_Observation], list[_Observation]]
        ] = {}
        for size in shapes:
            choices = []
            for candidate_id in sorted(regime_eligible_ids):
                shape_holdouts = qualified_holdouts[(size, candidate_id)]
                choices.append(
                    (
                        float(
                            median(
                                [item.median_ns for item in shape_holdouts]
                            )
                        ),
                        candidate_id,
                        candidate_family_by_id[candidate_id],
                        search_by_shape_candidate[(size, candidate_id)],
                        shape_holdouts,
                    )
                )
            _, winner_id, winner_family, search_records, shape_holdouts = min(
                choices
            )
            winners[size] = (
                winner_id,
                winner_family,
                search_records,
                shape_holdouts,
            )

        selected_ids = {value[0] for value in winners.values()}
        selected_families = {value[1] for value in winners.values()}
        if len(selected_ids) != 1 or len(selected_families) != 1:
            raise OperatorFrontierQualificationError(
                "anchor winners change candidate family inside the proposed regime",
                reason_code="candidate-family-regime-boundary",
            )
        selected_candidate_id = next(iter(selected_ids))
        selected_candidate_family = next(iter(selected_families))

        candidate_set_digest = _canonical_digest(
            {
                candidate_id: sorted(
                    {
                        item.candidate_protocol_digest
                        for item in all_observations
                        if item.candidate_id == candidate_id
                    }
                )
                for candidate_id in attempted_ids
            }
        )
        selected_candidate_protocol_digest = next(
            item.candidate_protocol_digest
            for item in all_observations
            if item.candidate_id == selected_candidate_id
        )
        execution_protocol_digest = searches[0].execution_protocol_digest
        validity_key = _canonical_digest(
            {
                "hardware_cohort": cohort_id,
                "runtime_device_name": searches[0].runtime_device_name,
                "logical_device": searches[0].logical_device,
                "dtype": searches[0].dtype,
                "layout": searches[0].layout,
                "candidate_family": selected_candidate_family,
                "selected_candidate_protocol_digest": (
                    selected_candidate_protocol_digest
                ),
                "candidate_set_digest": candidate_set_digest,
                "execution_protocol_digest": execution_protocol_digest,
                **(
                    {
                        "fixed_n": searches[0].n,
                        "fixed_k": searches[0].k,
                        "response_target": "latency",
                        "response_kind": "setup-plus-throughput",
                        "response_version": "v1",
                    }
                    if latency_response
                    else {}
                ),
            }
        )
        evidence_version_digest = _canonical_digest(
            {
                "validity_key": validity_key,
                "qualification_policy_digest": policy.digest,
                "source_manifest_digests": sorted(
                    item.manifest_sha256 for item in all_observations
                ),
            }
        )
        surface_version = f"v-{evidence_version_digest[:16]}"
        surface_id = (
            f"surface://{cohort_id}/matmul/fixed-nk/{validity_key[:16]}"
            if latency_response
            else f"surface://{cohort_id}/matmul/square/{validity_key[:16]}"
        )

        domain = {
            "semantic_operation": "MatMul",
            "dtype": searches[0].dtype,
            "layout": searches[0].layout,
            "alignment_regime": "minimum-64-byte",
            "alignment_validated": True,
            "working_set_regime": (
                f"fixed-n{searches[0].n}-k{searches[0].k}-m{shapes[0]}-{shapes[1]}"
                if latency_response
                else f"square-{shapes[0]}-{shapes[1]}"
            ),
            "working_set_validated": True,
            "kernel_dispatch_regime": selected_candidate_family,
            "kernel_dispatch_validated": True,
            "regime_validated": True,
            "execution_mode": searches[0].execution_mode,
            "threads": 1,
            "candidate_protocol_digest": selected_candidate_protocol_digest,
            "candidate_set_digest": candidate_set_digest,
            "execution_protocol_digest": execution_protocol_digest,
        }
        if latency_response:
            domain.update(
                {
                    "fixed_n": searches[0].n,
                    "fixed_k": searches[0].k,
                    "varying_axis": "m",
                }
            )
        anchors: list[dict[str, object]] = []
        anchor_rates: list[float] = []
        anchor_latencies: list[float] = []
        anchor_rate_variances: list[float] = []
        anchor_latency_variances: list[float] = []
        for size in shapes:
            (
                winner_id,
                winner_family,
                search_records,
                shape_holdouts,
            ) = winners[size]
            holdout_medians = [item.median_ns for item in shape_holdouts]
            latency_ns = float(median(holdout_medians))
            work = float(
                2 * size * searches[0].n * searches[0].k
                if latency_response
                else 2 * size**3
            )
            rates = [work / (value * 1e-9) for value in holdout_medians]
            effective_rate = float(median(rates))
            standard_rate = float(stdev(rates))
            standard_latency = float(stdev(holdout_medians))
            anchor_id = (
                f"ascend-matmul-fixed-n{searches[0].n}-k{searches[0].k}-m{size}-{validity_key[:12]}"
                if latency_response
                else f"ascend-matmul-square-{size}-{validity_key[:12]}"
            )
            anchor_rate_variances.append(standard_rate**2)
            anchor_latency_variances.append(standard_latency**2)
            anchor_rates.append(effective_rate)
            anchor_latencies.append(latency_ns)
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    "anchor_version": surface_version,
                    "shape": {"m" if latency_response else "s": size},
                    "effective_rate": effective_rate,
                    "rate_unit": "FLOP/s",
                    "candidate_id": winner_id,
                    "candidate_family": winner_family,
                    "candidate_digest": search_records[0].candidate_digest,
                    "candidate_protocol_digest": (
                        search_records[0].candidate_protocol_digest
                    ),
                    "execution_protocol_digest": execution_protocol_digest,
                    "cohort_id": cohort_id,
                    "domain": domain,
                    "observation_validity": "QUALIFIED",
                    "frontier_role": "ACTIVE",
                    "evidence_ref": "artifact://frontier/qualification.json",
                    "state_transitions": _active_transitions(anchor_id),
                    "latency_ns": latency_ns,
                    "standard_uncertainty_rate": standard_rate,
                    "standard_uncertainty_latency_ns": standard_latency,
                    "search_run_ids": [item.run_id for item in search_records],
                    "holdout_run_ids": [item.run_id for item in shape_holdouts],
                    "holdout_session_medians_ns": holdout_medians,
                }
            )

        if (
            len(confirmations) < policy.minimum_confirmation_sessions
            or not _same_shape_input_and_contract(confirmations)
            or any(
                item.candidate_id != selected_candidate_id
                or item.candidate_family != selected_candidate_family
                or item.correctness != "passed"
                or item.timing_quality != "passed"
                for item in confirmations
            )
            or _relative_range([item.median_ns for item in confirmations])
            > policy.maximum_session_median_relative_range
        ):
            raise OperatorFrontierQualificationError(
                "interior regime confirmation did not pass",
                reason_code="shape-regime-confirmation-failed",
            )
        left_weight = (shapes[1] - confirmation_size) / (
            shapes[1] - shapes[0]
        )
        interpolated_rate = (
            left_weight * anchor_rates[0]
            + (1.0 - left_weight) * anchor_rates[1]
        )
        confirmation_work = float(
            2 * confirmation_size * searches[0].n * searches[0].k
            if latency_response
            else 2 * confirmation_size**3
        )
        confirmation_rates = [
            confirmation_work / (item.median_ns * 1e-9)
            for item in confirmations
        ]
        confirmation_rate = float(median(confirmation_rates))
        interpolation_standard_rate = hypot(
            abs(confirmation_rate - interpolated_rate),
            float(stdev(confirmation_rates)),
        )
        maximum_rate = max(anchor_rates)
        minimum_latency = min(anchor_latencies)
        instrumentation_standard_rate = maximum_rate * (
            searches[0].timer_resolution_ns / minimum_latency
        )
        covariance = [
            [
                variance if row == column else 0.0
                for column, variance in enumerate(anchor_rate_variances)
            ]
            for row in range(len(anchor_rate_variances))
        ]
        latency_covariance = [
            [
                variance if row == column else 0.0
                for column, variance in enumerate(anchor_latency_variances)
            ]
            for row in range(len(anchor_latency_variances))
        ]
        cell_id = (
            (
                f"ascend-matmul-fixed-n{searches[0].n}-k{searches[0].k}-"
                f"m{shapes[0]}-{shapes[1]}-{validity_key[:12]}"
            )
            if latency_response
            else (
                f"ascend-matmul-square-{shapes[0]}-{shapes[1]}-"
                f"{validity_key[:12]}"
            )
        )
        regime_id = (
            (
                f"{cohort_id}-fixed-n{searches[0].n}-k{searches[0].k}-"
                f"m{shapes[0]}-{shapes[1]}-{validity_key[:12]}"
            )
            if latency_response
            else (
                f"{cohort_id}-square-{shapes[0]}-{shapes[1]}-"
                f"{validity_key[:12]}"
            )
        )
        confirmation_refs = [item.evidence_ref for item in confirmations]
        response: dict[str, object] | None = None
        response_model_standard_latency_ns = 0.0
        if latency_response:
            anchor_work = [
                float(2 * size * searches[0].n * searches[0].k)
                for size in shapes
            ]
            search_latencies = [
                float(
                    median(
                        [
                            item.median_ns
                            for item in winners[size][2]
                        ]
                    )
                )
                for size in shapes
            ]
            slope_ns_per_work = (
                search_latencies[1] - search_latencies[0]
            ) / (anchor_work[1] - anchor_work[0])
            if not isfinite(slope_ns_per_work) or slope_ns_per_work <= 0:
                raise OperatorFrontierQualificationError(
                    "latency response requires a positive asymptotic rate",
                    reason_code="invalid-latency-response",
                )
            asymptotic_rate = 1_000_000_000.0 / slope_ns_per_work
            setup_latency_ns = search_latencies[0] - (
                anchor_work[0] * slope_ns_per_work
            )
            if not isfinite(setup_latency_ns) or setup_latency_ns < 0:
                raise OperatorFrontierQualificationError(
                    "latency response requires non-negative Setup Latency",
                    reason_code="invalid-latency-response",
                )
            confirmation_work = float(
                2 * confirmation_size * searches[0].n * searches[0].k
            )
            modeled_confirmation_latency = (
                setup_latency_ns + confirmation_work * slope_ns_per_work
            )
            confirmation_latencies = [item.median_ns for item in confirmations]
            observed_confirmation_latency = float(median(confirmation_latencies))
            modeled_validation_latencies = [
                setup_latency_ns + work * slope_ns_per_work
                for work in anchor_work
            ] + [modeled_confirmation_latency]
            observed_validation_latencies = [
                *anchor_latencies,
                observed_confirmation_latency,
            ]
            validation_relative_errors = [
                abs(observed - modeled) / observed
                for observed, modeled in zip(
                    observed_validation_latencies,
                    modeled_validation_latencies,
                    strict=True,
                )
            ]
            response_model_standard_latency_ns = hypot(
                max(
                    abs(observed - modeled)
                    for observed, modeled in zip(
                        observed_validation_latencies,
                        modeled_validation_latencies,
                        strict=True,
                    )
                ),
                float(stdev(confirmation_latencies)),
            )
            maximum_relative_error = policy.document.get(
                "maximum_relative_error"
            )
            if (
                not isinstance(maximum_relative_error, (int, float))
                or isinstance(maximum_relative_error, bool)
                or not 0 <= float(maximum_relative_error) <= 1
                or any(
                    modeled != observed
                    for modeled, observed in zip(
                        modeled_validation_latencies[:2],
                        observed_validation_latencies[:2],
                        strict=True,
                    )
                )
                or max(validation_relative_errors)
                > float(maximum_relative_error)
            ):
                raise OperatorFrontierQualificationError(
                    "latency response failed the declared Error Budget",
                    reason_code="latency-response-error-budget-failed",
                )
            maximum_setup_fraction = max(
                setup_latency_ns / latency
                for latency in observed_validation_latencies
            )
            classification = (
                "steady"
                if len(observed_validation_latencies) >= 3
                and maximum_setup_fraction
                <= float(maximum_setup_fraction_for_steady)
                else "ramp"
            )
            response = {
                "target": "latency",
                "kind": "setup-plus-throughput",
                "version": "v1",
                "setup_latency_ns": setup_latency_ns,
                "asymptotic_rate": asymptotic_rate,
                "rate_unit": "FLOP/s",
                "shape_regime": {
                    "identity": regime_id,
                    "classification": classification,
                },
                "fit_evidence_refs": [
                    item.evidence_ref for item in searches
                ],
                "holdout_evidence_refs": [
                    item.evidence_ref for item in holdouts
                ],
                "confirmation_evidence_refs": confirmation_refs,
                "maximum_relative_error": float(maximum_relative_error),
                "maximum_setup_fraction_for_steady": float(
                    maximum_setup_fraction_for_steady
                ),
                "observed_confirmation_latency_ns": observed_confirmation_latency,
                "modeled_confirmation_latency_ns": modeled_confirmation_latency,
                "validation_relative_errors": validation_relative_errors,
            }
        surface: dict[str, object] = {
            "surface_id": surface_id,
            "version": surface_version,
            "previous_version": None,
            "qualification_status": "qualified",
            "cohort_id": cohort_id,
            "domain": domain,
            "candidate_family": selected_candidate_family,
            "anchor_lifecycle_policy": {
                "policy_id": "frontier-anchor-lifecycle",
                "version": "v2",
                "scope": (
                    f"{cohort_id}-fixed-nk-matmul"
                    if latency_response
                    else f"{cohort_id}-square-matmul"
                ),
                "change_reason": "issue-31 first qualified NPU Frontier",
                "revalidation": (
                    "on cohort, candidate, execution contract, anchor, or cell change"
                ),
            },
            "coordinate": {
                "axis": "m" if latency_response else "s",
                "transform": "identity",
                "transform_version": "v1",
            },
            "work_formula": (
                {
                    "kind": "matmul-2mnk-fixed-nk",
                    "version": "v1",
                    "fixed_n": searches[0].n,
                    "fixed_k": searches[0].k,
                    "work_unit": "FLOP",
                }
                if latency_response
                else {
                    "kind": "square-matmul-2s3",
                    "version": "v1",
                    "work_unit": "FLOP",
                }
            ),
            "anchors": anchors,
            "cells": [
                {
                    "cell_id": cell_id,
                    "anchor_ids": [anchor["anchor_id"] for anchor in anchors],
                    "status": "retained",
                    "regime_id": regime_id,
                    "confirmation_shape": {
                        "m" if latency_response else "s": confirmation_size
                    },
                    "confirmation_observed_rate": confirmation_rate,
                    "confirmation_evidence_refs": confirmation_refs,
                    "interpolation_standard_uncertainty_rate": (
                        interpolation_standard_rate
                    ),
                    **({"response": response} if response is not None else {}),
                }
            ],
            "uncertainty_policy": {
                "policy_id": "ascend-matmul-surface-uncertainty",
                "version": "v1",
                "scope": "single validated square MatMul cell",
                "change_reason": "independent holdout and interior confirmation",
                "revalidation": (
                    "on cohort, candidate, contract, anchor, or confirmation change"
                ),
                "combination": policy.document["uncertainty_combination"],
                "target_coverage": policy.target_coverage,
                "anchor_covariance": covariance,
                "instrumentation_standard_uncertainty_rate": (
                    instrumentation_standard_rate
                ),
                "calibration_evidence_refs": confirmation_refs,
                **(
                    {
                        "response_model_standard_uncertainty_latency_ns": (
                            response_model_standard_latency_ns
                        ),
                        "boundary_uncertainty": {
                            "status": "not_applicable",
                            "reason_code": "single-retained-regime-has-no-internal-boundary",
                        },
                        "anchor_latency_covariance": latency_covariance,
                        "instrumentation_standard_uncertainty_latency_ns": (
                            searches[0].timer_resolution_ns
                        ),
                    }
                    if latency_response
                    else {}
                ),
            },
            "evidence_refs": ["artifact://frontier/qualification.json"],
        }
        surface["input_digest"] = _canonical_digest(surface)

        source_records = [
            *(
                _source_record(item, lane="search", bundle_root=destination)
                for item in searches
            ),
            *(
                _source_record(item, lane="holdout", bundle_root=destination)
                for item in holdouts
            ),
            *(
                _source_record(item, lane="confirmation", bundle_root=destination)
                for item in confirmations
            ),
        ]
        attempted_level = _coverage_level(
            set(attempted_ids), attempted_families
        )
        policy_document = {
            **policy.document,
            "input_digest": policy.digest,
        }
        qualification: dict[str, object] = {
            "schema": QUALIFICATION_SCHEMA,
            "status": "qualified",
            "policy": policy_document,
            "hardware_cohort": cohort_id,
            "candidate_coverage": {
                "attempted_level": attempted_level,
                "eligible_level": eligible_level,
                "attempted_candidate_ids": attempted_ids,
                "eligible_candidate_ids": sorted(regime_eligible_ids),
                "selected_candidate_id": selected_candidate_id,
                "selected_candidate_family": selected_candidate_family,
            },
            "candidate_records": candidate_records,
            "anchors": anchors,
            "validated_shape_regime": {
                "regime_id": regime_id,
                "status": "validated",
                "anchor_shapes": [
                    {"m" if latency_response else "s": size}
                    for size in shapes
                ],
                "confirmation_shape": {
                    "m" if latency_response else "s": confirmation_size
                },
                "cell_id": cell_id,
                "candidate_family": selected_candidate_family,
                "hardware_cohort": cohort_id,
                "domain": domain,
            },
            "surface": surface,
            "source_runs": source_records,
        }

        queries = [
            {
                "query_id": (
                    f"ascend-matmul-fixed-nk-m{size}"
                    if latency_response
                    else f"ascend-matmul-square-{size}"
                ),
                "surface_id": surface["surface_id"],
                "surface_version": surface["version"],
                "shape": {"m" if latency_response else "s": size},
                "domain": domain,
            }
            for size in query_sizes
        ]
        inputs = {
            "resolved_configuration": {
                "analysis_plan": "issue-31-ascend-matmul-frontier",
                "qualification_policy": qualification["policy"],
            },
            "resolved_ir": {
                "semantic_operation": "MatMul",
                "shape_coordinate": (
                    "fixed-nk-m" if latency_response else "square-s"
                ),
            },
            "hardware": {
                "device": searches[0].runtime_device_name,
                "partition": searches[0].logical_device,
                "cohort_id": cohort_id,
            },
            "cohort_id": cohort_id,
            "execution_domain": domain,
        }
        evidence = {
            "capability_surfaces": [surface],
            "surface_queries": queries,
            "operator_frontier_qualification_ref": (
                "artifact://frontier/qualification.json"
            ),
        }
        diagnostic = {
            "schema": DIAGNOSTIC_EVIDENCE_SCHEMA,
            **inputs,
            **evidence,
            "digests": {
                "input_sha256": _canonical_digest(inputs),
                "evidence_sha256": _canonical_digest(evidence),
            },
        }

        runs_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))
        artifacts: list[dict[str, object]] = []

        def write_artifact(
            role: str,
            relative_path: str,
            document: dict[str, object],
            *,
            inputs: Sequence[str] = (),
        ) -> None:
            path = temporary / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_json_bytes(document))
            artifacts.append(
                {
                    "role": role,
                    "path": relative_path,
                    "media_type": "application/json",
                    "schema": document["schema"],
                    "sha256": _sha256(path),
                    "produced_by": "groundupscale-operator-frontier-v1",
                    "inputs": list(inputs),
                }
            )

        write_artifact(
            "operator-frontier-qualification",
            "frontier/qualification.json",
            qualification,
        )
        write_artifact(
            "diagnostic-evidence",
            "diagnostic/evidence.json",
            diagnostic,
            inputs=("operator-frontier-qualification",),
        )
        manifest = {
            "schema": "groundupscale.dev/run-manifest/v1alpha1",
            "bundle_kind": "operator-frontier",
            "run_id": run_id,
            "status": "completed",
            "created_at": datetime.now(UTC).isoformat(),
            "device": "ascend-npu",
            "hardware_cohort": cohort_id,
            "surface": {
                "surface_id": surface["surface_id"],
                "version": surface["version"],
                "input_digest": surface["input_digest"],
            },
            "source_runs": source_records,
            "artifacts": artifacts,
            "immutability": (
                "writer refuses an existing run_id; source and artifact digests "
                "are authoritative"
            ),
        }
        (temporary / "run.manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(temporary, destination)
        return destination


__all__ = [
    "OperatorFrontierBundleWriter",
    "OperatorFrontierQualificationError",
]
