"""Evidence-qualified Ascend NPU MatMul Frontier construction."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from math import hypot
from pathlib import Path
from statistics import median, stdev
from typing import Any

from groundupscale.ir import content_fingerprint
from groundupscale.run_bundle import RunBundleExistsError, verify_run_bundle

QUALIFICATION_SCHEMA = (
    "groundupscale.dev/operator-frontier-qualification/v1alpha1"
)
DIAGNOSTIC_EVIDENCE_SCHEMA = (
    "groundupscale.dev/diagnostic-evidence/v1alpha1"
)
MINIMUM_SESSIONS = 3
MAXIMUM_SESSION_MEDIAN_RELATIVE_RANGE = 0.10


class OperatorFrontierQualificationError(ValueError):
    """Source measurement evidence cannot qualify an authoritative Frontier."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class _Observation:
    root: Path
    manifest_sha256: str
    run_id: str
    cohort_id: str
    size: int
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
    input_identity: tuple[str, str]
    execution_contract_digest: str
    execution_protocol_digest: str
    execution_mode: str
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

    shape = case.get("shape")
    left = shape.get("left") if isinstance(shape, dict) else None
    right = shape.get("right") if isinstance(shape, dict) else None
    if (
        not isinstance(left, list)
        or not isinstance(right, list)
        or len(left) != 2
        or len(right) != 2
        or left[0] != left[1]
        or left != right
        or not isinstance(left[0], int)
        or left[0] <= 0
    ):
        raise OperatorFrontierQualificationError(
            f"{root}: only square exact-Shape MatMul is supported",
            reason_code="unsupported-shape-regime",
        )
    candidate_body = dict(candidate)
    candidate_digest = candidate_body.pop("candidate_digest", None)
    candidate_protocol_body = {
        key: value
        for key, value in candidate_body.items()
        if key not in {"shape", "minimum_alignment_bytes"}
    }
    session = environment.get("measurement_session")
    summary = raw_timing.get("summary")
    samples = raw_timing.get("samples")
    timer_resolution_ns = raw_timing.get("timer_resolution_ns")
    seed = case.get("seed")
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
        or preflight.get("eligible") is not True
        or completion.get("closed") is not True
        or instrumentation.get("lane") != "baseline-timing"
        or not isinstance(observation_validity, dict)
    ):
        raise OperatorFrontierQualificationError(
            f"{root}: incomplete qualification identity",
            reason_code="incomplete-qualification-identity",
        )
    left_identity = input_corpus.get("left_sha256")
    right_identity = input_corpus.get("right_sha256")
    if (
        not isinstance(left_identity, str)
        or len(left_identity) != 64
        or not isinstance(right_identity, str)
        or len(right_identity) != 64
    ):
        raise OperatorFrontierQualificationError(
            f"{root}: incomplete input identity",
            reason_code="incomplete-qualification-identity",
        )
    input_identity = (left_identity, right_identity)
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
    return _Observation(
        root=root,
        manifest_sha256=_sha256(manifest_path),
        run_id=str(manifest["run_id"]),
        cohort_id=str(manifest["hardware_cohort"]),
        size=int(left[0]),
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
        timer_resolution_ns=float(timer_resolution_ns),
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
        process_identity=(
            int(session["process_id"]),
            str(session["process_started_at"]),
        ),
    )


def _relative_range(values: Sequence[float]) -> float:
    center = float(median(values))
    return (max(values) - min(values)) / center


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
                item.size,
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
            "reason_code": "exact-shape-best-of-correct-search-winner",
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
        "shape": {"s": observation.size},
        "candidate_id": observation.candidate_id,
        "candidate_family": observation.candidate_family,
        "candidate_digest": observation.candidate_digest,
        "candidate_identity": observation.candidate_identity,
        "candidate_evidence_sha256": observation.candidate_evidence_sha256,
        "correctness": observation.correctness,
        "timing_quality": observation.timing_quality,
        "raw_samples_ns": list(observation.samples_ns),
        "median_ns": observation.median_ns,
        "process_identity": {
            "process_id": observation.process_identity[0],
            "process_started_at": observation.process_identity[1],
        },
        "evidence_ref": observation.evidence_ref,
    }


class OperatorFrontierBundleWriter:
    """Qualify exact-Shape Anchors and publish one replayable Surface Bundle."""

    def run(
        self,
        artifact_store: str | Path,
        *,
        run_id: str,
        search_runs: Iterable[str | Path],
        holdout_runs: Iterable[str | Path],
        confirmation_runs: Iterable[str | Path],
        query_sizes: Sequence[int],
    ) -> Path:
        searches = [_observation(path) for path in search_runs]
        holdouts = [_observation(path) for path in holdout_runs]
        confirmations = [_observation(path) for path in confirmation_runs]
        all_observations = [*searches, *holdouts, *confirmations]
        if not searches or not holdouts or not confirmations:
            raise OperatorFrontierQualificationError(
                "search, holdout, and confirmation evidence are required",
                reason_code="missing-qualification-lane",
            )
        _require_independent_sessions(all_observations)
        cohort_id = _require_common_identity(all_observations)

        runs_root = Path(artifact_store).resolve() / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        destination = runs_root / run_id
        if destination.exists():
            raise RunBundleExistsError(f"Run Bundle already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs_root))

        search_by_shape_candidate: dict[
            tuple[int, str], list[_Observation]
        ] = {}
        for item in searches:
            search_by_shape_candidate.setdefault(
                (item.size, item.candidate_id), []
            ).append(item)
        shapes = sorted({item.size for item in searches})
        if len(shapes) != 2:
            raise OperatorFrontierQualificationError(
                "the minimal one-dimensional Surface requires exactly two anchor Shapes",
                reason_code="invalid-minimal-surface-anchor-count",
            )

        candidate_records: list[dict[str, object]] = []
        winners: dict[int, tuple[str, str, list[_Observation]]] = {}
        attempted_ids = sorted({item.candidate_id for item in searches})
        attempted_families = {item.candidate_family for item in searches}
        eligible_ids: set[str] = set()
        for size in shapes:
            eligible: list[tuple[float, str, str, list[_Observation]]] = []
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
                if len(records) < MINIMUM_SESSIONS:
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
                    len(medians) >= MINIMUM_SESSIONS
                    and _relative_range(medians)
                    > MAXIMUM_SESSION_MEDIAN_RELATIVE_RANGE
                ):
                    reasons.append("candidate-repeatability-failed")
                status = "eligible" if not reasons else "excluded"
                family = next(iter(families)) if len(families) == 1 else "mixed"
                aggregate = float(median(medians))
                candidate_records.append(
                    {
                        "shape": {"s": size},
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
                    }
                )
                if not reasons:
                    eligible.append((aggregate, candidate_id, family, records))
                    eligible_ids.add(candidate_id)
            if not eligible:
                raise OperatorFrontierQualificationError(
                    f"Shape {size} has no correct repeatable candidate",
                    reason_code="no-best-of-correct-candidate",
                )
            _, winner_id, winner_family, winner_records = min(eligible)
            winners[size] = (winner_id, winner_family, winner_records)

        selected_ids = {value[0] for value in winners.values()}
        selected_families = {value[1] for value in winners.values()}
        if len(selected_ids) != 1 or len(selected_families) != 1:
            raise OperatorFrontierQualificationError(
                "anchor winners change candidate family inside the proposed regime",
                reason_code="candidate-family-regime-boundary",
            )
        selected_candidate_id = next(iter(selected_ids))
        selected_candidate_family = next(iter(selected_families))

        domain = {
            "semantic_operation": "MatMul",
            "dtype": searches[0].dtype,
            "layout": searches[0].layout,
            "alignment_regime": "minimum-64-byte",
            "alignment_validated": True,
            "working_set_regime": f"square-{shapes[0]}-{shapes[1]}",
            "working_set_validated": True,
            "kernel_dispatch_regime": selected_candidate_family,
            "kernel_dispatch_validated": True,
            "regime_validated": True,
            "execution_mode": searches[0].execution_mode,
            "threads": 1,
        }
        anchors: list[dict[str, object]] = []
        anchor_rates: list[float] = []
        anchor_latencies: list[float] = []
        anchor_rate_variances: list[float] = []
        for size in shapes:
            winner_id, winner_family, search_records = winners[size]
            shape_holdouts = [
                item
                for item in holdouts
                if item.size == size and item.candidate_id == winner_id
            ]
            if shape_holdouts and not _same_shape_input_and_contract(
                shape_holdouts
            ):
                raise OperatorFrontierQualificationError(
                    f"Shape {size} holdout execution contract changed",
                    reason_code="execution-contract-mismatch",
                )
            if (
                len(shape_holdouts) < MINIMUM_SESSIONS
                or any(
                    item.correctness != "passed"
                    or item.timing_quality != "passed"
                    or item.candidate_family != winner_family
                    for item in shape_holdouts
                )
                or _relative_range([item.median_ns for item in shape_holdouts])
                > MAXIMUM_SESSION_MEDIAN_RELATIVE_RANGE
            ):
                raise OperatorFrontierQualificationError(
                    f"Shape {size} independent holdout did not pass",
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
            holdout_medians = [item.median_ns for item in shape_holdouts]
            latency_ns = float(median(holdout_medians))
            work = float(2 * size**3)
            rates = [work / (value * 1e-9) for value in holdout_medians]
            effective_rate = float(median(rates))
            standard_rate = float(stdev(rates))
            anchor_id = f"ascend-matmul-square-{size}"
            anchor_rate_variances.append(standard_rate**2)
            anchor_rates.append(effective_rate)
            anchor_latencies.append(latency_ns)
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    "anchor_version": "v1",
                    "shape": {"s": size},
                    "effective_rate": effective_rate,
                    "rate_unit": "FLOP/s",
                    "candidate_id": winner_id,
                    "candidate_family": winner_family,
                    "cohort_id": cohort_id,
                    "domain": domain,
                    "observation_validity": "QUALIFIED",
                    "frontier_role": "ACTIVE",
                    "evidence_ref": "artifact://frontier/qualification.json",
                    "state_transitions": _active_transitions(anchor_id),
                    "latency_ns": latency_ns,
                    "standard_uncertainty_rate": standard_rate,
                    "search_run_ids": [item.run_id for item in search_records],
                    "holdout_run_ids": [item.run_id for item in shape_holdouts],
                    "holdout_session_medians_ns": holdout_medians,
                }
            )

        confirmation_shapes = sorted({item.size for item in confirmations})
        if (
            len(confirmation_shapes) != 1
            or not shapes[0] < confirmation_shapes[0] < shapes[1]
        ):
            raise OperatorFrontierQualificationError(
                "one independent interior confirmation Shape is required",
                reason_code="invalid-regime-confirmation-shape",
            )
        confirmation_size = confirmation_shapes[0]
        if (
            len(confirmations) < MINIMUM_SESSIONS
            or not _same_shape_input_and_contract(confirmations)
            or any(
                item.candidate_id != selected_candidate_id
                or item.candidate_family != selected_candidate_family
                or item.correctness != "passed"
                or item.timing_quality != "passed"
                for item in confirmations
            )
            or _relative_range([item.median_ns for item in confirmations])
            > MAXIMUM_SESSION_MEDIAN_RELATIVE_RANGE
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
        confirmation_work = float(2 * confirmation_size**3)
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
        cell_id = f"ascend-matmul-square-{shapes[0]}-{shapes[1]}"
        regime_id = f"ascend-910b2-square-{shapes[0]}-{shapes[1]}-v1"
        confirmation_refs = [item.evidence_ref for item in confirmations]
        surface: dict[str, object] = {
            "surface_id": "surface://ascend-910b2/matmul/square/v1",
            "version": "v1",
            "previous_version": None,
            "cohort_id": cohort_id,
            "domain": domain,
            "candidate_family": selected_candidate_family,
            "anchor_lifecycle_policy": {
                "policy_id": "frontier-anchor-lifecycle",
                "version": "v2",
                "scope": "ascend-910b2-square-matmul",
                "change_reason": "issue-31 first qualified NPU Frontier",
                "revalidation": (
                    "on cohort, candidate, execution contract, anchor, or cell change"
                ),
            },
            "coordinate": {
                "axis": "s",
                "transform": "identity",
                "transform_version": "v1",
            },
            "work_formula": {
                "kind": "square-matmul-2s3",
                "version": "v1",
                "work_unit": "FLOP",
            },
            "anchors": anchors,
            "cells": [
                {
                    "cell_id": cell_id,
                    "anchor_ids": [anchor["anchor_id"] for anchor in anchors],
                    "status": "retained",
                    "regime_id": regime_id,
                    "confirmation_shape": {"s": confirmation_size},
                    "confirmation_observed_rate": confirmation_rate,
                    "confirmation_evidence_refs": confirmation_refs,
                    "interpolation_standard_uncertainty_rate": (
                        interpolation_standard_rate
                    ),
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
                "combination": "root-sum-of-squares",
                "target_coverage": 0.68,
                "anchor_covariance": covariance,
                "instrumentation_standard_uncertainty_rate": (
                    instrumentation_standard_rate
                ),
                "calibration_evidence_refs": confirmation_refs,
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
        attempted_level = (
            "C2_MULTI_FAMILY"
            if len(attempted_families) >= 2
            else "C0_SINGLE"
        )
        eligible_families = {
            record["candidate_family"]
            for record in candidate_records
            if record["status"] == "eligible"
        }
        eligible_level = (
            "C2_MULTI_FAMILY"
            if len(eligible_families) >= 2
            else "C0_SINGLE"
        )
        qualification: dict[str, object] = {
            "schema": QUALIFICATION_SCHEMA,
            "status": "qualified",
            "policy": {
                "policy_id": "ascend-exact-shape-frontier-qualification",
                "version": "v1",
                "minimum_search_sessions": MINIMUM_SESSIONS,
                "minimum_holdout_sessions": MINIMUM_SESSIONS,
                "minimum_confirmation_sessions": MINIMUM_SESSIONS,
                "maximum_session_median_relative_range": (
                    MAXIMUM_SESSION_MEDIAN_RELATIVE_RANGE
                ),
                "estimator": "median(independent-holdout-session-medians)",
                "sample_exclusion": "none-preserve-all-raw-samples",
            },
            "hardware_cohort": cohort_id,
            "candidate_coverage": {
                "attempted_level": attempted_level,
                "eligible_level": eligible_level,
                "attempted_candidate_ids": attempted_ids,
                "eligible_candidate_ids": sorted(eligible_ids),
                "selected_candidate_id": selected_candidate_id,
                "selected_candidate_family": selected_candidate_family,
            },
            "candidate_records": candidate_records,
            "anchors": anchors,
            "validated_shape_regime": {
                "regime_id": regime_id,
                "status": "validated",
                "anchor_shapes": [{"s": size} for size in shapes],
                "confirmation_shape": {"s": confirmation_size},
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
                "query_id": f"ascend-matmul-square-{size}",
                "surface_id": surface["surface_id"],
                "surface_version": surface["version"],
                "shape": {"s": size},
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
                "shape_coordinate": "square-s",
            },
            "hardware": {
                "device": "Ascend 910B2",
                "partition": "single-npu:0",
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
