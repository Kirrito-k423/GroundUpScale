"""Strict, bounded resolution of versioned YAML Spec documents."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError
import yaml
from groundupscale.schemas.v1alpha1 import (
    AnalysisCaseDocument,
    AnalysisPlanDocument,
    BenchmarkCaseDocument,
    DOCUMENT_TYPES,
    DeploymentIntentDocument,
    FabricGraphDocument,
    HardwareSpecDocument,
    HardwareCapabilityProfileDocument,
    OperatorFrontierProfileBody,
    OperatorFrontierProfileDocument,
    ModelCallNodeSpec,
    ModelSpecDocument,
    SequenceNodeSpec,
    SpecDocument,
    SpecReference,
    WorkloadSpecDocument,
)


class SpecValidationError(ValueError):
    """A YAML document or explicit reference violated the public Spec contract."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise SpecValidationError(
                f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True)
class SpecSource:
    path: str
    sha256: str
    kind: str
    name: str
    version: str


@dataclass(frozen=True)
class AnalysisBundle:
    plan: AnalysisPlanDocument
    workload: WorkloadSpecDocument
    analysis_case: AnalysisCaseDocument
    deployment_intent: DeploymentIntentDocument
    hardware: tuple[HardwareSpecDocument, ...]
    hardware_capability_profiles: tuple[HardwareCapabilityProfileDocument, ...]
    operator_frontier_profiles: tuple[OperatorFrontierProfileDocument, ...]
    fabric_graph: FabricGraphDocument
    benchmark_cases: tuple[BenchmarkCaseDocument, ...]
    models: dict[str, ModelSpecDocument]
    models_by_reference: dict[str, ModelSpecDocument]
    sources: dict[str, SpecSource]


class SpecRepository:
    """Load one analysis plan and its finite, repository-local reference graph."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._sources: dict[str, SpecSource] = {}

    def _bounded_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as error:
            raise SpecValidationError(
                f"reference escapes repository root: {candidate}"
            ) from error
        return candidate

    @staticmethod
    def _validation_message(path: Path, error: ValidationError) -> str:
        details: list[str] = []
        for item in error.errors(include_url=False):
            location = ".".join(str(part) for part in item["loc"])
            details.append(f"{location}: {item['msg']}")
        return f"{path}: " + "; ".join(details)

    def load_document(self, path: str | Path) -> SpecDocument:
        resolved = self._bounded_path(path)
        try:
            raw_bytes = resolved.read_bytes()
        except OSError as error:
            raise SpecValidationError(f"cannot read spec {resolved}: {error}") from error
        try:
            raw = yaml.load(raw_bytes.decode("utf-8"), Loader=_UniqueKeyLoader)
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise SpecValidationError(f"invalid YAML in {resolved}: {error}") from error
        if not isinstance(raw, dict):
            raise SpecValidationError(f"{resolved}: document root must be a mapping")
        kind = raw.get("kind")
        if not isinstance(kind, str) or kind not in DOCUMENT_TYPES:
            raise SpecValidationError(f"{resolved}: unknown spec kind {kind!r}")
        document_type = DOCUMENT_TYPES[kind]
        try:
            document = document_type.model_validate(raw)
        except ValidationError as error:
            raise SpecValidationError(self._validation_message(resolved, error)) from error

        relative_path = resolved.relative_to(self.root).as_posix()
        self._sources[relative_path] = SpecSource(
            path=relative_path,
            sha256=sha256(raw_bytes).hexdigest(),
            kind=document.kind,
            name=document.metadata.name,
            version=document.metadata.version,
        )
        return document  # type: ignore[return-value]

    def _load_reference(self, reference: SpecReference, expected_kind: str) -> SpecDocument:
        document = self.load_document(reference.path)
        if document.kind != expected_kind:
            raise SpecValidationError(
                f"{reference.path}: expected {expected_kind}, found {document.kind}"
            )
        if document.metadata.version != reference.version:
            raise SpecValidationError(
                f"{reference.path}: expected version {reference.version}, "
                f"found {document.metadata.version}"
            )
        if reference.sha256 is not None:
            source = self._sources[self._bounded_path(reference.path).relative_to(self.root).as_posix()]
            if source.sha256 != reference.sha256:
                raise SpecValidationError(
                    f"{reference.path}: expected sha256 {reference.sha256}, found {source.sha256}"
                )
        return document

    def _model_references(
        self, node: ModelCallNodeSpec | SequenceNodeSpec
    ) -> tuple[SpecReference, ...]:
        if isinstance(node, ModelCallNodeSpec):
            return (node.model,)
        references: list[SpecReference] = []
        for child in node.children:
            references.extend(self._model_references(child))
        return tuple(references)

    def load_analysis_plan(self, path: str | Path) -> AnalysisBundle:
        self._sources = {}
        plan = self.load_document(path)
        if not isinstance(plan, AnalysisPlanDocument):
            raise SpecValidationError(f"{path}: expected AnalysisPlan, found {plan.kind}")

        workload = self._load_reference(plan.spec.workload, "WorkloadSpec")
        analysis_case = self._load_reference(plan.spec.analysis_case, "AnalysisCase")
        deployment = self._load_reference(
            plan.spec.deployment_intent, "DeploymentIntent"
        )
        hardware = tuple(
            self._load_reference(reference, "HardwareSpec")
            for reference in plan.spec.hardware
        )
        hardware_capability_profiles = tuple(
            self._load_reference(reference, "HardwareCapabilityProfile")
            for reference in plan.spec.hardware_capability_profiles
        )
        for profile in hardware_capability_profiles:
            assert isinstance(profile, HardwareCapabilityProfileDocument)
            from groundupscale.benchmark.hardware_microbenchmark import (
                CapabilityAggregationError,
                aggregate_capability_envelope,
            )

            evidence_path = self._bounded_path(profile.spec.source.path)
            try:
                evidence_bytes = evidence_path.read_bytes()
            except OSError as error:
                raise SpecValidationError(
                    f"cannot read capability evidence {evidence_path}: {error}"
                ) from error
            evidence_digest = sha256(evidence_bytes).hexdigest()
            if evidence_digest != profile.spec.source.sha256:
                raise SpecValidationError(
                    f"{evidence_path}: expected capability evidence sha256 "
                    f"{profile.spec.source.sha256}, found {evidence_digest}"
                )
            cohort_evidence = profile.spec.cohort_evidence
            if cohort_evidence is not None:
                cohort_path = self._bounded_path(cohort_evidence.path)
                try:
                    cohort_bytes = cohort_path.read_bytes()
                except OSError as error:
                    raise SpecValidationError(
                        f"cannot read capability cohort evidence {cohort_path}: {error}"
                    ) from error
                cohort_digest = sha256(cohort_bytes).hexdigest()
                if cohort_digest != cohort_evidence.sha256:
                    raise SpecValidationError(
                        f"{cohort_path}: expected cohort evidence sha256 "
                        f"{cohort_evidence.sha256}, found {cohort_digest}"
                    )
                try:
                    cohort_document = json.loads(cohort_bytes)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise SpecValidationError(
                        f"invalid capability cohort evidence {cohort_path}: {error}"
                    ) from error
                if not isinstance(cohort_document, dict) or (
                    cohort_document.get("schema") != cohort_evidence.schema_name
                    or cohort_document.get("cohort_id")
                    != profile.spec.hardware_cohort
                ):
                    raise SpecValidationError(
                        f"{cohort_path}: capability cohort evidence identity mismatch"
                    )
            try:
                observation = json.loads(evidence_bytes)
                rederived = aggregate_capability_envelope(
                    observation,
                    profile_name=profile.metadata.name,
                    profile_version=profile.metadata.version,
                    source_path=profile.spec.source.path,
                    source_sha256=profile.spec.source.sha256,
                )
                expected_profile = HardwareCapabilityProfileDocument.model_validate(
                    rederived
                )
            except (json.JSONDecodeError, CapabilityAggregationError, ValidationError) as error:
                raise SpecValidationError(
                    f"{evidence_path}: cannot rederive capability profile: {error}"
                ) from error
            if profile != expected_profile:
                raise SpecValidationError(
                    f"{evidence_path}: derived capability profile does not match "
                    "raw observation"
                )
        operator_frontier_profiles = tuple(
            self._load_reference(reference, "OperatorFrontierProfile")
            for reference in plan.spec.operator_frontier_profiles
        )
        for profile in operator_frontier_profiles:
            assert isinstance(profile, OperatorFrontierProfileDocument)
            evidence_path = self._bounded_path(profile.spec.source.path)
            try:
                evidence_bytes = evidence_path.read_bytes()
                evidence_digest = sha256(evidence_bytes).hexdigest()
            except OSError as error:
                raise SpecValidationError(
                    f"cannot read operator frontier evidence {evidence_path}: {error}"
                ) from error
            if evidence_digest != profile.spec.source.sha256:
                raise SpecValidationError(
                    f"{evidence_path}: expected operator frontier evidence sha256 "
                    f"{profile.spec.source.sha256}, found {evidence_digest}"
                )
            try:
                observation = json.loads(evidence_bytes)
                if observation.get("schema") != profile.spec.source.observation_schema:
                    raise ValueError("operator frontier observation schema mismatch")
                expected_spec = OperatorFrontierProfileBody.model_validate(
                    {
                        "target": observation.get("target"),
                        "hardware_cohort": observation.get("hardware_cohort"),
                        "source": profile.spec.source.model_dump(by_alias=True),
                        "anchors": observation.get("anchors"),
                    }
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                raise SpecValidationError(
                    f"{evidence_path}: cannot rederive operator frontier profile: "
                    f"{error}"
                ) from error
            if profile.spec != expected_spec:
                raise SpecValidationError(
                    f"{evidence_path}: derived operator frontier profile does not "
                    "match raw observation"
                )
        execution_domain = plan.spec.operator_frontier_execution_domain
        if execution_domain is not None:
            for stable_path, failure in execution_domain.qualification_failures.items():
                source = failure.source
                evidence_path = self._bounded_path(source.path)
                try:
                    evidence_bytes = evidence_path.read_bytes()
                    evidence_digest = sha256(evidence_bytes).hexdigest()
                    evidence = json.loads(evidence_bytes)
                except (OSError, json.JSONDecodeError) as error:
                    raise SpecValidationError(
                        f"cannot read Frontier qualification failure evidence "
                        f"{evidence_path}: {error}"
                    ) from error
                if evidence_digest != source.sha256:
                    raise SpecValidationError(
                        f"{evidence_path}: Frontier qualification failure digest mismatch"
                    )
                if evidence.get("schema") != source.evidence_schema:
                    raise SpecValidationError(
                        f"{evidence_path}: Frontier qualification failure schema mismatch"
                    )
                attempts = evidence.get("attempts")
                matches = [
                    item
                    for item in attempts
                    if isinstance(item, dict)
                    and item.get("stable_path") == stable_path
                ] if isinstance(attempts, list) else []
                if (
                    len(matches) != 1
                    or matches[0].get("status") != failure.status
                    or tuple(matches[0].get("reason_codes", ()))
                    != failure.reason_codes
                ):
                    raise SpecValidationError(
                        f"{evidence_path}: Frontier qualification failure does not "
                        f"bind {stable_path}"
                    )
                relative_path = evidence_path.relative_to(self.root).as_posix()
                self._sources[relative_path] = SpecSource(
                    path=relative_path,
                    sha256=evidence_digest,
                    kind="OperatorFrontierQualificationEvidence",
                    name=stable_path,
                    version="v1alpha1",
                )
        fabric = self._load_reference(plan.spec.fabric_graph, "FabricGraph")
        benchmarks = tuple(
            self._load_reference(reference, "BenchmarkCase")
            for reference in plan.spec.benchmark_cases
        )

        assert isinstance(workload, WorkloadSpecDocument)
        models: dict[str, ModelSpecDocument] = {}
        models_by_reference: dict[str, ModelSpecDocument] = {}
        model_paths: dict[str, str] = {}
        for reference in self._model_references(workload.spec.root):
            model = self._load_reference(reference, "ModelSpec")
            assert isinstance(model, ModelSpecDocument)
            previous_path = model_paths.get(model.metadata.name)
            if previous_path is not None and previous_path != reference.path:
                raise SpecValidationError(
                    f"model name {model.metadata.name!r} resolves from both "
                    f"{previous_path!r} and {reference.path!r}"
                )
            model_paths[model.metadata.name] = reference.path
            models[model.metadata.name] = model
            models_by_reference[reference.path] = model

        assert isinstance(analysis_case, AnalysisCaseDocument)
        assert isinstance(deployment, DeploymentIntentDocument)
        assert all(isinstance(document, HardwareSpecDocument) for document in hardware)
        assert all(
            isinstance(document, HardwareCapabilityProfileDocument)
            for document in hardware_capability_profiles
        )
        assert all(
            isinstance(document, OperatorFrontierProfileDocument)
            for document in operator_frontier_profiles
        )
        assert isinstance(fabric, FabricGraphDocument)
        assert all(isinstance(document, BenchmarkCaseDocument) for document in benchmarks)
        return AnalysisBundle(
            plan=plan,
            workload=workload,
            analysis_case=analysis_case,
            deployment_intent=deployment,
            hardware=hardware,  # type: ignore[arg-type]
            hardware_capability_profiles=hardware_capability_profiles,  # type: ignore[arg-type]
            operator_frontier_profiles=operator_frontier_profiles,  # type: ignore[arg-type]
            fabric_graph=fabric,
            benchmark_cases=benchmarks,  # type: ignore[arg-type]
            models=models,
            models_by_reference=dict(sorted(models_by_reference.items())),
            sources=dict(sorted(self._sources.items())),
        )
