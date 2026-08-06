"""Deep public Semantic Compiler interface and owned lowering phases."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from groundupscale.ir.common import (
    DerivationRecord,
    canonical_json,
    content_fingerprint,
    derivation_identity,
    node_identity,
)
from groundupscale.ir.model import IREntrypoint, IRModule, IRTensorType, ModelIR
from groundupscale.ir.semantic import (
    ProvenanceGraph,
    SemanticCompilationResult,
    SemanticOperation,
    SemanticProgram,
    SemanticRegion,
    SemanticStateArtifact,
    SemanticStateEffect,
    SemanticTensorType,
    SemanticValue,
    ValidationResult,
)
from groundupscale.ir.workload import IRModelCall, IRSequence, WorkloadIR, WorkloadNode
from groundupscale.schemas.v1alpha1 import (
    AnalysisCaseDocument,
    DeploymentIntentDocument,
)


SEMANTIC_COMPILER_VERSION = "core.semantic-compiler/v1alpha1"


class SemanticCompileError(ValueError):
    """The verified structural inputs cannot form a valid Semantic IR."""


@dataclass(frozen=True)
class CompilationContext:
    compiler_version: str
    plugin_versions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class LogicalStrategy:
    type: str
    version: str
    config_json: str


@dataclass(frozen=True)
class LogicalStrategyBinding:
    scope: str
    strategies: tuple[LogicalStrategy, ...]


@dataclass(frozen=True)
class SemanticDeploymentPlan:
    bindings: tuple[LogicalStrategyBinding, ...]


def semantic_deployment_plan(intent: DeploymentIntentDocument) -> SemanticDeploymentPlan:
    """Project only logical strategy effects; deliberately discard placement."""
    bindings = tuple(
        LogicalStrategyBinding(
            scope=binding.scope,
            strategies=tuple(
                LogicalStrategy(
                    type=strategy.type,
                    version=strategy.version,
                    config_json=canonical_json(strategy.config),
                )
                for strategy in binding.strategies
            ),
        )
        for binding in intent.spec.bindings
        if binding.strategies
    )
    return SemanticDeploymentPlan(bindings=bindings)


@dataclass(frozen=True)
class SemanticCompileRequest:
    workload: WorkloadIR
    models: tuple[ModelIR, ...]
    analysis_case: AnalysisCaseDocument
    deployment: SemanticDeploymentPlan
    context: CompilationContext


@dataclass
class _ValueDraft:
    value_id: str
    node_id: str
    stable_path: str
    kind: str
    tensor: SemanticTensorType
    producer_id: str | None
    consumers: list[str] = field(default_factory=list)
    alias_of: str | None = None
    derivation_ids: tuple[str, ...] = ()

    def freeze(self) -> SemanticValue:
        return SemanticValue(
            value_id=self.value_id,
            node_id=self.node_id,
            stable_path=self.stable_path,
            kind=self.kind,
            tensor=self.tensor,
            producer_id=self.producer_id,
            consumer_ids=tuple(self.consumers),
            alias_of=self.alias_of,
            derivation_ids=self.derivation_ids,
        )


class SemanticCompiler:
    """Own all phases behind one immutable compile request/result seam."""

    def __init__(self) -> None:
        self._fingerprint = ""
        self._bindings: dict[str, int] = {}
        self._values: dict[str, _ValueDraft] = {}
        self._artifacts: dict[str, SemanticStateArtifact] = {}
        self._effects: list[SemanticStateEffect] = []
        self._records: list[DerivationRecord] = []
        self._models: dict[tuple[str, str], ModelIR] = {}
        self._workload_artifacts: dict[str, SemanticStateArtifact] = {}
        self._artifact_versions: dict[str, int | None] = {}

    def compile(self, request: SemanticCompileRequest) -> SemanticCompilationResult:
        if request.deployment.bindings:
            strategy_types = sorted(
                {
                    strategy.type
                    for binding in request.deployment.bindings
                    for strategy in binding.strategies
                }
            )
            raise SemanticCompileError(
                "no registered semantic strategy plugin for: "
                + ", ".join(strategy_types)
            )
        self._reset(request)
        self._validate_symbols_and_constraints(request)
        self._create_workload_artifacts(request.workload)
        workload_region = self._expand_workload_node(request.workload.root)
        root_path = f"semantic/analysis/{request.analysis_case.metadata.name}"
        root_id = node_identity("semantic-region", self._fingerprint, root_path)
        root_derivations = self._record(
            phase="semantic-analysis",
            rule=f"{SEMANTIC_COMPILER_VERSION}:bind-analysis-case",
            source_path=(
                f"AnalysisCase/{request.analysis_case.metadata.name}@"
                f"{request.analysis_case.metadata.version}"
            ),
            source_stable_path=root_path,
            target_node_id=root_id,
        )
        analysis = request.analysis_case.spec
        root = SemanticRegion(
            local_id=request.analysis_case.metadata.name,
            kind="analysis_case",
            definition_id=(
                f"analysis-definition/{request.analysis_case.metadata.name}@"
                f"{request.analysis_case.metadata.version}"
            ),
            stable_path=root_path,
            node_id=root_id,
            inputs=(),
            outputs=workload_region.outputs,
            items=(workload_region,),
            state_effect_ids=(),
            attributes=(
                ("driver_kind", analysis.driver.kind),
                ("measured_iterations", analysis.driver.measured_iterations),
                ("observation_iterations", analysis.observation_window.value),
                ("warmup_iterations", analysis.driver.warmup_iterations),
            ),
            derivation_ids=root_derivations,
        )
        values = tuple(value.freeze() for value in self._values.values())
        program = SemanticProgram(
            schema="groundupscale.dev/semantic-ir/v1alpha1",
            name=f"{request.workload.name}:{request.analysis_case.metadata.name}",
            compilation_fingerprint=self._fingerprint,
            symbols=tuple(sorted(self._bindings.items())),
            root=root,
            values=values,
            state_artifacts=tuple(self._artifacts.values()),
            state_effects=tuple(self._effects),
        )
        validations = self._verify(program)
        failed = [validation for validation in validations if not validation.passed]
        if failed:
            raise SemanticCompileError(
                "; ".join(f"{item.check}: {item.detail}" for item in failed)
            )
        return SemanticCompilationResult(
            semantic_ir=program,
            provenance=ProvenanceGraph(
                schema="groundupscale.dev/provenance-graph/v1alpha1",
                records=tuple(self._records),
            ),
            diagnostics=(),
            validation_results=validations,
            compilation_fingerprint=self._fingerprint,
        )

    def _reset(self, request: SemanticCompileRequest) -> None:
        model_fingerprints = tuple(
            sorted((model.name, model.version, model.compilation_fingerprint) for model in request.models)
        )
        analysis_semantics = {
            "shape": request.analysis_case.spec.shape.model_dump(mode="json"),
            "driver": request.analysis_case.spec.driver.model_dump(mode="json"),
            "observation_window": request.analysis_case.spec.observation_window.model_dump(
                mode="json"
            ),
        }
        self._fingerprint = content_fingerprint(
            SEMANTIC_COMPILER_VERSION,
            request.context,
            request.workload.compilation_fingerprint,
            model_fingerprints,
            analysis_semantics,
            request.deployment,
        )
        self._bindings = dict(request.analysis_case.spec.shape.bindings)
        self._values = {}
        self._artifacts = {}
        self._effects = []
        self._models = {(model.name, model.version): model for model in request.models}
        if len(self._models) != len(request.models):
            raise SemanticCompileError("duplicate model name/version in compile request")
        self._records = list(request.workload.provenance)
        for model in sorted(request.models, key=lambda item: (item.name, item.version)):
            self._records.extend(model.provenance)
        self._workload_artifacts = {}
        self._artifact_versions = {}

    def _validate_symbols_and_constraints(self, request: SemanticCompileRequest) -> None:
        analysis_dtype = request.analysis_case.spec.shape.dtype
        boundary_dtypes = {
            artifact.tensor.dtype for artifact in request.workload.artifacts
        }
        for model in request.models:
            for entrypoint in model.root.entrypoints:
                boundary_dtypes.update(port.tensor.dtype for port in entrypoint.inputs)
                boundary_dtypes.update(port.tensor.dtype for port in entrypoint.outputs)
        mismatches = sorted(dtype for dtype in boundary_dtypes if dtype != analysis_dtype)
        if mismatches:
            raise SemanticCompileError(
                f"AnalysisCase dtype {analysis_dtype} does not match boundary tensor "
                f"dtypes: {', '.join(mismatches)}"
            )
        for model in request.models:
            declared = {name for name, _ in model.symbols}
            missing = sorted(declared - self._bindings.keys())
            if missing:
                raise SemanticCompileError(
                    f"model {model.name} has unbound symbols: {', '.join(missing)}"
                )
            for constraint in model.constraints:
                match = re.fullmatch(
                    r"([A-Za-z_][A-Za-z0-9_]*) == "
                    r"([A-Za-z_][A-Za-z0-9_]*) \* ([A-Za-z_][A-Za-z0-9_]*)",
                    constraint,
                )
                if match is None:
                    raise SemanticCompileError(f"unsupported constraint: {constraint}")
                left, first, second = match.groups()
                if self._bindings[left] != self._bindings[first] * self._bindings[second]:
                    raise SemanticCompileError(
                        f"constraint failed: {constraint} with bindings {self._bindings}"
                    )

    def _tensor(self, tensor: IRTensorType) -> SemanticTensorType:
        shape: list[int] = []
        for dimension in tensor.shape:
            if isinstance(dimension, int):
                shape.append(dimension)
                continue
            try:
                shape.append(self._bindings[dimension])
            except KeyError as error:
                raise SemanticCompileError(f"unbound shape symbol: {dimension}") from error
        return SemanticTensorType(dtype=tensor.dtype, shape=tuple(shape), layout=tensor.layout)

    @staticmethod
    def _expect_tensor(
        expected: SemanticTensorType, actual: SemanticTensorType, location: str
    ) -> None:
        if expected != actual:
            raise SemanticCompileError(
                f"type mismatch at {location}: expected {expected}, found {actual}"
            )

    def _record(
        self,
        *,
        phase: str,
        rule: str,
        source_path: str,
        source_stable_path: str,
        target_node_id: str,
    ) -> tuple[str, ...]:
        derivation_id = derivation_identity(rule, self._fingerprint, source_stable_path)
        self._records.append(
            DerivationRecord(
                derivation_id=derivation_id,
                phase=phase,
                rule=rule,
                source_path=source_path,
                source_stable_path=source_stable_path,
                target_node_ids=(target_node_id,),
            )
        )
        return (derivation_id,)

    def _create_value(
        self,
        *,
        stable_path: str,
        kind: str,
        tensor: SemanticTensorType,
        producer_id: str | None,
        source_path: str,
        source_stable_path: str,
        alias_of: str | None = None,
    ) -> str:
        value_id = node_identity("semantic-value", self._fingerprint, stable_path)
        if value_id in self._values:
            raise SemanticCompileError(f"duplicate semantic value at {stable_path}")
        derivations = self._record(
            phase="semantic-link",
            rule=f"{SEMANTIC_COMPILER_VERSION}:typed-value",
            source_path=source_path,
            source_stable_path=source_stable_path,
            target_node_id=value_id,
        )
        self._values[value_id] = _ValueDraft(
            value_id=value_id,
            node_id=value_id,
            stable_path=stable_path,
            kind=kind,
            tensor=tensor,
            producer_id=producer_id,
            alias_of=alias_of,
            derivation_ids=derivations,
        )
        return value_id

    def _create_workload_artifacts(self, workload: WorkloadIR) -> None:
        for artifact in workload.artifacts:
            stable_path = f"semantic/{workload.root.stable_path}/artifact/{artifact.name}"
            artifact_id = node_identity(
                "semantic-state-artifact", self._fingerprint, stable_path
            )
            derivations = self._record(
                phase="semantic-link",
                rule=f"{SEMANTIC_COMPILER_VERSION}:workload-artifact",
                source_path=f"WorkloadIR/{workload.name}@{workload.version}",
                source_stable_path=stable_path,
                target_node_id=artifact_id,
            )
            semantic_artifact = SemanticStateArtifact(
                artifact_id=artifact_id,
                node_id=artifact_id,
                stable_path=stable_path,
                role=artifact.role,
                tensor=self._tensor(artifact.tensor),
                source_kind="workload_artifact",
                initial_version=0 if artifact.role in {"input", "state"} else None,
                derivation_ids=derivations,
            )
            self._artifacts[artifact_id] = semantic_artifact
            self._artifact_versions[artifact_id] = semantic_artifact.initial_version
            self._workload_artifacts[artifact.name] = semantic_artifact

    def _read_artifact(
        self, artifact: SemanticStateArtifact, stable_path: str, owner: str | None
    ) -> tuple[str, str]:
        version = self._artifact_versions.get(artifact.artifact_id)
        if version is None:
            raise SemanticCompileError(
                f"artifact {artifact.stable_path} is read before its first write"
            )
        effect_id = node_identity("semantic-state-effect", self._fingerprint, stable_path)
        value_id = self._create_value(
            stable_path=f"{stable_path}/value",
            kind="artifact",
            tensor=artifact.tensor,
            producer_id=effect_id,
            source_path=artifact.stable_path,
            source_stable_path=stable_path,
        )
        derivations = self._record(
            phase="semantic-link",
            rule=f"{SEMANTIC_COMPILER_VERSION}:state-read",
            source_path=artifact.stable_path,
            source_stable_path=stable_path,
            target_node_id=effect_id,
        )
        self._effects.append(
            SemanticStateEffect(
                effect_id=effect_id,
                node_id=effect_id,
                stable_path=stable_path,
                kind="read",
                artifact_id=artifact.artifact_id,
                input_value_id=None,
                output_value_id=value_id,
                owner_node_id=owner,
                version_before=version,
                version_after=None,
                derivation_ids=derivations,
            )
        )
        return effect_id, value_id

    def _write_artifact(
        self, artifact: SemanticStateArtifact, value_id: str, stable_path: str
    ) -> str:
        effect_id = node_identity("semantic-state-effect", self._fingerprint, stable_path)
        self._expect_tensor(
            artifact.tensor, self._values[value_id].tensor, f"artifact {artifact.stable_path}"
        )
        self._values[value_id].consumers.append(effect_id)
        version_before = self._artifact_versions.get(artifact.artifact_id)
        version_after = 0 if version_before is None else version_before + 1
        self._artifact_versions[artifact.artifact_id] = version_after
        derivations = self._record(
            phase="semantic-link",
            rule=f"{SEMANTIC_COMPILER_VERSION}:state-write",
            source_path=artifact.stable_path,
            source_stable_path=stable_path,
            target_node_id=effect_id,
        )
        self._effects.append(
            SemanticStateEffect(
                effect_id=effect_id,
                node_id=effect_id,
                stable_path=stable_path,
                kind="write",
                artifact_id=artifact.artifact_id,
                input_value_id=value_id,
                output_value_id=None,
                owner_node_id=None,
                version_before=version_before,
                version_after=version_after,
                derivation_ids=derivations,
            )
        )
        return effect_id

    def _expand_workload_node(self, node: WorkloadNode) -> SemanticRegion:
        stable_path = f"semantic/{node.stable_path}"
        region_id = node_identity("semantic-region", self._fingerprint, stable_path)
        if isinstance(node, IRSequence):
            items = tuple(self._expand_workload_node(child) for child in node.children)
            derivations = self._record(
                phase="semantic-skeleton",
                rule=f"{SEMANTIC_COMPILER_VERSION}:workload-sequence",
                source_path=node.definition_id,
                source_stable_path=node.stable_path,
                target_node_id=region_id,
            )
            return SemanticRegion(
                local_id=node.local_id,
                kind="sequence",
                definition_id=node.definition_id,
                stable_path=stable_path,
                node_id=region_id,
                inputs=items[0].inputs if items else (),
                outputs=items[-1].outputs if items else (),
                items=items,
                state_effect_ids=(),
                attributes=(),
                derivation_ids=derivations,
            )
        return self._expand_model_call(node, stable_path, region_id)

    def _expand_model_call(
        self, call: IRModelCall, stable_path: str, region_id: str
    ) -> SemanticRegion:
        try:
            model = self._models[(call.model_name, call.model_version)]
        except KeyError as error:
            raise SemanticCompileError(
                f"model call {call.stable_path} cannot resolve "
                f"{call.model_name}@{call.model_version}"
            ) from error
        input_bindings: dict[str, str] = {}
        effect_ids: list[str] = []
        for port, artifact_name in call.inputs:
            try:
                artifact = self._workload_artifacts[artifact_name]
            except KeyError as error:
                raise SemanticCompileError(
                    f"unknown input artifact {artifact_name!r} at {call.stable_path}"
                ) from error
            effect_id, value_id = self._read_artifact(
                artifact, f"{stable_path}/artifact-read/{port}", region_id
            )
            effect_ids.append(effect_id)
            input_bindings[port] = value_id

        model_region, model_outputs = self._expand_entrypoint(
            model.root,
            call.entrypoint,
            input_bindings,
            stable_path=f"{stable_path}/model/{model.root.local_id}",
        )
        output_values: list[str] = []
        for port, artifact_name in call.outputs:
            try:
                artifact = self._workload_artifacts[artifact_name]
                value_id = model_outputs[port]
            except KeyError as error:
                raise SemanticCompileError(
                    f"invalid output binding {port!r}->{artifact_name!r} at {call.stable_path}"
                ) from error
            effect_ids.append(
                self._write_artifact(
                    artifact, value_id, f"{stable_path}/artifact-write/{port}"
                )
            )
            output_values.append(value_id)
        derivations = self._record(
            phase="semantic-skeleton",
            rule=f"{SEMANTIC_COMPILER_VERSION}:expand-model-call",
            source_path=call.definition_id,
            source_stable_path=call.stable_path,
            target_node_id=region_id,
        )
        return SemanticRegion(
            local_id=call.local_id,
            kind="model_call",
            definition_id=call.definition_id,
            stable_path=stable_path,
            node_id=region_id,
            inputs=tuple(input_bindings.values()),
            outputs=tuple(output_values),
            items=(model_region,),
            state_effect_ids=tuple(effect_ids),
            attributes=(
                ("entrypoint", call.entrypoint),
                ("model", call.model_name),
                ("model_version", call.model_version),
            ),
            derivation_ids=derivations,
        )

    def _expand_entrypoint(
        self,
        module: IRModule,
        entrypoint_name: str,
        inputs: dict[str, str],
        *,
        stable_path: str,
    ) -> tuple[SemanticRegion, dict[str, str]]:
        if module.module_kind != "composite":
            raise SemanticCompileError(f"cannot open entrypoint on primitive {module.stable_path}")
        try:
            entrypoint = module.entrypoint(entrypoint_name)
        except KeyError as error:
            raise SemanticCompileError(str(error)) from error
        self._validate_entry_inputs(entrypoint, inputs, module.stable_path)
        environment = dict(inputs)
        children = {child.local_id: child for child in module.children}
        if len(children) != len(module.children):
            raise SemanticCompileError(f"duplicate child id in {module.stable_path}")
        items: list[SemanticOperation | SemanticRegion] = []
        for step in entrypoint.steps:
            try:
                target = children[step.target]
            except KeyError as error:
                raise SemanticCompileError(
                    f"entrypoint {module.stable_path}:{entrypoint_name} references "
                    f"unknown child {step.target!r}"
                ) from error
            step_inputs: dict[str, str] = {}
            for port, value_name in step.inputs:
                try:
                    step_inputs[port] = environment[value_name]
                except KeyError as error:
                    raise SemanticCompileError(
                        f"undefined value {value_name!r} before {module.stable_path}/{step.local_id}"
                    ) from error
            item_path = f"{stable_path}/{target.local_id}"
            if target.module_kind == "primitive":
                item, step_outputs = self._expand_primitive(
                    target, step_inputs, stable_path=item_path, local_id=step.local_id
                )
            else:
                item, step_outputs = self._expand_entrypoint(
                    target, step.entrypoint, step_inputs, stable_path=item_path
                )
            items.append(item)
            for port, value_name in step.outputs:
                try:
                    environment[value_name] = step_outputs[port]
                except KeyError as error:
                    raise SemanticCompileError(
                        f"child {target.stable_path} did not produce port {port!r}"
                    ) from error
        outputs: dict[str, str] = {}
        for port in entrypoint.outputs:
            try:
                value_id = environment[port.name]
            except KeyError as error:
                raise SemanticCompileError(
                    f"entrypoint {module.stable_path}:{entrypoint_name} did not bind "
                    f"output {port.name!r}"
                ) from error
            self._expect_tensor(
                self._tensor(port.tensor),
                self._values[value_id].tensor,
                f"{module.stable_path}:{entrypoint_name} output {port.name}",
            )
            outputs[port.name] = value_id
        region_id = node_identity("semantic-region", self._fingerprint, stable_path)
        derivations = self._record(
            phase="semantic-model-expand",
            rule=f"{SEMANTIC_COMPILER_VERSION}:composite-entrypoint",
            source_path=module.definition_id,
            source_stable_path=module.stable_path,
            target_node_id=region_id,
        )
        return (
            SemanticRegion(
                local_id=module.local_id,
                kind="model_entrypoint",
                definition_id=module.definition_id,
                stable_path=stable_path,
                node_id=region_id,
                inputs=tuple(inputs[port.name] for port in entrypoint.inputs),
                outputs=tuple(outputs[port.name] for port in entrypoint.outputs),
                items=tuple(items),
                state_effect_ids=(),
                attributes=(("entrypoint", entrypoint_name),),
                derivation_ids=derivations,
            ),
            outputs,
        )

    def _validate_entry_inputs(
        self, entrypoint: IREntrypoint, inputs: dict[str, str], location: str
    ) -> None:
        expected_names = {port.name for port in entrypoint.inputs}
        if set(inputs) != expected_names:
            raise SemanticCompileError(
                f"entrypoint input mismatch at {location}: expected "
                f"{sorted(expected_names)}, found {sorted(inputs)}"
            )
        for port in entrypoint.inputs:
            self._expect_tensor(
                self._tensor(port.tensor),
                self._values[inputs[port.name]].tensor,
                f"{location} input {port.name}",
            )

    def _parameter_artifact(
        self, module: IRModule, state_name: str, tensor: SemanticTensorType, role: str
    ) -> SemanticStateArtifact:
        stable_path = f"semantic/{module.stable_path}/state/{state_name}"
        artifact_id = node_identity("semantic-state-artifact", self._fingerprint, stable_path)
        existing = self._artifacts.get(artifact_id)
        if existing is not None:
            return existing
        derivations = self._record(
            phase="semantic-link",
            rule=f"{SEMANTIC_COMPILER_VERSION}:model-state",
            source_path=module.definition_id,
            source_stable_path=f"{module.stable_path}/state/{state_name}",
            target_node_id=artifact_id,
        )
        artifact = SemanticStateArtifact(
            artifact_id=artifact_id,
            node_id=artifact_id,
            stable_path=stable_path,
            role=role,
            tensor=tensor,
            source_kind="model_state",
            initial_version=0,
            derivation_ids=derivations,
        )
        self._artifacts[artifact_id] = artifact
        self._artifact_versions[artifact_id] = artifact.initial_version
        return artifact

    def _expand_primitive(
        self,
        module: IRModule,
        inputs: dict[str, str],
        *,
        stable_path: str,
        local_id: str,
    ) -> tuple[SemanticOperation, dict[str, str]]:
        expected_names = {port.name for port in module.inputs}
        if set(inputs) != expected_names:
            raise SemanticCompileError(
                f"primitive input mismatch at {module.stable_path}: expected "
                f"{sorted(expected_names)}, found {sorted(inputs)}"
            )
        operation_id = node_identity("semantic-op", self._fingerprint, stable_path)
        operands: list[str] = []
        for port in module.inputs:
            value_id = inputs[port.name]
            self._expect_tensor(
                self._tensor(port.tensor),
                self._values[value_id].tensor,
                f"{module.stable_path} input {port.name}",
            )
            self._values[value_id].consumers.append(operation_id)
            operands.append(value_id)
        effect_ids: list[str] = []
        for state in module.state:
            artifact = self._parameter_artifact(
                module, state.name, self._tensor(state.tensor), state.role
            )
            effect_id, value_id = self._read_artifact(
                artifact, f"{stable_path}/state-read/{state.name}", operation_id
            )
            self._values[value_id].consumers.append(operation_id)
            effect_ids.append(effect_id)
            operands.append(value_id)
        outputs: dict[str, str] = {}
        result_ids: list[str] = []
        alias_of = operands[0] if module.operation in {"View", "Transpose"} else None
        for port in module.outputs:
            value_id = self._create_value(
                stable_path=f"{stable_path}/result/{port.name}",
                kind="tensor",
                tensor=self._tensor(port.tensor),
                producer_id=operation_id,
                source_path=module.definition_id,
                source_stable_path=f"{module.stable_path}/output/{port.name}",
                alias_of=alias_of,
            )
            outputs[port.name] = value_id
            result_ids.append(value_id)
        attributes = list(module.attributes)
        if alias_of is not None:
            attributes.append(("materialization", "zero"))
        derivations = self._record(
            phase="semantic-model-expand",
            rule=f"{SEMANTIC_COMPILER_VERSION}:primitive-entrypoint",
            source_path=module.definition_id,
            source_stable_path=module.stable_path,
            target_node_id=operation_id,
        )
        return (
            SemanticOperation(
                local_id=local_id,
                operation=module.operation or "",
                definition_id=module.definition_id,
                stable_path=stable_path,
                node_id=operation_id,
                operands=tuple(operands),
                results=tuple(result_ids),
                attributes=tuple(sorted(attributes)),
                state_effect_ids=tuple(effect_ids),
                derivation_ids=derivations,
            ),
            outputs,
        )

    def _verify(self, program: SemanticProgram) -> tuple[ValidationResult, ...]:
        items = tuple(program.root.walk_items()) + (program.root,)
        value_ids = {value.value_id for value in program.values}
        effect_ids = {effect.effect_id for effect in program.state_effects}
        all_operands = all(
            set(item.operands) <= value_ids and set(item.results) <= value_ids
            for item in items
            if isinstance(item, SemanticOperation)
        )
        all_effects = all(
            set(item.state_effect_ids) <= effect_ids
            for item in items
            if isinstance(item, SemanticOperation)
        )
        all_provenance = all(item.derivation_ids for item in items) and all(
            value.derivation_ids for value in program.values
        )
        concrete_shapes = all(
            all(isinstance(dimension, int) and dimension > 0 for dimension in value.tensor.shape)
            for value in program.values
        )
        serialized = canonical_json(program)
        no_physical = all(
            marker not in serialized
            for marker in ("local-m4/cpu", "local-m4/gpu", '"latency"', '"schedule"')
        )
        return (
            ValidationResult("operands-resolve", all_operands, "all operands/results resolve"),
            ValidationResult("state-effects-resolve", all_effects, "all operation effects resolve"),
            ValidationResult("provenance-complete", all_provenance, "all entities have derivation"),
            ValidationResult("shapes-concrete", concrete_shapes, "all value shapes are positive integers"),
            ValidationResult("hardware-independent", no_physical, "no placement/latency/schedule fact present"),
        )
