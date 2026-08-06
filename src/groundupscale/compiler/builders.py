"""Deterministic structural Spec-to-IR builders."""

from __future__ import annotations

from hashlib import sha256
from typing import Iterable

from groundupscale.ir.common import (
    DerivationRecord,
    canonical_json,
    content_fingerprint,
    derivation_identity,
    node_identity,
)
from groundupscale.ir.model import (
    IRCallStep,
    IREntrypoint,
    IRModule,
    IRPort,
    IRState,
    IRTensorType,
    ModelIR,
)
from groundupscale.ir.workload import (
    IRArtifact,
    IRModelCall,
    IRSequence,
    WorkloadIR,
    WorkloadNode,
)
from groundupscale.schemas.v1alpha1 import (
    CompositeModuleSpec,
    EntrypointSpec,
    ModelCallNodeSpec,
    ModelSpecDocument,
    ModuleRepeatSpec,
    PortSpec,
    PrimitiveModuleSpec,
    RepeatCallStepSpec,
    SequenceNodeSpec,
    TensorSpec,
    WorkloadSpecDocument,
)


MODEL_BUILDER_VERSION = "core.model-builder/v1alpha1"
WORKLOAD_BUILDER_VERSION = "core.workload-builder/v1alpha1"


def _tensor(spec: TensorSpec) -> IRTensorType:
    return IRTensorType(dtype=spec.dtype, shape=spec.shape, layout=spec.layout)


def _port(spec: PortSpec) -> IRPort:
    return IRPort(name=spec.name, tensor=_tensor(spec.tensor))


def _sorted_bindings(bindings: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(bindings.items()))


class ModelBuilder:
    """Expand declarative model repeat while retaining definition identity."""

    def __init__(self) -> None:
        self._provenance: list[DerivationRecord] = []
        self._fingerprint = ""
        self._source_path = ""

    def build(self, document: ModelSpecDocument) -> ModelIR:
        self._provenance = []
        self._source_path = f"ModelSpec/{document.metadata.name}@{document.metadata.version}"
        effective = document.model_dump(mode="json")
        self._fingerprint = content_fingerprint(MODEL_BUILDER_VERSION, effective)
        root_path = f"model/{document.metadata.name}/{document.spec.root.id}"
        definition_path = (
            f"model-definition/{document.metadata.name}@{document.metadata.version}/"
            f"{document.spec.root.id}"
        )
        root = self._expand_module(
            document.spec.root,
            stable_path=root_path,
            definition_path=definition_path,
        )
        symbols = tuple(
            (
                name,
                tuple(
                    sorted(
                        {
                            "type": symbol.type,
                            "minimum": symbol.minimum,
                            "maximum": symbol.maximum,
                        }.items()
                    )
                ),
            )
            for name, symbol in sorted(document.spec.symbols.items())
        )
        return ModelIR(
            schema="groundupscale.dev/model-ir/v1alpha1",
            name=document.metadata.name,
            version=document.metadata.version,
            compilation_fingerprint=self._fingerprint,
            source_sha256=sha256(canonical_json(effective).encode("utf-8")).hexdigest(),
            symbols=symbols,
            constraints=document.spec.constraints,
            root=root,
            provenance=tuple(self._provenance),
        )

    def _record(
        self, stable_path: str, node_id: str, rule: str
    ) -> tuple[str, ...]:
        derivation_id = derivation_identity(rule, self._fingerprint, stable_path)
        self._provenance.append(
            DerivationRecord(
                derivation_id=derivation_id,
                phase="model-build",
                rule=rule,
                source_path=self._source_path,
                source_stable_path=stable_path,
                target_node_ids=(node_id,),
            )
        )
        return (derivation_id,)

    def _entrypoint(
        self, entrypoint: EntrypointSpec, children: tuple[IRModule, ...]
    ) -> IREntrypoint:
        steps: list[IRCallStep] = []
        for step in entrypoint.steps:
            if not isinstance(step, RepeatCallStepSpec):
                steps.append(
                    IRCallStep(
                        local_id=step.id,
                        target=step.target,
                        entrypoint=step.entrypoint,
                        inputs=_sorted_bindings(step.inputs),
                        outputs=_sorted_bindings(step.outputs),
                    )
                )
                continue

            repeated = [child for child in children if child.repeat_group == step.group]
            if not repeated:
                raise ValueError(
                    f"repeat_call {step.id!r} references empty group {step.group!r}"
                )
            carried_value = step.initial
            for index, child in enumerate(repeated):
                is_last = index == len(repeated) - 1
                output_value = (
                    step.result if is_last else f"{step.initial}__{child.local_id}"
                )
                steps.append(
                    IRCallStep(
                        local_id=f"{step.id}_{index}",
                        target=child.local_id,
                        entrypoint=step.entrypoint,
                        inputs=((step.input_port, carried_value),),
                        outputs=((step.output_port, output_value),),
                        source_kind="repeat_call",
                    )
                )
                carried_value = output_value
        return IREntrypoint(
            name=entrypoint.name,
            inputs=tuple(_port(port) for port in entrypoint.inputs),
            outputs=tuple(_port(port) for port in entrypoint.outputs),
            steps=tuple(steps),
        )

    def _expand_module(
        self,
        module: PrimitiveModuleSpec | CompositeModuleSpec,
        *,
        stable_path: str,
        definition_path: str,
        local_id: str | None = None,
        repeat_group: str | None = None,
        repeat_index: int | None = None,
    ) -> IRModule:
        concrete_id = local_id or module.id
        children: list[IRModule] = []
        if isinstance(module, CompositeModuleSpec):
            for child in module.children:
                if isinstance(child, ModuleRepeatSpec):
                    for index in range(child.count):
                        try:
                            child_id = child.id_template.format(index=index)
                        except (KeyError, ValueError) as error:
                            raise ValueError(
                                f"invalid id_template {child.id_template!r} in {stable_path}"
                            ) from error
                        children.append(
                            self._expand_module(
                                child.template,
                                stable_path=f"{stable_path}/{child_id}",
                                definition_path=f"{definition_path}/{child.id}[*]",
                                local_id=child_id,
                                repeat_group=child.id,
                                repeat_index=index,
                            )
                        )
                else:
                    children.append(
                        self._expand_module(
                            child,
                            stable_path=f"{stable_path}/{child.id}",
                            definition_path=f"{definition_path}/{child.id}",
                        )
                    )

        node_id = node_identity("modelir", self._fingerprint, stable_path)
        rule = (
            f"{MODEL_BUILDER_VERSION}:expand-repeat"
            if repeat_group is not None
            else f"{MODEL_BUILDER_VERSION}:instantiate-module"
        )
        derivation_ids = self._record(stable_path, node_id, rule)
        if isinstance(module, PrimitiveModuleSpec):
            entrypoints: tuple[IREntrypoint, ...] = ()
            inputs = tuple(_port(port) for port in module.inputs)
            outputs = tuple(_port(port) for port in module.outputs)
            state = tuple(
                IRState(
                    name=item.name,
                    role=item.role,
                    tensor=_tensor(item.tensor),
                    trainable=item.trainable,
                )
                for item in module.state
            )
            attributes = tuple(sorted(module.attributes.items()))
            operation = module.operation
            module_kind = "primitive"
        else:
            entrypoints = tuple(
                self._entrypoint(entrypoint, tuple(children))
                for entrypoint in module.entrypoints
            )
            inputs = ()
            outputs = ()
            state = ()
            attributes = ()
            operation = None
            module_kind = "composite"
        return IRModule(
            local_id=concrete_id,
            module_kind=module_kind,
            operation=operation,
            definition_id=definition_path,
            stable_path=stable_path,
            node_id=node_id,
            derivation_ids=derivation_ids,
            inputs=inputs,
            outputs=outputs,
            state=state,
            attributes=attributes,
            entrypoints=entrypoints,
            children=tuple(children),
            repeat_group=repeat_group,
            repeat_index=repeat_index,
        )


class WorkloadBuilder:
    """Build logical workload control without expanding referenced models."""

    def __init__(self) -> None:
        self._provenance: list[DerivationRecord] = []
        self._fingerprint = ""
        self._source_path = ""

    def build(
        self,
        document: WorkloadSpecDocument,
        *,
        models_by_reference: dict[str, ModelSpecDocument],
    ) -> WorkloadIR:
        self._provenance = []
        self._source_path = f"WorkloadSpec/{document.metadata.name}@{document.metadata.version}"
        effective = document.model_dump(mode="json")
        model_versions = tuple(
            (path, model.metadata.name, model.metadata.version)
            for path, model in sorted(models_by_reference.items())
        )
        self._fingerprint = content_fingerprint(
            WORKLOAD_BUILDER_VERSION, effective, model_versions
        )
        root = self._expand_node(
            document.spec.root,
            parent_path=f"workload/{document.metadata.name}",
            models_by_reference=models_by_reference,
        )
        if not isinstance(root, IRSequence):
            raise ValueError("WorkloadSpec root must lower to a Sequence")
        return WorkloadIR(
            schema="groundupscale.dev/workload-ir/v1alpha1",
            name=document.metadata.name,
            version=document.metadata.version,
            compilation_fingerprint=self._fingerprint,
            source_sha256=sha256(canonical_json(effective).encode("utf-8")).hexdigest(),
            artifacts=tuple(
                IRArtifact(
                    name=artifact.name,
                    tensor=_tensor(artifact.tensor),
                    role=artifact.role,
                )
                for artifact in document.spec.artifacts
            ),
            root=root,
            provenance=tuple(self._provenance),
        )

    def _record(self, stable_path: str, node_id: str, rule: str) -> tuple[str, ...]:
        derivation_id = derivation_identity(rule, self._fingerprint, stable_path)
        self._provenance.append(
            DerivationRecord(
                derivation_id=derivation_id,
                phase="workload-build",
                rule=rule,
                source_path=self._source_path,
                source_stable_path=stable_path,
                target_node_ids=(node_id,),
            )
        )
        return (derivation_id,)

    def _expand_node(
        self,
        node: ModelCallNodeSpec | SequenceNodeSpec,
        *,
        parent_path: str,
        models_by_reference: dict[str, ModelSpecDocument],
    ) -> WorkloadNode:
        stable_path = f"{parent_path}/{node.id}"
        node_id = node_identity("workloadir", self._fingerprint, stable_path)
        definition_id = f"workload-definition/{self._source_path}/{stable_path}"
        if isinstance(node, ModelCallNodeSpec):
            try:
                model = models_by_reference[node.model.path]
            except KeyError as error:
                raise ValueError(
                    f"unresolved model reference {node.model.path!r} at {stable_path}"
                ) from error
            derivations = self._record(
                stable_path, node_id, f"{WORKLOAD_BUILDER_VERSION}:model-call"
            )
            return IRModelCall(
                local_id=node.id,
                definition_id=definition_id,
                stable_path=stable_path,
                node_id=node_id,
                derivation_ids=derivations,
                model_name=model.metadata.name,
                model_version=model.metadata.version,
                model_reference=node.model.path,
                entrypoint=node.entrypoint,
                inputs=_sorted_bindings(node.inputs),
                outputs=_sorted_bindings(node.outputs),
            )
        children = tuple(
            self._expand_node(
                child,
                parent_path=stable_path,
                models_by_reference=models_by_reference,
            )
            for child in node.children
        )
        derivations = self._record(
            stable_path, node_id, f"{WORKLOAD_BUILDER_VERSION}:sequence"
        )
        return IRSequence(
            local_id=node.id,
            definition_id=definition_id,
            stable_path=stable_path,
            node_id=node_id,
            derivation_ids=derivations,
            children=children,
        )
