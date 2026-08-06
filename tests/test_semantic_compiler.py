from __future__ import annotations

from pathlib import Path

import pytest

from groundupscale.compiler import (
    CompilationContext,
    ModelBuilder,
    SemanticCompileError,
    SemanticCompileRequest,
    SemanticCompiler,
    WorkloadBuilder,
    semantic_deployment_plan,
)
from groundupscale.ir import SemanticOperation, SemanticRegion, canonical_json
from groundupscale.specs import SpecRepository


REPOSITORY_ROOT = Path(__file__).parents[1]


def _request(plan_name: str = "mac-cpu-prefill.yaml") -> SemanticCompileRequest:
    bundle = SpecRepository(REPOSITORY_ROOT).load_analysis_plan(
        REPOSITORY_ROOT / "specs/plans" / plan_name
    )
    models = tuple(
        ModelBuilder().build(document)
        for _, document in sorted(bundle.models.items())
    )
    workload = WorkloadBuilder().build(
        bundle.workload, models_by_reference=bundle.models_by_reference
    )
    return SemanticCompileRequest(
        workload=workload,
        models=models,
        analysis_case=bundle.analysis_case,
        deployment=semantic_deployment_plan(bundle.deployment_intent),
        context=CompilationContext(compiler_version="0.1.0", plugin_versions=()),
    )


def _walk_items(region: SemanticRegion):
    yield region
    for item in region.items:
        if isinstance(item, SemanticRegion):
            yield from _walk_items(item)
        else:
            yield item


def test_semantic_compiler_preserves_nested_regions_and_expands_all_primitives() -> None:
    result = SemanticCompiler().compile(_request())
    program = result.semantic_ir

    assert result.compilation_fingerprint == (
        "02d0facf395b847acc2bb850e039136b27b696a476d33bd26d58369eba1b2233"
    )

    assert program.root.kind == "analysis_case"
    workload_region = program.root.items[0]
    assert isinstance(workload_region, SemanticRegion)
    assert workload_region.kind == "sequence"
    model_call_region = workload_region.items[0]
    assert isinstance(model_call_region, SemanticRegion)
    assert model_call_region.kind == "model_call"
    model_region = model_call_region.items[0]
    assert isinstance(model_region, SemanticRegion)
    assert model_region.kind == "model_entrypoint"
    assert [item.local_id for item in model_region.items] == ["layer_0", "layer_1"]

    operations = [
        item for item in _walk_items(program.root) if isinstance(item, SemanticOperation)
    ]
    assert len(operations) == 52
    assert {operation.operation for operation in operations} == {
        "MatMul",
        "Add",
        "RMSNorm",
        "Softmax",
        "SiLU",
        "Mul",
        "View",
        "Transpose",
    }
    assert all(
        isinstance(dimension, int)
        for value in program.values
        for dimension in value.tensor.shape
    )
    assert len(program.values) == 73
    assert len(program.state_artifacts) == 22
    assert len(program.state_effects) == 22


def test_values_state_effects_and_provenance_are_closed_and_queryable() -> None:
    result = SemanticCompiler().compile(_request())
    program = result.semantic_ir
    items = list(_walk_items(program.root))
    values = {value.value_id: value for value in program.values}
    effects = {effect.effect_id: effect for effect in program.state_effects}

    assert all(item.derivation_ids for item in items)
    assert all(value.derivation_ids for value in program.values)
    assert all(artifact.derivation_ids for artifact in program.state_artifacts)
    assert all(effect.derivation_ids for effect in program.state_effects)
    assert all(record.target_node_ids for record in result.provenance.records)
    assert len({item.node_id for item in items}) == len(items)

    for item in items:
        if isinstance(item, SemanticOperation):
            assert all(value_id in values for value_id in item.operands)
            assert all(value_id in values for value_id in item.results)
            assert all(effect_id in effects for effect_id in item.state_effect_ids)
            for operand in item.operands:
                assert item.node_id in values[operand].consumer_ids

    effect_kinds = {effect.kind for effect in program.state_effects}
    assert {"read", "write"} <= effect_kinds
    assert any(artifact.role == "parameter" for artifact in program.state_artifacts)
    assert any(artifact.role == "input" for artifact in program.state_artifacts)
    assert any(artifact.role == "output" for artifact in program.state_artifacts)
    assert sum(artifact.role == "buffer" for artifact in program.state_artifacts) == 2
    assert all(validation.passed for validation in result.validation_results)
    assert all(value.producer_id is not None for value in program.values)
    assert all(value.consumer_ids for value in program.values)

    reads = [effect for effect in program.state_effects if effect.kind == "read"]
    writes = [effect for effect in program.state_effects if effect.kind == "write"]
    assert all(effect.version_before == 0 and effect.version_after is None for effect in reads)
    assert len(writes) == 1
    assert writes[0].version_before is None
    assert writes[0].version_after == 0

    alias_operations = [
        item
        for item in items
        if isinstance(item, SemanticOperation)
        and item.operation in {"View", "Transpose"}
    ]
    assert alias_operations
    for operation in alias_operations:
        assert dict(operation.attributes)["materialization"] == "zero"
        assert values[operation.results[0]].alias_of == operation.operands[0]


def test_physical_cpu_or_mps_placement_does_not_change_semantic_ir() -> None:
    cpu = SemanticCompiler().compile(_request("mac-cpu-prefill.yaml"))
    mps = SemanticCompiler().compile(_request("mac-mps-prefill.yaml"))

    assert cpu.compilation_fingerprint == mps.compilation_fingerprint
    assert canonical_json(cpu.semantic_ir) == canonical_json(mps.semantic_ir)
    serialized = canonical_json(cpu.semantic_ir)
    assert "local-m4/cpu" not in serialized
    assert "local-m4/gpu" not in serialized


def test_unbound_or_inconsistent_shape_is_rejected() -> None:
    request = _request()
    bad_shape = request.analysis_case.spec.shape.model_copy(
        update={"bindings": {"B": 1, "S": 512, "H": 513, "NH": 8, "D": 64, "I": 2048}}
    )
    bad_spec = request.analysis_case.spec.model_copy(update={"shape": bad_shape})
    bad_case = request.analysis_case.model_copy(update={"spec": bad_spec})

    with pytest.raises(SemanticCompileError, match=r"H == NH \* D"):
        SemanticCompiler().compile(
            SemanticCompileRequest(
                workload=request.workload,
                models=request.models,
                analysis_case=bad_case,
                deployment=request.deployment,
                context=request.context,
            )
        )


def test_analysis_dtype_must_match_workload_entry_tensors() -> None:
    request = _request()
    bad_shape = request.analysis_case.spec.shape.model_copy(
        update={"dtype": "bfloat16"}
    )
    bad_spec = request.analysis_case.spec.model_copy(update={"shape": bad_shape})
    bad_case = request.analysis_case.model_copy(update={"spec": bad_spec})

    with pytest.raises(SemanticCompileError, match="AnalysisCase dtype bfloat16"):
        SemanticCompiler().compile(
            SemanticCompileRequest(
                workload=request.workload,
                models=request.models,
                analysis_case=bad_case,
                deployment=request.deployment,
                context=request.context,
            )
        )
