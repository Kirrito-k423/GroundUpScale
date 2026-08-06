from __future__ import annotations

from pathlib import Path

from groundupscale.compiler import (
    CompilationContext,
    CostLowerer,
    CostLoweringRequest,
    ModelBuilder,
    SemanticCompileRequest,
    SemanticCompiler,
    WorkloadBuilder,
    semantic_deployment_plan,
)
from groundupscale.ir import canonical_json
from groundupscale.specs import SpecRepository


REPOSITORY_ROOT = Path(__file__).parents[1]


def _lower_cost(plan_name: str = "mac-cpu-prefill.yaml"):
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
    semantic = SemanticCompiler().compile(
        SemanticCompileRequest(
            workload=workload,
            models=models,
            analysis_case=bundle.analysis_case,
            deployment=semantic_deployment_plan(bundle.deployment_intent),
            context=CompilationContext(compiler_version="0.1.0", plugin_versions=()),
        )
    )
    return CostLowerer().lower(
        CostLoweringRequest(
            semantic_ir=semantic.semantic_ir,
            provenance=semantic.provenance,
        )
    )


def _operation(result, suffix: str):
    matches = [
        operation
        for operation in result.cost_ir.walk_operations()
        if operation.stable_path.endswith(suffix)
    ]
    assert len(matches) == 1, suffix
    return matches[0]


def test_atomic_cost_rules_match_independent_literal_examples() -> None:
    result = _lower_cost()

    q_proj = _operation(result, "/layer_0/attention/q_proj")
    assert q_proj.metrics.flops == 268_435_456
    assert q_proj.metrics.logical_read_bytes == 2_097_152
    assert q_proj.metrics.logical_write_bytes == 1_048_576
    assert q_proj.metrics.parameter_read_bytes == 1_048_576
    assert q_proj.formula.rule_id == "core.cost-rule.matmul/v1alpha1"

    qk = _operation(result, "/layer_0/attention/qk_matmul")
    assert qk.metrics.flops == 268_435_456
    assert qk.metrics.logical_read_bytes == 2_097_152
    assert qk.metrics.logical_write_bytes == 8_388_608

    input_norm = _operation(result, "/layer_0/input_norm")
    assert input_norm.metrics.flops == 1_049_600
    assert input_norm.metrics.logical_read_bytes == 1_050_624
    assert input_norm.metrics.parameter_read_bytes == 2_048

    softmax = _operation(result, "/layer_0/attention/softmax")
    assert softmax.metrics.flops == 10_477_568
    assert softmax.metrics.logical_read_bytes == 8_388_608
    assert softmax.metrics.logical_write_bytes == 8_388_608

    causal_mask = _operation(result, "/layer_0/attention/causal_mask")
    assert causal_mask.metrics.flops == 2_097_152
    assert causal_mask.metrics.logical_read_bytes == 9_437_184
    assert causal_mask.metrics.logical_write_bytes == 8_388_608
    assert causal_mask.metrics.buffer_read_bytes == 1_048_576


def test_alias_rules_expose_logical_shape_without_fake_materialization() -> None:
    result = _lower_cost()
    q_view = _operation(result, "/layer_0/attention/q_view")

    assert q_view.metrics.flops == 0
    assert q_view.metrics.logical_read_bytes == 1_048_576
    assert q_view.metrics.logical_write_bytes == 1_048_576
    assert q_view.metrics.materialized_read_bytes == 0
    assert q_view.metrics.materialized_write_bytes == 0
    assert q_view.metrics.alias_result_bytes == 1_048_576
    assert q_view.formula.rule_id == "core.cost-rule.alias/v1alpha1"


def test_cost_dependencies_and_provenance_follow_semantic_values() -> None:
    result = _lower_cost()
    q_proj = _operation(result, "/layer_0/attention/q_proj")
    q_view = _operation(result, "/layer_0/attention/q_view")
    q_transpose = _operation(result, "/layer_0/attention/q_transpose")

    assert q_view.dependency_cost_node_ids == (q_proj.node_id,)
    assert q_transpose.dependency_cost_node_ids == (q_view.node_id,)
    assert all(operation.derivation_ids for operation in result.cost_ir.walk_operations())
    assert all(record.target_node_ids for record in result.provenance.records)
    assert all(validation.passed for validation in result.validation_results)
    serialized = canonical_json(result.cost_ir)
    assert "local-m4/cpu" not in serialized
    assert "local-m4/gpu" not in serialized
    assert '"latency"' not in serialized
