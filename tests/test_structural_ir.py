from __future__ import annotations

from pathlib import Path

from groundupscale.compiler import ModelBuilder, WorkloadBuilder
from groundupscale.ir import IRModelCall, canonical_json
from groundupscale.specs import SpecRepository


REPOSITORY_ROOT = Path(__file__).parents[1]
PLAN = REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"


def _build_irs():
    bundle = SpecRepository(REPOSITORY_ROOT).load_analysis_plan(PLAN)
    model_document = bundle.models["two-layer-transformer"]
    model_ir = ModelBuilder().build(model_document)
    workload_ir = WorkloadBuilder().build(
        bundle.workload, models_by_reference=bundle.models_by_reference
    )
    return model_ir, workload_ir


def test_model_repeat_expands_to_nested_layers_with_three_identities() -> None:
    model_ir, _ = _build_irs()

    assert model_ir.root.stable_path == "model/two-layer-transformer/transformer"
    assert [child.local_id for child in model_ir.root.children] == ["layer_0", "layer_1"]
    layer_0, layer_1 = model_ir.root.children
    assert layer_0.stable_path.endswith("/layer_0")
    assert layer_1.stable_path.endswith("/layer_1")
    assert layer_0.definition_id == layer_1.definition_id
    assert layer_0.node_id != layer_1.node_id
    assert [child.local_id for child in layer_0.children] == [
        "input_norm",
        "attention",
        "residual_1",
        "post_norm",
        "mlp",
        "residual_2",
    ]
    assert [child.local_id for child in layer_0.children[1].children][:3] == [
        "q_proj",
        "k_proj",
        "v_proj",
    ]
    assert all(module.derivation_ids for module in model_ir.walk_modules())


def test_repeat_call_is_expanded_but_model_call_remains_a_workload_leaf() -> None:
    model_ir, workload_ir = _build_irs()

    prefill = model_ir.entrypoint("prefill")
    assert [step.target for step in prefill.steps] == ["layer_0", "layer_1"]
    assert dict(prefill.steps[0].inputs) == {"hidden": "hidden"}
    assert dict(prefill.steps[0].outputs) == {"hidden": "hidden__layer_0"}
    assert dict(prefill.steps[1].inputs) == {"hidden": "hidden__layer_0"}
    assert dict(prefill.steps[1].outputs) == {"hidden": "hidden"}

    assert len(workload_ir.root.children) == 1
    model_call = workload_ir.root.children[0]
    assert isinstance(model_call, IRModelCall)
    assert model_call.model_name == "two-layer-transformer"
    assert model_call.entrypoint == "prefill"
    assert not hasattr(model_call, "children")
    assert model_call.derivation_ids


def test_structural_ir_is_canonical_and_deterministic() -> None:
    first_model, first_workload = _build_irs()
    second_model, second_workload = _build_irs()

    assert first_model == second_model
    assert first_workload == second_workload
    assert canonical_json(first_model) == canonical_json(second_model)
    assert canonical_json(first_workload) == canonical_json(second_workload)
    assert first_model.compilation_fingerprint == second_model.compilation_fingerprint
    assert first_workload.compilation_fingerprint == second_workload.compilation_fingerprint
