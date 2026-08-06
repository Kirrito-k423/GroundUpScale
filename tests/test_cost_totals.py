from __future__ import annotations

from groundupscale.ir import CostRegion, canonical_json

from test_cost_lowerer import _lower_cost


def _region(result, suffix: str) -> CostRegion:
    regions = [result.cost_ir.root]
    regions.extend(
        item
        for item in result.cost_ir.root.walk_items()
        if isinstance(item, CostRegion)
    )
    matches = [region for region in regions if region.stable_path.endswith(suffix)]
    assert len(matches) == 1, suffix
    return matches[0]


def test_one_layer_totals_match_hand_calculated_literals() -> None:
    result = _lower_cost()
    layer = _region(result, "/layer_0")
    metrics = layer.metrics

    assert metrics.flops == 4_855_425_024
    assert metrics.logical_read_bytes == 93_327_360
    assert metrics.logical_write_bytes == 70_254_592
    assert metrics.materialized_read_bytes == 83_890_176
    assert metrics.materialized_write_bytes == 60_817_408
    assert metrics.parameter_read_bytes == 16_781_312
    assert metrics.buffer_read_bytes == 1_048_576
    assert metrics.activation_read_bytes == 75_497_472
    assert metrics.explicit_activation_bytes == 60_817_408
    assert metrics.alias_result_bytes == 9_437_184


def test_two_layer_summary_matches_hand_calculated_literals() -> None:
    result = _lower_cost()
    summary = result.cost_ir.summary
    metrics = summary.metrics

    assert result.compilation_fingerprint == (
        "65dba693c740d94cfbc5add162c2545e50b65953fd59c8fd5949ff7cd8f8c46e"
    )
    assert metrics.flops == 9_710_850_048
    assert metrics.logical_read_bytes == 186_654_720
    assert metrics.logical_write_bytes == 140_509_184
    assert metrics.materialized_read_bytes == 167_780_352
    assert metrics.materialized_write_bytes == 121_634_816
    assert metrics.parameter_read_bytes == 33_562_624
    assert metrics.buffer_read_bytes == 2_097_152
    assert metrics.activation_read_bytes == 150_994_944
    assert metrics.explicit_activation_bytes == 121_634_816
    assert metrics.alias_result_bytes == 18_874_368
    assert summary.parameter_bytes == 33_562_624
    assert summary.buffer_bytes == 2_097_152
    assert summary.workload_artifact_bytes == 2_097_152
    assert summary.serial_flops == 9_710_850_048
    assert summary.ideal_parallel_critical_path_flops == 6_489_624_576
    assert metrics.logical_read_bytes == (
        metrics.parameter_read_bytes
        + metrics.buffer_read_bytes
        + metrics.activation_read_bytes
    )
    assert metrics.logical_write_bytes == (
        metrics.explicit_activation_bytes + metrics.alias_result_bytes
    )


def test_cpu_and_mps_cost_ir_are_identical() -> None:
    cpu = _lower_cost("mac-cpu-prefill.yaml")
    mps = _lower_cost("mac-mps-prefill.yaml")

    assert cpu.compilation_fingerprint == mps.compilation_fingerprint
    assert canonical_json(cpu.cost_ir) == canonical_json(mps.cost_ir)
