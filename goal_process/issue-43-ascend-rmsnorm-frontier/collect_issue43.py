#!/usr/bin/env python3
"""Bounded Issue 43 RMSNorm phase evidence collection on Ascend 910B2."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from groundupscale.ascend_rmsnorm_frontier import (
    RmsNormOperatorFrontierBundleWriter,
    RmsNormPhaseMeasurementBundleWriter,
    rmsnorm_memory_pattern_probe,
)
from groundupscale.pipeline import compile_analysis_plan
from groundupscale.run_bundle import verify_run_bundle


COHORT = "ascend-npu-23b93a89d5fecc79"
PLAN = Path("specs/plans/ascend-npu-transformer-demo.yaml")
PHASES = (
    "square",
    "reduce_sum",
    "mean_scale",
    "epsilon_add",
    "rsqrt",
    "input_scale",
    "weight_scale",
)
LANES = ("search", "independent-holdout")
WARMUP = 20
SAMPLES = 20
INNER = 20


def _phase_callable(name: str, x: torch.Tensor, weight: torch.Tensor):
    rows = x.reshape(-1, x.shape[-1])
    squared = rows * rows
    reduced = squared.sum(dim=-1, keepdim=True)
    mean = reduced / rows.shape[-1]
    stabilized = mean + 1e-5
    reciprocal = torch.rsqrt(stabilized)
    scaled = rows * reciprocal
    calls = {
        "square": lambda: rows * rows,
        "reduce_sum": lambda: squared.sum(dim=-1, keepdim=True),
        "mean_scale": lambda: reduced / rows.shape[-1],
        "epsilon_add": lambda: mean + 1e-5,
        "rsqrt": lambda: torch.rsqrt(stabilized),
        "input_scale": lambda: rows * reciprocal,
        "weight_scale": lambda: scaled * weight,
    }
    return calls[name]


def _measure(call) -> list[float]:
    for _ in range(WARMUP):
        call()
    torch.npu.synchronize()
    samples: list[float] = []
    for _ in range(SAMPLES):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(INNER):
            call()
        end.record()
        end.synchronize()
        torch.npu.synchronize()
        samples.append(float(start.elapsed_time(end) * 1_000_000 / INNER))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-store", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0":
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES must be exactly 0")
    root = Path.cwd()
    compiled = compile_analysis_plan(root, root / PLAN)
    operation = next(
        item
        for item in compiled.cost.cost_ir.walk_operations()
        if item.stable_path.endswith("/layer_0/input_norm")
    )
    assert operation.phase_graph is not None
    domain = {
        "hardware_cohort": COHORT,
        "stable_path": operation.stable_path,
        "operand_shapes": [list(item.shape) for item in operation.operand_types],
        "result_shapes": [list(item.shape) for item in operation.result_types],
        "dtype": "float32",
        "layout": "row-major-contiguous",
        "execution_mode": "pytorch-eager",
        "logical_device": "npu:0",
    }
    plan = {
        "schema": "groundupscale.dev/issue43-collection-plan/v1alpha1",
        "run_tag": args.run_tag,
        "phases": list(PHASES),
        "lanes": list(LANES),
        "warmup": WARMUP,
        "samples": SAMPLES,
        "inner_iterations": INNER,
        "cohort": COHORT,
        "visibility": "0",
        "acceptance": {
            "correctness": "torch_npu result matches CPU float32 oracle",
            "timing": "positive finite samples; IQR/median <= 0.10",
            "coverage": "7 phases x search+independent-holdout",
            "frontier": "public verify_run_bundle passes",
        },
    }
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
    if args.dry_run:
        return 0

    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    generator = torch.Generator(device="cpu").manual_seed(43)
    cpu_x = torch.randn((1, 512, 512), generator=generator, dtype=torch.float32)
    cpu_weight = torch.randn((512,), generator=generator, dtype=torch.float32)
    x = cpu_x.npu()
    weight = cpu_weight.npu()
    metadata = {
        "issue": 43,
        "lock_owner": Path(
            "/home/t00906153/.groundupscale/locks/ascend-910b2-host.owner"
        ).read_text(encoding="utf-8"),
        "started_at": datetime.now(UTC).isoformat(),
        "hardware_cohort": COHORT,
        "device_visibility": "0",
    }
    sources = []
    for lane_index, lane in enumerate(LANES):
        torch.manual_seed(43 + lane_index)
        for phase in operation.phase_graph.phases:
            call = _phase_callable(phase.phase_name, x, weight)
            target = call()
            torch.npu.synchronize()
            reference = _phase_callable(
                phase.phase_name, cpu_x, cpu_weight
            )()
            torch.testing.assert_close(target.cpu(), reference, rtol=1e-4, atol=1e-5)
            samples = _measure(call)
            median = float(statistics.median(samples))
            q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive")
            if (q3 - q1) / median > 0.10:
                raise RuntimeError(f"{phase.phase_name}/{lane}: timing dispersion")
            rows = x.reshape(-1, x.shape[-1])
            memory_pattern_call = rmsnorm_memory_pattern_probe(
                phase.memory_capability_resource,
                rows,
                rows * rows,
                weight,
            )
            memory_samples = _measure(memory_pattern_call)
            memory_median = float(statistics.median(memory_samples))
            memory_q1, _, memory_q3 = statistics.quantiles(
                memory_samples, n=4, method="inclusive"
            )
            if (memory_q3 - memory_q1) / memory_median > 0.10:
                raise RuntimeError(
                    f"{phase.phase_name}/{lane}: memory-pattern timing dispersion"
                )
            phase_run_id = f"issue43-{args.run_tag}-{phase.phase_name}-{lane}"
            sources.append(
                RmsNormPhaseMeasurementBundleWriter().run(
                    args.artifact_store,
                    run_id=phase_run_id,
                    phase=phase,
                    execution_domain=domain,
                    lane=lane,
                    evidence_kind="exact-operation-probe",
                    candidate={
                        "candidate_id": f"torch-npu-{phase.phase_name}-eager-v1",
                        "candidate_family": f"torch-npu.{phase.phase_name}",
                        "candidate_version": "v1",
                    },
                    compute_or_exact_duration_ns=median,
                    memory_pattern_floor_ns=memory_median,
                    standard_uncertainty_ns=float(statistics.stdev(samples)),
                    raw_samples_ns=samples,
                    memory_pattern_raw_samples_ns=memory_samples,
                    run_metadata=metadata,
                    compilation_fingerprint=compiled.cost.compilation_fingerprint,
                )
            )
    frontier = RmsNormOperatorFrontierBundleWriter().run(
        args.artifact_store,
        run_id=f"issue43-{args.run_tag}-rmsnorm-frontier",
        operation=operation,
        execution_domain=domain,
        source_runs=sources,
        compilation_fingerprint=compiled.cost.compilation_fingerprint,
    )
    verification = verify_run_bundle(frontier)
    if verification["passed"] is not True:
        raise RuntimeError(verification)
    metadata["finished_at"] = datetime.now(UTC).isoformat()
    print(json.dumps({"frontier": str(frontier), "verification": verification}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
