#!/usr/bin/env python3
"""Collect one independent Ascend 910B2 diagnostic-ablation session.

The script intentionally emits raw measurements only.  Bundle construction and
Verdict evaluation happen later on the Mac through GroundUpScale's normal
digest-verifying diagnostic seam.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import median

import torch
import torch_npu

SHAPE = 512
SEED = 20260811
WARMUP = 100
SAMPLES = 20
INNER_ITERATIONS = 100
Q_PATH = (
    "semantic/model/two-layer-transformer/transformer/"
    "layer-0/attention/q-proj"
)
K_PATH = (
    "semantic/model/two-layer-transformer/transformer/"
    "layer-0/attention/k-proj"
)
V_PATH = (
    "semantic/model/two-layer-transformer/transformer/"
    "layer-0/attention/v-proj"
)


def _tensor_sha256(value: torch.Tensor) -> str:
    payload = value.detach().cpu().contiguous().numpy().tobytes()
    return sha256(payload).hexdigest()


def _device_samples(operation, *, profiled: bool = False) -> list[float]:
    profiler = (
        torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        if profiled
        else nullcontext()
    )
    samples: list[float] = []
    with profiler:
        for _ in range(SAMPLES):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(INNER_ITERATIONS):
                operation()
            end.record()
            end.synchronize()
            torch.npu.synchronize()
            samples.append(
                float(start.elapsed_time(end))
                * 1_000_000.0
                / INNER_ITERATIONS
            )
    return samples


def _correctness(
    left_cpu: torch.Tensor,
    right_cpu: torch.Tensor,
    observed: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    expected = torch.matmul(left_cpu.unsqueeze(0), right_cpu)
    actual = observed.detach().cpu()
    difference = (actual - expected).abs()
    tolerance = atol + rtol * expected.abs()
    return {
        "passed": bool(torch.all(difference <= tolerance).item()),
        "atol": atol,
        "rtol": rtol,
        "expected_sha256": _tensor_sha256(expected),
        "observed_sha256": _tensor_sha256(actual),
        "max_abs_difference": float(difference.max().item()),
        "mismatched_elements": int((difference > tolerance).sum().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0":
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES must be exactly 0")
    torch.manual_seed(SEED)
    left_cpu = torch.randn((SHAPE, SHAPE), dtype=torch.float32)
    right_cpu = torch.randn((SHAPE, SHAPE), dtype=torch.float32)
    k_left_cpu = torch.randn((SHAPE, SHAPE), dtype=torch.float32)
    k_right_cpu = torch.randn((SHAPE, SHAPE), dtype=torch.float32)
    v_left_cpu = torch.randn((SHAPE, SHAPE), dtype=torch.float32)
    v_right_cpu = torch.randn((SHAPE, SHAPE), dtype=torch.float32)
    left = left_cpu.to("npu:0")
    right = right_cpu.to("npu:0")
    k_left = k_left_cpu.to("npu:0")
    k_right = k_right_cpu.to("npu:0")
    v_left = v_left_cpu.to("npu:0")
    v_right = v_right_cpu.to("npu:0")
    left_batched = left.unsqueeze(0)
    frontier_left = left_batched[0]
    k_frontier_left = k_left.unsqueeze(0)[0]
    v_frontier_left = v_left.unsqueeze(0)[0]

    def frontier_adapter() -> torch.Tensor:
        """Measure #31's 2-D kernel; logical batch views stay outside timing."""
        return torch.matmul(frontier_left, right)

    def standalone() -> torch.Tensor:
        return torch.ops.aten.matmul.default(frontier_left, right)

    def dispatched() -> torch.Tensor:
        return torch.matmul(frontier_left, right)

    def k_baseline() -> torch.Tensor:
        return torch.ops.aten.matmul.default(k_frontier_left, k_right)

    def v_baseline() -> torch.Tensor:
        return torch.ops.aten.matmul.default(v_frontier_left, v_right)

    def copied() -> torch.Tensor:
        return dispatched().clone().clone()

    def synchronized() -> torch.Tensor:
        output = copied()
        for _ in range(4):
            torch.npu.synchronize()
        return output

    def injected_bias() -> torch.Tensor:
        return v_baseline() + 0.01

    for operation in (
        frontier_adapter,
        standalone,
        dispatched,
        k_baseline,
        v_baseline,
        copied,
        synchronized,
        injected_bias,
    ):
        for _ in range(WARMUP):
            operation()
        torch.npu.synchronize()

    variants = {
        "frontier_adapter": _device_samples(frontier_adapter),
        "standalone": _device_samples(standalone),
        "dispatch": _device_samples(dispatched),
        "k_baseline": _device_samples(k_baseline),
        "v_baseline": _device_samples(v_baseline),
        "copy": _device_samples(copied),
        "sync": _device_samples(synchronized),
        "profiling": _device_samples(synchronized, profiled=True),
        "negative_control": _device_samples(injected_bias),
    }
    torch.npu.synchronize()

    q_good = dispatched().unsqueeze(0)
    k_good = k_baseline().unsqueeze(0)
    v_good = v_baseline().unsqueeze(0)
    bad = v_good + 0.01
    torch.npu.synchronize()
    output = {
        "schema": "groundupscale.dev/ascend-diagnostic-session/v1alpha1",
        "session_id": args.session_id,
        "process_id": os.getpid(),
        "process_started_at": datetime.now(UTC).isoformat(),
        "cohort_id": "ascend-npu-23b93a89d5fecc79",
        "device": {
            "logical": "npu:0",
            "name": torch.npu.get_device_name(0),
        },
        "software": {
            "python": ".".join(str(part) for part in os.sys.version_info[:3]),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "execution_contract": {
            "semantic": "batch-one Q projection MatMul",
            "stable_path": Q_PATH,
            "shape": {
                "left": [1, SHAPE, SHAPE],
                "right": [SHAPE, SHAPE],
                "output": [1, SHAPE, SHAPE],
            },
            "dtype": "float32",
            "layout": "row-major-contiguous",
            "alignment_bytes": 512,
            "execution_mode": "pytorch-eager",
            "completion_boundary": (
                "device-event-end-synchronize-plus-device-synchronize"
            ),
            "warmup_iterations": WARMUP,
            "samples": SAMPLES,
            "inner_iterations": INNER_ITERATIONS,
            "operator_baseline_adapter": (
                "batch-one input/output are zero-copy views outside the timed "
                "2-D torch.matmul kernel"
            ),
            "cumulative_variant_order": [
                "frontier_adapter",
                "dispatch",
                "copy",
                "sync",
                "profiling",
            ],
            "variant_contracts": {
                "k_baseline": {
                    "semantic": "batch-one K projection MatMul",
                    "stable_path": K_PATH,
                    "lane": "baseline",
                    "input_identity": {
                        "left_sha256": _tensor_sha256(k_left_cpu),
                        "right_sha256": _tensor_sha256(k_right_cpu),
                    },
                },
                "v_baseline": {
                    "semantic": "batch-one V projection MatMul",
                    "stable_path": V_PATH,
                    "lane": "baseline",
                    "input_identity": {
                        "left_sha256": _tensor_sha256(v_left_cpu),
                        "right_sha256": _tensor_sha256(v_right_cpu),
                    },
                },
                "negative_control": {
                    "semantic": (
                        "batch-one V projection MatMul negative control"
                    ),
                    "stable_path": V_PATH,
                    "lane": "diagnostic",
                    "input_identity": {
                        "left_sha256": _tensor_sha256(v_left_cpu),
                        "right_sha256": _tensor_sha256(v_right_cpu),
                    },
                },
            },
        },
        "input": {
            "seed": SEED,
            "left_sha256": _tensor_sha256(left_cpu),
            "right_sha256": _tensor_sha256(right_cpu),
        },
        "path_inputs": {
            "q": {
                "left_sha256": _tensor_sha256(left_cpu),
                "right_sha256": _tensor_sha256(right_cpu),
            },
            "k": {
                "left_sha256": _tensor_sha256(k_left_cpu),
                "right_sha256": _tensor_sha256(k_right_cpu),
            },
            "v": {
                "left_sha256": _tensor_sha256(v_left_cpu),
                "right_sha256": _tensor_sha256(v_right_cpu),
            },
        },
        "timer": {
            "source": "torch.npu.Event.elapsed_time",
            "resolution_ns": 20.0,
            "unit": "nanoseconds",
        },
        "variants": {
            name: {
                "raw_samples_ns": samples,
                "median_ns": median(samples),
                "semantic": (
                    "batch-one K projection MatMul baseline"
                    if name == "k_baseline"
                    else "batch-one V projection MatMul baseline"
                    if name == "v_baseline"
                    else "batch-one V projection MatMul negative control"
                    if name == "negative_control"
                    else "cumulative Q projection 2-D Frontier-kernel wrapper stage"
                    if name in {
                        "frontier_adapter",
                        "dispatch",
                        "copy",
                        "sync",
                        "profiling",
                    }
                    else "2-D Frontier-kernel diagnostic control"
                ),
            }
            for name, samples in variants.items()
        },
        "correctness": _correctness(
            left_cpu,
            right_cpu,
            q_good,
            atol=0.001,
            rtol=0.001,
        ),
        "path_correctness": {
            "q": _correctness(
                left_cpu,
                right_cpu,
                q_good,
                atol=0.001,
                rtol=0.001,
            ),
            "k": _correctness(
                k_left_cpu,
                k_right_cpu,
                k_good,
                atol=0.001,
                rtol=0.001,
            ),
            "v": _correctness(
                v_left_cpu,
                v_right_cpu,
                v_good,
                atol=0.001,
                rtol=0.001,
            ),
        },
        "negative_control": {
            "kind": "injected-output-bias",
            "injected_bias": 0.01,
            "correctness": _correctness(
                v_left_cpu,
                v_right_cpu,
                bad,
                atol=0.001,
                rtol=0.001,
            ),
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
