#!/usr/bin/env python3
"""Collect one independent Ascend 910B2 diagnostic-ablation session.

The script intentionally emits raw measurements only.  Bundle construction and
Verdict evaluation happen later on the Mac through GroundUpScale's normal
digest-verifying diagnostic seam.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median

import torch
import torch_npu


SHAPE = 512
SEED = 20260811
WARMUP = 100
SAMPLES = 20
INNER_ITERATIONS = 100


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
    expected = torch.matmul(left_cpu, right_cpu)
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
    left = left_cpu.to("npu:0")
    right = right_cpu.to("npu:0")
    left_batched = left.unsqueeze(0)

    def standalone() -> torch.Tensor:
        return torch.matmul(left, right)

    def dispatched() -> torch.Tensor:
        return torch.matmul(left_batched, right)

    def copied() -> torch.Tensor:
        return dispatched().clone().clone()

    def synchronized() -> torch.Tensor:
        output = copied()
        for _ in range(4):
            torch.npu.synchronize()
        return output

    def injected_bias() -> torch.Tensor:
        return standalone() + 0.01

    for operation in (
        standalone,
        dispatched,
        copied,
        synchronized,
        injected_bias,
    ):
        for _ in range(WARMUP):
            operation()
        torch.npu.synchronize()

    variants = {
        "standalone": _device_samples(standalone),
        "dispatch": _device_samples(dispatched),
        "copy": _device_samples(copied),
        "sync": _device_samples(synchronized),
        "profiling": _device_samples(synchronized, profiled=True),
        "negative_control": _device_samples(injected_bias),
    }
    torch.npu.synchronize()

    good = standalone()
    bad = good + 0.01
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
        },
        "input": {
            "seed": SEED,
            "left_sha256": _tensor_sha256(left_cpu),
            "right_sha256": _tensor_sha256(right_cpu),
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
            }
            for name, samples in variants.items()
        },
        "correctness": _correctness(
            left_cpu,
            right_cpu,
            good,
            atol=0.001,
            rtol=0.001,
        ),
        "negative_control": {
            "kind": "injected-output-bias",
            "injected_bias": 0.01,
            "correctness": _correctness(
                left_cpu,
                right_cpu,
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
