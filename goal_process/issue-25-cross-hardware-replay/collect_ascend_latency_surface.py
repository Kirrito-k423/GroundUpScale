#!/usr/bin/env python3
"""Collect one real Ascend 910B2 fixed-N/K latency-surface session."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import median

import torch
import torch_npu


N = 512
K = 512
SEED = 20260812
WARMUP = 100
SAMPLES = 100
INNER_ITERATIONS = 100
COHORT_ID = "ascend-npu-23b93a89d5fecc79"
Q_PATH = (
    "semantic/model/two-layer-transformer/transformer/"
    "layer-0/attention/q-proj"
)


def _tensor_sha256(value: torch.Tensor) -> str:
    return sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--lane", choices=("search", "holdout", "confirmation"), required=True)
    parser.add_argument("--m", type=int, choices=(256, 384, 512), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0":
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES must be exactly 0")

    torch.manual_seed(SEED + args.m)
    left_cpu = torch.randn((args.m, K), dtype=torch.float32)
    right_cpu = torch.randn((K, N), dtype=torch.float32)
    expected = torch.matmul(left_cpu.double(), right_cpu.double()).float()
    left = left_cpu.to("npu:0")
    right = right_cpu.to("npu:0")

    def operation() -> torch.Tensor:
        return torch.matmul(left, right)

    for _ in range(WARMUP):
        operation()
    torch.npu.synchronize()

    raw_samples_ns: list[float] = []
    for _ in range(SAMPLES):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(INNER_ITERATIONS):
            operation()
        end.record()
        end.synchronize()
        torch.npu.synchronize()
        raw_samples_ns.append(
            float(start.elapsed_time(end)) * 1_000_000.0 / INNER_ITERATIONS
        )

    observed = operation().detach().cpu()
    difference = (observed - expected).abs()
    tolerance = 0.001 + 0.001 * expected.abs()
    correctness = {
        "passed": bool(torch.all(difference <= tolerance).item()),
        "atol": 0.001,
        "rtol": 0.001,
        "expected_sha256": _tensor_sha256(expected),
        "observed_sha256": _tensor_sha256(observed),
        "max_abs_difference": float(difference.max().item()),
        "mismatched_elements": int((difference > tolerance).sum().item()),
    }
    if not correctness["passed"]:
        raise RuntimeError("correctness failed")

    document = {
        "schema": "groundupscale.dev/ascend-latency-surface-session/v1alpha1",
        "session_id": args.session_id,
        "process_id": os.getpid(),
        "process_started_at": datetime.now(UTC).isoformat(),
        "lane": args.lane,
        "cohort_id": COHORT_ID,
        "device": {"logical": "npu:0", "name": torch.npu.get_device_name(0)},
        "software": {
            "python": ".".join(str(part) for part in os.sys.version_info[:3]),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        },
        "execution_contract": {
            "semantic": "batch-one Q projection MatMul",
            "stable_path": Q_PATH,
            "shape": {"m": args.m, "n": N, "k": K},
            "dtype": "float32",
            "layout": "row-major-contiguous",
            "alignment_bytes": 512,
            "execution_mode": "pytorch-eager",
            "candidate_id": "torch.matmul",
            "candidate_family": "pytorch-ascend-matmul",
            "completion_boundary": "device-event-end-synchronize-plus-device-synchronize",
            "warmup_iterations": WARMUP,
            "samples": SAMPLES,
            "inner_iterations": INNER_ITERATIONS,
            "response_identity": "ascend-q-proj-device-event-duration-v1",
            "shape_regime_identity": "ascend-q-proj-fixed-nk-ramp-v1",
        },
        "input": {
            "seed": SEED + args.m,
            "left_sha256": _tensor_sha256(left_cpu),
            "right_sha256": _tensor_sha256(right_cpu),
        },
        "timer": {
            "source": "torch.npu.Event.elapsed_time",
            "resolution_ns": 20,
            "monotonic": True,
            "kind": "device-event",
        },
        "warmup": {"iterations": WARMUP, "converged": True},
        "raw_samples_ns": raw_samples_ns,
        "median_ns": float(median(raw_samples_ns)),
        "excluded_samples": [],
        "correctness": correctness,
    }
    output = Path(args.output)
    output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"session_id": args.session_id, "median_ns": document["median_ns"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
