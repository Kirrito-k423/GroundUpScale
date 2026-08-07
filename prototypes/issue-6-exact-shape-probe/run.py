#!/usr/bin/env python3.11
"""PROTOTYPE ONLY: collect and replay exact-Shape MatMul verdict evidence."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path(__file__).resolve().parent / "results" / "raw-results.json"
SOURCE_OBSERVATION = (
    ROOT
    / "goal_process"
    / "mac-transformer-ir-calibration-slice"
    / "evidence"
    / "apple-m4-cpu-microbenchmark-observation-v2.json"
)


def _ensure_project_python() -> None:
    if importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"):
        return
    completed = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    common_git = Path(completed.stdout.strip())
    if not common_git.is_absolute():
        common_git = (ROOT / common_git).resolve()
    project_python = common_git.parent / ".venv" / "bin" / "python"
    if not project_python.exists():
        raise RuntimeError(
            "project dependencies are unavailable and the repository .venv was not found"
        )
    os.execv(
        str(project_python),
        [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]],
    )


_ensure_project_python()

# The execution contract is fixed before importing either BLAS wrapper.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from groundupscale.environment import collect_environment_validity  # noqa: E402

from decision import classify_evidence  # noqa: E402


ANOMALY_N = 257
ALIGNED_N = 256
PAD_SIZES = (264, 272, 288)
THREADS = 1
WARMUPS = 10
WINDOWS = 12
TARGET_WINDOW_NS = 10_000_000
MAX_INNER_ITERATIONS = 1_000
SESSIONS = 3
RTOL = 1e-4
ATOL = 1e-4
SEED = 20260807


def _summary(per_call_ns: list[float], raw_windows_ns: list[int], inner: int) -> dict[str, Any]:
    q1, _, q3 = statistics.quantiles(per_call_ns, n=4, method="inclusive")
    median_ns = float(statistics.median(per_call_ns))
    return {
        "raw_window_ns": raw_windows_ns,
        "per_call_ns": per_call_ns,
        "inner_iterations": inner,
        "median_ns": median_ns,
        "q1_ns": float(q1),
        "q3_ns": float(q3),
        "iqr_ns": float(q3 - q1),
        "iqr_over_median": float((q3 - q1) / median_ns),
    }


def _layout(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "contiguous": value.is_contiguous(),
            "stride": list(value.stride()),
        }
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "contiguous": bool(array.flags.c_contiguous),
        "strides_bytes": list(array.strides),
    }


def _correctness(output: Any, reference: np.ndarray) -> dict[str, Any]:
    observed = (
        output.detach().cpu().numpy() if isinstance(output, torch.Tensor) else np.asarray(output)
    )
    delta = np.abs(observed.astype(np.float64) - reference)
    denominator = np.maximum(np.abs(reference), ATOL)
    return {
        "oracle": "numpy-fp64-matmul",
        "rtol": RTOL,
        "atol": ATOL,
        "passed": bool(np.allclose(observed, reference, rtol=RTOL, atol=ATOL)),
        "max_abs_error": float(delta.max()),
        "max_relative_error_with_atol_floor": float((delta / denominator).max()),
        "output_layout": _layout(output),
    }


def _measure_interleaved(
    invocations: dict[str, Callable[[], Any]], seed: int
) -> tuple[dict[str, dict[str, Any]], list[list[str]]]:
    inner_iterations: dict[str, int] = {}
    for key, invoke in invocations.items():
        for _ in range(WARMUPS):
            invoke()
        started = time.perf_counter_ns()
        for _ in range(10):
            invoke()
        pilot_per_call = max(1.0, (time.perf_counter_ns() - started) / 10)
        inner_iterations[key] = max(
            1,
            min(MAX_INNER_ITERATIONS, math.ceil(TARGET_WINDOW_NS / pilot_per_call)),
        )

    rng = random.Random(seed)
    raw: dict[str, list[int]] = {key: [] for key in invocations}
    orders: list[list[str]] = []
    keys = list(invocations)
    for _ in range(WINDOWS):
        order = keys.copy()
        rng.shuffle(order)
        orders.append(order)
        for key in order:
            invoke = invocations[key]
            started = time.perf_counter_ns()
            for _ in range(inner_iterations[key]):
                invoke()
            raw[key].append(time.perf_counter_ns() - started)

    return (
        {
            key: _summary(
                [window / inner_iterations[key] for window in raw[key]],
                raw[key],
                inner_iterations[key],
            )
            for key in keys
        },
        orders,
    )


def _anomaly_measurements(session_id: int) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    left_np = rng.standard_normal((ANOMALY_N, ANOMALY_N), dtype=np.float32)
    right_np = rng.standard_normal((ANOMALY_N, ANOMALY_N), dtype=np.float32)
    reference = left_np.astype(np.float64) @ right_np.astype(np.float64)
    left = torch.from_numpy(left_np)
    right = torch.from_numpy(right_np)

    direct_out = torch.empty((ANOMALY_N, ANOMALY_N), dtype=torch.float32)
    numpy_out = np.empty((ANOMALY_N, ANOMALY_N), dtype=np.float32)

    def torch_direct() -> torch.Tensor:
        return torch.mm(left, right, out=direct_out)

    def numpy_direct() -> np.ndarray:
        np.matmul(left_np, right_np, out=numpy_out)
        return numpy_out

    invocations: dict[str, Callable[[], Any]] = {
        "torch-direct": torch_direct,
        "numpy-direct": numpy_direct,
    }
    outputs: dict[str, Any] = {
        "torch-direct": direct_out,
        "numpy-direct": numpy_out,
    }

    for padded_n in PAD_SIZES:
        padded_left = torch.zeros((padded_n, padded_n), dtype=torch.float32)
        padded_right = torch.zeros((padded_n, padded_n), dtype=torch.float32)
        padded_result = torch.empty((padded_n, padded_n), dtype=torch.float32)
        exact_result = torch.empty((ANOMALY_N, ANOMALY_N), dtype=torch.float32)

        def padded(
            padded_left: torch.Tensor = padded_left,
            padded_right: torch.Tensor = padded_right,
            padded_result: torch.Tensor = padded_result,
            exact_result: torch.Tensor = exact_result,
        ) -> torch.Tensor:
            padded_left[:ANOMALY_N, :ANOMALY_N].copy_(left)
            padded_right[:ANOMALY_N, :ANOMALY_N].copy_(right)
            torch.mm(padded_left, padded_right, out=padded_result)
            exact_result.copy_(padded_result[:ANOMALY_N, :ANOMALY_N])
            return exact_result

        key = f"torch-pad-{padded_n}-slice-copy"
        invocations[key] = padded
        outputs[key] = exact_result

    truncated_core = torch.empty((ALIGNED_N, ALIGNED_N), dtype=torch.float32)
    truncated_out = torch.zeros((ANOMALY_N, ANOMALY_N), dtype=torch.float32)

    def truncated_negative_control() -> torch.Tensor:
        torch.mm(
            left[:ALIGNED_N, :ALIGNED_N],
            right[:ALIGNED_N, :ALIGNED_N],
            out=truncated_core,
        )
        truncated_out[:ALIGNED_N, :ALIGNED_N].copy_(truncated_core)
        return truncated_out

    invocations["truncated-256-negative-control"] = truncated_negative_control
    outputs["truncated-256-negative-control"] = truncated_out

    correctness: dict[str, dict[str, Any]] = {}
    for key, invoke in invocations.items():
        output = invoke()
        correctness[key] = _correctness(output, reference)

    summaries, orders = _measure_interleaved(invocations, SEED + session_id)
    return {
        "input": {
            "semantic": "C = A @ B",
            "shape_mkn": [ANOMALY_N, ANOMALY_N, ANOMALY_N],
            "dtype": "float32",
            "layout": "C-contiguous inputs and output",
            "threads": THREADS,
            "seed": SEED,
            "left_sha256": sha256(left_np.tobytes()).hexdigest(),
            "right_sha256": sha256(right_np.tobytes()).hexdigest(),
        },
        "measurement_contract": {
            "timer": "time.perf_counter_ns",
            "completion_boundary": "CPU call return; synchronous completion",
            "warmup_iterations": WARMUPS,
            "windows": WINDOWS,
            "target_window_ns": TARGET_WINDOW_NS,
            "instrumentation": "baseline timing lane; no profiler in timed region",
            "candidate_order": orders,
        },
        "measurements": {
            key: {
                "correctness": correctness[key],
                "summary": summaries[key],
            }
            for key in invocations
        },
    }


def _integration_measurements(session_id: int) -> dict[str, Any]:
    rng = np.random.default_rng(SEED + 1)
    left_np = rng.standard_normal((ALIGNED_N, ALIGNED_N), dtype=np.float32)
    right_np = rng.standard_normal((ALIGNED_N, ALIGNED_N), dtype=np.float32)
    reference = left_np.astype(np.float64) @ right_np.astype(np.float64)
    left = torch.from_numpy(left_np)
    right = torch.from_numpy(right_np)
    operator_out = torch.empty((ALIGNED_N, ALIGNED_N), dtype=torch.float32)
    scratch = torch.empty_like(operator_out)
    final_out = torch.empty_like(operator_out)

    def operator() -> torch.Tensor:
        return torch.mm(left, right, out=operator_out)

    operator()

    def copy_twice() -> torch.Tensor:
        scratch.copy_(operator_out)
        final_out.copy_(scratch)
        return final_out

    def operator_plus_copy_twice() -> torch.Tensor:
        torch.mm(left, right, out=operator_out)
        scratch.copy_(operator_out)
        final_out.copy_(scratch)
        return final_out

    invocations = {
        "operator": operator,
        "copy-twice": copy_twice,
        "operator-plus-copy-twice": operator_plus_copy_twice,
    }
    correctness = {
        "operator": _correctness(operator(), reference),
        "copy-twice": {
            **_correctness(copy_twice(), reference),
            "oracle_interpretation": "copy ablation preserves a previously computed output",
        },
        "operator-plus-copy-twice": _correctness(operator_plus_copy_twice(), reference),
    }
    summaries, orders = _measure_interleaved(invocations, SEED + 100 + session_id)
    return {
        "input": {
            "semantic": "standalone C=A@B versus wrapper C=copy(copy(A@B))",
            "shape_mkn": [ALIGNED_N, ALIGNED_N, ALIGNED_N],
            "dtype": "float32",
            "layout": "C-contiguous inputs and output",
            "threads": THREADS,
            "seed": SEED + 1,
        },
        "measurement_contract": {
            "timer": "time.perf_counter_ns",
            "completion_boundary": "CPU call return; synchronous completion",
            "warmup_iterations": WARMUPS,
            "windows": WINDOWS,
            "target_window_ns": TARGET_WINDOW_NS,
            "instrumentation": "baseline timing lane; no profiler in timed region",
            "candidate_order": orders,
        },
        "measurements": {
            key: {
                "correctness": correctness[key],
                "summary": summaries[key],
            }
            for key in invocations
        },
    }


def _worker(session_id: int) -> dict[str, Any]:
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(THREADS)
    with torch.inference_mode():
        return {
            "session_id": session_id,
            "process_id": os.getpid(),
            "captured_at": datetime.now(UTC).isoformat(),
            "runtime": {
                "python": platform.python_version(),
                "executable": sys.executable,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "thread_environment": {
                    key: os.environ[key]
                    for key in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
            },
            "anomaly": _anomaly_measurements(session_id),
            "integration": _integration_measurements(session_id),
        }


def _old_reference() -> dict[str, Any]:
    raw = SOURCE_OBSERVATION.read_bytes()
    observation = json.loads(raw)
    cases = next(
        probe["cases"]
        for probe in observation["probes"]
        if probe["probe_id"] == "matrix-fp32-cube"
    )

    def selected(n: int) -> dict[str, Any]:
        return next(
            case
            for case in cases
            if case["shape"] == [n, n, n] and case["threads"] == THREADS
        )

    aligned = selected(ALIGNED_N)
    anomaly = selected(ANOMALY_N)
    return {
        "path": str(SOURCE_OBSERVATION.relative_to(ROOT)),
        "sha256": sha256(raw).hexdigest(),
        "hardware_cohort": observation["hardware_cohort"],
        "environment_eligible": observation["environment"]["eligible"],
        "environment_reason_codes": observation["environment"]["reason_codes"],
        "aligned_256": aligned,
        "anomaly_257": anomaly,
        "rate_drop_fraction": 1.0 - anomaly["achieved_rate"] / aligned["achieved_rate"],
    }


def _blas_identity() -> dict[str, Any]:
    numpy_config = np.__config__.show(mode="dicts")
    torch_config = torch.__config__.show()
    return {
        "numpy_blas": numpy_config["Build Dependencies"]["blas"],
        "torch_blas_info": "accelerate" if "BLAS_INFO=accelerate" in torch_config else "unknown",
        "coverage_conclusion": "C1_MULTIPLE_WRAPPERS_SINGLE_ACCELERATE_LIBRARY",
    }


def _candidate_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "torch-direct": {
            "role": "target",
            "provider": "PyTorch",
            "library": "Accelerate",
            "timed_work": "preallocated torch.mm",
        },
        "numpy-direct": {
            "role": "alternative",
            "provider": "NumPy",
            "library": "Accelerate",
            "timed_work": "preallocated numpy.matmul",
        },
        "truncated-256-negative-control": {
            "role": "negative_control",
            "provider": "PyTorch",
            "library": "Accelerate",
            "timed_work": "incorrect truncated 256-cube plus partial output copy",
        },
    }
    for padded_n in PAD_SIZES:
        manifest[f"torch-pad-{padded_n}-slice-copy"] = {
            "role": "alternative",
            "provider": "PyTorch",
            "library": "Accelerate",
            "timed_work": (
                f"copy exact inputs into zero-padded {padded_n}-cube, torch.mm, "
                "slice-copy to exact contiguous output"
            ),
        }
    return manifest


def _run_parent() -> dict[str, Any]:
    old = _old_reference()
    environment = collect_environment_validity(
        sample_interval_seconds=0.2,
        process_sample_count=3,
    )
    sessions: list[dict[str, Any]] = []
    for session_id in range(1, SESSIONS + 1):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker-session", str(session_id)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        sessions.append(json.loads(completed.stdout))

    evidence: dict[str, Any] = {
        "schema": "groundupscale.dev/throwaway-exact-shape-probe/v0",
        "prototype_status": "THROWAWAY — MUST NOT BECOME PRODUCTION RUNNER",
        "captured_at": datetime.now(UTC).isoformat(),
        "run_command": "python3.11 prototypes/issue-6-exact-shape-probe/run.py --batch",
        "input_source": old,
        "environment": environment,
        "blas_identity": _blas_identity(),
        "candidate_manifest": _candidate_manifest(),
        "protocol": {
            "anomaly_shape_mkn": [ANOMALY_N, ANOMALY_N, ANOMALY_N],
            "aligned_control_shape_mkn": [ALIGNED_N, ALIGNED_N, ALIGNED_N],
            "dtype": "float32",
            "layout": "C-contiguous",
            "threads": THREADS,
            "sessions": SESSIONS,
            "warmups": WARMUPS,
            "windows": WINDOWS,
            "correctness": {"oracle": "numpy-fp64-matmul", "rtol": RTOL, "atol": ATOL},
            "old_256_reference_rate_flops_per_s": old["aligned_256"]["achieved_rate"],
            "anomaly_work_flops": 2 * ANOMALY_N**3,
            "aligned_work_flops": 2 * ALIGNED_N**3,
            "locked_thresholds": {
                "headroom_minimum_fraction": 0.05,
                "headroom_recovery_of_old_reference": 0.90,
                "operator_reference_tolerance_fraction": 0.10,
                "integration_minimum_gap_fraction": 0.10,
                "copy_ablation_relative_error_fraction": 0.35,
            },
        },
        "sessions": sessions,
    }
    evidence["decision"] = classify_evidence(evidence)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def _render_batch(evidence: dict[str, Any]) -> None:
    print("PROTOTYPE — exact-Shape decision evidence")
    print(json.dumps({"environment": evidence["environment"]}, indent=2, ensure_ascii=False))
    for scenario in evidence["decision"]["scenarios"]:
        print("\nSCENARIO STATE")
        print(json.dumps(scenario, indent=2, sort_keys=True, ensure_ascii=False))
    print("\nASSERTIONS")
    print(json.dumps(evidence["decision"]["assertions"], indent=2, sort_keys=True))
    print(f"\nraw evidence: {RESULT_PATH}")
    print(f"exit criteria: {evidence['decision']['exit_criteria_passed']}")


def _interactive() -> int:
    evidence = json.loads(RESULT_PATH.read_text()) if RESULT_PATH.exists() else None
    index = 0
    while True:
        print("\033[2J\033[H", end="")
        print("\033[1mPROTOTYPE — Exact-Shape Probe Evidence\033[0m")
        if evidence is None:
            print("\n\033[2mNo captured run yet. Press r to run the locked protocol.\033[0m")
        else:
            scenarios = evidence["decision"]["scenarios"]
            scenario = scenarios[index % len(scenarios)]
            print(json.dumps(scenario, indent=2, sort_keys=True, ensure_ascii=False))
            print("\n\033[2mAssertions\033[0m")
            print(json.dumps(evidence["decision"]["assertions"], indent=2, sort_keys=True))
        print("\n\033[1m[n]\033[0m next  \033[1m[r]\033[0m rerun  \033[1m[q]\033[0m quit")
        command = input("> ").strip().lower()
        if command == "q":
            return 0
        if command == "r":
            evidence = _run_parent()
            index = 0
        elif command == "n" and evidence is not None:
            index += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--worker-session", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_session is not None:
        print(json.dumps(_worker(args.worker_session), ensure_ascii=False))
        return 0
    if args.batch:
        evidence = _run_parent()
        _render_batch(evidence)
        return 0 if evidence["decision"]["exit_criteria_passed"] else 1
    return _interactive()


if __name__ == "__main__":
    raise SystemExit(main())
