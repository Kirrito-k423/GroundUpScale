from __future__ import annotations

import json
import subprocess
import sys


REQUIRED_OPERATIONS = {
    "matmul",
    "add",
    "rmsnorm",
    "softmax",
    "silu",
    "mul",
    "view",
    "transpose",
}


def test_probe_cli_emits_machine_readable_cpu_report() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "probe",
            "--device",
            "cpu",
            "--warmup",
            "1",
            "--repeats",
            "5",
            "--inner-iterations",
            "3",
            "--windows-per-sample",
            "2",
            "--matrix-size",
            "16",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)

    assert report["schema"] == "groundupscale.dev/environment-probe/v1alpha1"
    assert report["python"]["implementation"] == "CPython"
    assert report["torch"]["version"]
    assert report["torch"]["num_threads"] > 0
    assert report["torch"]["num_interop_threads"] > 0
    assert report["configuration"]["inner_iterations"] == 3
    assert report["configuration"]["windows_per_sample"] == 2
    assert report["mps"]["built"] in {True, False}
    assert report["mps"]["available"] in {True, False}

    cpu = report["devices"]["cpu"]
    assert cpu["available"] is True
    assert set(cpu["operations"]) == REQUIRED_OPERATIONS
    assert all(operation["status"] == "passed" for operation in cpu["operations"].values())
    assert len(cpu["latency_ns"]["samples"]) == 5
    assert len(cpu["latency_ns"]["window_samples"]) == 5
    assert all(len(windows) == 2 for windows in cpu["latency_ns"]["window_samples"])
    assert cpu["latency_ns"]["median"] > 0
    assert cpu["latency_ns"]["iqr_over_median"] >= 0
    assert cpu["memory"]["observer"] == "process_rss"
    assert cpu["memory"]["after_bytes"] > 0
