from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

import groundupscale.environment as environment
from groundupscale.environment import (
    EnvironmentValidityPolicy,
    collect_environment_validity,
    evaluate_environment_validity,
)
from groundupscale.cli import main


REPOSITORY_ROOT = Path(__file__).parents[1]


def _nominal_observations() -> dict[str, object]:
    return {
        "platform": {
            "system": "Darwin",
            "machine": "arm64",
            "logical_cpu_count": 10,
        },
        "power": {"source": "ac", "battery_percent": 100.0},
        "thermal": {
            "status": "nominal",
            "thermal_warning": False,
            "performance_warning": False,
        },
        "load": {
            "one_minute": 0.8,
            "five_minutes": 0.9,
            "fifteen_minutes": 1.0,
        },
        "competitors": {
            "sample_interval_seconds": 1.0,
            "sample_count": 3,
            "total_cpu_percent_samples": [9.5, 8.0, 7.0],
            "top": [
                {"pid": 7, "name": "WindowServer", "cpu_percent": 9.5},
            ],
        },
    }


def test_nominal_mac_environment_is_eligible_and_explains_every_check() -> None:
    report = evaluate_environment_validity(_nominal_observations())

    assert report["schema"] == "groundupscale.dev/environment-validity/v1alpha1"
    assert report["eligible"] is True
    assert report["reason_codes"] == []
    assert {check["check_id"] for check in report["checks"]} == {
        "trusted-platform",
        "ac-power",
        "thermal-nominal",
        "settled-load",
        "bounded-total-competing-cpu",
    }
    assert all(check["passed"] for check in report["checks"])
    assert report["observations"]["load"]["normalized_one_minute"] == 0.08
    assert report["policy"]["policy_id"] == "local-apple-silicon-v2"


def test_short_single_process_burst_with_low_total_machine_contention_is_eligible() -> None:
    observations = _nominal_observations()
    observations["competitors"] = {
        "sample_interval_seconds": 1.0,
        "sample_count": 3,
        "total_cpu_percent_samples": [75.0, 80.0, 70.0],
        "top": [
            {
                "pid": 501,
                "name": "Code Helper (Renderer)",
                "cpu_percent": 50.3,
            },
        ],
    }

    report = evaluate_environment_validity(observations)

    assert report["eligible"] is True
    assert report["policy"]["policy_id"] == "local-apple-silicon-v2"
    check = next(
        item
        for item in report["checks"]
        if item["check_id"] == "bounded-total-competing-cpu"
    )
    assert check["passed"] is True
    assert check["observed"] == 0.08
    assert report["observations"]["competitors"][
        "maximum_single_process_cpu_percent"
    ] == 50.3


def test_competing_process_and_unsettled_load_make_environment_ineligible() -> None:
    observations = _nominal_observations()
    observations["load"] = {
        "one_minute": 3.1,
        "five_minutes": 2.0,
        "fifteen_minutes": 1.0,
    }
    observations["competitors"] = {
        "sample_interval_seconds": 1.0,
        "sample_count": 3,
        "total_cpu_percent_samples": [160.0, 175.0, 155.0],
        "top": [
            {"pid": 101, "name": "python", "cpu_percent": 84.5},
            {"pid": 102, "name": "mediaanalysisd", "cpu_percent": 75.9},
        ],
    }

    report = evaluate_environment_validity(observations)

    assert report["eligible"] is False
    assert report["reason_codes"] == [
        "load-above-policy",
        "total-competing-cpu-above-policy",
    ]
    failed = {check["check_id"]: check for check in report["checks"] if not check["passed"]}
    assert failed["settled-load"]["observed"] == 0.31
    assert failed["bounded-total-competing-cpu"]["observed"] == 0.175


def test_missing_total_competitor_samples_fail_closed() -> None:
    observations = _nominal_observations()
    observations["competitors"] = {
        "sample_interval_seconds": 1.0,
        "sample_count": 3,
        "top": [{"pid": 7, "name": "WindowServer", "cpu_percent": 9.5}],
    }

    report = evaluate_environment_validity(observations)

    assert report["eligible"] is False
    assert report["reason_codes"] == ["competitor-load-unverified"]


def test_public_collector_records_each_intervals_total_competing_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCoordinator:
        pid = 999

        def parents(self) -> list[object]:
            return []

    class FakeProcess:
        def __init__(self, pid: int, name: str, samples: list[float]) -> None:
            self.pid = pid
            self.info = {"pid": pid, "name": name}
            self._samples = iter([0.0, *samples])

        def cpu_percent(self, _: object) -> float:
            return next(self._samples)

    processes = [
        FakeProcess(101, "worker-a", [40.0, 20.0, 10.0]),
        FakeProcess(102, "worker-b", [30.0, 50.0, 20.0]),
    ]
    monkeypatch.setattr(environment.psutil, "Process", lambda: FakeCoordinator())
    monkeypatch.setattr(
        environment.psutil,
        "process_iter",
        lambda _: iter(processes),
    )
    monkeypatch.setattr(environment.time, "sleep", lambda _: None)
    monkeypatch.setattr(environment.os, "cpu_count", lambda: 10)
    monkeypatch.setattr(environment.os, "getloadavg", lambda: (0.8, 0.9, 1.0))
    monkeypatch.setattr(environment.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(environment.platform, "release", lambda: "test-release")
    monkeypatch.setattr(environment.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        environment,
        "_pmset_output",
        lambda mode: (
            "Now drawing from 'AC Power'\n 100%"
            if mode == "batt"
            else "No thermal warning level has been recorded\n"
            "No performance warning level has been recorded\n"
        ),
    )

    report = collect_environment_validity(
        sample_interval_seconds=0.001,
        process_sample_count=3,
    )

    assert report["eligible"] is True
    assert report["observations"]["competitors"][
        "total_cpu_percent_samples"
    ] == [70.0, 70.0, 30.0]
    assert report["observations"]["competitors"][
        "normalized_maximum_total_cpu"
    ] == 0.07


def test_unknown_required_signal_is_never_assumed_valid() -> None:
    observations = _nominal_observations()
    observations["power"] = {"source": "unknown", "battery_percent": None}
    observations["thermal"] = {
        "status": "unknown",
        "thermal_warning": None,
        "performance_warning": None,
    }

    report = evaluate_environment_validity(observations)

    assert report["eligible"] is False
    assert report["reason_codes"] == ["ac-power-unverified", "thermal-unverified"]


def test_policy_thresholds_are_explicit_and_versioned() -> None:
    policy = EnvironmentValidityPolicy(
        maximum_normalized_one_minute_load=0.5,
        maximum_normalized_total_competing_cpu=0.2,
    )
    observations = _nominal_observations()
    observations["load"] = {
        "one_minute": 4.0,
        "five_minutes": 4.0,
        "fifteen_minutes": 4.0,
    }
    observations["competitors"] = {
        "sample_interval_seconds": 1.0,
        "sample_count": 3,
        "total_cpu_percent_samples": [145.0, 150.0, 140.0],
        "top": [{"pid": 7, "name": "worker", "cpu_percent": 45.0}],
    }

    report = evaluate_environment_validity(observations, policy=policy)

    assert report["eligible"] is True
    assert report["policy"]["maximum_normalized_one_minute_load"] == 0.5
    assert report["policy"]["maximum_normalized_total_competing_cpu"] == 0.2


def test_preflight_cli_always_emits_a_machine_readable_decision() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "preflight",
            "--sample-interval-seconds",
            "0.01",
            "--json",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode in {0, 2}, completed.stderr
    report = json.loads(completed.stdout)
    assert report["schema"] == "groundupscale.dev/environment-validity/v1alpha1"
    assert report["eligible"] is (completed.returncode == 0)
    assert report["policy"]["allowlist_only"] is True


def test_trusted_run_rejection_is_machine_readable_and_publishes_no_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = evaluate_environment_validity(_nominal_observations())
    invalid["eligible"] = False
    invalid["reason_codes"] = ["total-competing-cpu-above-policy"]

    exit_code = main(
        [
            "run",
            str(REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--artifact-store",
            str(tmp_path),
            "--run-id",
            "rejected-before-benchmark",
            "--require-valid-environment",
            "--json",
        ],
        environment_collector=lambda **_: invalid,
    )

    assert exit_code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["schema"] == "groundupscale.dev/run-rejection/v1alpha1"
    assert output["status"] == "rejected-before-benchmark"
    assert output["reason_codes"] == ["total-competing-cpu-above-policy"]
    assert output["environment_validity"]["eligible"] is False
    assert not (tmp_path / "runs/rejected-before-benchmark").exists()
