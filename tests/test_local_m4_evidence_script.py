from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/run-local-m4-evidence.sh"
REPORT_NAME = "trusted-hardware-ci-report.json"
REPORT_SCHEMA = "groundupscale.dev/trusted-hardware-ci-report/v1alpha1"
POLICIES = {
    "collection": "groundupscale.dev/trusted-hardware-ci-policy/v1alpha1",
    "noise": "groundupscale.dev/trusted-hardware-noise-check/v1alpha1",
    "promotion": "groundupscale.dev/trusted-hardware-promotion/v1alpha1",
    "retention": "groundupscale.dev/trusted-hardware-retention/v1alpha1",
}


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


@dataclass(frozen=True)
class LaneHarness:
    artifact_store: Path
    environment: dict[str, str]

    def invoke(
        self,
        tag: str,
        *,
        device: str = "cpu",
        **environment: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "--device",
                device,
                "--tag",
                tag,
                "--artifact-store",
                str(self.artifact_store),
            ],
            cwd=REPOSITORY_ROOT,
            env={**self.environment, **environment},
            text=True,
            capture_output=True,
            check=False,
        )

    def report_dir(self, tag: str) -> Path:
        return self.artifact_store / "trusted-hardware-ci" / tag

    def report(self, tag: str) -> dict[str, object]:
        return json.loads(
            (self.report_dir(tag) / REPORT_NAME).read_text(encoding="utf-8")
        )

    def bundle(self, tag: str) -> Path:
        return self.artifact_store / "runs" / f"{tag}-cpu"


@pytest.fixture
def lane(tmp_path: Path) -> LaneHarness:
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    _write_executable(
        binary_dir / "uname",
        """#!/bin/sh
case "$1" in
  -s) printf '%s\n' "${FAKE_UNAME_SYSTEM:-Darwin}" ;;
  -m) printf '%s\n' "${FAKE_UNAME_MACHINE:-arm64}" ;;
  *) exit 64 ;;
esac
""",
    )
    _write_executable(
        binary_dir / "uv",
        """#!/bin/sh
if [ "$1" = "run" ] && [ "$2" = "python" ] \
  && [ "${FAKE_MPS_UNAVAILABLE:-}" = "1" ]; then
  printf '%s\n' 'MPS unavailable' >&2
  exit 8
fi
if [ "$1" = "run" ] && [ "$2" = "pytest" ] \
  && [ -n "${FAKE_LOAD_STATE:-}" ]; then
  : > "$FAKE_LOAD_STATE"
fi
if [ "$1" = "run" ] && [ "$2" = "groundupscale" ] \
  && [ "$3" = "preflight" ]; then
  if [ "${FAKE_UV_FAILURE:-}" = "preflight" ] \
    || { [ -n "${FAKE_LOAD_STATE:-}" ] && [ -f "$FAKE_LOAD_STATE" ]; }; then
    printf '%s\n' '{"reason_codes":["load-above-policy"]}'
    exit 2
  fi
fi
if [ "$1" = "run" ] && [ "$2" = "groundupscale" ] && [ "$3" = "run" ]; then
  if [ "${FAKE_UV_FAILURE:-}" = "collection" ]; then
    printf '%s\n' 'simulated collection failure' >&2
    exit 7
  fi
  if [ "${FAKE_UV_FAILURE:-}" = "collection-exit-2" ]; then
    printf '%s\n' 'simulated non-environment collection failure' >&2
    exit 2
  fi
  if [ "${FAKE_UV_FAILURE:-}" = "run_environment" ]; then
    printf '%s\n' '{"reason_codes":["total-competing-cpu-above-policy"]}'
    exit 2
  fi
  shift 3
  artifact_store=
  run_id=
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --artifact-store) artifact_store="$2"; shift 2 ;;
      --run-id) run_id="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  benchmark_path="${FAKE_BENCHMARK_PATH:-observation/raw/benchmark.json}"
  benchmark_role="${FAKE_BENCHMARK_ROLE:-benchmark-observation}"
  run_root="$artifact_store/runs/$run_id"
  mkdir -p "$run_root/reports" "$run_root/$(dirname "$benchmark_path")"
  printf '{"run_id":"%s","status":"completed","artifacts":[{"role":"%s","path":"%s"}]}\n' \
    "$run_id" "$benchmark_role" "$benchmark_path" > "$run_root/run.manifest.json"
  if [ "${FAKE_BENCHMARK_INVALID:-}" = "1" ]; then
    printf '%s\n' '{"cases":[{"case_id":"matmul"}]}' > "$run_root/$benchmark_path"
  else
    printf '{"cases":[{"case_id":"matmul","latency":{"iqr_over_median":%s}}]}\n' \
      "${FAKE_BENCHMARK_NOISE:-0.01}" > "$run_root/$benchmark_path"
  fi
  printf '<html>report</html>\n' > "$run_root/reports/report.html"
fi
if [ "$1" = "run" ] && [ "$2" = "groundupscale" ] \
  && [ "$3" = "verify-run" ] \
  && [ "${FAKE_UV_FAILURE:-}" = "verification" ]; then
  printf '%s\n' 'simulated verification failure' >&2
  exit 9
fi
printf '{"passed":true}\n'
""",
    )
    return LaneHarness(
        artifact_store=tmp_path / "artifacts",
        environment={
            **os.environ,
            "PATH": f"{binary_dir}:{os.environ['PATH']}",
        },
    )


def test_cpu_only_lane_collects_verified_bundle_and_writes_versioned_report(
    lane: LaneHarness,
) -> None:
    completed = lane.invoke("slice-001", FAKE_MPS_UNAVAILABLE="1")
    manifest = (lane.bundle("slice-001") / "run.manifest.json").read_bytes()

    assert completed.returncode == 0, completed.stderr
    assert lane.report("slice-001") == {
        "schema": REPORT_SCHEMA,
        "run_tag": "slice-001",
        "device": "cpu",
        "status": "evidence_collected",
        "promotion_allowed": False,
        "reason_codes": [],
        "run_bundle": {
            "run_id": "slice-001-cpu",
            "manifest_sha256": sha256(manifest).hexdigest(),
            "artifact_ref": "artifact://run-bundle/run.manifest.json",
        },
        "run_manifest": {
            "run_id": "slice-001-cpu",
            "manifest_sha256": sha256(manifest).hexdigest(),
            "artifact_count": 1,
            "artifact_roles": ["benchmark-observation"],
        },
        "failure_evidence": [],
        "previous_qualified_evidence": [],
        "policies": POLICIES,
    }
    assert sorted(path.name for path in lane.bundle("slice-001").parent.iterdir()) == [
        "slice-001-cpu"
    ]
    assert not [
        path.name
        for path in lane.report_dir("slice-001").iterdir()
        if "failure" in path.name
    ]


def test_collection_failure_is_quarantined_with_replayable_failure_evidence(
    lane: LaneHarness,
) -> None:
    completed = lane.invoke("collection-001", FAKE_UV_FAILURE="collection")

    assert completed.returncode == 1
    evidence = lane.report_dir("collection-001") / "collection.log"
    assert lane.report("collection-001") == {
        "schema": REPORT_SCHEMA,
        "run_tag": "collection-001",
        "device": "cpu",
        "status": "quarantined",
        "promotion_allowed": False,
        "reason_codes": ["collection-failed"],
        "run_bundle": None,
        "run_manifest": None,
        "failure_evidence": ["artifact://attempt-evidence/collection.log"],
        "previous_qualified_evidence": [],
        "policies": POLICIES,
    }
    assert "simulated collection failure" in evidence.read_text(encoding="utf-8")


def test_unavailable_hardware_preserves_and_references_previous_evidence(
    lane: LaneHarness,
) -> None:
    assert lane.invoke("qualified-001").returncode == 0
    previous_manifest = (lane.bundle("qualified-001") / "run.manifest.json").read_bytes()

    completed = lane.invoke(
        "unavailable-001",
        FAKE_UNAME_SYSTEM="Linux",
        FAKE_UNAME_MACHINE="x86_64",
    )

    assert completed.returncode == 2
    receipt = lane.report_dir("unavailable-001") / "availability-failure.log"
    assert lane.report("unavailable-001") == {
        "schema": REPORT_SCHEMA,
        "run_tag": "unavailable-001",
        "device": "cpu",
        "status": "hardware_unavailable",
        "promotion_allowed": False,
        "reason_codes": ["unsupported-platform"],
        "run_bundle": None,
        "run_manifest": None,
        "failure_evidence": ["artifact://attempt-evidence/availability-failure.log"],
        "previous_qualified_evidence": [
            {
                "artifact_ref": "artifact://run-bundle/run.manifest.json",
                "run_id": "qualified-001-cpu",
                "manifest_sha256": sha256(previous_manifest).hexdigest(),
            }
        ],
        "policies": POLICIES,
    }
    assert receipt.read_text(encoding="utf-8") == (
        "system=Linux\narchitecture=x86_64\n"
    )
    assert (lane.bundle("qualified-001") / "run.manifest.json").read_bytes() == (
        previous_manifest
    )


def test_verification_failure_quarantines_the_preserved_bundle(
    lane: LaneHarness,
) -> None:
    completed = lane.invoke("verify-001", FAKE_UV_FAILURE="verification")

    assert completed.returncode == 1
    evidence = lane.report_dir("verify-001") / "verification.log"
    report = lane.report("verify-001")
    assert report["status"] == "quarantined"
    assert report["reason_codes"] == ["verification-failed"]
    assert report["run_bundle"] == {
        "run_id": "verify-001-cpu",
        "manifest_sha256": sha256(
            (lane.bundle("verify-001") / "run.manifest.json").read_bytes()
        ).hexdigest(),
        "artifact_ref": "artifact://run-bundle/run.manifest.json",
    }
    assert report["failure_evidence"] == ["artifact://attempt-evidence/verification.log"]
    assert lane.bundle("verify-001").is_dir()


def test_noisy_cpu_bundle_is_quarantined_without_becoming_promotion_evidence(
    lane: LaneHarness,
) -> None:
    completed = lane.invoke("noisy-001", FAKE_BENCHMARK_NOISE="0.031")

    assert completed.returncode == 1
    evidence = lane.report_dir("noisy-001") / "noise-policy-check.json"
    report = lane.report("noisy-001")
    assert report["status"] == "quarantined"
    assert report["promotion_allowed"] is False
    assert report["reason_codes"] == ["measurement-noise-above-policy"]
    assert report["run_bundle"]["run_id"] == "noisy-001-cpu"
    assert report["failure_evidence"] == ["artifact://attempt-evidence/noise-policy-check.json"]
    assert json.loads(evidence.read_text(encoding="utf-8")) == {
        "schema": "groundupscale.dev/trusted-hardware-noise-check/v1alpha1",
        "policy_id": "local-m4-benchmark-noise-v1",
        "maximum_iqr_over_median": 0.03,
        "passed": False,
        "failures": [{"case_id": "matmul", "iqr_over_median": 0.031}],
    }


def test_ineligible_environment_is_quarantined_before_cpu_collection(
    lane: LaneHarness,
) -> None:
    completed = lane.invoke("environment-001", FAKE_UV_FAILURE="preflight")

    assert completed.returncode == 1
    evidence = lane.report_dir("environment-001") / "environment-preflight.log"
    report = lane.report("environment-001")
    assert report["status"] == "quarantined"
    assert report["reason_codes"] == ["environment-preflight-failed"]
    assert report["run_bundle"] is None
    assert report["failure_evidence"] == ["artifact://attempt-evidence/environment-preflight.log"]
    assert "load-above-policy" in evidence.read_text(encoding="utf-8")
    assert not lane.bundle("environment-001").exists()


def test_unsupported_device_is_reported_as_hardware_unavailable(
    lane: LaneHarness,
) -> None:
    completed = lane.invoke("device-001", device="cuda")

    assert completed.returncode == 2
    report = lane.report("device-001")
    assert report["status"] == "hardware_unavailable"
    assert report["reason_codes"] == ["unsupported-device"]
    assert report["promotion_allowed"] is False
    assert report["run_bundle"] is None
    assert not (lane.artifact_store / "runs").exists()


def test_hardware_attempt_is_not_self_quarantined_by_deterministic_ci_load(
    lane: LaneHarness,
    tmp_path: Path,
) -> None:
    completed = lane.invoke(
        "no-self-load-001",
        FAKE_LOAD_STATE=str(tmp_path / "deterministic-load-active"),
    )

    assert completed.returncode == 0, completed.stderr
    assert lane.report("no-self-load-001")["status"] == "evidence_collected"


def test_reusing_a_run_tag_never_overwrites_previous_report_or_bundle(
    lane: LaneHarness,
) -> None:
    assert lane.invoke("immutable-001").returncode == 0
    report_path = lane.report_dir("immutable-001") / REPORT_NAME
    manifest_path = lane.bundle("immutable-001") / "run.manifest.json"
    report_before = report_path.read_bytes()
    manifest_before = manifest_path.read_bytes()

    completed = lane.invoke("immutable-001", FAKE_UV_FAILURE="collection")

    assert completed.returncode == 64
    assert "already exists" in completed.stderr
    assert report_path.read_bytes() == report_before
    assert manifest_path.read_bytes() == manifest_before


def test_environment_drift_during_collection_keeps_machine_reason_codes(
    lane: LaneHarness,
) -> None:
    completed = lane.invoke("drift-001", FAKE_UV_FAILURE="run_environment")

    assert completed.returncode == 1
    report = lane.report("drift-001")
    assert report["status"] == "quarantined"
    assert report["reason_codes"] == [
        "environment-drift-during-collection",
        "total-competing-cpu-above-policy",
    ]
    assert report["run_bundle"] is None


def test_non_environment_exit_two_is_reported_as_collection_failure(
    lane: LaneHarness,
) -> None:
    completed = lane.invoke("collection-exit-002", FAKE_UV_FAILURE="collection-exit-2")

    assert completed.returncode == 1
    report = lane.report("collection-exit-002")
    assert report["reason_codes"] == ["collection-failed"]


def test_malformed_noise_input_writes_replayable_invalid_evidence(
    lane: LaneHarness,
) -> None:
    completed = lane.invoke("malformed-noise-001", FAKE_BENCHMARK_INVALID="1")

    assert completed.returncode == 1
    evidence = lane.report_dir("malformed-noise-001") / "noise-policy-check.json"
    assert evidence.is_file()
    assert json.loads(evidence.read_text(encoding="utf-8"))["reason_codes"] == [
        "benchmark-observation-invalid"
    ]


def test_noise_gate_resolves_benchmark_evidence_by_manifest_role(
    lane: LaneHarness,
) -> None:
    completed = lane.invoke(
        "manifest-role-001",
        FAKE_BENCHMARK_PATH="measurements/cpu-observation.json",
    )

    assert completed.returncode == 0, completed.stderr
    report = lane.report("manifest-role-001")
    assert report["status"] == "evidence_collected"
    assert report["run_bundle"]["run_id"] == "manifest-role-001-cpu"


def test_corrupt_previous_report_fails_closed_without_hiding_current_bundle(
    lane: LaneHarness,
) -> None:
    corrupt_report = (
        lane.report_dir("corrupt-previous") / "trusted-hardware-ci-report.json"
    )
    corrupt_report.parent.mkdir(parents=True)
    corrupt_report.write_text("not-json\n", encoding="utf-8")

    completed = lane.invoke("current-001")

    assert completed.returncode == 1
    report = lane.report("current-001")
    assert report["status"] == "quarantined"
    assert report["promotion_allowed"] is False
    assert report["reason_codes"] == ["previous-evidence-unreadable"]
    assert report["run_bundle"]["run_id"] == "current-001-cpu"
    assert report["failure_evidence"] == [
        "artifact://attempt-evidence/trusted-hardware-ci-report.json"
    ]
    assert corrupt_report.read_text(encoding="utf-8") == "not-json\n"


def test_invalid_benchmark_role_is_quarantined_with_existing_failure_evidence(
    lane: LaneHarness,
) -> None:
    completed = lane.invoke("invalid-role-001", FAKE_BENCHMARK_ROLE="wrong-role")

    assert completed.returncode == 1
    evidence = lane.report_dir("invalid-role-001") / "noise-policy-check.json"
    report = lane.report("invalid-role-001")
    assert report["status"] == "quarantined"
    assert report["reason_codes"] == ["measurement-evidence-invalid"]
    assert report["failure_evidence"] == ["artifact://attempt-evidence/noise-policy-check.json"]
    assert json.loads(evidence.read_text(encoding="utf-8"))["reason_codes"] == [
        "benchmark-observation-role-invalid"
    ]


def test_unsafe_run_tag_is_rejected_before_creating_any_artifact(
    lane: LaneHarness,
) -> None:
    completed = lane.invoke("../escape")

    assert completed.returncode == 64
    assert "unsafe --tag" in completed.stderr
    assert not lane.artifact_store.exists()


def test_tampered_previous_manifest_is_not_listed_as_qualified_evidence(
    lane: LaneHarness,
) -> None:
    assert lane.invoke("qualified-before-tamper").returncode == 0
    prior_report = lane.report_dir("qualified-before-tamper") / REPORT_NAME
    prior_manifest = lane.bundle("qualified-before-tamper") / "run.manifest.json"
    prior_manifest.write_text(
        prior_manifest.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    completed = lane.invoke("after-tamper")

    assert completed.returncode == 1
    report = lane.report("after-tamper")
    assert report["status"] == "quarantined"
    assert report["reason_codes"] == ["previous-evidence-unreadable"]
    assert report["previous_qualified_evidence"] == []
    assert report["failure_evidence"] == [
        "artifact://attempt-evidence/trusted-hardware-ci-report.json"
    ]
    assert report["run_bundle"]["run_id"] == "after-tamper-cpu"
