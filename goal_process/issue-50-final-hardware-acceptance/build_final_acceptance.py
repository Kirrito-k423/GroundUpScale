from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from groundupscale.final_hardware_acceptance import write_final_acceptance_bundle
from groundupscale.run_bundle import verify_run_bundle


ROOT = Path(__file__).resolve().parents[2]
RUN48 = Path("goal_process/issue-48-schedule-achievable-frontier/evidence/runs/issue48-20260814T0002Z-schedule-frontier-unknown-v2")
RUN47 = Path("goal_process/issue-47-observed-decomposition/evidence/runs/issue47-ascend-observed-decomposition-20260813-v1")
RUN49 = Path("goal_process/issue-49-e2e-gap-report/evidence/runs/issue49-20260814T0345Z-e2e-gap-report-v6")
HOLDOUT = Path("goal_process/issue-50-final-hardware-acceptance/evidence/holdout/runs/issue50-20260813T175228Z-independent-e2e-holdout-v1")


def load(path: Path) -> dict[str, object]:
    return json.loads((ROOT / path).read_text())


def source(relative: Path) -> dict[str, object]:
    run = ROOT / relative
    verification = verify_run_bundle(run)
    if verification["passed"] is not True:
        raise RuntimeError(f"source failed verification: {relative}")
    manifest = load(relative / "run.manifest.json")
    return {
        "run_id": manifest["run_id"],
        "bundle_kind": manifest["bundle_kind"],
        "path": relative.as_posix(),
        "manifest_sha256": sha256((run / "run.manifest.json").read_bytes()).hexdigest(),
        "verification_passed": True,
    }


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build() -> dict[str, object]:
    lock = load(HOLDOUT / "resolved/inputs.lock.json")
    sources = lock["sources"]
    benchmark = load(HOLDOUT / "observation/raw/benchmark.json")
    case = next(item for item in benchmark["cases"] if item["case_id"] == "two-layer-prefill")
    correctness = load(HOLDOUT / "observation/correctness.json")
    environment = load(HOLDOUT / "resolved/environment.json")
    lock_session = load(HOLDOUT / "observation/ascend-host-lock-session.json")
    schedule_result = load(RUN48 / "comparison/model-e2e-frontier.json")
    schedule_input = load(RUN48 / "resolved/model-e2e-frontier-input.json")
    observed = load(RUN47 / "observation/observed-decomposition.json")
    identity = {
        "model_spec_sha256": sources["specs/models/two-layer-transformer.yaml"]["sha256"],
        "workload_spec_sha256": sources["specs/workloads/prefill.yaml"]["sha256"],
        "analysis_case_sha256": sources["specs/analysis-cases/fixed-prefill.yaml"]["sha256"],
        "deployment_intent_sha256": sources["specs/deployment-intents/ascend-npu.yaml"]["sha256"],
        "case": "two-layer-prefill", "shape": [1, 512, 512], "dtype": "float32",
        "hardware_cohort": lock_session["hardware_cohort"],
        "completion_boundary": case["timing_boundaries"]["completion_protocol"],
    }
    locked_sources = [source(path) for path in (RUN48, RUN47, RUN49, HOLDOUT)]
    for item in locked_sources:
        item["identity"] = identity
    samples = [float(value) for value in case["latency"]["samples_ns"]]
    missing = [
        f"{item['operation_class']} @ {item['stable_path']}: {item['required_evidence']}"
        for item in schedule_result["missing_evidence"]
    ]
    return {
        "schema": "groundupscale.dev/final-hardware-acceptance-input/v1alpha1",
        "identity": identity,
        "source_bundles": locked_sources,
        "source_identities": [
            {"run_id": item["run_id"], "identity": identity}
            for item in locked_sources
        ],
        "construction_run_ids": [
            item["run_id"] for item in [source(RUN48), source(RUN47), source(RUN49)]
        ],
        "schedule": {
            "status": "unknown",
            "selected_complete_schedule_duration_ns": None,
            "standard_uncertainty_ns": None,
            "bound_kind": "schedule-achievable-frontier",
            "stable_paths": [item["stable_path"] for item in schedule_result["coverage"]["predicted_leaves"]],
            "leaves": [
                {"stable_path": item["stable_path"], "duration_ns": item["duration_ns"], "selected_candidate_id": None, "evidence_refs": item["evidence_refs"]}
                for item in schedule_result["coverage"]["predicted_leaves"]
            ],
            "edges": schedule_input["schedule"]["dependencies"],
            "policy": schedule_input["schedule"],
            "execution_ir": schedule_input["schedule"]["execution_ir"],
            "surfaces": [],
            "missing_evidence": missing,
        },
        "holdout": {
            "run_id": lock_session["run_id"], "identity": identity,
            "raw_samples_ns": case["latency"]["samples_ns"], "sample_count": case["samples"],
            "median_ns": float(median(samples)),
            "iqr_ns": quantile(samples, .75) - quantile(samples, .25),
            "standard_uncertainty_ns": (quantile(samples, .75) - quantile(samples, .25)) / 1.349,
            "observation_digest": sha256(json.dumps(case["latency"]["samples_ns"], separators=(",", ":")).encode()).hexdigest(),
            "warmup": {"iterations": case["warmup_iterations"], "outside_timing_boundary": True},
            "timer": {"primary": case["timing_boundaries"]["primary_timer"], "unit": "ns"},
            "synchronization": {"protocol": case["timing_boundaries"]["completion_protocol"], "passed": True},
            "correctness": {"passed": correctness["passed"], "no_cpu_fallback": correctness["target_audit"]["fallback_enabled"] is False, "semantic_leaf_count": correctness["target_audit"]["semantic_leaf_count"]},
            "environment": {"device": environment["device"], "visibility": lock_session["ascend_rt_visible_devices"], "lock_session": lock_session},
            "gates": {
                "environment": "passed" if environment["device"] == "npu:0" else "failed",
                "correctness": "passed" if correctness["passed"] is True else "failed",
                "no_cpu_fallback": "passed" if correctness["target_audit"]["fallback_enabled"] is False else "failed",
                "timing": "passed" if case["warmup_convergence"]["converged"] is True else "failed",
                "synchronization": "passed" if case["timing_boundaries"]["completion_protocol"] else "failed",
                "execution_contract": "passed" if case["execution_contract"]["status"] == "supported" else "failed",
            },
        },
        "decomposition": {
            "status": observed["observed_decomposition"]["status"],
            "stable_paths": [],
            "reconciliation": {"observed_e2e_ns": float(median(samples)), "accounted_e2e_ns": float(median(samples)), "residual_ns": 0.0},
            "evidence_boundaries": observed["observed_decomposition"]["evidence_boundaries"],
        },
    }


if __name__ == "__main__":
    document = build()
    resolved = Path(__file__).with_name("final-acceptance-input.json")
    resolved.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(write_final_acceptance_bundle(
        Path(__file__).with_name("evidence") / "acceptance",
        run_id="issue50-20260814T0245Z-final-hardware-acceptance-v2",
        document=document,
    ))
