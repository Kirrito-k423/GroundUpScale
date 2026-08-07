"""GroundUpScale command line interface."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml

from groundupscale.calibration import (
    fit_calibration,
    load_calibration_yaml,
    promote_calibration,
    validate_calibration,
    write_calibration_yaml,
)
from groundupscale.environment import collect_environment_validity
from groundupscale.benchmark.hardware_microbenchmark import (
    HardwareMicrobenchmarkRunner,
    aggregate_capability_envelope,
)
from groundupscale.ir import canonical_data
from groundupscale.pipeline import compile_analysis_plan
from groundupscale.probe import run_environment_probe
from groundupscale.run_bundle import RunBundleWriter, verify_run_bundle
from groundupscale.schemas.v1alpha1 import (
    HardwareBenchmarkSuiteDocument,
    HardwareCapabilityProfileDocument,
)
from groundupscale.specs import SpecRepository


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="groundupscale")
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser(
        "probe", help="verify CPU/MPS operations, timing, and memory observers"
    )
    probe.add_argument("--device", action="append", choices=("cpu", "mps"))
    probe.add_argument("--require-mps", action="store_true")
    probe.add_argument("--warmup", type=int, default=5)
    probe.add_argument("--repeats", type=int, default=20)
    probe.add_argument("--inner-iterations", type=int, default=1)
    probe.add_argument("--windows-per-sample", type=int, default=1)
    probe.add_argument("--matrix-size", type=int, default=512)
    probe.add_argument("--seed", type=int, default=20260806)
    probe.add_argument("--json", action="store_true", dest="as_json")
    preflight = subparsers.add_parser(
        "preflight",
        help="check whether the local Mac environment is eligible for trusted evidence",
    )
    preflight.add_argument("--sample-interval-seconds", type=float, default=1.0)
    preflight.add_argument("--process-samples", type=int, default=3)
    preflight.add_argument("--json", action="store_true", dest="as_json")
    hardware_benchmark = subparsers.add_parser(
        "benchmark-hardware",
        help="measure multi-Shape CPU resource envelopes and write a YAML profile",
    )
    hardware_benchmark.add_argument("suite")
    hardware_benchmark.add_argument("--repository-root", default=".")
    hardware_benchmark.add_argument("--observation-output", required=True)
    hardware_benchmark.add_argument("--profile-output", required=True)
    hardware_benchmark.add_argument("--profile-name", required=True)
    hardware_benchmark.add_argument("--profile-version", default="0.1.0")
    hardware_benchmark.add_argument(
        "--preflight-sample-interval-seconds", type=float, default=0.2
    )
    hardware_benchmark.add_argument("--preflight-process-samples", type=int, default=3)
    hardware_benchmark.add_argument("--require-valid-environment", action="store_true")
    hardware_benchmark.add_argument("--json", action="store_true", dest="as_json")
    compile_command = subparsers.add_parser(
        "compile", help="compile a YAML AnalysisPlan through Semantic IR"
    )
    compile_command.add_argument("plan")
    compile_command.add_argument("--repository-root", default=".")
    compile_command.add_argument("--output", required=True)
    compile_command.add_argument("--json", action="store_true", dest="as_json")
    run_command = subparsers.add_parser(
        "run", help="compile and execute one YAML AnalysisPlan into a Run Bundle"
    )
    run_command.add_argument("plan")
    run_command.add_argument("--repository-root", default=".")
    run_command.add_argument("--artifact-store", default=".groundupscale")
    run_command.add_argument("--run-id")
    run_command.add_argument("--samples", type=int)
    run_command.add_argument("--warmup", type=int)
    run_command.add_argument("--windows-per-sample", type=int, default=5)
    run_command.add_argument("--target-window-ms", type=float, default=20.0)
    run_command.add_argument("--collect-environment", action="store_true")
    run_command.add_argument("--require-valid-environment", action="store_true")
    run_command.add_argument(
        "--preflight-sample-interval-seconds", type=float, default=1.0
    )
    run_command.add_argument("--preflight-process-samples", type=int, default=3)
    run_command.add_argument("--json", action="store_true", dest="as_json")
    verify_command = subparsers.add_parser(
        "verify-run", help="verify every artifact digest in a Run Bundle"
    )
    verify_command.add_argument("run_bundle")
    verify_command.add_argument("--json", action="store_true", dest="as_json")
    explain_command = subparsers.add_parser(
        "explain", help="show the headline metrics and report path for a Run Bundle"
    )
    explain_command.add_argument("run_bundle")
    explain_command.add_argument("--json", action="store_true", dest="as_json")
    fit_command = subparsers.add_parser(
        "fit-calibration", help="fit a candidate profile from declared Run Bundles"
    )
    fit_command.add_argument("--run-bundle", action="append", required=True)
    fit_command.add_argument("--output", required=True)
    fit_command.add_argument("--json", action="store_true", dest="as_json")
    validate_command = subparsers.add_parser(
        "validate-calibration", help="validate a candidate against independent holdouts"
    )
    validate_command.add_argument("profile")
    validate_command.add_argument("--run-bundle", action="append", required=True)
    validate_command.add_argument("--output", required=True)
    validate_command.add_argument("--json", action="store_true", dest="as_json")
    promote_command = subparsers.add_parser(
        "promote-calibration", help="promote a candidate only after a passing validation"
    )
    promote_command.add_argument("profile")
    promote_command.add_argument("validation")
    promote_command.add_argument("--output", required=True)
    promote_command.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _run_probe(args: argparse.Namespace) -> int:
    devices = list(args.device or ["cpu"])
    if args.require_mps and "mps" not in devices:
        devices.append("mps")
    report = run_environment_probe(
        devices,
        warmup=args.warmup,
        repeats=args.repeats,
        inner_iterations=args.inner_iterations,
        windows_per_sample=args.windows_per_sample,
        matrix_size=args.matrix_size,
        seed=args.seed,
    )
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"environment probe: {'PASS' if report['ok'] else 'FAIL'}")
        for device, result in report["devices"].items():
            status = "PASS" if result.get("available") and not result.get("error") else "FAIL"
            print(f"  {device}: {status}")

    if args.require_mps and not report["mps"]["available"]:
        return 2
    return 0 if report["ok"] else 1


def _run_preflight(args: argparse.Namespace) -> int:
    report = collect_environment_validity(
        sample_interval_seconds=args.sample_interval_seconds,
        process_sample_count=args.process_samples,
    )
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            "trusted measurement preflight: "
            f"{'PASS' if report['eligible'] else 'FAIL'}"
        )
        for check in report["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(
                f"  {check['check_id']}: {status} "
                f"(observed={check['observed']!r})"
            )
        if report["reason_codes"]:
            print(f"  reasons: {', '.join(report['reason_codes'])}")
    return 0 if report["eligible"] else 2


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(canonical_data(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _run_hardware_benchmark(
    args: argparse.Namespace,
    *,
    environment_collector: Callable[..., dict[str, object]],
) -> int:
    repository_root = Path(args.repository_root).resolve()
    suite = SpecRepository(repository_root).load_document(Path(args.suite))
    if not isinstance(suite, HardwareBenchmarkSuiteDocument):
        raise ValueError(f"{args.suite}: expected HardwareBenchmarkSuite")
    environment = environment_collector(
        sample_interval_seconds=args.preflight_sample_interval_seconds,
        process_sample_count=args.preflight_process_samples,
    )
    if args.require_valid_environment and environment.get("eligible") is not True:
        if args.as_json:
            print(
                json.dumps(
                    {
                        "status": "rejected-before-benchmark",
                        "reason_codes": environment.get("reason_codes", []),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print("hardware benchmark rejected before measurement")
        return 2
    observation = HardwareMicrobenchmarkRunner(
        suite, environment=dict(environment)
    ).run()
    observation_path = Path(args.observation_output).resolve()
    _write_json(observation_path, observation)
    try:
        source_path = observation_path.relative_to(repository_root).as_posix()
    except ValueError:
        source_path = str(observation_path)
    profile_data = aggregate_capability_envelope(
        observation,
        profile_name=args.profile_name,
        profile_version=args.profile_version,
        source_path=source_path,
        source_sha256=sha256(observation_path.read_bytes()).hexdigest(),
    )
    profile = HardwareCapabilityProfileDocument.model_validate(profile_data)
    profile_path = Path(args.profile_output).resolve()
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        yaml.safe_dump(
            profile.model_dump(mode="json", by_alias=True),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    summary = {
        "schema": "groundupscale.dev/hardware-benchmark-summary/v1alpha1",
        "status": "completed",
        "environment_eligible": environment.get("eligible"),
        "hardware_cohort": observation["hardware_cohort"],
        "probe_count": len(observation["probes"]),
        "resource_count": len(profile.spec.resources),
        "observation": str(observation_path),
        "profile": str(profile_path),
    }
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"hardware benchmark: {summary['probe_count']} probes -> "
            f"{summary['resource_count']} resource envelopes"
        )
        print(f"  observation: {observation_path}")
        print(f"  profile: {profile_path}")
    return 0


def _run_compile(args: argparse.Namespace) -> int:
    repository_root = Path(args.repository_root).resolve()
    compiled = compile_analysis_plan(repository_root, Path(args.plan))
    bundle = compiled.bundle
    models = compiled.models
    workload = compiled.workload
    result = compiled.semantic
    cost_result = compiled.cost
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_payload: object = models[0] if len(models) == 1 else {"models": models}
    _write_json(output / "model-ir.json", model_payload)
    _write_json(output / "workload-ir.json", workload)
    _write_json(output / "semantic-ir.json", result.semantic_ir)
    _write_json(output / "cost-ir.json", cost_result.cost_ir)
    if compiled.hardware_prediction is not None:
        _write_json(
            output / "hardware-prediction.json", compiled.hardware_prediction
        )
    _write_json(output / "provenance.json", cost_result.provenance)
    compilation = {
        "schema": "groundupscale.dev/semantic-compilation/v1alpha1",
        "compilation_fingerprint": result.compilation_fingerprint,
        "cost_compilation_fingerprint": cost_result.compilation_fingerprint,
        "hardware_compilation_fingerprint": (
            compiled.hardware_prediction.compilation_fingerprint
            if compiled.hardware_prediction is not None
            else None
        ),
        "diagnostics": result.diagnostics,
        "semantic_validation_results": result.validation_results,
        "cost_validation_results": cost_result.validation_results,
        "sources": bundle.sources,
    }
    _write_json(output / "compilation.json", compilation)
    summary = {
        "schema": "groundupscale.dev/compile-summary/v1alpha1",
        "plan": bundle.plan.metadata.name,
        "model_count": len(models),
        "model_module_count": sum(
            1 for model in models for _ in model.walk_modules()
        ),
        "workload_node_count": sum(1 for _ in workload.walk_nodes()),
        "semantic_operation_count": sum(
            1 for _ in result.semantic_ir.walk_operations()
        ),
        "semantic_compilation_fingerprint": result.compilation_fingerprint,
        "cost_compilation_fingerprint": cost_result.compilation_fingerprint,
        "hardware_compilation_fingerprint": (
            compiled.hardware_prediction.compilation_fingerprint
            if compiled.hardware_prediction is not None
            else None
        ),
        "total_flops": cost_result.cost_ir.summary.metrics.flops,
        "parameter_bytes": cost_result.cost_ir.summary.parameter_bytes,
        "buffer_bytes": cost_result.cost_ir.summary.buffer_bytes,
        "explicit_activation_bytes": (
            cost_result.cost_ir.summary.metrics.explicit_activation_bytes
        ),
        "hardware_prediction_status": (
            compiled.hardware_prediction.status
            if compiled.hardware_prediction is not None
            else "unsupported-hardware-backend"
        ),
        "output": str(output),
    }
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"compiled {summary['plan']}: "
            f"{summary['semantic_operation_count']} semantic operations"
        )
        print(f"  fingerprint: {result.compilation_fingerprint}")
        print(f"  output: {output}")
    return 0


def _run_analysis(
    args: argparse.Namespace,
    *,
    environment_collector: Callable[..., dict[str, object]] = (
        collect_environment_validity
    ),
) -> int:
    repository_root = Path(args.repository_root).resolve()
    compiled = compile_analysis_plan(repository_root, Path(args.plan))
    environment_validity = (
        environment_collector(
            sample_interval_seconds=args.preflight_sample_interval_seconds,
            process_sample_count=args.preflight_process_samples,
        )
        if args.require_valid_environment or args.collect_environment
        else None
    )
    if (
        args.require_valid_environment
        and environment_validity is not None
        and environment_validity.get("eligible") is not True
    ):
        rejection = {
            "schema": "groundupscale.dev/run-rejection/v1alpha1",
            "status": "rejected-before-benchmark",
            "reason_codes": environment_validity.get("reason_codes", []),
            "environment_validity": environment_validity,
        }
        if args.as_json:
            print(json.dumps(rejection, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print("trusted measurement rejected before benchmark")
            print(f"  reasons: {', '.join(rejection['reason_codes'])}")
        return 2
    run = RunBundleWriter(compiled).run(
        Path(args.artifact_store),
        run_id=args.run_id,
        samples_override=args.samples,
        warmup_override=args.warmup,
        windows_per_sample=args.windows_per_sample,
        target_window_ns=int(args.target_window_ms * 1_000_000),
        environment_validity=environment_validity,
        require_valid_environment=args.require_valid_environment,
    )
    verification = verify_run_bundle(run)
    manifest = json.loads((run / "run.manifest.json").read_text(encoding="utf-8"))
    summary = {
        "schema": "groundupscale.dev/run-summary/v1alpha1",
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "device": manifest["device"],
        "artifact_count": verification["artifact_count"],
        "digests_verified": verification["passed"],
        "run_bundle": str(run),
        "report": str(run / "reports/report.html"),
    }
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"run {summary['run_id']}: {summary['status']} on {summary['device']}")
        print(f"  bundle: {summary['run_bundle']}")
        print(f"  report: {summary['report']}")
    return 0 if verification["passed"] else 1


def _run_verify(args: argparse.Namespace) -> int:
    result = verify_run_bundle(args.run_bundle)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"run {result['run_id']}: "
            f"{'PASS' if result['passed'] else 'FAIL'} ({result['artifact_count']} artifacts)"
        )
        for failure in result["failures"]:
            print(f"  {failure}")
    return 0 if result["passed"] else 1


def _run_explain(args: argparse.Namespace) -> int:
    run = Path(args.run_bundle).resolve()
    manifest = json.loads((run / "run.manifest.json").read_text(encoding="utf-8"))
    benchmark = json.loads(
        (run / "observation/raw/benchmark.json").read_text(encoding="utf-8")
    )
    prediction = json.loads(
        (run / "prediction/metrics.json").read_text(encoding="utf-8")
    )
    hardware_prediction_path = run / "prediction/hardware-backend.json"
    hardware_prediction = (
        json.loads(hardware_prediction_path.read_text(encoding="utf-8"))
        if hardware_prediction_path.is_file()
        else None
    )
    comparison_path = run / "comparison/predicted-vs-observed.json"
    comparison = (
        json.loads(comparison_path.read_text(encoding="utf-8"))
        if comparison_path.is_file()
        else None
    )
    summary = {
        "schema": "groundupscale.dev/explain-summary/v1alpha1",
        "run_id": manifest["run_id"],
        "device": manifest["device"],
        "cases": [
            {
                "case_id": case["case_id"],
                "stable_path": case["resolved_scope"],
                "median_ns": case["latency"]["median_ns"],
                "iqr_over_median": case["latency"]["iqr_over_median"],
                "throughput_per_second": case["latency"]["throughput_per_second"],
            }
            for case in benchmark["cases"]
        ],
        "predicted_framework_peak_bytes": prediction["live_set"][
            "predicted_framework_peak_bytes"
        ],
        "duration_status": prediction.get("duration_status", "uncalibrated"),
        "hardware_empirical_floor_ns": (
            prediction.get("duration", {}).get(
                "empirical_hardware_floor_ns"
            )
            if isinstance(prediction.get("duration"), dict)
            else None
        ),
        "full_duration_ns": (
            prediction.get("duration", {}).get("full_duration_ns")
            if isinstance(prediction.get("duration"), dict)
            else None
        ),
        "hardware_capability_environment_eligible": (
            all(
                item["environment_eligible"]
                for item in hardware_prediction["measured_capabilities"]
            )
            if hardware_prediction is not None
            and hardware_prediction.get("measured_capabilities")
            else None
        ),
        "comparison_status": (
            comparison["status"] if comparison is not None else "unavailable"
        ),
        "latency_comparisons": (
            [
                {
                    "case_id": item["case_id"],
                    "scope": item["scope"],
                    "empirical_hardware_floor_ns": item["predicted"][
                        "empirical_hardware_floor_ns"
                    ],
                    "empirical_compute_time_ns": item["predicted"][
                        "empirical_compute_time_ns"
                    ],
                    "empirical_memory_time_ns": item["predicted"][
                        "empirical_memory_time_ns"
                    ],
                    "limiting_resource": item["predicted"]["limiting_resource"],
                    "observed_median_ns": item["observed"]["median_ns"],
                    "observed_to_hardware_floor_ratio": item["comparison"][
                        "observed_to_hardware_floor_ratio"
                    ],
                    "error_status": item["comparison"]["error_status"],
                }
                for item in comparison["latency_cases"]
            ]
            if comparison is not None
            else []
        ),
        "memory_comparison": (
            {
                "predicted_framework_peak_bytes": comparison["memory"][
                    "predicted"
                ]["framework_peak_bytes"],
                "observed_framework_peak_bytes": comparison["memory"][
                    "observed"
                ]["framework_peak_bytes"],
                **comparison["memory"]["comparison"],
            }
            if comparison is not None
            else None
        ),
        "report": str(run / "reports/report.html"),
        "explanation_graph": str(run / "prediction/explanation.graph.json"),
    }
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"run {summary['run_id']} on {summary['device']}")
        for case in summary["cases"]:
            print(
                f"  {case['case_id']}: {case['median_ns'] / 1_000_000:.3f} ms "
                f"(IQR/median {case['iqr_over_median']:.2%})"
            )
        if summary["hardware_empirical_floor_ns"] is not None:
            print(
                "  algorithm-independent empirical hardware floor: "
                f"{summary['hardware_empirical_floor_ns'] / 1_000_000:.3f} ms "
                "(not the current implementation duration)"
            )
            print(
                "  capability evidence: "
                + (
                    "trusted"
                    if summary["hardware_capability_environment_eligible"]
                    else "exploratory (measurement preflight did not pass)"
                )
            )
        for item in summary["latency_comparisons"]:
            floor = item["empirical_hardware_floor_ns"]
            ratio = item["observed_to_hardware_floor_ratio"]
            floor_text = (
                f"{floor / 1_000_000:.3f} ms" if floor is not None else "N/A"
            )
            ratio_text = f"{ratio:.2f}x" if ratio is not None else "N/A"
            print(
                f"  compare {item['case_id']}: floor "
                f"{floor_text}, "
                f"observed {item['observed_median_ns'] / 1_000_000:.3f} ms, "
                f"distance {ratio_text}, limiting {item['limiting_resource']} "
                "(hardware-floor headroom; not prediction error)"
            )
        if summary["memory_comparison"] is not None:
            memory = summary["memory_comparison"]
            print(
                "  compare framework peak memory: predicted "
                f"{memory['predicted_framework_peak_bytes']} B, observed "
                f"{memory['observed_framework_peak_bytes']} B, absolute relative "
                f"error {memory['absolute_relative_error']:.2%}"
            )
        print(f"  report: {summary['report']}")
    return 0


def _run_fit_calibration(args: argparse.Namespace) -> int:
    profile = fit_calibration(args.run_bundle)
    write_calibration_yaml(args.output, profile)
    summary = {
        "profile_id": profile["metadata"]["profile_id"],
        "status": profile["metadata"]["status"],
        "device": profile["spec"]["applicability"]["device"],
        "fit_runs": len(profile["spec"]["fit_evidence"]),
        "output": str(Path(args.output).resolve()),
    }
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"candidate {summary['profile_id'][:12]}: {summary['fit_runs']} fit runs "
            f"for {summary['device']}"
        )
        print(f"  output: {summary['output']}")
    return 0


def _run_validate_calibration(args: argparse.Namespace) -> int:
    profile = load_calibration_yaml(args.profile)
    validation = validate_calibration(profile, args.run_bundle)
    _write_json(Path(args.output).resolve(), validation)
    summary = {
        "profile_id": validation["profile_id"],
        "passed": validation["passed"],
        "valid_holdout_runs": validation["valid_holdout_runs"],
        "quarantined_noisy_runs": validation["quarantined_noisy_runs"],
        "output": str(Path(args.output).resolve()),
    }
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"validation: {'PASS' if summary['passed'] else 'FAIL'} "
            f"({summary['valid_holdout_runs']} valid holdouts)"
        )
        print(f"  output: {summary['output']}")
    return 0 if validation["passed"] else 1


def _run_promote_calibration(args: argparse.Namespace) -> int:
    profile = load_calibration_yaml(args.profile)
    validation = json.loads(Path(args.validation).read_text(encoding="utf-8"))
    promoted = promote_calibration(profile, validation)
    write_calibration_yaml(args.output, promoted)
    summary = {
        "profile_id": promoted["metadata"]["profile_id"],
        "status": promoted["metadata"]["status"],
        "output": str(Path(args.output).resolve()),
    }
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"profile {summary['profile_id'][:12]} promoted to active")
        print(f"  output: {summary['output']}")
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    environment_collector: Callable[..., dict[str, object]] = (
        collect_environment_validity
    ),
) -> int:
    args = _parser().parse_args(argv)
    if args.command == "probe":
        return _run_probe(args)
    if args.command == "preflight":
        return _run_preflight(args)
    if args.command == "benchmark-hardware":
        return _run_hardware_benchmark(
            args, environment_collector=environment_collector
        )
    if args.command == "compile":
        return _run_compile(args)
    if args.command == "run":
        return _run_analysis(args, environment_collector=environment_collector)
    if args.command == "verify-run":
        return _run_verify(args)
    if args.command == "explain":
        return _run_explain(args)
    if args.command == "fit-calibration":
        return _run_fit_calibration(args)
    if args.command == "validate-calibration":
        return _run_validate_calibration(args)
    if args.command == "promote-calibration":
        return _run_promote_calibration(args)
    raise AssertionError(f"unhandled command: {args.command}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
