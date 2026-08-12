"""GroundUpScale command line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from hashlib import sha256
from pathlib import Path

import yaml

from groundupscale.benchmark.ascend_hardware_microbenchmark import (
    AscendNpuHardwareMicrobenchmarkRunner,
)
from groundupscale.benchmark.hardware_microbenchmark import (
    HardwareMicrobenchmarkRunner,
    aggregate_capability_envelope,
)
from groundupscale.benchmark.measurement import resolve_device
from groundupscale.calibration import (
    fit_calibration,
    load_calibration_yaml,
    promote_calibration,
    validate_calibration,
    write_calibration_yaml,
)
from groundupscale.frontier_qualification import (
    FrontierQualificationError,
    qualify_exact_shape_frontier,
)
from groundupscale.diagnostics import (
    diagnose_run_bundle,
    render_diagnostic_report,
)
from groundupscale.environment import collect_environment_validity
from groundupscale.execution_runtime import (
    ExecutionRuntime,
    create_execution_runtime,
)
from groundupscale.ir import canonical_data
from groundupscale.measurement_adapters import (
    available_measurement_devices,
    create_measurement_adapter,
)
from groundupscale.measurement_contract import MeasurementAdapter
from groundupscale.measurement_run import MeasurementRunBundleWriter
from groundupscale.operator_frontier import (
    OperatorFrontierBundleWriter,
    OperatorFrontierQualificationError,
)
from groundupscale.physical_floor_bundle import PhysicalFloorComparisonBundleWriter
from groundupscale.pipeline import compile_analysis_plan
from groundupscale.probe import run_environment_probe
from groundupscale.run_bundle import (
    NpuRunEvidence,
    RunBundleWriter,
    verify_run_bundle,
    write_blocked_transformer_run,
)
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
    hardware_benchmark.add_argument("--cohort-output")
    hardware_benchmark.add_argument("--profile-output", required=True)
    hardware_benchmark.add_argument("--profile-name", required=True)
    hardware_benchmark.add_argument("--profile-version", default="0.1.0")
    hardware_benchmark.add_argument("--logical-device-index", type=int, default=0)
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
    run_command.add_argument(
        "--case-id",
        action="append",
        help="collect only the named Benchmark Case; repeat to select more than one",
    )
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
    measure_command = subparsers.add_parser(
        "measure",
        help="run an exact-Shape case through an explicit Measurement Adapter",
    )
    measure_command.add_argument(
        "--device", required=True, choices=available_measurement_devices()
    )
    measure_command.add_argument("--logical-device-index", type=int, default=0)
    measure_command.add_argument("--m", type=int, required=True)
    measure_command.add_argument("--n", type=int, required=True)
    measure_command.add_argument("--k", type=int, required=True)
    measure_command.add_argument("--dtype", default="float32")
    measure_command.add_argument("--layout", default="row-major-contiguous")
    measure_command.add_argument(
        "--candidate",
        choices=("torch.matmul", "torch.matmul.k-split-2"),
        default="torch.matmul",
    )
    measure_command.add_argument("--seed", type=int, default=20260810)
    measure_command.add_argument("--warmup", type=int, default=20)
    measure_command.add_argument("--repetitions", type=int, default=100)
    measure_command.add_argument("--inner-iterations", type=int, default=1)
    measure_command.add_argument("--artifact-store", default=".groundupscale")
    measure_command.add_argument("--run-id", required=True)
    measure_command.add_argument("--json", action="store_true", dest="as_json")
    compare_measurement = subparsers.add_parser(
        "compare-measurement",
        help="replay a verified exact-Shape measurement beside its Physical Floor",
    )
    compare_measurement.add_argument("plan")
    compare_measurement.add_argument("measurement_bundle")
    compare_measurement.add_argument("--repository-root", default=".")
    compare_measurement.add_argument("--artifact-store", default=".groundupscale")
    compare_measurement.add_argument("--run-id", required=True)
    compare_measurement.add_argument("--json", action="store_true", dest="as_json")
    qualify_frontier = subparsers.add_parser(
        "qualify-frontier",
        help=(
            "qualify an exact-Shape frontier profile, or publish a policy-qualified "
            "operator Capability Surface"
        ),
    )
    qualify_frontier.add_argument("--policy")
    qualify_frontier.add_argument("--search-run", action="append", required=True)
    qualify_frontier.add_argument("--holdout-run", action="append", required=True)
    qualify_frontier.add_argument("--confirmation-run", action="append")
    qualify_frontier.add_argument("--query-size", action="append", type=int)
    qualify_frontier.add_argument("--artifact-store", default=".groundupscale")
    qualify_frontier.add_argument("--run-id")
    qualify_frontier.add_argument("--case-id")
    qualify_frontier.add_argument("--stable-path-pattern")
    qualify_frontier.add_argument("--candidate-family")
    qualify_frontier.add_argument("--profile-name")
    qualify_frontier.add_argument("--profile-version")
    qualify_frontier.add_argument("--observation-output")
    qualify_frontier.add_argument("--profile-output")
    qualify_frontier.add_argument("--repository-root", default=".")
    qualify_frontier.add_argument("--json", action="store_true", dest="as_json")
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
    diagnose_command = subparsers.add_parser(
        "diagnose",
        help="derive four-axis diagnostics from an evidence-qualified Run Bundle",
    )
    diagnose_command.add_argument("run_bundle")
    diagnose_command.add_argument("--json", action="store_true", dest="as_json")
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
    observation_path = Path(args.observation_output).resolve()
    if suite.spec.target.device.startswith("npu-"):
        if args.cohort_output is None:
            raise ValueError("Ascend hardware benchmark requires --cohort-output")
        adapter = create_measurement_adapter(
            "ascend-npu", logical_device_index=args.logical_device_index
        )
        cohort = dict(adapter.fingerprint_cohort())
        environment = dict(adapter.preflight())
        if cohort.get("status") != "completed" or environment.get("eligible") is not True:
            reason_codes = list(
                environment.get(
                    "reason_codes", cohort.get("reason_codes", ["ascend-preflight-failed"])
                )
            )
            if args.as_json:
                print(
                    json.dumps(
                        {
                            "status": "rejected-before-benchmark",
                            "reason_codes": reason_codes,
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
            return 2
        cohort_path = Path(args.cohort_output).resolve()
        _write_json(cohort_path, cohort)
        try:
            cohort_source_path = cohort_path.relative_to(repository_root).as_posix()
        except ValueError:
            cohort_source_path = str(cohort_path)
        runner: HardwareMicrobenchmarkRunner | AscendNpuHardwareMicrobenchmarkRunner = (
            AscendNpuHardwareMicrobenchmarkRunner(
                suite,
                environment=environment,
                cohort=cohort,
                cohort_evidence={
                    "path": cohort_source_path,
                    "sha256": sha256(cohort_path.read_bytes()).hexdigest(),
                    "schema": cohort["schema"],
                },
                logical_device_index=args.logical_device_index,
            )
        )
    else:
        environment = environment_collector(
            sample_interval_seconds=args.preflight_sample_interval_seconds,
            process_sample_count=args.preflight_process_samples,
        )
        runner = HardwareMicrobenchmarkRunner(
            suite, environment=dict(environment)
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
    observation = runner.run()
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
    measurement_adapter_factory: Callable[..., MeasurementAdapter] = (
        create_measurement_adapter
    ),
    execution_runtime_factory: Callable[[str], ExecutionRuntime] = (
        create_execution_runtime
    ),
) -> int:
    repository_root = Path(args.repository_root).resolve()
    compiled = compile_analysis_plan(repository_root, Path(args.plan))
    device = resolve_device(compiled.bundle)
    execution_runtime: ExecutionRuntime | None = None
    npu_evidence: NpuRunEvidence | None = None
    if device.startswith("npu:"):
        logical_device_index = int(device.partition(":")[2])
        adapter = measurement_adapter_factory(
            "ascend-npu", logical_device_index=logical_device_index
        )
        capabilities = dict(adapter.discover_capabilities())
        cohort = dict(adapter.fingerprint_cohort())
        preflight = dict(adapter.preflight())
        profile_cohorts = {
            profile.spec.hardware_cohort
            for profile in compiled.bundle.hardware_capability_profiles
        }
        if (
            profile_cohorts
            and cohort.get("status") == "completed"
            and cohort.get("cohort_id") not in profile_cohorts
        ):
            preflight.update(
                {
                    "status": "blocked",
                    "eligible": False,
                    "reason_codes": ["hardware-cohort-profile-mismatch"],
                    "expected_hardware_cohorts": sorted(profile_cohorts),
                    "observed_hardware_cohort": cohort.get("cohort_id"),
                }
            )

        def publish_blocked_npu_run() -> int:
            selected_run_id = args.run_id or (
                f"blocked-{device.replace(':', '-')}-"
                f"{compiled.cost.compilation_fingerprint[:8]}"
            )
            blocked_run = write_blocked_transformer_run(
                compiled,
                Path(args.artifact_store),
                run_id=selected_run_id,
                npu_evidence=NpuRunEvidence(
                    capabilities=capabilities,
                    cohort=cohort,
                    preflight=preflight,
                ),
            )
            verification = verify_run_bundle(blocked_run)
            manifest = json.loads(
                (blocked_run / "run.manifest.json").read_text(encoding="utf-8")
            )
            summary = {
                "schema": "groundupscale.dev/run-summary/v1alpha1",
                "run_id": manifest["run_id"],
                "status": manifest["status"],
                "device": manifest["device"],
                "reason_codes": manifest["reason_codes"],
                "artifact_count": verification["artifact_count"],
                "digests_verified": verification["passed"],
                "run_bundle": str(blocked_run),
                "report": None,
            }
            if args.as_json:
                print(
                    json.dumps(
                        summary, ensure_ascii=False, indent=2, sort_keys=True
                    )
                )
            else:
                print(
                    f"run {summary['run_id']}: blocked on {summary['device']}"
                )
                print(f"  reasons: {', '.join(summary['reason_codes'])}")
                print(f"  bundle: {summary['run_bundle']}")
            return 2 if verification["passed"] else 1

        if (
            capabilities.get("status") != "completed"
            or cohort.get("status") != "completed"
            or preflight.get("eligible") is not True
        ):
            return publish_blocked_npu_run()
        try:
            execution_runtime = execution_runtime_factory(device)
        except Exception as error:
            raw_reason = str(error).strip()
            reason = (
                raw_reason
                if raw_reason
                and len(raw_reason) <= 160
                and all(character.isprintable() for character in raw_reason)
                else f"runtime-initialization-failed:{type(error).__name__}"
            )
            preflight.update(
                {
                    "status": "blocked",
                    "eligible": False,
                    "reason_codes": [reason],
                    "runtime_initialization_error_type": type(error).__name__,
                }
            )
            return publish_blocked_npu_run()
        npu_evidence = NpuRunEvidence(
            capabilities=capabilities,
            cohort=cohort,
            preflight=preflight,
        )
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
        raw_rejection_reasons = environment_validity.get("reason_codes")
        rejection_reasons = (
            list(raw_rejection_reasons)
            if isinstance(raw_rejection_reasons, list)
            else []
        )
        rejection = {
            "schema": "groundupscale.dev/run-rejection/v1alpha1",
            "status": "rejected-before-benchmark",
            "reason_codes": rejection_reasons,
            "environment_validity": environment_validity,
        }
        if args.as_json:
            print(json.dumps(rejection, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print("trusted measurement rejected before benchmark")
            print(f"  reasons: {', '.join(str(reason) for reason in rejection_reasons)}")
        return 2
    run = RunBundleWriter(
        compiled,
        execution_runtime=execution_runtime,
        npu_evidence=npu_evidence,
    ).run(
        Path(args.artifact_store),
        run_id=args.run_id,
        samples_override=args.samples,
        warmup_override=args.warmup,
        windows_per_sample=args.windows_per_sample,
        target_window_ns=int(args.target_window_ms * 1_000_000),
        selected_case_ids=tuple(args.case_id) if args.case_id else None,
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
        "report": (
            str(run / "reports/report.html")
            if (run / "reports/report.html").is_file()
            else None
        ),
        "reason_codes": manifest.get("reason_codes", []),
    }
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"run {summary['run_id']}: {summary['status']} on {summary['device']}")
        print(f"  bundle: {summary['run_bundle']}")
        if summary["report"] is not None:
            print(f"  report: {summary['report']}")
        if summary["reason_codes"]:
            print(f"  reasons: {', '.join(summary['reason_codes'])}")
    if not verification["passed"]:
        return 1
    return 0 if manifest["status"] == "completed" else 2


def _run_exact_shape_frontier_qualification(args: argparse.Namespace) -> int:
    try:
        document = qualify_exact_shape_frontier(
            search_runs=args.search_run,
            holdout_runs=args.holdout_run,
            case_id=args.case_id,
            stable_path_pattern=args.stable_path_pattern,
            candidate_family=args.candidate_family,
            profile_name=args.profile_name,
            profile_version=args.profile_version,
            observation_output=args.observation_output,
            profile_output=args.profile_output,
            repository_root=args.repository_root,
        )
    except FrontierQualificationError as error:
        rejection = {
            "schema": "groundupscale.dev/frontier-qualification-rejection/v1alpha1",
            "status": "insufficient_evidence",
            "reason_code": error.reason_code,
            "detail": str(error),
        }
        if args.as_json:
            print(json.dumps(rejection, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(
                f"frontier qualification rejected: {rejection['reason_code']} "
                f"({rejection['detail']})"
            )
        return 2
    anchor = document.spec.anchors[0]
    summary = {
        "schema": "groundupscale.dev/frontier-qualification-summary/v1alpha1",
        "status": anchor.observation_validity,
        "frontier_role": anchor.frontier_role,
        "anchor_id": anchor.anchor_id,
        "latency_ns": anchor.latency_ns,
        "standard_uncertainty_ns": anchor.standard_uncertainty_ns,
        "observation": str(Path(args.observation_output).resolve()),
        "profile": str(Path(args.profile_output).resolve()),
    }
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"qualified {summary['anchor_id']}: {summary['latency_ns']:.3f} ns "
            f"± {summary['standard_uncertainty_ns']:.3f} ns"
        )
        print(f"  profile: {summary['profile']}")
    return 0


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


def _run_measurement(args: argparse.Namespace) -> int:
    case = {
        "schema": "groundupscale.dev/exact-shape-matmul-case/v1alpha1",
        "operation": "MatMul",
        "shape": {
            "left": [args.m, args.k],
            "right": [args.k, args.n],
        },
        "dtype": args.dtype,
        "layout": args.layout,
        "seed": args.seed,
        "candidate": args.candidate,
        "warmup_iterations": args.warmup,
        "repetitions": args.repetitions,
        "inner_iterations": args.inner_iterations,
    }
    adapter = create_measurement_adapter(
        args.device,
        logical_device_index=args.logical_device_index,
    )
    run = MeasurementRunBundleWriter(adapter).run(
        Path(args.artifact_store),
        case=case,
        run_id=args.run_id,
    )
    verification = verify_run_bundle(run)
    manifest = json.loads(
        (run / "run.manifest.json").read_text(encoding="utf-8")
    )
    reason_codes: list[str] = []
    if manifest["status"] == "blocked":
        failure = json.loads(
            (run / "adapter/failure.json").read_text(encoding="utf-8")
        )
        reason_codes = list(failure["reason_codes"])
    summary = {
        "schema": "groundupscale.dev/measurement-run-summary/v1alpha1",
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "device": manifest["device"],
        "hardware_cohort": manifest["hardware_cohort"],
        "reason_codes": reason_codes,
        "verification_passed": verification["passed"],
        "run_bundle": str(run),
    }
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"measurement {summary['run_id']}: {summary['status']} "
            f"on {summary['device']}"
        )
        print(f"  bundle: {summary['run_bundle']}")
        if reason_codes:
            print(f"  reasons: {', '.join(reason_codes)}")
    if not verification["passed"]:
        return 1
    return 2 if manifest["status"] == "blocked" else 0


def _run_compare_measurement(args: argparse.Namespace) -> int:
    compiled = compile_analysis_plan(
        Path(args.repository_root).resolve(), Path(args.plan)
    )
    run = PhysicalFloorComparisonBundleWriter(compiled).run(
        args.artifact_store,
        measurement_bundle=args.measurement_bundle,
        run_id=args.run_id,
    )
    verification = verify_run_bundle(run)
    comparison = json.loads(
        (run / "comparison/physical-floor-vs-observation.json").read_text(
            encoding="utf-8"
        )
    )
    summary = {
        "schema": "groundupscale.dev/physical-floor-comparison-summary/v1alpha1",
        "status": "completed" if verification["passed"] else "verification-failed",
        "run_id": args.run_id,
        "stable_path": comparison["stable_path"],
        "hardware_cohort": comparison["hardware_cohort"],
        "resource_physical_floor_ns": comparison["physical_floor"][
            "resource_physical_floor_ns"
        ],
        "full_duration_ns": comparison["physical_floor"]["full_duration_ns"],
        "observation_median_ns": comparison["observation"]["median_ns"],
        "run_bundle": str(run),
        "verification_passed": verification["passed"],
    }
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"comparison {summary['run_id']}: {summary['status']} "
            f"for {summary['stable_path']}"
        )
        print(f"  bundle: {summary['run_bundle']}")
    return 0 if verification["passed"] else 1


def _run_explain(args: argparse.Namespace) -> int:
    run = Path(args.run_bundle).resolve()
    manifest = json.loads((run / "run.manifest.json").read_text(encoding="utf-8"))
    if manifest.get("bundle_kind") == "physical-floor-observation-comparison":
        comparison = json.loads(
            (run / "comparison/physical-floor-vs-observation.json").read_text(
                encoding="utf-8"
            )
        )
        summary = {
            "schema": "groundupscale.dev/explain-summary/v1alpha1",
            "bundle_kind": manifest["bundle_kind"],
            "run_id": manifest["run_id"],
            "device": manifest["device"],
            "hardware_cohort": manifest["hardware_cohort"],
            "stable_path": comparison["stable_path"],
            "resource_physical_floor_ns": comparison["physical_floor"][
                "resource_physical_floor_ns"
            ],
            "full_duration_ns": comparison["physical_floor"][
                "full_duration_ns"
            ],
            "capability_quality": comparison["physical_floor"]["quality"],
            "observation_median_ns": comparison["observation"]["median_ns"],
            "observed_to_physical_floor_ratio": comparison["comparison"][
                "observed_to_physical_floor_ratio"
            ],
            "unsupported_region_count": comparison["unsupported_regions"][
                "count"
            ],
            "report": str(run / "reports/report.html"),
            "explanation_graph": str(
                run / "prediction/explanation.graph.json"
            ),
        }
        if args.as_json:
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"run {summary['run_id']} on {summary['device']}")
            print(
                "  Resource Physical Floor: "
                f"{summary['resource_physical_floor_ns'] / 1_000:.3f} μs"
            )
            print(
                "  Observation: "
                f"{summary['observation_median_ns'] / 1_000:.3f} μs"
            )
            print("  full implementation duration: unknown")
        return 0
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
    prediction_schema = prediction.get("schema")
    if prediction_schema not in {
        "groundupscale.dev/prediction/v1alpha1",
        "groundupscale.dev/prediction/v1alpha2",
    }:
        raise ValueError(f"unsupported prediction schema: {prediction_schema}")
    comparison_schema = comparison.get("schema") if comparison is not None else None
    if comparison is not None and comparison_schema not in {
        "groundupscale.dev/prediction-observation-comparison/v1alpha1",
        "groundupscale.dev/prediction-observation-comparison/v1alpha2",
    }:
        raise ValueError(f"unsupported comparison schema: {comparison_schema}")
    summary = {
        "schema": "groundupscale.dev/explain-summary/v1alpha2",
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
        "hardware_resource_physical_floor_ns": (
            prediction.get("duration", {}).get("resource_physical_floor_ns")
            if isinstance(prediction.get("duration"), dict)
            else None
        ),
        "hardware_floor_schedule": (
            prediction.get("duration", {}).get("schedule")
            if isinstance(prediction.get("duration"), dict)
            else None
        ),
        "hardware_serialized_floor_ns": (
            prediction.get("duration", {}).get("serialized_hardware_floor_ns")
            if isinstance(prediction.get("duration"), dict)
            else None
        ),
        "hardware_ideal_dag_floor_ns": (
            prediction.get("duration", {}).get("ideal_dag_hardware_floor_ns")
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
                    "empirical_hardware_floor_ns": item["predicted"].get(
                        "empirical_hardware_floor_ns"
                    ),
                    "resource_physical_floor_ns": item["predicted"].get(
                        "resource_physical_floor_ns"
                    ),
                    "schedule": item["predicted"].get("schedule"),
                    "serialized_hardware_floor_ns": item["predicted"].get(
                        "serialized_hardware_floor_ns"
                    ),
                    "critical_path_hardware_floor_ns": item["predicted"].get(
                        "critical_path_hardware_floor_ns"
                    ),
                    "resource_hardware_floor_ns": item["predicted"].get(
                        "resource_hardware_floor_ns"
                    ),
                    "ideal_dag_hardware_floor_ns": item["predicted"].get(
                        "ideal_dag_hardware_floor_ns"
                    ),
                    "empirical_compute_time_ns": item["predicted"].get(
                        "empirical_compute_time_ns"
                    ),
                    "empirical_memory_time_ns": item["predicted"].get(
                        "empirical_memory_time_ns"
                    ),
                    "limiting_resource": item["predicted"].get(
                        "limiting_resource"
                    ),
                    "resource_limiting_resource": item["predicted"].get(
                        "resource_limiting_resource"
                    ),
                    "operator_achievable_frontier_ns": item["predicted"].get(
                        "operator_achievable_frontier_ns"
                    ),
                    "operator_frontier_standard_uncertainty_ns": item[
                        "predicted"
                    ].get("operator_frontier_standard_uncertainty_ns"),
                    "operator_frontier_match_status": item["predicted"].get(
                        "operator_frontier_match_status"
                    ),
                    "operator_frontier_anchor_ids": item["predicted"].get(
                        "operator_frontier_anchor_ids", []
                    ),
                    "operator_frontier_hardware_cohort": item["predicted"].get(
                        "operator_frontier_hardware_cohort"
                    ),
                    "operator_frontier_candidate_digest": item["predicted"].get(
                        "operator_frontier_candidate_digest"
                    ),
                    "operator_frontier_input_corpus_digest": item["predicted"].get(
                        "operator_frontier_input_corpus_digest"
                    ),
                    "operator_frontier_execution_contract_digest": item[
                        "predicted"
                    ].get("operator_frontier_execution_contract_digest"),
                    "observed_median_ns": item["observed"]["median_ns"],
                    "observed_to_hardware_floor_ratio": item["comparison"][
                        "observed_to_hardware_floor_ratio"
                    ],
                    "error_status": item["comparison"]["error_status"],
                    "operator_frontier_efficiency": item["comparison"].get(
                        "operator_frontier_efficiency"
                    ),
                    "frontier_efficiency_status": item["comparison"].get(
                        "frontier_efficiency_status"
                    ),
                    "operator_frontier_gap_status": item["comparison"].get(
                        "operator_frontier_gap_status"
                    ),
                    "operator_frontier_combined_uncertainty_ns": item[
                        "comparison"
                    ].get("operator_frontier_combined_uncertainty_ns"),
                    "operator_frontier_uncertainty_components_ns": item[
                        "comparison"
                    ].get("operator_frontier_uncertainty_components_ns"),
                    "operator_frontier_uncertainty_policy": item[
                        "comparison"
                    ].get("operator_frontier_uncertainty_policy"),
                    "operator_frontier_comparison_reason_codes": item[
                        "comparison"
                    ].get("operator_frontier_comparison_reason_codes", []),
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
                f"({summary['hardware_floor_schedule']}; not the current "
                "implementation duration)"
            )
            if summary["hardware_ideal_dag_floor_ns"] is not None:
                print(
                    "  ideal DAG reference floor: "
                    f"{summary['hardware_ideal_dag_floor_ns'] / 1_000_000:.3f} ms"
                )
            print(
                "  capability evidence: "
                + (
                    "trusted"
                    if summary["hardware_capability_environment_eligible"]
                    else "exploratory (measurement preflight did not pass)"
                )
            )
        elif summary["hardware_resource_physical_floor_ns"] is not None:
            print(
                "  selected compound duration: unknown "
                f"({summary['duration_status']})"
            )
            print(
                "  resource physical floor (not adoptable as a point prediction): "
                f"{summary['hardware_resource_physical_floor_ns'] / 1_000_000:.3f} ms"
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


def _run_operator_frontier_qualification(args: argparse.Namespace) -> int:
    try:
        policy = yaml.safe_load(Path(args.policy).read_text(encoding="utf-8"))
        if not isinstance(policy, dict):
            raise OperatorFrontierQualificationError(
                "qualification policy must be a JSON/YAML object",
                reason_code="invalid-qualification-policy",
            )
        run = OperatorFrontierBundleWriter().run(
            args.artifact_store,
            run_id=args.run_id,
            qualification_policy=policy,
            search_runs=args.search_run,
            holdout_runs=args.holdout_run,
            confirmation_runs=args.confirmation_run,
            query_sizes=tuple(args.query_size),
        )
    except OperatorFrontierQualificationError as error:
        failure = {
            "schema": (
                "groundupscale.dev/operator-frontier-qualification-failure/"
                "v1alpha1"
            ),
            "status": "insufficient_evidence",
            "reason_code": error.reason_code,
            "message": str(error),
        }
        if args.as_json:
            print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(
                "operator Frontier qualification failed: "
                f"{failure['reason_code']}"
            )
        return 2
    verification = verify_run_bundle(run)
    result = diagnose_run_bundle(run)
    qualification = json.loads(
        (run / "frontier/qualification.json").read_text(encoding="utf-8")
    )
    summary = {
        "schema": (
            "groundupscale.dev/operator-frontier-run-summary/v1alpha1"
        ),
        "run_id": args.run_id,
        "status": qualification["status"],
        "hardware_cohort": qualification["hardware_cohort"],
        "anchor_count": len(qualification["anchors"]),
        "surface": qualification["surface"]["surface_id"],
        "reason_code": qualification.get("reason_code"),
        "stopping_decision": qualification.get("stopping_decision"),
        "query_statuses": {
            str(next(iter(query["query_shape"].values()))): query["status"]
            for query in result["capability_surface_queries"]
        },
        "verification_passed": verification["passed"],
        "run_bundle": str(run),
    }
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"operator Frontier {summary['run_id']}: {summary['status']} "
            f"({summary['anchor_count']} anchors)"
        )
        print(f"  bundle: {summary['run_bundle']}")
    return 0 if verification["passed"] else 1


def _run_diagnose(args: argparse.Namespace) -> int:
    result = diagnose_run_bundle(args.run_bundle)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_diagnostic_report(result), end="")
    return 0


def _run_qualify_frontier(args: argparse.Namespace) -> int:
    if args.policy is not None:
        missing = tuple(
            option
            for option, value in (
                ("--confirmation-run", args.confirmation_run),
                ("--query-size", args.query_size),
                ("--run-id", args.run_id),
            )
            if not value
        )
        if missing:
            raise ValueError(
                "policy-qualified frontier requires " + ", ".join(missing)
            )
        return _run_operator_frontier_qualification(args)

    missing = tuple(
        option
        for option, value in (
            ("--case-id", args.case_id),
            ("--stable-path-pattern", args.stable_path_pattern),
            ("--candidate-family", args.candidate_family),
            ("--profile-name", args.profile_name),
            ("--profile-version", args.profile_version),
            ("--observation-output", args.observation_output),
            ("--profile-output", args.profile_output),
        )
        if not value
    )
    if missing:
        raise ValueError(
            "exact-Shape frontier qualification requires " + ", ".join(missing)
        )
    return _run_exact_shape_frontier_qualification(args)


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
    measurement_adapter_factory: Callable[..., MeasurementAdapter] = (
        create_measurement_adapter
    ),
    execution_runtime_factory: Callable[[str], ExecutionRuntime] = (
        create_execution_runtime
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
        return _run_analysis(
            args,
            environment_collector=environment_collector,
            measurement_adapter_factory=measurement_adapter_factory,
            execution_runtime_factory=execution_runtime_factory,
        )
    if args.command == "measure":
        return _run_measurement(args)
    if args.command == "compare-measurement":
        return _run_compare_measurement(args)
    if args.command == "qualify-frontier":
        return _run_qualify_frontier(args)
    if args.command == "verify-run":
        return _run_verify(args)
    if args.command == "explain":
        return _run_explain(args)
    if args.command == "diagnose":
        return _run_diagnose(args)
    if args.command == "fit-calibration":
        return _run_fit_calibration(args)
    if args.command == "validate-calibration":
        return _run_validate_calibration(args)
    if args.command == "promote-calibration":
        return _run_promote_calibration(args)
    raise AssertionError(f"unhandled command: {args.command}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
