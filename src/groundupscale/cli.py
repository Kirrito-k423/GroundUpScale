"""GroundUpScale command line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from groundupscale.ir import canonical_data
from groundupscale.pipeline import compile_analysis_plan
from groundupscale.probe import run_environment_probe
from groundupscale.run_bundle import RunBundleWriter, verify_run_bundle


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


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(canonical_data(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


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
    _write_json(output / "provenance.json", cost_result.provenance)
    compilation = {
        "schema": "groundupscale.dev/semantic-compilation/v1alpha1",
        "compilation_fingerprint": result.compilation_fingerprint,
        "cost_compilation_fingerprint": cost_result.compilation_fingerprint,
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
        "total_flops": cost_result.cost_ir.summary.metrics.flops,
        "parameter_bytes": cost_result.cost_ir.summary.parameter_bytes,
        "buffer_bytes": cost_result.cost_ir.summary.buffer_bytes,
        "explicit_activation_bytes": (
            cost_result.cost_ir.summary.metrics.explicit_activation_bytes
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


def _run_analysis(args: argparse.Namespace) -> int:
    repository_root = Path(args.repository_root).resolve()
    compiled = compile_analysis_plan(repository_root, Path(args.plan))
    run = RunBundleWriter(compiled).run(
        Path(args.artifact_store),
        run_id=args.run_id,
        samples_override=args.samples,
        warmup_override=args.warmup,
        windows_per_sample=args.windows_per_sample,
        target_window_ns=int(args.target_window_ms * 1_000_000),
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
        print(f"  report: {summary['report']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "probe":
        return _run_probe(args)
    if args.command == "compile":
        return _run_compile(args)
    if args.command == "run":
        return _run_analysis(args)
    if args.command == "verify-run":
        return _run_verify(args)
    if args.command == "explain":
        return _run_explain(args)
    raise AssertionError(f"unhandled command: {args.command}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
