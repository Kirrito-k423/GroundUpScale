"""GroundUpScale command line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from groundupscale.compiler import (
    CompilationContext,
    ModelBuilder,
    SemanticCompileRequest,
    SemanticCompiler,
    WorkloadBuilder,
    semantic_deployment_plan,
)
from groundupscale.ir import canonical_data
from groundupscale.probe import run_environment_probe
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
    compile_command = subparsers.add_parser(
        "compile", help="compile a YAML AnalysisPlan through Semantic IR"
    )
    compile_command.add_argument("plan")
    compile_command.add_argument("--repository-root", default=".")
    compile_command.add_argument("--output", required=True)
    compile_command.add_argument("--json", action="store_true", dest="as_json")
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
    bundle = SpecRepository(repository_root).load_analysis_plan(Path(args.plan))
    models = tuple(
        ModelBuilder().build(document)
        for _, document in sorted(bundle.models.items())
    )
    workload = WorkloadBuilder().build(
        bundle.workload, models_by_reference=bundle.models_by_reference
    )
    result = SemanticCompiler().compile(
        SemanticCompileRequest(
            workload=workload,
            models=models,
            analysis_case=bundle.analysis_case,
            deployment=semantic_deployment_plan(bundle.deployment_intent),
            context=CompilationContext(
                compiler_version="0.1.0", plugin_versions=()
            ),
        )
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_payload: object = models[0] if len(models) == 1 else {"models": models}
    _write_json(output / "model-ir.json", model_payload)
    _write_json(output / "workload-ir.json", workload)
    _write_json(output / "semantic-ir.json", result.semantic_ir)
    _write_json(output / "provenance.json", result.provenance)
    compilation = {
        "schema": "groundupscale.dev/semantic-compilation/v1alpha1",
        "compilation_fingerprint": result.compilation_fingerprint,
        "diagnostics": result.diagnostics,
        "validation_results": result.validation_results,
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "probe":
        return _run_probe(args)
    if args.command == "compile":
        return _run_compile(args)
    raise AssertionError(f"unhandled command: {args.command}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
