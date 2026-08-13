#!/usr/bin/env python3
"""Run the reproducible CPU dispatch and instrumentation ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from groundupscale.benchmark.cpu_ablation import run_cpu_ablation
from groundupscale.pipeline import compile_analysis_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("specs/plans/mac-cpu-prefill.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--target-window-ms", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    repository_root = args.repository_root.resolve()
    plan = args.plan if args.plan.is_absolute() else repository_root / args.plan
    compiled = compile_analysis_plan(repository_root, plan)
    result = run_cpu_ablation(
        compiled.bundle,
        samples=args.samples,
        warmup=args.warmup,
        target_window_ns=int(args.target_window_ms * 1_000_000),
        seed=args.seed,
    )
    output = args.output
    if not output.is_absolute():
        output = repository_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "matmul_fit": result["matmul_scaling"]["affine_fit_all_shapes"],
                "instrumentation_verdict": result["instrumentation"]["verdict"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
