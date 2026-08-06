"""GroundUpScale command line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from groundupscale.probe import run_environment_probe


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "probe":  # pragma: no cover - argparse enforces this.
        raise AssertionError(f"unhandled command: {args.command}")

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


if __name__ == "__main__":
    raise SystemExit(main())
