#!/usr/bin/env python3
"""Render a cross-hardware diagnostic report from two result or bundle paths."""

from __future__ import annotations

import argparse
import json

from groundupscale.cross_hardware import (
    compare_cross_hardware_inputs,
    render_cross_hardware_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("m4_input")
    parser.add_argument("ascend_input")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    report = compare_cross_hardware_inputs(args.m4_input, args.ascend_input)
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_cross_hardware_report(report), end="")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
