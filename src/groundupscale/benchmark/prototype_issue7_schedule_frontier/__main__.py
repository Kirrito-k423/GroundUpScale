"""Terminal shell for the throwaway issue 7 schedule-frontier prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .model import evaluate


BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"


def _summary(result: dict[str, Any], scenario: str) -> str:
    boundaries = result["boundaries_ms"]
    ledger = result["observed_time_ledger"]["by_kind_ms"]
    counterfactual = result["counterfactual"]
    if scenario == "counterfactual":
        e2e = counterfactual["after_e2e_ms"]
        removed_schedule_wait = counterfactual["removed_bucket_total_ms"]
        operator_execution = counterfactual["operator_observed_after_ms"]
        schedule_wait = ledger["schedule_wait"] - removed_schedule_wait
        integration = ledger["integration"]
        ledger_total = counterfactual["after_e2e_ms"]
        scenario_note = (
            f"explicit batched dispatch removes {counterfactual['observed_delta_ms']:.6f} ms"
        )
    else:
        e2e = boundaries["observed_trace_e2e"]
        operator_execution = ledger["operator_execution"]
        schedule_wait = ledger["schedule_wait"]
        integration = ledger["integration"]
        ledger_total = result["observed_time_ledger"]["total_ms"]
        scenario_note = "published E2E plus explicitly marked prototype trace leaves"
    lines = [
        f"{BOLD}PROTOTYPE ONLY — two-layer M4 schedule ledger{RESET}",
        f"{BOLD}scenario{RESET}: {scenario}",
        f"{DIM}{scenario_note}{RESET}",
        "",
        f"{BOLD}Four separate boundaries (ms){RESET}",
        f"  Resource Physical Floor       {boundaries['resource_physical_floor']:10.6f}",
        f"  Operator Achievable Frontier  {boundaries['operator_achievable_frontier']:10.6f}",
        f"  Schedule Achievable Frontier  {boundaries['schedule_achievable_frontier']:10.6f}",
        f"  Observed E2E for scenario      {e2e:10.6f}",
        "",
        f"{BOLD}Exclusive observed ledger ({scenario} ms){RESET}",
        f"  operator_execution            {operator_execution:10.6f}",
        f"  schedule_wait                 {schedule_wait:10.6f}",
        f"  integration                   {integration:10.6f}",
        f"  total                         {ledger_total:10.6f}",
        "",
        f"{BOLD}Diagnosis{RESET}: {result['diagnosis']['verdict']}",
        f"{BOLD}Checks{RESET}: "
        + "  ".join(
            f"{name}={'PASS' if passed else 'FAIL'}"
            for name, passed in result["checks"].items()
        ),
        "",
        f"{BOLD}Top predicted / observed operators{RESET}",
    ]
    for predicted, observed in zip(
        result["top_10"]["predicted_operator_frontier"][:3],
        result["top_10"]["observed_operator_trace"][:3],
        strict=True,
    ):
        lines.append(
            f"  P{predicted['rank']} {predicted['operator_id']}={predicted['duration_ms']:.3f}  "
            f"O{observed['rank']} {observed['operator_id']}={observed['duration_ms']:.3f}"
        )
    lines.extend(
        [
            "",
            f"{BOLD}Keys{RESET}: [b] baseline  [c] counterfactual  [x] rejected cases  [q] quit",
        ]
    )
    return "\n".join(lines)


def _counterexample_summary(result: dict[str, Any]) -> str:
    lines = [
        f"{BOLD}PROTOTYPE ONLY — rejected implicit semantics{RESET}",
        "",
    ]
    for item in result["counterexamples"]:
        lines.append(
            f"  {item['id']}: {'REJECTED' if not item['accepted'] else 'ACCEPTED'} "
            f"({item['actual_reason']})"
        )
    lines.extend(
        [
            "",
            f"{BOLD}Keys{RESET}: [b] baseline  [c] counterfactual  [x] rejected cases  [q] quit",
        ]
    )
    return "\n".join(lines)


def _interactive(result: dict[str, Any]) -> int:
    scenario = "baseline"
    while True:
        print("\033[2J\033[H", end="")
        if scenario == "rejected":
            print(_counterexample_summary(result))
        else:
            print(_summary(result, scenario))
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            return 0
        if choice == "q":
            return 0
        if choice == "b":
            scenario = "baseline"
        elif choice == "c":
            scenario = "counterfactual"
        elif choice == "x":
            scenario = "rejected"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PROTOTYPE ONLY: evaluate the issue 7 schedule-frontier ledger"
    )
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate()
    if args.run_all:
        if args.output is None:
            parser.error("--run-all requires --output")
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(_summary(result, "baseline"))
        print(f"\nraw_result={args.output}")
        raise SystemExit(0 if result["question_answered"] else 1)
    raise SystemExit(_interactive(result))


if __name__ == "__main__":
    main()
