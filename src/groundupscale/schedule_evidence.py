"""Shared validation primitives for Schedule Frontier evidence modules."""

from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite


SCHEDULE_FRONTIER_INPUT_SCHEMA = (
    "groundupscale.dev/schedule-frontier-input/v1alpha1"
)
SCHEDULE_FRONTIER_RESULT_SCHEMA = (
    "groundupscale.dev/schedule-frontier-result/v1alpha1"
)

class ScheduleFrontierError(ValueError):
    """Schedule evidence cannot produce a trustworthy result."""


class ScheduleEvidenceUnknown(Exception):
    """A well-formed schedule is missing evidence required for a value."""

    def __init__(self, reason_code: str, **context: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.context = context


def finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
        and value >= 0
    )


def valid_evidence_refs(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(reference, str) and reference for reference in value)
    )


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


__all__ = [
    "SCHEDULE_FRONTIER_INPUT_SCHEMA",
    "SCHEDULE_FRONTIER_RESULT_SCHEMA",
    "ScheduleEvidenceUnknown",
    "ScheduleFrontierError",
    "canonical_digest",
    "finite_nonnegative",
    "valid_evidence_refs",
]
