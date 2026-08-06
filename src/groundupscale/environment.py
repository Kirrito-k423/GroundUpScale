"""Allowlisted environment-validity evidence for trusted Mac measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import os
import platform
import re
import subprocess
import time
from typing import Any, Mapping

import psutil


ENVIRONMENT_VALIDITY_SCHEMA = "groundupscale.dev/environment-validity/v1alpha1"


@dataclass(frozen=True)
class EnvironmentValidityPolicy:
    """Versioned, predeclared thresholds for the initial Apple Silicon cohort."""

    policy_id: str = "local-apple-silicon-v1"
    required_system: str = "Darwin"
    required_machine: str = "arm64"
    require_ac_power: bool = True
    require_nominal_thermal_status: bool = True
    maximum_normalized_one_minute_load: float = 0.25
    maximum_competing_process_cpu_percent: float = 25.0
    allowlist_only: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.maximum_normalized_one_minute_load:
            raise ValueError("maximum normalized load must be non-negative")
        if not 0 <= self.maximum_competing_process_cpu_percent:
            raise ValueError("maximum competing CPU percent must be non-negative")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _check(
    check_id: str,
    *,
    passed: bool,
    observed: object,
    required: object,
    failure_reason: str,
) -> dict[str, Any]:
    result = {
        "check_id": check_id,
        "passed": passed,
        "observed": observed,
        "required": required,
    }
    if not passed:
        result["reason_code"] = failure_reason
    return result


def evaluate_environment_validity(
    observations: Mapping[str, object],
    *,
    policy: EnvironmentValidityPolicy | None = None,
) -> dict[str, Any]:
    """Evaluate supplied allowlisted observations without assuming unknowns pass."""
    selected = policy or EnvironmentValidityPolicy()
    normalized_observations: dict[str, Any] = {
        key: dict(_mapping(value)) for key, value in observations.items()
    }

    platform_observation = _mapping(observations.get("platform"))
    system = platform_observation.get("system")
    machine = platform_observation.get("machine")
    platform_ok = (
        system == selected.required_system and machine == selected.required_machine
    )
    checks = [
        _check(
            "trusted-platform",
            passed=platform_ok,
            observed={"system": system, "machine": machine},
            required={
                "system": selected.required_system,
                "machine": selected.required_machine,
            },
            failure_reason=(
                "platform-unverified"
                if system is None or machine is None
                else "platform-outside-policy"
            ),
        )
    ]

    power_observation = _mapping(observations.get("power"))
    power_source = power_observation.get("source")
    power_ok = power_source == "ac" if selected.require_ac_power else True
    checks.append(
        _check(
            "ac-power",
            passed=power_ok,
            observed=power_source,
            required="ac" if selected.require_ac_power else "any",
            failure_reason=(
                "ac-power-unverified"
                if power_source in {None, "unknown"}
                else "not-on-ac-power"
            ),
        )
    )

    thermal_observation = _mapping(observations.get("thermal"))
    thermal_status = thermal_observation.get("status")
    thermal_ok = (
        thermal_status == "nominal"
        if selected.require_nominal_thermal_status
        else True
    )
    checks.append(
        _check(
            "thermal-nominal",
            passed=thermal_ok,
            observed=thermal_status,
            required="nominal" if selected.require_nominal_thermal_status else "any",
            failure_reason=(
                "thermal-unverified"
                if thermal_status in {None, "unknown"}
                else "thermal-warning"
            ),
        )
    )

    logical_cpu_count = _number(platform_observation.get("logical_cpu_count"))
    load_observation = _mapping(observations.get("load"))
    one_minute_load = _number(load_observation.get("one_minute"))
    normalized_load = (
        one_minute_load / logical_cpu_count
        if one_minute_load is not None
        and logical_cpu_count is not None
        and logical_cpu_count > 0
        else None
    )
    if "load" not in normalized_observations:
        normalized_observations["load"] = dict(load_observation)
    normalized_observations["load"]["normalized_one_minute"] = normalized_load
    load_ok = (
        normalized_load is not None
        and normalized_load <= selected.maximum_normalized_one_minute_load
    )
    checks.append(
        _check(
            "settled-load",
            passed=load_ok,
            observed=normalized_load,
            required={
                "maximum": selected.maximum_normalized_one_minute_load,
                "normalization": "one-minute load / logical CPU count",
            },
            failure_reason=(
                "load-unverified" if normalized_load is None else "load-above-policy"
            ),
        )
    )

    competitor_observation = _mapping(observations.get("competitors"))
    top = competitor_observation.get("top")
    competitor_cpu_values: list[float] = []
    if isinstance(top, list):
        for item in top:
            if not isinstance(item, Mapping):
                continue
            cpu_percent = _number(item.get("cpu_percent"))
            if cpu_percent is not None:
                competitor_cpu_values.append(cpu_percent)
    maximum_competitor_cpu = (
        max(competitor_cpu_values, default=0.0) if isinstance(top, list) else None
    )
    competitor_ok = (
        maximum_competitor_cpu is not None
        and maximum_competitor_cpu
        <= selected.maximum_competing_process_cpu_percent
    )
    checks.append(
        _check(
            "no-competing-process",
            passed=competitor_ok,
            observed=maximum_competitor_cpu,
            required={
                "maximum_cpu_percent": (
                    selected.maximum_competing_process_cpu_percent
                ),
                "scope": "single non-coordinator process",
            },
            failure_reason=(
                "competitor-load-unverified"
                if maximum_competitor_cpu is None
                else "competing-process-above-policy"
            ),
        )
    )

    failures = [check for check in checks if not check["passed"]]
    return {
        "schema": ENVIRONMENT_VALIDITY_SCHEMA,
        "eligible": not failures,
        "reason_codes": [check["reason_code"] for check in failures],
        "policy": asdict(selected),
        "observations": normalized_observations,
        "checks": checks,
        "interpretation": (
            "Preflight eligibility does not replace per-run benchmark noise gates."
        ),
    }


def _pmset_output(mode: str) -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        completed = subprocess.run(
            ["/usr/bin/pmset", "-g", mode],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _power_observation() -> dict[str, Any]:
    output = _pmset_output("batt")
    if output is None:
        return {"source": "unknown", "battery_percent": None}
    lowered = output.lower()
    source = (
        "ac"
        if "ac power" in lowered
        else "battery"
        if "battery power" in lowered
        else "unknown"
    )
    percentage = re.search(r"(\d+(?:\.\d+)?)%", output)
    return {
        "source": source,
        "battery_percent": float(percentage.group(1)) if percentage else None,
    }


def _thermal_observation() -> dict[str, Any]:
    output = _pmset_output("therm")
    if output is None:
        return {
            "status": "unknown",
            "thermal_warning": None,
            "performance_warning": None,
            "speed_limits_percent": {},
        }
    lowered = output.lower()
    no_thermal_warning = "no thermal warning level has been recorded" in lowered
    no_performance_warning = (
        "no performance warning level has been recorded" in lowered
    )
    speed_limits = {
        name.lower(): int(value)
        for name, value in re.findall(
            r"(CPU_Speed_Limit|GPU_Speed_Limit|Scheduler_Limit)\s*=\s*(\d+)",
            output,
            flags=re.IGNORECASE,
        )
    }
    restricted = any(value < 100 for value in speed_limits.values())
    if no_thermal_warning and no_performance_warning and not restricted:
        status = "nominal"
    elif restricted or (
        "thermal warning level" in lowered and not no_thermal_warning
    ) or (
        "performance warning level" in lowered and not no_performance_warning
    ):
        status = "warning"
    else:
        status = "unknown"
    return {
        "status": status,
        "thermal_warning": (
            False if no_thermal_warning else True if status == "warning" else None
        ),
        "performance_warning": (
            False
            if no_performance_warning
            else True
            if status == "warning"
            else None
        ),
        "speed_limits_percent": speed_limits,
    }


def _coordinator_process_ids() -> set[int]:
    coordinator = psutil.Process()
    process_ids = {coordinator.pid}
    try:
        process_ids.update(parent.pid for parent in coordinator.parents())
    except (psutil.Error, RuntimeError):
        pass
    return process_ids


def _competitor_observation(
    *, sample_interval_seconds: float, sample_count: int
) -> dict[str, Any]:
    excluded = _coordinator_process_ids()
    tracked: dict[int, psutil.Process] = {}
    names: dict[int, str] = {}
    maxima: dict[int, float] = {}
    for process in psutil.process_iter(["pid", "name"]):
        if process.pid in excluded:
            continue
        try:
            process.cpu_percent(None)
            tracked[process.pid] = process
            names[process.pid] = process.info.get("name") or "unknown"
        except (psutil.Error, RuntimeError):
            continue

    for _ in range(sample_count):
        time.sleep(sample_interval_seconds)
        for process_id, process in list(tracked.items()):
            try:
                cpu_percent = float(process.cpu_percent(None))
            except (psutil.Error, RuntimeError):
                tracked.pop(process_id, None)
                continue
            maxima[process_id] = max(maxima.get(process_id, 0.0), cpu_percent)

    top = sorted(
        (
            {
                "pid": process_id,
                "name": names.get(process_id, "unknown"),
                "cpu_percent": cpu_percent,
            }
            for process_id, cpu_percent in maxima.items()
            if cpu_percent > 0.0
        ),
        key=lambda item: (-item["cpu_percent"], item["pid"]),
    )[:10]
    return {
        "sample_interval_seconds": sample_interval_seconds,
        "sample_count": sample_count,
        "excluded_process_scope": "coordinator process and ancestors",
        "top": top,
    }


def collect_environment_validity(
    *,
    policy: EnvironmentValidityPolicy | None = None,
    sample_interval_seconds: float = 1.0,
    process_sample_count: int = 3,
) -> dict[str, Any]:
    """Collect an allowlisted preflight snapshot and evaluate it fail-closed."""
    if sample_interval_seconds <= 0:
        raise ValueError("sample interval must be positive")
    if process_sample_count <= 0:
        raise ValueError("process sample count must be positive")
    logical_cpu_count = os.cpu_count()
    try:
        one_minute, five_minutes, fifteen_minutes = os.getloadavg()
    except OSError:
        one_minute = five_minutes = fifteen_minutes = None
    observations: dict[str, object] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "logical_cpu_count": logical_cpu_count,
        },
        "power": _power_observation(),
        "thermal": _thermal_observation(),
        "load": {
            "one_minute": one_minute,
            "five_minutes": five_minutes,
            "fifteen_minutes": fifteen_minutes,
        },
        "competitors": _competitor_observation(
            sample_interval_seconds=sample_interval_seconds,
            sample_count=process_sample_count,
        ),
    }
    report = evaluate_environment_validity(observations, policy=policy)
    report["captured_at"] = datetime.now(UTC).isoformat()
    report["collector"] = {
        "name": "groundupscale-local-environment",
        "version": "v1alpha1",
        "privacy": (
            "allowlisted fields only; no command arguments, environment variables, "
            "or unrestricted process paths"
        ),
    }
    return report


__all__ = [
    "ENVIRONMENT_VALIDITY_SCHEMA",
    "EnvironmentValidityPolicy",
    "collect_environment_validity",
    "evaluate_environment_validity",
]
