"""Explicit registry for hardware Measurement Adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from groundupscale.measurement_contract import MeasurementAdapter


AdapterFactory = Callable[..., MeasurementAdapter]


def _ascend_npu_factory(**configuration: Any) -> MeasurementAdapter:
    from groundupscale.measurement_adapters.ascend_npu import (
        AscendNpuMeasurementAdapter,
    )

    return AscendNpuMeasurementAdapter(**configuration)


_ADAPTER_FACTORIES: dict[str, AdapterFactory] = {
    "ascend-npu": _ascend_npu_factory,
}


def available_measurement_devices() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTER_FACTORIES))


def create_measurement_adapter(
    device: str, **configuration: Any
) -> MeasurementAdapter:
    try:
        factory = _ADAPTER_FACTORIES[device]
    except KeyError as error:
        raise ValueError(f"unsupported measurement device: {device}") from error
    return factory(**configuration)


__all__ = [
    "available_measurement_devices",
    "create_measurement_adapter",
]
