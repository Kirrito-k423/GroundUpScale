"""Reference execution and measurement interfaces."""

from groundupscale.benchmark.measurement import BenchmarkRunner
from groundupscale.benchmark.memory import observe_tensor_storage_peak
from groundupscale.benchmark.prediction import predict_live_set
from groundupscale.benchmark.trace import TraceRunner
from groundupscale.benchmark.reference import (
    AliasAudit,
    CorrectnessReport,
    DeviceExecutionAudit,
    DeviceRun,
    ReferenceConfig,
    ReferenceRunner,
    TwoLayerTransformer,
)

__all__ = [
    "AliasAudit",
    "BenchmarkRunner",
    "CorrectnessReport",
    "DeviceExecutionAudit",
    "DeviceRun",
    "ReferenceConfig",
    "ReferenceRunner",
    "TraceRunner",
    "TwoLayerTransformer",
    "observe_tensor_storage_peak",
    "predict_live_set",
]
