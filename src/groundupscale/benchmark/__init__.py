"""Reference execution and measurement interfaces."""

from groundupscale.benchmark.comparison import (
    build_prediction_observation_comparison,
)
from groundupscale.benchmark.measurement import BenchmarkRunner
from groundupscale.benchmark.hardware_microbenchmark import (
    CapabilityAggregationError,
    HardwareMicrobenchmarkRunner,
    aggregate_capability_envelope,
)
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
    "CapabilityAggregationError",
    "CorrectnessReport",
    "DeviceExecutionAudit",
    "DeviceRun",
    "ReferenceConfig",
    "ReferenceRunner",
    "TraceRunner",
    "TwoLayerTransformer",
    "HardwareMicrobenchmarkRunner",
    "aggregate_capability_envelope",
    "build_prediction_observation_comparison",
    "observe_tensor_storage_peak",
    "predict_live_set",
]
