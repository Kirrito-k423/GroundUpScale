"""Hardware backend registry for translating CostIR into implementation candidates."""

from collections.abc import Callable

from groundupscale.backends.apple_m4_cpu import compile_apple_m4_cpu_prediction
from groundupscale.backends.ascend_910b2 import compile_ascend_910b2_prediction
from groundupscale.ir import CostProgram, HardwareBackendPrediction
from groundupscale.specs import AnalysisBundle


BackendCompiler = Callable[
    [AnalysisBundle, CostProgram], HardwareBackendPrediction | None
]

_BACKEND_COMPILERS: tuple[BackendCompiler, ...] = (
    compile_apple_m4_cpu_prediction,
    compile_ascend_910b2_prediction,
)


def compile_hardware_prediction(
    bundle: AnalysisBundle, cost: CostProgram
) -> HardwareBackendPrediction | None:
    """Return the first compatible backend result from the explicit registry."""

    for compile_backend in _BACKEND_COMPILERS:
        prediction = compile_backend(bundle, cost)
        if prediction is not None:
            return prediction
    return None

__all__ = ["compile_hardware_prediction"]
