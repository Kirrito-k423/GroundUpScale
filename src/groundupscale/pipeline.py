"""Public orchestration seam for deterministic AnalysisPlan compilation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from groundupscale.backends import compile_hardware_prediction
from groundupscale.compiler import (
    CompilationContext,
    CostLowerer,
    CostLoweringRequest,
    ModelBuilder,
    SemanticCompileRequest,
    SemanticCompiler,
    WorkloadBuilder,
    semantic_deployment_plan,
)
from groundupscale.ir import (
    CostLoweringResult,
    HardwareBackendPrediction,
    ModelIR,
    SemanticCompilationResult,
)
from groundupscale.ir import WorkloadIR
from groundupscale.specs import AnalysisBundle, SpecRepository


@dataclass(frozen=True)
class CompiledAnalysis:
    bundle: AnalysisBundle
    models: tuple[ModelIR, ...]
    workload: WorkloadIR
    semantic: SemanticCompilationResult
    cost: CostLoweringResult
    hardware_prediction: HardwareBackendPrediction | None


def compile_analysis_bundle(bundle: AnalysisBundle) -> CompiledAnalysis:
    models = tuple(
        ModelBuilder().build(document)
        for _, document in sorted(bundle.models.items())
    )
    workload = WorkloadBuilder().build(
        bundle.workload, models_by_reference=bundle.models_by_reference
    )
    semantic = SemanticCompiler().compile(
        SemanticCompileRequest(
            workload=workload,
            models=models,
            analysis_case=bundle.analysis_case,
            deployment=semantic_deployment_plan(bundle.deployment_intent),
            context=CompilationContext(
                compiler_version="0.1.0", plugin_versions=()
            ),
        )
    )
    cost = CostLowerer().lower(
        CostLoweringRequest(
            semantic_ir=semantic.semantic_ir,
            provenance=semantic.provenance,
        )
    )
    hardware_prediction = compile_hardware_prediction(bundle, cost.cost_ir)
    return CompiledAnalysis(
        bundle=bundle,
        models=models,
        workload=workload,
        semantic=semantic,
        cost=cost,
        hardware_prediction=hardware_prediction,
    )


def compile_analysis_plan(
    repository_root: str | Path, plan: str | Path
) -> CompiledAnalysis:
    root = Path(repository_root).resolve()
    bundle = SpecRepository(root).load_analysis_plan(Path(plan))
    return compile_analysis_bundle(bundle)
