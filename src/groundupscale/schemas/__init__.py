"""Versioned, strict schemas for all human-authored GroundUpScale specs."""

from groundupscale.schemas.v1alpha1 import (
    AnalysisCaseDocument,
    AnalysisPlanDocument,
    BenchmarkCaseDocument,
    DeploymentIntentDocument,
    FabricGraphDocument,
    HardwareSpecDocument,
    HardwareBenchmarkSuiteDocument,
    HardwareCapabilityProfileDocument,
    ModelSpecDocument,
    SpecDocument,
    WorkloadSpecDocument,
    document_schema,
)

__all__ = [
    "AnalysisCaseDocument",
    "AnalysisPlanDocument",
    "BenchmarkCaseDocument",
    "DeploymentIntentDocument",
    "FabricGraphDocument",
    "HardwareSpecDocument",
    "HardwareBenchmarkSuiteDocument",
    "HardwareCapabilityProfileDocument",
    "ModelSpecDocument",
    "SpecDocument",
    "WorkloadSpecDocument",
    "document_schema",
]
