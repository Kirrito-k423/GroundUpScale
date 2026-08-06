"""SemanticIR to hardware-independent CostIR lowering."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Protocol

from groundupscale.ir.common import (
    DerivationRecord,
    canonical_json,
    content_fingerprint,
    derivation_identity,
    node_identity,
)
from groundupscale.ir.cost import (
    CostFormula,
    CostLoweringResult,
    CostMetrics,
    CostOperation,
    CostProgram,
    CostRegion,
    CostSummary,
)
from groundupscale.ir.semantic import (
    ProvenanceGraph,
    SemanticOperation,
    SemanticProgram,
    SemanticRegion,
    SemanticStateArtifact,
    SemanticTensorType,
    SemanticValue,
    ValidationResult,
)


COST_LOWERER_VERSION = "core.cost-lowerer/v1alpha1"
DTYPE_BYTES = {
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "int64": 8,
    "bool": 1,
}


class CostLoweringError(ValueError):
    """Semantic facts cannot be lowered under the registered cost rules."""


@dataclass(frozen=True)
class CostLoweringRequest:
    semantic_ir: SemanticProgram
    provenance: ProvenanceGraph


@dataclass(frozen=True)
class CostRuleContext:
    operation: SemanticOperation
    operands: tuple[SemanticValue, ...]
    results: tuple[SemanticValue, ...]


@dataclass(frozen=True)
class RuleEstimate:
    flops: int
    expression: str
    assumptions: tuple[str, ...]


class CostRule(Protocol):
    operation: str
    rule_id: str

    def estimate(self, context: CostRuleContext) -> RuleEstimate: ...


def _elements(tensor: SemanticTensorType) -> int:
    return prod(tensor.shape)


def _bytes(tensor: SemanticTensorType) -> int:
    try:
        width = DTYPE_BYTES[tensor.dtype]
    except KeyError as error:
        raise CostLoweringError(f"unsupported dtype for byte cost: {tensor.dtype}") from error
    return _elements(tensor) * width


@dataclass(frozen=True)
class _MatMulRule:
    operation: str = "MatMul"
    rule_id: str = "core.cost-rule.matmul/v1alpha1"

    def estimate(self, context: CostRuleContext) -> RuleEstimate:
        if len(context.operands) != 2 or len(context.results) != 1:
            raise CostLoweringError("MatMul requires two operands and one result")
        left = context.operands[0].tensor.shape
        right = context.operands[1].tensor.shape
        output = context.results[0].tensor.shape
        if len(left) < 2 or len(right) < 2 or len(output) < 2:
            raise CostLoweringError("MatMul tensors must have rank >= 2")
        m, n, k = output[-2], output[-1], left[-1]
        if right[-2] != k or right[-1] != n:
            raise CostLoweringError(
                f"MatMul shape mismatch: left={left}, right={right}, output={output}"
            )
        batch = prod(output[:-2]) if len(output) > 2 else 1
        flops = 2 * batch * m * n * k
        return RuleEstimate(
            flops=flops,
            expression=f"2 * {batch} * {m} * {n} * {k} = {flops}",
            assumptions=("one multiply and one add count as two FLOPs",),
        )


@dataclass(frozen=True)
class _ElementwiseRule:
    operation: str
    rule_id: str

    def estimate(self, context: CostRuleContext) -> RuleEstimate:
        if len(context.results) != 1:
            raise CostLoweringError(f"{self.operation} requires one result")
        elements = _elements(context.results[0].tensor)
        return RuleEstimate(
            flops=elements,
            expression=f"result_elements = {elements}",
            assumptions=(f"one {self.operation} arithmetic operation per result element",),
        )


@dataclass(frozen=True)
class _RMSNormRule:
    operation: str = "RMSNorm"
    rule_id: str = "core.cost-rule.rmsnorm/v1alpha1"

    def estimate(self, context: CostRuleContext) -> RuleEstimate:
        if not context.operands or len(context.results) != 1:
            raise CostLoweringError("RMSNorm requires an activation operand and one result")
        shape = context.operands[0].tensor.shape
        width = shape[-1]
        outer = prod(shape[:-1]) if len(shape) > 1 else 1
        flops = outer * (4 * width + 2)
        return RuleEstimate(
            flops=flops,
            expression=f"{outer} * (4 * {width} + 2) = {flops}",
            assumptions=(
                "per row: square H, reduce H-1, divide 1, epsilon add 1, rsqrt 1, two multiplies H",
                "rsqrt and divide each count as one equivalent FLOP",
            ),
        )


@dataclass(frozen=True)
class _SoftmaxRule:
    operation: str = "Softmax"
    rule_id: str = "core.cost-rule.softmax/v1alpha1"

    def estimate(self, context: CostRuleContext) -> RuleEstimate:
        if len(context.results) != 1:
            raise CostLoweringError("Softmax requires one result")
        shape = context.results[0].tensor.shape
        width = shape[-1]
        outer = prod(shape[:-1]) if len(shape) > 1 else 1
        flops = outer * (5 * width - 2)
        return RuleEstimate(
            flops=flops,
            expression=f"{outer} * (5 * {width} - 2) = {flops}",
            assumptions=(
                "per row: max N-1, subtract N, exp N, sum N-1, divide N",
                "comparison and exp each count as one equivalent FLOP",
            ),
        )


@dataclass(frozen=True)
class _SiLURule:
    operation: str = "SiLU"
    rule_id: str = "core.cost-rule.silu/v1alpha1"

    def estimate(self, context: CostRuleContext) -> RuleEstimate:
        if len(context.results) != 1:
            raise CostLoweringError("SiLU requires one result")
        elements = _elements(context.results[0].tensor)
        flops = 5 * elements
        return RuleEstimate(
            flops=flops,
            expression=f"5 * {elements} = {flops}",
            assumptions=("negate, exp, add, divide, multiply each count as one",),
        )


@dataclass(frozen=True)
class _AliasRule:
    operation: str
    rule_id: str = "core.cost-rule.alias/v1alpha1"

    def estimate(self, context: CostRuleContext) -> RuleEstimate:
        return RuleEstimate(
            flops=0,
            expression="0",
            assumptions=("View/Transpose changes metadata and aliases storage",),
        )


def default_cost_rules() -> tuple[CostRule, ...]:
    return (
        _MatMulRule(),
        _ElementwiseRule("Add", "core.cost-rule.add/v1alpha1"),
        _RMSNormRule(),
        _SoftmaxRule(),
        _SiLURule(),
        _ElementwiseRule("Mul", "core.cost-rule.mul/v1alpha1"),
        _AliasRule("View"),
        _AliasRule("Transpose"),
    )


class CostRuleRegistry:
    """Immutable operation-to-rule registry; duplicate operations are rejected."""

    def __init__(self, rules: tuple[CostRule, ...]) -> None:
        by_operation: dict[str, CostRule] = {}
        for rule in rules:
            if rule.operation in by_operation:
                raise CostLoweringError(f"duplicate CostRule for {rule.operation}")
            by_operation[rule.operation] = rule
        self._rules = by_operation
        self.identities = tuple(
            sorted((operation, rule.rule_id) for operation, rule in by_operation.items())
        )

    def resolve(self, operation: str) -> CostRule:
        try:
            return self._rules[operation]
        except KeyError as error:
            raise CostLoweringError(f"no CostRule registered for {operation}") from error


class CostLowerer:
    """Lower semantic operations and hierarchy without selecting hardware."""

    def __init__(self, rules: tuple[CostRule, ...] | None = None) -> None:
        self._registry = CostRuleRegistry(rules or default_cost_rules())
        self._fingerprint = ""
        self._values: dict[str, SemanticValue] = {}
        self._artifacts: dict[str, SemanticStateArtifact] = {}
        self._value_roles: dict[str, str] = {}
        self._effect_artifacts: dict[str, str] = {}
        self._cost_node_ids: dict[str, str] = {}
        self._records: list[DerivationRecord] = []

    def lower(self, request: CostLoweringRequest) -> CostLoweringResult:
        semantic = request.semantic_ir
        self._fingerprint = content_fingerprint(
            COST_LOWERER_VERSION,
            semantic.compilation_fingerprint,
            self._registry.identities,
        )
        self._values = {value.value_id: value for value in semantic.values}
        self._artifacts = {
            artifact.artifact_id: artifact for artifact in semantic.state_artifacts
        }
        self._records = list(request.provenance.records)
        self._effect_artifacts = {
            effect.effect_id: effect.artifact_id for effect in semantic.state_effects
        }
        self._value_roles = {}
        for effect in semantic.state_effects:
            if effect.output_value_id is not None:
                self._value_roles[effect.output_value_id] = self._artifacts[
                    effect.artifact_id
                ].role
        operations = tuple(semantic.walk_operations())
        self._cost_node_ids = {
            operation.node_id: node_identity(
                "cost-op", self._fingerprint, f"cost/{operation.stable_path}"
            )
            for operation in operations
        }
        root = self._lower_region(semantic.root)
        parameter_bytes = sum(
            _bytes(artifact.tensor)
            for artifact in semantic.state_artifacts
            if artifact.role == "parameter"
        )
        buffer_bytes = sum(
            _bytes(artifact.tensor)
            for artifact in semantic.state_artifacts
            if artifact.role == "buffer"
        )
        workload_artifact_bytes = sum(
            _bytes(artifact.tensor)
            for artifact in semantic.state_artifacts
            if artifact.source_kind == "workload_artifact"
        )
        critical_path_flops = self._critical_path_flops(tuple(root.walk_items()))
        program = CostProgram(
            schema="groundupscale.dev/cost-ir/v1alpha1",
            name=semantic.name,
            compilation_fingerprint=self._fingerprint,
            semantic_compilation_fingerprint=semantic.compilation_fingerprint,
            root=root,
            summary=CostSummary(
                metrics=root.metrics,
                parameter_bytes=parameter_bytes,
                buffer_bytes=buffer_bytes,
                workload_artifact_bytes=workload_artifact_bytes,
                serial_flops=root.metrics.flops,
                ideal_parallel_critical_path_flops=critical_path_flops,
            ),
        )
        validations = self._verify(program, len(operations))
        failed = [validation for validation in validations if not validation.passed]
        if failed:
            raise CostLoweringError(
                "; ".join(f"{item.check}: {item.detail}" for item in failed)
            )
        return CostLoweringResult(
            cost_ir=program,
            provenance=ProvenanceGraph(
                schema="groundupscale.dev/provenance-graph/v1alpha1",
                records=tuple(self._records),
            ),
            validation_results=validations,
            compilation_fingerprint=self._fingerprint,
        )

    def _record(
        self, *, rule: str, source: SemanticOperation | SemanticRegion, target_id: str
    ) -> tuple[str, ...]:
        derivation_id = derivation_identity(rule, self._fingerprint, source.stable_path)
        self._records.append(
            DerivationRecord(
                derivation_id=derivation_id,
                phase="cost-lower",
                rule=rule,
                source_path="SemanticIR",
                source_stable_path=source.stable_path,
                target_node_ids=(target_id,),
                assumptions=("hardware-independent logical cost",),
            )
        )
        return (derivation_id,)

    def _lower_region(self, region: SemanticRegion) -> CostRegion:
        items = tuple(
            self._lower_region(item)
            if isinstance(item, SemanticRegion)
            else self._lower_operation(item)
            for item in region.items
        )
        metrics = CostMetrics()
        for item in items:
            metrics = metrics + item.metrics
        stable_path = f"cost/{region.stable_path}"
        node_id = node_identity("cost-region", self._fingerprint, stable_path)
        derivations = self._record(
            rule=f"{COST_LOWERER_VERSION}:region-aggregate",
            source=region,
            target_id=node_id,
        )
        return CostRegion(
            local_id=region.local_id,
            kind=region.kind,
            semantic_node_id=region.node_id,
            definition_id=region.definition_id,
            stable_path=stable_path,
            node_id=node_id,
            items=items,
            metrics=metrics,
            derivation_ids=derivations,
        )

    @staticmethod
    def _unique(values: list[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(values))

    @staticmethod
    def _critical_path_flops(items: tuple[object, ...]) -> int:
        operations = {
            item.node_id: item for item in items if isinstance(item, CostOperation)
        }
        memo: dict[str, int] = {}
        visiting: set[str] = set()

        def longest(node_id: str) -> int:
            if node_id in memo:
                return memo[node_id]
            if node_id in visiting:
                raise CostLoweringError("cycle detected in CostIR dependencies")
            visiting.add(node_id)
            operation = operations[node_id]
            missing = [
                dependency
                for dependency in operation.dependency_cost_node_ids
                if dependency not in operations
            ]
            if missing:
                raise CostLoweringError(
                    f"CostIR dependency does not resolve for {operation.stable_path}: {missing}"
                )
            prefix = max(
                (longest(dependency) for dependency in operation.dependency_cost_node_ids),
                default=0,
            )
            visiting.remove(node_id)
            memo[node_id] = prefix + operation.metrics.flops
            return memo[node_id]

        return max((longest(node_id) for node_id in operations), default=0)

    def _lower_operation(self, operation: SemanticOperation) -> CostOperation:
        operands = tuple(self._values[value_id] for value_id in operation.operands)
        results = tuple(self._values[value_id] for value_id in operation.results)
        rule = self._registry.resolve(operation.operation)
        estimate = rule.estimate(
            CostRuleContext(operation=operation, operands=operands, results=results)
        )
        logical_read = sum(_bytes(value.tensor) for value in operands)
        logical_write = sum(_bytes(value.tensor) for value in results)
        parameter_read = sum(
            _bytes(value.tensor)
            for value in operands
            if self._value_roles.get(value.value_id) == "parameter"
        )
        buffer_read = sum(
            _bytes(value.tensor)
            for value in operands
            if self._value_roles.get(value.value_id) == "buffer"
        )
        activation_read = logical_read - parameter_read - buffer_read
        aliases = any(value.alias_of is not None for value in results)
        metrics = CostMetrics(
            flops=estimate.flops,
            logical_read_bytes=logical_read,
            logical_write_bytes=logical_write,
            materialized_read_bytes=0 if aliases else logical_read,
            materialized_write_bytes=0 if aliases else logical_write,
            parameter_read_bytes=parameter_read,
            buffer_read_bytes=buffer_read,
            activation_read_bytes=activation_read,
            explicit_activation_bytes=0 if aliases else logical_write,
            alias_result_bytes=logical_write if aliases else 0,
        )
        dependencies: list[str] = []
        for value in operands:
            if value.producer_id in self._cost_node_ids:
                dependencies.append(self._cost_node_ids[value.producer_id])
        state_artifacts = self._unique(
            [
                self._effect_artifacts[effect_id]
                for effect_id in operation.state_effect_ids
            ]
        )
        stable_path = f"cost/{operation.stable_path}"
        node_id = self._cost_node_ids[operation.node_id]
        derivations = self._record(rule=rule.rule_id, source=operation, target_id=node_id)
        return CostOperation(
            local_id=operation.local_id,
            operation=operation.operation,
            semantic_node_id=operation.node_id,
            definition_id=operation.definition_id,
            stable_path=stable_path,
            node_id=node_id,
            operands=operation.operands,
            results=operation.results,
            operand_types=tuple(value.tensor for value in operands),
            result_types=tuple(value.tensor for value in results),
            state_artifact_ids=state_artifacts,
            dependency_cost_node_ids=self._unique(dependencies),
            metrics=metrics,
            formula=CostFormula(
                rule_id=rule.rule_id,
                flops_expression=estimate.expression,
                logical_read_expression="sum(bytes(each logical operand tensor))",
                logical_write_expression="sum(bytes(each logical result tensor))",
                assumptions=estimate.assumptions
                + (
                    "logical bytes count tensor operands/results once and are not cache traffic",
                    "materialized bytes are zero for alias-only View/Transpose",
                ),
            ),
            derivation_ids=derivations,
        )

    def _verify(
        self, program: CostProgram, semantic_operation_count: int
    ) -> tuple[ValidationResult, ...]:
        operations = tuple(program.walk_operations())
        count_matches = len(operations) == semantic_operation_count
        nonnegative = all(
            all(value >= 0 for value in operation.metrics.__dict__.values())
            for operation in operations
        )
        alias_zero = all(
            operation.metrics.materialized_read_bytes == 0
            and operation.metrics.materialized_write_bytes == 0
            for operation in operations
            if operation.metrics.alias_result_bytes > 0
        )
        provenance = all(operation.derivation_ids for operation in operations)
        serialized = canonical_json(program)
        hardware_independent = all(
            marker not in serialized
            for marker in ("local-m4/cpu", "local-m4/gpu", '"latency"', '"duration"')
        )
        return (
            ValidationResult("operation-count", count_matches, "one Cost op per Semantic op"),
            ValidationResult("metrics-nonnegative", nonnegative, "all metrics are non-negative"),
            ValidationResult("alias-zero-materialization", alias_zero, "alias ops materialize zero bytes"),
            ValidationResult("provenance-complete", provenance, "all Cost ops have derivation"),
            ValidationResult("hardware-independent", hardware_independent, "no device or duration fact present"),
        )
