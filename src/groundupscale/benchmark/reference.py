"""Real two-layer causal Transformer matching the frozen SemanticIR."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from typing import Any, cast

import torch
from torch import Tensor, nn

from groundupscale.schemas.v1alpha1 import ModuleRepeatSpec
from groundupscale.specs import AnalysisBundle
from groundupscale.execution_runtime import ExecutionRuntime


@dataclass(frozen=True)
class ReferenceConfig:
    batch_size: int
    sequence_length: int
    hidden_size: int
    heads: int
    head_dim: int
    intermediate_size: int
    layers: int
    model_root: str
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_analysis_bundle(cls, bundle: AnalysisBundle) -> ReferenceConfig:
        bindings = bundle.analysis_case.spec.shape.bindings
        if bundle.analysis_case.spec.shape.dtype != "float32":
            raise ValueError("reference slice currently requires float32")
        if len(bundle.models) != 1:
            raise ValueError("reference slice requires exactly one model")
        model = next(iter(bundle.models.values()))
        repeats = [
            child
            for child in model.spec.root.children
            if isinstance(child, ModuleRepeatSpec) and child.id == "layers"
        ]
        if len(repeats) != 1:
            raise ValueError("model must contain one layers repeat")
        workload_root = bundle.workload.spec.root
        if workload_root.kind != "sequence" or len(workload_root.children) != 1:
            raise ValueError("reference slice requires one ModelCall in a Sequence")
        model_call = workload_root.children[0]
        if model_call.kind != "model_call":
            raise ValueError("reference slice requires a ModelCall leaf")
        return cls(
            batch_size=bindings["B"],
            sequence_length=bindings["S"],
            hidden_size=bindings["H"],
            heads=bindings["NH"],
            head_dim=bindings["D"],
            intermediate_size=bindings["I"],
            layers=repeats[0].count,
            model_root=(
                f"semantic/workload/{bundle.workload.metadata.name}/"
                f"{workload_root.id}/{model_call.id}/model/{model.spec.root.id}"
            ),
        )


class SemanticLeaf(nn.Module):
    def __init__(self, stable_path: str, operation: str) -> None:
        super().__init__()
        self.stable_path = stable_path
        self.operation = operation


class StateMatMul(SemanticLeaf):
    def __init__(
        self,
        stable_path: str,
        input_size: int,
        output_size: int,
        generator: torch.Generator,
    ) -> None:
        super().__init__(stable_path, "MatMul")
        weight = torch.randn(
            input_size,
            output_size,
            generator=generator,
            dtype=torch.float32,
        ) * 0.02
        self.weight = nn.Parameter(weight)

    def forward(self, x: Tensor) -> Tensor:
        return torch.matmul(x, self.weight)


class TensorMatMul(SemanticLeaf):
    def __init__(self, stable_path: str, equation: str | None = None) -> None:
        super().__init__(stable_path, "MatMul")
        self.equation = equation

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if self.equation is None:
            return torch.matmul(left, right)
        if self.equation == "bhqk,bhkd->bqhd":
            return torch.matmul(left, right).transpose(1, 2).contiguous()
        return torch.einsum(self.equation, left, right)


class AddOp(SemanticLeaf):
    def __init__(self, stable_path: str) -> None:
        super().__init__(stable_path, "Add")

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return torch.add(left, right)


class CausalMaskAdd(SemanticLeaf):
    def __init__(self, stable_path: str, batch: int, sequence: int) -> None:
        super().__init__(stable_path, "Add")
        mask = torch.zeros(batch, 1, sequence, sequence, dtype=torch.float32)
        upper = torch.triu(
            torch.ones(sequence, sequence, dtype=torch.bool), diagonal=1
        )
        mask.masked_fill_(upper, float("-inf"))
        self.register_buffer("mask", mask, persistent=True)

    def forward(self, scores: Tensor) -> Tensor:
        return torch.add(scores, cast(Tensor, self.mask))


class RMSNormOp(SemanticLeaf):
    def __init__(self, stable_path: str, hidden_size: int, epsilon: float) -> None:
        super().__init__(stable_path, "RMSNorm")
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        variance = x.square().mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.epsilon) * self.weight


class SoftmaxOp(SemanticLeaf):
    def __init__(self, stable_path: str, dimension: int) -> None:
        super().__init__(stable_path, "Softmax")
        self.dimension = dimension

    def forward(self, x: Tensor) -> Tensor:
        return torch.softmax(x, dim=self.dimension)


class SiLUOp(SemanticLeaf):
    def __init__(self, stable_path: str) -> None:
        super().__init__(stable_path, "SiLU")

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.silu(x)


class MulOp(SemanticLeaf):
    def __init__(self, stable_path: str, scalar: float | None = None) -> None:
        super().__init__(stable_path, "Mul")
        self.scalar = scalar

    def forward(self, left: Tensor, right: Tensor | None = None) -> Tensor:
        multiplier: Tensor | float
        if right is not None:
            multiplier = right
        elif self.scalar is not None:
            multiplier = self.scalar
        else:
            raise ValueError("Mul requires a Tensor operand or configured scalar")
        return torch.mul(left, multiplier)


class ViewOp(SemanticLeaf):
    def __init__(self, stable_path: str, target_shape: tuple[int, ...]) -> None:
        super().__init__(stable_path, "View")
        self.target_shape = target_shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(self.target_shape)


class TransposeOp(SemanticLeaf):
    def __init__(self, stable_path: str, first: int, second: int) -> None:
        super().__init__(stable_path, "Transpose")
        self.first = first
        self.second = second

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(self.first, self.second)


class Attention(nn.Module):
    def __init__(
        self, config: ReferenceConfig, layer_path: str, generator: torch.Generator
    ) -> None:
        super().__init__()
        self.stable_path = f"{layer_path}/attention"
        hidden = config.hidden_size
        self.q_proj = StateMatMul(f"{self.stable_path}/q_proj", hidden, hidden, generator)
        self.k_proj = StateMatMul(f"{self.stable_path}/k_proj", hidden, hidden, generator)
        self.v_proj = StateMatMul(f"{self.stable_path}/v_proj", hidden, hidden, generator)
        heads_shape = (
            config.batch_size,
            config.sequence_length,
            config.heads,
            config.head_dim,
        )
        self.q_view = ViewOp(f"{self.stable_path}/q_view", heads_shape)
        self.q_transpose = TransposeOp(f"{self.stable_path}/q_transpose", 1, 2)
        self.k_view = ViewOp(f"{self.stable_path}/k_view", heads_shape)
        self.k_transpose = TransposeOp(f"{self.stable_path}/k_transpose", 1, 2)
        self.k_key_transpose = TransposeOp(
            f"{self.stable_path}/k_key_transpose", 2, 3
        )
        self.qk_matmul = TensorMatMul(f"{self.stable_path}/qk_matmul")
        self.scale = MulOp(
            f"{self.stable_path}/scale", scalar=config.head_dim**-0.5
        )
        self.causal_mask = CausalMaskAdd(
            f"{self.stable_path}/causal_mask",
            config.batch_size,
            config.sequence_length,
        )
        self.softmax = SoftmaxOp(f"{self.stable_path}/softmax", -1)
        self.v_view = ViewOp(f"{self.stable_path}/v_view", heads_shape)
        self.v_transpose = TransposeOp(f"{self.stable_path}/v_transpose", 1, 2)
        self.context_matmul = TensorMatMul(
            f"{self.stable_path}/context_matmul", "bhqk,bhkd->bqhd"
        )
        self.context_view = ViewOp(
            f"{self.stable_path}/context_view",
            (
                config.batch_size,
                config.sequence_length,
                config.hidden_size,
            ),
        )
        self.out_proj = StateMatMul(
            f"{self.stable_path}/out_proj", hidden, hidden, generator
        )

    def forward(self, hidden: Tensor) -> Tensor:
        q = self.q_proj(hidden)
        k = self.k_proj(hidden)
        v = self.v_proj(hidden)
        q_heads = self.q_transpose(self.q_view(q))
        k_heads = self.k_transpose(self.k_view(k))
        k_keys = self.k_key_transpose(k_heads)
        scores = self.qk_matmul(q_heads, k_keys)
        masked_scores = self.causal_mask(self.scale(scores))
        probabilities = self.softmax(masked_scores)
        v_heads = self.v_transpose(self.v_view(v))
        context_heads = self.context_matmul(probabilities, v_heads)
        context = self.context_view(context_heads)
        return self.out_proj(context)


class MLP(nn.Module):
    def __init__(
        self, config: ReferenceConfig, layer_path: str, generator: torch.Generator
    ) -> None:
        super().__init__()
        self.stable_path = f"{layer_path}/mlp"
        self.gate_proj = StateMatMul(
            f"{self.stable_path}/gate_proj",
            config.hidden_size,
            config.intermediate_size,
            generator,
        )
        self.up_proj = StateMatMul(
            f"{self.stable_path}/up_proj",
            config.hidden_size,
            config.intermediate_size,
            generator,
        )
        self.silu = SiLUOp(f"{self.stable_path}/silu")
        self.gate_mul = MulOp(f"{self.stable_path}/gate_mul")
        self.down_proj = StateMatMul(
            f"{self.stable_path}/down_proj",
            config.intermediate_size,
            config.hidden_size,
            generator,
        )

    def forward(self, hidden: Tensor) -> Tensor:
        gate = self.gate_proj(hidden)
        up = self.up_proj(hidden)
        return self.down_proj(self.gate_mul(self.silu(gate), up))


class TransformerLayer(nn.Module):
    def __init__(
        self, config: ReferenceConfig, index: int, generator: torch.Generator
    ) -> None:
        super().__init__()
        self.stable_path = f"{config.model_root}/layer_{index}"
        self.input_norm = RMSNormOp(
            f"{self.stable_path}/input_norm", config.hidden_size, 1e-6
        )
        self.attention = Attention(config, self.stable_path, generator)
        self.residual_1 = AddOp(f"{self.stable_path}/residual_1")
        self.post_norm = RMSNormOp(
            f"{self.stable_path}/post_norm", config.hidden_size, 1e-6
        )
        self.mlp = MLP(config, self.stable_path, generator)
        self.residual_2 = AddOp(f"{self.stable_path}/residual_2")

    def forward(self, hidden: Tensor) -> Tensor:
        attention_output = self.attention(self.input_norm(hidden))
        post_attention = self.residual_1(hidden, attention_output)
        mlp_output = self.mlp(self.post_norm(post_attention))
        return self.residual_2(post_attention, mlp_output)


class TwoLayerTransformer(nn.Module):
    def __init__(self, config: ReferenceConfig, seed: int) -> None:
        super().__init__()
        if config.hidden_size != config.heads * config.head_dim:
            raise ValueError("hidden_size must equal heads * head_dim")
        self.config = config
        self.stable_path = config.model_root
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.layers = nn.ModuleList(
            TransformerLayer(config, index, generator)
            for index in range(config.layers)
        )

    def forward(self, hidden: Tensor) -> Tensor:
        expected = (
            self.config.batch_size,
            self.config.sequence_length,
            self.config.hidden_size,
        )
        if tuple(hidden.shape) != expected:
            raise ValueError(f"expected input shape {expected}, found {tuple(hidden.shape)}")
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


@dataclass(frozen=True)
class TensorExecutionContract:
    device: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    is_contiguous: bool


@dataclass(frozen=True)
class AliasTensorExecutionContract:
    device: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    is_contiguous: bool
    layout: str


@dataclass(frozen=True)
class AliasAudit:
    stable_path: str
    operation: str
    aliases_input_storage: bool
    selected_candidate_id: str
    candidate_selection_evidence: CandidateSelectionEvidence
    input_storage_identity: str
    output_storage_identity: str
    input_contract: AliasTensorExecutionContract
    output_contract: AliasTensorExecutionContract


@dataclass(frozen=True)
class CandidateSelectionEvidence:
    kind: str
    evidence_ref: str


@dataclass(frozen=True)
class DeviceExecutionAudit:
    requested_device: str
    input_device: str
    output_device: str
    parameter_devices: tuple[str, ...]
    buffer_devices: tuple[str, ...]
    leaf_output_devices: tuple[tuple[str, str], ...]
    input_contract: TensorExecutionContract
    output_contract: TensorExecutionContract
    leaf_output_contracts: tuple[tuple[str, TensorExecutionContract], ...]
    alias_checks: tuple[AliasAudit, ...]
    semantic_leaf_count: int
    parameter_bytes: int
    buffer_bytes: int
    fallback_enabled: bool


@dataclass(frozen=True, eq=False)
class DeviceRun:
    output: Tensor
    output_sha256: str
    audit: DeviceExecutionAudit


@dataclass(frozen=True, eq=False)
class CorrectnessReport:
    cpu: DeviceRun
    mps: DeviceRun
    passed: bool
    max_absolute_error: float
    max_relative_error: float
    atol: float
    rtol: float


class ReferenceRunner:
    def __init__(self, config: ReferenceConfig, seed: int = 20260806) -> None:
        self.config = config
        self.seed = seed

    @classmethod
    def from_analysis_bundle(
        cls, bundle: AnalysisBundle, seed: int = 20260806
    ) -> ReferenceRunner:
        return cls(ReferenceConfig.from_analysis_bundle(bundle), seed=seed)

    def _input(self) -> Tensor:
        generator = torch.Generator(device="cpu").manual_seed(self.seed + 1)
        return torch.randn(
            self.config.batch_size,
            self.config.sequence_length,
            self.config.hidden_size,
            generator=generator,
            dtype=self.config.dtype,
        )

    @staticmethod
    def _tensor_sha256(tensor: Tensor) -> str:
        data = tensor.detach().cpu().contiguous().numpy().tobytes()
        return sha256(data).hexdigest()

    def run_device(
        self,
        device: str,
        *,
        execution_runtime: ExecutionRuntime | None = None,
        lane: str = "correctness",
    ) -> DeviceRun:
        if device not in {"cpu", "mps"} and not device.startswith("npu:"):
            raise ValueError(f"unsupported reference device: {device}")
        if device.startswith("npu:") and execution_runtime is None:
            raise RuntimeError("NPU correctness requires an explicit ExecutionRuntime")
        fallback_enabled = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        if device == "mps":
            if fallback_enabled:
                raise RuntimeError("refusing MPS correctness run with fallback enabled")
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS is not available")
        model = TwoLayerTransformer(self.config, self.seed)
        hidden = self._input()
        if execution_runtime is not None:
            model = cast(
                TwoLayerTransformer,
                execution_runtime.prepare_model(model, lane=lane).eval(),
            )
            hidden = execution_runtime.prepare_tensor(
                hidden, lane=lane, role="input"
            )
        else:
            target = torch.device(device)
            model = model.to(target).eval()
            hidden = hidden.to(target)
        leaf_devices: dict[str, str] = {}
        leaf_contracts: dict[str, TensorExecutionContract] = {}
        alias_checks: list[AliasAudit] = []
        handles: list[Any] = []

        def tensor_contract(tensor: Tensor) -> TensorExecutionContract:
            return TensorExecutionContract(
                device=(
                    execution_runtime.tensor_device(tensor)
                    if execution_runtime is not None
                    else str(tensor.device)
                ),
                dtype=str(tensor.dtype).removeprefix("torch."),
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                is_contiguous=tensor.is_contiguous(),
            )

        def alias_tensor_contract(tensor: Tensor) -> AliasTensorExecutionContract:
            contract = tensor_contract(tensor)
            return AliasTensorExecutionContract(
                device=contract.device,
                dtype=contract.dtype,
                shape=contract.shape,
                stride=contract.stride,
                is_contiguous=contract.is_contiguous,
                layout=("contiguous" if contract.is_contiguous else "strided"),
            )

        def storage_identity(tensor: Tensor) -> str:
            storage = tensor.untyped_storage()
            return f"{tensor_contract(tensor).device}:{storage.data_ptr()}:{storage.nbytes()}"

        def audit_hook(module: SemanticLeaf, inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, Tensor):
                raise RuntimeError(f"semantic leaf {module.stable_path} returned non-Tensor")
            leaf_devices[module.stable_path] = (
                execution_runtime.tensor_device(output)
                if execution_runtime is not None
                else str(output.device)
            )
            leaf_contracts[module.stable_path] = tensor_contract(output)
            if module.operation in {"View", "Transpose"}:
                input_tensor = inputs[0]
                if not isinstance(input_tensor, Tensor):
                    raise RuntimeError(f"alias op {module.stable_path} has non-Tensor input")
                aliases = (
                    input_tensor.untyped_storage().data_ptr()
                    == output.untyped_storage().data_ptr()
                )
                alias_checks.append(
                    AliasAudit(
                        stable_path=module.stable_path,
                        operation=module.operation,
                        aliases_input_storage=aliases,
                        selected_candidate_id=(
                            f"runtime-candidate:{module.operation.casefold()}:"
                            f"{device}:eager"
                        ),
                        candidate_selection_evidence=CandidateSelectionEvidence(
                            kind="executed-runtime-leaf",
                            evidence_ref=(
                                f"correctness-observation://{module.stable_path}"
                            ),
                        ),
                        input_storage_identity=storage_identity(input_tensor),
                        output_storage_identity=storage_identity(output),
                        input_contract=alias_tensor_contract(input_tensor),
                        output_contract=alias_tensor_contract(output),
                    )
                )

        for module in model.modules():
            if isinstance(module, SemanticLeaf):
                handles.append(module.register_forward_hook(audit_hook))
        try:
            with torch.inference_mode():
                output = (
                    execution_runtime.execute_checked(lambda: model(hidden))
                    if execution_runtime is not None
                    else model(hidden)
                )
                if execution_runtime is not None:
                    execution_runtime.synchronize()
                elif device == "mps":
                    torch.mps.synchronize()
        finally:
            for handle in handles:
                handle.remove()
        output_device = (
            execution_runtime.tensor_device(output)
            if execution_runtime is not None
            else str(output.device)
        )
        cpu_output = (
            execution_runtime.copy_to_cpu(output, lane=lane, role="output")
            if execution_runtime is not None
            else output.detach().cpu()
        )
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        buffer_bytes = sum(
            buffer.numel() * buffer.element_size() for buffer in model.buffers()
        )
        semantic_leaf_count = sum(
            isinstance(module, SemanticLeaf) for module in model.modules()
        )
        audit = DeviceExecutionAudit(
            requested_device=device,
            input_device=(
                execution_runtime.tensor_device(hidden)
                if execution_runtime is not None
                else str(hidden.device)
            ),
            output_device=output_device,
            parameter_devices=tuple(
                sorted(
                    {
                        execution_runtime.tensor_device(parameter)
                        if execution_runtime is not None
                        else str(parameter.device)
                        for parameter in model.parameters()
                    }
                )
            ),
            buffer_devices=tuple(
                sorted(
                    {
                        execution_runtime.tensor_device(buffer)
                        if execution_runtime is not None
                        else str(buffer.device)
                        for buffer in model.buffers()
                    }
                )
            ),
            leaf_output_devices=tuple(sorted(leaf_devices.items())),
            input_contract=tensor_contract(hidden),
            output_contract=tensor_contract(output),
            leaf_output_contracts=tuple(sorted(leaf_contracts.items())),
            alias_checks=tuple(alias_checks),
            semantic_leaf_count=semantic_leaf_count,
            parameter_bytes=parameter_bytes,
            buffer_bytes=buffer_bytes,
            fallback_enabled=(
                fallback_enabled
                if execution_runtime is None
                else any(
                    execution_runtime.tensor_device_type(value) != "npu"
                    for value in (
                        hidden,
                        output,
                        *model.parameters(),
                        *model.buffers(),
                    )
                )
            ),
        )
        return DeviceRun(
            output=cpu_output,
            output_sha256=self._tensor_sha256(cpu_output),
            audit=audit,
        )

    def compare_cpu_mps(self, *, atol: float, rtol: float) -> CorrectnessReport:
        cpu = self.run_device("cpu")
        mps = self.run_device("mps")
        absolute = (cpu.output - mps.output).abs()
        significant = cpu.output.abs() >= (atol / rtol if rtol > 0 else float("inf"))
        relative = torch.zeros_like(absolute)
        relative[significant] = absolute[significant] / cpu.output[significant].abs()
        return CorrectnessReport(
            cpu=cpu,
            mps=mps,
            passed=bool(torch.allclose(cpu.output, mps.output, atol=atol, rtol=rtol)),
            max_absolute_error=float(absolute.max().item()),
            max_relative_error=float(relative.max().item()),
            atol=atol,
            rtol=rtol,
        )

    def compare_cpu_target(
        self,
        execution_runtime: ExecutionRuntime,
        *,
        atol: float,
        rtol: float,
    ) -> CorrectnessReport:
        cpu = self.run_device("cpu")
        target = self.run_device(
            execution_runtime.logical_device,
            execution_runtime=execution_runtime,
        )
        cpu_leaf_contracts = dict(cpu.audit.leaf_output_contracts)
        target_leaf_contracts = dict(target.audit.leaf_output_contracts)
        if (
            cpu.audit.semantic_leaf_count != target.audit.semantic_leaf_count
            or cpu_leaf_contracts.keys() != target_leaf_contracts.keys()
        ):
            raise RuntimeError("semantic-operation-coverage-failed")

        def layout_signature(
            contract: TensorExecutionContract,
        ) -> tuple[str, tuple[int, ...], tuple[int, ...], bool]:
            return (
                contract.dtype,
                contract.shape,
                contract.stride,
                contract.is_contiguous,
            )

        contracts_match = (
            layout_signature(cpu.audit.input_contract)
            == layout_signature(target.audit.input_contract)
            and layout_signature(cpu.audit.output_contract)
            == layout_signature(target.audit.output_contract)
            and all(
                layout_signature(cpu_leaf_contracts[path])
                == layout_signature(target_leaf_contracts[path])
                for path in cpu_leaf_contracts
            )
        )
        if not contracts_match:
            raise RuntimeError("dtype-layout-substitution-detected")
        if target.audit.fallback_enabled:
            raise RuntimeError("cpu-fallback-detected")
        absolute = (cpu.output - target.output).abs()
        significant = cpu.output.abs() >= (
            atol / rtol if rtol > 0 else float("inf")
        )
        relative = torch.zeros_like(absolute)
        relative[significant] = absolute[significant] / cpu.output[
            significant
        ].abs()
        return CorrectnessReport(
            cpu=cpu,
            mps=target,
            passed=bool(
                torch.allclose(cpu.output, target.output, atol=atol, rtol=rtol)
            ),
            max_absolute_error=float(absolute.max().item()),
            max_relative_error=float(relative.max().item()),
            atol=atol,
            rtol=rtol,
        )
