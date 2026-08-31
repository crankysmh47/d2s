"""Parameter-preserving Qwen3.5 text-FFN conversion."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

from ppmoe.adapters.qwen35_runtime import iter_text_mlps


@dataclass(frozen=True, slots=True)
class ConversionRecord:
    layer_index: int
    seed: int
    permutation: tuple[int, ...]
    expert_original_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class ConversionProbeEvidence:
    layer_index: int
    seed: int
    tokens: int
    max_abs_difference: float


@dataclass(frozen=True, slots=True)
class ConversionManifest:
    converted_layers: int
    num_experts: int
    expert_width: int
    parameter_count_before: int
    parameter_count_after: int
    records: tuple[ConversionRecord, ...]
    probe: ConversionProbeEvidence


def probe_qwen35_ffn_conversion(
    model: Qwen3_5ForConditionalGeneration,
    *,
    num_experts: int,
    seed_base: int,
    tokens: int = 16,
) -> ConversionProbeEvidence:
    """Check the layer-zero all-expert reconstruction using CPU FP32 weights."""

    if num_experts <= 0 or tokens <= 0:
        raise ValueError("num_experts and tokens must be positive")
    first_layer, dense = next(iter_text_mlps(model), (None, None))
    if first_layer is None or not isinstance(dense, Qwen3_5MLP):
        raise TypeError("layer zero MLP must be Qwen3_5MLP")
    intermediate_size, hidden_size = dense.gate_proj.weight.shape
    if intermediate_size % num_experts:
        raise ValueError("intermediate size must be divisible by num_experts")
    generator = torch.Generator(device="cpu").manual_seed(seed_base)
    inputs = torch.randn(tokens, hidden_size, generator=generator, dtype=torch.float32)
    gate_weight = dense.gate_proj.weight.detach().to(device="cpu", dtype=torch.float32)
    up_weight = dense.up_proj.weight.detach().to(device="cpu", dtype=torch.float32)
    down_weight = dense.down_proj.weight.detach().to(device="cpu", dtype=torch.float32)
    gate_bias = None if dense.gate_proj.bias is None else dense.gate_proj.bias.detach().to(device="cpu", dtype=torch.float32)
    up_bias = None if dense.up_proj.bias is None else dense.up_proj.bias.detach().to(device="cpu", dtype=torch.float32)
    down_bias = None if dense.down_proj.bias is None else dense.down_proj.bias.detach().to(device="cpu", dtype=torch.float32)
    activated = dense.act_fn(F.linear(inputs, gate_weight, gate_bias)) * F.linear(inputs, up_weight, up_bias)
    dense_output = F.linear(activated, down_weight, down_bias)
    permutation = torch.randperm(intermediate_size, generator=torch.Generator(device="cpu").manual_seed(seed_base))
    permuted_activated = activated.index_select(-1, permutation)
    permuted_down = down_weight.index_select(1, permutation)
    expert_width = intermediate_size // num_experts
    all_expert = sum(
        (
            F.linear(
                permuted_activated[..., start:start + expert_width],
                permuted_down[:, start:start + expert_width],
            )
            for start in range(0, intermediate_size, expert_width)
        ),
        torch.zeros_like(dense_output),
    )
    if down_bias is not None:
        all_expert = all_expert + down_bias
    return ConversionProbeEvidence(
        layer_index=0,
        seed=seed_base,
        tokens=tokens,
        max_abs_difference=float((dense_output - all_expert).abs().max().item()),
    )


class PartitionedQwen35MLP(nn.Module):
    """Physically permuted Qwen MLP with exact expert contribution capture."""

    def __init__(
        self,
        dense: Qwen3_5MLP,
        *,
        permutation: Tensor,
        num_experts: int,
    ) -> None:
        super().__init__()
        if any(
            projection.bias is not None
            for projection in (dense.gate_proj, dense.up_proj, dense.down_proj)
        ):
            raise ValueError("Qwen3.5 Phase 1 conversion requires bias-free MLPs")
        intermediate_size, hidden_size = dense.gate_proj.weight.shape
        if intermediate_size % num_experts:
            raise ValueError("intermediate size must be divisible by num_experts")
        if tuple(permutation.shape) != (intermediate_size,):
            raise ValueError("permutation has the wrong length")
        permutation = permutation.to(device=dense.gate_proj.weight.device, dtype=torch.long)
        if not torch.equal(
            permutation.sort().values,
            torch.arange(intermediate_size, device=permutation.device),
        ):
            raise ValueError("permutation must contain every neuron exactly once")
        factory = {
            "device": dense.gate_proj.weight.device,
            "dtype": dense.gate_proj.weight.dtype,
        }
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.expert_width = intermediate_size // num_experts
        self.config = dense.config
        self.act_fn = dense.act_fn
        self._output_transform: Callable[[Tensor, Tensor], Tensor] | None = None
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, **factory)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, **factory)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, **factory)
        self.register_buffer("permutation", permutation.detach().clone(), persistent=False)
        with torch.no_grad():
            self.gate_proj.weight.copy_(dense.gate_proj.weight.index_select(0, permutation))
            self.up_proj.weight.copy_(dense.up_proj.weight.index_select(0, permutation))
            self.down_proj.weight.copy_(dense.down_proj.weight.index_select(1, permutation))

    def expert_outputs(self, x: Tensor) -> Tensor:
        activated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        activated = activated.reshape(
            *activated.shape[:-1],
            self.num_experts,
            self.expert_width,
        )
        outputs = []
        for expert_index in range(self.num_experts):
            start = expert_index * self.expert_width
            stop = start + self.expert_width
            outputs.append(
                F.linear(
                    activated[..., expert_index, :],
                    self.down_proj.weight[:, start:stop],
                )
            )
        return torch.stack(outputs, dim=-2)

    @contextmanager
    def output_transform(
        self,
        transform: Callable[[Tensor, Tensor], Tensor],
    ) -> Iterator[None]:
        """Temporarily replace all-expert output without changing model state."""

        if self._output_transform is not None:
            raise RuntimeError("an output transform is already active")
        self._output_transform = transform
        try:
            yield
        finally:
            self._output_transform = None

    def forward(
        self,
        x: Tensor,
        selected_experts: Sequence[int] | Tensor | None = None,
    ) -> Tensor:
        outputs = self.expert_outputs(x)
        if selected_experts is None:
            if self._output_transform is not None:
                transformed = self._output_transform(x, outputs)
                if not isinstance(transformed, Tensor):
                    raise TypeError("output transform must return a tensor")
                if transformed.shape != x.shape:
                    raise ValueError("output transform returned a tensor with the wrong shape")
                if transformed.device != x.device:
                    raise ValueError("output transform returned a tensor on the wrong device")
                if transformed.dtype != x.dtype:
                    raise ValueError("output transform returned a tensor with the wrong dtype")
                return transformed
            return outputs.sum(dim=-2)
        if isinstance(selected_experts, Tensor):
            selections = selected_experts.to(device=outputs.device, dtype=torch.long)
            if tuple(selections.shape[:-1]) != tuple(outputs.shape[:-2]):
                raise ValueError("token selections must match input leading dimensions")
            if selections.shape[-1] == 0:
                raise ValueError("token selections must not be empty")
            if int(selections.min()) < 0 or int(selections.max()) >= self.num_experts:
                raise ValueError("token selections contain an out-of-range expert")
            sorted_selections = selections.sort(dim=-1).values
            if sorted_selections.shape[-1] > 1 and torch.any(
                sorted_selections[..., 1:] == sorted_selections[..., :-1]
            ):
                raise ValueError("token selections must not contain duplicates")
            gather_indices = selections[..., None].expand(
                *selections.shape,
                self.hidden_size,
            )
            return torch.gather(outputs, -2, gather_indices).sum(dim=-2)
        indices = tuple(int(index) for index in selected_experts)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("selected_experts must be non-empty and unique")
        if min(indices) < 0 or max(indices) >= self.num_experts:
            raise ValueError("selected_experts contains an out-of-range expert")
        return outputs[..., list(indices), :].sum(dim=-2)


def convert_qwen35_text_ffns(
    model: Qwen3_5ForConditionalGeneration,
    *,
    num_experts: int,
    seed_base: int,
) -> ConversionManifest:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    probe = probe_qwen35_ffn_conversion(
        model,
        num_experts=num_experts,
        seed_base=seed_base,
    )
    if probe.max_abs_difference > 1e-5:
        raise RuntimeError(
            "layer-zero FP32 all-expert conversion probe exceeds 1e-5: "
            f"{probe.max_abs_difference}"
        )
    parameter_count_before = sum(parameter.numel() for parameter in model.parameters())
    visual = model.model.visual
    records: list[ConversionRecord] = []
    mlps = tuple(iter_text_mlps(model))
    for layer_index, dense in mlps:
        if not isinstance(dense, Qwen3_5MLP):
            raise TypeError(f"text layer {layer_index} MLP is not Qwen3_5MLP")
        generator = torch.Generator(device="cpu").manual_seed(seed_base + layer_index)
        permutation = torch.randperm(dense.intermediate_size, generator=generator)
        converted = PartitionedQwen35MLP(
            dense,
            permutation=permutation,
            num_experts=num_experts,
        )
        model.model.language_model.layers[layer_index].mlp = converted
        records.append(
            ConversionRecord(
                layer_index=layer_index,
                seed=seed_base + layer_index,
                permutation=tuple(int(value) for value in permutation.tolist()),
                expert_original_indices=tuple(
                    tuple(int(value) for value in permutation[start:start + converted.expert_width].tolist())
                    for start in range(0, dense.intermediate_size, converted.expert_width)
                ),
            )
        )
    parameter_count_after = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count_before != parameter_count_after:
        raise RuntimeError("Qwen3.5 conversion changed the parameter count")
    if model.model.visual is not visual:
        raise RuntimeError("Qwen3.5 conversion replaced the vision module")
    expert_width = mlps[0][1].intermediate_size // num_experts
    return ConversionManifest(
        converted_layers=len(records),
        num_experts=num_experts,
        expert_width=expert_width,
        parameter_count_before=parameter_count_before,
        parameter_count_after=parameter_count_after,
        records=tuple(records),
        probe=probe,
    )
