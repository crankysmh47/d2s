"""Unoptimized reference SwiGLU and partitioned-expert modules."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ppmoe.moefication.neuron import validate_permutation


class DenseSwiGLU(nn.Module):
    """A small dense SwiGLU FFN with explicit Qwen-style projections."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        gate_bias: bool = False,
        up_bias: bool = False,
        down_bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=gate_bias,
            **factory_kwargs,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=up_bias,
            **factory_kwargs,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=down_bias,
            **factory_kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertSwiGLU(nn.Module):
    """One independently executable set of complete SwiGLU neurons."""

    def __init__(self, dense: DenseSwiGLU, indices: Tensor) -> None:
        super().__init__()
        device = dense.gate_proj.weight.device
        dtype = dense.gate_proj.weight.dtype
        indices = indices.to(device=device, dtype=torch.long)
        width = indices.numel()
        self.register_buffer("indices", indices.clone())
        self.gate_proj = nn.Linear(
            dense.hidden_size,
            width,
            bias=dense.gate_proj.bias is not None,
            device=device,
            dtype=dtype,
        )
        self.up_proj = nn.Linear(
            dense.hidden_size,
            width,
            bias=dense.up_proj.bias is not None,
            device=device,
            dtype=dtype,
        )
        self.down_proj = nn.Linear(
            width,
            dense.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            self.gate_proj.weight.copy_(
                dense.gate_proj.weight.index_select(0, indices)
            )
            self.up_proj.weight.copy_(dense.up_proj.weight.index_select(0, indices))
            self.down_proj.weight.copy_(
                dense.down_proj.weight.index_select(1, indices)
            )
            if self.gate_proj.bias is not None:
                self.gate_proj.bias.copy_(
                    dense.gate_proj.bias.index_select(0, indices)
                )
            if self.up_proj.bias is not None:
                self.up_proj.bias.copy_(dense.up_proj.bias.index_select(0, indices))

    def forward(self, x: Tensor) -> Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(hidden)


class PartitionedSwiGLU(nn.Module):
    """Reference expert execution preserving one dense SwiGLU parameter budget."""

    def __init__(
        self,
        dense: DenseSwiGLU,
        partitions: Sequence[Tensor],
    ) -> None:
        super().__init__()
        validated = self._validate_partitions(dense, partitions)
        self.hidden_size = dense.hidden_size
        self.intermediate_size = dense.intermediate_size
        self.experts = nn.ModuleList(
            [ExpertSwiGLU(dense, indices) for indices in validated]
        )
        if dense.down_proj.bias is None:
            self.register_parameter("down_bias", None)
        else:
            self.down_bias = nn.Parameter(dense.down_proj.bias.detach().clone())

    @staticmethod
    def _validate_partitions(
        dense: DenseSwiGLU,
        partitions: Sequence[Tensor],
    ) -> tuple[Tensor, ...]:
        if not partitions:
            raise ValueError("partitions must not be empty")
        if any(partition.ndim != 1 for partition in partitions):
            raise ValueError("each partition must be rank 1")
        widths = {partition.numel() for partition in partitions}
        if 0 in widths or len(widths) != 1:
            raise ValueError("partitions must have equal nonzero widths")
        source_device = dense.gate_proj.weight.device
        if any(partition.device != source_device for partition in partitions):
            raise ValueError("partition indices must share the dense module device")
        flattened = torch.cat(tuple(partitions))
        try:
            validate_permutation(flattened, dense.intermediate_size)
        except ValueError as error:
            raise ValueError(f"invalid partition indices: {error}") from error
        return tuple(partition.to(dtype=torch.long).clone() for partition in partitions)

    @classmethod
    def from_dense(
        cls,
        dense: DenseSwiGLU,
        partitions: Sequence[Tensor],
    ) -> "PartitionedSwiGLU":
        return cls(dense=dense, partitions=partitions)

    def expert_outputs(self, x: Tensor) -> Tensor:
        return torch.stack([expert(x) for expert in self.experts], dim=-2)

    def forward(
        self,
        x: Tensor,
        selected_experts: Sequence[int] | None = None,
        *,
        accumulation_dtype: torch.dtype | None = None,
    ) -> Tensor:
        indices = (
            tuple(range(len(self.experts)))
            if selected_experts is None
            else tuple(selected_experts)
        )
        if not indices:
            raise ValueError("selected_experts must not be empty")
        if len(set(indices)) != len(indices):
            raise ValueError("selected_experts must not contain duplicates")
        if any(index < 0 or index >= len(self.experts) for index in indices):
            raise ValueError("selected_experts contains an out-of-range index")
        outputs = [self.experts[index](x) for index in indices]
        target_dtype = accumulation_dtype or outputs[0].dtype
        result = torch.stack(
            [output.to(dtype=target_dtype) for output in outputs]
        ).sum(dim=0)
        if self.down_bias is not None:
            result = result + self.down_bias.to(dtype=target_dtype)
        return result.to(dtype=x.dtype)
