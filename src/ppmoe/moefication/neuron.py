"""Validated SwiGLU neuron-triplet parameter operations."""

from dataclasses import dataclass

import torch
from torch import Tensor


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclass(frozen=True)
class SwiGLUParameters:
    """The three inseparable parameter axes that define SwiGLU neurons."""

    gate_weight: Tensor
    up_weight: Tensor
    down_weight: Tensor
    gate_bias: Tensor | None = None
    up_bias: Tensor | None = None
    down_bias: Tensor | None = None

    def __post_init__(self) -> None:
        weights = (self.gate_weight, self.up_weight, self.down_weight)
        if any(weight.ndim != 2 for weight in weights):
            raise ValueError("gate, up, and down weights must all be rank 2")

        intermediate_size, hidden_size = self.gate_weight.shape
        if self.up_weight.shape != (intermediate_size, hidden_size):
            raise ValueError(
                "up weight has an inconsistent intermediate or hidden axis"
            )
        if self.down_weight.shape != (hidden_size, intermediate_size):
            raise ValueError(
                "down weight has an inconsistent hidden or intermediate axis"
            )

        expected_bias_sizes = (
            (self.gate_bias, intermediate_size, "gate"),
            (self.up_bias, intermediate_size, "up"),
            (self.down_bias, hidden_size, "down"),
        )
        for bias, expected_size, name in expected_bias_sizes:
            if bias is not None and bias.shape != (expected_size,):
                raise ValueError(f"{name} bias must have shape ({expected_size},)")

        tensors = [
            tensor
            for tensor in (*weights, self.gate_bias, self.up_bias, self.down_bias)
            if tensor is not None
        ]
        if len({tensor.dtype for tensor in tensors}) != 1:
            raise ValueError("all SwiGLU parameters must have the same dtype")
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all SwiGLU parameters must be on the same device")

    @property
    def intermediate_size(self) -> int:
        return self.gate_weight.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.gate_weight.shape[1]


def validate_permutation(permutation: Tensor, intermediate_size: int) -> Tensor:
    """Return a detached long copy after validating a complete permutation."""

    if intermediate_size <= 0:
        raise ValueError("intermediate_size must be positive")
    if permutation.ndim != 1:
        raise ValueError("permutation must be rank 1")
    if permutation.dtype not in _INTEGER_DTYPES:
        raise ValueError("permutation must use an integer dtype")
    if permutation.numel() != intermediate_size:
        raise ValueError(f"permutation length must equal {intermediate_size}")
    if torch.any(permutation < 0) or torch.any(permutation >= intermediate_size):
        raise ValueError(
            f"permutation indices must be in range [0, {intermediate_size})"
        )
    if torch.unique(permutation).numel() != intermediate_size:
        raise ValueError("permutation must contain each index exactly once")
    return permutation.detach().to(dtype=torch.long).clone()
