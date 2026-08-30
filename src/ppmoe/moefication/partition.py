"""Deterministic balanced SwiGLU neuron partitions."""

from torch import Tensor

from ppmoe.moefication.neuron import validate_permutation


def balanced_partition(
    permutation: Tensor,
    num_experts: int,
) -> tuple[Tensor, ...]:
    """Split a complete neuron permutation into equal ordered groups."""

    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    intermediate_size = permutation.numel()
    validated = validate_permutation(permutation, intermediate_size)
    if intermediate_size % num_experts != 0:
        raise ValueError(
            f"intermediate size {intermediate_size} must be divisible by "
            f"num_experts {num_experts}"
        )
    expert_width = intermediate_size // num_experts
    return tuple(group.clone() for group in validated.split(expert_width))
