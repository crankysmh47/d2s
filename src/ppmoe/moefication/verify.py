"""Numerically stable tensor fidelity measurements."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class NumericalMetrics:
    max_absolute_error: float
    rms_error: float
    relative_error: float
    normalized_mse: float
    cosine_similarity: float


def compare_tensors(reference: Tensor, candidate: Tensor) -> NumericalMetrics:
    """Compare equal-shaped tensors with calculations performed in FP64."""

    if reference.shape != candidate.shape:
        raise ValueError(
            f"reference and candidate shapes differ: "
            f"{tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    if reference.numel() == 0:
        raise ValueError("cannot compare empty tensors")

    reference64 = reference.detach().to(dtype=torch.float64).reshape(-1)
    candidate64 = candidate.detach().to(dtype=torch.float64).reshape(-1)
    difference = candidate64 - reference64
    squared_error = difference.square()
    reference_energy = reference64.square().mean()
    reference_norm = torch.linalg.vector_norm(reference64)
    candidate_norm = torch.linalg.vector_norm(candidate64)
    epsilon = torch.finfo(torch.float64).eps

    if reference_norm == 0 and candidate_norm == 0:
        cosine_similarity = 1.0
    else:
        denominator = torch.clamp(reference_norm * candidate_norm, min=epsilon)
        cosine_similarity = torch.dot(reference64, candidate64) / denominator

    return NumericalMetrics(
        max_absolute_error=squared_error.max().sqrt().item(),
        rms_error=squared_error.mean().sqrt().item(),
        relative_error=(
            torch.linalg.vector_norm(difference)
            / torch.clamp(reference_norm, min=epsilon)
        ).item(),
        normalized_mse=(
            squared_error.mean() / torch.clamp(reference_energy, min=epsilon)
        ).item(),
        cosine_similarity=(
            cosine_similarity
            if isinstance(cosine_similarity, float)
            else cosine_similarity.item()
        ),
    )
