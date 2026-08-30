"""Exact fixed-K subset oracles for small expert counts."""

from dataclasses import dataclass
from itertools import combinations

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class OracleResult:
    subsets: Tensor
    squared_errors: Tensor


def _validate_oracle_inputs(
    target: Tensor,
    expert_outputs: Tensor,
    k: int,
) -> tuple[int, int, int]:
    if target.ndim != 2:
        raise ValueError("target must have shape [tokens, hidden]")
    if expert_outputs.ndim != 3:
        raise ValueError(
            "expert_outputs must have shape [tokens, experts, hidden]"
        )
    tokens, num_experts, hidden_size = expert_outputs.shape
    if target.shape != (tokens, hidden_size):
        raise ValueError(
            "target and expert_outputs tokens and hidden dimensions must match"
        )
    if target.device != expert_outputs.device:
        raise ValueError("target and expert_outputs must share a device")
    if not 1 <= k <= num_experts:
        raise ValueError(f"k must satisfy 1 <= k <= {num_experts}; received {k}")
    return tokens, num_experts, hidden_size


def _combination_indices(
    num_experts: int,
    k: int,
    *,
    device: torch.device,
) -> Tensor:
    return torch.tensor(
        tuple(combinations(range(num_experts), k)),
        dtype=torch.long,
        device=device,
    )


def exact_direct_oracle(
    target: Tensor,
    expert_outputs: Tensor,
    k: int,
) -> OracleResult:
    """Choose each token's lowest-error K-expert sum by direct reconstruction."""

    _validate_oracle_inputs(target, expert_outputs, k)
    combination_indices = _combination_indices(
        expert_outputs.shape[1],
        k,
        device=expert_outputs.device,
    )
    scoring_dtype = (
        torch.float64 if expert_outputs.dtype == torch.float64 else torch.float32
    )
    experts = expert_outputs.to(dtype=scoring_dtype)
    targets = target.to(dtype=scoring_dtype)
    reconstructions = experts[:, combination_indices, :].sum(dim=2)
    errors = (reconstructions - targets[:, None, :]).square().sum(dim=-1)
    best_combination = errors.argmin(dim=1)
    token_indices = torch.arange(target.shape[0], device=target.device)
    return OracleResult(
        subsets=combination_indices.index_select(0, best_combination),
        squared_errors=errors[token_indices, best_combination],
    )
