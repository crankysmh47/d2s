"""Exact fixed-K coefficient reconstruction oracles for eight experts.

The free-reconstruction condition intentionally returns the best member of a
registered finite candidate pool.  It is not an assertion of the exact span
optimum: the pseudoinverse cutoff can discard low-energy modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Literal, Mapping

import torch
from torch import Tensor


_FAMILY = Literal["unit", "ridge", "bounded", "free_bound"]
_DEFAULT_RHOS = (10.0, 1.0, 0.1, 0.01, 0.001)


@dataclass(frozen=True, slots=True)
class ReconstructionCondition:
    """A registered coefficient family and, for ridge, its dimensionless rho."""

    family: _FAMILY
    rho: float | None = None

    def __post_init__(self) -> None:
        if self.family not in {"unit", "ridge", "bounded", "free_bound"}:
            raise ValueError(f"unsupported reconstruction family: {self.family}")
        if self.family == "ridge":
            if self.rho is None or not isfinite(self.rho) or self.rho <= 0.0:
                raise ValueError("ridge reconstruction requires a finite rho > 0")
        elif self.rho is not None:
            raise ValueError(f"{self.family} reconstruction does not accept rho")


@dataclass(frozen=True, slots=True)
class SolverSettings:
    """Numerical constants frozen by the Phase 1A.5 protocol."""

    tau_g: float = 1e-12
    epsilon_zero: float = 1e-15
    rho_values: tuple[float, ...] = _DEFAULT_RHOS
    residual_tolerance: float = 1e-9
    objective_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if not isfinite(self.tau_g) or self.tau_g <= 0.0:
            raise ValueError("tau_g must be finite and positive")
        if not isfinite(self.epsilon_zero) or self.epsilon_zero < 0.0:
            raise ValueError("epsilon_zero must be finite and non-negative")
        if not self.rho_values or any(
            not isfinite(rho) or rho <= 0.0 for rho in self.rho_values
        ):
            raise ValueError("rho_values must contain only finite positive values")
        if self.residual_tolerance <= 0.0 or self.objective_tolerance <= 0.0:
            raise ValueError("solver tolerances must be positive")


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    """The selected subset and coefficients for every input token."""

    subsets: Tensor
    coefficients: Tensor
    squared_error: Tensor
    effective_rank: Tensor | None
    solver_source: tuple[str, ...]
    diagnostics: Mapping[str, object]


def enumerate_subsets(num_experts: int, k: int, device: torch.device) -> Tensor:
    """Return lexicographically ordered fixed-K expert subsets."""

    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    if not 1 <= k <= num_experts:
        raise ValueError(f"k must satisfy 1 <= k <= {num_experts}; received {k}")
    return torch.tensor(
        tuple(combinations(range(num_experts), k)), dtype=torch.long, device=device
    )


def _validate_inputs(target: Tensor, experts: Tensor, k: int) -> None:
    if target.ndim != 2:
        raise ValueError("target must have shape [tokens, hidden]")
    if experts.ndim != 3:
        raise ValueError("experts must have shape [tokens, 8, hidden]")
    tokens, num_experts, hidden = experts.shape
    if num_experts != 8:
        raise ValueError("reconstruction oracle requires exactly 8 experts")
    if target.shape != (tokens, hidden):
        raise ValueError("target and experts tokens and hidden dimensions must match")
    if target.device != experts.device:
        raise ValueError("target and experts must share a device")
    if not target.is_floating_point() or not experts.is_floating_point():
        raise ValueError("target and experts must use floating-point dtypes")
    if not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(experts).all()):
        raise ValueError("target and experts must be finite")
    if not 1 <= k <= num_experts:
        raise ValueError(f"k must satisfy 1 <= k <= {num_experts}; received {k}")


def _scoring_dtype(target: Tensor, experts: Tensor) -> torch.dtype:
    if target.dtype == torch.float64 or experts.dtype == torch.float64:
        return torch.float64
    return torch.float32


def _quadratic_loss(
    target_norm: Tensor, gs: Tensor, bs: Tensor, coefficients: Tensor
) -> Tensor:
    loss = (
        target_norm[:, None]
        - 2.0 * (bs * coefficients).sum(dim=-1)
        + torch.einsum("tci,tcij,tcj->tc", coefficients, gs, coefficients)
    )
    tolerance = 1e-8 * target_norm[:, None].clamp_min(1.0)
    if bool((loss < -tolerance).any()):
        raise FloatingPointError("quadratic reconstruction loss is materially negative")
    return loss.clamp_min(0.0)


def _materialized_loss(target: Tensor, selected_experts: Tensor, coefficients: Tensor) -> Tensor:
    reconstruction = torch.einsum("tck,tckh->tch", coefficients, selected_experts)
    return (reconstruction - target[:, None, :]).square().sum(dim=-1)


def _pseudoinverse_candidates(gs: Tensor, bs: Tensor, settings: SolverSettings) -> tuple[Tensor, Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(gs)
    maximum = eigenvalues[..., -1]
    keep = (maximum > 0.0)[..., None] & (
        eigenvalues > settings.tau_g * maximum[..., None]
    )
    inverse = torch.where(keep, eigenvalues.reciprocal(), torch.zeros_like(eigenvalues))
    coefficients = torch.einsum(
        "tcij,tcj,tcj->tci", eigenvectors, inverse, torch.einsum("tcij,tcj->tci", eigenvectors.transpose(-2, -1), bs)
    )
    return coefficients, keep.sum(dim=-1)


def _ridge_candidates(
    gs: Tensor,
    bs: Tensor,
    target_norm: Tensor,
    rho: float,
    settings: SolverSettings,
) -> tuple[Tensor, Mapping[str, object]]:
    tokens, candidates, k, _ = gs.shape
    trace = torch.diagonal(gs, dim1=-2, dim2=-1).sum(dim=-1)
    zero = trace <= settings.epsilon_zero * target_norm[:, None].clamp_min(1.0)
    lambda_eff = rho * trace / k
    identity = torch.eye(k, dtype=gs.dtype, device=gs.device)
    system = gs + lambda_eff[..., None, None] * identity
    rhs = bs + lambda_eff[..., None]
    coefficients = torch.ones((tokens, candidates, k), dtype=gs.dtype, device=gs.device)

    active = ~zero
    fallback_count = 0
    if bool(active.any()):
        active_system = system[active]
        active_rhs = rhs[active]
        cholesky, info = torch.linalg.cholesky_ex(active_system)
        solved = torch.empty_like(active_rhs)
        direct = info == 0
        if bool(direct.any()):
            solved[direct] = torch.cholesky_solve(
                active_rhs[direct, :, None], cholesky[direct]
            ).squeeze(-1)
        if bool((~direct).any()):
            fallback_count = int((~direct).sum().item())
            fallback_system = active_system[~direct]
            eigenvalues, eigenvectors = torch.linalg.eigh(fallback_system)
            scale = torch.diagonal(fallback_system, dim1=-2, dim2=-1).sum(dim=-1)
            if bool((eigenvalues[..., 0] <= 0.0).any()):
                raise FloatingPointError("ridge eigensolve fallback found non-positive system")
            solution = torch.einsum(
                "bij,bj->bi",
                eigenvectors,
                torch.einsum(
                    "bij,bi->bj", eigenvectors.transpose(-2, -1), active_rhs[~direct]
                )
                / eigenvalues,
            )
            if bool(~torch.isfinite(solution).all()) or bool((scale <= 0.0).any()):
                raise FloatingPointError("ridge eigensolve fallback was non-finite")
            solved[~direct] = solution
        coefficients[active] = solved

    residual = (system @ coefficients[..., None]).squeeze(-1) - rhs
    relative_residual = residual.norm(dim=-1) / rhs.norm(dim=-1).clamp_min(1.0)
    if bool((relative_residual[active] > settings.residual_tolerance).any()):
        raise FloatingPointError("ridge linear-system residual exceeds tolerance")
    if not bool(torch.isfinite(coefficients).all()):
        raise FloatingPointError("ridge solve produced non-finite coefficients")
    return coefficients, {
        "cholesky_fallback_count": fallback_count,
        "degenerate_ridge_count": int(zero.sum().item()),
        "max_relative_system_residual": float(relative_residual.max().item()),
    }


def _lexicographically_smaller(left: Tensor, right: Tensor) -> Tensor:
    """Return whether each left coefficient vector is lexicographically smaller."""

    unresolved = torch.ones(left.shape[:-1], dtype=torch.bool, device=left.device)
    smaller = torch.zeros_like(unresolved)
    for coordinate in range(left.shape[-1]):
        difference = left[..., coordinate] - right[..., coordinate]
        smaller |= unresolved & (difference < -1e-12)
        unresolved &= difference.abs() <= 1e-12
    return smaller


def _pick_registered_candidate(
    coefficients: list[Tensor],
    losses: list[Tensor],
    sources: tuple[str, ...],
    target_norm: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Choose among candidates exactly tied at the materialized FP64 minimum."""

    all_losses = torch.stack(losses, dim=-1)
    minimum_loss = all_losses.min(dim=-1).values
    chosen_coefficients = coefficients[0].clone()
    chosen_losses = losses[0].clone()
    chosen_source = torch.zeros(
        chosen_losses.shape, dtype=torch.long, device=chosen_losses.device
    )
    chosen = chosen_losses == minimum_loss
    for source_index, contender_coefficients in enumerate(coefficients[1:], start=1):
        contender_losses = losses[source_index]
        tied_loss = contender_losses == minimum_loss
        contender_norm = contender_coefficients.norm(dim=-1)
        chosen_norm = chosen_coefficients.norm(dim=-1)
        better_norm = contender_norm < chosen_norm - 1e-12
        tied_norm = (contender_norm - chosen_norm).abs() <= 1e-12
        replace = tied_loss & (
            ~chosen
            | better_norm
            | (
                tied_norm
                & _lexicographically_smaller(contender_coefficients, chosen_coefficients)
            )
        )
        chosen_coefficients = torch.where(
            replace[..., None], contender_coefficients, chosen_coefficients
        )
        chosen_losses = torch.where(replace, contender_losses, chosen_losses)
        chosen_source = torch.where(
            replace, torch.full_like(chosen_source, source_index), chosen_source
        )
        chosen |= tied_loss
    return chosen_coefficients, chosen_losses, chosen_source


def _choose_subsets(losses: Tensor) -> Tensor:
    """Use the first minimum, which is the lexicographically first subset."""

    return losses.argmin(dim=1)


def _gather_by_subset(values: Tensor, subset_indices: Tensor) -> Tensor:
    return values[torch.arange(values.shape[0], device=values.device), subset_indices]


def score_reconstruction(
    target: Tensor,
    experts: Tensor,
    k: int,
    condition: ReconstructionCondition,
    settings: SolverSettings,
) -> ReconstructionResult:
    """Exhaustively choose each token's best subset under one coefficient family."""

    _validate_inputs(target, experts, k)
    if condition.family == "bounded":
        raise NotImplementedError("bounded reconstruction is implemented by the box-QP task")

    combinations = enumerate_subsets(experts.shape[1], k, experts.device)
    score_dtype = _scoring_dtype(target, experts)
    score_target = target.to(score_dtype)
    score_experts = experts.to(score_dtype)
    target_norm = score_target.square().sum(dim=-1)
    full_gram = torch.einsum("teh,tfh->tef", score_experts, score_experts)
    full_linear = torch.einsum("th,teh->te", score_target, score_experts)
    gs = full_gram[:, combinations[:, :, None], combinations[:, None, :]]
    bs = full_linear[:, combinations]

    unit_coefficients = torch.ones(
        bs.shape, dtype=score_dtype, device=experts.device
    )
    unit_loss = _quadratic_loss(target_norm, gs, bs, unit_coefficients)

    if condition.family == "unit":
        subset_indices = _choose_subsets(unit_loss)
        coefficients = _gather_by_subset(unit_coefficients, subset_indices)
        selected_subsets = _gather_by_subset(
            combinations.expand(target.shape[0], -1, -1), subset_indices
        )
        selected_experts = score_experts[
            torch.arange(target.shape[0], device=experts.device)[:, None],
            selected_subsets,
        ]
        squared_error = (
            (
                torch.einsum("tk,tkh->th", coefficients, selected_experts)
                - score_target
            )
            .square()
            .sum(dim=-1)
        )
        return ReconstructionResult(
            subsets=selected_subsets,
            coefficients=coefficients,
            squared_error=squared_error,
            effective_rank=None,
            solver_source=("unit",) * target.shape[0],
            diagnostics={
            "cholesky_fallback_count": 0,
            "degenerate_ridge_count": 0,
            "max_relative_system_residual": 0.0,
            },
        )

    # Numerical coefficient systems are constructed from promoted vectors,
    # never from an FP32 Gram matrix cast after accumulation.
    target64 = target.double()
    experts64 = experts.double()
    target_norm64 = target64.square().sum(dim=-1)
    full_gram64 = torch.einsum("teh,tfh->tef", experts64, experts64)
    full_linear64 = torch.einsum("th,teh->te", target64, experts64)
    gs64 = full_gram64[:, combinations[:, :, None], combinations[:, None, :]]
    bs64 = full_linear64[:, combinations]

    if condition.family == "ridge":
        ridge_coefficients, ridge_diagnostics = _ridge_candidates(
            gs64, bs64, target_norm64, condition.rho, settings
        )
        ridge_loss = _quadratic_loss(target_norm64, gs64, bs64, ridge_coefficients)
        subset_indices = _choose_subsets(ridge_loss)
        coefficients = _gather_by_subset(ridge_coefficients, subset_indices)
        selected_subsets = _gather_by_subset(
            combinations.expand(target.shape[0], -1, -1), subset_indices
        )
        selected_experts64 = experts64[
            torch.arange(target.shape[0], device=experts.device)[:, None],
            selected_subsets,
        ]
        squared_error = (
            (
                torch.einsum("tk,tkh->th", coefficients, selected_experts64)
                - target64
            )
            .square()
            .sum(dim=-1)
        )
        return ReconstructionResult(
            subsets=selected_subsets,
            coefficients=coefficients,
            squared_error=squared_error,
            effective_rank=None,
            solver_source=(f"ridge:{condition.rho:g}",) * target.shape[0],
            diagnostics=dict(ridge_diagnostics),
        )

    pseudoinverse_coefficients, pseudoinverse_rank = _pseudoinverse_candidates(
        gs64, bs64, settings
    )
    selected_experts64 = experts64[:, combinations, :]
    unit_coefficients64 = torch.ones_like(bs64)
    pseudoinverse_loss = _materialized_loss(
        target64, selected_experts64, pseudoinverse_coefficients
    )
    candidate_coefficients = [pseudoinverse_coefficients, unit_coefficients64]
    candidate_losses = [
        pseudoinverse_loss,
        _materialized_loss(target64, selected_experts64, unit_coefficients64),
    ]
    sources = ("pseudoinverse", "unit") + tuple(
        f"ridge:{rho:g}" for rho in settings.rho_values
    )
    total_fallbacks = 0
    total_degenerate = 0
    largest_residual = 0.0
    for rho in settings.rho_values:
        ridge_coefficients, ridge_diagnostics = _ridge_candidates(
            gs64, bs64, target_norm64, rho, settings
        )
        candidate_coefficients.append(ridge_coefficients)
        candidate_losses.append(
            _materialized_loss(target64, selected_experts64, ridge_coefficients)
        )
        total_fallbacks += int(ridge_diagnostics["cholesky_fallback_count"])
        total_degenerate += int(ridge_diagnostics["degenerate_ridge_count"])
        largest_residual = max(
            largest_residual,
            float(ridge_diagnostics["max_relative_system_residual"]),
        )
    free_coefficients, free_loss, source_indices_all = _pick_registered_candidate(
        candidate_coefficients,
        candidate_losses,
        sources,
        target_norm64,
    )
    subset_indices = _choose_subsets(free_loss)
    coefficients = _gather_by_subset(free_coefficients, subset_indices)
    squared_error = _gather_by_subset(free_loss, subset_indices)
    source_indices = _gather_by_subset(source_indices_all, subset_indices)
    diagnostics: dict[str, object] = {
        "cholesky_fallback_count": total_fallbacks,
        "degenerate_ridge_count": total_degenerate,
        "max_relative_system_residual": largest_residual,
        "candidate_sources": sources,
    }
    selected_subsets = _gather_by_subset(
        combinations.expand(target.shape[0], -1, -1), subset_indices
    )
    selected_pseudoinverse_coefficients = _gather_by_subset(
        pseudoinverse_coefficients, subset_indices
    )
    selected_pseudoinverse_rank = _gather_by_subset(pseudoinverse_rank, subset_indices)
    selected_experts64 = experts.double()[
        torch.arange(target.shape[0], device=experts.device)[:, None], selected_subsets
    ]
    selected_pseudoinverse_loss = (
        (torch.einsum("tk,tkh->th", selected_pseudoinverse_coefficients, selected_experts64) - target64)
        .square()
        .sum(dim=-1)
    )
    diagnostics.update(
        {
            "tau_g": settings.tau_g,
            "pseudoinverse_coefficients": selected_pseudoinverse_coefficients,
            "pseudoinverse_squared_error": selected_pseudoinverse_loss,
            "pseudoinverse_effective_rank": selected_pseudoinverse_rank,
        }
    )
    return ReconstructionResult(
        subsets=selected_subsets,
        coefficients=coefficients,
        squared_error=squared_error,
        effective_rank=selected_pseudoinverse_rank,
        solver_source=tuple(sources[index] for index in source_indices.tolist()),
        diagnostics=diagnostics,
    )


def verify_pseudoinverse_svd(
    target: Tensor,
    experts: Tensor,
    result: ReconstructionResult,
    settings: SolverSettings,
) -> Mapping[str, object]:
    """Cross-check stored primary pseudoinverse candidates with direct FP64 SVD."""

    _validate_inputs(target, experts, result.subsets.shape[1])
    if result.subsets.shape != result.coefficients.shape:
        raise ValueError("result subsets and coefficients must have matching shapes")
    pseudoinverse_coefficients = result.diagnostics.get("pseudoinverse_coefficients")
    pseudoinverse_loss = result.diagnostics.get("pseudoinverse_squared_error")
    pseudoinverse_rank = result.diagnostics.get("pseudoinverse_effective_rank")
    if not all(isinstance(value, Tensor) for value in (
        pseudoinverse_coefficients, pseudoinverse_loss, pseudoinverse_rank
    )):
        raise ValueError("result does not expose primary pseudoinverse candidates")

    rows = torch.arange(target.shape[0], device=target.device)
    selected = experts.double()[rows[:, None], result.subsets].transpose(-2, -1)
    u, singular_values, vh = torch.linalg.svd(selected, full_matrices=False)
    maximum = singular_values[..., 0]
    keep = (maximum > 0.0)[..., None] & (
        singular_values > settings.tau_g**0.5 * maximum[..., None]
    )
    inverse = torch.where(keep, singular_values.reciprocal(), torch.zeros_like(singular_values))
    projected = (u.transpose(-2, -1) @ target.double()[..., None]).squeeze(-1)
    svd_coefficients = vh.transpose(-2, -1) @ (projected * inverse)[..., None]
    svd_coefficients = svd_coefficients.squeeze(-1)
    reconstruction = torch.einsum("tk,thk->th", svd_coefficients, selected)
    svd_loss = (reconstruction - target.double()).square().sum(dim=-1)
    rank_equal = bool(torch.equal(keep.sum(dim=-1), pseudoinverse_rank))
    relative_objective_error = (svd_loss - pseudoinverse_loss).abs() / target.double().square().sum(dim=-1).clamp_min(1.0)
    maximum_error = float(relative_objective_error.max().item())
    if not rank_equal or maximum_error > settings.objective_tolerance:
        raise FloatingPointError("pseudoinverse and direct SVD verification disagree")
    return {
        "rank_equal": rank_equal,
        "relative_objective_error": relative_objective_error,
        "max_relative_objective_error": maximum_error,
    }
