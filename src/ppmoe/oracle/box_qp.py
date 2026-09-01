"""Deterministic, certified active-set solves for small convex box QPs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class BoxQPSettings:
    """Protocol-fixed numerical controls for the bounded reconstruction solve."""

    kkt_tolerance: float = 1e-7
    max_iterations: int = 64
    pseudoinverse_rtol: float = 1e-12
    objective_tolerance: float = 1e-10
    norm_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not isfinite(self.kkt_tolerance) or self.kkt_tolerance <= 0.0:
            raise ValueError("kkt_tolerance must be finite and positive")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.max_iterations > 64:
            raise ValueError("max_iterations must be at most 64")
        if not isfinite(self.pseudoinverse_rtol) or self.pseudoinverse_rtol <= 0.0:
            raise ValueError("pseudoinverse_rtol must be finite and positive")
        if not isfinite(self.objective_tolerance) or self.objective_tolerance <= 0.0:
            raise ValueError("objective_tolerance must be finite and positive")
        if not isfinite(self.norm_tolerance) or self.norm_tolerance <= 0.0:
            raise ValueError("norm_tolerance must be finite and positive")


@dataclass(frozen=True, slots=True)
class BoxQPSolution:
    """A batched box-QP candidate together with its separate KKT certificate."""

    coefficients: Tensor
    objective: Tensor
    primal_residual: Tensor
    stationarity_residual: Tensor
    complementarity_residual: Tensor
    iterations: Tensor
    certified: Tensor
    face_enumeration_count: Tensor


def _validate_inputs(
    gs: Tensor, bs: Tensor, objective_scale: Tensor, lower: float, upper: float
) -> None:
    if gs.ndim < 2 or gs.shape[-1] != gs.shape[-2]:
        raise ValueError("gs must have shape [..., k, k]")
    if bs.shape != gs.shape[:-1]:
        raise ValueError("bs must have shape [..., k] matching gs")
    k = bs.shape[-1]
    if not 2 <= k <= 6:
        raise ValueError(f"box-QP requires 2 <= k <= 6; received {k}")
    if objective_scale.shape != bs.shape[:-1]:
        raise ValueError("objective_scale must have one value per system")
    if not gs.is_floating_point() or not bs.is_floating_point() or not objective_scale.is_floating_point():
        raise ValueError("gs, bs, and objective_scale must be floating-point tensors")
    if gs.device != bs.device or gs.device != objective_scale.device:
        raise ValueError("gs, bs, and objective_scale must share a device")
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        raise ValueError("box bounds must be finite and satisfy lower < upper")
    if (
        not bool(torch.isfinite(gs).all())
        or not bool(torch.isfinite(bs).all())
        or not bool(torch.isfinite(objective_scale).all())
        or bool((objective_scale < 0.0).any())
    ):
        raise ValueError("gs, bs, and non-negative objective_scale must be finite")


def _first(mask: Tensor) -> Tensor:
    """Return each row's first true position; callers guard rows with no match."""

    return mask.to(torch.int64).argmax(dim=-1)


def _canonical_face_choice(
    coefficients: Tensor,
    objectives: Tensor,
    *,
    objective_scale: float,
    objective_tolerance: float,
    norm_tolerance: float,
) -> int:
    """Choose a tied active face by objective, L2 norm, then coordinates."""

    minimum_objective = objectives.min()
    objective_tied = objectives <= minimum_objective + objective_tolerance * max(
        objective_scale, 1.0
    )
    norms = coefficients.norm(dim=-1)
    minimum_norm = norms[objective_tied].min()
    norm_tied = objective_tied & (norms <= minimum_norm + norm_tolerance)
    selected = int(torch.nonzero(norm_tied, as_tuple=False)[0].item())
    for contender in torch.nonzero(norm_tied, as_tuple=False).flatten().tolist():
        left = coefficients[contender]
        right = coefficients[selected]
        for coordinate in range(coefficients.shape[-1]):
            if bool(left[coordinate] < right[coordinate]):
                selected = contender
                break
            if bool(left[coordinate] > right[coordinate]):
                break
    return selected


def _kkt_residuals(
    gs: Tensor,
    bs: Tensor,
    coefficients: Tensor,
    lower: float,
    upper: float,
    tolerance: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return gradient and the three registered, independently scaled residuals."""

    gradient = (gs @ coefficients[..., None]).squeeze(-1) - bs
    box_width = upper - lower
    if box_width <= 0.0:
        raise ValueError("box width must be positive")
    primal = torch.maximum(
        (lower - coefficients).clamp_min(0.0),
        (coefficients - upper).clamp_min(0.0),
    ).amax(dim=-1) / box_width
    gradient_scale = torch.maximum(
        bs.abs().amax(dim=-1),
        torch.diagonal(gs, dim1=-2, dim2=-1).sum(dim=-1),
    ).clamp_min(1.0)

    at_lower = coefficients <= lower + tolerance
    at_upper = coefficients >= upper - tolerance
    interior = ~(at_lower | at_upper)
    projected_gradient = torch.where(
        at_lower,
        (-gradient).clamp_min(0.0),
        torch.where(at_upper, gradient.clamp_min(0.0), gradient.abs()),
    )
    stationarity = projected_gradient.amax(dim=-1) / gradient_scale

    lower_multiplier = gradient.clamp_min(0.0)
    upper_multiplier = (-gradient).clamp_min(0.0)
    complementarity = torch.maximum(
        ((coefficients - lower).clamp_min(0.0) * lower_multiplier).amax(dim=-1),
        ((upper - coefficients).clamp_min(0.0) * upper_multiplier).amax(dim=-1),
    ) / gradient_scale
    # Keeping this value materialized prevents an accidental change of the
    # certificate from treating a point in the box interior as a bound point.
    del interior
    return gradient, primal, stationarity, complementarity


def _canonical_rank_deficient_solution(
    gs: Tensor,
    bs: Tensor,
    lower: float,
    upper: float,
    objective_scale: Tensor,
    settings: BoxQPSettings,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None:
    """Resolve certified rank-deficient active faces to the registered order."""

    k = bs.shape[0]
    candidates: list[Tensor] = []
    objectives: list[Tensor] = []
    residuals: list[tuple[Tensor, Tensor, Tensor]] = []
    for face in product((-1, 0, 1), repeat=k):
        state = torch.tensor(face, dtype=torch.int8, device=gs.device)
        fixed = state != 0
        free = ~fixed
        coefficients = torch.zeros(k, dtype=gs.dtype, device=gs.device)
        coefficients[state < 0] = lower
        coefficients[state > 0] = upper
        if bool(free.any()):
            rhs = bs[free] - gs[free][:, fixed] @ coefficients[fixed]
            coefficients[free] = torch.linalg.pinv(
                gs[free][:, free], hermitian=True, rtol=settings.pseudoinverse_rtol
            ) @ rhs
        _, primal, stationarity, complementarity = _kkt_residuals(
            gs.unsqueeze(0),
            bs.unsqueeze(0),
            coefficients.unsqueeze(0),
            lower,
            upper,
            settings.kkt_tolerance,
        )
        if not bool(
            (primal <= settings.kkt_tolerance)
            & (stationarity <= settings.kkt_tolerance)
            & (complementarity <= settings.kkt_tolerance)
        ):
            continue
        candidates.append(coefficients)
        objectives.append(coefficients @ gs @ coefficients - 2.0 * bs @ coefficients)
        residuals.append((primal, stationarity, complementarity))
    if not candidates:
        return None
    all_coefficients = torch.stack(candidates)
    all_objectives = torch.stack(objectives)
    selected = _canonical_face_choice(
        all_coefficients,
        all_objectives,
        objective_scale=float(objective_scale.item()),
        objective_tolerance=settings.objective_tolerance,
        norm_tolerance=settings.norm_tolerance,
    )
    primal, stationarity, complementarity = residuals[selected]
    return (
        all_coefficients[selected],
        primal.squeeze(0),
        stationarity.squeeze(0),
        complementarity.squeeze(0),
        all_objectives[selected],
    )


def solve_box_qp(
    gs: Tensor,
    bs: Tensor,
    lower: float,
    upper: float,
    objective_scale: Tensor,
    settings: BoxQPSettings,
) -> BoxQPSolution:
    """Solve ``min a'Ga - 2b'a`` over a finite box with deterministic updates.

    The leading dimensions are independent systems.  Each active-set step solves
    the free coordinates using a minimum-L2 pseudoinverse, clamps the first
    violating coordinate, or releases the first bound with a wrong KKT sign.
    An iteration cap leaves the result explicitly uncertified.
    """

    _validate_inputs(gs, bs, objective_scale, lower, upper)
    leading_shape = bs.shape[:-1]
    k = bs.shape[-1]
    flat_gs = gs.reshape(-1, k, k)
    flat_bs = bs.reshape(-1, k)
    flat_objective_scale = objective_scale.reshape(-1)
    systems = flat_bs.shape[0]
    dtype = torch.promote_types(flat_gs.dtype, flat_bs.dtype)
    flat_gs = flat_gs.to(dtype)
    flat_bs = flat_bs.to(dtype)
    flat_objective_scale = flat_objective_scale.to(dtype)

    # -1/lower, 0/free, +1/upper.  The specified initial candidate is one.
    state = torch.zeros((systems, k), dtype=torch.int8, device=bs.device)
    coefficients = torch.ones((systems, k), dtype=dtype, device=bs.device).clamp(
        lower, upper
    )
    iterations = torch.zeros(systems, dtype=torch.int64, device=bs.device)
    face_enumeration_count = torch.zeros(
        systems, dtype=torch.int64, device=bs.device
    )
    unresolved = torch.ones(systems, dtype=torch.bool, device=bs.device)

    identity = torch.eye(k, dtype=dtype, device=bs.device).expand(systems, -1, -1)
    row_indices = torch.arange(systems, device=bs.device)
    for iteration in range(settings.max_iterations):
        if not bool(unresolved.any()):
            break
        free = state == 0
        fixed_coefficients = torch.where(
            state < 0,
            torch.full_like(coefficients, lower),
            torch.where(state > 0, torch.full_like(coefficients, upper), torch.zeros_like(coefficients)),
        )
        free_outer = free[..., :, None] & free[..., None, :]
        system = torch.where(free_outer, flat_gs, identity * (~free).to(dtype)[..., None])
        rhs = (flat_bs - (flat_gs @ fixed_coefficients[..., None]).squeeze(-1)) * free
        free_solution = (
            torch.linalg.pinv(system, hermitian=True, rtol=settings.pseudoinverse_rtol)
            @ rhs[..., None]
        ).squeeze(-1)
        candidate = fixed_coefficients + torch.where(free, free_solution, torch.zeros_like(free_solution))
        coefficients = torch.where(unresolved[:, None], candidate, coefficients)

        gradient, primal, stationarity, complementarity = _kkt_residuals(
            flat_gs, flat_bs, coefficients, lower, upper, settings.kkt_tolerance
        )
        certified = (
            (primal <= settings.kkt_tolerance)
            & (stationarity <= settings.kkt_tolerance)
            & (complementarity <= settings.kkt_tolerance)
        )
        newly_certified = unresolved & certified
        iterations[newly_certified] = iteration + 1
        unresolved &= ~certified
        if not bool(unresolved.any()):
            break

        scale = torch.maximum(
            flat_bs.abs().amax(dim=-1),
            torch.diagonal(flat_gs, dim1=-2, dim2=-1).sum(dim=-1),
        ).clamp_min(1.0)
        violating_lower = free & (coefficients < lower - settings.kkt_tolerance)
        violating_upper = free & (coefficients > upper + settings.kkt_tolerance)
        violating = violating_lower | violating_upper
        has_violation = violating.any(dim=-1) & unresolved
        clamp_index = _first(violating)
        state[row_indices[has_violation], clamp_index[has_violation]] = torch.where(
            violating_lower[row_indices[has_violation], clamp_index[has_violation]],
            torch.full_like(clamp_index[has_violation], -1, dtype=torch.int8),
            torch.full_like(clamp_index[has_violation], 1, dtype=torch.int8),
        )

        wrong_lower = (state < 0) & (gradient < -settings.kkt_tolerance * scale[:, None])
        wrong_upper = (state > 0) & (gradient > settings.kkt_tolerance * scale[:, None])
        release = (wrong_lower | wrong_upper) & ~has_violation[:, None]
        has_release = release.any(dim=-1) & unresolved
        release_index = _first(release)
        state[row_indices[has_release], release_index[has_release]] = 0
        changed = has_violation | has_release
        stalled = unresolved & ~changed
        iterations[stalled] = iteration + 1
        unresolved &= ~stalled

    gradient, primal, stationarity, complementarity = _kkt_residuals(
        flat_gs, flat_bs, coefficients, lower, upper, settings.kkt_tolerance
    )
    del gradient
    certified = (
        (primal <= settings.kkt_tolerance)
        & (stationarity <= settings.kkt_tolerance)
        & (complementarity <= settings.kkt_tolerance)
        & ~unresolved
    )
    eigenvalues = torch.linalg.eigvalsh(flat_gs)
    largest_eigenvalue = eigenvalues.abs().amax(dim=-1)
    rank_deficient = (largest_eigenvalue == 0.0) | (
        eigenvalues[..., 0].abs()
        <= settings.pseudoinverse_rtol * largest_eigenvalue.clamp_min(1.0)
    )
    for system_index in torch.nonzero(certified & rank_deficient, as_tuple=False).flatten().tolist():
        face_enumeration_count[system_index] = 3**k
        canonical = _canonical_rank_deficient_solution(
            flat_gs[system_index],
            flat_bs[system_index],
            lower,
            upper,
            flat_objective_scale[system_index],
            settings,
        )
        if canonical is None:
            certified[system_index] = False
            continue
        (
            coefficients[system_index],
            primal[system_index],
            stationarity[system_index],
            complementarity[system_index],
            _,
        ) = canonical
    iterations[unresolved] = settings.max_iterations
    objective = torch.einsum("bi,bij,bj->b", coefficients, flat_gs, coefficients) - 2.0 * (
        flat_bs * coefficients
    ).sum(dim=-1)
    return BoxQPSolution(
        coefficients=coefficients.reshape(*leading_shape, k),
        objective=objective.reshape(*leading_shape),
        primal_residual=primal.reshape(*leading_shape),
        stationarity_residual=stationarity.reshape(*leading_shape),
        complementarity_residual=complementarity.reshape(*leading_shape),
        iterations=iterations.reshape(*leading_shape),
        certified=certified.reshape(*leading_shape),
        face_enumeration_count=face_enumeration_count.reshape(*leading_shape),
    )
