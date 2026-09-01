from __future__ import annotations

from itertools import product

import pytest
import torch

from ppmoe.oracle.box_qp import BoxQPSettings, solve_box_qp


SETTINGS = BoxQPSettings()


def _reference_box_qp(gs: torch.Tensor, bs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent exhaustive active-face reference for one small PSD system."""

    k = gs.shape[0]
    best: tuple[torch.Tensor, torch.Tensor] | None = None
    for state in product((-1, 0, 1), repeat=k):
        coefficients = torch.empty(k, dtype=torch.float64)
        fixed = torch.tensor([entry != 0 for entry in state])
        lower = torch.tensor([entry == -1 for entry in state])
        upper = torch.tensor([entry == 1 for entry in state])
        coefficients[lower] = 0.0
        coefficients[upper] = 2.0
        free = ~fixed
        if bool(free.any()):
            rhs = bs[free] - gs[free][:, fixed] @ coefficients[fixed]
            coefficients[free] = torch.linalg.pinv(
                gs[free][:, free], hermitian=True, rtol=1e-12
            ) @ rhs
        gradient = gs @ coefficients - bs
        if (
            bool((coefficients < -1e-9).any())
            or bool((coefficients > 2.0 + 1e-9).any())
            or bool((gradient[lower] < -1e-9).any())
            or bool((gradient[upper] > 1e-9).any())
            or bool((gradient[free].abs() > 1e-9).any())
        ):
            continue
        objective = coefficients @ gs @ coefficients - 2.0 * (bs @ coefficients)
        if best is None:
            best = coefficients, objective
            continue
        best_coefficients, best_objective = best
        if objective < best_objective - 1e-10:
            best = coefficients, objective
        elif abs(objective - best_objective) <= 1e-10:
            if coefficients.norm() < best_coefficients.norm() - 1e-12:
                best = coefficients, objective
            elif (
                abs(coefficients.norm() - best_coefficients.norm()) <= 1e-12
                and tuple(coefficients.tolist()) < tuple(best_coefficients.tolist())
            ):
                best = coefficients, objective
    assert best is not None
    return best


def _deterministic_system(k: int, case: str) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1100 + 10 * k + len(case))
    basis = torch.randn(k + 2, k, generator=generator, dtype=torch.float64)
    if case == "rank_deficient":
        basis[:, -1] = basis[:, 0] + basis[:, 1]
    gs = basis.T @ basis
    desired = torch.ones(k, dtype=torch.float64)
    if case == "lower":
        desired[0] = 0.0
    elif case == "upper":
        desired[-1] = 2.0
    elif case not in {"interior", "rank_deficient"}:
        raise ValueError(case)
    return gs, gs @ desired


@pytest.mark.parametrize("k", range(2, 7))
@pytest.mark.parametrize("case", ("lower", "upper", "interior", "rank_deficient"))
def test_box_qp_matches_exhaustive_active_faces_for_psd_systems(
    k: int, case: str
) -> None:
    """Catches a wrong clamp/release branch or noncanonical singular solve."""

    gs, bs = _deterministic_system(k, case)
    expected_coefficients, expected_objective = _reference_box_qp(gs, bs)

    solution = solve_box_qp(
        gs.unsqueeze(0), bs.unsqueeze(0), lower=0.0, upper=2.0, settings=SETTINGS
    )

    assert solution.certified.tolist() == [True]
    torch.testing.assert_close(solution.objective, expected_objective.unsqueeze(0), rtol=0, atol=1e-10)
    torch.testing.assert_close(solution.coefficients[0], expected_coefficients, rtol=0, atol=1e-10)
    assert solution.primal_residual.max().item() <= 1e-7
    assert solution.stationarity_residual.max().item() <= 1e-7
    assert solution.complementarity_residual.max().item() <= 1e-7


def test_box_qp_preserves_leading_batch_dimensions() -> None:
    """Catches flattening a token-by-subset batch into the wrong result shape."""

    gs, bs = _deterministic_system(3, "upper")
    solution = solve_box_qp(
        gs.expand(2, 4, -1, -1).clone(),
        bs.expand(2, 4, -1).clone(),
        lower=0.0,
        upper=2.0,
        settings=SETTINGS,
    )

    assert solution.coefficients.shape == (2, 4, 3)
    assert solution.objective.shape == (2, 4)
    assert solution.certified.shape == (2, 4)
    assert bool(solution.certified.all())


def test_box_qp_rejects_approximate_solution_after_iteration_cap() -> None:
    """Catches incorrectly certifying a result when the active-set budget is exhausted."""

    gs, bs = _deterministic_system(4, "upper")
    solution = solve_box_qp(
        gs.unsqueeze(0),
        bs.unsqueeze(0),
        lower=0.0,
        upper=2.0,
        settings=BoxQPSettings(max_iterations=0),
    )

    assert solution.certified.tolist() == [False]
