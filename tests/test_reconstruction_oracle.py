import pytest
import torch

from ppmoe.oracle.exact import exact_gram_oracle
from ppmoe.oracle.reconstruction import (
    ReconstructionCondition,
    SolverSettings,
    score_reconstruction,
    verify_pseudoinverse_svd,
)


SETTINGS = SolverSettings()
UNIT = ReconstructionCondition("unit")


def deterministic_case(
    *, tokens: int = 7, experts: int = 8, hidden: int = 11
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(913)
    expert_outputs = torch.randn(
        tokens, experts, hidden, generator=generator, dtype=torch.float64
    )
    target = expert_outputs.sum(dim=1) + 0.03 * torch.randn(
        tokens, hidden, generator=generator, dtype=torch.float64
    )
    return target, expert_outputs


def materialized_loss(
    target: torch.Tensor, experts: torch.Tensor, result: object
) -> torch.Tensor:
    subsets = result.subsets
    coefficients = result.coefficients
    rows = torch.arange(target.shape[0], device=target.device)
    selected = experts[rows[:, None], subsets]
    reconstruction = (selected * coefficients[..., None]).sum(dim=1)
    return (reconstruction - target).square().sum(dim=-1)


def test_unit_reconstruction_matches_existing_exact_oracle() -> None:
    target, experts = deterministic_case()

    old = exact_gram_oracle(target, experts, 4)
    new = score_reconstruction(target, experts, 4, UNIT, SETTINGS)

    assert torch.equal(new.subsets, old.subsets)
    torch.testing.assert_close(
        new.squared_error,
        materialized_loss(target, experts, new),
        rtol=1e-8,
        atol=1e-8,
    )
    torch.testing.assert_close(new.coefficients, torch.ones_like(new.coefficients))


def test_scale_normalized_ridge_is_invariant_to_common_scale() -> None:
    target, experts = deterministic_case()
    condition = ReconstructionCondition("ridge", rho=0.1)

    first = score_reconstruction(target, experts, 4, condition, SETTINGS)
    second = score_reconstruction(target * 17, experts * 17, 4, condition, SETTINGS)

    torch.testing.assert_close(
        first.coefficients, second.coefficients, rtol=1e-9, atol=1e-10
    )
    assert first.diagnostics["cholesky_fallback_count"] == 0
    assert first.diagnostics["max_relative_system_residual"] <= 1e-9


def test_ridge_uses_unit_coefficients_for_registered_zero_trace_degeneracy() -> None:
    target = torch.tensor([[2.0, -1.0]])
    experts = torch.zeros(1, 8, 2)

    result = score_reconstruction(
        target, experts, 4, ReconstructionCondition("ridge", rho=0.1), SETTINGS
    )

    torch.testing.assert_close(result.coefficients, torch.ones_like(result.coefficients))
    assert result.diagnostics["degenerate_ridge_count"] == 70


def test_free_bound_records_zero_and_rank_deficient_pseudoinverse_candidates() -> None:
    target = torch.tensor([[1.0, -2.0], [3.0, 1.0]])
    experts = torch.zeros(2, 8, 2)
    experts[1, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])

    result = score_reconstruction(
        target, experts, 4, ReconstructionCondition("free_bound"), SETTINGS
    )

    assert result.diagnostics["pseudoinverse_effective_rank"].tolist()[0] == 0
    assert result.diagnostics["pseudoinverse_effective_rank"].tolist()[1] == 1
    assert result.diagnostics["tau_g"] == 1e-12
    assert all(source in {"pseudoinverse", "unit", *{
        f"ridge:{rho:g}" for rho in SETTINGS.rho_values
    }} for source in result.solver_source)


def test_free_bound_primary_pseudoinverse_matches_direct_svd_cutoff() -> None:
    target, experts = deterministic_case(tokens=3, hidden=6)
    result = score_reconstruction(
        target, experts, 4, ReconstructionCondition("free_bound"), SETTINGS
    )

    verification = verify_pseudoinverse_svd(target, experts, result, SETTINGS)

    assert verification["rank_equal"]
    assert verification["max_relative_objective_error"] <= 1e-9


def test_free_bound_is_no_worse_than_registered_unit_and_ridge_candidates() -> None:
    target, experts = deterministic_case(tokens=5, hidden=7)
    free = score_reconstruction(
        target, experts, 4, ReconstructionCondition("free_bound"), SETTINGS
    )
    unit = score_reconstruction(target, experts, 4, UNIT, SETTINGS)

    # The comparison uses every token's selected subset; the free candidate
    # pool additionally contains the same unit vector and every ridge vector.
    tolerance = 1e-9 * target.square().sum(dim=-1).clamp_min(1.0)
    assert torch.all(free.squared_error <= unit.squared_error + tolerance)
    for rho in SETTINGS.rho_values:
        ridge = score_reconstruction(
            target, experts, 4, ReconstructionCondition("ridge", rho=rho), SETTINGS
        )
        assert torch.all(free.squared_error <= ridge.squared_error + tolerance)


def test_reconstruction_rejects_nonfinite_and_non_eight_expert_inputs() -> None:
    target, experts = deterministic_case(experts=8)
    experts[0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        score_reconstruction(target, experts, 4, UNIT, SETTINGS)
    with pytest.raises(ValueError, match="8 experts"):
        score_reconstruction(target, torch.zeros(1, 7, 2), 4, UNIT, SETTINGS)
