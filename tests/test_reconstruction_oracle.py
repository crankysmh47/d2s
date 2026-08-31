import pytest
import torch

from ppmoe.oracle.exact import exact_gram_oracle
from ppmoe.oracle.reconstruction import (
    ReconstructionCondition,
    SolverSettings,
    _pick_registered_candidate,
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


def test_unit_reconstruction_matches_fp32_gram_oracle() -> None:
    target, experts = deterministic_case()
    target = target.float()
    experts = experts.float()

    old = exact_gram_oracle(target, experts, 4)
    new = score_reconstruction(target, experts, 4, UNIT, SETTINGS)

    assert torch.equal(new.subsets, old.subsets)
    torch.testing.assert_close(new.squared_error, materialized_loss(target, experts, new))


def test_unit_reconstruction_preserves_cancellation_and_lexicographic_ties() -> None:
    target = torch.zeros(1, 2)
    experts = torch.zeros(1, 8, 2)
    experts[0, 0] = torch.tensor([2.0, -3.0])
    experts[0, 1] = torch.tensor([-2.0, 3.0])
    experts[0, 2] = torch.tensor([2.0, -3.0])
    experts[0, 3] = torch.tensor([-2.0, 3.0])

    result = score_reconstruction(target, experts, 2, UNIT, SETTINGS)

    assert result.subsets.tolist() == [[0, 1]]
    assert result.squared_error.tolist() == [0.0]
    assert result.effective_rank is None
    assert "pseudoinverse_coefficients" not in result.diagnostics


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
    verification = verify_pseudoinverse_svd(target, experts, result, SETTINGS)
    assert verification["rank_equal"]
    assert verification["max_relative_objective_error"] <= 1e-9
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


def test_free_bound_promotes_fp32_inputs_before_eigen_cutoff() -> None:
    generator = torch.Generator().manual_seed(71)
    left, _ = torch.linalg.qr(
        torch.randn(8, 8, generator=generator, dtype=torch.float64)
    )
    right, _ = torch.linalg.qr(
        torch.randn(8, 8, generator=generator, dtype=torch.float64)
    )
    singular_values = torch.tensor(
        [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01, 5e-7], dtype=torch.float64
    )
    matrix = (left @ torch.diag(singular_values) @ right.T).float()
    experts = matrix.T.unsqueeze(0)
    target = torch.randn(1, 8, generator=generator).float()

    result = score_reconstruction(
        target, experts, 8, ReconstructionCondition("free_bound"), SETTINGS
    )
    verification = verify_pseudoinverse_svd(target, experts, result, SETTINGS)

    assert result.effective_rank.tolist() == [7]
    assert verification["rank_equal"]
    assert verification["max_relative_objective_error"] <= 1e-9


def test_ridge_builds_fp64_system_from_promoted_fp32_vectors() -> None:
    generator = torch.Generator().manual_seed(101)
    target = torch.randn(1, 8, generator=generator).float()
    experts = torch.randn(1, 8, 8, generator=generator).float()
    condition = ReconstructionCondition("ridge", rho=0.1)

    result = score_reconstruction(target, experts, 8, condition, SETTINGS)

    matrix = experts[0].double().T
    target64 = target[0].double()
    gram = matrix.T @ matrix
    lambda_eff = condition.rho * torch.trace(gram) / 8
    expected = torch.linalg.solve(
        gram + lambda_eff * torch.eye(8, dtype=torch.float64),
        matrix.T @ target64 + lambda_eff,
    )
    torch.testing.assert_close(result.coefficients[0], expected, rtol=1e-12, atol=1e-12)


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


def test_free_candidate_tie_breaking_cannot_chain_past_global_objective_tolerance() -> None:
    coefficients = [
        torch.tensor([[[3.0]]]),
        torch.tensor([[[2.0]]]),
        torch.tensor([[[1.0]]]),
    ]
    losses = [
        torch.tensor([[0.0]], dtype=torch.float64),
        torch.tensor([[0.9e-9]], dtype=torch.float64),
        torch.tensor([[1.8e-9]], dtype=torch.float64),
    ]

    chosen_coefficients, chosen_losses, chosen_source = _pick_registered_candidate(
        coefficients, losses, ("minimum", "tie", "outside_tie"), torch.ones(1)
    )

    assert chosen_source.tolist() == [[1]]
    assert chosen_losses.tolist() == [[0.9e-9]]
    assert chosen_coefficients.tolist() == [[[2.0]]]


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_solver_paths_and_svd_verification_run_on_available_devices(
    device: torch.device,
) -> None:
    target, experts = deterministic_case(tokens=3, hidden=6)
    target, experts = target.to(device), experts.to(device)

    unit = score_reconstruction(target, experts, 4, UNIT, SETTINGS)
    ridge = score_reconstruction(
        target, experts, 4, ReconstructionCondition("ridge", 0.1), SETTINGS
    )
    free = score_reconstruction(
        target, experts, 4, ReconstructionCondition("free_bound"), SETTINGS
    )

    assert unit.subsets.device.type == device.type
    assert ridge.coefficients.device.type == device.type
    assert free.squared_error.device.type == device.type
    assert verify_pseudoinverse_svd(target, experts, free, SETTINGS)["rank_equal"]


def test_reconstruction_rejects_nonfinite_and_non_eight_expert_inputs() -> None:
    target, experts = deterministic_case(experts=8)
    experts[0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        score_reconstruction(target, experts, 4, UNIT, SETTINGS)
    with pytest.raises(ValueError, match="8 experts"):
        score_reconstruction(target, torch.zeros(1, 7, 2), 4, UNIT, SETTINGS)
