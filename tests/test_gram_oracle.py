import torch

from ppmoe.oracle.exact import exact_direct_oracle, exact_gram_oracle


def test_gram_oracle_matches_direct_oracle_across_k():
    generator = torch.Generator().manual_seed(23)
    experts = torch.randn(5, 8, 6, generator=generator)
    target = experts.sum(dim=1) + 0.01 * torch.randn(5, 6, generator=generator)

    for k in (1, 2, 4, 8):
        direct = exact_direct_oracle(target, experts, k)
        gram = exact_gram_oracle(target, experts, k)

        assert gram.subsets.tolist() == direct.subsets.tolist()
        torch.testing.assert_close(
            gram.squared_errors,
            direct.squared_errors,
            rtol=1e-5,
            atol=1e-5,
        )


def test_gram_oracle_accounts_for_expert_cancellation():
    experts = torch.tensor([[[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0]]])
    target = torch.tensor([[0.0, 0.0]])

    result = exact_gram_oracle(target, experts, k=2)

    assert result.subsets.tolist() == [[0, 1]]
    assert result.squared_errors.tolist() == [0.0]


def test_gram_oracle_uses_lexicographic_tie_breaking():
    experts = torch.zeros(1, 4, 3)

    result = exact_gram_oracle(torch.zeros(1, 3), experts, k=2)

    assert result.subsets.tolist() == [[0, 1]]
