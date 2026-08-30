import pytest
import torch

from ppmoe.oracle.exact import exact_direct_oracle


def test_direct_oracle_selects_best_subset_per_token():
    experts = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 2.0]],
        ]
    )
    target = torch.tensor([[1.0, 1.0], [1.0, 2.0]])

    result = exact_direct_oracle(target, experts, k=2)

    assert result.subsets.tolist() == [[0, 1], [1, 2]]
    torch.testing.assert_close(result.squared_errors, torch.zeros(2))


def test_direct_oracle_uses_lexicographic_tie_breaking():
    experts = torch.zeros(1, 3, 2)

    result = exact_direct_oracle(torch.zeros(1, 2), experts, k=1)

    assert result.subsets.tolist() == [[0]]
    assert result.squared_errors.tolist() == [0.0]


def test_direct_oracle_selecting_all_experts_matches_literal_error():
    experts = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [-0.5, 0.5]]])
    target = torch.tensor([[1.0, 1.0]])

    result = exact_direct_oracle(target, experts, k=3)

    assert result.subsets.tolist() == [[0, 1, 2]]
    torch.testing.assert_close(result.squared_errors, torch.tensor([2.5]))


@pytest.mark.parametrize("k", [0, 4])
def test_direct_oracle_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="1 <= k <= 3"):
        exact_direct_oracle(
            torch.zeros(1, 2),
            torch.zeros(1, 3, 2),
            k=k,
        )


def test_direct_oracle_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="tokens and hidden"):
        exact_direct_oracle(
            torch.zeros(2, 4),
            torch.zeros(3, 8, 4),
            k=2,
        )
