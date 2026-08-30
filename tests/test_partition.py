import pytest
import torch

from ppmoe.moefication.partition import balanced_partition


def test_balanced_partition_preserves_order_and_covers_indices():
    permutation = torch.tensor([4, 0, 6, 2, 7, 1, 5, 3])

    groups = balanced_partition(permutation, 4)

    assert [group.tolist() for group in groups] == [
        [4, 0],
        [6, 2],
        [7, 1],
        [5, 3],
    ]


def test_balanced_partition_rejects_non_divisible_width():
    with pytest.raises(ValueError, match="divisible"):
        balanced_partition(torch.arange(7), 4)


def test_balanced_partition_rejects_non_positive_expert_count():
    with pytest.raises(ValueError, match="positive"):
        balanced_partition(torch.arange(8), 0)


def test_balanced_partition_rejects_non_permutation():
    with pytest.raises(ValueError, match="exactly once"):
        balanced_partition(torch.tensor([0, 0, 1, 2]), 2)
