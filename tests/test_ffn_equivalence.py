import pytest
import torch

from ppmoe.moefication.reference import DenseSwiGLU, PartitionedSwiGLU


def test_all_experts_match_dense_fp32_with_down_bias_added_once():
    torch.manual_seed(11)
    dense = DenseSwiGLU(
        16,
        32,
        gate_bias=True,
        up_bias=True,
        down_bias=True,
    )
    partitioned = PartitionedSwiGLU.from_dense(
        dense,
        partitions=tuple(torch.arange(32).reshape(4, 8)),
    )
    x = torch.randn(2, 5, 16)

    expected = dense(x)
    actual = partitioned(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert partitioned.down_bias is not None
    assert all(expert.down_proj.bias is None for expert in partitioned.experts)


def test_selected_experts_returns_requested_sum_in_input_dtype():
    torch.manual_seed(13)
    dense = DenseSwiGLU(8, 16, down_bias=False)
    partitioned = PartitionedSwiGLU.from_dense(
        dense,
        partitions=tuple(torch.arange(16).reshape(4, 4)),
    )
    x = torch.randn(3, 8)

    expected = partitioned.expert_outputs(x)[:, [0, 2], :].sum(dim=1)
    actual = partitioned(x, selected_experts=(0, 2))

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == x.dtype


def test_down_bias_is_added_once_for_sparse_or_all_expert_execution():
    dense = DenseSwiGLU(2, 4, down_bias=True)
    with torch.no_grad():
        dense.gate_proj.weight.zero_()
        dense.up_proj.weight.zero_()
        dense.down_proj.weight.zero_()
        dense.down_proj.bias.copy_(torch.tensor([2.0, -3.0]))
    partitioned = PartitionedSwiGLU.from_dense(
        dense,
        partitions=(torch.tensor([0, 1]), torch.tensor([2, 3])),
    )
    x = torch.ones(1, 2)

    torch.testing.assert_close(partitioned(x), torch.tensor([[2.0, -3.0]]))
    torch.testing.assert_close(
        partitioned(x, selected_experts=(0,)),
        torch.tensor([[2.0, -3.0]]),
    )


@pytest.mark.parametrize(
    "partitions",
    [
        (torch.tensor([0, 1]), torch.tensor([1, 2])),
        (torch.tensor([0, 1]), torch.tensor([2])),
        (torch.tensor([0, 1]), torch.tensor([2, 4])),
    ],
)
def test_partitioned_swiglu_rejects_invalid_partitions(partitions):
    dense = DenseSwiGLU(2, 4)

    with pytest.raises(ValueError, match="partition"):
        PartitionedSwiGLU.from_dense(dense, partitions)


def test_partitioned_swiglu_rejects_empty_selection():
    dense = DenseSwiGLU(2, 4)
    partitioned = PartitionedSwiGLU.from_dense(
        dense,
        partitions=(torch.tensor([0, 1]), torch.tensor([2, 3])),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        partitioned(torch.ones(1, 2), selected_experts=())
