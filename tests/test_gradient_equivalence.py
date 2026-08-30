import torch

from ppmoe.moefication.reference import DenseSwiGLU, PartitionedSwiGLU


def _restore_neuron_axis_gradients(
    dense: DenseSwiGLU,
    partitioned: PartitionedSwiGLU,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate = torch.zeros_like(dense.gate_proj.weight)
    up = torch.zeros_like(dense.up_proj.weight)
    down = torch.zeros_like(dense.down_proj.weight)
    for expert in partitioned.experts:
        gate.index_copy_(0, expert.indices, expert.gate_proj.weight.grad)
        up.index_copy_(0, expert.indices, expert.up_proj.weight.grad)
        down.index_copy_(1, expert.indices, expert.down_proj.weight.grad)
    return gate, up, down


def test_all_expert_input_and_parameter_gradients_match_dense():
    torch.manual_seed(17)
    dense = DenseSwiGLU(8, 16, gate_bias=True, up_bias=True, down_bias=True)
    partitions = tuple(torch.tensor([7, 1, 12, 3, 9, 0, 15, 5, 2, 11, 4, 14, 6, 10, 8, 13]).reshape(4, 4))
    partitioned = PartitionedSwiGLU.from_dense(dense, partitions)
    dense_input = torch.randn(2, 3, 8, requires_grad=True)
    partitioned_input = dense_input.detach().clone().requires_grad_(True)

    dense(dense_input).square().mean().backward()
    partitioned(partitioned_input).square().mean().backward()
    restored_gate, restored_up, restored_down = _restore_neuron_axis_gradients(
        dense,
        partitioned,
    )

    torch.testing.assert_close(
        partitioned_input.grad,
        dense_input.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        restored_gate,
        dense.gate_proj.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        restored_up,
        dense.up_proj.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        restored_down,
        dense.down_proj.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        partitioned.down_bias.grad,
        dense.down_proj.bias.grad,
        rtol=1e-5,
        atol=1e-6,
    )

