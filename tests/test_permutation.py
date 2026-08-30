import torch
import torch.nn.functional as F

from ppmoe.moefication.neuron import SwiGLUParameters, permute_swiglu_parameters


def _evaluate(parameters: SwiGLUParameters, x: torch.Tensor) -> torch.Tensor:
    gate = F.linear(x, parameters.gate_weight, parameters.gate_bias)
    up = F.linear(x, parameters.up_weight, parameters.up_bias)
    return F.linear(
        F.silu(gate) * up,
        parameters.down_weight,
        parameters.down_bias,
    )


def test_permutation_uses_rows_rows_and_columns_in_requested_order():
    parameters = SwiGLUParameters(
        gate_weight=torch.tensor([[10.0], [20.0], [30.0]]),
        up_weight=torch.tensor([[1.0], [2.0], [3.0]]),
        down_weight=torch.tensor([[100.0, 200.0, 300.0]]),
        gate_bias=torch.tensor([4.0, 5.0, 6.0]),
        up_bias=torch.tensor([7.0, 8.0, 9.0]),
        down_bias=torch.tensor([11.0]),
    )

    result = permute_swiglu_parameters(parameters, torch.tensor([2, 0, 1]))

    torch.testing.assert_close(
        result.gate_weight,
        torch.tensor([[30.0], [10.0], [20.0]]),
    )
    torch.testing.assert_close(
        result.up_weight,
        torch.tensor([[3.0], [1.0], [2.0]]),
    )
    torch.testing.assert_close(
        result.down_weight,
        torch.tensor([[300.0, 100.0, 200.0]]),
    )
    torch.testing.assert_close(result.gate_bias, torch.tensor([6.0, 4.0, 5.0]))
    torch.testing.assert_close(result.up_bias, torch.tensor([9.0, 7.0, 8.0]))
    torch.testing.assert_close(result.down_bias, torch.tensor([11.0]))


def test_permuting_complete_triplets_preserves_output_without_mutation():
    generator = torch.Generator().manual_seed(7)
    parameters = SwiGLUParameters(
        gate_weight=torch.randn(8, 4, generator=generator),
        up_weight=torch.randn(8, 4, generator=generator),
        down_weight=torch.randn(4, 8, generator=generator),
        gate_bias=torch.randn(8, generator=generator),
        up_bias=torch.randn(8, generator=generator),
        down_bias=torch.randn(4, generator=generator),
    )
    x = torch.randn(2, 3, 4, generator=generator)
    permutation = torch.tensor([5, 1, 7, 0, 3, 2, 6, 4])
    original_tensors = {
        field: getattr(parameters, field).clone()
        for field in (
            "gate_weight",
            "up_weight",
            "down_weight",
            "gate_bias",
            "up_bias",
            "down_bias",
        )
    }

    permuted = permute_swiglu_parameters(parameters, permutation)

    torch.testing.assert_close(
        _evaluate(permuted, x),
        _evaluate(parameters, x),
        rtol=1e-5,
        atol=1e-6,
    )
    for field, original in original_tensors.items():
        torch.testing.assert_close(getattr(parameters, field), original)
        assert getattr(permuted, field).data_ptr() != getattr(parameters, field).data_ptr()


def test_permutation_preserves_absent_biases():
    parameters = SwiGLUParameters(
        gate_weight=torch.zeros(3, 2),
        up_weight=torch.zeros(3, 2),
        down_weight=torch.zeros(2, 3),
    )

    result = permute_swiglu_parameters(parameters, torch.tensor([2, 1, 0]))

    assert result.gate_bias is None
    assert result.up_bias is None
    assert result.down_bias is None
