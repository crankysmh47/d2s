import pytest
import torch

from ppmoe.moefication.neuron import SwiGLUParameters, validate_permutation


def test_validate_permutation_returns_detached_long_tensor():
    permutation = torch.tensor([2, 0, 1], dtype=torch.int32, requires_grad=False)

    result = validate_permutation(permutation, 3)

    assert result.dtype == torch.long
    assert result.tolist() == [2, 0, 1]
    assert result.data_ptr() != permutation.data_ptr()


@pytest.mark.parametrize(
    ("values", "size", "message"),
    [
        ([0, 1], 3, "length"),
        ([0, 0, 2], 3, "exactly once"),
        ([-1, 0, 1], 3, "range"),
        ([0, 1, 3], 3, "range"),
    ],
)
def test_validate_permutation_rejects_invalid_indices(values, size, message):
    with pytest.raises(ValueError, match=message):
        validate_permutation(torch.tensor(values), size)


def test_validate_permutation_rejects_non_integer_values():
    with pytest.raises(ValueError, match="integer"):
        validate_permutation(torch.tensor([0.0, 1.0, 2.0]), 3)


def test_swiglu_parameters_checks_neuron_axes():
    with pytest.raises(ValueError, match="intermediate"):
        SwiGLUParameters(
            gate_weight=torch.zeros(4, 3),
            up_weight=torch.zeros(5, 3),
            down_weight=torch.zeros(3, 4),
        )


def test_swiglu_parameters_checks_optional_bias_axes():
    with pytest.raises(ValueError, match="gate bias"):
        SwiGLUParameters(
            gate_weight=torch.zeros(4, 3),
            up_weight=torch.zeros(4, 3),
            down_weight=torch.zeros(3, 4),
            gate_bias=torch.zeros(3),
        )


def test_swiglu_parameters_reports_sizes():
    parameters = SwiGLUParameters(
        gate_weight=torch.zeros(4, 3),
        up_weight=torch.zeros(4, 3),
        down_weight=torch.zeros(3, 4),
    )

    assert parameters.intermediate_size == 4
    assert parameters.hidden_size == 3
