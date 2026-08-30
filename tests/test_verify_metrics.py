import math

import pytest
import torch

from ppmoe.moefication.verify import compare_tensors


def test_compare_tensors_reports_hand_computed_error_metrics():
    reference = torch.tensor([1.0, 2.0])
    candidate = torch.tensor([2.0, 2.0])

    metrics = compare_tensors(reference, candidate)

    assert metrics.max_absolute_error == 1.0
    assert math.isclose(metrics.rms_error, math.sqrt(0.5))
    assert math.isclose(metrics.relative_error, 1.0 / math.sqrt(5.0))
    assert math.isclose(metrics.normalized_mse, 1.0 / 5.0)
    assert math.isclose(metrics.cosine_similarity, 6.0 / math.sqrt(40.0))


def test_compare_tensors_treats_two_zero_tensors_as_identical():
    metrics = compare_tensors(torch.zeros(3), torch.zeros(3))

    assert metrics.max_absolute_error == 0.0
    assert metrics.rms_error == 0.0
    assert metrics.relative_error == 0.0
    assert metrics.normalized_mse == 0.0
    assert metrics.cosine_similarity == 1.0


def test_compare_tensors_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        compare_tensors(torch.zeros(2), torch.zeros(3))


def test_compare_tensors_rejects_empty_tensors():
    with pytest.raises(ValueError, match="empty"):
        compare_tensors(torch.tensor([]), torch.tensor([]))
