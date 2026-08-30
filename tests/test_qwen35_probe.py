from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ppmoe.adapters.qwen35 import inspect_qwen35_config, inspect_qwen35_mlp


def _composite_config(intermediate_size: int = 3584) -> SimpleNamespace:
    return SimpleNamespace(
        model_type="qwen3_5",
        text_config=SimpleNamespace(
            model_type="qwen3_5_text",
            hidden_size=1024,
            intermediate_size=intermediate_size,
            num_hidden_layers=24,
            hidden_act="silu",
        ),
    )


def test_config_inspection_selects_text_config_and_derives_expert_width():
    report = inspect_qwen35_config(_composite_config(), num_experts=8)

    assert report.model_type == "qwen3_5"
    assert report.text_model_type == "qwen3_5_text"
    assert report.hidden_size == 1024
    assert report.intermediate_size == 3584
    assert report.expert_width == 448
    assert report.num_hidden_layers == 24
    assert report.hidden_act == "silu"


def test_config_inspection_rejects_non_divisible_expert_width():
    with pytest.raises(ValueError, match="divisible"):
        inspect_qwen35_config(_composite_config(intermediate_size=3585), num_experts=8)


class _TextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 16,
        intermediate_size: int = 32,
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)


def test_mlp_inspection_reports_row_row_column_layout_and_biases():
    report = inspect_qwen35_mlp(_TextMLP(hidden_size=16, intermediate_size=32))

    assert report.hidden_size == 16
    assert report.intermediate_size == 32
    assert report.gate_weight_shape == (32, 16)
    assert report.up_weight_shape == (32, 16)
    assert report.down_weight_shape == (16, 32)
    assert report.gate_bias is False
    assert report.up_bias is False
    assert report.down_bias is False


def test_mlp_inspection_reports_present_biases():
    report = inspect_qwen35_mlp(_TextMLP(bias=True))

    assert report.gate_bias is True
    assert report.up_bias is True
    assert report.down_bias is True


def test_mlp_inspection_rejects_missing_projection():
    module = _TextMLP()
    del module.up_proj

    with pytest.raises(ValueError, match="up_proj"):
        inspect_qwen35_mlp(module)


def test_mlp_inspection_rejects_incompatible_projection_shape():
    module = _TextMLP()
    module.down_proj = nn.Linear(31, 16, bias=False)

    with pytest.raises(ValueError, match=r"expected.*\(16, 32\).*actual.*\(16, 31\)"):
        inspect_qwen35_mlp(module)
