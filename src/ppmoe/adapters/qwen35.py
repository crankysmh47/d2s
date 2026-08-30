"""Read-only Qwen3.5 text configuration and MLP layout inspection."""

from dataclasses import dataclass
from typing import Any

from torch import nn


@dataclass(frozen=True, slots=True)
class Qwen35ConfigReport:
    model_type: str
    text_model_type: str
    hidden_size: int
    intermediate_size: int
    expert_width: int
    num_experts: int
    num_hidden_layers: int
    hidden_act: str


@dataclass(frozen=True, slots=True)
class Qwen35MLPReport:
    hidden_size: int
    intermediate_size: int
    gate_weight_shape: tuple[int, int]
    up_weight_shape: tuple[int, int]
    down_weight_shape: tuple[int, int]
    gate_bias: bool
    up_bias: bool
    down_bias: bool


def _required_attribute(value: Any, name: str) -> Any:
    if not hasattr(value, name):
        raise ValueError(f"Qwen3.5 text config is missing {name}")
    return getattr(value, name)


def inspect_qwen35_config(
    config: Any,
    *,
    num_experts: int,
) -> Qwen35ConfigReport:
    """Inspect only public Qwen3.5 config fields needed by conversion."""

    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    text_config = getattr(config, "text_config", config)
    hidden_size = int(_required_attribute(text_config, "hidden_size"))
    intermediate_size = int(
        _required_attribute(text_config, "intermediate_size")
    )
    num_hidden_layers = int(
        _required_attribute(text_config, "num_hidden_layers")
    )
    hidden_act = str(_required_attribute(text_config, "hidden_act"))
    if hidden_size <= 0 or intermediate_size <= 0 or num_hidden_layers <= 0:
        raise ValueError("Qwen3.5 text dimensions and layer count must be positive")
    if intermediate_size % num_experts != 0:
        raise ValueError(
            f"intermediate size {intermediate_size} must be divisible by "
            f"num_experts {num_experts}"
        )
    return Qwen35ConfigReport(
        model_type=str(getattr(config, "model_type", "")),
        text_model_type=str(getattr(text_config, "model_type", "")),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        expert_width=intermediate_size // num_experts,
        num_experts=num_experts,
        num_hidden_layers=num_hidden_layers,
        hidden_act=hidden_act,
    )


def inspect_qwen35_mlp(module: nn.Module) -> Qwen35MLPReport:
    """Validate Qwen-style gate/up row and down column projection shapes."""

    projections: dict[str, nn.Linear] = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        projection = getattr(module, name, None)
        if not isinstance(projection, nn.Linear):
            raise ValueError(f"Qwen3.5 text MLP requires nn.Linear {name}")
        projections[name] = projection

    gate = projections["gate_proj"]
    up = projections["up_proj"]
    down = projections["down_proj"]
    intermediate_size, hidden_size = gate.weight.shape
    expected_gate_up = (intermediate_size, hidden_size)
    expected_down = (hidden_size, intermediate_size)
    if tuple(up.weight.shape) != expected_gate_up:
        raise ValueError(
            f"up_proj expected shape {expected_gate_up}; "
            f"actual {tuple(up.weight.shape)}"
        )
    if tuple(down.weight.shape) != expected_down:
        raise ValueError(
            f"down_proj expected shape {expected_down}; "
            f"actual {tuple(down.weight.shape)}"
        )
    return Qwen35MLPReport(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        gate_weight_shape=tuple(gate.weight.shape),
        up_weight_shape=tuple(up.weight.shape),
        down_weight_shape=tuple(down.weight.shape),
        gate_bias=gate.bias is not None,
        up_bias=up.bias is not None,
        down_bias=down.bias is not None,
    )
