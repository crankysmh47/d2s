from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ppmoe.distill.feasibility import (
    assert_only_adapters_trainable,
    validate_result,
    validate_student_config,
)


def _student_config(**overrides: object) -> SimpleNamespace:
    values = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 28,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _valid_result() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "passed",
        "student_update": {
            "loss": 1.25,
            "adapter_gradient_norm": 0.5,
            "changed_parameter_count": 1,
            "duration_seconds": 12.0,
            "peak_memory_bytes": 1024,
        },
        "teacher_generation": {
            "generated_tokens": 16,
            "duration_seconds": 3.0,
            "peak_memory_bytes": 2048,
        },
    }


def test_result_validator_accepts_finite_completed_smoke() -> None:
    validate_result(_valid_result())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("loss", float("nan"), "loss"),
        ("adapter_gradient_norm", 0.0, "gradient"),
        ("changed_parameter_count", 0, "parameter"),
        ("duration_seconds", -1.0, "duration"),
    ],
)
def test_result_validator_rejects_invalid_student_evidence(
    field: str, value: object, message: str
) -> None:
    result = _valid_result()
    result["student_update"][field] = value  # type: ignore[index]
    with pytest.raises(ValueError, match=message):
        validate_result(result)


def test_student_config_accepts_pinned_qwen3_architecture() -> None:
    validate_student_config(_student_config())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "qwen2"),
        ("num_hidden_layers", 27),
        ("hidden_size", 1024),
        ("num_attention_heads", 8),
        ("num_key_value_heads", 4),
    ],
)
def test_student_config_rejects_architecture_drift(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="student config"):
        validate_student_config(_student_config(**{field: value}))


def test_frozen_base_check_accepts_trainable_lora_and_frozen_base() -> None:
    model = torch.nn.Module()
    model.base_weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    model.lora_A = torch.nn.Parameter(torch.ones(2, 2), requires_grad=True)

    assert assert_only_adapters_trainable(model) == ["lora_A"]


def test_frozen_base_check_rejects_trainable_base_parameter() -> None:
    model = torch.nn.Module()
    model.base_weight = torch.nn.Parameter(torch.ones(2), requires_grad=True)
    model.lora_A = torch.nn.Parameter(torch.ones(2, 2), requires_grad=True)

    with pytest.raises(ValueError, match="non-adapter parameters are trainable"):
        assert_only_adapters_trainable(model)


def test_frozen_base_check_rejects_missing_trainable_adapter() -> None:
    model = torch.nn.Module()
    model.base_weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)

    with pytest.raises(ValueError, match="no trainable adapter parameters"):
        assert_only_adapters_trainable(model)
