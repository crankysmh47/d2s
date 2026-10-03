"""Validation primitives used by the bounded QLoRA compatibility smoke."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


STUDENT_MODEL_ID = "Qwen/Qwen3-1.7B"
STUDENT_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TEACHER_MODEL_ID = "Qwen/Qwen3-4B"
TEACHER_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"

_STUDENT_CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "num_hidden_layers": 28,
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
}


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def validate_student_config(config: Any) -> None:
    """Reject a config that is not the fixed Qwen3-1.7B architecture."""
    mismatches = {
        name: (_config_value(config, name), expected)
        for name, expected in _STUDENT_CONFIG.items()
        if _config_value(config, name) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in mismatches.items()
        )
        raise ValueError(f"student config does not match pinned architecture: {details}")


def assert_only_adapters_trainable(model: Any) -> list[str]:
    """Assert the base is frozen and at least one LoRA adapter is trainable."""
    named_parameters = list(model.named_parameters())
    trainable = [name for name, parameter in named_parameters if parameter.requires_grad]
    invalid = [name for name in trainable if "lora_" not in name.lower()]
    if invalid:
        raise ValueError(f"non-adapter parameters are trainable: {invalid[:8]}")
    if not trainable:
        raise ValueError("no trainable adapter parameters")
    return trainable


def validate_result(result: Mapping[str, Any]) -> None:
    """Check durable smoke evidence for complete and physically plausible metrics."""
    if result.get("schema_version") != 1:
        raise ValueError("unsupported feasibility result schema")
    if result.get("status") != "passed":
        raise ValueError("feasibility result status must be passed")

    student = result.get("student_update")
    if not isinstance(student, Mapping):
        raise ValueError("student update evidence is missing")
    for field in ("loss", "adapter_gradient_norm", "duration_seconds"):
        value = student.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"student {field} must be finite")
        if field == "loss" and value < 0:
            raise ValueError("student loss must be nonnegative")
        if field == "adapter_gradient_norm" and value <= 0:
            raise ValueError("student adapter gradient must be nonzero")
        if field == "duration_seconds" and value < 0:
            raise ValueError("student duration must be nonnegative")
    changed = student.get("changed_parameter_count")
    if not isinstance(changed, int) or isinstance(changed, bool) or changed <= 0:
        raise ValueError("student changed parameter count must be positive")
    for field in ("peak_memory_bytes",):
        value = student.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"student {field} must be a nonnegative integer")

    teacher = result.get("teacher_generation")
    if not isinstance(teacher, Mapping):
        raise ValueError("teacher generation evidence is missing")
    if teacher.get("generated_tokens") != 16:
        raise ValueError("teacher must generate exactly 16 new tokens")
    duration = teacher.get("duration_seconds")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
        raise ValueError("teacher duration must be finite and nonnegative")
    memory = teacher.get("peak_memory_bytes")
    if not isinstance(memory, int) or isinstance(memory, bool) or memory < 0:
        raise ValueError("teacher peak_memory_bytes must be a nonnegative integer")
