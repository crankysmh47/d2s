"""Small compatibility checks for the verified-distillation QLoRA pilot."""

from .feasibility import (
    STUDENT_MODEL_ID,
    STUDENT_REVISION,
    assert_only_adapters_trainable,
    validate_result,
    validate_student_config,
)

__all__ = [
    "STUDENT_MODEL_ID",
    "STUDENT_REVISION",
    "assert_only_adapters_trainable",
    "validate_result",
    "validate_student_config",
]
