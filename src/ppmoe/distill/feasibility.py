"""Validation primitives used by the bounded QLoRA compatibility smoke."""

from __future__ import annotations

import math
from collections.abc import Mapping
import hashlib
from pathlib import Path
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

REQUIRED_STUDENT_FILES = frozenset({
    "LICENSE",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
})
VRAM_LIMIT_BYTES = 16 * 1024**3
VRAM_HEADROOM_BYTES = 2 * 1024**3


def compute_memory_cap(
    physical_capacity_bytes: int,
    *,
    configured_limit_bytes: int = VRAM_LIMIT_BYTES,
    headroom_bytes: int = VRAM_HEADROOM_BYTES,
) -> int:
    """Return min(configured limit, physical VRAM) with reserved headroom."""
    values = (physical_capacity_bytes, configured_limit_bytes, headroom_bytes)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise ValueError("memory capacities and headroom must be nonnegative integers")
    cap = min(physical_capacity_bytes, configured_limit_bytes) - headroom_bytes
    if cap <= 0:
        raise ValueError("physical VRAM is not greater than required headroom")
    return cap


def validate_memory_peaks(
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    cap_bytes: int,
    *,
    phase: str,
) -> None:
    """Reject either PyTorch allocator peak above the configured phase cap."""
    for name, value in (
        ("peak allocated", peak_allocated_bytes),
        ("peak reserved", peak_reserved_bytes),
        ("cap", cap_bytes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{phase} {name} bytes must be a nonnegative integer")
    if peak_allocated_bytes > cap_bytes or peak_reserved_bytes > cap_bytes:
        raise ValueError(
            f"{phase} memory exceeds cap: allocated={peak_allocated_bytes}, "
            f"reserved={peak_reserved_bytes}, cap={cap_bytes}"
        )


def validate_source_inventory(
    snapshot_root: Path,
    manifest: Mapping[str, Any],
    *,
    expected_repo_id: str = STUDENT_MODEL_ID,
    expected_revision: str = STUDENT_REVISION,
) -> dict[str, str]:
    """Verify the explicit model/license/tokenizer inventory used by the smoke."""
    if manifest.get("repo_id") != expected_repo_id or manifest.get("revision") != expected_revision:
        raise ValueError("student source manifest does not match the pinned repository revision")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("student source manifest has no valid file inventory")
    listed: dict[str, Mapping[str, Any]] = {}
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("student source inventory entry is malformed")
        relative = item.get("relative_path")
        if not isinstance(relative, str) or relative in listed:
            raise ValueError("student source inventory has an invalid or duplicate path")
        listed[relative] = item
    missing_entries = REQUIRED_STUDENT_FILES.difference(listed)
    missing_files = sorted(
        relative for relative in REQUIRED_STUDENT_FILES
        if not (snapshot_root / relative).is_file()
    )
    if missing_entries or missing_files:
        missing = sorted(missing_entries.union(missing_files))
        raise ValueError(f"required student source files missing: {missing}")

    inventory: dict[str, str] = {}
    for relative in sorted(REQUIRED_STUDENT_FILES):
        item = listed[relative]
        path = snapshot_root / relative
        expected_size = item.get("bytes")
        expected_sha = item.get("sha256")
        if not isinstance(expected_size, int) or isinstance(expected_size, bool):
            raise ValueError(f"student source inventory byte count is invalid: {relative}")
        if path.stat().st_size != expected_size:
            raise ValueError(f"student source size mismatch: {relative}")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"student source hash is invalid: {relative}")
        hasher = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
        if digest != expected_sha:
            raise ValueError(f"student source hash mismatch: {relative}")
        inventory[relative] = digest
    return inventory


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
    cap = student.get("memory_cap_bytes")
    for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
        value = student.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"student {field} must be a nonnegative integer")
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
        raise ValueError("student memory_cap_bytes must be a nonnegative integer")
    validate_memory_peaks(
        student["peak_allocated_bytes"], student["peak_reserved_bytes"], cap, phase="student"
    )

    teacher = result.get("teacher_generation")
    if not isinstance(teacher, Mapping):
        raise ValueError("teacher generation evidence is missing")
    generated_tokens = teacher.get("generated_tokens")
    stop_reason = teacher.get("stop_reason")
    if (
        not isinstance(generated_tokens, int)
        or isinstance(generated_tokens, bool)
        or not 1 <= generated_tokens <= 16
        or stop_reason not in {"eos", "max_new_tokens"}
        or (stop_reason == "max_new_tokens" and generated_tokens != 16)
    ):
        raise ValueError("teacher generation stop evidence is invalid")
    duration = teacher.get("duration_seconds")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
        raise ValueError("teacher duration must be finite and nonnegative")
    cap = teacher.get("memory_cap_bytes")
    allocated = teacher.get("peak_allocated_bytes")
    reserved = teacher.get("peak_reserved_bytes")
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
        raise ValueError("teacher memory_cap_bytes must be a nonnegative integer")
    validate_memory_peaks(allocated, reserved, cap, phase="teacher")
