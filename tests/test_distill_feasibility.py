from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ppmoe.distill.feasibility import (
    REQUIRED_STUDENT_FILES,
    assert_only_adapters_trainable,
    compute_memory_cap,
    validate_memory_peaks,
    validate_result,
    validate_source_inventory,
    validate_student_config,
)
from scripts.distillation_feasibility import (
    _environment_record,
    _write_failure_artifacts,
    run_workers_in_order,
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
            "peak_allocated_bytes": 1024,
            "peak_reserved_bytes": 2048,
            "memory_cap_bytes": 4096,
        },
        "teacher_generation": {
            "generated_tokens": 7,
            "stop_reason": "eos",
            "duration_seconds": 3.0,
            "peak_allocated_bytes": 1024,
            "peak_reserved_bytes": 2048,
            "memory_cap_bytes": 4096,
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
        ("peak_reserved_bytes", 4097, "memory exceeds cap"),
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


def test_memory_cap_is_physical_capacity_limited_with_two_gib_headroom() -> None:
    cap = compute_memory_cap(physical_capacity_bytes=20 * 1024**3)
    assert cap == 14 * 1024**3
    assert compute_memory_cap(physical_capacity_bytes=15 * 1024**3) == 13 * 1024**3


@pytest.mark.parametrize(
    ("allocated", "reserved"),
    [(101, 100), (100, 101)],
)
def test_memory_gate_rejects_either_allocator_peak_over_cap(allocated: int, reserved: int) -> None:
    with pytest.raises(ValueError, match="memory exceeds cap"):
        validate_memory_peaks(allocated, reserved, 100, phase="student")


def test_memory_gate_accepts_peaks_within_cap() -> None:
    validate_memory_peaks(100, 100, 100, phase="teacher")


def test_teacher_result_accepts_early_eos() -> None:
    validate_result(_valid_result())


def test_teacher_result_accepts_eos_at_sixteen_tokens() -> None:
    result = _valid_result()
    result["teacher_generation"]["generated_tokens"] = 16  # type: ignore[index]
    validate_result(result)


@pytest.mark.parametrize(
    ("count", "reason"),
    [(0, "eos"), (17, "length"), (7, "length"), (16, "other")],
)
def test_teacher_result_rejects_invalid_token_stop_evidence(count: int, reason: str) -> None:
    result = _valid_result()
    result["teacher_generation"]["generated_tokens"] = count  # type: ignore[index]
    result["teacher_generation"]["stop_reason"] = reason  # type: ignore[index]
    with pytest.raises(ValueError, match="teacher generation stop evidence"):
        validate_result(result)


def test_source_inventory_requires_all_declared_model_files(tmp_path: Path) -> None:
    import hashlib

    records = []
    for relative in REQUIRED_STUDENT_FILES:
        if relative == "model-00002-of-00002.safetensors":
            continue
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        records.append({"relative_path": relative, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {
        "repo_id": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "files": records,
    }
    with pytest.raises(ValueError, match="required student source files missing"):
        validate_source_inventory(
            tmp_path,
            manifest,
            expected_repo_id="Qwen/Qwen3-1.7B",
            expected_revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        )


def test_missing_teacher_config_still_writes_environment_and_failure_evidence(tmp_path: Path) -> None:
    env = _environment_record(
        project_root=Path.cwd(),
        teacher_path=tmp_path / "missing-teacher",
        torch_environment={"cuda_available": False, "bf16_supported": False},
        device_snapshot={"status": "unavailable"},
    )
    assert env["torch_environment"]["cuda_available"] is False
    assert env["model_hashes"]["teacher_config"] is None
    assert env["hashes"]["uv.lock"]
    assert env["dependencies_lock"]["accelerate"]["version"] == "1.15.0"
    _write_failure_artifacts(tmp_path / "failure", RuntimeError("download failed"), env)
    assert (tmp_path / "failure" / "environment.json").is_file()
    assert (tmp_path / "failure" / "diagnostic.json").is_file()
    assert (tmp_path / "failure" / "result.json").is_file()


def test_worker_phases_are_ordered_and_student_failure_skips_teacher(tmp_path: Path) -> None:
    calls: list[str] = []

    def passing_worker(phase: str, result_path: Path) -> int:
        calls.append(phase)
        result_path.write_text('{"status":"passed"}', encoding="utf-8")
        return 0

    assert run_workers_in_order(tmp_path / "passing", passing_worker) == ["student_update", "teacher_generation"]
    assert calls == ["student_update", "teacher_generation"]

    calls.clear()

    def failing_worker(phase: str, _result_path: Path) -> int:
        calls.append(phase)
        return 1

    with pytest.raises(RuntimeError, match="student_update worker exited"):
        run_workers_in_order(tmp_path / "failing", failing_worker)
    assert calls == ["student_update"]
