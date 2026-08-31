from pathlib import Path

import pytest
import torch
from torch import nn

from ppmoe.config import Phase1Config
from ppmoe.adapters.qwen35_runtime import (
    iter_text_mlps,
    load_qwen35_pair,
    validate_loaded_qwen35,
)
import ppmoe.adapters.qwen35_runtime as runtime
from tests.helpers import tiny_qwen35


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "phase1-v0.toml"


def test_validation_accepts_concrete_composite_class_and_actual_bf16() -> None:
    model = tiny_qwen35(dtype=torch.bfloat16)

    report = validate_loaded_qwen35(
        model,
        expected_dtype=torch.bfloat16,
        expected_num_hidden_layers=2,
        expected_hidden_size=16,
        expected_intermediate_size=32,
    )

    assert report.model_class == "Qwen3_5ForConditionalGeneration"
    assert report.text_layers == 2
    assert report.floating_dtypes == ("torch.bfloat16",)
    assert report.parameter_count == sum(p.numel() for p in model.parameters())
    assert report.validated_num_hidden_layers == 2
    assert report.validated_hidden_size == 16
    assert report.validated_intermediate_size == 32
    assert len(report.mlp_shapes) == 2
    assert report.mlp_shapes[0].gate_proj == (32, 16)
    assert report.mlp_shapes[0].up_proj == (32, 16)
    assert report.mlp_shapes[0].down_proj == (16, 32)
    assert [index for index, _ in iter_text_mlps(model)] == [0, 1]


def test_validation_rejects_wrong_class_or_parameter_dtype() -> None:
    with pytest.raises(TypeError, match="Qwen3_5ForConditionalGeneration"):
        validate_loaded_qwen35(
            nn.Linear(2, 2),
            expected_dtype=torch.float32,
            expected_num_hidden_layers=2,
            expected_hidden_size=16,
            expected_intermediate_size=32,
        )

    model = tiny_qwen35(dtype=torch.float32)
    with pytest.raises(ValueError, match="parameter dtypes"):
        validate_loaded_qwen35(
            model,
            expected_dtype=torch.bfloat16,
            expected_num_hidden_layers=2,
            expected_hidden_size=16,
            expected_intermediate_size=32,
        )

    model = tiny_qwen35(dtype=torch.bfloat16)
    model.model.language_model.layers[1].mlp.down_proj = nn.Linear(31, 16, bias=False)
    with pytest.raises(ValueError, match="layer 1 down projection"):
        validate_loaded_qwen35(
            model,
            expected_dtype=torch.bfloat16,
            expected_num_hidden_layers=2,
            expected_hidden_size=16,
            expected_intermediate_size=32,
        )


def test_pair_loader_passes_locked_expectations_and_returns_distinct_models(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    validation_calls: list[dict[str, object]] = []

    def loader(repo_id: str, **kwargs):
        calls.append((repo_id, kwargs))
        return tiny_qwen35(dtype=torch.bfloat16)

    def validate(_model, **kwargs):
        validation_calls.append(kwargs)
        return object()

    monkeypatch.setattr(runtime, "validate_loaded_qwen35", validate)

    pair = load_qwen35_pair(
        Phase1Config.load(CONFIG_PATH),
        device=torch.device("cpu"),
        loader=loader,
    )

    assert pair.teacher is not pair.student
    assert len(calls) == 2
    for repo_id, kwargs in calls:
        assert repo_id == "Qwen/Qwen3.5-0.8B-Base"
        assert kwargs == {
            "revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
            "dtype": torch.bfloat16,
            "trust_remote_code": False,
        }
    assert pair.free_cuda_bytes is None
    assert pair.total_cuda_bytes is None
    assert pair.peak_cuda_bytes == 0
    assert validation_calls == [
        {
            "expected_dtype": torch.bfloat16,
            "expected_num_hidden_layers": 24,
            "expected_hidden_size": 1024,
            "expected_intermediate_size": 3584,
        },
        {
            "expected_dtype": torch.bfloat16,
            "expected_num_hidden_layers": 24,
            "expected_hidden_size": 1024,
            "expected_intermediate_size": 3584,
        },
    ]


def test_pair_loader_requires_four_gib_cuda_headroom(monkeypatch) -> None:
    models = iter([tiny_qwen35(dtype=torch.bfloat16), tiny_qwen35(dtype=torch.bfloat16)])
    insufficient_models = iter([tiny_qwen35(dtype=torch.bfloat16), tiny_qwen35(dtype=torch.bfloat16)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 123)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (4 * 1024**3, 8 * 1024**3))
    monkeypatch.setattr(nn.Module, "to", lambda self, _device: self)
    monkeypatch.setattr(runtime, "validate_loaded_qwen35", lambda *_args, **_kwargs: object())

    pair = load_qwen35_pair(
        Phase1Config.load(CONFIG_PATH),
        device=torch.device("cuda"),
        loader=lambda *_args, **_kwargs: next(models),
    )

    assert pair.free_cuda_bytes == 4 * 1024**3
    assert pair.total_cuda_bytes == 8 * 1024**3
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (4 * 1024**3 - 1, 8 * 1024**3))
    with pytest.raises(RuntimeError, match="4 GiB"):
        load_qwen35_pair(
            Phase1Config.load(CONFIG_PATH),
            device=torch.device("cuda"),
            loader=lambda *_args, **_kwargs: next(insufficient_models),
        )
