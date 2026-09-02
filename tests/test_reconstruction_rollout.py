from copy import deepcopy

import pytest
import torch

import ppmoe.oracle.reconstruction_rollout as reconstruction_rollout
from ppmoe.evaluation.perplexity import EvaluationBatch
from ppmoe.moefication.qwen35 import convert_qwen35_text_ffns
from ppmoe.oracle.reconstruction import ReconstructionCondition, SolverSettings
from ppmoe.oracle.reconstruction_rollout import (
    run_reconstruction_rollout,
    select_reconstruction_h,
)
from tests.helpers import tiny_qwen35


def test_reconstruction_h_selects_teacher_next_minus_student_baseline() -> None:
    """The rollout objective is a correction, not the teacher FFN output."""

    target_next = torch.tensor([[2.0]])
    residual_baseline = torch.tensor([[10.0]])
    experts = torch.tensor([[[2.0], [-8.0], [5.0], [6.0], [7.0], [8.0], [9.0], [11.0]]])

    result = select_reconstruction_h(
        target_next,
        residual_baseline,
        experts,
        ReconstructionCondition("unit"),
        1,
        SolverSettings(),
    )

    assert result.subsets.tolist() == [[1]]


def _models_and_batch() -> tuple[torch.nn.Module, torch.nn.Module, EvaluationBatch]:
    torch.manual_seed(97)
    teacher = tiny_qwen35()
    student = deepcopy(teacher)
    convert_qwen35_text_ffns(student, num_experts=8, seed_base=97)
    return teacher, student, EvaluationBatch(
        0,
        torch.tensor([[1, 2, 3, 0]]),
        torch.tensor([[1, 1, 1, 0]]),
    )


def test_reconstruction_h_rollout_is_atomic_and_resume_does_not_double_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    teacher, student, batch = _models_and_batch()
    condition = ReconstructionCondition("ridge", 1.0)
    first = run_reconstruction_rollout(
        teacher,
        student,
        (batch,),
        condition,
        4,
        tmp_path,
        "ridge-k4",
        source_hashes={"stage:local": "a" * 64},
    )

    assert first.complete is True
    assert first.completed_batches == 1
    assert first.failed_batches == 0
    assert first.scored_tokens == 2
    assert first.condition_id == "ridge:1"
    assert all(metric.token_count == 3 for metric in first.layer_metrics.values())
    assert set(first.reference_distances[0]) == {"D0|E8", "D0|R8[ridge:1]", "E8|R8[ridge:1]"}
    assert first.peak_cuda_bytes == 0
    assert first.source_hashes == {"stage:local": "a" * 64}

    monkeypatch.setattr(
        student.model.language_model,
        "forward",
        lambda *_args, **_kwargs: pytest.fail("completed batch was rescored"),
    )
    assert run_reconstruction_rollout(
        teacher, student, (batch,), condition, 4, tmp_path, "ridge-k4"
    ) == first

    teacher, student, batch = _models_and_batch()
    real_select = reconstruction_rollout.select_reconstruction_h
    calls = 0

    def fail_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected layer-one failure")
        return real_select(*args, **kwargs)

    monkeypatch.setattr(reconstruction_rollout, "select_reconstruction_h", fail_on_second)
    failed = run_reconstruction_rollout(
        teacher, student, (batch,), condition, 4, tmp_path / "failed", "failed"
    )
    assert failed.completed_batches == 0
    assert failed.failed_batches == 1
    assert all(metric.token_count == 0 for metric in failed.layer_metrics.values())
