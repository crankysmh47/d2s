from copy import deepcopy
import math

import pytest
import torch
import torch.nn.functional as F

import ppmoe.oracle.rollout as rollout
from ppmoe.evaluation.perplexity import EvaluationBatch
from ppmoe.moefication.qwen35 import convert_qwen35_text_ffns
from ppmoe.oracle.rollout import run_rollout_oracle, select_rollout_subsets
from tests.helpers import tiny_qwen35


def _models_and_batch():
    torch.manual_seed(97)
    teacher = tiny_qwen35()
    student = deepcopy(teacher)
    convert_qwen35_text_ffns(student, num_experts=8, seed_base=97)
    batch = EvaluationBatch(
        0,
        torch.tensor([[1, 2, 3, 0]]),
        torch.tensor([[1, 1, 1, 0]]),
    )
    return teacher, student, batch


def test_select_rollout_subsets_accounts_for_residual_and_breaks_ties() -> None:
    target_next = torch.tensor([[3.0], [1.0]])
    residual = torch.tensor([[1.0], [1.0]])
    experts = torch.tensor([[[2.0], [1.0]], [[1.0], [-1.0]]])

    selected = select_rollout_subsets(target_next, residual, experts, k=1)

    assert selected.subsets.tolist() == [[0], [0]]


def test_rollout_k8_advances_sequentially_uses_text_stack_and_scores_masks(
    monkeypatch, tmp_path
) -> None:
    teacher, student, batch = _models_and_batch()
    monkeypatch.setattr(
        teacher,
        "forward",
        lambda *_args, **_kwargs: pytest.fail("composite forward must not run"),
    )

    result = run_rollout_oracle(
        teacher, student, (batch,), k=8, checkpoint_dir=tmp_path, run_signature="k8"
    )

    assert result.complete is True
    assert result.completed_batches == 1
    assert result.scored_tokens == 2
    assert result.ppl is not None and math.isfinite(result.ppl)
    with torch.inference_mode():
        dense_hidden = teacher.model.language_model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
        ).last_hidden_state
        dense_logits = teacher.lm_head(dense_hidden)
        valid_labels = batch.attention_mask[..., 1:].bool() & batch.attention_mask[..., :-1].bool()
        dense_nll = F.cross_entropy(
            dense_logits[..., :-1, :][valid_labels],
            batch.input_ids[..., 1:][valid_labels],
            reduction="sum",
        )
    assert result.ppl == pytest.approx(math.exp(float(dense_nll) / 2), rel=1e-6)
    for layer in range(2):
        assert result.layer_metrics[layer].token_count == 3
        assert result.layer_metrics[layer].normalized_mse <= 1e-8
        assert sum(result.selection_counts[layer]) == 3 * 8


def test_rollout_k4_is_finite_and_batch_failure_is_transactional(monkeypatch, tmp_path) -> None:
    teacher, student, batch = _models_and_batch()
    result = run_rollout_oracle(
        teacher, student, (batch,), k=4, checkpoint_dir=tmp_path / "finite", run_signature="k4"
    )
    assert result.complete is True
    assert result.ppl is not None and math.isfinite(result.ppl)
    assert all(math.isfinite(metric.normalized_mse) for metric in result.layer_metrics.values())

    teacher, student, batch = _models_and_batch()
    real_select = rollout.select_rollout_subsets
    calls = 0

    def fail_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected layer-one failure")
        return real_select(*args, **kwargs)

    monkeypatch.setattr(rollout, "select_rollout_subsets", fail_on_second)
    failed = run_rollout_oracle(
        teacher, student, (batch,), k=4, checkpoint_dir=tmp_path / "failed", run_signature="failed"
    )
    assert failed.failed_batches == 1
    assert failed.layer_metrics[0].token_count == 0


def test_rollout_resume_skips_completed_batch_and_guards_signature(monkeypatch, tmp_path) -> None:
    teacher, student, batch = _models_and_batch()
    first = run_rollout_oracle(
        teacher, student, (batch,), k=4, checkpoint_dir=tmp_path, run_signature="resume"
    )
    monkeypatch.setattr(
        student.model.language_model,
        "forward",
        lambda *_args, **_kwargs: pytest.fail("completed batch was rescored"),
    )
    resumed = run_rollout_oracle(
        teacher, student, (batch,), k=4, checkpoint_dir=tmp_path, run_signature="resume"
    )
    assert resumed == first
    with pytest.raises(ValueError, match="signature"):
        run_rollout_oracle(
            teacher, student, (), k=4, checkpoint_dir=tmp_path, run_signature="different"
        )
