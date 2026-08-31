from math import log
from pathlib import Path

import pytest
import torch
from torch import nn

from ppmoe.evaluation.perplexity import (
    EvaluationBatch,
    M0GateError,
    NLLAccumulator,
    evaluate_m0,
    evaluate_perplexity,
)


def test_nll_accumulator_uses_causal_shift_masks_and_fp64_totals() -> None:
    accumulator = NLLAccumulator()
    logits = torch.tensor(
        [[[0.0, log(3.0)], [log(3.0), 0.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[0, 1, 0]])
    attention_mask = torch.tensor([[1, 1, 1]])

    accumulator.update(logits, input_ids, attention_mask, batch_index=0)
    result = accumulator.finalize()

    assert result.scored_tokens == 2
    assert result.nll_sum == pytest.approx(2 * -log(0.75), rel=1e-6)
    assert result.ppl == pytest.approx(4 / 3, rel=1e-6)
    assert result.failed_batches == 0
    assert result.complete is True


def test_nll_accumulator_excludes_padding_and_rejects_duplicate_batch() -> None:
    accumulator = NLLAccumulator()
    logits = torch.zeros(1, 4, 2)
    input_ids = torch.tensor([[0, 1, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 0, 0]])

    accumulator.update(logits, input_ids, attention_mask, batch_index=3)

    assert accumulator.finalize().scored_tokens == 1
    with pytest.raises(ValueError, match="already completed"):
        accumulator.update(logits, input_ids, attention_mask, batch_index=3)


def test_nll_accumulator_sums_fp32_per_token_losses_on_cpu_float64(monkeypatch) -> None:
    accumulator = NLLAccumulator()
    logits = torch.zeros(1, 5, 2, dtype=torch.bfloat16)
    input_ids = torch.tensor([[0, 1, 0, 1, 0]])
    attention_mask = torch.ones(1, 5, dtype=torch.long)

    def per_token_cross_entropy(values, labels, *, reduction):
        assert values.dtype == torch.float32
        assert reduction == "none"
        assert labels.numel() == 4
        return torch.tensor([1e8, 1.0, 1.0, -1e8], dtype=torch.float32, device=values.device)

    monkeypatch.setattr("ppmoe.evaluation.perplexity.F.cross_entropy", per_token_cross_entropy)
    accumulator.update(logits, input_ids, attention_mask, batch_index=0)

    assert accumulator.finalize().nll_sum == 2.0


def test_failed_batch_invalidates_aggregate() -> None:
    accumulator = NLLAccumulator()

    accumulator.record_failure(7, "synthetic failure")
    result = accumulator.finalize()

    assert result.complete is False
    assert result.failed_batches == 1
    assert result.ppl is None


class ConstantModel(nn.Module):
    def __init__(self, vocab_size: int = 4, *, fail: bool = False) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.fail = fail
        self.calls = 0

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model failure")
        return type(
            "Output",
            (),
            {"logits": torch.zeros(*input_ids.shape, self.vocab_size)},
        )()


def _batches() -> tuple[EvaluationBatch, ...]:
    return (
        EvaluationBatch(0, torch.tensor([[0, 1, 2]]), torch.ones(1, 3, dtype=torch.long)),
        EvaluationBatch(1, torch.tensor([[2, 1, 0]]), torch.ones(1, 3, dtype=torch.long)),
    )


def test_perplexity_resume_does_not_score_completed_batches_twice(tmp_path: Path) -> None:
    first_model = ConstantModel()
    first = evaluate_perplexity(first_model, _batches(), tmp_path)
    resumed_model = ConstantModel()

    resumed = evaluate_perplexity(resumed_model, _batches(), tmp_path)

    assert first.scored_tokens == resumed.scored_tokens == 4
    assert first_model.calls == 2
    assert resumed_model.calls == 0


def test_perplexity_records_model_failure_as_incomplete(tmp_path: Path) -> None:
    result = evaluate_perplexity(ConstantModel(fail=True), _batches(), tmp_path)

    assert result.complete is False
    assert result.failed_batches == 1
    assert result.scored_tokens == 0


def test_m0_rejects_perplexity_ratio_outside_locked_interval(tmp_path: Path) -> None:
    teacher = ConstantModel(vocab_size=4)
    student = ConstantModel(vocab_size=5)

    with pytest.raises(M0GateError, match="perplexity ratio"):
        evaluate_m0(teacher, student, _batches(), tmp_path)


def test_m0_persists_fixed_batch_discrepancy_evidence(tmp_path: Path) -> None:
    result = evaluate_m0(ConstantModel(), ConstantModel(), _batches(), tmp_path)

    assert result.fixed_batch_index == 0
    assert len(result.input_ids_sha256) == 64
    assert len(result.attention_mask_sha256) == 64
    assert result.logits_finite is True
    assert result.max_abs_logit_difference == 0.0
    assert result.logits_cosine == 1.0


def test_m0_rejects_nonfinite_discrepancy_logits_before_cosine(tmp_path: Path) -> None:
    class NonfiniteAfterEvaluationModel(ConstantModel):
        def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
            self.calls += 1
            logits = torch.zeros(*input_ids.shape, self.vocab_size)
            if self.calls > 2:
                logits.fill_(float("nan"))
            return type("Output", (), {"logits": logits})()

    with pytest.raises(M0GateError, match="logits are not finite"):
        evaluate_m0(
            NonfiniteAfterEvaluationModel(),
            NonfiniteAfterEvaluationModel(),
            _batches(),
            tmp_path,
        )
