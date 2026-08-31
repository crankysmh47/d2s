from copy import deepcopy

import pytest
import torch

import ppmoe.oracle.local as local_oracle
from ppmoe.evaluation.perplexity import EvaluationBatch
from ppmoe.moefication.qwen35 import convert_qwen35_text_ffns
from ppmoe.oracle.local import LocalOracleAccumulator, run_local_oracle
from tests.helpers import tiny_qwen35


def test_local_accumulator_aggregates_selection_interaction_and_cancellation() -> None:
    accumulator = LocalOracleAccumulator(
        num_layers=1,
        num_experts=2,
        k_values=(2, 1),
    )
    experts = torch.tensor(
        [
            [[1.0, 0.0], [-1.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    target = experts.sum(dim=1)

    accumulator.update_layer(0, target, experts)
    result = accumulator.finalize()

    assert result.layer_k[(0, 2)].token_count == 2
    assert result.layer_k[(0, 2)].normalized_mse == pytest.approx(0.0, abs=1e-7)
    assert sum(result.selection_counts[(0, 1)]) == 2
    assert result.co_selection[(0, 2)] == ((2, 2), (2, 2))
    assert result.interaction_sum[0][0][1] < 0
    assert result.cancellation_count[0][0][1] == 1


def test_local_accumulator_resume_and_signature_guard(tmp_path) -> None:
    experts = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    target = experts.sum(dim=1)
    first = LocalOracleAccumulator(
        num_layers=1,
        num_experts=2,
        k_values=(2, 1),
        checkpoint_dir=tmp_path,
        run_signature="locked-inputs",
    )
    first.update_layer(0, target, experts)
    first.mark_batch_completed(0)

    resumed = LocalOracleAccumulator(
        num_layers=1,
        num_experts=2,
        k_values=(2, 1),
        checkpoint_dir=tmp_path,
        run_signature="locked-inputs",
    )

    assert resumed.completed_batches == {0}
    assert resumed.finalize().layer_k[(0, 2)].token_count == 1
    assert resumed.finalize().direct_check_count == 2
    with pytest.raises(ValueError, match="signature"):
        LocalOracleAccumulator(
            num_layers=1,
            num_experts=2,
            k_values=(2, 1),
            checkpoint_dir=tmp_path,
            run_signature="different-inputs",
        )


def test_local_accumulator_scores_bf16_in_fp32_and_preserves_float64(monkeypatch) -> None:
    observed_dtypes: list[torch.dtype] = []
    real_gram = local_oracle.exact_gram_oracle

    def record_gram(target, experts, k):
        observed_dtypes.extend((target.dtype, experts.dtype))
        return real_gram(target, experts, k)

    monkeypatch.setattr(local_oracle, "exact_gram_oracle", record_gram)
    accumulator = LocalOracleAccumulator(num_layers=1, num_experts=2, k_values=(2, 1))
    experts = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.bfloat16)
    accumulator.update_layer(0, experts.sum(dim=1), experts)
    assert observed_dtypes == [torch.float32] * 4

    observed_dtypes.clear()
    experts64 = experts.to(torch.float64)
    accumulator.update_layer(0, experts64.sum(dim=1), experts64)
    assert observed_dtypes == [torch.float64] * 4


def test_local_accumulator_direct_checks_each_layer_k_only_once(monkeypatch) -> None:
    calls: list[int] = []
    real_direct = local_oracle.exact_direct_oracle

    def record_direct(target, experts, k):
        calls.append(k)
        return real_direct(target, experts, k)

    monkeypatch.setattr(local_oracle, "exact_direct_oracle", record_direct)
    accumulator = LocalOracleAccumulator(num_layers=1, num_experts=2, k_values=(2, 1))
    experts = torch.tensor([[[1.0], [0.0]]])
    accumulator.update_layer(0, experts.sum(dim=1), experts)
    accumulator.update_layer(0, experts.sum(dim=1), experts)

    assert calls == [2, 1]
    assert accumulator.finalize().direct_check_count == 2


def test_local_accumulator_resume_keeps_direct_checked_pairs(monkeypatch, tmp_path) -> None:
    experts = torch.tensor([[[1.0], [0.0]]])
    first = LocalOracleAccumulator(
        num_layers=1,
        num_experts=2,
        k_values=(2, 1),
        checkpoint_dir=tmp_path,
        run_signature="direct-check-resume",
    )
    first.update_layer(0, experts.sum(dim=1), experts)
    first.mark_batch_completed(0)
    resumed = LocalOracleAccumulator(
        num_layers=1,
        num_experts=2,
        k_values=(2, 1),
        checkpoint_dir=tmp_path,
        run_signature="direct-check-resume",
    )

    monkeypatch.setattr(
        local_oracle,
        "exact_direct_oracle",
        lambda *_: pytest.fail("already checked layer/K was checked again"),
    )
    resumed.update_layer(0, experts.sum(dim=1), experts)
    assert resumed.finalize().direct_check_count == 2


def test_local_accumulator_promotes_per_token_fp32_metrics_before_reduction() -> None:
    token_count = 4096
    values = torch.ones((token_count, 1), dtype=torch.float32)
    values[0] = 4096.0
    experts = torch.stack((values, values), dim=1)
    accumulator = LocalOracleAccumulator(num_layers=1, num_experts=2, k_values=(2,))

    accumulator.update_layer(0, values, experts)
    result = accumulator.finalize()
    expected = float(4096**2 + token_count - 1)

    assert result.layer_k[(0, 2)].target_squared_norm_sum == expected
    assert result.layer_k[(0, 2)].squared_error_sum == expected
    assert result.interaction_sum[0][0][1] == expected


def test_local_runner_rolls_back_earlier_layers_before_persisting_failure(
    monkeypatch, tmp_path
) -> None:
    teacher, student, batch = _converted_models_and_batch()
    failing_mlp = student.model.language_model.layers[1].mlp

    def fail_after_layer_zero(_):
        raise RuntimeError("injected layer-one failure")

    monkeypatch.setattr(failing_mlp, "expert_outputs", fail_after_layer_zero)
    result = run_local_oracle(
        teacher,
        student,
        (batch,),
        k_values=(8, 4),
        checkpoint_dir=tmp_path,
        run_signature="transaction-test",
    )

    assert result.failed_batches == 1
    assert result.layer_k[(0, 8)].token_count == 0
    resumed = LocalOracleAccumulator(
        num_layers=2,
        num_experts=8,
        k_values=(8, 4),
        checkpoint_dir=tmp_path,
        run_signature="transaction-test",
    )
    assert resumed.finalize().layer_k[(0, 8)].token_count == 0


def test_local_runner_uses_text_stack_not_composite_forward(monkeypatch, tmp_path) -> None:
    teacher, student, batch = _converted_models_and_batch()

    def reject_composite_forward(*args, **kwargs):
        raise AssertionError("full composite forward must not run")

    monkeypatch.setattr(teacher, "forward", reject_composite_forward)
    result = run_local_oracle(
        teacher,
        student,
        (batch,),
        k_values=(8, 4),
        checkpoint_dir=tmp_path,
        run_signature="text-stack-only",
    )
    assert result.complete is True


def _converted_models_and_batch():
    torch.manual_seed(53)
    teacher = tiny_qwen35()
    student = deepcopy(teacher)
    convert_qwen35_text_ffns(student, num_experts=8, seed_base=53)
    batch = EvaluationBatch(
        0,
        torch.tensor([[1, 2, 3, 0]]),
        torch.tensor([[1, 1, 1, 0]]),
    )
    return teacher, student, batch


def test_local_runner_uses_teacher_inputs_passes_k8_and_checks_direct_gram(tmp_path) -> None:
    teacher, student, batch = _converted_models_and_batch()

    result = run_local_oracle(
        teacher,
        student,
        (batch,),
        k_values=(8, 4),
        checkpoint_dir=tmp_path,
        run_signature="tiny-qwen",
    )

    assert result.complete is True
    assert result.completed_batches == 1
    assert result.failed_batches == 0
    assert result.direct_check_count == 12
    for layer in range(2):
        assert result.layer_k[(layer, 8)].token_count == 3
        assert result.layer_k[(layer, 8)].normalized_mse <= 1e-8
