from dataclasses import replace
import json
from math import sqrt
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
import torch

import ppmoe.oracle.reconstruction_local as reconstruction_local
from ppmoe.evaluation.perplexity import EvaluationBatch
from ppmoe.oracle.reconstruction import ReconstructionCondition, ReconstructionResult
from ppmoe.oracle.reconstruction_local import (
    ReconstructionLocalAccumulator,
    condition_id,
)


RIDGE = ReconstructionCondition("ridge", rho=0.1)


def _result(
    subsets: torch.Tensor,
    coefficients: torch.Tensor,
    squared_error: torch.Tensor,
    *,
    diagnostics: dict[str, object] | None = None,
) -> ReconstructionResult:
    return ReconstructionResult(
        subsets=subsets,
        coefficients=coefficients,
        squared_error=squared_error,
        effective_rank=None,
        solver_source=("test",) * subsets.shape[0],
        diagnostics=diagnostics or {},
    )


def test_accumulator_commits_only_completed_batches_and_resumes_exactly(
    tmp_path: Path,
) -> None:
    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (ReconstructionCondition("unit"), RIDGE)},
        checkpoint_dir=tmp_path,
        run_signature="locked",
        source_hashes={"source": "source-hash"},
    )
    target = torch.tensor([[1.0, 0.0]])
    experts = torch.zeros(1, 8, 2)
    experts[0, 0] = target[0]

    accumulator.update_layer(0, target, experts)
    assert accumulator.finalize().metrics[(4, condition_id(RIDGE))].token_count == 1
    assert (
        not list((tmp_path / "coefficients").glob("*.json"))
        if (tmp_path / "coefficients").exists()
        else True
    )
    accumulator.mark_batch_completed(7)

    resumed = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (ReconstructionCondition("unit"), RIDGE)},
        checkpoint_dir=tmp_path,
        run_signature="locked",
    )
    assert resumed.completed_batches == {7}
    assert resumed.finalize().metrics[(4, condition_id(RIDGE))].token_count == 1
    assert resumed.finalize().source_hashes == {"source": "source-hash"}
    with pytest.raises(ValueError, match="signature"):
        ReconstructionLocalAccumulator(
            num_layers=1,
            num_experts=8,
            conditions={4: (RIDGE,)},
            checkpoint_dir=tmp_path,
            run_signature="other",
        )


def test_restore_discards_reference_coefficient_and_quantile_deltas(
    tmp_path: Path,
) -> None:
    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (RIDGE,)},
        checkpoint_dir=tmp_path,
        run_signature="rollback",
    )
    target = torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16)
    experts = torch.zeros(1, 8, 2, dtype=torch.bfloat16)
    experts[0, 0, 0] = 1
    snapshot = accumulator.snapshot()
    accumulator.update_layer(0, target, experts)
    accumulator.restore(snapshot)
    accumulator.record_failure(0, "injected layer failure")

    result = accumulator.finalize()
    metric = result.metrics[(4, condition_id(RIDGE))]
    assert metric.token_count == metric.coefficient_count == 0
    assert result.reference_distances == {}
    assert result.coefficient_quantiles[(4, condition_id(RIDGE))] == {}


def test_reference_labels_remain_distinct_for_bf16_independent_experts(
    tmp_path: Path,
) -> None:
    condition = ReconstructionCondition("unit")
    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={8: (condition,)},
        checkpoint_dir=tmp_path,
        run_signature="references",
    )
    # BF16's independent sums differ from the dense reference in this synthetic case.
    d0 = torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16)
    experts = torch.zeros(1, 8, 2, dtype=torch.bfloat16)
    experts[0, :, 0] = torch.tensor([1.0, 0.0078125, 0.0078125, 0, 0, 0, 0, 0])
    accumulator.update_layer(0, d0, experts)
    accumulator.mark_batch_completed(0)

    references = accumulator.finalize().reference_distances
    assert set(references) == {
        (0, "D0", "E8"),
        (0, "D0", f"R8[{condition_id(condition)}]"),
        (0, "E8", f"R8[{condition_id(condition)}]"),
    }
    assert references[(0, "D0", "E8")].squared_error_sum > 0.0


def test_every_sparse_condition_has_a_separately_solved_all_expert_r8_reference(
    tmp_path: Path,
) -> None:
    conditions = (
        ReconstructionCondition("unit"),
        ReconstructionCondition("ridge", rho=0.1),
        ReconstructionCondition("bounded"),
        ReconstructionCondition("free_bound"),
    )
    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: conditions},
        checkpoint_dir=tmp_path,
        run_signature="all-r8-references",
    )
    d0 = torch.tensor([[1.0, -0.5]])
    experts = torch.tensor(
        [
            [
                [0.2, -0.1],
                [0.1, -0.1],
                [0.2, 0.0],
                [0.0, -0.1],
                [0.1, 0.0],
                [0.1, -0.1],
                [0.2, 0.0],
                [0.1, -0.1],
            ]
        ]
    )

    accumulator.update_layer(0, d0, experts)
    references = accumulator.finalize().reference_distances

    for condition in conditions:
        r8 = f"R8[{condition_id(condition)}]"
        assert (0, "D0", r8) in references
        assert (0, "E8", r8) in references


def test_r8_references_deduplicate_k4_k6_and_persist_full_solver_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ridge = ReconstructionCondition("ridge", rho=0.1)
    bounded = ReconstructionCondition("bounded")
    free = ReconstructionCondition("free_bound")
    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (ridge, bounded, free), 6: (ridge, bounded, free)},
        checkpoint_dir=tmp_path,
        run_signature="deduplicated-r8",
    )
    generator = torch.Generator().manual_seed(33)
    experts = torch.randn(2, 8, 8, generator=generator)
    d0 = experts.sum(dim=1) + 0.01
    calls = 0
    original = reconstruction_local.score_full_reference

    def count_references(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(reconstruction_local, "score_full_reference", count_references)
    accumulator.update_layer(0, d0, experts)
    accumulator.mark_batch_completed(0)
    first = accumulator.finalize()
    resumed = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (ridge, bounded, free), 6: (ridge, bounded, free)},
        checkpoint_dir=tmp_path,
        run_signature="deduplicated-r8",
    ).finalize()

    assert calls == 3
    expected_norm = float(d0.double().square().sum().item())
    for condition in (ridge, bounded, free):
        r8 = f"R8[{condition_id(condition)}]"
        assert first.reference_distances[(0, "D0", r8)].token_count == 2
        assert first.reference_distances[
            (0, "D0", r8)
        ].left_squared_norm_sum == pytest.approx(expected_norm)
        diagnostics = first.reference_diagnostics[(0, r8)]
        assert diagnostics["solver_source_counts"]
        if condition.family == "bounded":
            assert diagnostics["bounded_certified_count"] == 2
            assert "bounded_max_stationarity_residual" in diagnostics
        if condition.family == "free_bound":
            assert diagnostics["pseudoinverse_effective_rank_counts"] == {"8": 2}
        assert diagnostics == resumed.reference_diagnostics[(0, r8)]


def test_tensor_solver_diagnostics_persist_rank_distribution_across_resume(
    tmp_path: Path,
) -> None:
    condition = ReconstructionCondition("free_bound")
    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (condition,)},
        checkpoint_dir=tmp_path,
        run_signature="tensor-diagnostics",
    )
    d0 = torch.tensor([[1.0], [0.0]])
    experts = torch.zeros(2, 8, 1)
    experts[0, :, 0] = torch.arange(1, 9, dtype=torch.float32)

    accumulator.update_layer(0, d0, experts)
    accumulator.mark_batch_completed(0)
    first = accumulator.finalize().solver_diagnostics[(4, condition_id(condition))]
    resumed = (
        ReconstructionLocalAccumulator(
            num_layers=1,
            num_experts=8,
            conditions={4: (condition,)},
            checkpoint_dir=tmp_path,
            run_signature="tensor-diagnostics",
        )
        .finalize()
        .solver_diagnostics[(4, condition_id(condition))]
    )

    assert first["pseudoinverse_effective_rank_counts"] == {"0": 1, "1": 1}
    assert first["sequence_diagnostics"]["candidate_sources"] == [
        "pseudoinverse",
        "unit",
        "ridge:10",
        "ridge:1",
        "ridge:0.1",
        "ridge:0.01",
        "ridge:0.001",
    ]
    assert first == resumed


def test_persist_failure_rolls_back_the_entire_batch_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    condition = ReconstructionCondition("unit")
    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (condition,)},
        checkpoint_dir=tmp_path,
        run_signature="persist-failure",
    )
    target = torch.tensor([[1.0]])
    experts = torch.zeros(1, 8, 1)
    calls = 0
    original = accumulator._persist

    def fail_once() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected checkpoint failure")
        original()

    accumulator.update_layer(0, target, experts)
    monkeypatch.setattr(accumulator, "_persist", fail_once)
    with pytest.raises(OSError, match="checkpoint failure"):
        accumulator.mark_batch_completed(0)

    assert accumulator.completed_batches == set()
    assert accumulator.failures == {}
    assert accumulator.finalize().metrics[(4, condition_id(condition))].token_count == 0
    assert not list((tmp_path / "coefficients").glob("*.json"))
    assert not list((tmp_path / "coefficients").glob("*.bin"))
    accumulator.update_layer(0, target, experts)
    accumulator.mark_batch_completed(0)
    assert accumulator.completed_batches == {0}


def test_coefficient_chunks_are_sorted_and_deterministic(tmp_path: Path) -> None:
    condition = ReconstructionCondition("unit")
    kwargs = dict(
        num_layers=1,
        num_experts=8,
        conditions={4: (condition,)},
        run_signature="chunks",
    )
    target = torch.tensor([[0.0]])
    experts = torch.zeros(1, 8, 1)
    first = ReconstructionLocalAccumulator(checkpoint_dir=tmp_path / "one", **kwargs)
    second = ReconstructionLocalAccumulator(checkpoint_dir=tmp_path / "two", **kwargs)
    for accumulator in (first, second):
        accumulator.update_layer(0, target, experts)
        accumulator.mark_batch_completed(3)
    first_chunk = next((tmp_path / "one" / "coefficients").glob("*.json"))
    second_chunk = next((tmp_path / "two" / "coefficients").glob("*.json"))
    assert first_chunk.read_bytes() == second_chunk.read_bytes()
    assert (
        first.finalize().coefficient_quantiles[(4, condition_id(condition))]["p99"]
        == 1.0
    )


def test_cross_condition_subset_jaccard_is_known_and_resumable(tmp_path: Path) -> None:
    unit = ReconstructionCondition("unit")
    ridge = ReconstructionCondition("ridge", rho=0.1)
    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (unit, ridge)},
        checkpoint_dir=tmp_path,
        run_signature="jaccard",
    )
    target = torch.tensor([[1.0], [0.0]])
    experts = torch.zeros(2, 8, 1)
    experts[:, 0, 0] = 1.0
    accumulator.update_layer(0, target, experts)
    accumulator.mark_batch_completed(0)

    key = (4, "ridge:0.1", "unit")
    first = accumulator.finalize().subset_stability[key]
    resumed = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (unit, ridge)},
        checkpoint_dir=tmp_path,
        run_signature="jaccard",
    ).finalize().subset_stability[key]

    assert first.token_count == 2
    assert first.mean_jaccard == pytest.approx(1.0)
    assert resumed == first


def test_jaccard_transaction_rolls_back_on_persist_failure_then_retries_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unit, ridge = ReconstructionCondition("unit"), ReconstructionCondition("ridge", rho=0.1)
    accumulator = ReconstructionLocalAccumulator(num_layers=1, num_experts=8, conditions={4: (unit, ridge)}, checkpoint_dir=tmp_path, run_signature="jaccard-rollback")
    target, experts = torch.tensor([[1.0], [0.0]]), torch.zeros(2, 8, 1)
    experts[:, 0, 0] = 1.0
    accumulator.update_layer(0, target, experts)
    original, calls = accumulator._persist, 0

    def fail_once() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected persist failure")
        original()

    monkeypatch.setattr(accumulator, "_persist", fail_once)
    with pytest.raises(OSError, match="injected persist failure"):
        accumulator.mark_batch_completed(0)
    assert accumulator.finalize().subset_stability[(4, "ridge:0.1", "unit")].token_count == 0
    accumulator.update_layer(0, target, experts)
    accumulator.mark_batch_completed(0)
    record = accumulator.finalize().subset_stability[(4, "ridge:0.1", "unit")]
    assert record.token_count == 2 and record.mean_jaccard == pytest.approx(1.0)


def test_pooled_improvement_and_gain_recovery_are_analysis_only(tmp_path: Path) -> None:
    unit = ReconstructionCondition("unit")
    ridge = ReconstructionCondition("ridge", rho=0.1)
    free = ReconstructionCondition("free_bound")
    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (unit, ridge, free)},
        checkpoint_dir=tmp_path,
        run_signature="gain",
    )
    target = torch.tensor([[1.0]])
    experts = torch.full((1, 8, 1), 0.2)
    accumulator.update_layer(0, target, experts)
    accumulator.mark_batch_completed(0)
    result = accumulator.finalize()
    assert result.relative_improvement(4, condition_id(ridge)) is not None
    assert result.gain_recovery(4, condition_id(ridge)) is not None
    zero_denominator = dict(result.metrics)
    zero_denominator[(4, condition_id(free))] = replace(
        zero_denominator[(4, condition_id(free))],
        total_squared_error=zero_denominator[
            (4, condition_id(unit))
        ].total_squared_error,
    )
    assert (
        replace(result, metrics=zero_denominator).gain_recovery(4, condition_id(ridge))
        is None
    )


def test_free_bound_svd_checks_first_sixteen_tokens_per_layer_once_and_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    free = ReconstructionCondition("free_bound")
    calls: list[tuple[int, int]] = []

    def verify(
        target: torch.Tensor,
        _experts: torch.Tensor,
        result: ReconstructionResult,
        _settings: object,
    ) -> dict[str, object]:
        k = result.subsets.shape[1]
        calls.append((k, target.shape[0]))
        base = {4: 2e-10, 6: 3e-10, 8: 4e-10}[k]
        error = base * calls.count((k, target.shape[0]))
        return {
            "rank_equal": True,
            "relative_objective_error": torch.full(
                (target.shape[0],), error, dtype=torch.float64
            ),
            "max_relative_objective_error": error,
        }

    monkeypatch.setattr(reconstruction_local, "verify_pseudoinverse_svd", verify)
    kwargs = dict(
        num_layers=2,
        num_experts=8,
        conditions={4: (free,), 6: (free,)},
        checkpoint_dir=tmp_path,
        run_signature="svd-resume",
    )
    generator = torch.Generator().manual_seed(103)
    first = ReconstructionLocalAccumulator(**kwargs)
    for layer, tokens in ((0, 20), (1, 8)):
        experts = torch.randn(tokens, 8, 3, generator=generator)
        first.update_layer(layer, experts.sum(dim=1), experts)
    first.mark_batch_completed(0)

    resumed = ReconstructionLocalAccumulator(**kwargs)
    experts = torch.randn(10, 8, 3, generator=generator)
    resumed.update_layer(1, experts.sum(dim=1), experts)
    resumed.mark_batch_completed(1)
    evidence = resumed.finalize().svd_verification

    expected = {
        "sparse:K4:free_bound": 4e-10,
        "sparse:K6:free_bound": 6e-10,
        "R8[free_bound]": 8e-10,
    }
    for identity, maximum in expected.items():
        record = evidence[identity]
        assert record.identity == identity
        assert record.sample_count == 32
        assert record.sample_count_by_layer == {0: 16, 1: 16}
        assert record.rank_equal is True
        assert record.max_relative_objective_error == pytest.approx(maximum)
    assert calls.count((4, 16)) == 1
    assert calls.count((6, 16)) == 1
    assert calls.count((8, 16)) == 1
    assert calls.count((4, 8)) == calls.count((6, 8)) == calls.count((8, 8)) == 2


def test_svd_mismatch_is_a_failed_batch_and_rolls_back_all_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class TeacherMLP(torch.nn.Identity):
        pass

    class Teacher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = TeacherMLP()
            self.model = SimpleNamespace(language_model=self.forward_language)

        def forward_language(self, *, input_ids: torch.Tensor, **_kwargs: object) -> None:
            values = input_ids.to(torch.float32).unsqueeze(-1)
            self.mlp(values)

    class StudentMLP(torch.nn.Module):
        num_experts = 8

        def expert_outputs(self, values: torch.Tensor) -> torch.Tensor:
            return values.unsqueeze(-2).expand(*values.shape[:-1], 8, values.shape[-1])

    class Student(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = StudentMLP()

    monkeypatch.setattr(
        reconstruction_local, "iter_text_mlps", lambda model: ((0, model.mlp),)
    )
    monkeypatch.setattr(reconstruction_local, "PartitionedQwen35MLP", StudentMLP)
    monkeypatch.setattr(
        reconstruction_local,
        "verify_pseudoinverse_svd",
        lambda *_args, **_kwargs: {
            "rank_equal": False,
            "relative_objective_error": torch.zeros(4, dtype=torch.float64),
            "max_relative_objective_error": 0.0,
        },
    )
    free = ReconstructionCondition("free_bound")
    result = reconstruction_local.run_reconstruction_local(
        Teacher(),
        Student(),
        (
            EvaluationBatch(
                index=0,
                input_ids=torch.ones((1, 4), dtype=torch.long),
                attention_mask=torch.ones((1, 4), dtype=torch.long),
            ),
        ),
        {4: (free,), 6: (free,)},
        tmp_path,
        "svd-mismatch",
    )

    assert result.complete is False
    assert result.completed_batches == 0
    assert result.failed_batches == 1
    assert all(metric.token_count == 0 for metric in result.metrics.values())
    assert result.reference_distances == {}
    assert all(record.sample_count == 0 for record in result.svd_verification.values())


def test_condition_metrics_are_exact_known_values_pooled_per_layer_and_resumable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    subsets = torch.tensor([[0, 1, 2, 3]] * 3)
    coefficients = torch.tensor(
        [[1.0, -1.0, 3.0, 5.0], [1.0, 0.0, 2.0, 4.0], [0.0, 1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    squared_error = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64)
    sparse = _result(
        subsets,
        coefficients,
        squared_error,
        diagnostics={
            "cholesky_fallback_count": 2,
            "degenerate_ridge_count": 1,
            "max_relative_system_residual": 1e-12,
        },
    )
    full = _result(
        torch.arange(8).expand(3, -1),
        torch.ones((3, 8), dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
    )
    monkeypatch.setattr(reconstruction_local, "score_reconstruction", lambda *_args: sparse)
    monkeypatch.setattr(reconstruction_local, "score_full_reference", lambda *_args: full)
    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (RIDGE,)},
        checkpoint_dir=tmp_path,
        run_signature="known-metrics",
    )
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    experts = torch.zeros(3, 8, 2)
    experts[0, 0] = torch.tensor([1.0, 0.0])
    experts[1, 0] = torch.tensor([1.0, 0.0])
    accumulator.update_layer(0, target, experts)
    accumulator.mark_batch_completed(0)

    result = accumulator.finalize()
    metric = result.metrics[(4, "ridge:0.1")]
    assert metric == result.layer_metrics[(0, 4, "ridge:0.1")]
    assert metric.coefficient_count == metric.l2_deviation_denominator == 12
    assert metric.coefficient_mean == pytest.approx(1.5)
    assert metric.coefficient_std == pytest.approx(sqrt(2.75))
    assert metric.coefficient_p01 == -1.0
    assert metric.coefficient_p05 == -1.0
    assert metric.coefficient_median == 1.0
    assert metric.coefficient_p95 == 5.0
    assert metric.coefficient_p99 == 5.0
    assert metric.coefficient_p99_abs == 5.0
    assert metric.coefficient_max_abs == 5.0
    assert (metric.negative_count, metric.negative_rate) == (1, pytest.approx(1 / 12))
    assert (metric.above_two_count, metric.above_two_rate) == (3, pytest.approx(3 / 12))
    assert (metric.above_four_count, metric.above_four_rate) == (1, pytest.approx(1 / 12))
    assert metric.mean_absolute_deviation_from_one == pytest.approx(14 / 12)
    assert metric.l2_deviation_from_one == pytest.approx(sqrt(36 / 12))
    assert metric.relative_error_defined_count == 2
    assert metric.relative_error_undefined_count == 1
    assert metric.relative_error_mean == pytest.approx(sqrt(2) / 2)
    assert metric.relative_error_median == 0.0
    assert metric.relative_error_p90 == pytest.approx(sqrt(2))
    assert metric.relative_error_p95 == pytest.approx(sqrt(2))
    assert metric.relative_error_p99 == pytest.approx(sqrt(2))
    assert metric.cosine_defined_count == 2
    assert metric.cosine_undefined_count == 1
    assert metric.cosine_mean == pytest.approx(0.5)
    assert metric.cosine_median == 0.0
    assert metric.cosine_p90 == 1.0
    assert metric.cosine_p95 == 1.0
    assert metric.cosine_p99 == 1.0
    assert metric.gram_condition_defined_count == 2
    assert metric.gram_condition_undefined_count == 1
    assert metric.gram_condition_median == 1.0
    assert metric.gram_condition_p90 == 1.0
    assert metric.gram_condition_p99 == 1.0
    assert metric.gram_condition_max == 1.0
    assert metric.effective_rank_counts == {"0": 1, "1": 2}
    diagnostics = result.layer_solver_diagnostics[(0, 4, "ridge:0.1")]
    assert diagnostics["cholesky_fallback_count"] == 2
    assert diagnostics["degenerate_ridge_count"] == 1
    resumed = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (RIDGE,)},
        checkpoint_dir=tmp_path,
        run_signature="known-metrics",
    ).finalize()
    assert resumed.metrics == result.metrics
    assert resumed.layer_metrics == result.layer_metrics
    assert resumed.layer_solver_diagnostics == result.layer_solver_diagnostics


def test_bounded_saturation_rates_use_exact_strict_thresholds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bounded = ReconstructionCondition("bounded")
    coefficients = torch.tensor(
        [[0.0, 0.5, 1.9999995, 2.0], [1e-6, 0.1, 1.0, 1.5]],
        dtype=torch.float64,
    )
    sparse = _result(
        torch.tensor([[0, 1, 2, 3]] * 2),
        coefficients,
        torch.zeros(2, dtype=torch.float64),
        diagnostics={
            "bounded_lower_saturation_count": 1,
            "bounded_upper_saturation_count": 2,
            "bounded_failure_count": 0,
            "bounded_uncertified_count": 0,
            "bounded_certified_count": 2,
            "bounded_solver_candidate_count": 2,
            "bounded_coefficient_bound_violation_count": 0,
        },
    )
    full = _result(
        torch.arange(8).expand(2, -1),
        torch.ones((2, 8), dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
    )
    monkeypatch.setattr(reconstruction_local, "score_reconstruction", lambda *_args: sparse)
    monkeypatch.setattr(reconstruction_local, "score_full_reference", lambda *_args: full)
    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (bounded,)},
        checkpoint_dir=tmp_path,
        run_signature="bounded-rates",
    )
    target = torch.zeros(2, 1)
    experts = torch.zeros(2, 8, 1)
    accumulator.update_layer(0, target, experts)
    accumulator.mark_batch_completed(0)

    metric = accumulator.finalize().metrics[(4, "bounded")]
    assert metric.bounded_lower_saturation_count == 1
    assert metric.bounded_lower_saturation_rate == pytest.approx(1 / 8)
    assert metric.bounded_upper_saturation_count == 2
    assert metric.bounded_upper_saturation_rate == pytest.approx(2 / 8)


def test_finalization_uses_exact_nearest_rank_k_way_merge_without_materializing_chunks(
    tmp_path: Path,
) -> None:
    """Regression: replacing the merge with concatenation must fail this test."""

    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (RIDGE,)},
        checkpoint_dir=tmp_path,
        run_signature="streamed-quantiles",
    )

    class OnePassChunk:
        def __init__(self, values: tuple[float, ...]) -> None:
            self.values = values
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("a sorted checkpoint chunk was re-materialized")
            yield from self.values

    chunks = [OnePassChunk((1.0, 5.0, 9.0)), OnePassChunk((2.0, 6.0, 10.0)), OnePassChunk((3.0, 7.0, 11.0))]
    quantiles = accumulator._nearest_from_sorted_chunks(  # type: ignore[attr-defined]
        (iter(chunk) for chunk in chunks),
        total_count=9,
        percentiles=(0.01, 0.5, 0.9, 0.99),
    )

    assert quantiles == {0.01: 1.0, 0.5: 6.0, 0.9: 11.0, 0.99: 11.0}
    assert all(chunk.iterations == 1 for chunk in chunks)


def test_production_distribution_chunks_stream_fixed_binary_buffers_and_reject_corruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Persisted chunks are a small manifest plus sequential FP64 data, never arrays."""

    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (RIDGE,)},
        checkpoint_dir=tmp_path,
        run_signature="binary-chunks",
    )
    key = (4, condition_id(RIDGE))
    distribution_names = (
        "coefficients",
        "coefficient_abs",
        "relative_error",
        "cosine",
        "gram_condition",
    )
    for batch, values in enumerate(((1.0, 5.0, 9.0), (2.0, 6.0, 10.0), (3.0, 7.0, 11.0))):
        accumulator._pending_distributions[key] = {  # type: ignore[attr-defined]
            0: {name: list(values) for name in distribution_names}
        }
        accumulator.mark_batch_completed(batch)

    manifest_paths = sorted((tmp_path / "coefficients").glob("*.json"))
    assert len(manifest_paths) == 3
    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    assert manifest["format"] == "phase1a5-distribution-fp64-v1"
    entries = manifest["layers"]["0"]
    assert set(entries) == set(distribution_names)
    assert all(isinstance(entry, dict) and not any(isinstance(value, list) for value in entry.values()) for entry in entries.values())
    data_path = manifest_paths[0].with_suffix(".bin")
    assert data_path.is_file()
    original_data = data_path.read_bytes()

    original_open = Path.open

    class FixedReadHandle:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def read(self, size: int = -1):
            assert 0 < size <= 8 * 1024, "reader attempted to materialize a binary chunk"
            return self._handle.read(size)

        def __getattr__(self, name: str):
            return getattr(self._handle, name)

    def guarded_open(path: Path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if Path(path).suffix == ".bin" and args and args[0] == "rb":
            return FixedReadHandle(handle)
        return handle

    def fail_array_read(*_args, **_kwargs):
        raise AssertionError("persisted distributions must not be loaded as JSON arrays")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", fail_array_read)
    monkeypatch.setattr(Path, "read_bytes", fail_array_read)
    quantiles = accumulator._nearest_from_sorted_chunks(  # type: ignore[attr-defined]
        accumulator._iter_distribution_chunks(key, "coefficients"),  # type: ignore[attr-defined]
        total_count=9,
        percentiles=(0.01, 0.5, 0.9, 0.99),
    )
    assert quantiles == {0.01: 1.0, 0.5: 6.0, 0.9: 11.0, 0.99: 11.0}

    monkeypatch.undo()
    data_path.write_bytes(original_data + b"\x00")
    with pytest.raises(RuntimeError, match="trailing"):
        list(accumulator._iter_distribution_chunks(key, "coefficients"))  # type: ignore[attr-defined]
    data_path.write_bytes(original_data)

    truncated = manifest_paths[1].with_suffix(".bin")
    original_truncated_data = truncated.read_bytes()
    truncated.write_bytes(original_truncated_data[:-1])
    with pytest.raises(RuntimeError, match="truncated"):
        list(accumulator._iter_distribution_chunks(key, "coefficients"))  # type: ignore[attr-defined]
    truncated.write_bytes(original_truncated_data)
    malformed = json.loads(manifest_paths[2].read_text(encoding="utf-8"))
    malformed["layers"]["0"]["coefficients"]["count"] = "three"
    manifest_paths[2].write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="malformed"):
        list(accumulator._iter_distribution_chunks(key, "coefficients"))  # type: ignore[attr-defined]


def test_distribution_reader_rejects_in_range_order_preserving_binary_corruption(
    tmp_path: Path,
) -> None:
    """Manifest metadata must authenticate every streamed distribution value."""

    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (RIDGE,)},
        checkpoint_dir=tmp_path,
        run_signature="binary-chunk-authentication",
    )
    key = (4, condition_id(RIDGE))
    accumulator._pending_distributions[key] = {  # type: ignore[attr-defined]
        0: {
            "coefficients": [1.0, 5.0, 9.0],
            "coefficient_abs": [1.0, 5.0, 9.0],
            "relative_error": [1.0, 5.0, 9.0],
            "cosine": [1.0, 5.0, 9.0],
            "gram_condition": [1.0, 5.0, 9.0],
        }
    }
    accumulator.mark_batch_completed(0)
    data_path = next((tmp_path / "coefficients").glob("*.bin"))
    data = bytearray(data_path.read_bytes())
    data[8:16] = struct.pack("<d", 6.0)
    data_path.write_bytes(data)

    with pytest.raises(RuntimeError, match="malformed"):
        list(
            value
            for chunk in accumulator._iter_distribution_chunks(key, "coefficients")  # type: ignore[attr-defined]
            for value in chunk
        )


def test_checkpoint_loader_explicitly_rejects_legacy_distribution_format(
    tmp_path: Path,
) -> None:
    """A state written before the binary chunk contract cannot be resumed silently."""

    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (RIDGE,)},
        checkpoint_dir=tmp_path,
        run_signature="legacy-chunk-format",
    )
    accumulator._persist()  # type: ignore[attr-defined]
    state_path = tmp_path / "reconstruction_local_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("distribution_chunk_format", None)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="distribution checkpoint format"):
        ReconstructionLocalAccumulator(
            num_layers=1,
            num_experts=8,
            conditions={4: (RIDGE,)},
            checkpoint_dir=tmp_path,
            run_signature="legacy-chunk-format",
        )


def test_zero_d0_relative_errors_are_undefined_and_resume_without_double_counting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: zero-norm D0 tokens must be excluded, never coerced to zero error."""

    sparse = _result(
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
    )
    full = _result(
        torch.arange(8).expand(2, -1),
        torch.ones((2, 8), dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
    )
    monkeypatch.setattr(reconstruction_local, "score_reconstruction", lambda *_args: sparse)
    monkeypatch.setattr(reconstruction_local, "score_full_reference", lambda *_args: full)
    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (RIDGE,)},
        checkpoint_dir=tmp_path,
        run_signature="zero-relative-error",
    )
    target = torch.tensor([[1.0], [0.0]])
    experts = torch.zeros(2, 8, 1)
    experts[:, 0, 0] = 1.0
    accumulator.update_layer(0, target, experts)
    accumulator.mark_batch_completed(0)

    metric = accumulator.finalize().metrics[(4, "ridge:0.1")]
    assert metric.relative_error_defined_count == 1
    assert metric.relative_error_undefined_count == 1
    assert metric.relative_error_mean == 0.0
    assert metric.relative_error_median == 0.0
    resumed = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (RIDGE,)},
        checkpoint_dir=tmp_path,
        run_signature="zero-relative-error",
    ).finalize().metrics[(4, "ridge:0.1")]
    assert resumed.relative_error_defined_count == 1
    assert resumed.relative_error_undefined_count == 1


def test_rollback_after_zero_d0_discards_defined_and_undefined_relative_error_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed microbatch cannot leak a zero-target accounting delta into resume."""

    sparse = _result(
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
    )
    full = _result(
        torch.arange(8).expand(2, -1),
        torch.ones((2, 8), dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
    )
    monkeypatch.setattr(reconstruction_local, "score_reconstruction", lambda *_args: sparse)
    monkeypatch.setattr(reconstruction_local, "score_full_reference", lambda *_args: full)
    accumulator = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (RIDGE,)},
        checkpoint_dir=tmp_path,
        run_signature="rollback-zero-relative-error",
    )
    snapshot = accumulator.snapshot()
    target = torch.tensor([[1.0], [0.0]])
    experts = torch.zeros(2, 8, 1)
    experts[:, 0, 0] = 1.0
    accumulator.update_layer(0, target, experts)
    accumulator.restore(snapshot)
    accumulator.record_failure(0, "injected after zero-D0")

    result = accumulator.finalize().metrics[(4, "ridge:0.1")]
    assert result.token_count == 0
    assert result.relative_error_defined_count == 0
    assert result.relative_error_undefined_count == 0
    resumed = ReconstructionLocalAccumulator(
        num_layers=1,
        num_experts=8,
        conditions={4: (RIDGE,)},
        checkpoint_dir=tmp_path,
        run_signature="rollback-zero-relative-error",
    ).finalize().metrics[(4, "ridge:0.1")]
    assert resumed.relative_error_defined_count == 0
    assert resumed.relative_error_undefined_count == 0
