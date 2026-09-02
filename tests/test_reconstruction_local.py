from dataclasses import replace
from pathlib import Path

import pytest
import torch

import ppmoe.oracle.reconstruction_local as reconstruction_local
from ppmoe.oracle.reconstruction import ReconstructionCondition
from ppmoe.oracle.reconstruction_local import (
    ReconstructionLocalAccumulator,
    condition_id,
)


RIDGE = ReconstructionCondition("ridge", rho=0.1)


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
