from pathlib import Path
from dataclasses import replace

import pytest
import torch

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
