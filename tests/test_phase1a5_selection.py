from pathlib import Path

import pytest

from ppmoe.experiments.phase1a5_config import Phase1A5Config
from ppmoe.experiments.phase1a5_selection import select_calibration_conditions
from ppmoe.oracle.reconstruction import ReconstructionCondition
from ppmoe.oracle.reconstruction_local import (
    ConditionMetrics,
    ReconstructionLocalResult,
    condition_id,
)


CONFIG = Phase1A5Config.load(Path(__file__).parents[1] / "configs" / "phase1-v0_5.toml")


def metric(
    sse: float,
    *,
    coefficients: int = 100,
    negative: int = 0,
    above_two: int = 0,
    deviation: float = 0.2,
    failures: int = 0,
    lower: int = 0,
    upper: int = 0,
) -> ConditionMetrics:
    return ConditionMetrics(
        10,
        sse,
        100.0,
        coefficients,
        negative,
        above_two,
        deviation * coefficients,
        failures,
        lower,
        upper,
        True,
    )


def result_for(
    metrics: dict[tuple[int, str], ConditionMetrics],
) -> ReconstructionLocalResult:
    return ReconstructionLocalResult(
        metrics=metrics,
        layer_metrics={},
        selection_counts={},
        co_selection={},
        solver_diagnostics={},
        reference_distances={},
        coefficient_quantiles={},
        completed_batches=1,
        failed_batches=0,
        complete=True,
        run_signature="local",
        source_hashes={"source": "a", "local": "b"},
    )


def test_selection_records_no_eligible_condition(tmp_path: Path) -> None:
    unit = ReconstructionCondition("unit")
    ridge = ReconstructionCondition("ridge", rho=0.1)
    result = result_for(
        {
            (4, condition_id(unit)): metric(100),
            (4, condition_id(ridge)): metric(80),
            (6, condition_id(unit)): metric(100),
        }
    )

    manifest = select_calibration_conditions(
        result, CONFIG, tmp_path / "selection-manifest.json"
    )
    assert manifest.practical_conditions == {}
    assert "does not meet 30%" in manifest.rejected[(4, condition_id(ridge))]


def test_selection_picks_ridge_then_ties_by_deviation_and_stronger_rho(
    tmp_path: Path,
) -> None:
    unit = ReconstructionCondition("unit")
    weaker = ReconstructionCondition("ridge", rho=0.1)
    stronger = ReconstructionCondition("ridge", rho=1.0)
    result = result_for(
        {
            (4, condition_id(unit)): metric(100),
            (4, condition_id(weaker)): metric(60, deviation=0.1),
            (4, condition_id(stronger)): metric(60, deviation=0.1),
            (6, condition_id(unit)): metric(100),
        }
    )
    manifest = select_calibration_conditions(
        result, CONFIG, tmp_path / "selection-manifest.json"
    )
    assert manifest.practical_conditions[4] == condition_id(stronger)


def test_bounded_is_eligible_by_construction_and_free_bound_advances_alone(
    tmp_path: Path,
) -> None:
    unit = ReconstructionCondition("unit")
    bounded = ReconstructionCondition("bounded")
    free = ReconstructionCondition("free_bound")
    result = result_for(
        {
            (4, condition_id(unit)): metric(100),
            (4, condition_id(bounded)): metric(
                60, negative=50, above_two=0, lower=30, upper=20
            ),
            (6, condition_id(unit)): metric(100),
            (6, condition_id(free)): metric(60, negative=99, above_two=99),
        }
    )
    manifest = select_calibration_conditions(
        result, CONFIG, tmp_path / "selection-manifest.json"
    )
    assert manifest.practical_conditions[4] == condition_id(bounded)
    assert manifest.free_bound_conditions[6] == condition_id(free)
    assert "bounded saturation" in manifest.diagnostics[(4, condition_id(bounded))]


def test_selection_is_immutable_and_refuses_evaluation_artifacts(
    tmp_path: Path,
) -> None:
    unit = ReconstructionCondition("unit")
    ridge = ReconstructionCondition("ridge", rho=0.1)
    result = result_for(
        {
            (4, condition_id(unit)): metric(100),
            (4, condition_id(ridge)): metric(60),
            (6, condition_id(unit)): metric(100),
        }
    )
    path = tmp_path / "selection-manifest.json"
    select_calibration_conditions(result, CONFIG, path)
    evaluation = tmp_path / "evaluation-results.json"
    evaluation.write_text('{"ppl": 1.0}', encoding="utf-8")
    assert select_calibration_conditions(result, CONFIG, path).practical_conditions[
        4
    ] == condition_id(ridge)
    changed = result_for(
        {
            (4, condition_id(unit)): metric(100),
            (4, condition_id(ridge)): metric(70),
            (6, condition_id(unit)): metric(100),
        }
    )
    with pytest.raises(ValueError, match="immutable"):
        select_calibration_conditions(changed, CONFIG, path)
