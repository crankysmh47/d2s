import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ppmoe.experiments.phase1a5_config import Phase1A5Config
from ppmoe.oracle.reconstruction_local import (
    ConditionMetrics,
    ReconstructionLocalResult,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "phase1-v0_5.toml"


def _source_manifest() -> dict[str, object]:
    return {
        "source_root": "C:/frozen-phase1",
        "stage_sha256": {"local": "a" * 64, "rollout": "b" * 64, "m0": "c" * 64},
        "dataset_manifest_sha256": {"calibration": "d" * 64},
        "model_manifest_sha256": {"conversion": "e" * 64, "model": "f" * 64},
        "code_revision": {"commit": "1" * 40, "dirty": False},
        "solver_version": "phase1a5-oracle-v1",
    }


def _result(
    run_signature: str, source_hashes: dict[str, str]
) -> ReconstructionLocalResult:
    unit = ConditionMetrics(2_048, 10.0, 20.0, 8_192, 0, 0, 0.0, 0, 0, 0, True)
    free = ConditionMetrics(2_048, 5.0, 20.0, 8_192, 0, 0, 1.0, 0, 0, 0, True)
    return ReconstructionLocalResult(
        metrics={
            (4, "unit"): unit,
            (4, "free_bound"): free,
            (6, "unit"): unit,
            (6, "free_bound"): free,
        },
        layer_metrics={},
        selection_counts={},
        co_selection={},
        solver_diagnostics={},
        reference_distances={},
        reference_diagnostics={},
        coefficient_quantiles={},
        completed_batches=1,
        failed_batches=0,
        complete=True,
        run_signature=run_signature,
        source_hashes=source_hashes,
    )


def _dependencies():
    from ppmoe.experiments.phase1a5 import Phase1A5Dependencies

    calls: list[dict[str, object]] = []

    def local_runner(
        _teacher,
        _student,
        batches,
        conditions,
        checkpoint_dir,
        run_signature,
        *,
        source_hashes,
    ):
        token_count = sum(int(batch.attention_mask.sum()) for batch in batches)
        calls.append(
            {
                "checkpoint_dir": checkpoint_dir,
                "conditions": conditions,
                "tokens": token_count,
                "source_hashes": source_hashes,
            }
        )
        return _result(run_signature, dict(source_hashes))

    def selection_runner(result, _config, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "practical_conditions": {},
                    "free_bound_conditions": {"4": "free_bound"},
                    "frozen": result.complete,
                }
            ),
            encoding="utf-8",
        )
        return {"practical_conditions": {}, "free_bound_conditions": {4: "free_bound"}}

    return Phase1A5Dependencies(
        source_verifier=lambda *_args, **_kwargs: _source_manifest(),
        calibration_loader=lambda *_args: (
            torch.ones((100, 1_000), dtype=torch.long),
            torch.ones((100, 1_000), dtype=torch.long),
        ),
        converted_pair_loader=lambda _config: (object(), object()),
        local_runner=local_runner,
        unit_regression_runner=lambda *_args, **_kwargs: {
            "passed": True,
            "checked_tokens": 2_048,
        },
        cuda_peak_probe=lambda: 123,
        clock=lambda: 1.0,
        selection_runner=selection_runner,
    ), calls


def test_stages_enforce_phase1a5_order_and_selection_is_immutable(
    tmp_path: Path,
) -> None:
    from ppmoe.experiments.phase1a5 import StagePrerequisiteError, run_phase1a5

    config = Phase1A5Config.load(CONFIG_PATH)
    dependencies, _ = _dependencies()
    with pytest.raises(StagePrerequisiteError, match="source"):
        run_phase1a5(config, "smoke", False, tmp_path, 100, dependencies=dependencies)
    run_phase1a5(config, "source", False, tmp_path, 100, dependencies=dependencies)
    with pytest.raises(StagePrerequisiteError, match="feasible smoke"):
        run_phase1a5(config, "local", False, tmp_path, 100, dependencies=dependencies)
    run_phase1a5(config, "smoke", False, tmp_path, 100, dependencies=dependencies)
    run_phase1a5(config, "local", False, tmp_path, 100, dependencies=dependencies)
    selected = run_phase1a5(
        config, "select", False, tmp_path, 100, dependencies=dependencies
    )
    assert selected["selection_manifest_sha256"]
    (tmp_path / "selection-manifest.json").write_text(
        '{"mutated":true}', encoding="utf-8"
    )
    with pytest.raises(StagePrerequisiteError, match="immutable"):
        run_phase1a5(config, "rollout", False, tmp_path, 100, dependencies=dependencies)


def test_smoke_and_local_use_exact_token_counts_and_pass_verified_source_hashes(
    tmp_path: Path,
) -> None:
    from ppmoe.experiments.phase1a5 import run_phase1a5

    config = Phase1A5Config.load(CONFIG_PATH)
    dependencies, calls = _dependencies()
    run_phase1a5(config, "source", False, tmp_path, 100, dependencies=dependencies)
    smoke = run_phase1a5(
        config, "smoke", False, tmp_path, 100, dependencies=dependencies
    )
    local = run_phase1a5(
        config, "local", False, tmp_path, 100, dependencies=dependencies
    )

    assert smoke["valid_calibration_tokens"] == 2_048
    assert set(smoke["families"]) == {"unit", "ridge", "bounded", "free_bound"}
    assert all(value["peak_cuda_bytes"] == 123 for value in smoke["families"].values())
    assert local["valid_calibration_tokens"] == 100_000
    assert {call["tokens"] for call in calls[:4]} == {2_048}
    assert calls[-1]["tokens"] == 100_000
    assert calls[-1]["source_hashes"]["stage:local"] == "a" * 64


def test_local_omits_bounded_only_after_the_bounded_smoke_gate_fails(
    tmp_path: Path,
) -> None:
    from ppmoe.experiments.phase1a5 import run_phase1a5

    config = Phase1A5Config.load(CONFIG_PATH)
    dependencies, calls = _dependencies()
    original_runner = dependencies.local_runner

    def bounded_infeasible(*args, **kwargs):
        assert original_runner is not None
        result = original_runner(*args, **kwargs)
        conditions = args[3]
        if any(
            condition.family == "bounded"
            for values in conditions.values()
            for condition in values
        ):
            return replace(
                result,
                solver_diagnostics={
                    (4, "bounded"): {
                        "bounded_failure_count": 1,
                        "bounded_uncertified_count": 1,
                    }
                },
                reference_diagnostics={
                    (0, "R8[bounded]"): {"bounded_face_enumeration_count": 6_561}
                },
            )
        return result

    dependencies.local_runner = bounded_infeasible
    run_phase1a5(config, "source", False, tmp_path, 100, dependencies=dependencies)
    smoke = run_phase1a5(
        config, "smoke", False, tmp_path, 100, dependencies=dependencies
    )
    local = run_phase1a5(
        config, "local", False, tmp_path, 100, dependencies=dependencies
    )

    assert smoke["families"]["bounded"]["feasible"] is False
    assert smoke["families"]["bounded"]["r8_fallback_face_enumeration_count"] == 6_561
    assert local["bounded_included"] is False
    assert all(
        condition.family != "bounded"
        for values in calls[-1]["conditions"].values()
        for condition in values
    )


def test_execution_signature_isolates_checkpoint_paths_for_code_and_microbatch(
    tmp_path: Path,
) -> None:
    from ppmoe.experiments.phase1a5 import StagePrerequisiteError, run_phase1a5

    config = Phase1A5Config.load(CONFIG_PATH)
    dependencies, calls = _dependencies()
    run_phase1a5(config, "source", False, tmp_path, 100, dependencies=dependencies)
    first_smoke = run_phase1a5(
        config, "smoke", False, tmp_path, 100, dependencies=dependencies
    )
    first = run_phase1a5(
        config, "local", False, tmp_path, 100, dependencies=dependencies
    )
    with pytest.raises(StagePrerequisiteError, match="microbatch"):
        run_phase1a5(config, "local", False, tmp_path, 50, dependencies=dependencies)
    second_smoke = run_phase1a5(
        config, "smoke", False, tmp_path, 50, dependencies=dependencies
    )
    second = run_phase1a5(
        config, "local", False, tmp_path, 50, dependencies=dependencies
    )
    assert first["input_signature"] != second["input_signature"]
    assert first_smoke["input_signature"] != second_smoke["input_signature"]
    assert calls[-2]["checkpoint_dir"] != calls[-1]["checkpoint_dir"]
    first_local_checkpoint = calls[-2]["checkpoint_dir"]

    changed = replace(
        dependencies,
        source_verifier=lambda *_args, **_kwargs: {
            **_source_manifest(),
            "code_revision": {"commit": "2" * 40, "dirty": False},
        },
    )
    run_phase1a5(config, "source", False, tmp_path, 100, dependencies=changed)
    with pytest.raises(StagePrerequisiteError, match="source"):
        run_phase1a5(config, "local", False, tmp_path, 100, dependencies=changed)
    changed_smoke = run_phase1a5(
        config, "smoke", False, tmp_path, 100, dependencies=changed
    )
    changed_local = run_phase1a5(
        config, "local", False, tmp_path, 100, dependencies=changed
    )
    assert changed_smoke["input_signature"] != first_smoke["input_signature"]
    assert changed_local["input_signature"] != first["input_signature"]
    assert calls[-1]["checkpoint_dir"] != first_local_checkpoint


def test_compatible_smoke_evidence_is_immutable_on_resume_and_nonresume(
    tmp_path: Path,
) -> None:
    from ppmoe.experiments.phase1a5 import StageError, run_phase1a5

    config = Phase1A5Config.load(CONFIG_PATH)
    dependencies, calls = _dependencies()
    run_phase1a5(config, "source", False, tmp_path, 100, dependencies=dependencies)
    first = run_phase1a5(
        config, "smoke", False, tmp_path, 100, dependencies=dependencies
    )
    smoke_path = tmp_path / "stages" / "smoke.json"
    original_bytes = smoke_path.read_bytes()
    calls_after_first = len(calls)

    resumed = run_phase1a5(
        config, "smoke", True, tmp_path, 100, dependencies=dependencies
    )
    assert resumed == first
    assert smoke_path.read_bytes() == original_bytes
    assert len(calls) == calls_after_first
    with pytest.raises(StageError, match="immutable"):
        run_phase1a5(config, "smoke", False, tmp_path, 100, dependencies=dependencies)
    assert smoke_path.read_bytes() == original_bytes
    assert len(calls) == calls_after_first


def test_report_requires_terminal_predecessors_and_allows_selection_stop(
    tmp_path: Path,
) -> None:
    from ppmoe.experiments.phase1a5 import (
        StageError,
        StagePrerequisiteError,
        run_phase1a5,
    )

    config = Phase1A5Config.load(CONFIG_PATH)
    dependencies, _ = _dependencies()
    run_phase1a5(
        config, "source", False, tmp_path, 100, dependencies=dependencies
    )
    with pytest.raises(StagePrerequisiteError, match="local"):
        run_phase1a5(config, "select", False, tmp_path, 100, dependencies=dependencies)
    run_phase1a5(config, "smoke", False, tmp_path, 100, dependencies=dependencies)
    with pytest.raises(StagePrerequisiteError, match="local"):
        run_phase1a5(config, "report", False, tmp_path, 100, dependencies=dependencies)
    run_phase1a5(config, "local", False, tmp_path, 100, dependencies=dependencies)
    with pytest.raises(StagePrerequisiteError, match="select"):
        run_phase1a5(config, "report", False, tmp_path, 100, dependencies=dependencies)
    run_phase1a5(config, "select", False, tmp_path, 100, dependencies=dependencies)
    with pytest.raises(StagePrerequisiteError, match="rollout"):
        run_phase1a5(config, "report", False, tmp_path, 100, dependencies=dependencies)

    stopped = replace(
        dependencies,
        selection_runner=lambda _result, _config, output_path: (
            output_path.write_text(
                '{"practical_conditions":{},"free_bound_conditions":{}}',
                encoding="utf-8",
            ),
            {"practical_conditions": {}, "free_bound_conditions": {}},
        )[1],
    )
    run_phase1a5(config, "select", False, tmp_path, 100, dependencies=stopped)
    with pytest.raises(StageError, match="not implemented"):
        run_phase1a5(config, "report", False, tmp_path, 100, dependencies=stopped)
