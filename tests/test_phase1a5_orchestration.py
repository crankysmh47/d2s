import json
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ppmoe.experiments.phase1a5_config import Phase1A5Config
from ppmoe.oracle.reconstruction_local import (
    ConditionMetrics,
    ReconstructionLocalResult,
    SVDVerificationEvidence,
)
from ppmoe.provenance import sha256_file


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
        solver_diagnostics={
            (k, "bounded"): {
                "bounded_failure_count": 0,
                "bounded_uncertified_count": 0,
                "bounded_certified_count": 1,
                "bounded_solver_candidate_count": 1,
                "bounded_coefficient_bound_violation_count": 0,
            }
            for k in (4, 6)
        },
        reference_distances={},
        reference_diagnostics={},
        coefficient_quantiles={},
        completed_batches=1,
        failed_batches=0,
        complete=True,
        run_signature=run_signature,
        source_hashes=source_hashes,
        svd_verification={
            identity: SVDVerificationEvidence(
                identity=identity,
                sample_count=384,
                sample_count_by_layer={layer: 16 for layer in range(24)},
                rank_equal=True,
                max_relative_objective_error=5e-10,
            )
            for identity in (
                "sparse:K4:free_bound",
                "sparse:K6:free_bound",
                "R8[free_bound]",
            )
        },
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
    evidence = smoke["families"]["free_bound"]["svd_verification"]
    assert set(evidence) == {
        "sparse:K4:free_bound",
        "sparse:K6:free_bound",
        "R8[free_bound]",
    }
    assert all(record["sample_count"] == 384 for record in evidence.values())


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


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("absent", None),
        ("sample_count", 0),
        ("sample_count", 383),
        ("sample_count", 385),
        ("rank_equal", False),
        ("max_relative_objective_error", float("nan")),
        ("max_relative_objective_error", 1.01e-9),
    ],
)
def test_local_rejects_tampered_smoke_svd_evidence(
    tmp_path: Path, mutation: str, value: object
) -> None:
    from ppmoe.experiments.phase1a5 import StagePrerequisiteError, run_phase1a5

    config = Phase1A5Config.load(CONFIG_PATH)
    dependencies, _ = _dependencies()
    run_phase1a5(config, "source", False, tmp_path, 100, dependencies=dependencies)
    run_phase1a5(config, "smoke", False, tmp_path, 100, dependencies=dependencies)
    path = tmp_path / "stages" / "smoke.json"
    smoke = json.loads(path.read_text(encoding="utf-8"))
    evidence = smoke["families"]["free_bound"]["svd_verification"]
    if mutation == "absent":
        evidence.pop("sparse:K4:free_bound")
    else:
        evidence["sparse:K4:free_bound"][mutation] = value
    path.write_text(json.dumps(smoke), encoding="utf-8")

    with pytest.raises(StagePrerequisiteError, match="feasible smoke"):
        run_phase1a5(config, "local", False, tmp_path, 100, dependencies=dependencies)


def test_local_stage_rejects_incomplete_own_svd_evidence(tmp_path: Path) -> None:
    from ppmoe.experiments.phase1a5 import StageError, run_phase1a5

    config = Phase1A5Config.load(CONFIG_PATH)
    dependencies, _ = _dependencies()
    original = dependencies.local_runner
    calls = 0

    def drop_local_evidence(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert original is not None
        result = original(*args, **kwargs)
        return replace(result, svd_verification={}) if calls == 5 else result

    dependencies.local_runner = drop_local_evidence
    run_phase1a5(config, "source", False, tmp_path, 100, dependencies=dependencies)
    run_phase1a5(config, "smoke", False, tmp_path, 100, dependencies=dependencies)
    with pytest.raises(StageError, match="SVD verification"):
        run_phase1a5(config, "local", False, tmp_path, 100, dependencies=dependencies)


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


def test_unit_regression_consumes_captured_inference_tensors_inside_inference_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trainable linear rejects captured inference tensors outside this scope."""
    import ppmoe.experiments.phase1a5 as experiment

    class TeacherMLP(torch.nn.Identity):
        pass

    class Teacher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = TeacherMLP()
            self.model = SimpleNamespace(language_model=self._forward)

        def _forward(self, **_kwargs):
            return self.mlp(torch.ones((1, 1, 4)))

    class StudentMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def expert_outputs(self, values: torch.Tensor) -> torch.Tensor:
            return self.linear(values).unsqueeze(-2).expand(-1, -1, 8, -1)

    class Student(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = StudentMLP()

    monkeypatch.setattr(
        experiment, "iter_text_mlps", lambda model: ((0, model.mlp),)
    )
    monkeypatch.setattr(experiment, "PartitionedQwen35MLP", StudentMLP)
    subsets = torch.zeros((1, 4), dtype=torch.long)
    monkeypatch.setattr(
        experiment, "score_reconstruction", lambda *_args, **_kwargs: SimpleNamespace(subsets=subsets)
    )
    monkeypatch.setattr(
        experiment, "exact_gram_oracle", lambda *_args, **_kwargs: SimpleNamespace(subsets=subsets)
    )
    batch = SimpleNamespace(
        input_ids=torch.ones((1, 1), dtype=torch.long),
        attention_mask=torch.ones((1, 1), dtype=torch.long),
    )

    result = experiment._default_unit_regression(
        Teacher(), Student(), (batch,), Phase1A5Config.load(CONFIG_PATH)
    )

    assert result["checked_tokens"] == 1


def test_smoke_failure_is_preserved_without_a_success_stage(tmp_path: Path) -> None:
    from ppmoe.experiments.phase1a5 import StageError, run_phase1a5

    config = Phase1A5Config.load(CONFIG_PATH)
    dependencies, _ = _dependencies()
    dependencies.unit_regression_runner = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("captured inference tensor")
    )
    run_phase1a5(config, "source", False, tmp_path, 1, dependencies=dependencies)

    with pytest.raises(RuntimeError, match="captured inference tensor"):
        run_phase1a5(config, "smoke", False, tmp_path, 1, dependencies=dependencies)

    failures = list((tmp_path / "failures" / "smoke").glob("*.json"))
    assert len(failures) == 1
    evidence = json.loads(failures[0].read_text(encoding="utf-8"))
    assert evidence["status"] == "failed"
    assert evidence["exception_type"] == "RuntimeError"
    assert evidence["completed_batches"] is None
    assert not (tmp_path / "stages" / "smoke.json").exists()
    original = failures[0].read_bytes()
    with pytest.raises(StageError, match="immutable failure evidence"):
        run_phase1a5(config, "smoke", False, tmp_path, 1, dependencies=dependencies)
    assert failures[0].read_bytes() == original


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

    def stopped_selection_runner(result, _config, output_path):
        output_path.write_text(
            json.dumps(
                {
                    "practical_conditions": {},
                    "free_bound_conditions": {},
                    "source_hashes": dict(result.source_hashes),
                    "local_hashes": {"run_signature": result.run_signature},
                }
            ),
            encoding="utf-8",
        )
        return {"practical_conditions": {}, "free_bound_conditions": {}}

    stopped = replace(dependencies, selection_runner=stopped_selection_runner)
    run_phase1a5(config, "select", False, tmp_path, 100, dependencies=stopped)
    with pytest.raises(StageError, match="pinned M0 dense baseline hash drift"):
        run_phase1a5(config, "report", False, tmp_path, 100, dependencies=stopped)


@dataclass(frozen=True)
class _RolloutResult:
    complete: bool = True
    failed_batches: int = 0
    completed_batches: int = 1
    scored_tokens: int = 2
    nll_sum: float = 1.0
    ppl: float = 1.5
    peak_cuda_bytes: int = 123
    layer_metrics: dict[int, object] | None = None
    source_hashes: dict[str, str] | None = None


def test_rollout_uses_only_byte_frozen_registered_conditions_and_corpora(
    tmp_path: Path,
) -> None:
    from ppmoe.experiments.phase1a5 import Phase1A5Dependencies, run_phase1a5

    config = Phase1A5Config.load(CONFIG_PATH)
    base, _ = _dependencies()
    calls: list[tuple[str, int, str]] = []

    def selection_runner(_result, _config, output_path):
        output_path.write_bytes(
            b'{"free_bound_conditions":{"4":"free_bound"},"practical_conditions":{"4":"ridge:1","6":"bounded"}}\n'
        )
        return {
            "practical_conditions": {4: "ridge:1", 6: "bounded"},
            "free_bound_conditions": {4: "free_bound"},
        }

    def evaluation_loader(_root, _manifest):
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones_like(ids)
        return {"wikitext": (ids, mask), "fineweb_heldout": (ids, mask)}

    def rollout_runner(_teacher, _student, batches, condition, k, _checkpoint, _signature, *, source_hashes):
        materialized = tuple(batches)
        assert len(materialized) == 1
        calls.append(("unknown", k, condition.family if condition.rho is None else f"ridge:{condition.rho:g}"))
        return _RolloutResult(layer_metrics={0: {}}, source_hashes=dict(source_hashes))

    dependencies = Phase1A5Dependencies(
        source_verifier=base.source_verifier,
        calibration_loader=base.calibration_loader,
        converted_pair_loader=base.converted_pair_loader,
        local_runner=base.local_runner,
        unit_regression_runner=base.unit_regression_runner,
        selection_runner=selection_runner,
        cuda_peak_probe=base.cuda_peak_probe,
        cuda_peak_reset=base.cuda_peak_reset,
        clock=base.clock,
        evaluation_loader=evaluation_loader,
        rollout_runner=rollout_runner,
    )
    run_phase1a5(config, "source", False, tmp_path, 100, dependencies=dependencies)
    run_phase1a5(config, "smoke", False, tmp_path, 100, dependencies=dependencies)
    run_phase1a5(config, "local", False, tmp_path, 100, dependencies=dependencies)
    run_phase1a5(config, "select", False, tmp_path, 100, dependencies=dependencies)
    rollout = run_phase1a5(config, "rollout", False, tmp_path, 100, dependencies=dependencies)

    assert len(calls) == 10
    assert {item[1:] for item in calls} == {
        (4, "unit"), (4, "ridge:1"), (4, "free_bound"), (6, "unit"), (6, "bounded")
    }
    assert set(rollout["results"]) == {
        f"{corpus}:k{k}:{condition}"
        for corpus in ("wikitext", "fineweb_heldout")
        for k, condition in ((4, "unit"), (4, "ridge:1"), (4, "free_bound"), (6, "unit"), (6, "bounded"))
    }


def _write_evaluation_source(root: Path) -> dict[str, object]:
    hashes: dict[str, str] = {}
    for directory in ("wikitext", "fineweb-heldout"):
        token_dir = root / "tokens" / directory
        token_dir.mkdir(parents=True)
        input_ids = token_dir / "input_ids.pt"
        attention_mask = token_dir / "attention_mask.pt"
        torch.save(torch.tensor([[1, 2, 3]]), input_ids)
        torch.save(torch.ones((1, 3), dtype=torch.long), attention_mask)
        manifest = token_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "input_ids_sha256": sha256_file(input_ids),
                    "attention_mask_sha256": sha256_file(attention_mask),
                }
            ),
            encoding="utf-8",
        )
        hashes[directory] = sha256_file(manifest)
    return {"dataset_manifest_sha256": hashes}


@pytest.mark.parametrize("tensor_name", ("input_ids.pt", "attention_mask.pt"))
def test_evaluation_loader_rejects_mutated_tensor_after_manifest_hash(
    tmp_path: Path, tensor_name: str
) -> None:
    from ppmoe.experiments.phase1a5 import StageError, _load_evaluation_tokens

    source_manifest = _write_evaluation_source(tmp_path)
    torch.save(torch.tensor([[99, 99, 99]]), tmp_path / "tokens" / "wikitext" / tensor_name)

    with pytest.raises(StageError, match="tensor hash"):
        _load_evaluation_tokens(tmp_path, source_manifest)
