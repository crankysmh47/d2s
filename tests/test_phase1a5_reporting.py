"""Integrity tests for the Phase 1A.5 hashed evidence report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppmoe.experiments.phase1a5_config import Phase1A5Config


CONFIG = Phase1A5Config.load(Path(__file__).parents[1] / "configs" / "phase1-v0_5.toml")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _metric(*, error: float = 50.0, tokens: int = 100_000) -> dict[str, object]:
    return {
        "token_count": tokens,
        "total_squared_error": error,
        "total_d0_squared_norm": 100.0,
        "coefficient_count": 100,
        "negative_count": 0,
        "above_two_count": 0,
        "absolute_deviation_sum": 1.0,
        "unresolved_failures": 0,
        "bounded_lower_saturation_count": 0,
        "bounded_upper_saturation_count": 0,
        "finite": True,
    }


def _stage(name: str, **payload: object) -> dict[str, object]:
    return {
        "stage": name,
        "status": "complete",
        "config_signature": "locked-config",
        "input_signature": f"{name}-signature",
        **payload,
    }


def _local_result() -> dict[str, object]:
    layers = {
        str((layer, k, condition)): _metric(tokens=100_000)
        for layer in range(24)
        for k in (4, 6)
        for condition in ("unit", "free_bound", "ridge:10", "ridge:1", "ridge:0.1", "ridge:0.01", "ridge:0.001", "bounded")
    }
    return {
        "metrics": {
            str((k, condition)): _metric(error=100.0 if condition == "unit" else 50.0)
            for k in (4, 6)
            for condition in ("unit", "free_bound", "ridge:10", "ridge:1", "ridge:0.1", "ridge:0.01", "ridge:0.001", "bounded")
        },
        "layer_metrics": layers,
        "selection_counts": {
            str((layer, k, condition)): [1] * 8
            for layer in range(24)
            for k in (4, 6)
            for condition in ("unit", "free_bound", "ridge:10", "ridge:1", "ridge:0.1", "ridge:0.01", "ridge:0.001", "bounded")
        },
        "co_selection": {},
        "solver_diagnostics": {
            str((k, "bounded")): {"bounded_failure_count": 0, "bounded_uncertified_count": 0, "effective_rank_sum": 4, "condition_max": 2.0}
            for k in (4, 6)
        },
        "reference_distances": {str((0, "D0", "E8")): {"squared_error_sum": 1.0, "left_squared_norm_sum": 2.0, "right_squared_norm_sum": 2.0, "dot_sum": 1.0, "token_count": 100_000}},
        "reference_diagnostics": {},
        "coefficient_quantiles": {str((k, condition)): {"p99_abs": 1.0} for k in (4, 6) for condition in ("unit", "free_bound", "ridge:10", "ridge:1", "ridge:0.1", "ridge:0.01", "ridge:0.001", "bounded")},
        "completed_batches": 4,
        "failed_batches": 0,
        "complete": True,
        "run_signature": "local-signature",
        "source_hashes": {"stage:local": "a" * 64},
    }


def _rollout_result() -> dict[str, object]:
    return {
        "complete": True,
        "failed_batches": 0,
        "completed_batches": 1,
        "scored_tokens": 100,
        "ppl": 2.0,
        "layer_metrics": {
            str(layer): {
                "token_count": 100,
                "hidden_squared_error_sum": 1.0,
                "teacher_squared_norm_sum": 2.0,
            }
            for layer in range(24)
        },
        "coefficient_diagnostics": {"coefficient_count": 100, "negative_count": 0},
        "solver_diagnostics": {str(layer): {} for layer in range(24)},
        "reference_distances": {str(layer): {} for layer in range(24)},
    }


def _write_complete_artifacts(root: Path) -> None:
    source_hashes = {"stage:local": "a" * 64}
    _write_json(root / "config.json", {"name": CONFIG.name})
    _write_json(
        root / "stages" / "source.json",
        _stage("source", source_hashes=source_hashes, source_manifest={"solver_version": CONFIG.solver_version}),
    )
    _write_json(
        root / "stages" / "smoke.json",
        _stage(
            "smoke",
            source_hashes=source_hashes,
            source_stage_input_signature="source-signature",
            valid_calibration_tokens=2_048,
            unit_regression={"passed": True, "checked_tokens": 2_048},
            families={
                family: {"failures": 0, "all_certified": True, "feasible": True,
                         "elapsed_seconds": 1.0, "peak_cuda_bytes": 1}
                for family in ("unit", "ridge", "free_bound", "bounded")
            },
        ),
    )
    local_result = _local_result()
    _write_json(
        root / "stages" / "local.json",
        _stage(
            "local",
            source_hashes=source_hashes,
            source_stage_input_signature="source-signature",
            valid_calibration_tokens=100_000,
            bounded_included=True,
            selection_metrics=[
                {"k": k, "condition": condition, "metric": local_result["metrics"][str((k, condition))]}
                for k in (4, 6)
                for condition in ("unit", "free_bound", "ridge:10", "ridge:1", "ridge:0.1", "ridge:0.01", "ridge:0.001", "bounded")
            ],
            result=local_result,
        ),
    )
    selection = {
        "practical_conditions": {"4": "ridge:1"},
        "free_bound_conditions": {"4": "free_bound", "6": "free_bound"},
        "source_hashes": source_hashes,
        "local_hashes": {"run_signature": "local-signature"},
    }
    _write_json(root / "selection-manifest.json", selection)
    from ppmoe.provenance import sha256_file

    _write_json(
        root / "stages" / "select.json",
        _stage(
            "select",
            source_hashes=source_hashes,
            source_stage_input_signature="source-signature",
            selection_manifest_sha256=sha256_file(root / "selection-manifest.json"),
        ),
    )
    _write_json(
        root / "stages" / "rollout.json",
        _stage(
            "rollout",
            source_hashes=source_hashes,
            source_stage_input_signature="source-signature",
            selection_manifest_sha256=sha256_file(root / "selection-manifest.json"),
            results={
                f"{corpus}:k{k}:{condition}": _rollout_result()
                for corpus in ("wikitext", "fineweb_heldout")
                for k, conditions in ((4, ("unit", "ridge:1", "free_bound")), (6, ("unit", "free_bound")))
                for condition in conditions
            },
        ),
    )


def _render(root: Path):
    from ppmoe.reporting.phase1a5 import render_phase1a5_report

    return render_phase1a5_report(root, CONFIG)


def test_report_writes_hashed_authoritative_sources_and_conservative_decision(tmp_path: Path) -> None:
    _write_complete_artifacts(tmp_path)

    manifest = _render(tmp_path)

    assert manifest.output_hashes
    assert all((tmp_path / path).is_file() for path in manifest.output_hashes)
    assert "plot-source/local-metrics.json" in manifest.output_hashes
    assert "plots/01-local-nmse-rho.png" in manifest.output_hashes
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Registered Free-Reconstruction Candidate Bound" in report
    assert "no claim of an exact-span optimum" in report
    assert "trained router" in report and "inference speedup" in report


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda root: _replace(root / "stages" / "local.json", ("source_hashes", "stage:local"), "b" * 64), "source hash"),
        (lambda root: _drop_local_condition(root), "missing local condition"),
        (lambda root: _mutate_selection(root), "immutable selection"),
        (lambda root: _add_unregistered_rollout(root), "evaluation condition leakage"),
        (lambda root: _replace(root / "stages" / "local.json", ("result", "metrics", "(4, 'ridge:1')", "total_squared_error"), float("nan")), "non-finite"),
        (lambda root: _replace(root / "stages" / "local.json", ("result", "layer_metrics"), {}), "layer coverage"),
        (lambda root: _replace(root / "stages" / "local.json", ("result", "failed_batches"), 1), "failed batches"),
        (lambda root: _replace(root / "stages" / "local.json", ("result", "solver_diagnostics", "(4, 'bounded')", "bounded_uncertified_count"), 1), "bounded KKT"),
    ],
)
def test_report_rejects_integrity_and_completeness_defects(tmp_path: Path, mutate, message: str) -> None:
    from ppmoe.reporting.phase1a5 import ReportValidationError

    _write_complete_artifacts(tmp_path)
    mutate(tmp_path)

    with pytest.raises(ReportValidationError, match=message):
        _render(tmp_path)


def test_report_rejects_existing_output_hash_mismatch(tmp_path: Path) -> None:
    from ppmoe.reporting.phase1a5 import ReportValidationError

    _write_complete_artifacts(tmp_path)
    manifest = _render(tmp_path)
    victim = tmp_path / next(iter(manifest.output_hashes))
    victim.write_bytes(b"tampered")

    with pytest.raises(ReportValidationError, match="output hash mismatch"):
        _render(tmp_path)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(path: Path, keys: tuple[str, ...], value: object) -> None:
    payload = _load(path)
    target = payload
    for key in keys[:-1]:
        target = target[key]  # type: ignore[index]
    target[keys[-1]] = value  # type: ignore[index]
    _write_json(path, payload)


def _drop_local_condition(root: Path) -> None:
    path = root / "stages" / "local.json"
    payload = _load(path)
    payload["selection_metrics"] = [
        item for item in payload["selection_metrics"]
        if not (item["k"] == 4 and item["condition"] == "ridge:1")
    ]
    _write_json(path, payload)


def _mutate_selection(root: Path) -> None:
    path = root / "selection-manifest.json"
    payload = _load(path)
    payload["practical_conditions"] = {"4": "bounded"}
    _write_json(path, payload)


def _add_unregistered_rollout(root: Path) -> None:
    path = root / "stages" / "rollout.json"
    payload = _load(path)
    payload["results"] = {"wikitext:k6:ridge:1": {"complete": True}}
    _write_json(path, payload)
