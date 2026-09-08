"""Integrity tests for the Phase 1A.5 hashed evidence report."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pytest

from ppmoe.experiments.phase1a5_config import Phase1A5Config


CONFIG = Phase1A5Config.load(Path(__file__).parents[1] / "configs" / "phase1-v0_5.toml")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _metric(
    *, error: float = 50.0, tokens: int = 100_000, k: int = 4
) -> dict[str, object]:
    coefficients = tokens * k
    return {
        "token_count": tokens,
        "total_squared_error": error,
        "total_d0_squared_norm": 100.0,
        "coefficient_count": coefficients,
        "negative_count": 0,
        "above_two_count": 0,
        "above_four_count": 0,
        "absolute_deviation_sum": 0.0,
        "squared_deviation_sum": 0.0,
        "unresolved_failures": 0,
        "bounded_lower_saturation_count": 0,
        "bounded_upper_saturation_count": 0,
        "coefficient_sum": float(coefficients),
        "coefficient_squared_sum": float(coefficients),
        "coefficient_mean": 1.0,
        "coefficient_std": 0.0,
        "coefficient_p01": 1.0,
        "coefficient_p05": 1.0,
        "coefficient_median": 1.0,
        "coefficient_p95": 1.0,
        "coefficient_p99": 1.0,
        "coefficient_p99_abs": 1.0,
        "coefficient_max_abs": 1.0,
        "negative_rate": 0.0,
        "above_two_rate": 0.0,
        "above_four_rate": 0.0,
        "bounded_lower_saturation_rate": 0.0,
        "bounded_upper_saturation_rate": 0.0,
        "mean_absolute_deviation_from_one": 0.0,
        "l2_deviation_from_one": 0.0,
        "l2_deviation_denominator": coefficients,
        "relative_error_sum": 0.0,
        "relative_error_mean": 0.0,
        "relative_error_median": 0.0,
        "relative_error_p90": 0.0,
        "relative_error_p95": 0.0,
        "relative_error_p99": 0.0,
        "cosine_sum": float(tokens),
        "cosine_defined_count": tokens,
        "cosine_undefined_count": 0,
        "cosine_mean": 1.0,
        "cosine_median": 1.0,
        "cosine_p90": 1.0,
        "cosine_p95": 1.0,
        "cosine_p99": 1.0,
        "gram_condition_defined_count": tokens,
        "gram_condition_undefined_count": 0,
        "gram_condition_median": 1.0,
        "gram_condition_p90": 1.0,
        "gram_condition_p99": 1.0,
        "gram_condition_max": 1.0,
        "effective_rank_counts": {str(k): tokens},
        "finite": True,
    }


def _solver(condition: str, tokens: int) -> dict[str, object]:
    values: dict[str, object] = {
        "solver_source_counts": {condition: tokens},
        "cholesky_fallback_count": 0,
        "degenerate_ridge_count": 0,
        "max_relative_system_residual": 0.0,
    }
    if condition == "bounded":
        values.update(
            {
                "bounded_failure_count": 0,
                "bounded_uncertified_count": 0,
                "bounded_certified_count": tokens,
                "bounded_solver_candidate_count": tokens,
                "bounded_coefficient_bound_violation_count": 0,
                "bounded_face_enumeration_count": tokens,
            }
        )
    return values


def _svd_evidence() -> dict[str, object]:
    return {
        identity: {
            "identity": identity,
            "sample_count": 384,
            "sample_count_by_layer": {str(layer): 16 for layer in range(24)},
            "rank_equal": True,
            "max_relative_objective_error": 5e-10,
        }
        for identity in (
            "sparse:K4:free_bound",
            "sparse:K6:free_bound",
            "R8[free_bound]",
        )
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
        str((layer, k, condition)): _metric(tokens=100_000, k=k)
        for layer in range(24)
        for k in (4, 6)
        for condition in ("unit", "free_bound", "ridge:10", "ridge:1", "ridge:0.1", "ridge:0.01", "ridge:0.001", "bounded")
    }
    return {
        "metrics": {
            str((k, condition)): _metric(
                error=100.0 if condition == "unit" else 50.0,
                tokens=2_400_000,
                k=k,
            )
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
            str((k, condition)): _solver(condition, 2_400_000)
            for k in (4, 6)
            for condition in ("unit", "free_bound", "ridge:10", "ridge:1", "ridge:0.1", "ridge:0.01", "ridge:0.001", "bounded")
        },
        "layer_solver_diagnostics": {
            str((layer, k, condition)): _solver(condition, 100_000)
            for layer in range(24)
            for k in (4, 6)
            for condition in ("unit", "free_bound", "ridge:10", "ridge:1", "ridge:0.1", "ridge:0.01", "ridge:0.001", "bounded")
        },
        "reference_distances": {
            str((0, "D0", "E8")): {"squared_error_sum": 1.0, "left_squared_norm_sum": 2.0, "right_squared_norm_sum": 2.0, "dot_sum": 1.0, "token_count": 100_000},
            str((0, "D0", "R8[free_bound]")): {"squared_error_sum": 1.0, "left_squared_norm_sum": 2.0, "right_squared_norm_sum": 2.0, "dot_sum": 1.0, "token_count": 100_000},
        },
        "subset_stability": {
            str((k, left, right)): {"jaccard_sum": 50.0, "token_count": 100}
            for k in (4, 6)
            for left, right in combinations(sorted(("unit", "free_bound", "ridge:10", "ridge:1", "ridge:0.1", "ridge:0.01", "ridge:0.001", "bounded")), 2)
        },
        "reference_diagnostics": {},
        "coefficient_quantiles": {
            str((k, condition)): {
                "p01": 1.0,
                "p05": 1.0,
                "median": 1.0,
                "p95": 1.0,
                "p99": 1.0,
                "p99_signed": 1.0,
                "max_abs": 1.0,
            }
            for k in (4, 6)
            for condition in ("unit", "free_bound", "ridge:10", "ridge:1", "ridge:0.1", "ridge:0.01", "ridge:0.001", "bounded")
        },
        "svd_verification": _svd_evidence(),
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
    from ppmoe.provenance import sha256_file

    source_root = root / "phase1-source"
    _write_json(source_root / "stages" / "m0.json", {
        "results": {
            "wikitext": {"teacher": {"ppl": 2.0}},
            "fineweb_heldout": {"teacher": {"ppl": 2.5}},
        }
    })
    source_hashes = {"stage:local": "a" * 64, "stage:m0": sha256_file(source_root / "stages" / "m0.json")}
    _write_json(root / "config.json", {"name": CONFIG.name})
    _write_json(
        root / "stages" / "source.json",
        _stage("source", source_hashes=source_hashes, source_manifest={"solver_version": CONFIG.solver_version, "source_root": str(source_root)}),
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
                         "elapsed_seconds": 1.0, "peak_cuda_bytes": 1,
                         "processed_tokens": 2_048, "completed_batches": 1,
                         **({"svd_verification": _svd_evidence()} if family == "free_bound" else {}),
                         "kkt_by_k": {
                             "4": {"bounded_failure_count": 0, "bounded_uncertified_count": 0, "bounded_certified_count": 1, "bounded_solver_candidate_count": 1, "bounded_coefficient_bound_violation_count": 0},
                             "6": {"bounded_failure_count": 0, "bounded_uncertified_count": 0, "bounded_certified_count": 1, "bounded_solver_candidate_count": 1, "bounded_coefficient_bound_violation_count": 0},
                         } if family == "bounded" else {}}
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
    assert "plot-source/primary-table.csv" in manifest.output_hashes
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


def test_report_refuses_fabricated_subset_stability(tmp_path: Path) -> None:
    from ppmoe.reporting.phase1a5 import ReportValidationError

    _write_complete_artifacts(tmp_path)
    _replace(tmp_path / "stages" / "local.json", ("result", "subset_stability"), {})

    with pytest.raises(ReportValidationError, match="subset stability evidence is absent"):
        _render(tmp_path)


def test_report_accepts_bounded_engineering_stop_when_bounded_is_absent(tmp_path: Path) -> None:
    _write_complete_artifacts(tmp_path)
    _make_bounded_engineering_stop(tmp_path)

    manifest = _render(tmp_path)

    assert "plot-source/smoke-feasibility.json" in manifest.output_hashes
    assert "engineering_infeasible_under_v0_5" in (tmp_path / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("keys", "value", "message"),
    [
        (("families", "unit", "elapsed_seconds"), None, "smoke elapsed"),
        (("families", "unit", "elapsed_seconds"), -1.0, "smoke elapsed"),
        (("families", "ridge", "peak_cuda_bytes"), -1, "smoke peak"),
        (("families", "free_bound", "processed_tokens"), 2_047, "smoke processed"),
        (("families", "bounded", "completed_batches"), 0, "smoke completed"),
    ],
)
def test_report_rejects_incomplete_smoke_family_measurements(
    tmp_path: Path, keys: tuple[str, ...], value: object, message: str
) -> None:
    from ppmoe.reporting.phase1a5 import ReportValidationError

    _write_complete_artifacts(tmp_path)
    _replace(tmp_path / "stages" / "smoke.json", keys, value)

    with pytest.raises(ReportValidationError, match=message):
        _render(tmp_path)


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
@pytest.mark.parametrize("stage", ("smoke", "local"))
def test_report_rejects_missing_false_or_out_of_range_svd_evidence(
    tmp_path: Path, stage: str, mutation: str, value: object
) -> None:
    from ppmoe.reporting.phase1a5 import ReportValidationError

    _write_complete_artifacts(tmp_path)
    path = tmp_path / "stages" / f"{stage}.json"
    payload = _load(path)
    evidence = (
        payload["families"]["free_bound"]["svd_verification"]
        if stage == "smoke"
        else payload["result"]["svd_verification"]
    )
    if mutation == "absent":
        evidence.pop("sparse:K4:free_bound")
    else:
        evidence["sparse:K4:free_bound"][mutation] = value
    _write_json(path, payload)

    with pytest.raises(ReportValidationError, match="SVD verification"):
        _render(tmp_path)


@pytest.mark.parametrize(
    ("scope", "field", "value"),
    [
        ("metrics", "coefficient_mean", None),
        ("metrics", "coefficient_std", float("nan")),
        ("metrics", "coefficient_p99_abs", float("inf")),
        ("metrics", "finite", "true"),
        ("metrics", "relative_error_p95", None),
        ("metrics", "cosine_p99", float("nan")),
        ("metrics", "gram_condition_max", None),
        ("layer_metrics", "above_four_rate", None),
        ("layer_metrics", "effective_rank_counts", {}),
    ],
)
def test_report_rejects_incomplete_or_nonfinite_local_metric_schema(
    tmp_path: Path, scope: str, field: str, value: object
) -> None:
    from ppmoe.reporting.phase1a5 import ReportValidationError

    _write_complete_artifacts(tmp_path)
    path = tmp_path / "stages" / "local.json"
    payload = _load(path)
    key = "(4, 'ridge:1')" if scope == "metrics" else "(0, 4, 'ridge:1')"
    payload["result"][scope][key][field] = value
    if scope == "metrics":
        for entry in payload["selection_metrics"]:
            if entry["k"] == 4 and entry["condition"] == "ridge:1":
                entry["metric"][field] = value
    _write_json(path, payload)

    with pytest.raises(ReportValidationError, match="local metric schema"):
        _render(tmp_path)


def test_report_rejects_missing_per_layer_solver_evidence(tmp_path: Path) -> None:
    from ppmoe.reporting.phase1a5 import ReportValidationError

    _write_complete_artifacts(tmp_path)
    _replace(
        tmp_path / "stages" / "local.json",
        ("result", "layer_solver_diagnostics"),
        {},
    )

    with pytest.raises(ReportValidationError, match="layer solver diagnostics"):
        _render(tmp_path)


def test_report_discloses_undefined_cosines_and_uses_real_condition_field(
    tmp_path: Path,
) -> None:
    _write_complete_artifacts(tmp_path)
    path = tmp_path / "stages" / "local.json"
    payload = _load(path)
    for scope in ("metrics", "layer_metrics"):
        for metric in payload["result"][scope].values():
            metric["cosine_defined_count"] -= 1
            metric["cosine_undefined_count"] = 1
            metric["cosine_sum"] -= 1.0
    for entry in payload["selection_metrics"]:
        entry["metric"]["cosine_defined_count"] -= 1
        entry["metric"]["cosine_undefined_count"] = 1
        entry["metric"]["cosine_sum"] -= 1.0
    _write_json(path, payload)

    manifest = _render(tmp_path)

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "undefined cosine" in report
    rank_source = json.loads(
        (tmp_path / "plot-source" / "rank-conditioning.json").read_text(
            encoding="utf-8"
        )
    )
    assert rank_source
    assert all("gram_condition_max" in row for row in rank_source)
    assert all("condition_max" not in row for row in rank_source)
    assert "plot-source/rank-conditioning.json" in manifest.output_hashes


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


def _make_bounded_engineering_stop(root: Path) -> None:
    local_path = root / "stages" / "local.json"
    local = _load(local_path)
    local["bounded_included"] = False
    local["selection_metrics"] = [row for row in local["selection_metrics"] if row["condition"] != "bounded"]
    for name in ("metrics", "layer_metrics", "selection_counts", "coefficient_quantiles"):
        local["result"][name] = {key: value for key, value in local["result"][name].items() if "bounded" not in key}
    local["result"]["solver_diagnostics"] = {key: value for key, value in local["result"]["solver_diagnostics"].items() if "bounded" not in key}
    local["result"]["layer_solver_diagnostics"] = {
        key: value
        for key, value in local["result"]["layer_solver_diagnostics"].items()
        if "bounded" not in key
    }
    local["result"]["subset_stability"] = {key: value for key, value in local["result"]["subset_stability"].items() if "bounded" not in key}
    _write_json(local_path, local)
    smoke_path = root / "stages" / "smoke.json"
    smoke = _load(smoke_path)
    smoke["families"]["bounded"].update({"feasible": False, "all_certified": False, "failures": 1, "kkt_by_k": {}, "bounded_stop_reason": "engineering_infeasible_under_v0_5"})
    _write_json(smoke_path, smoke)
