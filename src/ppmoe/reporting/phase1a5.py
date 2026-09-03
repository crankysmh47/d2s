"""Validated, hash-addressed reporting for the Phase 1A.5 oracle."""

from __future__ import annotations

import ast
import csv
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ppmoe.experiments.phase1a5_config import Phase1A5Config
from ppmoe.provenance import atomic_write_json, sha256_file


class ReportValidationError(RuntimeError):
    """Persisted evidence is incomplete, inconsistent, or unsafe to report."""


@dataclass(frozen=True, slots=True)
class Phase1A5ReportManifest:
    output_hashes: Mapping[str, str]
    source_stage_hashes: Mapping[str, str]
    decision: str
    local_conditions: Mapping[str, tuple[str, ...]]
    rollout_conditions: Mapping[str, tuple[str, ...]]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportValidationError(f"{label} is missing or invalid") from error
    if not isinstance(payload, dict):
        raise ReportValidationError(f"{label} is invalid")
    return payload


def _stage(root: Path, name: str, *, allow_scientific_stop: bool = False) -> dict[str, Any]:
    stage = _read_json(root / "stages" / f"{name}.json", f"{name} stage")
    allowed = {"complete", "scientifically_stopped"} if allow_scientific_stop else {"complete"}
    if stage.get("stage") != name or stage.get("status") not in allowed:
        raise ReportValidationError(f"{name} stage is incomplete")
    return stage


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(float(value)):
        raise ReportValidationError(f"{label} is non-finite")
    result = float(value)
    if positive and result <= 0:
        raise ReportValidationError(f"{label} must be positive")
    return result


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or (positive and value <= 0):
        raise ReportValidationError(f"{label} is invalid")
    return value


def _key(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, str):
        raise ReportValidationError(f"{label} has invalid key")
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as error:
        raise ReportValidationError(f"{label} has invalid key") from error
    if not isinstance(parsed, tuple):
        raise ReportValidationError(f"{label} has invalid key")
    return parsed


def _hashes(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ReportValidationError(f"{label} is missing")
    result = {str(key): str(item) for key, item in value.items()}
    if any(len(item) != 64 for item in result.values()):
        raise ReportValidationError(f"{label} is malformed")
    return result


def _validate_existing_manifest(root: Path) -> None:
    path = root / "report-manifest.json"
    if not path.exists():
        return
    manifest = _read_json(path, "existing report manifest")
    hashes = manifest.get("output_hashes")
    if not isinstance(hashes, Mapping):
        raise ReportValidationError("existing report manifest is malformed")
    for relative, expected in hashes.items():
        output = root / str(relative)
        if not output.is_file() or not isinstance(expected, str) or sha256_file(output) != expected:
            raise ReportValidationError("output hash mismatch")


def _metric(metric: object, label: str) -> dict[str, float | int | bool]:
    if not isinstance(metric, Mapping):
        raise ReportValidationError(f"{label} is missing")
    values = {
        "token_count": _integer(metric.get("token_count"), f"{label} token count", positive=True),
        "total_squared_error": _finite(metric.get("total_squared_error"), f"{label} error"),
        "total_d0_squared_norm": _finite(metric.get("total_d0_squared_norm"), f"{label} norm", positive=True),
        "coefficient_count": _integer(metric.get("coefficient_count"), f"{label} coefficient count", positive=True),
        "negative_count": _integer(metric.get("negative_count"), f"{label} negative count"),
        "above_two_count": _integer(metric.get("above_two_count"), f"{label} above-two count"),
        "absolute_deviation_sum": _finite(metric.get("absolute_deviation_sum"), f"{label} coefficient deviation"),
        "unresolved_failures": _integer(metric.get("unresolved_failures"), f"{label} unresolved failures"),
        "finite": bool(metric.get("finite")),
        "bounded_lower_saturation_count": _integer(metric.get("bounded_lower_saturation_count", 0), f"{label} lower saturation"),
        "bounded_upper_saturation_count": _integer(metric.get("bounded_upper_saturation_count", 0), f"{label} upper saturation"),
    }
    if not values["finite"]:
        raise ReportValidationError(f"{label} is non-finite")
    physical = ("total_squared_error", "total_d0_squared_norm", "absolute_deviation_sum", "negative_count", "above_two_count", "unresolved_failures", "bounded_lower_saturation_count", "bounded_upper_saturation_count")
    if any(float(values[name]) < 0 for name in physical):
        raise ReportValidationError(f"{label} has negative physical metric")
    if any(int(values[name]) > int(values["coefficient_count"]) for name in ("negative_count", "above_two_count", "bounded_lower_saturation_count", "bounded_upper_saturation_count")):
        raise ReportValidationError(f"{label} coefficient counts are invalid")
    return values


def _selection_conditions(selection: Mapping[str, Any], config: Phase1A5Config) -> dict[int, set[str]]:
    selected: dict[int, set[str]] = {k: {"unit"} for k in config.k_values}
    for field in ("practical_conditions", "free_bound_conditions"):
        values = selection.get(field)
        if not isinstance(values, Mapping):
            raise ReportValidationError("selection manifest is malformed")
        for raw_k, identifier in values.items():
            try:
                k = int(raw_k)
            except (TypeError, ValueError) as error:
                raise ReportValidationError("selection manifest names invalid K") from error
            if k not in selected or not isinstance(identifier, str):
                raise ReportValidationError("selection manifest names an unregistered condition")
            if field == "free_bound_conditions" and identifier != "free_bound":
                raise ReportValidationError("selection manifest free-bound condition is invalid")
            if identifier not in {"bounded", "free_bound", *(f"ridge:{rho:g}" for rho in config.rho_values)}:
                raise ReportValidationError("selection manifest names an unregistered condition")
            selected[k].add(identifier)
    return selected


def _write_source(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        raise ReportValidationError(f"authoritative source {path.name} is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_plot(path: Path, title: str, rows: list[Mapping[str, object]], x: str, y: str) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    values = [row for row in rows if isinstance(row.get(x), (int, float)) and isinstance(row.get(y), (int, float))]
    if values:
        axis.scatter([float(row[x]) for row in values], [float(row[y]) for row in values])
    axis.set(title=title, xlabel=x, ylabel=y)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _decision(selection: Mapping[str, Any]) -> str:
    practical = selection["practical_conditions"]
    free = selection["free_bound_conditions"]
    if not free:
        return "No registered numerical reconstruction qualifies; proceed to Phase 1B."
    if not practical:
        return "Registered free bound only; preserve the Registered Free-Reconstruction Candidate Bound and proceed to Phase 1B."
    practical_ids = set(practical.values())
    if "4" in practical:
        return f"K=4 practical frontier with {practical['4']} as the registered practical condition."
    if set(practical) == {"6"}:
        return "K=6-only frontier; retain the 75%-active FFN as the supported fallback frontier."
    if practical_ids == {"bounded"}:
        return "Practical bounded qualifies under the registered local protocol."
    return "Practical ridge qualifies under the registered local protocol."


def _dense_ppl_from_pinned_m0(source: Mapping[str, Any], source_hashes: Mapping[str, str]) -> dict[str, float]:
    root = source.get("source_root")
    expected = source_hashes.get("stage:m0")
    if not isinstance(root, str) or not isinstance(expected, str):
        raise ReportValidationError("pinned M0 dense baseline is missing")
    path = Path(root) / "stages" / "m0.json"
    if not path.is_file() or sha256_file(path) != expected:
        raise ReportValidationError("pinned M0 dense baseline hash drift")
    results = _read_json(path, "pinned M0 stage").get("results")
    if not isinstance(results, Mapping):
        raise ReportValidationError("pinned M0 dense baseline is invalid")
    dense: dict[str, float] = {}
    for corpus in ("wikitext", "fineweb_heldout"):
        entry = results.get(corpus)
        teacher = entry.get("teacher") if isinstance(entry, Mapping) else None
        dense[corpus] = _finite(teacher.get("ppl") if isinstance(teacher, Mapping) else None, f"pinned M0 {corpus} PPL", positive=True)
    return dense


def render_phase1a5_report(artifact_root: Path, config: Phase1A5Config) -> Phase1A5ReportManifest:
    """Render only from complete, immutable, provenance-compatible stage evidence."""
    root = Path(artifact_root)
    _validate_existing_manifest(root)
    source, smoke, local = (_stage(root, name) for name in ("source", "smoke", "local"))
    select = _stage(root, "select", allow_scientific_stop=True)
    selection_stopped = select.get("status") == "scientifically_stopped"
    rollout = None if selection_stopped else _stage(root, "rollout")
    source_hashes = _hashes(source.get("source_hashes"), "source hash manifest")
    source_manifest = source.get("source_manifest")
    if not isinstance(source_manifest, Mapping):
        raise ReportValidationError("source manifest is missing")
    dense_ppl = _dense_ppl_from_pinned_m0(source_manifest, source_hashes)
    stages_to_validate = (("smoke", smoke), ("local", local), ("select", select))
    if rollout is not None:
        stages_to_validate += (("rollout", rollout),)
    for name, stage in stages_to_validate:
        if _hashes(stage.get("source_hashes"), f"{name} source hashes") != source_hashes:
            raise ReportValidationError("source hash drift")
        if stage.get("source_stage_input_signature") != source.get("input_signature"):
            raise ReportValidationError("source stage signature drift")
    if local.get("valid_calibration_tokens") != 100_000 or smoke.get("valid_calibration_tokens") != 2_048:
        raise ReportValidationError("bad token coverage")
    unit = smoke.get("unit_regression")
    if not isinstance(unit, Mapping) or unit.get("passed") is not True or unit.get("checked_tokens") != 2_048:
        raise ReportValidationError("unit regression is incomplete")
    families = smoke.get("families")
    if not isinstance(families, Mapping):
        raise ReportValidationError("smoke family evidence is missing")
    for family in ("unit", "ridge", "free_bound", "bounded"):
        evidence = families.get(family)
        if not isinstance(evidence, Mapping):
            raise ReportValidationError("smoke failed batches")
        elapsed = _finite(evidence.get("elapsed_seconds"), "smoke elapsed")
        if elapsed < 0.0:
            raise ReportValidationError("smoke elapsed is negative")
        peak = evidence.get("peak_cuda_bytes")
        if not isinstance(peak, int) or isinstance(peak, bool) or peak < 0:
            raise ReportValidationError("smoke peak CUDA is invalid")
        if evidence.get("processed_tokens") != 2_048:
            raise ReportValidationError("smoke processed token coverage is invalid")
        if not isinstance(evidence.get("completed_batches"), int) or evidence["completed_batches"] <= 0:
            raise ReportValidationError("smoke completed batches are invalid")
        bounded_included = local.get("bounded_included") is True
        if family != "bounded" and evidence.get("failures") != 0:
            raise ReportValidationError("smoke failed batches")
        if family == "bounded" and bounded_included and evidence.get("all_certified") is not True:
            raise ReportValidationError("bounded KKT evidence is invalid")
        if family == "bounded" and bounded_included:
            if evidence.get("failures") != 0:
                raise ReportValidationError("bounded smoke failed batches")
            if evidence.get("feasible") is not True:
                raise ReportValidationError("bounded smoke is not feasible")
            kkt = evidence.get("kkt_by_k")
            if not isinstance(kkt, Mapping) or set(kkt) != {str(k) for k in config.k_values}:
                raise ReportValidationError("bounded KKT evidence is incomplete")
            for k, record in kkt.items():
                if not isinstance(record, Mapping):
                    raise ReportValidationError("bounded KKT evidence is malformed")
                required = ("bounded_failure_count", "bounded_uncertified_count", "bounded_certified_count", "bounded_solver_candidate_count", "bounded_coefficient_bound_violation_count")
                if any(not isinstance(record.get(name), int) or record[name] < 0 for name in required):
                    raise ReportValidationError("bounded KKT evidence is malformed")
                if record["bounded_failure_count"] or record["bounded_uncertified_count"] or record["bounded_coefficient_bound_violation_count"] or record["bounded_solver_candidate_count"] <= 0 or record["bounded_certified_count"] != record["bounded_solver_candidate_count"]:
                    raise ReportValidationError(f"bounded KKT evidence is invalid for K={k}")
        elif family == "bounded" and evidence.get("feasible") is not True:
            if not isinstance(evidence.get("bounded_stop_reason"), str) or not evidence["bounded_stop_reason"]:
                raise ReportValidationError("bounded engineering stop reason is missing")
        elif family == "bounded":
            raise ReportValidationError("bounded exclusion requires engineering stop")

    selection_path = root / "selection-manifest.json"
    expected_selection_hash = select.get("selection_manifest_sha256")
    if not isinstance(expected_selection_hash, str) or not selection_path.is_file() or sha256_file(selection_path) != expected_selection_hash:
        raise ReportValidationError("immutable selection manifest does not match its stage")
    selection = _read_json(selection_path, "selection manifest")
    if _hashes(selection.get("source_hashes"), "selection source hashes") != source_hashes:
        raise ReportValidationError("selection source hash drift")
    if selection.get("local_hashes", {}).get("run_signature") != local.get("result", {}).get("run_signature"):
        raise ReportValidationError("selection local signature drift")
    selected = _selection_conditions(selection, config)
    if selection_stopped:
        if select.get("scientific_stop_reason") != "selection_no_eligible_conditions":
            raise ReportValidationError("selection scientific stop is invalid")
        if selection.get("practical_conditions") or selection.get("free_bound_conditions"):
            raise ReportValidationError("selection scientific stop has eligible conditions")

    local_result = local.get("result")
    if not isinstance(local_result, Mapping) or local_result.get("complete") is not True or local_result.get("failed_batches") != 0:
        raise ReportValidationError("local failed batches")
    metrics = local_result.get("metrics")
    staged_metrics = local.get("selection_metrics")
    if not isinstance(metrics, Mapping) or not isinstance(staged_metrics, list):
        raise ReportValidationError("local metrics are missing")
    expected_ids = {"unit", "free_bound", *(f"ridge:{rho:g}" for rho in config.rho_values)}
    if local.get("bounded_included") is True:
        expected_ids.add("bounded")
    local_rows: list[dict[str, object]] = []
    observed: set[tuple[int, str]] = set()
    for entry in staged_metrics:
        if not isinstance(entry, Mapping):
            raise ReportValidationError("missing local condition")
        k, identifier = entry.get("k"), entry.get("condition")
        if not isinstance(k, int) or k not in config.k_values or not isinstance(identifier, str):
            raise ReportValidationError("missing local condition")
        parsed = _metric(entry.get("metric"), f"local K={k} {identifier}")
        if (k, identifier) in observed:
            raise ReportValidationError("duplicate local metric key")
        observed.add((k, identifier))
        family, separator, rho_value = identifier.partition(":")
        local_rows.append({"k": k, "condition": identifier, "family": family, "rho": float(rho_value) if separator else None, "constraint": "[0,2]" if identifier == "bounded" else None, **parsed, "nmse": parsed["total_squared_error"] / parsed["total_d0_squared_norm"], "active_fraction": k / 8})
    if observed != {(k, identifier) for k in config.k_values for identifier in expected_ids}:
        raise ReportValidationError("missing local condition")
    by_condition = {(int(row["k"]), str(row["condition"])): row for row in local_rows}
    gain_recovery_rows: list[dict[str, object]] = []
    for row in local_rows:
        k, identifier = int(row["k"]), str(row["condition"])
        unit_row = by_condition[(k, "unit")]
        free_row = by_condition[(k, "free_bound")]
        denominator = float(unit_row["total_squared_error"]) - float(free_row["total_squared_error"])
        tolerance = 1e-9 * max(abs(float(unit_row["total_squared_error"])), abs(float(free_row["total_squared_error"])), 1.0)
        recovery = None if identifier == "free_bound" or denominator <= tolerance else (float(unit_row["total_squared_error"]) - float(row["total_squared_error"])) / denominator
        gain_recovery_rows.append({"k": k, "condition": identifier, "gain_recovery": recovery})
    persisted_metrics: dict[tuple[int, str], dict[str, float | int | bool]] = {}
    for key, value in metrics.items():
        parsed = _key(key, "local metric")
        if len(parsed) != 2 or not isinstance(parsed[0], int) or not isinstance(parsed[1], str):
            raise ReportValidationError("local persisted metric key is invalid")
        metric_key = (parsed[0], parsed[1])
        if metric_key in persisted_metrics:
            raise ReportValidationError("duplicate local persisted metric key")
        persisted_metrics[metric_key] = _metric(value, "local persisted metric")
    if set(persisted_metrics) != observed:
        raise ReportValidationError("local metric reconciliation is incomplete")
    for row in local_rows:
        key = (int(row["k"]), str(row["condition"]))
        if persisted_metrics[key] != {name: row[name] for name in persisted_metrics[key]}:
            raise ReportValidationError("local metric reconciliation mismatch")
    for k, values in selected.items():
        if any((k, identifier) not in observed for identifier in values):
            raise ReportValidationError("selected condition is absent from local metrics")
        if "bounded" in values and local.get("bounded_included") is not True:
            raise ReportValidationError("bounded selection lacks feasible smoke evidence")

    layer_metrics = local_result.get("layer_metrics")
    if not isinstance(layer_metrics, Mapping):
        raise ReportValidationError("layer coverage is missing")
    observed_layers: set[int] = set()
    layer_rows: list[dict[str, object]] = []
    for key, value in layer_metrics.items():
        parsed = _key(key, "layer metric")
        if len(parsed) != 3 or not isinstance(parsed[0], int):
            raise ReportValidationError("layer coverage is invalid")
        metric = _metric(value, "layer metric")
        observed_layers.add(parsed[0])
        layer_rows.append({"layer": parsed[0], "k": parsed[1], "condition": parsed[2], "nmse": metric["total_squared_error"] / metric["total_d0_squared_norm"]})
    if observed_layers != set(range(24)):
        raise ReportValidationError("layer coverage is incomplete")
    solver = local_result.get("solver_diagnostics")
    if not isinstance(solver, Mapping):
        raise ReportValidationError("solver diagnostics are missing")
    solver_rows: list[dict[str, object]] = []
    for key, value in solver.items():
        parsed = _key(key, "solver diagnostic")
        if not isinstance(value, Mapping):
            raise ReportValidationError("solver diagnostics are invalid")
        if len(parsed) == 2 and parsed[1] == "bounded" and (value.get("bounded_failure_count", 0) != 0 or value.get("bounded_uncertified_count", 0) != 0):
            raise ReportValidationError("bounded KKT evidence is invalid")
        solver_rows.append({"key": key, **{str(k): v for k, v in value.items()}})

    expected_rollout = set() if selection_stopped else {f"{corpus}:k{k}:{condition}" for corpus in ("wikitext", "fineweb_heldout") for k, conditions in selected.items() for condition in conditions}
    results = {} if rollout is None else rollout.get("results")
    if not isinstance(results, Mapping) or set(results) != expected_rollout:
        raise ReportValidationError("evaluation condition leakage or incomplete registered rollout")
    evaluation_rows: list[dict[str, object]] = []
    for name, result in results.items():
        if not isinstance(result, Mapping) or result.get("complete") is not True or result.get("failed_batches") != 0:
            raise ReportValidationError("evaluation failed batches")
        ppl = _finite(result.get("ppl"), f"evaluation PPL {name}", positive=True)
        scored = _integer(result.get("scored_tokens"), f"evaluation scored tokens {name}", positive=True)
        layers = result.get("layer_metrics")
        if not isinstance(layers, Mapping) or {int(layer) for layer in layers} != set(range(24)):
            raise ReportValidationError("evaluation layer coverage is incomplete")
        for metric in layers.values():
            if not isinstance(metric, Mapping):
                raise ReportValidationError("evaluation layer metric is invalid")
            _finite(metric.get("hidden_squared_error_sum"), "evaluation hidden error")
            _finite(metric.get("teacher_squared_norm_sum"), "evaluation teacher norm", positive=True)
        corpus, k_token, condition = str(name).split(":", 2)
        evaluation_rows.append({"corpus": corpus, "k": int(k_token.removeprefix("k")), "condition": condition, "ppl": ppl, "scored_tokens": scored})

    reference_rows = []
    references = local_result.get("reference_distances", {})
    if not isinstance(references, Mapping):
        raise ReportValidationError("reference distances are invalid")
    for key, value in references.items():
        if not isinstance(value, Mapping):
            raise ReportValidationError("reference distances are invalid")
        reference_rows.append({"key": key, **{str(k): v for k, v in value.items()}})
    reference_labels = {
        str(item)
        for row in reference_rows
        for item in _key(str(row["key"]), "reference distance")[1:]
    }
    if not {"D0", "E8"}.issubset(reference_labels) or not any(
        label.startswith("R8[") for label in reference_labels
    ):
        raise ReportValidationError("D0/E8/R8 distance evidence is incomplete")
    stability = local_result.get("subset_stability")
    if not isinstance(stability, Mapping) or not stability:
        raise ReportValidationError("subset stability evidence is absent")
    stability_rows: list[dict[str, object]] = []
    expected_stability = {
        (k, left, right)
        for k in config.k_values
        for index, left in enumerate(sorted(identifier for candidate_k, identifier in observed if candidate_k == k))
        for right in sorted(identifier for candidate_k, identifier in observed if candidate_k == k)[index + 1 :]
    }
    seen_stability: set[tuple[int, str, str]] = set()
    for key, value in stability.items():
        if not isinstance(value, Mapping):
            raise ReportValidationError("subset stability evidence is malformed")
        parsed = _key(key, "subset stability")
        if len(parsed) != 3 or not isinstance(parsed[0], int) or not isinstance(parsed[1], str) or not isinstance(parsed[2], str):
            raise ReportValidationError("subset stability evidence is malformed")
        identity = (parsed[0], parsed[1], parsed[2])
        if identity in seen_stability:
            raise ReportValidationError("duplicate subset stability evidence")
        seen_stability.add(identity)
        total = _finite(value.get("jaccard_sum"), "subset stability")
        tokens = _integer(value.get("token_count"), "subset stability token count", positive=True)
        score = total / tokens
        if score < 0.0 or score > 1.0:
            raise ReportValidationError("subset stability is outside [0,1]")
        stability_rows.append({"key": key, "jaccard": score, **{str(name): item for name, item in value.items()}})
    if seen_stability != expected_stability:
        raise ReportValidationError("subset stability pair coverage is incomplete")
    plot_source = root / "plot-source"
    sources = {
        "local-metrics.json": local_rows,
        "coefficient-distributions.json": [{key: row[key] for key in ("k", "condition", "coefficient_count", "negative_count", "above_two_count", "bounded_lower_saturation_count", "bounded_upper_saturation_count")} for row in local_rows],
        "bound-saturation.json": [{key: row[key] for key in ("k", "condition", "bounded_lower_saturation_count", "bounded_upper_saturation_count")} for row in local_rows],
        "coefficient-distortion-gain.json": [{"k": row["k"], "condition": row["condition"], "coefficient_distortion": row["absolute_deviation_sum"], "gain_recovery": next(item["gain_recovery"] for item in gain_recovery_rows if item["k"] == row["k"] and item["condition"] == row["condition"])} for row in local_rows],
        "gain-recovery.json": gain_recovery_rows,
        "rank-conditioning.json": solver_rows,
        "layer-improvements.json": layer_rows,
        "subset-stability.json": stability_rows,
        "reference-distances.json": reference_rows,
        "evaluation-ppl.json": evaluation_rows,
    }
    for name, value in sources.items():
        _write_source(plot_source / name, value)
    feasibility_rows = [
        {"family": family, "feasible": evidence.get("feasible"), "bounded_stop_reason": evidence.get("bounded_stop_reason")}
        for family, evidence in families.items()
        if isinstance(evidence, Mapping)
    ]
    _write_source(plot_source / "smoke-feasibility.json", feasibility_rows)
    primary_rows: list[dict[str, object]] = []
    for row in local_rows:
        matching = [item for item in evaluation_rows if item["k"] == row["k"] and item["condition"] == row["condition"]]
        if not matching:
            matching = [{"corpus": None, "ppl": None, "scored_tokens": None}]
        unit_sse = float(by_condition[(int(row["k"]), "unit")]["total_squared_error"])
        for evaluation in matching:
            corpus, ppl = evaluation["corpus"], evaluation["ppl"]
            evaluated = corpus is not None
            baseline = dense_ppl[str(corpus)] if evaluated else None
            primary_rows.append({**row, "local_relative_improvement": (unit_sse - float(row["total_squared_error"])) / unit_sse, "corpus": corpus, "ppl": ppl, "ppl_delta": float(ppl) - baseline if evaluated else None, "ppl_ratio": float(ppl) / baseline if evaluated else None, "scored_tokens": evaluation["scored_tokens"], "deployment_status": "evaluated" if evaluated else "not_evaluated"})
    _write_source(plot_source / "primary-table.json", primary_rows)
    _write_csv(plot_source / "primary-table.csv", primary_rows)
    plots = root / "plots"
    specs = (
        ("01-local-nmse-rho.png", "Local NMSE versus ridge rho", local_rows, "rho", "nmse"),
        ("02-local-active-fraction.png", "Local NMSE versus active fraction", local_rows, "active_fraction", "nmse"),
        ("03-evaluation-ppl.png", "Evaluation PPL", evaluation_rows, "k", "ppl"),
        ("04-coefficient-distributions.png", "Coefficient above-two rate", local_rows, "active_fraction", "above_two_count"),
        ("05-coefficient-distortion-gain.png", "Coefficient distortion versus gain", local_rows, "absolute_deviation_sum", "nmse"),
        ("06-rank-conditioning-gain.png", "Rank and conditioning diagnostics", solver_rows, "effective_rank_sum", "condition_max"),
        ("07-layer-improvements.png", "Layerwise local NMSE", layer_rows, "layer", "nmse"),
        ("08-reference-distances.png", "D0/E8/R8 reference distances", reference_rows, "token_count", "squared_error_sum"),
        ("09-subset-stability.png", "Cross-condition subset Jaccard", stability_rows, "token_count", "jaccard"),
    )
    for name, title, rows, x, y in specs:
        _save_plot(plots / name, title, rows, x, y)
    outputs = [*sorted(plot_source.glob("*.json")), *sorted(plot_source.glob("*.csv")), *sorted(plots.glob("*.png"))]
    output_hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in outputs}
    stage_names = ("source", "smoke", "local", "select") if selection_stopped else ("source", "smoke", "local", "select", "rollout")
    source_stage_hashes = {f"stages/{name}.json": sha256_file(root / "stages" / f"{name}.json") for name in stage_names}
    decision = _decision(selection)
    bounded_stop_reason = families["bounded"].get("bounded_stop_reason") if isinstance(families.get("bounded"), Mapping) else None
    report_lines = [
        "# Phase 1A.5 sparse reconstruction oracle report", "",
        f"Decision: {decision}", "",
        "The Registered Free-Reconstruction Candidate Bound is a registered candidate bound, not an exact-span optimum.",
        "This report makes no claim of an exact-span optimum, a trained router, or an inference speedup.",
        "Gain recovery is analysis-only and never affects eligibility.",
    ]
    if isinstance(bounded_stop_reason, str):
        report_lines.append(f"Bounded engineering stop: {bounded_stop_reason}.")
    report = "\n".join(report_lines)
    (root / "report.md").write_text(report, encoding="utf-8")
    output_hashes["report.md"] = sha256_file(root / "report.md")
    manifest = Phase1A5ReportManifest(
        output_hashes=output_hashes,
        source_stage_hashes=source_stage_hashes,
        decision=decision,
        local_conditions={str(k): tuple(sorted(identifier for candidate_k, identifier in observed if candidate_k == k)) for k in config.k_values},
        rollout_conditions={str(k): tuple(sorted(values)) for k, values in selected.items()},
    )
    atomic_write_json(root / "report-manifest.json", asdict(manifest))
    return manifest
