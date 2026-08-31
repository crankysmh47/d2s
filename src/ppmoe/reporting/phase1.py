"""Validated, value-preserving Phase 1 V0 CSV and plot rendering."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ppmoe.config import Phase1Config
from ppmoe.provenance import atomic_write_json, sha256_file


class ReportValidationError(ValueError):
    """Finalized stage data cannot support a scientifically valid report."""


@dataclass(frozen=True, slots=True)
class ReportManifest:
    output_hashes: Mapping[str, str]
    source_stage_hashes: Mapping[str, str]
    k_values: tuple[int, ...]
    layers: tuple[int, ...]
    corpora: tuple[str, ...]
    local_k4_normalized_mse: float
    ppl_ratios: Mapping[str, float]


def _config_signature(config: Phase1Config) -> str:
    from ppmoe.experiments.phase1_v0 import _config_signature as signature

    return signature(config)


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if value is None:
        raise ReportValidationError(f"{name} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ReportValidationError(f"{name} must be finite") from error
    if not isfinite(numeric) or (positive and numeric <= 0.0):
        raise ReportValidationError(f"{name} must be finite" + (" and positive" if positive else ""))
    return numeric


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError) as error:
        raise ReportValidationError(f"{name} must be an integer") from error
    if positive and numeric <= 0:
        raise ReportValidationError(f"{name} must be positive")
    return numeric


def _read_stage(artifact_root: Path, stage: str, config_signature: str) -> tuple[dict[str, Any], str]:
    path = artifact_root / "stages" / f"{stage}.json"
    if not path.exists():
        raise ReportValidationError(f"missing {stage} stage")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReportValidationError(f"invalid {stage} stage JSON") from error
    if state.get("status") != "complete":
        raise ReportValidationError(f"{stage} stage is incomplete")
    if state.get("config_signature") != config_signature:
        raise ReportValidationError(f"{stage} stage signature does not match")
    return state, sha256_file(path)


def _parse_layer_k(key: str) -> tuple[int, int]:
    match = re.fullmatch(r"\(?\s*(\d+)\s*[, :]\s*(\d+)\s*\)?", str(key))
    if match is None:
        raise ReportValidationError(f"invalid layer/K key: {key}")
    return int(match.group(1)), int(match.group(2))


def _matrix(value: Any, name: str, experts: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != experts:
        raise ReportValidationError(f"{name} matrix has the wrong shape")
    rows = []
    for row in value:
        if not isinstance(row, list) or len(row) != experts:
            raise ReportValidationError(f"{name} matrix has the wrong shape")
        rows.append([_finite(item, name) for item in row])
    return np.array(rows, dtype=np.float64)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_plot(path: Path, render: Any) -> None:
    figure = render()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def render_phase1_report(artifact_root: Path, config: Phase1Config) -> ReportManifest:
    """Validate finalized metrics and render only direct, unsmoothed summaries."""

    artifact_root = Path(artifact_root)
    config_signature = _config_signature(config)
    m0, m0_hash = _read_stage(artifact_root, "m0", config_signature)
    local, local_hash = _read_stage(artifact_root, "local", config_signature)
    rollout, rollout_hash = _read_stage(artifact_root, "rollout", config_signature)
    source_stage_hashes = {"m0": m0_hash, "local": local_hash, "rollout": rollout_hash}
    k_values = tuple(config.k_values)
    experts = config.num_experts
    corpora = ("wikitext", "fineweb_heldout")

    m0_results = m0.get("results")
    if not isinstance(m0_results, Mapping):
        raise ReportValidationError("m0 results are missing")
    dense_ppl: dict[str, float] = {}
    for corpus in corpora:
        result = m0_results.get(corpus)
        if not isinstance(result, Mapping) or not result.get("passed", False):
            raise ReportValidationError(f"missing or failed M0 result for {corpus}")
        m0_models: dict[str, Mapping[str, Any]] = {}
        for role in ("teacher", "student"):
            model_result = result.get(role)
            if not isinstance(model_result, Mapping) or not model_result.get("complete", False):
                raise ReportValidationError(f"M0 {role} result for {corpus} is incomplete")
            if _integer(model_result.get("failed_batches"), f"M0 {corpus} {role} failed batches") != 0:
                raise ReportValidationError(f"M0 {role} result for {corpus} is incomplete")
            _integer(model_result.get("scored_tokens"), f"M0 {corpus} {role} scored tokens", positive=True)
            _finite(model_result.get("ppl"), f"M0 {role} PPL for {corpus}", positive=True)
            m0_models[role] = model_result
        dense_ppl[corpus] = _finite(m0_models["teacher"].get("ppl"), f"M0 dense PPL for {corpus}", positive=True)

    local_result = local.get("result")
    if not isinstance(local_result, Mapping) or not local_result.get("complete", False):
        raise ReportValidationError("local result is incomplete")
    if _integer(local_result.get("failed_batches"), "local failed batches") != 0:
        raise ReportValidationError("local result is incomplete")
    if _integer(local_result.get("direct_check_count"), "local direct checks", positive=True) <= 0:
        raise ReportValidationError("local direct checks are missing")
    layer_k_raw = local_result.get("layer_k")
    if not isinstance(layer_k_raw, Mapping):
        raise ReportValidationError("local layer/K metrics are missing")
    parsed_layer_k: dict[tuple[int, int], Mapping[str, Any]] = {}
    for key, value in layer_k_raw.items():
        if not isinstance(value, Mapping):
            raise ReportValidationError("local layer/K metric is invalid")
        parsed_layer_k[_parse_layer_k(str(key))] = value
    expected_layers = tuple(range(config.num_hidden_layers))
    observed_local_layers = tuple(sorted({layer for layer, _ in parsed_layer_k}))
    if observed_local_layers != expected_layers:
        raise ReportValidationError("missing or unexpected local layer coverage")
    layers = expected_layers
    for layer in layers:
        for k in k_values:
            if (layer, k) not in parsed_layer_k:
                raise ReportValidationError(f"missing local layer/K entry for layer {layer}, K={k}")

    local_rows: list[dict[str, object]] = []
    layer_k_rows: list[dict[str, object]] = []
    local_summary: dict[int, tuple[float, float]] = {}
    local_token_counts: dict[int, int] = {}
    for k in k_values:
        error_sum = 0.0
        norm_sum = 0.0
        token_counts: set[int] = set()
        for layer in layers:
            metric = parsed_layer_k[(layer, k)]
            error = _finite(metric.get("squared_error_sum"), f"local error L{layer} K{k}")
            norm = _finite(metric.get("target_squared_norm_sum"), f"local norm L{layer} K{k}", positive=True)
            tokens = _integer(metric.get("token_count"), f"local tokens L{layer} K{k}", positive=True)
            token_counts.add(tokens)
            error_sum += error
            norm_sum += norm
            layer_k_rows.append({
                "layer": layer, "k": k, "active_fraction": k / experts,
                "squared_error_sum": error, "target_squared_norm_sum": norm,
                "normalized_mse": error / norm, "token_count": tokens,
            })
        if norm_sum == 0.0:
            raise ReportValidationError(f"local aggregate denominator for K={k} is zero")
        if len(token_counts) != 1:
            raise ReportValidationError(f"local token counts differ across layers for K={k}")
        local_token_counts[k] = next(iter(token_counts))
        local_summary[k] = (error_sum, norm_sum)
        local_rows.append({
            "k": k, "active_fraction": k / experts, "squared_error_sum": error_sum,
            "target_squared_norm_sum": norm_sum, "normalized_mse": error_sum / norm_sum,
        })

    interaction_layers = local_result.get("interaction_sum")
    cancellation_layers = local_result.get("cancellation_count")
    co_selection_raw = local_result.get("co_selection")
    if not isinstance(interaction_layers, list) or not isinstance(cancellation_layers, list) or not isinstance(co_selection_raw, Mapping):
        raise ReportValidationError("local expert matrices are missing")
    if len(interaction_layers) != len(layers) or len(cancellation_layers) != len(layers):
        raise ReportValidationError("local expert matrix layer coverage is incomplete")
    interaction = sum((_matrix(value, "interaction", experts) for value in interaction_layers), np.zeros((experts, experts)))
    cancellation = sum((_matrix(value, "cancellation", experts) for value in cancellation_layers), np.zeros((experts, experts)))
    co_selection = np.zeros((experts, experts))
    for layer in layers:
        key = f"({layer}, 4)"
        if key not in co_selection_raw:
            raise ReportValidationError("missing K4 co-selection matrix")
        co_selection += _matrix(co_selection_raw[key], "K4 co-selection", experts)
    expert_rows: list[dict[str, object]] = []
    for name, matrix in (("interaction_sum", interaction), ("cancellation_count", cancellation), ("k4_co_selection", co_selection)):
        for expert_i in range(experts):
            for expert_j in range(experts):
                expert_rows.append({"matrix": name, "expert_i": expert_i, "expert_j": expert_j, "value": float(matrix[expert_i, expert_j])})

    rollout_results = rollout.get("results")
    if not isinstance(rollout_results, Mapping):
        raise ReportValidationError("rollout results are missing")
    rollout_rows: list[dict[str, object]] = []
    perplexity_rows: list[dict[str, object]] = []
    ppl_ratios: dict[str, float] = {}
    for corpus in corpora:
        for k in k_values:
            key = f"{corpus}:k{k}"
            result = rollout_results.get(key)
            if not isinstance(result, Mapping):
                raise ReportValidationError(f"missing rollout result for {key}")
            if not result.get("complete", False) or _integer(result.get("failed_batches"), f"rollout failures {key}") != 0:
                raise ReportValidationError(f"rollout result for {key} is incomplete")
            scored = _integer(result.get("scored_tokens"), f"rollout scored tokens {key}", positive=True)
            sparse_ppl = _finite(result.get("ppl"), f"rollout PPL {key}", positive=True)
            layer_metrics = result.get("layer_metrics")
            if not isinstance(layer_metrics, Mapping):
                raise ReportValidationError(f"rollout layer metrics for {key} are missing")
            try:
                observed_rollout_layers = tuple(sorted(int(layer) for layer in layer_metrics))
            except (TypeError, ValueError) as error:
                raise ReportValidationError(f"invalid rollout layer coverage for {key}") from error
            if len(layer_metrics) != len(layers) or observed_rollout_layers != layers:
                raise ReportValidationError(f"missing or unexpected rollout layer coverage for {key}")
            final_layer = layers[-1]
            final_metric = layer_metrics[str(final_layer)]
            if not isinstance(final_metric, Mapping):
                raise ReportValidationError(f"invalid final rollout metric for {key}")
            error = _finite(final_metric.get("hidden_squared_error_sum"), f"rollout error {key}")
            norm = _finite(final_metric.get("teacher_squared_norm_sum"), f"rollout norm {key}", positive=True)
            tokens = _integer(final_metric.get("token_count"), f"rollout tokens {key}", positive=True)
            ratio = sparse_ppl / dense_ppl[corpus]
            if not isfinite(ratio):
                raise ReportValidationError(f"rollout PPL ratio {key} is nonfinite")
            rollout_rows.append({
                "corpus": corpus, "k": k, "active_fraction": k / experts, "layer": final_layer,
                "hidden_squared_error_sum": error, "teacher_squared_norm_sum": norm,
                "normalized_mse": error / norm, "token_count": tokens,
            })
            perplexity_rows.append({
                "corpus": corpus, "k": k, "active_fraction": k / experts,
                "dense_ppl": dense_ppl[corpus], "sparse_ppl": sparse_ppl,
                "ppl_ratio": ratio, "delta_percent": (ratio - 1.0) * 100.0,
                "scored_tokens": scored,
            })
            ppl_ratios[f"{corpus}:{k}"] = ratio

    plot_source = artifact_root / "plot-source"
    plots = artifact_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    _write_csv(plot_source / "local-k.csv", list(local_rows[0]), local_rows)
    _write_csv(plot_source / "rollout-hidden.csv", list(rollout_rows[0]), rollout_rows)
    _write_csv(plot_source / "perplexity.csv", list(perplexity_rows[0]), perplexity_rows)
    _write_csv(plot_source / "layer-k.csv", list(layer_k_rows[0]), layer_k_rows)
    _write_csv(plot_source / "expert-matrices.csv", list(expert_rows[0]), expert_rows)

    local_plot_rows = [
        {**row, "token_count": local_token_counts[int(row["k"])]}
        for row in local_rows
    ]
    _save_plot(plots / "01-local-k.png", lambda: _plot_local(local_plot_rows))
    _save_plot(plots / "02-rollout-hidden.png", lambda: _plot_rollout_hidden(rollout_rows, corpora))
    _save_plot(plots / "03-perplexity.png", lambda: _plot_perplexity(perplexity_rows, corpora))
    _save_plot(plots / "04-layer-k-heatmap.png", lambda: _plot_layer_k(layer_k_rows, layers, k_values))
    _save_plot(plots / "05-expert-matrices.png", lambda: _plot_expert_matrices(interaction, cancellation, co_selection))

    output_paths = [
        plot_source / "local-k.csv", plot_source / "rollout-hidden.csv", plot_source / "perplexity.csv",
        plot_source / "layer-k.csv", plot_source / "expert-matrices.csv",
        plots / "01-local-k.png", plots / "02-rollout-hidden.png", plots / "03-perplexity.png",
        plots / "04-layer-k-heatmap.png", plots / "05-expert-matrices.png",
    ]
    output_hashes = {path.relative_to(artifact_root).as_posix(): sha256_file(path) for path in output_paths}
    manifest = ReportManifest(
        output_hashes=output_hashes,
        source_stage_hashes=source_stage_hashes,
        k_values=k_values,
        layers=layers,
        corpora=corpora,
        local_k4_normalized_mse=local_summary[4][0] / local_summary[4][1],
        ppl_ratios=ppl_ratios,
    )
    atomic_write_json(artifact_root / "report-manifest.json", asdict(manifest))
    return manifest


def _plot_local(rows: list[Mapping[str, object]]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot([row["active_fraction"] for row in rows], [row["normalized_mse"] for row in rows], marker="o")
    axis.axhline(0.0, color="black", linestyle="--")
    for row in rows:
        axis.annotate(
            f"K={row['k']}\ntokens={row['token_count']}",
            (row["active_fraction"], row["normalized_mse"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set(xlabel="Active FFN fraction (K / 8)", ylabel="Weighted local normalized MSE")
    return figure


def _plot_rollout_hidden(rows: list[Mapping[str, object]], corpora: tuple[str, ...]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6, 4))
    for corpus in corpora:
        subset = [row for row in rows if row["corpus"] == corpus]
        axis.plot([row["active_fraction"] for row in subset], [row["normalized_mse"] for row in subset], marker="o", label=corpus)
        for row in subset:
            axis.annotate(
                f"K={row['k']}\ntokens={row['token_count']}",
                (row["active_fraction"], row["normalized_mse"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axis.axhline(0.0, color="black", linestyle="--")
    axis.set(xlabel="Active FFN fraction (K / 8)", ylabel="Final-layer normalized MSE")
    axis.legend()
    return figure


def _plot_perplexity(rows: list[Mapping[str, object]], corpora: tuple[str, ...]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6, 4))
    for corpus in corpora:
        subset = [row for row in rows if row["corpus"] == corpus]
        axis.plot([row["active_fraction"] for row in subset], [row["ppl_ratio"] for row in subset], marker="o", label=corpus)
        for row in subset:
            axis.annotate(
                f"K={row['k']}\nscored={row['scored_tokens']}",
                (row["active_fraction"], row["ppl_ratio"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axis.axhline(1.0, color="black", linestyle="--", label="dense reference")
    axis.set(xlabel="Active FFN fraction (K / 8)", ylabel="Sparse / dense perplexity")
    axis.legend()
    return figure


def _plot_layer_k(rows: list[Mapping[str, object]], layers: tuple[int, ...], k_values: tuple[int, ...]) -> plt.Figure:
    values = np.array([[next(float(row["normalized_mse"]) for row in rows if row["layer"] == layer and row["k"] == k) for k in k_values] for layer in layers])
    figure, axis = plt.subplots(figsize=(7, 4))
    image = axis.imshow(values, aspect="auto")
    axis.set(xticks=range(len(k_values)), xticklabels=k_values, yticks=range(len(layers)), yticklabels=layers, xlabel="K", ylabel="Text layer")
    figure.colorbar(image, ax=axis, label="Normalized MSE")
    return figure


def _plot_expert_matrices(interaction: np.ndarray, cancellation: np.ndarray, co_selection: np.ndarray) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, matrix, title in zip(axes, (interaction, cancellation, co_selection), ("Interaction sum (signed)", "Cancellation count", "K=4 co-selection"), strict=True):
        image = axis.imshow(matrix, aspect="equal")
        axis.set(title=title, xlabel="Expert j", ylabel="Expert i")
        figure.colorbar(image, ax=axis)
    return figure
