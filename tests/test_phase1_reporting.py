import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from ppmoe.config import Phase1Config
from ppmoe.experiments.phase1_v0 import _config_signature
from ppmoe.experiments.phase1_v0 import run_phase1
from ppmoe.provenance import atomic_write_json, sha256_file
from ppmoe.reporting.phase1 import (
    ReportValidationError,
    _plot_local,
    _plot_perplexity,
    _plot_rollout_hidden,
    render_phase1_report,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "phase1-v0.toml"
K_VALUES = (8, 6, 4, 3, 2)


def _matrix(value: float) -> list[list[float]]:
    return [[value for _ in range(8)] for _ in range(8)]


def _write_finalized_stages(root: Path, config: Phase1Config) -> None:
    signature = _config_signature(config)
    m0_results = {}
    rollout_results = {}
    for corpus, dense_ppl in (("wikitext", 10.0), ("fineweb_heldout", 20.0)):
        m0_results[corpus] = {
            "passed": True,
            "teacher": {"complete": True, "failed_batches": 0, "ppl": dense_ppl, "scored_tokens": 5},
            "student": {"complete": True, "failed_batches": 0, "ppl": dense_ppl, "scored_tokens": 5},
        }
        for k in K_VALUES:
            sparse_ppl = dense_ppl * (1.0 + (8 - k) / 80.0)
            rollout_results[f"{corpus}:k{k}"] = {
                "complete": True,
                "failed_batches": 0,
                "ppl": sparse_ppl,
                "scored_tokens": 5,
                "layer_metrics": {
                    str(layer): {
                        "hidden_squared_error_sum": float((layer + 1) * k),
                        "teacher_squared_norm_sum": float((layer + 1) * 100),
                        "token_count": 5,
                    }
                    for layer in range(config.num_hidden_layers)
                },
                "selection_counts": {
                    str(layer): [1] * 8
                    for layer in range(config.num_hidden_layers)
                },
            }
    layer_k = {}
    selection_counts = {}
    co_selection = {}
    for layer in range(config.num_hidden_layers):
        for k in K_VALUES:
            error = float((layer + 1) * (9 - k))
            norm = float((layer + 1) * 100)
            layer_k[f"({layer}, {k})"] = {
                "squared_error_sum": error,
                "target_squared_norm_sum": norm,
                "token_count": 5,
            }
            selection_counts[f"({layer}, {k})"] = [k] * 8
            co_selection[f"({layer}, {k})"] = _matrix(float(k))
    local = {
        "result": {
            "complete": True,
            "failed_batches": 0,
            "direct_check_count": 10,
            "all_expert_check_count": 10,
            "layer_k": layer_k,
            "selection_counts": selection_counts,
            "co_selection": co_selection,
            "interaction_sum": [
                _matrix(float(-(layer + 1)))
                for layer in range(config.num_hidden_layers)
            ],
            "cancellation_count": [
                _matrix(float(layer + 1))
                for layer in range(config.num_hidden_layers)
            ],
        }
    }
    for name, payload in (("m0", {"results": m0_results}), ("local", local), ("rollout", {"results": rollout_results})):
        atomic_write_json(
            root / "stages" / f"{name}.json",
            {"config_signature": signature, "input_signature": name, "stage": name, "status": "complete", **payload},
        )


def _drop_local_layer(value: dict) -> None:
    for k in K_VALUES:
        value["result"]["layer_k"].pop(f"(23, {k})")


def test_reporting_writes_exact_sources_plots_and_hashes(tmp_path: Path) -> None:
    config = Phase1Config.load(CONFIG_PATH)
    _write_finalized_stages(tmp_path, config)

    manifest = render_phase1_report(tmp_path, config)

    assert manifest.local_k4_normalized_mse == pytest.approx(15 / 300)
    assert manifest.layers == tuple(range(24))
    assert manifest.ppl_ratios["wikitext:4"] == pytest.approx(1.05)
    assert manifest.ppl_ratios["fineweb_heldout:4"] == pytest.approx(1.05)
    local_rows = list(csv.DictReader((tmp_path / "plot-source" / "local-k.csv").open()))
    assert tuple(local_rows[0]) == (
        "k",
        "active_fraction",
        "squared_error_sum",
        "target_squared_norm_sum",
        "normalized_mse",
    )
    k4 = next(row for row in local_rows if row["k"] == "4")
    assert float(k4["active_fraction"]) == pytest.approx(0.5)
    assert float(k4["normalized_mse"]) == pytest.approx(15 / 300)
    ppl_rows = list(csv.DictReader((tmp_path / "plot-source" / "perplexity.csv").open()))
    wiki_k4 = next(row for row in ppl_rows if row["corpus"] == "wikitext" and row["k"] == "4")
    assert float(wiki_k4["ppl_ratio"]) == pytest.approx(1.05)
    assert float(wiki_k4["delta_percent"]) == pytest.approx(5.0)
    assert len(manifest.output_hashes) == 10
    for relative, digest in manifest.output_hashes.items():
        path = tmp_path / relative
        assert path.exists() and path.stat().st_size > 0
        assert sha256_file(path) == digest
    saved = json.loads((tmp_path / "report-manifest.json").read_text())
    assert saved["source_stage_hashes"] == manifest.source_stage_hashes
    assert saved["layers"] == list(range(24))

    previous_hash = manifest.source_stage_hashes["m0"]
    m0_path = tmp_path / "stages" / "m0.json"
    m0 = json.loads(m0_path.read_text())
    m0["extra_provenance"] = "changed"
    atomic_write_json(m0_path, m0)
    changed = render_phase1_report(tmp_path, config)
    assert changed.source_stage_hashes["m0"] != previous_hash


def test_report_stage_and_cli_accept_report(monkeypatch, tmp_path: Path) -> None:
    from scripts import run_phase1_v0

    config = Phase1Config.load(CONFIG_PATH)
    _write_finalized_stages(tmp_path, config)
    stage = run_phase1(config, "report", False, tmp_path, 1)
    assert stage["status"] == "complete"
    assert (tmp_path / "stages" / "report.json").exists()

    monkeypatch.setattr(run_phase1_v0.Phase1Config, "load", lambda _path: config)
    monkeypatch.setattr(run_phase1_v0, "run_phase1", lambda *_args, **_kwargs: {"status": "complete"})
    assert run_phase1_v0.main(["--stage", "report"]) == 0


def test_metric_plots_annotate_k_and_counts_with_locked_references() -> None:
    local = _plot_local([
        {"k": 4, "active_fraction": 0.5, "normalized_mse": 0.1, "token_count": 120},
    ])
    rollout = _plot_rollout_hidden([
        {"corpus": "wikitext", "k": 4, "active_fraction": 0.5, "normalized_mse": 0.2, "token_count": 80},
    ], ("wikitext",))
    perplexity = _plot_perplexity([
        {"corpus": "wikitext", "k": 4, "active_fraction": 0.5, "ppl_ratio": 1.05, "scored_tokens": 64},
    ], ("wikitext",))

    assert {text.get_text() for text in local.axes[0].texts} == {"K=4\ntokens=120"}
    assert {text.get_text() for text in rollout.axes[0].texts} == {"K=4\ntokens=80"}
    assert {text.get_text() for text in perplexity.axes[0].texts} == {"K=4\nscored=64"}
    assert any(set(line.get_ydata()) == {0.0} for line in local.axes[0].lines)
    assert any(set(line.get_ydata()) == {0.0} for line in rollout.axes[0].lines)
    assert any(set(line.get_ydata()) == {1.0} for line in perplexity.axes[0].lines)
    assert all(
        figure.axes[0].get_xlabel() == "Active FFN fraction (K / 8)"
        for figure in (local, rollout, perplexity)
    )
    for figure in (local, rollout, perplexity):
        plt.close(figure)


@pytest.mark.parametrize(
    ("stage", "mutation", "message"),
    [
        ("m0", lambda value: value["results"]["wikitext"]["teacher"].__setitem__("ppl", None), "finite"),
        ("m0", lambda value: value["results"]["wikitext"]["student"].__setitem__("ppl", None), "finite"),
        ("local", lambda value: value["result"].__setitem__("complete", False), "incomplete"),
        ("local", lambda value: value["result"].pop("all_expert_check_count"), "all-expert"),
        (
            "local",
            lambda value: value["result"].__setitem__("all_expert_check_count", 0),
            "all-expert",
        ),
        ("local", _drop_local_layer, "missing"),
        ("rollout", lambda value: value["results"].pop("wikitext:k4"), "missing"),
        (
            "rollout",
            lambda value: value["results"]["wikitext:k4"]["layer_metrics"].pop("23"),
            "missing",
        ),
    ],
)
def test_reporting_rejects_incomplete_missing_and_nonfinite_inputs(tmp_path: Path, stage, mutation, message: str) -> None:
    config = Phase1Config.load(CONFIG_PATH)
    _write_finalized_stages(tmp_path, config)
    path = tmp_path / "stages" / f"{stage}.json"
    value = json.loads(path.read_text())
    mutation(value)
    atomic_write_json(path, value)

    with pytest.raises(ReportValidationError, match=message):
        render_phase1_report(tmp_path, config)
