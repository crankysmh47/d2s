from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from ppmoe.recurrence.config_v2 import load_v2_config
from ppmoe.recurrence.data import Example, interpret
from ppmoe.recurrence.data_v1 import make_pilot as make_v1_pilot
from ppmoe.recurrence.config_v1 import load_v1_config
from ppmoe.recurrence.data_v2 import make_training_view_v2, render_v2_list_prompt


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "recurrence-v2.toml"
V1_CONFIG = ROOT / "configs" / "recurrence-v1.toml"


def test_v2_config_exposes_registered_values_and_is_distinct_from_v1() -> None:
    config = load_v2_config(CONFIG)

    assert config.source_run == "artifacts/recurrence-v0/129d39646933"
    assert config.v1_pilot_root == (
        "artifacts/recurrence-v1/4f1ddbedf7b4cbd11d73699d579a3e299a049aa8fb2b036e9124842d695eb2ca"
    )
    assert config.data.output_root == "artifacts/recurrence-v2"
    assert config.model_revision == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert (config.model_repo_id, config.dtype, config.attention_implementation) == (
        "Qwen/Qwen3-1.7B", "bfloat16", "eager"
    )
    assert (config.layer_start, config.layer_stop) == (12, 16)
    assert not config.use_cache and not config.enable_thinking and config.greedy
    assert (config.max_new_tokens, config.max_sequence_tokens) == (256, 512)
    assert config.data.pilot_per_candidate == 30
    assert config.data.development_per_family == 100
    assert config.data.sealed_per_family == 200
    assert config.data.maximum_attempts_per_example == 100
    assert config.data.maximum_absolute_expression_answer == 9999
    assert config.data.graph_edge_probability == 0.20
    assert config.data.expression_leaf_counts == (6, 7)
    assert config.data.list_item_counts == (4, 5)
    assert config.data.list_operation_counts == (1, 2)
    assert config.data.graph_vertex_counts == (7, 8)
    assert config.data.pilot_seeds == {
        "expressions": (721001, 721002),
        "lists": (722001, 722002),
        "graphs": (723001, 723002),
    }
    assert config.data.development_seeds == {"expressions": 731001, "lists": 731002, "graphs": 731003}
    assert config.data.sealed_seeds == {"expressions": 741001, "lists": 741002, "graphs": 741003}
    assert config.lora.seeds == (47, 48)
    assert (config.lora.rank, config.lora.alpha, config.lora.dropout) == (8, 16, 0.0)
    assert config.lora.target_modules == ("q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    assert config.optimizer.updates == 1000
    assert config.optimizer.microbatch_size == 1
    assert config.optimizer.accumulation_steps == 8
    assert config.optimizer.warmup_updates == 20
    assert config.resources.maximum_calibration_cuda_seconds == 7200
    assert config.resources.maximum_calibration_wall_seconds == 7200
    assert config.resources.maximum_study_cuda_seconds == 86400
    assert config.resources.maximum_artifact_bytes == 20 * 1024**3
    assert load_v1_config(V1_CONFIG).data.output_root == "artifacts/recurrence-v1"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('run = "artifacts/recurrence-v0/129d39646933"', 'run = "artifacts/elsewhere"', "source"),
        ('output_root = "artifacts/recurrence-v2"', 'output_root = "artifacts/recurrence-v1"', "output"),
        ('revision = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"', 'revision = "main"', "revision"),
        ("pilot_expressions = [721001, 721002]", "pilot_expressions = [621001, 621002]", "seed"),
        ("development = [731001, 731002, 731003]", "development = [631001, 631002, 631003]", "seed"),
        ("sealed = [741001, 741002, 741003]", "sealed = [641001, 641002, 641003]", "seed"),
        ("max_new_tokens = 256", "max_new_tokens = 255", "token"),
        ("max_sequence_tokens = 512", "max_sequence_tokens = 513", "token"),
        ("layer_start = 12", "layer_start = true", "integer"),
        ("pilot_per_candidate = 30", "rogue = 30", "unknown"),
        ("sealed_per_family = 200", "", "missing"),
        ('run = "artifacts/recurrence-v0/129d39646933"', 'run = "../outside"', "path"),
        ('output_root = "artifacts/recurrence-v2"', 'output_root = "C:\\\\outside"', "path"),
        (
            'v1_pilot_root = "artifacts/recurrence-v1/4f1ddbedf7b4cbd11d73699d579a3e299a049aa8fb2b036e9124842d695eb2ca"',
            'v1_pilot_root = "artifacts/recurrence-v1"',
            "pilot",
        ),
    ],
)
def test_v2_config_rejects_unregistered_or_unsafe_values(tmp_path, old, new, message) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    assert old in text
    invalid = tmp_path / "bad.toml"
    invalid.write_text(text.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_v2_config(invalid)


@pytest.mark.parametrize("value", ["7201", "true"])
def test_v2_config_rejects_changed_or_bool_calibration_wall_budget(tmp_path, value) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    old = "maximum_calibration_wall_seconds = 7200"
    assert old in text
    invalid = tmp_path / "bad-wall-budget.toml"
    invalid.write_text(text.replace(old, f"maximum_calibration_wall_seconds = {value}", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="wall"):
        load_v2_config(invalid)


def test_v2_list_prompt_matches_frozen_template_exactly() -> None:
    assert render_v2_list_prompt({"input": [3, 8, 7, 2], "ops": ["ADD1"]}) == (
        "Initial list: [3,8,7,2]. Operations to apply in order: ADD1. "
        "Apply exactly these operations once each, from left to right; do not apply any operation not listed. "
        "Definitions for listed operations: ADD1 adds 1 modulo 10 to each element. Return only the final list."
    )


def test_v2_list_prompt_preserves_repeated_operations_and_deduplicates_definitions() -> None:
    prompt = render_v2_list_prompt({"input": [1, 2], "ops": ["REVERSE", "ADD1", "REVERSE"]})

    assert "Operations to apply in order: REVERSE, ADD1, REVERSE." in prompt
    assert "Definitions for listed operations: REVERSE reverses the list; ADD1 adds 1 modulo 10 to each element." in prompt
    assert "ROTL1" not in prompt


def test_training_view_preserves_8000_rows_and_changes_only_list_prompts() -> None:
    v1 = load_v1_config(V1_CONFIG)
    rows = (
        replace(make_v1_pilot(v1, "expressions", 1, set())[0], split="train"),
        replace(make_v1_pilot(v1, "lists", 1, set())[0], split="train"),
        replace(make_v1_pilot(v1, "graphs", 1, set())[0], split="train"),
    )
    train_rows = tuple(rows[index % len(rows)] for index in range(8000))

    view = make_training_view_v2(train_rows)

    assert len(view) == 8000
    assert view is not train_rows
    for original, converted in zip(train_rows, view, strict=True):
        assert converted is not original
        assert (converted.id, converted.family, converted.split, converted.structure_id) == (
            original.id,
            original.family,
            original.split,
            original.structure_id,
        )
        assert converted.latent_spec == original.latent_spec
        assert converted.answer == original.answer
        assert converted.prompt == (
            render_v2_list_prompt(original.latent_spec) if original.family == "lists" else original.prompt
        )
        expected = interpret(converted.latent_spec, converted.family)
        assert converted.answer == (
            expected if converted.family == "graphs" else str(expected)
            if converted.family == "expressions" else json.dumps(expected, separators=(",", ":"))
        )


@pytest.mark.parametrize("bad_row", [None, "not an Example"])
def test_training_view_rejects_malformed_rows(bad_row) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_training_view_v2([bad_row])


def test_training_view_rejects_example_with_malformed_family() -> None:
    row = make_v1_pilot(load_v1_config(V1_CONFIG), "expressions", 1, set())[0]
    malformed = Example(row.id, [], "train", row.structure_id, row.prompt, row.answer, row.latent_spec)

    with pytest.raises(ValueError, match="malformed"):
        make_training_view_v2([malformed])


def test_training_view_rejects_answer_that_does_not_match_latent_spec() -> None:
    row = replace(make_v1_pilot(load_v1_config(V1_CONFIG), "expressions", 1, set())[0], split="train")
    mismatched = replace(row, answer="not the interpreted answer")

    with pytest.raises(ValueError, match="answer"):
        make_training_view_v2([mismatched])
