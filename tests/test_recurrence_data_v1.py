from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ppmoe.recurrence.config_v1 import load_v1_config
from ppmoe.recurrence.data import Example, interpret
from ppmoe.recurrence.data_v1 import (
    freeze_v1_split,
    make_pilot,
    make_selected_splits,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "recurrence-v1.toml"


class FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return list(range(len(messages[0]["content"].split()) + 3))

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split()) + 1))}


def test_v1_config_is_separate_and_exposes_registered_values():
    config = load_v1_config(CONFIG)
    assert config.source_run == "artifacts/recurrence-v0/129d39646933"
    assert config.model_revision == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert config.data.pilot_per_candidate == 30
    assert config.data.development_per_family == 100
    assert config.data.sealed_per_family == 200
    assert config.optimizer.updates == 1000
    assert config.lora.seeds == (47, 48)
    assert config.resources.maximum_study_cuda_seconds == 86400


@pytest.mark.parametrize(
    "old,new",
    [
        ('revision = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"', 'revision = "main"'),
        ("max_new_tokens = 256", "max_new_tokens = 480"),
        ("max_sequence_tokens = 512", "max_sequence_tokens = 520"),
        ("rank = 8", "rank = true"),
    ],
)
def test_v1_config_rejects_mutable_revision_and_impossible_budgets(tmp_path, old, new):
    text = CONFIG.read_text(encoding="utf-8")
    invalid = tmp_path / "bad.toml"
    invalid.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError):
        load_v1_config(invalid)


def test_v1_config_rejects_unknown_keys(tmp_path):
    invalid = tmp_path / "unknown.toml"
    invalid.write_text("rogue = true\n" + CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown configuration key"):
        load_v1_config(invalid)


@pytest.mark.parametrize("family,candidate", [("expressions", 1), ("lists", 2), ("graphs", 1)])
def test_pilot_generation_is_deterministic_and_has_valid_answers(family, candidate):
    config = load_v1_config(CONFIG)
    first = make_pilot(config, family, candidate, set())
    second = make_pilot(config, family, candidate, set())
    assert len(first) == 30
    assert first == second
    assert all(row.split == "pilot" and row.family == family for row in first)
    for row in first:
        assert row.answer == (interpret(row.latent_spec, family) if family == "graphs" else json.dumps(interpret(row.latent_spec, family), separators=(",", ":")) if family == "lists" else str(interpret(row.latent_spec, family)))


def test_registered_first_expression_row_is_exact():
    row = make_pilot(load_v1_config(CONFIG), "expressions", 1, set())[0]
    assert row.id == "dc20519ff9f8621a9f6411377b97face956ecd70bb83b07fa5da5317ca916337"
    assert row.structure_id == "chain6"
    assert row.prompt == "Compute (((((6 + 7) + 8) - 8) + 5) * 3). Return only the integer answer."
    assert row.answer == "54"


def test_independent_seed_streams_change_generated_rows():
    config = load_v1_config(CONFIG)
    one = make_pilot(config, "expressions", 1, set())
    two = make_pilot(config, "expressions", 2, set())
    assert {row.id for row in one}.isdisjoint(row.id for row in two)
    assert one != two


def test_pilots_alternate_expression_shape_and_balance_graph_labels():
    config = load_v1_config(CONFIG)
    expressions = make_pilot(config, "expressions", 1, set())
    assert [row.latent_spec["shape"] for row in expressions] == ["chain", "balanced"] * 15
    graphs = make_pilot(config, "graphs", 1, set())
    assert [row.answer for row in graphs] == ["yes", "no"] * 15


def test_selected_splits_have_exact_counts_independent_seeds_and_no_forbidden_ids():
    config = load_v1_config(CONFIG)
    forbidden = {make_pilot(config, "expressions", 1, set())[0].id}
    original = set(forbidden)
    result = make_selected_splits(config, {"expressions": 1, "lists": 1, "graphs": 1}, forbidden)
    assert forbidden == original
    assert set(result) == {"dev", "sealed"}
    assert [len(result["dev"]), len(result["sealed"])] == [300, 600]
    assert all(row.split == split for split, rows in result.items() for row in rows)
    all_ids = [row.id for rows in result.values() for row in rows]
    assert not (set(all_ids) & forbidden)
    assert len(all_ids) == len(set(all_ids))
    assert all(isinstance(rows, tuple) for rows in result.values())


def test_generation_rejects_forbidden_ids_without_mutating_input():
    config = load_v1_config(CONFIG)
    row_id = make_pilot(config, "expressions", 1, set())[0].id
    forbidden = {row_id}
    generated = make_pilot(config, "expressions", 1, forbidden)
    assert row_id not in {row.id for row in generated}
    assert len(generated) == config.data.pilot_per_candidate
    assert forbidden == {row_id}


def test_generation_fails_after_one_hundred_rejected_attempts(monkeypatch):
    import ppmoe.recurrence.data_v1 as data_v1

    config = load_v1_config(CONFIG)
    original = make_pilot(config, "expressions", 1, set())[0]
    monkeypatch.setattr(data_v1, "_candidate", lambda *args: original)
    with pytest.raises(ValueError, match="after 100 attempts"):
        make_pilot(config, "expressions", 1, {original.id})


def test_freeze_writes_jsonl_and_manifest_with_independent_raw_hash(tmp_path):
    config = load_v1_config(CONFIG)
    rows = tuple(replace(row, split="dev") for row in make_pilot(config, "expressions", 1, set())[:2])
    path = tmp_path / "dev.jsonl"
    result = freeze_v1_split(
        path, rows, FakeTokenizer(), revision=config.model_revision,
        tokenizer_sha256="a" * 64,
    )
    raw = path.read_bytes()
    assert result["count"] == len(rows)
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text(encoding="utf-8"))
    assert manifest["ordered_ids"] == [row.id for row in rows]
    assert manifest["tokenizer_revision"] == config.model_revision
    assert manifest["tokenizer_sha256"] == "a" * 64
    assert manifest["max_prompt_plus_answer_tokens"] > 0


def test_freeze_accepts_pilot_split_for_calibration_pool(tmp_path):
    config = load_v1_config(CONFIG)
    rows = make_pilot(config, "expressions", 1, set())[:1]
    result = freeze_v1_split(
        tmp_path / "pilot.jsonl", rows, FakeTokenizer(), revision=config.model_revision,
        tokenizer_sha256="a" * 64,
    )
    assert result["split"] == "pilot"


@pytest.mark.parametrize("prompt_length,should_pass", [(256, True), (257, False)])
def test_freeze_enforces_v1_prompt_plus_generation_budget(tmp_path, prompt_length, should_pass):
    class LongPromptTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return list(range(prompt_length))

    config = load_v1_config(CONFIG)
    row = make_pilot(config, "expressions", 1, set())[0]
    path = tmp_path / f"prompt-{prompt_length}.jsonl"
    if should_pass:
        freeze_v1_split(
            path, [row], LongPromptTokenizer(), revision=config.model_revision,
            tokenizer_sha256="a" * 64,
        )
        assert path.exists()
    else:
        with pytest.raises(ValueError, match="prompt-plus-generation"):
            freeze_v1_split(
                path, [row], LongPromptTokenizer(), revision=config.model_revision,
                tokenizer_sha256="a" * 64,
            )
        assert not path.exists()


def test_task3_must_bind_split_raw_hash_outside_the_mutable_manifest(tmp_path):
    config = load_v1_config(CONFIG)
    rows = tuple(replace(row, split="dev") for row in make_pilot(config, "expressions", 1, set())[:1])
    path = tmp_path / "dev.jsonl"
    result = freeze_v1_split(
        path, rows, FakeTokenizer(), revision=config.model_revision, tokenizer_sha256="a" * 64,
    )
    bound_hash = result["sha256"]
    path.write_bytes(path.read_bytes() + b"tampered\n")
    manifest_path = Path(str(path) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert hashlib.sha256(path.read_bytes()).hexdigest() != bound_hash


def test_freeze_refuses_existing_files_and_symlinks(tmp_path):
    config = load_v1_config(CONFIG)
    row = replace(make_pilot(config, "expressions", 1, set())[0], split="dev")
    path = tmp_path / "dev.jsonl"
    path.write_text("owned", encoding="utf-8")
    with pytest.raises(FileExistsError):
        freeze_v1_split(path, [row], FakeTokenizer(), revision=config.model_revision, tokenizer_sha256="a" * 64)
    link = tmp_path / "link.jsonl"
    try:
        link.symlink_to(path)
    except OSError as error:
        if getattr(error, "winerror", None) != 1314:
            raise
    else:
        with pytest.raises((FileExistsError, ValueError)):
            freeze_v1_split(link, [row], FakeTokenizer(), revision=config.model_revision, tokenizer_sha256="a" * 64)


def test_freeze_rejects_wrong_split_duplicate_ids_and_inaccurate_rows(tmp_path):
    config = load_v1_config(CONFIG)
    pilot = replace(make_pilot(config, "expressions", 1, set())[0], split="train")
    with pytest.raises(ValueError, match="split"):
        freeze_v1_split(tmp_path / "bad.jsonl", [pilot], FakeTokenizer(), revision=config.model_revision, tokenizer_sha256="a" * 64)
    row = replace(pilot, split="dev")
    duplicate = tmp_path / "duplicate.jsonl"
    with pytest.raises(ValueError, match="duplicate"):
        freeze_v1_split(duplicate, [row, row], FakeTokenizer(), revision=config.model_revision, tokenizer_sha256="a" * 64)
    invalid = Example(row.id, row.family, "dev", row.structure_id, row.prompt, "wrong", row.latent_spec)
    with pytest.raises(ValueError, match="answer"):
        freeze_v1_split(tmp_path / "inaccurate.jsonl", [invalid], FakeTokenizer(), revision=config.model_revision, tokenizer_sha256="a" * 64)
