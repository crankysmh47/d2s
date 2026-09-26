from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import random

import pytest

from ppmoe.recurrence.config_v2 import load_v2_config
from ppmoe.recurrence.data import Example, interpret
from ppmoe.recurrence.data_v1 import make_pilot as make_v1_pilot
from ppmoe.recurrence.config_v1 import load_v1_config
from ppmoe.recurrence import data_v2
from ppmoe.recurrence.data_v2 import (
    freeze_v2_split,
    make_pilot_v2,
    make_selected_splits_v2,
    make_training_view_v2,
    render_v2_list_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "recurrence-v2.toml"
V1_CONFIG = ROOT / "configs" / "recurrence-v1.toml"


class FakeTokenizer:
    eos_token_id = 2

    def __init__(self, prompt_length: int = 10) -> None:
        self.prompt_length = prompt_length

    def apply_chat_template(self, messages, **kwargs):
        return [1] * self.prompt_length

    def __call__(self, text, **kwargs):
        return {"input_ids": [5]}


def _example(family: str, latent: dict[str, object], answer: str, prompt: str) -> Example:
    from ppmoe.recurrence.provenance import sha256_payload

    return Example(
        sha256_payload({"family": family, "latent_spec": latent}),
        family,
        "pilot",
        "fixed",
        prompt,
        answer,
        latent,
    )


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


def test_six_pilots_have_registered_counts_families_and_difficulty_grid() -> None:
    config = load_v2_config(CONFIG)
    pilots = {
        (family, candidate): make_pilot_v2(config, family, candidate, set())
        for family in ("expressions", "lists", "graphs")
        for candidate in (1, 2)
    }

    assert all(len(rows) == 30 for rows in pilots.values())
    for (family, _candidate), rows in pilots.items():
        assert all(row.family == family and row.split == "pilot" for row in rows)
        assert len({row.id for row in rows}) == 30
    assert all(row.structure_id.endswith("6") for row in pilots[("expressions", 1)])
    assert all(row.structure_id.endswith("7") for row in pilots[("expressions", 2)])
    assert all(len(row.latent_spec["leaves"]) == 6 for row in pilots[("expressions", 1)])
    assert all(len(row.latent_spec["leaves"]) == 7 for row in pilots[("expressions", 2)])
    assert all(row.structure_id.startswith("n4:") for row in pilots[("lists", 1)])
    assert all(row.structure_id.startswith("n5:") for row in pilots[("lists", 2)])
    assert all(len(row.latent_spec["input"]) == 4 and len(row.latent_spec["ops"]) == 1 for row in pilots[("lists", 1)])
    assert all(len(row.latent_spec["input"]) == 5 and len(row.latent_spec["ops"]) == 2 for row in pilots[("lists", 2)])
    assert all(row.structure_id.startswith("n7:") for row in pilots[("graphs", 1)])
    assert all(row.structure_id.startswith("n8:") for row in pilots[("graphs", 2)])
    assert all(row.latent_spec["n"] == 7 for row in pilots[("graphs", 1)])
    assert all(row.latent_spec["n"] == 8 for row in pilots[("graphs", 2)])
    assert all(row.prompt.startswith("Initial list:") for row in pilots[("lists", 1)])


def test_pilots_are_deterministic_use_exact_seed_schedule_and_isolate_global_prng(monkeypatch) -> None:
    config = load_v2_config(CONFIG)
    real_random = random.Random
    used_seeds: list[int] = []

    def recording_random(seed: int):
        used_seeds.append(seed)
        return real_random(seed)

    state = random.getstate()
    monkeypatch.setattr(data_v2.random, "Random", recording_random)
    observed = []
    for family, seeds in (
        ("expressions", (721001, 721002)),
        ("lists", (722001, 722002)),
        ("graphs", (723001, 723002)),
    ):
        for candidate, seed in enumerate(seeds, start=1):
            first = make_pilot_v2(config, family, candidate, set())
            second = make_pilot_v2(config, family, candidate, set())
            assert first == second
            observed.extend(first)
            assert used_seeds[-2:] == [seed, seed]

    make_selected_splits_v2(
        config, {"expressions": 1, "lists": 1, "graphs": 1}, set()
    )
    assert used_seeds[-6:] == [731001, 731002, 731003, 741001, 741002, 741003]

    assert len(observed) == 180
    assert random.getstate() == state


def test_pilot_generation_alternates_expression_shapes_and_balances_graph_labels() -> None:
    config = load_v2_config(CONFIG)
    expressions = make_pilot_v2(config, "expressions", 1, set())
    graphs = make_pilot_v2(config, "graphs", 1, set())

    assert [row.latent_spec["shape"] for row in expressions] == ["chain", "balanced"] * 15
    assert [row.answer for row in graphs] == ["yes", "no"] * 15


def test_candidate_rejects_identity_and_adjacent_reverse_list_programs() -> None:
    config = load_v2_config(CONFIG)

    class FixedRandom:
        def __init__(self, values, operations):
            self.values = iter(values)
            self.operations = iter(operations)

        def randrange(self, stop):
            return next(self.values)

        def choice(self, choices):
            return next(self.operations)

    identity = data_v2._candidate_v2(
        config, "lists", 1, "pilot", 0, FixedRandom([3, 3, 3, 3], ["REVERSE"])
    )
    adjacent_reverse = data_v2._candidate_v2(
        config, "lists", 2, "pilot", 0, FixedRandom([3, 3, 3, 3, 3], ["REVERSE", "REVERSE"])
    )

    assert identity is None
    assert adjacent_reverse is None


def test_selected_splits_have_exact_family_order_counts_and_no_overlap() -> None:
    config = load_v2_config(CONFIG)
    forbidden = {
        row.id
        for family in ("expressions", "lists", "graphs")
        for row in make_pilot_v2(config, family, 1, set())[:1]
    }
    original_forbidden = set(forbidden)
    selected = make_selected_splits_v2(
        config,
        {"expressions": 1, "lists": 2, "graphs": 1},
        forbidden,
    )

    assert set(selected) == {"dev", "sealed"}
    assert [len(selected[name]) for name in ("dev", "sealed")] == [300, 600]
    assert [row.family for row in selected["dev"]] == [
        "expressions"
    ] * 100 + ["lists"] * 100 + ["graphs"] * 100
    assert [row.family for row in selected["sealed"]] == [
        "expressions"
    ] * 200 + ["lists"] * 200 + ["graphs"] * 200
    assert all(row.split == split for split, rows in selected.items() for row in rows)
    ids = [row.id for rows in selected.values() for row in rows]
    assert len(ids) == len(set(ids))
    assert not (set(ids) & forbidden)
    assert forbidden == original_forbidden
    assert all(row.structure_id.startswith("n5:") for row in selected["dev"] if row.family == "lists")


@pytest.mark.parametrize(
    "choices",
    [
        {"expressions": 1, "lists": 1},
        {"expressions": 1, "lists": 1, "graphs": 1, "other": 1},
        {"expressions": 1, "lists": 1, "graphs": 3},
        {"expressions": True, "lists": 1, "graphs": 1},
    ],
)
def test_selected_splits_reject_incomplete_or_invalid_choices(choices) -> None:
    with pytest.raises(ValueError, match="choices|candidate"):
        make_selected_splits_v2(load_v2_config(CONFIG), choices, set())


def test_generation_rejects_forbidden_ids_without_mutating_them() -> None:
    config = load_v2_config(CONFIG)
    excluded = make_pilot_v2(config, "expressions", 1, set())[0].id
    forbidden = {excluded}
    rows = make_pilot_v2(config, "expressions", 1, forbidden)

    assert len(rows) == 30
    assert excluded not in {row.id for row in rows}
    assert forbidden == {excluded}


def test_generation_fails_after_one_hundred_duplicate_attempts(monkeypatch) -> None:
    config = load_v2_config(CONFIG)
    duplicate = make_pilot_v2(config, "expressions", 1, set())[0]
    monkeypatch.setattr(data_v2, "_candidate_v2", lambda *args: duplicate)

    with pytest.raises(ValueError, match="after 100 attempts"):
        make_pilot_v2(config, "expressions", 1, {duplicate.id})


def test_v2_split_freeze_accepts_hand_computed_list_and_graph_answers(tmp_path) -> None:
    config = load_v2_config(CONFIG)
    list_row = _example(
        "lists",
        {"input": [3, 8], "ops": ["ADD1"]},
        "[4,9]",
        render_v2_list_prompt({"input": [3, 8], "ops": ["ADD1"]}),
    )
    graph_row = _example(
        "graphs",
        {"n": 2, "edges": ["A->B"], "start": "A", "end": "B"},
        "yes",
        "Directed edges: A->B. Is there a directed path from A to B? Return only yes or no.",
    )
    path = tmp_path / "pilot.jsonl"

    manifest = freeze_v2_split(
        path,
        [list_row, graph_row],
        FakeTokenizer(),
        revision=config.model_revision,
        tokenizer_sha256="a" * 64,
    )

    assert manifest["ordered_ids"] == [list_row.id, graph_row.id]
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["answer"] == "[4,9]"
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[1])["answer"] == "yes"


def test_freeze_manifest_binds_order_count_and_raw_jsonl_hash(tmp_path) -> None:
    config = load_v2_config(CONFIG)
    rows = make_pilot_v2(config, "lists", 1, set())[:2]
    path = tmp_path / "ordered.jsonl"

    manifest = freeze_v2_split(
        path, tuple(reversed(rows)), FakeTokenizer(), revision=config.model_revision, tokenizer_sha256="b" * 64
    )

    sidecar = json.loads(Path(str(path) + ".manifest.json").read_text(encoding="utf-8"))
    assert manifest == sidecar
    assert sidecar["ordered_ids"] == [row.id for row in reversed(rows)]
    assert sidecar["count"] == 2
    assert sidecar["sha256"] == sha256(path.read_bytes()).hexdigest()
    assert sidecar["tokenizer_sha256"] == "b" * 64


@pytest.mark.parametrize(
    ("revision", "tokenizer_sha256", "message"),
    [("main", "a" * 64, "revision"), ("70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "bad", "SHA-256")],
)
def test_freeze_requires_immutable_tokenizer_provenance(tmp_path, revision, tokenizer_sha256, message) -> None:
    row = make_pilot_v2(load_v2_config(CONFIG), "expressions", 1, set())[0]
    path = tmp_path / "bad-provenance.jsonl"

    with pytest.raises(ValueError, match=message):
        freeze_v2_split(path, [row], FakeTokenizer(), revision=revision, tokenizer_sha256=tokenizer_sha256)
    assert not path.exists()
    assert not Path(str(path) + ".manifest.json").exists()


def test_freeze_enforces_256_new_token_prompt_budget_before_writing(tmp_path) -> None:
    row = make_pilot_v2(load_v2_config(CONFIG), "expressions", 1, set())[0]
    path = tmp_path / "too-long.jsonl"

    with pytest.raises(ValueError, match="prompt-plus-generation"):
        freeze_v2_split(
            path,
            [row],
            FakeTokenizer(prompt_length=257),
            revision=load_v2_config(CONFIG).model_revision,
            tokenizer_sha256="a" * 64,
        )
    assert not path.exists()


def test_freeze_rejects_altered_answer_or_non_v2_list_prompt(tmp_path) -> None:
    config = load_v2_config(CONFIG)
    row = make_pilot_v2(config, "lists", 1, set())[0]
    wrong_answer = replace(row, answer="[9,9,9,9]")
    wrong_prompt = replace(row, prompt="Start with [1,2,3,4]. Apply ADD1.")

    for bad_row in (wrong_answer, wrong_prompt):
        path = tmp_path / f"bad-{bad_row.answer == row.answer}.jsonl"
        with pytest.raises(ValueError, match="answer|V2 list prompt"):
            freeze_v2_split(
                path, [bad_row], FakeTokenizer(), revision=config.model_revision, tokenizer_sha256="a" * 64
            )
        assert not path.exists()


def test_freeze_refuses_existing_data_or_manifest_and_preserves_owned_bytes(tmp_path) -> None:
    config = load_v2_config(CONFIG)
    row = make_pilot_v2(config, "expressions", 1, set())[0]
    path = tmp_path / "existing.jsonl"
    path.write_bytes(b"owned data")

    with pytest.raises(FileExistsError):
        freeze_v2_split(path, [row], FakeTokenizer(), revision=config.model_revision, tokenizer_sha256="a" * 64)
    assert path.read_bytes() == b"owned data"

    path.unlink()
    sidecar = Path(str(path) + ".manifest.json")
    sidecar.write_bytes(b"owned manifest")
    with pytest.raises(FileExistsError):
        freeze_v2_split(path, [row], FakeTokenizer(), revision=config.model_revision, tokenizer_sha256="a" * 64)
    assert not path.exists()
    assert sidecar.read_bytes() == b"owned manifest"


def test_freeze_cleans_only_its_data_file_when_manifest_already_exists(tmp_path) -> None:
    config = load_v2_config(CONFIG)
    row = make_pilot_v2(config, "expressions", 1, set())[0]
    path = tmp_path / "pilot.jsonl"
    sidecar = Path(str(path) + ".manifest.json")
    sidecar.write_bytes(b"preexisting")

    with pytest.raises(FileExistsError):
        freeze_v2_split(path, [row], FakeTokenizer(), revision=config.model_revision, tokenizer_sha256="a" * 64)

    assert not path.exists()
    assert sidecar.read_bytes() == b"preexisting"


def test_freeze_cleans_files_created_by_failed_manifest_fsync(tmp_path, monkeypatch) -> None:
    config = load_v2_config(CONFIG)
    row = make_pilot_v2(config, "expressions", 1, set())[0]
    path = tmp_path / "pilot.jsonl"
    real_fsync = data_v2.os.fsync
    calls = 0

    def fail_manifest_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected V2 manifest fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(data_v2.os, "fsync", fail_manifest_fsync)
    with pytest.raises(OSError, match="injected V2 manifest"):
        freeze_v2_split(path, [row], FakeTokenizer(), revision=config.model_revision, tokenizer_sha256="a" * 64)

    assert calls == 2
    assert not path.exists()
    assert not Path(str(path) + ".manifest.json").exists()


def test_freeze_rejects_symlinked_output_ancestor(tmp_path) -> None:
    config = load_v2_config(CONFIG)
    row = make_pilot_v2(config, "expressions", 1, set())[0]
    target = tmp_path / "real-directory"
    target.mkdir()
    linked = tmp_path / "linked-directory"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("creating symlinks requires an unavailable Windows privilege")
        raise

    path = linked / "pilot.jsonl"
    with pytest.raises(ValueError, match="symlink or reparse point"):
        freeze_v2_split(path, [row], FakeTokenizer(), revision=config.model_revision, tokenizer_sha256="a" * 64)
    assert not (target / "pilot.jsonl").exists()
