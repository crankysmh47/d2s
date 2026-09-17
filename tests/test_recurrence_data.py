"""Contracts for the deterministic exact-answer recurrence corpus."""

from collections import Counter
from copy import copy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from ppmoe.recurrence.config import RecurrenceConfig
import ppmoe.recurrence.data as recurrence_data
from ppmoe.recurrence.data import (
    Example,
    freeze_examples,
    interpret,
    make_examples,
    parse_answer,
    render_example,
)


def registered_config() -> RecurrenceConfig:
    return RecurrenceConfig.load(Path("configs/recurrence-v0.toml"))


def example_for(family: str, answer: str) -> Example:
    latent_specs = {
        "expressions": {"shape": "chain", "leaves": [1, 2], "operators": ["+"]},
        "lists": {"input": [1, 2, 3], "ops": ["ROTL1"]},
        "graphs": {"n": 2, "edges": [], "start": "A", "end": "B"},
    }
    return Example("fixture", family, "dev", "fixture", "question", answer, latent_specs[family])


def copied_generated(examples: object) -> object:
    """Copy test-owned row lists while retaining generation-owned evidence attributes."""
    clone = copy(examples)
    for split, rows in examples.items():  # type: ignore[union-attr]
        clone[split] = list(rows)
    return clone


def test_registered_generation_is_deterministic_and_globally_disjoint() -> None:
    """Would fail if a seed, count, or global duplicate guard changed."""
    first = make_examples(registered_config())
    second = make_examples(registered_config())

    assert first == second
    assert Counter(x.family for x in first["train"]) == {
        "expressions": 2667,
        "lists": 2667,
        "graphs": 2666,
    }
    assert Counter(x.family for x in first["dev"]) == {
        "expressions": 200,
        "lists": 200,
        "graphs": 200,
    }
    assert Counter(x.family for x in first["test"]) == {
        "expressions": 800,
        "lists": 800,
        "graphs": 800,
    }
    ids = [example.id for rows in first.values() for example in rows]
    assert len(ids) == len(set(ids))


def test_generated_examples_and_accounting_are_recursively_immutable() -> None:
    """Would fail if callers could mutate a latent payload or accounting after its ID was fixed."""
    examples = make_examples(registered_config())
    expression = next(row for row in examples["train"] if row.family == "expressions")
    with pytest.raises(TypeError):
        expression.latent_spec["shape"] = "balanced"  # type: ignore[index]
    with pytest.raises(TypeError):
        expression.latent_spec["leaves"][0] = 9  # type: ignore[index]
    with pytest.raises(TypeError):
        examples.rejections["train"]["expressions"] = 0  # type: ignore[attr-defined, index]


def test_registered_splits_shift_structure_without_breaking_family_rules() -> None:
    """Would fail if held-out instances reused training structures or invalid programs."""
    examples = make_examples(registered_config())
    expected = {"train": {3, 4}, "dev": {5}, "test": {6}}
    for split, leaves in expected.items():
        expression_rows = [x for x in examples[split] if x.family == "expressions"]
        assert {len(x.latent_spec["leaves"]) for x in expression_rows} == leaves
        assert [x.latent_spec["shape"] for x in expression_rows[:8]] == [
            "chain", "balanced", "chain", "balanced", "chain", "balanced", "chain", "balanced"
        ]

    list_rules = {"train": ({3, 4, 5}, {1, 2}), "dev": ({6}, {3}), "test": ({7}, {4})}
    graph_rules = {"train": {5, 6}, "dev": {7}, "test": {8}}
    for split, (lengths, program_lengths) in list_rules.items():
        list_rows = [x for x in examples[split] if x.family == "lists"]
        assert {len(x.latent_spec["input"]) for x in list_rows} == lengths
        assert {len(x.latent_spec["ops"]) for x in list_rows} == program_lengths
        for row in list_rows:
            ops = row.latent_spec["ops"]
            assert all(pair != ("REVERSE", "REVERSE") for pair in zip(ops, ops[1:]))
            assert interpret(row.latent_spec, "lists") != list(row.latent_spec["input"])

        graph_rows = [x for x in examples[split] if x.family == "graphs"]
        assert {x.latent_spec["n"] for x in graph_rows} == graph_rules[split]
        labels = Counter(x.answer for x in graph_rows)
        assert abs(labels["yes"] - labels["no"]) <= 1


def test_interpreters_recompute_the_canonical_answers_without_eval() -> None:
    """Would fail if the safe evaluators drifted from stored exact answers."""
    examples = make_examples(registered_config())
    for rows in examples.values():
        for row in rows:
            expected = row.answer if row.family == "graphs" else json.loads(row.answer)
            assert interpret(row.latent_spec, row.family) == expected


def test_expression_parser_requires_whole_string() -> None:
    """Would fail if integer parsing extracted an answer from prose or accepted noncanonical form."""
    example = example_for("expressions", "12")
    assert parse_answer(" 12\n", example) == 12
    for bad in ["+12", "012", "The answer is 12", "12.", "12 13"]:
        with pytest.raises(ValueError):
            parse_answer(bad, example)


def test_list_parser_rejects_bool_and_wrong_length() -> None:
    """Would fail if JSON booleans or malformed/full-text list answers were accepted."""
    example = example_for("lists", "[1,2,3]")
    for bad in ["[1, true, 3]", "[1,2]", "[1, 11, 3]", "result=[1,2,3]"]:
        with pytest.raises(ValueError):
            parse_answer(bad, example)


def test_graph_parser_is_case_insensitive_but_not_extractive() -> None:
    """Would fail if graph answers used substring extraction rather than a whole-string parser."""
    example = example_for("graphs", "yes")
    assert parse_answer(" YES ", example) == "yes"
    with pytest.raises(ValueError):
        parse_answer("the answer is yes", example)


def test_generation_attempt_cap_never_makes_a_cap_plus_one_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Would fail if an exhausted family/split made an unregistered extra sampling attempt."""
    calls = 0

    def always_reject(*args: object) -> None:
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(recurrence_data, "_candidate", always_reject)
    with pytest.raises(RuntimeError, match=r"expressions/dev exceeded 200 generation attempts"):
        recurrence_data._generate_family("expressions", "dev", 2, 1, set())
    assert calls == 200


class FakeTokenizer:
    eos_token_id = 99
    revision = "fake-tokenizer-revision"

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> list[int]:
        assert messages == [{"role": "user", "content": "question"}]
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
        return [10, 11]

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        assert kwargs == {"add_special_tokens": False}
        return {"input_ids": [ord(character) for character in text]}


class AuditTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> list[int]:
        assert len(messages) == 1 and messages[0]["role"] == "user"
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
        return [10, 11]


def test_rendering_uses_a_single_user_turn_and_shifted_answer_only_labels() -> None:
    """Would fail if training leaked prompt tokens, omitted EOS, or used a different chat setting."""
    example = example_for("expressions", "12")
    rendered = render_example(FakeTokenizer(), example)

    assert rendered.prompt_input_ids == (10, 11)
    assert rendered.train_input_ids == (10, 11, 49, 50, 99)
    assert rendered.labels == (-100, -100, 49, 50, 99)
    assert len(rendered.train_input_ids) <= 512
    assert len(rendered.train_input_ids) - len(rendered.prompt_input_ids) <= 32


def test_rendering_rejects_any_length_violation_without_truncation() -> None:
    """Would fail if overlong prompts or answers were silently truncated."""
    class LongTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> list[int]:
            return [1] * 481

    with pytest.raises(ValueError, match="prompt"):
        render_example(LongTokenizer(), example_for("expressions", "12"))


def test_freeze_is_deterministic_and_refuses_different_existing_evidence(tmp_path: Path) -> None:
    """Would fail if frozen JSONL order, hashes, or immutable replay protection changed."""
    examples = make_examples(registered_config())
    first = freeze_examples(
        tmp_path,
        examples,
        tokenizer=AuditTokenizer(),
        tokenizer_revision="a" * 40,
        tokenizer_sha256="b" * 64,
    )
    second = freeze_examples(
        tmp_path,
        examples,
        tokenizer=AuditTokenizer(),
        tokenizer_revision="a" * 40,
        tokenizer_sha256="b" * 64,
    )

    assert first == second
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"dev": 600, "test": 2400, "train": 8000}
    assert set(manifest["split_sha256"]) == {"train", "dev", "test"}
    assert set(manifest["rejections"]) == {"train", "dev", "test"}
    assert all(set(rows) == {"expressions", "lists", "graphs"} for rows in manifest["rejections"].values())
    assert sum(count for rows in manifest["rejections"].values() for count in rows.values()) > 0
    assert manifest["tokenizer"] == {"revision": "a" * 40, "sha256": "b" * 64}
    assert manifest["maximum_token_lengths"] == {"prompt": 2, "answer_plus_eos": 16, "sequence": 18}
    assert manifest["generator_sha256"] == sha256(
        json.dumps(manifest["generator"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert (tmp_path / "data" / "train.jsonl").read_bytes().endswith(b"\n")

    changed = copied_generated(examples)
    changed["dev"] = changed["dev"][:-1]
    with pytest.raises(ValueError, match="registered generation order"):
        freeze_examples(
            tmp_path,
            changed,
            tokenizer=AuditTokenizer(),
            tokenizer_revision="a" * 40,
            tokenizer_sha256="b" * 64,
        )


def test_final_freeze_requires_pinned_tokenizer_and_audits_before_writing(tmp_path: Path) -> None:
    """Would fail if a final data artifact lacked source provenance or wrote partial rows before audit."""
    examples = make_examples(registered_config())
    with pytest.raises(ValueError, match="tokenizer revision"):
        freeze_examples(tmp_path, examples)
    assert not (tmp_path / "data").exists()

    class OverlongTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> list[int]:
            return [1] * 481

    with pytest.raises(ValueError, match="prompt"):
        freeze_examples(
            tmp_path,
            examples,
            tokenizer=OverlongTokenizer(),
            tokenizer_revision="a" * 40,
            tokenizer_sha256="b" * 64,
        )
    assert not (tmp_path / "data").exists()


def test_final_freeze_revalidates_ids_and_registered_order_before_writing(tmp_path: Path) -> None:
    """Would fail if caller mutation could produce a differently ordered or mismatched identity artifact."""
    examples = make_examples(registered_config())
    reordered = copied_generated(examples)
    reordered["dev"].reverse()
    with pytest.raises(ValueError, match="registered generation order"):
        freeze_examples(
            tmp_path,
            reordered,
            tokenizer=AuditTokenizer(),
            tokenizer_revision="a" * 40,
            tokenizer_sha256="b" * 64,
        )
    assert not (tmp_path / "data").exists()

    tampered = copied_generated(examples)
    tampered["train"][0] = replace(tampered["train"][0], id="0" * 64)
    with pytest.raises(ValueError, match="canonical ID"):
        freeze_examples(
            tmp_path,
            tampered,
            tokenizer=AuditTokenizer(),
            tokenizer_revision="a" * 40,
            tokenizer_sha256="b" * 64,
        )
    assert not (tmp_path / "data").exists()
