from __future__ import annotations

from pathlib import Path

import pytest

from ppmoe.recurrence.data import Example
from ppmoe.recurrence.v1_runtime import score_v1_examples


class TinyTokenizer:
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [4]}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return [1, 3]

    def decode(self, token_ids, skip_special_tokens=True):
        return "Final answer: \\boxed{3}"


class TinyModel:
    def __call__(self, prefix, mode):
        raise AssertionError("generation is replaced by the CPU fake")


def test_score_v1_examples_writes_exact_rows_and_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ppmoe.recurrence.v1_runtime as runtime

    monkeypatch.setattr(runtime, "generate_answer", lambda *args, **kwargs: {
        "token_ids": [4, 2], "completion_token_ids": [4], "stop_reason": "eos",
        "generated_token_count": 2,
    })
    example = Example("id-1", "expressions", "pilot", "chain6", "Compute 1+2.", "3", {
        "shape": "chain", "leaves": [1, 2], "operators": ["+"],
    })
    output = tmp_path / "scores.jsonl"
    result = score_v1_examples(TinyModel(), TinyTokenizer(), [example], "single", output, {})
    assert result["validated_count"] == 1
    assert len(result["raw_sha256"]) == 64
    assert result["summary"]["correct"] == 1
    assert output.read_text(encoding="utf-8").count("id-1") == 1


def test_score_v1_examples_rejects_duplicate_ids(tmp_path: Path) -> None:
    row = Example("same", "expressions", "pilot", "x", "p", "1", {"shape": "chain", "leaves": [1], "operators": []})
    with pytest.raises(ValueError, match="duplicate"):
        score_v1_examples(TinyModel(), TinyTokenizer(), [row, row], "single", tmp_path / "raw.jsonl", {})


def test_score_v1_examples_preserves_partial_rows_after_generation_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import ppmoe.recurrence.v1_runtime as runtime

    first = Example("first", "expressions", "pilot", "x", "p", "3", {"shape": "chain", "leaves": [1, 2], "operators": ["+"]})
    second = Example("second", "expressions", "pilot", "x", "p", "3", {"shape": "chain", "leaves": [1, 2], "operators": ["+"]})
    calls = 0

    def generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic model failure")
        return {"token_ids": [4, 2], "completion_token_ids": [4], "stop_reason": "eos", "generated_token_count": 2}

    monkeypatch.setattr(runtime, "generate_answer", generate)
    path = tmp_path / "partial.jsonl"
    with pytest.raises(runtime.ScoreExecutionError) as caught:
        score_v1_examples(TinyModel(), TinyTokenizer(), [first, second], "single", path, {})
    assert caught.value.row_count == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["example_id"] == "first"
    assert "raw_sha256" in caught.value.__dict__


def test_ambiguous_answer_is_normalized_to_gate_invalid() -> None:
    from ppmoe.recurrence.v1_runtime import _normalize_score

    result = _normalize_score({"correct": False, "reason_code": "ambiguous", "failure_detail": None})
    assert result["reason_code"] == "invalid_format"
    assert result["failure_detail"] == "ambiguous final answer"


def test_deadline_failure_is_preserved_and_marked_resource_limited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ppmoe.recurrence.v1_runtime as runtime

    row = Example("deadline", "expressions", "pilot", "x", "p", "3", {"shape": "chain", "leaves": [1, 2], "operators": ["+"]})

    def expired(*args, **kwargs):
        kwargs["check_deadline"]()

    monkeypatch.setattr(runtime, "generate_answer", expired)
    with pytest.raises(runtime.ScoreExecutionError) as caught:
        score_v1_examples(TinyModel(), TinyTokenizer(), [row], "single", tmp_path / "timeout.jsonl", {}, deadline=0.0)
    assert caught.value.deadline_exceeded is True
    assert caught.value.row_count == 0
    assert len(caught.value.raw_sha256) == 64
