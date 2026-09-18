from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from ppmoe.recurrence.data import Example, RenderedExample, RetentionChunk
from ppmoe.recurrence.evaluate import (
    evaluate_examples,
    evaluate_retention,
    generate_answer,
)


EOS = 2


class ScriptedModel:
    def __init__(self, tokens: list[int], vocab: int = 16) -> None:
        self.tokens = tokens
        self.vocab = vocab
        self.calls: list[tuple[list[int], str, dict[str, object]]] = []
        self.tokenizer = _Tokenizer()
        self.config = type("Config", (), {"eos_token_id": EOS})()

    def __call__(self, input_ids: torch.Tensor, mode: str, **kwargs: object) -> torch.Tensor:
        self.calls.append((input_ids[0].tolist(), mode, kwargs))
        index = len(self.calls) - 1
        logits = torch.full((1, input_ids.shape[1], self.vocab), -100.0)
        logits[0, -1, self.tokens[min(index, len(self.tokens) - 1)]] = 100.0
        return logits


class _Tokenizer:
    def decode(self, tokens: list[int], **_: object) -> str:
        return "".join({7: "7", 8: "8", 3: "3"}.get(token, "") for token in tokens)


def _example(identifier: str = "e-1", answer: str = "7") -> RenderedExample:
    row = Example(
        id=identifier,
        family="expressions",
        split="test",
        structure_id="s-1",
        prompt="what is 3+4?",
        answer=answer,
        latent_spec={"leaves": [3, 4], "operators": ["+"] , "shape": "chain"},
    )
    return RenderedExample(row, (11, 12), (11, 12, 7, EOS), (-100, -100, 7, EOS))


def test_generation_recomputes_full_prefix() -> None:
    model = ScriptedModel([7, 8, EOS])
    result = generate_answer(model, torch.tensor([[1, 2]]), "repeat", EOS, 32)
    assert [call[0] for call in model.calls] == [[1, 2], [1, 2, 7], [1, 2, 7, 8]]
    assert result["token_ids"] == [7, 8, EOS]
    assert result["completion_token_ids"] == [7, 8]
    assert result["stop_reason"] == "eos"
    assert all(not kwargs for _, _, kwargs in model.calls)


def test_generation_length_cap() -> None:
    model = ScriptedModel([7] * 4)
    result = generate_answer(model, torch.tensor([[1]]), "single", EOS, 4)
    assert result["token_ids"] == [7, 7, 7, 7]
    assert result["stop_reason"] == "length"


def test_example_evaluation_persists_record_and_summary(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    result = evaluate_examples(ScriptedModel([7, EOS]), [_example()], "single", path)
    assert result["complete"] is True
    assert result["correct"] == 1
    assert len(result["records_sha256"]) == 64
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["example_id"] == "e-1"
    assert record["family"] == "expressions"
    assert record["reason_code"] == "correct"
    assert (tmp_path / "scores.summary.json").is_file()


@pytest.mark.parametrize("tokens, reason", [([EOS], "empty_completion"), ([3, EOS], "wrong_value")])
def test_example_reason_codes(tmp_path: Path, tokens: list[int], reason: str) -> None:
    result = evaluate_examples(ScriptedModel(tokens), [_example()], "single", tmp_path / "scores.jsonl")
    assert result["complete"] is True
    row = json.loads((tmp_path / "scores.jsonl").read_text(encoding="utf-8"))
    assert row["reason_code"] == reason
    assert row["correct"] is False


def test_retention_excludes_each_chunk_first_token(tmp_path: Path) -> None:
    class RetentionModel:
        def __call__(self, input_ids: torch.Tensor, mode: str) -> torch.Tensor:
            logits = torch.zeros((1, input_ids.shape[1], 3), dtype=torch.float64)
            logits[..., 0] = 0.0
            logits[..., 1] = 1.0
            logits[..., 2] = 2.0
            return logits

    chunks = [
        RetentionChunk(0, (0, 1, 2), ("d0",), 2),
        RetentionChunk(1, (2, 1, 0), ("d1",), 2),
    ]
    result = evaluate_retention(RetentionModel(), chunks, "single", tmp_path / "retention.jsonl")
    assert result["complete"] is True
    assert result["scored_tokens"] == 4
    assert len(result["records_sha256"]) == 64
    reference_targets = torch.tensor([1, 2, 1, 0])
    reference_logits = torch.tensor([[0.0, 1.0, 2.0]] * 4)
    reference_nll = -torch.log_softmax(reference_logits, dim=-1).gather(
        1, reference_targets.unsqueeze(1)
    ).sum()
    assert result["nll_sum"] == pytest.approx(reference_nll.item(), rel=1e-6)
    assert result["mean_nll"] == pytest.approx(result["nll_sum"] / 4)
    assert result["perplexity"] == pytest.approx(torch.exp(torch.tensor(result["mean_nll"])).item())


def test_duplicate_or_out_of_order_records_are_incomplete(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    row = {"example_id": "e-1"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    result = evaluate_examples(ScriptedModel([7, EOS]), [_example()], "single", path)
    assert result["complete"] is False
    assert "duplicate" in str(result["reconciliation_error"])


def _metadata() -> dict[str, object]:
    return {"arm": "a1", "seed": 47, "signatures": {"run": "sig-v1"}}


def test_explicit_metadata_conflict_is_rejected_before_writing(tmp_path: Path) -> None:
    model = ScriptedModel([7, EOS])
    model.arm = "stale-arm"
    with pytest.raises(ValueError, match="arm"):
        evaluate_examples(model, [_example()], "single", tmp_path / "scores.jsonl", **_metadata())
    assert not (tmp_path / "scores.jsonl").exists()


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("arm", "other", "arm"),
        ("family", "lists", "family"),
        ("prompt_hash", "tampered", "prompt"),
        ("correct", False, "correct"),
        ("reason_code", "made_up", "reason"),
    ],
)
def test_tampered_example_record_is_incomplete(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    path = tmp_path / "scores.jsonl"
    first = evaluate_examples(ScriptedModel([7, EOS]), [_example()], "single", path, **_metadata())
    assert first["complete"] is True
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0][field] = value
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    (tmp_path / "scores.summary.json").unlink()
    result = evaluate_examples(ScriptedModel([7, EOS]), [_example()], "single", path, **_metadata())
    assert result["complete"] is False
    assert message in str(result["reconciliation_error"])
    assert not (tmp_path / "scores.summary.json").exists()


def test_valid_partial_prefix_resume(tmp_path: Path) -> None:
    first_path = tmp_path / "scores.jsonl"
    first = evaluate_examples(ScriptedModel([7, EOS]), [_example("e-1")], "single", first_path, **_metadata())
    assert first["complete"] is True
    # Simulate a crash after the durable raw prefix but before its provisional
    # one-row aggregate can be considered the final expected-set summary.
    (tmp_path / "scores.summary.json").unlink()
    rows = [_example("e-1"), _example("e-2")]
    resumed = evaluate_examples(ScriptedModel([7, EOS]), rows, "single", first_path, **_metadata())
    assert resumed["complete"] is True
    assert resumed["record_count"] == 2
    assert [json.loads(line)["example_id"] for line in first_path.read_text(encoding="utf-8").splitlines()] == ["e-1", "e-2"]


def test_empty_raw_tokens_cannot_have_completion_tokens(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    first = evaluate_examples(ScriptedModel([7, EOS]), [_example()], "single", path, **_metadata())
    assert first["complete"] is True
    row = json.loads(path.read_text(encoding="utf-8"))
    row["token_ids"] = []
    row["completion_token_ids"] = [7]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (tmp_path / "scores.summary.json").unlink()
    result = evaluate_examples(ScriptedModel([7, EOS]), [_example()], "single", path, **_metadata())
    assert result["complete"] is False
    assert "token IDs" in str(result["reconciliation_error"])


def test_missing_eos_at_cap_cannot_contain_interior_eos(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    first = evaluate_examples(
        ScriptedModel([7, 7, 7]), [_example()], "single", path, max_new_tokens=3, **_metadata()
    )
    assert first["complete"] is True
    row = json.loads(path.read_text(encoding="utf-8"))
    row["token_ids"] = [7, EOS, 7]
    row["completion_token_ids"] = [7, EOS, 7]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (tmp_path / "scores.summary.json").unlink()
    result = evaluate_examples(
        ScriptedModel([7, 7, 7]), [_example()], "single", path, max_new_tokens=3, **_metadata()
    )
    assert result["complete"] is False


def _retention_model() -> object:
    class RetentionModel:
        def __call__(self, input_ids: torch.Tensor, mode: str) -> torch.Tensor:
            logits = torch.zeros((1, input_ids.shape[1], 3), dtype=torch.float64)
            logits[..., 0] = 0.0
            logits[..., 1] = 1.0
            logits[..., 2] = 2.0
            return logits

    return RetentionModel()


def _chunks() -> list[RetentionChunk]:
    return [
        RetentionChunk(0, (0, 1, 2), ("d0",), 2),
        RetentionChunk(1, (2, 1, 0), ("d1",), 2),
    ]


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("arm", "other", "arm"),
        ("token_count", 99, "token_count"),
        ("scored_tokens", 99, "scored_tokens"),
        ("nll_sum", -1.0, "nll"),
        ("source_document_ids", ["wrong"], "source"),
    ],
)
def test_tampered_retention_record_is_incomplete(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    path = tmp_path / "retention.jsonl"
    first = evaluate_retention(_retention_model(), _chunks(), "single", path, **_metadata())
    assert first["complete"] is True
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0][field] = value
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (tmp_path / "retention.summary.json").unlink()
    result = evaluate_retention(_retention_model(), _chunks(), "single", path, **_metadata())
    assert result["complete"] is False
    assert message.casefold() in str(result["reconciliation_error"]).casefold()
    assert not (tmp_path / "retention.summary.json").exists()


def test_negative_retention_token_is_a_failed_incomplete_record(tmp_path: Path) -> None:
    path = tmp_path / "retention.jsonl"
    chunks = [RetentionChunk(0, (0, -1, 2), ("d0",), 2)]
    result = evaluate_retention(_retention_model(), chunks, "single", path, **_metadata())
    assert result["complete"] is False
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["failure"]


def test_nonfinite_retention_record_is_incomplete(tmp_path: Path) -> None:
    path = tmp_path / "retention.jsonl"
    first = evaluate_retention(_retention_model(), _chunks(), "single", path, **_metadata())
    assert first["complete"] is True
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["mean_nll"] = math.nan
    path.write_text("\n".join(json.dumps(row, allow_nan=True) for row in rows) + "\n", encoding="utf-8")
    (tmp_path / "retention.summary.json").unlink()
    result = evaluate_retention(_retention_model(), _chunks(), "single", path, **_metadata())
    assert result["complete"] is False
