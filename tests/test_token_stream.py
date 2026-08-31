from pathlib import Path

import pytest
import torch

from ppmoe.config import DatasetSpec
from ppmoe.data.token_stream import (
    assert_disjoint_document_ids,
    iter_evaluation_batches,
    load_token_tensors,
    materialize_token_stream,
    with_stable_row_ids,
)
from ppmoe.evaluation.perplexity import EvaluationBatch


class IntegerTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [int(value) for value in text.split()]


FINEWEB = DatasetSpec(
    repo_id="HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    streaming=True,
)


def _rows() -> list[dict[str, str]]:
    return [
        {"id": "a", "text": "1 2 3", "url": "https://a.invalid"},
        {"id": "b", "text": "4 5 6 7", "url": "https://b.invalid"},
        {"id": "c", "text": "8 9 10", "url": "https://c.invalid"},
        {"id": "d", "text": "11 12 13", "url": "https://d.invalid"},
    ]


def test_materialization_stops_exactly_and_pads_only_the_final_context(
    tmp_path: Path,
) -> None:
    manifest = materialize_token_stream(
        _rows(),
        IntegerTokenizer(),
        dataset=FINEWEB,
        tokenizer_repo_id="Qwen/Qwen3.5-0.8B-Base",
        tokenizer_revision="model-sha",
        target_tokens=7,
        context_length=4,
        output_dir=tmp_path,
    )
    input_ids, attention_mask = load_token_tensors(tmp_path, manifest)

    assert input_ids.tolist() == [[1, 2, 3, 99], [4, 5, 6, 0]]
    assert attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert int(attention_mask.sum()) == 7
    assert manifest.total_tokens == 7
    assert manifest.context_count == 2
    assert manifest.final_context_length == 3
    assert manifest.documents[-1].id == "b"
    assert manifest.documents[-1].token_count == 3
    assert manifest.documents[-1].truncated is True
    assert manifest.next_row_index == 2


def test_heldout_starts_after_whole_truncated_calibration_document(
    tmp_path: Path,
) -> None:
    calibration = materialize_token_stream(
        _rows(),
        IntegerTokenizer(),
        dataset=FINEWEB,
        tokenizer_repo_id="model",
        tokenizer_revision="sha",
        target_tokens=7,
        context_length=4,
        output_dir=tmp_path / "calibration",
    )
    heldout = materialize_token_stream(
        _rows(),
        IntegerTokenizer(),
        dataset=FINEWEB,
        tokenizer_repo_id="model",
        tokenizer_revision="sha",
        target_tokens=7,
        context_length=4,
        output_dir=tmp_path / "heldout",
        start_row_index=calibration.next_row_index,
        excluded_document_ids=frozenset(record.id for record in calibration.documents),
    )

    assert [record.id for record in heldout.documents] == ["c", "d"]
    assert heldout.documents[-1].truncated is False
    assert_disjoint_document_ids(calibration, heldout)


def test_materialization_is_stable_and_hash_verified(tmp_path: Path) -> None:
    first = materialize_token_stream(
        _rows(),
        IntegerTokenizer(),
        dataset=FINEWEB,
        tokenizer_repo_id="model",
        tokenizer_revision="sha",
        target_tokens=6,
        context_length=4,
        output_dir=tmp_path / "first",
    )
    second = materialize_token_stream(
        _rows(),
        IntegerTokenizer(),
        dataset=FINEWEB,
        tokenizer_repo_id="model",
        tokenizer_revision="sha",
        target_tokens=6,
        context_length=4,
        output_dir=tmp_path / "second",
    )

    assert first.input_ids_sha256 == second.input_ids_sha256
    assert first.attention_mask_sha256 == second.attention_mask_sha256
    assert [record.text_sha256 for record in first.documents] == [
        record.text_sha256 for record in second.documents
    ]
    load_token_tensors(tmp_path / "first", first)
    (tmp_path / "first" / "input_ids.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="input_ids hash"):
        load_token_tensors(tmp_path / "first", first)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"text": "1 2"}], "id"),
        (
            [{"id": "same", "text": "1"}, {"id": "same", "text": "2"}],
            "duplicate",
        ),
        ([{"id": "empty", "text": ""}], "empty token stream"),
    ],
)
def test_materialization_rejects_untraceable_streams(
    tmp_path: Path,
    rows: list[dict[str, str]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        materialize_token_stream(
            rows,
            IntegerTokenizer(),
            dataset=FINEWEB,
            tokenizer_repo_id="model",
            tokenizer_revision="sha",
            target_tokens=3,
            context_length=2,
            output_dir=tmp_path,
        )


def test_overlap_check_rejects_shared_document_id(tmp_path: Path) -> None:
    first = materialize_token_stream(
        _rows(),
        IntegerTokenizer(),
        dataset=FINEWEB,
        tokenizer_repo_id="model",
        tokenizer_revision="sha",
        target_tokens=3,
        context_length=4,
        output_dir=tmp_path / "first",
    )
    second = materialize_token_stream(
        _rows(),
        IntegerTokenizer(),
        dataset=FINEWEB,
        tokenizer_repo_id="model",
        tokenizer_revision="sha",
        target_tokens=3,
        context_length=4,
        output_dir=tmp_path / "second",
    )

    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_document_ids(first, second)


def test_complete_stream_uses_synthetic_ids_and_preserves_final_partial_context(
    tmp_path: Path,
) -> None:
    rows = [{"text": "1 2"}, {"text": ""}, {"text": "3"}, {"text": "4 5"}]
    stable_rows = list(with_stable_row_ids(rows, "wikitext-test"))
    manifest = materialize_token_stream(
        stable_rows,
        IntegerTokenizer(),
        dataset=FINEWEB,
        tokenizer_repo_id="model",
        tokenizer_revision="sha",
        target_tokens=None,
        context_length=4,
        output_dir=tmp_path,
    )
    input_ids, attention_mask = load_token_tensors(tmp_path, manifest)

    assert [row["id"] for row in stable_rows] == [
        "wikitext-test:0",
        "wikitext-test:1",
        "wikitext-test:2",
        "wikitext-test:3",
    ]
    assert manifest.target_tokens == manifest.total_tokens == 7
    assert manifest.final_context_length == 3
    assert input_ids.tolist() == [[1, 2, 99, 3], [99, 4, 5, 0]]
    assert attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert [record.id for record in manifest.documents] == [
        "wikitext-test:0",
        "wikitext-test:2",
        "wikitext-test:3",
    ]


def test_evaluation_batches_are_deterministic_and_reject_invalid_microbatches() -> None:
    input_ids = torch.tensor([[1, 2], [3, 4], [5, 0]])
    attention_mask = torch.tensor([[1, 1], [1, 1], [1, 0]])

    batches = tuple(iter_evaluation_batches(input_ids, attention_mask, microbatch_size=2))

    assert [(batch.index, batch.input_ids.tolist()) for batch in batches] == [
        (0, [[1, 2], [3, 4]]),
        (1, [[5, 0]]),
    ]
    assert all(isinstance(batch, EvaluationBatch) for batch in batches)
    with pytest.raises(ValueError, match="microbatch"):
        tuple(iter_evaluation_batches(input_ids, attention_mask, microbatch_size=0))
    with pytest.raises(ValueError, match="same shape"):
        tuple(iter_evaluation_batches(input_ids, attention_mask[:2], microbatch_size=1))
