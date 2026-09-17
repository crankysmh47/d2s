"""Contracts for exact FineWeb retention recovery and re-tokenization."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

from ppmoe.recurrence.data import (
    DocumentRecord,
    RetentionSourceError,
    load_retention_chunks,
    materialize_retention,
    recover_retention_documents,
)


PINNED_REVISION = "a" * 40
REAL_MANIFEST_PATH = Path("artifacts/phase1-v0/tokens/fineweb-heldout/manifest.json")


class CharacterTokenizer:
    """Offline tokenizer whose token boundaries make stream assertions transparent."""

    eos_token_id = 0

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        assert kwargs == {"add_special_tokens": False}
        return {"input_ids": [ord(character) for character in text]}


def _old_root(tmp_path: Path, rows: list[dict[str, object]], *, revision: str = PINNED_REVISION) -> Path:
    root = tmp_path / "old"
    token_root = root / "tokens" / "fineweb-heldout"
    token_root.mkdir(parents=True)
    manifest = {
        "dataset": {
            "name": "sample-10BT",
            "repo_id": "HuggingFaceFW/fineweb-edu",
            "revision": revision,
            "split": "train",
            "streaming": True,
        },
        "ordering": "repository_order",
        "shuffle": False,
        "documents": [
            {
                "id": row["id"],
                "row_index": index,
                "text_sha256": sha256(str(row["text"]).encode("utf-8")).hexdigest(),
            }
            for index, row in enumerate(rows)
        ],
    }
    (token_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _write_cached_text(root: Path, rows: list[dict[str, object]]) -> None:
    path = root / "tokens" / "fineweb-heldout" / "raw_documents.jsonl"
    path.write_bytes(
        b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in rows
        )
    )


def test_recovery_prefers_hash_verified_cached_raw_text(tmp_path: Path) -> None:
    rows = [{"id": "one", "text": "alpha"}, {"id": "two", "text": "beta"}]
    root = _old_root(tmp_path, rows)
    _write_cached_text(root, list(reversed(rows)))

    def must_not_load(*args: object, **kwargs: object) -> list[dict[str, object]]:
        raise AssertionError(f"cached text should avoid loader: {args}, {kwargs}")

    recovered = recover_retention_documents(root, must_not_load)
    assert [record.id for record in recovered] == ["one", "two"]
    assert [record.text for record in recovered] == ["alpha", "beta"]
    assert all(record.text_sha256 == sha256(record.text.encode("utf-8")).hexdigest() for record in recovered)


@pytest.mark.parametrize(
    ("rows", "cached", "message"),
    [
        ([{"id": "one", "text": "alpha"}], [], "missing raw text"),
        ([{"id": "one", "text": "alpha"}], [{"id": "one", "text": "changed"}], "hash mismatch"),
        (
            [{"id": "one", "text": "alpha"}, {"id": "one", "text": "beta"}],
            [],
            "duplicate document id",
        ),
    ],
)
def test_recovery_rejects_missing_changed_or_duplicate_source_text(
    tmp_path: Path, rows: list[dict[str, object]], cached: list[dict[str, object]], message: str
) -> None:
    root = _old_root(tmp_path, rows)
    if cached:
        _write_cached_text(root, cached)
    with pytest.raises(RetentionSourceError, match=message):
        recover_retention_documents(root, lambda repo_id, **kwargs: [])


def test_recovery_uses_real_manifest_shape_and_production_loader_calling_convention(tmp_path: Path) -> None:
    """The historical config and production `load_dataset` adapter must stay aligned."""
    actual_manifest = json.loads(REAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    row = {"id": "one", "text": "alpha"}
    actual_manifest["documents"] = [
        {
            "id": row["id"],
            "row_index": 130,
            "text_sha256": sha256(row["text"].encode("utf-8")).hexdigest(),
            "token_count": 5,
            "truncated": False,
        }
    ]
    root = tmp_path / "old"
    token_root = root / "tokens" / "fineweb-heldout"
    token_root.mkdir(parents=True)
    (token_root / "manifest.json").write_text(json.dumps(actual_manifest), encoding="utf-8")
    observed: dict[str, object] = {}

    def production_style_loader(repo_id: str, **kwargs: object) -> list[dict[str, object]]:
        observed["repo_id"] = repo_id
        observed.update(kwargs)
        return [row]

    recovered = recover_retention_documents(root, production_style_loader)
    assert [record.id for record in recovered] == ["one"]
    assert observed == {
        "repo_id": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "revision": actual_manifest["dataset"]["revision"],
        "split": "train",
        "streaming": True,
    }


def test_recovery_rejects_token_ids_and_requires_pinned_loader_revision(tmp_path: Path) -> None:
    rows = [{"id": "one", "text": "alpha"}]
    root = _old_root(tmp_path, rows)
    observed: dict[str, object] = {}

    def token_only_loader(repo_id: str, **kwargs: object) -> list[dict[str, object]]:
        observed["repo_id"] = repo_id
        observed.update(kwargs)
        return [{"id": "one", "input_ids": [1, 2, 3]}]

    with pytest.raises(RetentionSourceError, match="raw text"):
        recover_retention_documents(root, token_only_loader)
    assert observed == {
        "repo_id": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "revision": PINNED_REVISION,
        "split": "train",
        "streaming": True,
    }


def test_recovery_rejects_mutable_dataset_revision(tmp_path: Path) -> None:
    root = _old_root(tmp_path, [{"id": "one", "text": "alpha"}], revision="main")
    with pytest.raises(RetentionSourceError, match="immutable"):
        recover_retention_documents(root, lambda **kwargs: [])


def test_recovery_rejects_manifest_with_wrong_dataset_ordering_or_shuffle(tmp_path: Path) -> None:
    root = _old_root(tmp_path, [{"id": "one", "text": "alpha"}])
    manifest_path = root / "tokens" / "fineweb-heldout" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["name"] = "other-config"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RetentionSourceError, match="sample-10BT"):
        recover_retention_documents(root, lambda *args, **kwargs: [])

    manifest["dataset"]["name"] = "sample-10BT"
    manifest["ordering"] = "shuffled"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RetentionSourceError, match="repository order"):
        recover_retention_documents(root, lambda *args, **kwargs: [])

    manifest["ordering"] = "repository_order"
    manifest["shuffle"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RetentionSourceError, match="shuffle"):
        recover_retention_documents(root, lambda *args, **kwargs: [])


def test_recovery_rejects_document_seen_in_cache_and_stream(tmp_path: Path) -> None:
    rows = [{"id": "one", "text": "alpha"}, {"id": "two", "text": "beta"}]
    root = _old_root(tmp_path, rows)
    _write_cached_text(root, rows[:1])

    with pytest.raises(RetentionSourceError, match="duplicate document id"):
        recover_retention_documents(
            root,
            lambda repo_id, **kwargs: [
                {"id": "one", "text": "alpha"},
                {"id": "two", "text": "beta"},
            ],
        )


def _documents_for_exact_stream() -> tuple[DocumentRecord, ...]:
    # 20 documents force several boundaries, then a deterministic partial final chunk.
    return tuple(
        DocumentRecord(
            id=f"document-{index}",
            text=chr(65 + index % 26) * 1000,
            text_sha256=sha256((chr(65 + index % 26) * 1000).encode("utf-8")).hexdigest(),
            row_index=index,
        )
        for index in range(25)
    )


def test_materialization_has_exact_budget_eos_chunk_provenance_and_stable_replay(tmp_path: Path) -> None:
    documents = _documents_for_exact_stream()
    output = tmp_path / "retention"
    manifest = materialize_retention(CharacterTokenizer(), documents, output)
    chunks = load_retention_chunks(output)

    assert manifest["valid_source_tokens"] == 20_000
    assert manifest["chunk_count"] == 40
    assert sum(chunk.scored_tokens for chunk in chunks) == 20_000 - manifest["chunk_count"]
    assert max(len(chunk.input_ids) for chunk in chunks) <= 512
    assert len(chunks[-1].input_ids) == 32
    assert tuple(chunk.index for chunk in chunks) == tuple(range(manifest["chunk_count"]))
    assert all(chunk.scored_tokens == len(chunk.input_ids) - 1 and chunk.scored_tokens > 0 for chunk in chunks)
    assert all(chunk.source_document_ids for chunk in chunks)
    assert manifest["source_document_row_indices"] == list(range(20))
    assert isinstance(manifest["source_document_order_sha256"], str)
    assert manifest["final_document_id"] == "document-19"
    assert manifest["final_document_full_token_count"] == 1000
    assert manifest["final_document_consumed_token_count"] == 981
    assert manifest["final_document_truncated"] is True

    flattened = tuple(token for chunk in chunks for token in chunk.input_ids)
    assert flattened[:1002] == (ord("A"),) * 1000 + (0,) + (ord("B"),)
    first_bytes = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert materialize_retention(CharacterTokenizer(), documents, output) == manifest
    assert {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    } == first_bytes


def test_retention_materialization_rejects_unverified_or_duplicate_documents(tmp_path: Path) -> None:
    with pytest.raises(RetentionSourceError, match="hash mismatch"):
        materialize_retention(
            CharacterTokenizer(),
            (DocumentRecord("one", "alpha", "0" * 64, 0),),
            tmp_path,
        )
    document = _documents_for_exact_stream()[0]
    with pytest.raises(RetentionSourceError, match="duplicate document id"):
        materialize_retention(CharacterTokenizer(), (document, document), tmp_path)


def test_materialization_rejects_non_increasing_historical_source_order(tmp_path: Path) -> None:
    documents = _documents_for_exact_stream()
    with pytest.raises(RetentionSourceError, match="strictly increasing"):
        materialize_retention(CharacterTokenizer(), tuple(reversed(documents)), tmp_path)


def test_final_document_cutoff_metadata_is_exact_at_document_boundary_and_validated(tmp_path: Path) -> None:
    text = "Z" * 20_000
    document = DocumentRecord("boundary", text, sha256(text.encode("utf-8")).hexdigest(), 7)
    output = tmp_path / "retention"
    manifest = materialize_retention(CharacterTokenizer(), (document,), output)
    assert manifest["final_document_id"] == "boundary"
    assert manifest["final_document_full_token_count"] == 20_000
    assert manifest["final_document_consumed_token_count"] == 20_000
    assert manifest["final_document_truncated"] is False
    assert load_retention_chunks(output)

    manifest_path = output / "retention-manifest.json"
    corrupted = json.loads(manifest_path.read_text(encoding="utf-8"))
    corrupted["final_document_truncated"] = True
    manifest_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="final-document cutoff"):
        load_retention_chunks(output)
