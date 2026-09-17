import json
from pathlib import Path

import pytest

from ppmoe.recurrence.provenance import (
    append_event,
    canonical_json,
    directory_size,
    sha256_file,
    sha256_payload,
    write_immutable_json,
)


def test_payload_hash_ignores_mapping_order() -> None:
    """Catches signatures depending on incidental insertion order."""
    assert sha256_payload({"b": 2, "a": 1}) == sha256_payload({"a": 1, "b": 2})


def test_canonical_json_is_compact_utf8_with_one_newline() -> None:
    """Catches manifest bytes changing through whitespace or text encoding."""
    assert canonical_json({"message": "μ", "value": 1}) == b'{"message":"\xce\xbc","value":1}\n'


def test_immutable_writer_accepts_identical_replay(tmp_path: Path) -> None:
    """Catches safe stage replay being rejected after its first completed write."""
    path = tmp_path / "manifest.json"
    first = write_immutable_json(path, {"value": 1})
    second = write_immutable_json(path, {"value": 1})

    assert first == second == sha256_file(path)


def test_immutable_writer_rejects_different_payload(tmp_path: Path) -> None:
    """Catches a completed manifest being silently overwritten by a new result."""
    path = tmp_path / "manifest.json"
    write_immutable_json(path, {"value": 1})

    with pytest.raises(FileExistsError, match="immutable"):
        write_immutable_json(path, {"value": 2})


def test_append_event_assigns_monotonic_indices_and_utc_timestamps(tmp_path: Path) -> None:
    """Catches a ledger replay losing event order or provenance time."""
    path = tmp_path / "events.jsonl"

    assert append_event(path, {"stage": "source"}) == 0
    assert append_event(path, {"stage": "data"}) == 1

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["index"] for event in events] == [0, 1]
    assert [event["stage"] for event in events] == ["source", "data"]
    assert all(event["timestamp"].endswith("+00:00") for event in events)


def test_append_event_rejects_malformed_final_history_line(tmp_path: Path) -> None:
    """Catches history corruption being hidden by truncating the final event."""
    path = tmp_path / "events.jsonl"
    path.write_text('{"index":0}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="malformed.*final"):
        append_event(path, {"stage": "data"})


def test_append_event_separates_valid_final_line_without_terminal_newline(
    tmp_path: Path,
) -> None:
    """Catches appending a second JSON object directly onto valid final JSON."""
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"index":0,"stage":"source"}')

    assert append_event(path, {"stage": "data"}) == 1

    records = path.read_bytes().splitlines()
    assert len(records) == 2
    assert json.loads(records[0]) == {"index": 0, "stage": "source"}
    assert json.loads(records[1])["stage"] == "data"


def test_directory_size_counts_only_file_bytes(tmp_path: Path) -> None:
    """Catches artifact budget checks counting directory metadata instead of artifacts."""
    (tmp_path / "one.bin").write_bytes(b"abc")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "two.bin").write_bytes(b"12345")

    assert directory_size(tmp_path) == 8
