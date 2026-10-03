from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from ppmoe.distill.evaluate import (
    ResourceLedger,
    ResourceLimitError,
    calibrate_candidates,
    export_review_packets,
    parser_review_metrics,
    reconcile_review,
    validate_calibration_rows,
)


def _rows(count: int = 60) -> list[dict[str, str]]:
    return [
        {"id": f"source-{i:03}", "question": f"Synthetic question {i}?", "gold": str(i), "solution": f"private {i}"}
        for i in range(count)
    ]


class FakeWorker:
    def __init__(self, *, eos: int = 60, overflow: set[str] | None = None) -> None:
        self.eos = eos
        self.overflow = overflow or set()

    def generate(self, row, candidate):
        assert "gold" not in row and "solution" not in row
        if row["id"] in self.overflow:
            return {"text": "", "stop_reason": "budget_overflow", "generated_tokens": 0}
        idx = int(row["id"].split("-")[1])
        reason = "eos" if idx < self.eos else "length"
        return {
            "text": f"Worked 1+2. Final answer: {idx}",
            "stop_reason": reason,
            "generated_tokens": 3,
            "wall_seconds": 0.01,
            "cuda_seconds": 0.01,
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
            "free_device_bytes": 300,
        }


def _worker_factory(role, candidate):
    return FakeWorker()


def test_calibration_uses_prespecified_first_passing_budget_and_no_gold() -> None:
    rows = _rows()
    selected = calibrate_candidates(
        rows,
        candidates=((256, 768), (512, 1024), (768, 1536)),
        worker_factory=_worker_factory,
        minimum_eos=57,
    )
    assert selected["candidate"] == [256, 768]
    assert selected["status"] == "awaiting_review"
    assert len(selected["attempts"]) == 1
    assert len(selected["attempts"][0]["rows"]) == 120


def test_calibration_row_validator_requires_exact_ids_prefix_and_signature() -> None:
    expected = [row["id"] for row in _rows(3)]
    rows = [{"source_id": row_id, "stage_signature": "abc"} for row_id in expected[:2]]
    assert validate_calibration_rows(rows, expected_ids=expected, stage_signature="abc") == 2
    with pytest.raises(ValueError, match="signature"):
        validate_calibration_rows(rows, expected_ids=expected, stage_signature="different")
    with pytest.raises(ValueError, match="prefix"):
        validate_calibration_rows([{"source_id": expected[1], "stage_signature": "abc"}], expected_ids=expected, stage_signature="abc")


def test_review_packet_is_gold_and_arm_free_and_tamper_evident(tmp_path: Path) -> None:
    rows = _rows(2)
    raw = [
        {"source_id": row["id"], "role": role, "candidate": [256, 768], "question": row["question"], "text": f"Final answer: {i}", "stop_reason": "eos"}
        for role in ("student", "teacher") for i, row in enumerate(rows)
    ]
    packet = export_review_packets(rows, raw, tmp_path, seed="stable-seed")
    packet_text = "\n".join(path.read_text(encoding="utf-8") for path in packet["packets"])
    assert "gold" not in packet_text and "student" not in packet_text and "teacher" not in packet_text
    assert "source-000" not in packet_text
    assert len(packet["row_ids"]) == 4
    assert all(set(json.loads(line)) == {"row_id", "question", "response", "stop_reason"} for line in packet_text.splitlines())
    packet["packets"][0].write_text(packet_text + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tamper"):
        reconcile_review(packet["packets"][0], [{"row_id": "unknown", "session_id": "s", "status": "valid", "value": "0", "evidence": "Final answer: 0"}])


def test_review_reconciliation_accepts_only_exact_substring_evidence(tmp_path: Path) -> None:
    review = [{"row_id": "opaque-a", "session_id": "opaque-session", "status": "valid", "value": "5", "evidence": "Final answer: 5"}]
    packet_path = tmp_path / "packet.jsonl"
    packet_bytes = b'{"row_id":"opaque-a","question":"q","response":"Worked. Final answer: 5","stop_reason":"eos"}\n'
    packet_path.write_bytes(packet_bytes)
    (tmp_path / "calibration.sha256.jsonl").write_text(
        json.dumps({"sha256": sha256(packet_bytes).hexdigest(), "row_count": 1}) + "\n",
        encoding="utf-8",
    )
    assert reconcile_review(packet_path, review)["row_count"] == 1
    review[0]["evidence"] = "citation not present"
    with pytest.raises(ValueError, match="substring"):
        reconcile_review(packet_path, review)


def test_parser_review_gate_accepts_equivalent_numeric_alias_without_gold() -> None:
    raw = [
        {"opaque_id": f"opaque-{i}", "text": f"Final answer: {i}.0", "stop_reason": "eos"}
        for i in range(60)
    ]
    reviews = [
        {"row_id": f"opaque-{i}", "session_id": "opaque-session", "status": "valid", "value": str(i), "evidence": f"Final answer: {i}.0"}
        for i in range(60)
    ]
    metrics = parser_review_metrics(raw, reviews)
    assert metrics["parser_reviewer_disagreements"] == 0
    assert metrics["invalid_responses"] == 0
    assert metrics["passes"]


def test_resource_ledger_preserves_initial_charges_and_caps() -> None:
    ledger = ResourceLedger.initial()
    assert ledger.cuda_seconds == 1168
    assert ledger.download_bytes == 8066699435
    ledger = ledger.charge_worker(cuda_seconds=None, wall_seconds=10, download_bytes=0, artifact_bytes=0)
    assert ledger.cuda_seconds == 1178
    assert ledger.wall_started_utc == "2026-10-03T08:01:07Z"
    with pytest.raises(ResourceLimitError, match="CUDA"):
        ledger.charge_worker(cuda_seconds=86400, wall_seconds=0, download_bytes=0, artifact_bytes=0)


def test_worker_lease_reclaims_only_proven_dead_pid_without_signalling(tmp_path: Path, monkeypatch) -> None:
    from ppmoe.distill.evaluate import WorkerLease

    lock = tmp_path / "worker.lock"
    lock.write_text('{"pid":123}', encoding="utf-8")
    monkeypatch.setattr("ppmoe.distill.evaluate._pid_is_alive", lambda pid: False)
    with WorkerLease(lock):
        assert lock.exists()
    assert not lock.exists()


def test_raw_journal_recovery_binds_failed_attempt_hash(tmp_path: Path) -> None:
    from ppmoe.distill.evaluate import RawRowJournal

    journal = RawRowJournal(tmp_path / "rows.jsonl", expected_ids=["x"], stage_signature="sig")
    journal.record_failure({"source_id": "x", "stage_signature": "sig", "status": "failed"})
    assert journal.failure_count() == 1
    attempts = journal.path.with_suffix(".jsonl.attempts.jsonl")
    attempts.write_text('{"stage_signature":"different"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="signature"):
        journal.failure_count()
