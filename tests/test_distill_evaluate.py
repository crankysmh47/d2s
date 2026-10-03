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
    from hashlib import sha256
    import json

    expected = [row["id"] for row in _rows(3)]
    rows = []
    for row_id in expected[:2]:
        row = {"source_id": row_id, "stage_signature": "abc"}
        row["payload_sha256"] = sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        rows.append(row)
    assert validate_calibration_rows(rows, expected_ids=expected, stage_signature="abc") == 2
    with pytest.raises(ValueError, match="signature"):
        validate_calibration_rows(rows, expected_ids=expected, stage_signature="different")
    out_of_order = {"source_id": expected[1], "stage_signature": "abc"}
    out_of_order["payload_sha256"] = sha256(json.dumps(out_of_order, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="prefix"):
        validate_calibration_rows([out_of_order], expected_ids=expected, stage_signature="abc")


def test_review_packet_is_gold_and_arm_free_and_tamper_evident(tmp_path: Path) -> None:
    rows = _rows(2)
    raw = [
        {"source_id": row["id"], "role": role, "candidate": [256, 768], "question": row["question"], "text": f"Final answer: {i}", "stop_reason": "eos"}
        for role in ("student", "teacher") for i, row in enumerate(rows)
    ]
    packet = export_review_packets(rows, raw, tmp_path, seed="stable-seed", private_gold_by_id={row["id"]: row["gold"] for row in rows})
    packet_text = "\n".join(path.read_text(encoding="utf-8") for path in packet["packets"])
    assert "gold" not in packet_text and "student" not in packet_text and "teacher" not in packet_text
    assert "source-000" not in packet_text
    assert len(packet["row_ids"]) == 4
    assert all(set(json.loads(line)) == {"row_id", "question", "response", "stop_reason"} for line in packet_text.splitlines())
    packet["packets"][0].write_text(packet_text + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tamper"):
        reconcile_review(packet["packets"][0], [{"row_id": "unknown", "session_id": "s", "status": "valid", "value": "0", "evidence": "Final answer: 0"}])


def test_review_export_uses_separate_private_gold_mapping(tmp_path: Path) -> None:
    source_rows = [{"id": "source-0", "question": "q?"}]
    raw = [{"source_id": "source-0", "role": "student", "question": "q?", "text": "Final answer: 0", "stop_reason": "eos"}]
    packet = export_review_packets(
        source_rows, raw, tmp_path, seed="seed", private_gold_by_id={"source-0": "0"},
    )
    packet_text = packet["packets"][0].read_text(encoding="utf-8")
    private_text = Path(packet["private_map"]).read_text(encoding="utf-8")
    assert '"gold"' not in packet_text and '"role"' not in packet_text
    assert '"gold":"0"' in private_text


def test_raw_journal_rejects_modified_payload_even_with_signature(tmp_path: Path) -> None:
    from ppmoe.distill.evaluate import RawRowJournal

    journal = RawRowJournal(tmp_path / "rows.jsonl", expected_ids=["x"], stage_signature="sig")
    journal.append({"source_id": "x", "stage_signature": "sig", "role": "student", "candidate": [256, 768], "text": "Final answer: 1"})
    saved = json.loads(journal.path.read_text(encoding="utf-8"))
    saved["text"] = "Final answer: 2"
    journal.path.write_text(json.dumps(saved) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        journal.read()


def test_failed_row_is_persisted_via_explicit_failure_sink_before_retry() -> None:
    class Flaky:
        def __init__(self):
            self.calls = 0

        def generate(self, row, candidate):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic row failure")
            return {"text": "Final answer: 0", "stop_reason": "eos", "generated_tokens": 2, "wall_seconds": 0.01}

    made = []
    failures = []
    reservations = []
    class FailureSink:
        def record_attempt(self, attempt):
            reservations.append(attempt)

        def record_failure(self, attempt):
            failures.append(attempt)

    def factory(role, candidate):
        worker = Flaky()
        # First worker per role fails; its retry worker succeeds.
        state = getattr(factory, "starts", {})
        count = state.get(role, 0)
        state[role] = count + 1
        factory.starts = state
        if count:
            worker.calls = 1
        else:
            worker.calls = 0
        made.append((role, worker))
        return worker

    result = calibrate_candidates(
        [{"id": "x", "question": "q?"}], candidates=((256, 768),),
        worker_factory=factory, minimum_eos=1, max_retries=2,
        failure_sink=FailureSink(),
    )
    assert result["status"] == "awaiting_review"
    assert failures[0]["status"] == "failed"
    assert failures[0]["wall_seconds"] >= 0
    assert reservations[0]["status"] == "worker_reserved"


def test_aggregate_worker_retry_exhaustion_is_environment_stop() -> None:
    class BrokenWorkerFactory:
        def __call__(self, role, candidate):
            raise RuntimeError("synthetic load failure")

    class FailureSink:
        def record_attempt(self, attempt):
            pass

        def record_failure(self, attempt):
            pass

    result = calibrate_candidates(
        [{"id": "x", "question": "q?"}], candidates=((256, 768),),
        worker_factory=BrokenWorkerFactory(), minimum_eos=1, max_retries=2,
        failure_sink=FailureSink(),
    )
    assert result["status"] == "stopped_environment"


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


def test_resource_ledger_wall_is_global_elapsed_and_cap_error_carries_attempted_totals() -> None:
    from datetime import datetime, timezone, timedelta

    started = (datetime.now(timezone.utc) - timedelta(seconds=100)).isoformat().replace("+00:00", "Z")
    ledger = ResourceLedger(wall_seconds=100, wall_started_utc=started)
    after_one = ledger.charge_worker(cuda_seconds=0, wall_seconds=40, download_bytes=0, artifact_bytes=0)
    after_two = after_one.charge_worker(cuda_seconds=0, wall_seconds=40, download_bytes=0, artifact_bytes=0)
    assert after_one.wall_seconds == after_two.wall_seconds

    capped = ResourceLedger(cuda_seconds=9, cuda_cap=10)
    with pytest.raises(ResourceLimitError, match="CUDA") as caught:
        capped.charge_worker(cuda_seconds=2, wall_seconds=2, download_bytes=0, artifact_bytes=5)
    assert caught.value.ledger.cuda_seconds == 11
    assert caught.value.ledger.artifact_bytes == 5


def test_failed_durable_row_append_retries_same_id_without_advancing_prefix() -> None:
    generated_ids = []
    saved = []
    class Worker:
        def generate(self, row, candidate):
            generated_ids.append(row["id"])
            return {"text": "Final answer: 1", "stop_reason": "eos", "generated_tokens": 2,
                    "wall_seconds": 0.001, "cuda_seconds": 0}

    failed_once = False
    def sink(row):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise OSError("synthetic append failure")
        saved.append(dict(row))

    result = calibrate_candidates(
        [{"id": "same-id", "question": "q?"}], candidates=((256, 768),),
        worker_factory=lambda role, candidate: Worker(), minimum_eos=1,
        row_sink=sink, max_retries=2,
    )
    assert generated_ids == ["same-id", "same-id", "same-id"]
    assert len(saved) == 2
    assert result["status"] == "awaiting_review"


def test_worker_close_failure_has_no_worker_complete_and_carries_cost() -> None:
    from ppmoe.distill.model import WorkerOwnershipError
    attempts = []
    failures = []
    class Sink:
        def record_attempt(self, row): attempts.append(dict(row))
        def record_failure(self, row): failures.append(dict(row))
    class Worker:
        startup_wall_seconds = 0.001
        def generate(self, row, candidate):
            return {"text": "Final answer: 1", "stop_reason": "eos", "generated_tokens": 2,
                    "wall_seconds": 0.001, "cuda_seconds": 0}
        def close(self): raise RuntimeError("synthetic teardown failure")

    with pytest.raises(WorkerOwnershipError) as caught:
        calibrate_candidates(
            [{"id": "x", "question": "q?"}], candidates=((256, 768),),
            worker_factory=lambda role, candidate: Worker(), minimum_eos=1,
            failure_sink=Sink(),
        )
    assert caught.value.ledger is not None
    assert not any(item["status"] == "worker_complete" for item in attempts)
    assert failures[-1]["status"] == "worker_close_failed"


def test_last_durable_row_over_artifact_cap_preserves_attempted_total() -> None:
    saved = []
    class Worker:
        def generate(self, row, candidate):
            return {"text": "Final answer: 1", "stop_reason": "eos", "generated_tokens": 2,
                    "wall_seconds": 0, "cuda_seconds": 0}
    ledger = ResourceLedger(artifact_bytes=0, artifact_cap=0)
    with pytest.raises(ResourceLimitError, match="artifact") as caught:
        calibrate_candidates(
            [{"id": "x", "question": "q?"}], candidates=((256, 768),),
            worker_factory=lambda role, candidate: Worker(), minimum_eos=1,
            row_sink=lambda row: saved.append(dict(row)), ledger=ledger,
        )
    assert len(saved) == 1
    assert caught.value.ledger.artifact_bytes > caught.value.ledger.artifact_cap


def test_worker_lease_reclaims_only_proven_dead_pid_without_signalling(tmp_path: Path, monkeypatch) -> None:
    from ppmoe.distill.evaluate import WorkerLease

    lock = tmp_path / "worker.lock"
    lock.write_text('{"pid":123}', encoding="utf-8")
    monkeypatch.setattr("ppmoe.distill.evaluate._pid_is_alive", lambda pid: False)
    with WorkerLease(lock):
        assert lock.exists()
    assert not lock.exists()


def test_worker_lease_blocks_dead_parent_with_live_child(tmp_path: Path, monkeypatch) -> None:
    from ppmoe.distill.evaluate import WorkerLease

    lock = tmp_path / "worker.lock"
    lock.write_text('{"pid":123,"child_pids":[456]}', encoding="utf-8")
    monkeypatch.setattr("ppmoe.distill.evaluate._pid_is_alive", lambda pid: pid == 456)
    with pytest.raises(ResourceLimitError, match="live model worker"):
        with WorkerLease(lock):
            pytest.fail("must not reclaim while a child is alive")
    assert lock.exists()


def test_raw_journal_recovery_binds_failed_attempt_hash(tmp_path: Path) -> None:
    from ppmoe.distill.evaluate import RawRowJournal

    journal = RawRowJournal(tmp_path / "rows.jsonl", expected_ids=["x"], stage_signature="sig")
    journal.record_failure({"source_id": "x", "stage_signature": "sig", "status": "failed"})
    assert journal.failure_count() == 1
    attempts = journal.path.with_suffix(".jsonl.attempts.jsonl")
    attempts.write_text('{"stage_signature":"different"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="signature"):
        journal.failure_count()


def test_raw_journal_rejects_tampered_failed_attempt_digest(tmp_path: Path) -> None:
    from ppmoe.distill.evaluate import RawRowJournal

    journal = RawRowJournal(tmp_path / "rows.jsonl", expected_ids=["x"], stage_signature="sig")
    journal.record_failure({"source_id": "x", "stage_signature": "sig", "status": "failed", "error_type": "RuntimeError"})
    attempts = journal.path.with_suffix(".jsonl.attempts.jsonl")
    item = json.loads(attempts.read_text(encoding="utf-8"))
    item["error_type"] = "DifferentError"
    attempts.write_text(json.dumps(item) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        journal.failure_count()
