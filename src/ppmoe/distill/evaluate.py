"""Blind numeric calibration, reviewer packets, and conservative resource ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Sequence

from .answers import extract_final_numeric


class ResourceLimitError(RuntimeError):
    """A registered study resource cap would be exceeded."""


@dataclass(frozen=True, slots=True)
class ResourceLedger:
    cuda_seconds: int = 1168
    wall_seconds: int = 0
    download_bytes: int = 8066699435
    artifact_bytes: int = 0
    wall_started_utc: str = "2026-10-03T08:01:07Z"
    cuda_cap: int = 86400
    wall_cap: int = 172800
    download_cap: int = 21474836480
    artifact_cap: int = 21474836480

    @classmethod
    def initial(cls) -> "ResourceLedger":
        started = datetime.fromisoformat("2026-10-03T08:01:07+00:00")
        elapsed = max(0, int((datetime.now(timezone.utc) - started).total_seconds()))
        return cls(wall_seconds=elapsed)

    def charge_worker(
        self, *, cuda_seconds: float | None, wall_seconds: float,
        download_bytes: int, artifact_bytes: int,
    ) -> "ResourceLedger":
        # When synchronized device timing is absent, wall time is the upper bound.
        cuda_charge = int((wall_seconds if cuda_seconds is None else cuda_seconds) + 0.999999)
        wall_start = datetime.fromisoformat(self.wall_started_utc.replace("Z", "+00:00"))
        wall_elapsed = max(0, int((datetime.now(timezone.utc) - wall_start).total_seconds()))
        next_ledger = ResourceLedger(
            cuda_seconds=self.cuda_seconds + cuda_charge,
            wall_seconds=max(
                self.wall_seconds + int(wall_seconds + 0.999999),
                wall_elapsed,
            ),
            download_bytes=self.download_bytes + download_bytes,
            artifact_bytes=self.artifact_bytes + artifact_bytes,
            wall_started_utc=self.wall_started_utc,
            cuda_cap=self.cuda_cap,
            wall_cap=self.wall_cap,
            download_cap=self.download_cap,
            artifact_cap=self.artifact_cap,
        )
        if next_ledger.cuda_seconds > next_ledger.cuda_cap:
            raise ResourceLimitError("CUDA resource cap would be exceeded")
        if next_ledger.wall_seconds > next_ledger.wall_cap:
            raise ResourceLimitError("wall-time resource cap would be exceeded")
        if next_ledger.download_bytes > next_ledger.download_cap:
            raise ResourceLimitError("download resource cap would be exceeded")
        if next_ledger.artifact_bytes > next_ledger.artifact_cap:
            raise ResourceLimitError("artifact resource cap would be exceeded")
        return next_ledger


def preflight_disk(path: Path, *, reserve_bytes: int) -> int:
    free = shutil.disk_usage(path).free
    if free < reserve_bytes:
        raise ResourceLimitError(f"insufficient free disk: need {reserve_bytes}, have {free}")
    return free


class WorkerLease:
    """Exclusive durable marker preventing two live model workers in one run."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def __enter__(self) -> "WorkerLease":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                reservation = json.loads(self.path.read_text(encoding="utf-8"))
                pid = reservation.get("pid")
            except (OSError, ValueError, AttributeError):
                raise ResourceLimitError("an unreadable model-worker reservation already exists") from None
            if not isinstance(pid, int) or pid <= 0 or _pid_is_alive(pid):
                raise ResourceLimitError("a simultaneous live model worker is already reserved") from None
            self.path.unlink(missing_ok=True)
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                raise ResourceLimitError("a simultaneous live model worker is already reserved") from None
        os.write(self._fd, json.dumps({"pid": os.getpid(), "reserved_ns": time.time_ns()}).encode())
        os.fsync(self._fd)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if exc_type is None:
            self.path.unlink(missing_ok=True)


def validate_calibration_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_ids: Sequence[str], stage_signature: str,
) -> int:
    """Authenticate resumable completed rows and require an exact ordered ID prefix."""
    if not stage_signature:
        raise ValueError("calibration stage signature is missing")
    if len(rows) > len(expected_ids):
        raise ValueError("calibration rows exceed expected ID prefix")
    seen: list[str] = []
    for row in rows:
        if row.get("stage_signature") != stage_signature:
            raise ValueError("calibration row stage signature mismatch")
        source_id = row.get("source_id")
        if not isinstance(source_id, str):
            raise ValueError("calibration row source ID is invalid")
        seen.append(source_id)
    if seen != list(expected_ids[:len(rows)]):
        raise ValueError("calibration rows are not an exact expected ID prefix")
    return len(rows)


class RawRowJournal:
    """Append-only durable completed-row journal, authenticated on every resume."""

    def __init__(self, path: Path, *, expected_ids: Sequence[str], stage_signature: str) -> None:
        self.path = Path(path)
        self.expected_ids = tuple(expected_ids)
        self.stage_signature = stage_signature

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        validate_calibration_rows(rows, expected_ids=self.expected_ids, stage_signature=self.stage_signature)
        return rows

    def append(self, row: Mapping[str, Any]) -> None:
        rows = self.read()
        expected = self.expected_ids[len(rows)] if len(rows) < len(self.expected_ids) else None
        if row.get("source_id") != expected:
            raise ValueError("calibration append is not the next expected ID")
        if row.get("stage_signature") != self.stage_signature:
            raise ValueError("calibration row stage signature mismatch")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as stream:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    def record_failure(self, attempt: Mapping[str, Any]) -> None:
        """Persist cost/error attempts separately so incomplete rows never count as completed."""
        if attempt.get("stage_signature") != self.stage_signature:
            raise ValueError("calibration failure attempt stage signature mismatch")
        path = self.path.with_suffix(self.path.suffix + ".attempts.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as stream:
            stream.write(json.dumps(dict(attempt), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    def failure_count(self) -> int:
        path = self.path.with_suffix(self.path.suffix + ".attempts.jsonl")
        if not path.exists():
            return 0
        attempts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if any(item.get("stage_signature") != self.stage_signature for item in attempts):
            raise ValueError("calibration failure attempt stage signature mismatch")
        return len(attempts)


def calibrate_candidates(
    rows: Sequence[Mapping[str, str]], *, candidates: Sequence[Sequence[int]],
    worker_factory: Callable[[str, Sequence[int]], Any], minimum_eos: int,
    stage_signature: str = "injected-test-signature", row_sink: Callable[[Mapping[str, Any]], None] | None = None,
    row_loader: Callable[[str, Sequence[int]], Sequence[Mapping[str, Any]]] | None = None,
    failure_count_loader: Callable[[str, Sequence[int]], int] | None = None,
    ledger: ResourceLedger | None = None, max_retries: int = 2,
) -> dict[str, Any]:
    """Try the registered budgets in order, student then teacher, without exposing gold."""
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("calibration rows must have unique IDs")
    current_ledger = ledger or ResourceLedger.initial()
    attempt_records: list[dict[str, Any]] = []
    for candidate in candidates:
        pair = [int(candidate[0]), int(candidate[1])]
        if len(pair) != 2 or min(pair) <= 0:
            raise ValueError("each calibration budget needs positive prompt and generation limits")
        all_raw: list[dict[str, Any]] = []
        role_stats: dict[str, dict[str, int]] = {}
        candidate_status = "candidate_failed"
        for role in ("student", "teacher"):
            completed: list[dict[str, Any]] = [dict(row) for row in (row_loader(role, pair) if row_loader else ())]
            validate_calibration_rows(
                completed, expected_ids=[row["id"] for row in rows], stage_signature=stage_signature,
            )
            if any(row.get("role") != role or row.get("candidate") != pair for row in completed):
                raise ValueError("calibration recovery role/budget mismatch")
            errors = failure_count_loader(role, pair) if failure_count_loader else 0
            while errors <= max_retries and len(completed) < len(rows):
                worker = None
                worker_started = time.perf_counter()
                try:
                    worker = worker_factory(role, pair)
                    startup_wall = float(getattr(worker, "startup_wall_seconds", 0.0))
                    if startup_wall:
                        current_ledger = current_ledger.charge_worker(
                            cuda_seconds=None, wall_seconds=startup_wall,
                            download_bytes=0, artifact_bytes=0,
                        )
                    while len(completed) < len(rows):
                        index = len(completed)
                        source = rows[index]
                        exposed = {"id": source["id"], "question": source["question"]}
                        started = time.perf_counter()
                        try:
                            output = worker.generate(exposed, pair)
                            wall = float(output.get("wall_seconds", time.perf_counter() - started))
                            cuda = output.get("cuda_seconds")
                            serialized_bytes = len(json.dumps(raw := {
                                "opaque_id": hmac.new(
                                    stage_signature.encode(),
                                    f"{candidate[0]}:{candidate[1]}\0{role}\0{source['id']}".encode(),
                                    sha256,
                                ).hexdigest()[:20],
                                "source_id": source["id"], "role": role, "candidate": pair,
                                "question": source["question"], "text": str(output.get("text", "")),
                                "stop_reason": str(output.get("stop_reason", "error")),
                                "generated_tokens": int(output.get("generated_tokens", 0)),
                                "wall_seconds": wall, "cuda_seconds": cuda,
                                "worker_startup_wall_seconds": startup_wall if not completed else 0.0,
                                "peak_allocated_bytes": output.get("peak_allocated_bytes"),
                                "peak_reserved_bytes": output.get("peak_reserved_bytes"),
                                "free_device_bytes": output.get("free_device_bytes"),
                                "stage_signature": stage_signature,
                            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()) + 1
                            current_ledger = current_ledger.charge_worker(
                                cuda_seconds=None if cuda is None else float(cuda),
                                wall_seconds=wall, download_bytes=0, artifact_bytes=serialized_bytes,
                            )
                            completed.append(raw)
                            if row_sink is not None:
                                row_sink(raw)
                        except ResourceLimitError:
                            raise
                        except Exception as exc:
                            errors += 1
                            failed_wall = time.perf_counter() - started
                            current_ledger = current_ledger.charge_worker(
                                cuda_seconds=None, wall_seconds=failed_wall,
                                download_bytes=0, artifact_bytes=0,
                            )
                            failed = {
                                "source_id": source["id"], "role": role,
                                "candidate": pair, "status": "failed",
                                "error_type": type(exc).__name__, "wall_seconds": failed_wall,
                                "cuda_seconds": None, "stage_signature": stage_signature,
                            }
                            if row_sink is not None and hasattr(row_sink, "record_failure"):
                                row_sink.record_failure(failed)
                            break
                    if len(completed) == len(rows):
                        break
                except ResourceLimitError:
                    raise
                except Exception as exc:
                    errors += 1
                    failed_wall = time.perf_counter() - worker_started
                    failed = {
                        "role": role, "candidate": pair, "status": "worker_load_failed",
                        "error_type": type(exc).__name__, "wall_seconds": failed_wall,
                        "cuda_seconds": None, "stage_signature": stage_signature,
                    }
                    if row_sink is not None and hasattr(row_sink, "record_failure"):
                        row_sink.record_failure(failed)
                    if errors > max_retries:
                        break
                finally:
                    close = getattr(worker, "close", None)
                    if callable(close):
                        close()
            if len(completed) != len(rows):
                candidate_status = "stopped_resource" if errors > max_retries else "interrupted"
                break
            eos = sum(item["stop_reason"] == "eos" for item in completed)
            overflow = sum(item["stop_reason"] == "budget_overflow" for item in completed)
            role_stats[role] = {"completed": len(completed), "eos": eos, "prompt_overflow": overflow}
            all_raw.extend(completed)
            if eos < minimum_eos or overflow:
                candidate_status = "candidate_failed"
        else:
            candidate_status = "selected" if all(
                stats["eos"] >= minimum_eos and stats["prompt_overflow"] == 0
                for stats in role_stats.values()
            ) else "candidate_failed"
        attempt_records.append({
            "candidate": pair, "status": candidate_status,
            "role_stats": role_stats, "rows": all_raw,
            "resource_ledger": current_ledger.__dict__ if hasattr(current_ledger, "__dict__") else {
                name: getattr(current_ledger, name) for name in current_ledger.__dataclass_fields__
            },
        })
        if candidate_status == "selected":
            return {
                "status": "awaiting_review", "candidate": pair,
                "parser_sha256": sha256(Path(__file__).with_name("answers.py").read_bytes()).hexdigest(),
                "attempts": attempt_records,
                "selection_rule": f"first candidate with >= {minimum_eos}/{len(rows)} EOS per role and zero prompt overflows",
                "resource_ledger": attempt_records[-1]["resource_ledger"],
            }
        if candidate_status == "interrupted":
            return {"status": "interrupted", "candidate": None, "attempts": attempt_records}
        if candidate_status == "stopped_resource":
            return {
                "status": "stopped_resource", "candidate": None,
                "attempts": attempt_records,
                "reason": "bounded worker retries were exhausted; partial raw rows are preserved",
                "resource_ledger": attempt_records[-1]["resource_ledger"],
            }
    return {
        "status": "stopped_scientific", "candidate": None, "attempts": attempt_records,
        "reason": "no prespecified token budget met EOS and prompt-overflow criteria",
    }


def export_review_packets(
    source_rows: Sequence[Mapping[str, str]], raw_rows: Sequence[Mapping[str, Any]],
    directory: Path, *, seed: str,
) -> dict[str, Any]:
    """Write an arm-blind packet and a separate private source/role/gold mapping."""
    root = Path(directory)
    packet_dir, private_dir = root / "packets", root / "private"
    packet_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    gold_by_id = {row["id"]: row["gold"] for row in source_rows}
    packet_rows: list[dict[str, str]] = []
    private_rows: list[dict[str, str]] = []
    for raw in raw_rows:
        source_id = str(raw["source_id"])
        opaque = str(raw.get("opaque_id") or hmac.new(seed.encode(), f"{raw['role']}\0{source_id}".encode(), sha256).hexdigest()[:20])
        packet_rows.append({
            "row_id": opaque,
            "question": str(raw["question"]),
            "response": str(raw["text"]),
            "stop_reason": str(raw["stop_reason"]),
        })
        private_rows.append({
            "row_id": opaque, "source_id": source_id,
            "role": str(raw["role"]), "gold": gold_by_id[source_id],
        })
    if len({row["row_id"] for row in packet_rows}) != len(packet_rows):
        raise ValueError("opaque review IDs collided")
    packet_rows.sort(key=lambda row: row["row_id"])
    packet_path = packet_dir / "calibration.jsonl"
    private_path = private_dir / "review-map.jsonl"
    _write_exclusive_jsonl(packet_path, packet_rows)
    _write_exclusive_jsonl(private_path, private_rows)
    packet_sha = _sha256(packet_path)
    _write_exclusive_jsonl(packet_dir / "calibration.sha256.jsonl", [{"sha256": packet_sha, "row_count": len(packet_rows)}])
    return {"status": "awaiting_review", "packets": [packet_path], "row_ids": [row["row_id"] for row in packet_rows], "packet_sha256": packet_sha, "private_map": private_path}


def reconcile_review(packet_path: Path, reviews: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Validate opaque reviewer rowsets and substring citations without reparsing them."""
    path = Path(packet_path)
    packet_sha = _sha256(path)
    sidecar = path.with_name("calibration.sha256.jsonl")
    if not sidecar.exists():
        raise ValueError("review packet digest sidecar is missing")
    metadata = json.loads(sidecar.read_text(encoding="utf-8").splitlines()[0])
    if metadata.get("sha256") != packet_sha:
        raise ValueError("review packet tamper detected")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if metadata.get("row_count") != len(rows):
        raise ValueError("review packet row-count tamper detected")
    by_id = {row["row_id"]: row for row in rows}
    seen: set[str] = set()
    sessions: set[str] = set()
    for review in reviews:
        row_id, session = review.get("row_id"), review.get("session_id")
        if row_id not in by_id or row_id in seen:
            raise ValueError("review row set contains unknown or duplicate opaque IDs")
        if not isinstance(session, str) or not session:
            raise ValueError("review session ID must be opaque and nonempty")
        seen.add(row_id)
        sessions.add(session)
        status = review.get("status")
        if status not in {"valid", "invalid"}:
            raise ValueError("review status must be valid or invalid")
        evidence = review.get("evidence") or ""
        if evidence and evidence not in by_id[row_id]["response"]:
            raise ValueError("review evidence is not an exact response substring")
        if status == "valid" and (not review.get("value") or not evidence):
            raise ValueError("valid review requires value and evidence")
        if status == "valid":
            try:
                Fraction(str(review["value"]))
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError("valid review value must be a numeric scalar") from exc
    if seen != set(by_id):
        raise ValueError("review row set is incomplete")
    return {"status": "reconciled", "row_count": len(seen), "valid_rows": len(seen), "sessions": sorted(sessions), "packet_sha256": packet_sha}


def parser_review_metrics(
    raw_rows: Sequence[Mapping[str, Any]], reviews: Sequence[Mapping[str, str]],
) -> dict[str, int | bool]:
    """Compare review labels to parser outputs; caller supplies no gold values."""
    review_by_id = {row["row_id"]: row for row in reviews}
    disagreements = 0
    invalid = 0
    for raw in raw_rows:
        row_id = str(raw["opaque_id"])
        extraction = extract_final_numeric(str(raw["text"]), str(raw["stop_reason"]))
        review = review_by_id[row_id]
        parsed_valid = extraction.status == "valid"
        invalid += int(not parsed_valid)
        disagreements += int(parsed_valid != (review["status"] == "valid"))
        if parsed_valid and review["status"] == "valid":
            try:
                values_match = Fraction(str(extraction.value)) == Fraction(str(review.get("value")))
            except (ValueError, ZeroDivisionError):
                values_match = False
            disagreements += int(not values_match)
    count = len(raw_rows)
    return {
        "rows": count,
        "parser_reviewer_disagreements": disagreements,
        "invalid_responses": invalid,
        "passes": count == 60 and disagreements <= 3 and invalid <= 6,
    }


def _write_exclusive_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing immutable JSONL differs: {path.name}")
        return
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pid_is_alive(pid: int) -> bool:
    """Read process state without sending a signal or mutating the target."""
    if os.name != "nt":
        # This study is Windows-bound; unknown platform state conservatively blocks.
        return True
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ctypes.get_last_error() == 5  # access denied: conservatively treat as live
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


__all__ = [
    "RawRowJournal", "ResourceLedger", "ResourceLimitError", "WorkerLease",
    "calibrate_candidates", "export_review_packets", "parser_review_metrics",
    "preflight_disk", "reconcile_review", "validate_calibration_rows",
]
