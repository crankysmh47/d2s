"""Registered statistical summaries and terminal recurrence decisions.

This module intentionally contains no model code.  It consumes immutable raw
records and performs all aggregate arithmetic with deterministic NumPy
generators so that a report can be recomputed independently of evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import numbers
import os
from pathlib import Path
import time
from threading import Lock
from typing import Callable, Iterator

import numpy as np
import torch

from .config import RecurrenceConfig
from .provenance import canonical_json, sha256_payload, write_immutable_json


_REPLICATES = 10_000
_LOWER_PERCENTILE = 2.5
_UPPER_PERCENTILE = 97.5
_REGISTERED_FAMILIES = ("expressions", "lists", "graphs")
_REGISTERED_CONFIRMATION_SEEDS = (47, 48)
_REGISTERED_COMPARISONS = ("a2-f0", "a2-a1", "a2-a3")
_CUDA_PROJECTION_MULTIPLIER = 1.5
_PILOT_WARMUPS = 5
_PILOT_SAMPLES = 20
_CUDA_LEDGER_FORMAT = "ppmoe.recurrence.cuda-cost-ledger.v1"
_CUDA_LEDGER_GENESIS = sha256_payload(
    {"domain": _CUDA_LEDGER_FORMAT, "kind": "genesis"}
)
_LEDGER_THREAD_LOCKS: dict[Path, Lock] = {}
_LEDGER_THREAD_LOCKS_GUARD = Lock()
SYNCHRONIZED_TIMING_MECHANISM = "torch.cuda.synchronize+time.perf_counter"


@dataclass(frozen=True, slots=True)
class CudaCostEntry:
    """One immutable CUDA-active attempt interval in the 24-hour ledger."""

    format: str
    index: int
    previous_integrity: str
    attempt_id: str
    stage: str
    elapsed_seconds: float
    outcome: str
    cumulative_seconds: float
    integrity: str


class CudaCostLedger:
    """Append-only CUDA-active wall-time accounting.

    The ledger deliberately receives only CUDA-job intervals.  Idle operator
    time and model-download time have no `record` call and therefore cannot be
    charged accidentally.  Failed attempts are durable entries rather than
    being rolled back when a fresh-process retry succeeds.
    """

    def __init__(self, path: Path, cap_seconds: float, entries: Sequence[CudaCostEntry] = ()) -> None:
        self._path = path
        self._cap_seconds = _positive_finite(cap_seconds, "cap_seconds")
        self._entries = tuple(entries)

    @classmethod
    def load(cls, path: Path, cap_seconds: float) -> "CudaCostLedger":
        """Recover then validate an append-only JSONL CUDA-cost ledger."""
        validated_cap = _positive_finite(cap_seconds, "cap_seconds")
        with _ledger_lock(path):
            entries = _recover_and_load_ledger(path)
        return cls(path, validated_cap, entries)

    @property
    def entries(self) -> tuple[CudaCostEntry, ...]:
        return self._entries

    @property
    def cap_seconds(self) -> float:
        return self._cap_seconds

    @property
    def total_seconds(self) -> float:
        return self._entries[-1].cumulative_seconds if self._entries else 0.0

    def record(self, attempt_id: str, stage: str, elapsed_seconds: float, outcome: str) -> None:
        """Append one finalized CUDA attempt exactly once."""
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("CUDA attempt ID must be a non-empty string")
        if not isinstance(stage, str) or not stage:
            raise ValueError("CUDA stage must be a non-empty string")
        if not isinstance(outcome, str) or not outcome:
            raise ValueError("CUDA outcome must be a non-empty string")
        elapsed = _nonnegative_finite(elapsed_seconds, "elapsed_seconds")
        with _ledger_lock(self._path):
            current = _recover_and_load_ledger(self._path)
            self._entries = tuple(current)
            if attempt_id in {entry.attempt_id for entry in current}:
                raise ValueError("CUDA ledger rejects duplicate attempt IDs")
            entry = _new_cuda_cost_entry(
                current, attempt_id=attempt_id, stage=stage, elapsed_seconds=elapsed, outcome=outcome
            )
            _write_pending_transaction(self._path, entry)
            write_immutable_json(_ledger_anchor_path(self._path, entry.index), _ledger_anchor_payload(entry))
            _append_ledger_record(self._path, entry)
            _clear_pending_transaction(self._path)
            self._entries = (*current, entry)


def measure_training_step(
    model: object,
    batch: Mapping[str, object],
    optimizer: torch.optim.Optimizer,
    warmups: int = _PILOT_WARMUPS,
    samples: int = _PILOT_SAMPLES,
    *,
    synchronizer: Callable[[], None] | None = None,
    clock: Callable[[], float] | None = None,
    clipper: Callable[[Sequence[torch.Tensor]], object] | None = None,
) -> tuple[float, ...]:
    """Measure 20 synchronized, preloaded full training updates after 5 warmups."""
    _require_registered_pilot_shape(warmups, samples)
    synchronize = synchronizer or _synchronize_cuda
    timer = clock or time.perf_counter
    apply_clip = clipper or _clip_registered_gradients
    for _ in range(warmups):
        _run_timed_training_update(model, batch, optimizer, synchronize, timer, apply_clip)
    return tuple(
        _run_timed_training_update(model, batch, optimizer, synchronize, timer, apply_clip)
        for _ in range(samples)
    )


def measure_training_arms(
    arms: Mapping[str, tuple[object, Mapping[str, object], torch.optim.Optimizer]],
    warmups: int = _PILOT_WARMUPS,
    samples: int = _PILOT_SAMPLES,
    *,
    synchronizer: Callable[[], None] | None = None,
    clock: Callable[[], float] | None = None,
    clipper: Callable[[Sequence[torch.Tensor]], object] | None = None,
) -> dict[str, tuple[float, ...]]:
    """Measure A1/A2 with warmups excluded and sample order A1,A2,A1,A2,..."""
    _require_registered_pilot_shape(warmups, samples)
    if set(arms) != {"a1", "a2"}:
        raise ValueError("timing pilot requires exactly A1 and A2 arms")
    synchronize = synchronizer or _synchronize_cuda
    timer = clock or time.perf_counter
    apply_clip = clipper or _clip_registered_gradients
    for arm in ("a1", "a2"):
        model, batch, optimizer = arms[arm]
        for _ in range(warmups):
            _run_timed_training_update(model, batch, optimizer, synchronize, timer, apply_clip)
    result: dict[str, list[float]] = {"a1": [], "a2": []}
    for _ in range(samples):
        for arm in ("a1", "a2"):
            model, batch, optimizer = arms[arm]
            result[arm].append(
                _run_timed_training_update(model, batch, optimizer, synchronize, timer, apply_clip)
            )
    return {arm: tuple(values) for arm, values in result.items()}


def derive_a3_steps(t1: Sequence[float], t2: Sequence[float]) -> int:
    """Freeze the registered A3 ceiling from exact 20-sample A1/A2 pilots."""
    one = _validated_pilot_samples(t1, "t1")
    two = _validated_pilot_samples(t2, "t2")
    return max(1_000, math.ceil(1_000 * float(np.median(two)) / float(np.median(one))))


def resource_decision(
    measurements: Mapping[str, object], ledger: CudaCostLedger, config: RecurrenceConfig
) -> dict[str, object]:
    """Apply registered inclusive VRAM, CUDA, download, and artifact ceilings.

    `projected_remaining_seconds` must include every not-yet-completed arm,
    longest-prompt-plus-32-token generation, retention chunk, one retry
    allowance, and report CUDA estimate.  The projection is multiplied by
    1.5 before it is compared with the 24-hour cap.
    """
    peak_reserved = _nonnegative_integer(measurements, "peak_reserved_bytes")
    total_vram = _positive_integer(measurements, "total_vram_bytes")
    downloads = _nonnegative_integer(measurements, "download_bytes")
    artifacts = _nonnegative_integer(measurements, "artifact_bytes")
    projected_remaining = _nonnegative_measurement(measurements, "projected_remaining_seconds")
    available_vram = total_vram - config.resources.reserved_vram_bytes
    projected_total = ledger.total_seconds + _CUDA_PROJECTION_MULTIPLIER * projected_remaining
    gates = {
        "reserved_vram": {
            "value": peak_reserved,
            "limit": available_vram,
            "passed": peak_reserved <= available_vram,
        },
        "cuda_budget": {
            "spent_seconds": ledger.total_seconds,
            "projected_remaining_seconds": projected_remaining,
            "safety_multiplier": _CUDA_PROJECTION_MULTIPLIER,
            "projected_total_seconds": projected_total,
            "limit": config.resources.maximum_cuda_seconds,
            "passed": projected_total <= config.resources.maximum_cuda_seconds,
        },
        "source_download_bytes": {
            "value": downloads,
            "limit": config.resources.maximum_download_bytes,
            "passed": downloads <= config.resources.maximum_download_bytes,
        },
        "recurrence_artifact_bytes": {
            "value": artifacts,
            "limit": config.resources.maximum_artifact_bytes,
            "passed": artifacts <= config.resources.maximum_artifact_bytes,
        },
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "status": "passed" if passed else "stopped_budget",
        "reason": "passed" if passed else "registered_resource_limit_exceeded",
        "passed": passed,
        "cumulative_cuda_seconds": ledger.total_seconds,
        "gates": gates,
    }


def _cuda_cost_entry(raw: object, line_number: int) -> CudaCostEntry:
    if not isinstance(raw, Mapping) or set(raw) != {
        "format",
        "index",
        "previous_integrity",
        "attempt_id",
        "stage",
        "elapsed_seconds",
        "outcome",
        "cumulative_seconds",
        "integrity",
    }:
        raise ValueError(f"CUDA ledger line {line_number} has an invalid integrity schema")
    format_name = raw["format"]
    index = raw["index"]
    previous_integrity = raw["previous_integrity"]
    attempt_id = raw["attempt_id"]
    stage = raw["stage"]
    outcome = raw["outcome"]
    integrity = raw["integrity"]
    if not isinstance(format_name, str) or not format_name:
        raise ValueError("CUDA ledger integrity format must be a non-empty string")
    if isinstance(index, bool) or not isinstance(index, int) or index != line_number:
        raise ValueError("CUDA ledger integrity index is invalid")
    if not _is_sha256(previous_integrity) or not _is_sha256(integrity):
        raise ValueError("CUDA ledger integrity value must be a SHA-256 digest")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("CUDA ledger attempt ID must be a non-empty string")
    if not isinstance(stage, str) or not stage:
        raise ValueError("CUDA ledger stage must be a non-empty string")
    if not isinstance(outcome, str) or not outcome:
        raise ValueError("CUDA ledger outcome must be a non-empty string")
    return CudaCostEntry(
        format=format_name,
        index=index,
        previous_integrity=previous_integrity,
        attempt_id=attempt_id,
        stage=stage,
        elapsed_seconds=_nonnegative_finite(raw["elapsed_seconds"], "elapsed_seconds"),
        outcome=outcome,
        cumulative_seconds=_nonnegative_finite(raw["cumulative_seconds"], "cumulative_seconds"),
        integrity=integrity,
    )


def _ledger_record_payload(entry: CudaCostEntry) -> dict[str, object]:
    return {
        "format": entry.format,
        "index": entry.index,
        "previous_integrity": entry.previous_integrity,
        "attempt_id": entry.attempt_id,
        "stage": entry.stage,
        "elapsed_seconds": entry.elapsed_seconds,
        "outcome": entry.outcome,
        "cumulative_seconds": entry.cumulative_seconds,
        "integrity": entry.integrity,
    }


def _entry_integrity(entry: CudaCostEntry) -> str:
    return sha256_payload(
        {
            "domain": _CUDA_LEDGER_FORMAT,
            "kind": "entry",
            "index": entry.index,
            "previous_integrity": entry.previous_integrity,
            "attempt_id": entry.attempt_id,
            "stage": entry.stage,
            "elapsed_seconds": entry.elapsed_seconds,
            "outcome": entry.outcome,
            "cumulative_seconds": entry.cumulative_seconds,
        }
    )


def _ledger_anchor_root(path: Path) -> Path:
    return path.parent / f"{path.name}.anchors"


def _ledger_anchor_path(path: Path, index: int) -> Path:
    return _ledger_anchor_root(path) / f"{index:08d}.json"


def _ledger_anchor_payload(entry: CudaCostEntry) -> dict[str, object]:
    return {
        "domain": _CUDA_LEDGER_FORMAT,
        "kind": "immutable-entry-anchor",
        "index": entry.index,
        "integrity": entry.integrity,
    }


def _ledger_pending_path(path: Path) -> Path:
    return path.parent / f"{path.name}.pending.json"


def _ledger_pending_payload(entry: CudaCostEntry) -> dict[str, object]:
    return {
        "domain": _CUDA_LEDGER_FORMAT,
        "kind": "pending-append",
        "entry": _ledger_record_payload(entry),
    }


def _new_cuda_cost_entry(
    entries: Sequence[CudaCostEntry], *, attempt_id: str, stage: str, elapsed_seconds: float, outcome: str
) -> CudaCostEntry:
    without_integrity = CudaCostEntry(
        format=_CUDA_LEDGER_FORMAT,
        index=len(entries) + 1,
        previous_integrity=entries[-1].integrity if entries else _CUDA_LEDGER_GENESIS,
        attempt_id=attempt_id,
        stage=stage,
        elapsed_seconds=elapsed_seconds,
        outcome=outcome,
        cumulative_seconds=entries[-1].cumulative_seconds + elapsed_seconds if entries else elapsed_seconds,
        integrity="",
    )
    return CudaCostEntry(
        format=without_integrity.format,
        index=without_integrity.index,
        previous_integrity=without_integrity.previous_integrity,
        attempt_id=without_integrity.attempt_id,
        stage=without_integrity.stage,
        elapsed_seconds=without_integrity.elapsed_seconds,
        outcome=without_integrity.outcome,
        cumulative_seconds=without_integrity.cumulative_seconds,
        integrity=_entry_integrity(without_integrity),
    )


def _recover_and_load_ledger(path: Path) -> list[CudaCostEntry]:
    _recover_pending_transaction(path)
    entries = _read_ledger_entries(path)
    _validate_ledger_anchors(path, entries)
    return entries


def _read_ledger_entries(path: Path) -> list[CudaCostEntry]:
    if not path.exists():
        return []
    entries: list[CudaCostEntry] = []
    seen_attempts: set[str] = set()
    previous_total = 0.0
    previous_integrity = _CUDA_LEDGER_GENESIS
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                raise ValueError(f"CUDA ledger line {line_number} is blank")
            try:
                raw = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"CUDA ledger line {line_number} is invalid JSON") from error
            entry = _cuda_cost_entry(raw, line_number)
            _validate_entry_integrity(entry, expected_index=line_number, previous_integrity=previous_integrity)
            if entry.attempt_id in seen_attempts:
                raise ValueError("CUDA ledger contains a duplicate attempt ID")
            if entry.cumulative_seconds < previous_total or not math.isclose(
                entry.cumulative_seconds,
                previous_total + entry.elapsed_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("CUDA ledger cumulative totals must be monotonic and exact")
            seen_attempts.add(entry.attempt_id)
            previous_total = entry.cumulative_seconds
            previous_integrity = entry.integrity
            entries.append(entry)
    return entries


def _recover_pending_transaction(path: Path) -> None:
    pending = _read_pending_transaction(path)
    if pending is None:
        return
    entries = _read_ledger_entries(path)
    committed = list(entries)
    expected_paths = {_ledger_anchor_path(path, entry.index) for entry in committed}
    actual_paths = _ledger_anchor_paths(path)
    if not expected_paths.issubset(actual_paths):
        raise ValueError("CUDA ledger integrity anchor set is incomplete or unexpected")
    for entry in committed:
        _validate_ledger_anchor(path, entry)

    pending_anchor = _ledger_anchor_path(path, pending.index)
    extra_paths = actual_paths - expected_paths
    if pending.index == len(committed) + 1:
        previous = committed[-1].integrity if committed else _CUDA_LEDGER_GENESIS
        _validate_entry_integrity(pending, expected_index=pending.index, previous_integrity=previous)
        if extra_paths == set():
            write_immutable_json(pending_anchor, _ledger_anchor_payload(pending))
            _append_ledger_record(path, pending)
            _clear_pending_transaction(path)
            return
        if extra_paths != {pending_anchor}:
            raise ValueError("CUDA ledger integrity anchor set is incomplete or unexpected")
        _validate_ledger_anchor(path, pending)
        _append_ledger_record(path, pending)
        _clear_pending_transaction(path)
        return
    if pending.index == len(committed) and committed and committed[-1] == pending:
        if extra_paths:
            raise ValueError("CUDA ledger integrity anchor set is incomplete or unexpected")
        _validate_ledger_anchor(path, pending)
        _clear_pending_transaction(path)
        return
    raise ValueError("CUDA ledger pending transaction does not match durable tail")


def _write_pending_transaction(path: Path, entry: CudaCostEntry) -> None:
    write_immutable_json(_ledger_pending_path(path), _ledger_pending_payload(entry))


def _read_pending_transaction(path: Path) -> CudaCostEntry | None:
    pending_path = _ledger_pending_path(path)
    if not pending_path.exists():
        return None
    encoded = pending_path.read_bytes()
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError("CUDA ledger pending transaction is invalid JSON") from error
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"domain", "kind", "entry"}
        or payload.get("domain") != _CUDA_LEDGER_FORMAT
        or payload.get("kind") != "pending-append"
        or not isinstance(payload.get("entry"), Mapping)
        or encoded != canonical_json(payload)
    ):
        raise ValueError("CUDA ledger pending transaction integrity is invalid")
    entry_payload = payload["entry"]
    index = entry_payload.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("CUDA ledger pending transaction integrity is invalid")
    return _cuda_cost_entry(entry_payload, index)


def _append_ledger_record(path: Path, entry: CudaCostEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical_json(_ledger_record_payload(entry)))
        stream.flush()
        os.fsync(stream.fileno())


def _clear_pending_transaction(path: Path) -> None:
    _ledger_pending_path(path).unlink(missing_ok=True)


def _validate_entry_integrity(
    entry: CudaCostEntry, *, expected_index: int, previous_integrity: str
) -> None:
    if entry.format != _CUDA_LEDGER_FORMAT or entry.index != expected_index:
        raise ValueError("CUDA ledger integrity format or index is invalid")
    if entry.previous_integrity != previous_integrity:
        raise ValueError("CUDA ledger integrity chain is invalid")
    if entry.integrity != _entry_integrity(entry):
        raise ValueError("CUDA ledger integrity value is invalid")


def _validate_ledger_anchors(path: Path, entries: Sequence[CudaCostEntry]) -> None:
    expected = {_ledger_anchor_path(path, entry.index) for entry in entries}
    actual = _ledger_anchor_paths(path)
    if actual != expected:
        raise ValueError("CUDA ledger integrity anchor set is incomplete or unexpected")
    for entry in entries:
        _validate_ledger_anchor(path, entry)


def _ledger_anchor_paths(path: Path) -> set[Path]:
    anchor_root = _ledger_anchor_root(path)
    return set(anchor_root.iterdir()) if anchor_root.exists() else set()


def _validate_ledger_anchor(path: Path, entry: CudaCostEntry) -> None:
    anchor_path = _ledger_anchor_path(path, entry.index)
    if not anchor_path.is_file() or anchor_path.read_bytes() != canonical_json(_ledger_anchor_payload(entry)):
        raise ValueError("CUDA ledger integrity anchor does not match entry")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@contextmanager
def _ledger_lock(path: Path) -> Iterator[None]:
    """Serialize one ledger's recovery, duplicate claim, and durable append.

    The persistent zero-byte/one-byte lock file is not the lock state: the
    kernel advisory lock is released automatically when a process crashes.
    Keeping the path avoids an unlink/recreate race that could split writers
    across two lock files.
    """
    destination = path.resolve()
    with _LEDGER_THREAD_LOCKS_GUARD:
        thread_lock = _LEDGER_THREAD_LOCKS.setdefault(destination, Lock())
    with thread_lock:
        lock_path = destination.parent / f"{destination.name}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
            stream.seek(0)
            _acquire_os_file_lock(stream)
            try:
                yield
            finally:
                _release_os_file_lock(stream)


def _acquire_os_file_lock(stream: object) -> None:
    fileno = getattr(stream, "fileno")
    seek = getattr(stream, "seek")
    seek(0)
    if os.name == "nt":
        import msvcrt

        while True:
            try:
                msvcrt.locking(fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(0.01)
    else:
        import fcntl

        fcntl.flock(fileno(), fcntl.LOCK_EX)


def _release_os_file_lock(stream: object) -> None:
    fileno = getattr(stream, "fileno")
    seek = getattr(stream, "seek")
    seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fileno(), fcntl.LOCK_UN)


def _require_registered_pilot_shape(warmups: int, samples: int) -> None:
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups != _PILOT_WARMUPS:
        raise ValueError("registered timing pilot requires exactly 5 warmups")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples != _PILOT_SAMPLES:
        raise ValueError("registered timing pilot requires exactly 20 samples")


def _run_timed_training_update(
    model: object,
    batch: Mapping[str, object],
    optimizer: torch.optim.Optimizer,
    synchronize: Callable[[], None],
    clock: Callable[[], float],
    clipper: Callable[[Sequence[torch.Tensor]], object],
) -> float:
    """Time the complete preloaded update, bounded by synchronization calls."""
    synchronize()
    started = _finite_clock(clock())
    optimizer.zero_grad(set_to_none=True)
    loss = _answer_loss(model, batch)
    if loss.numel() != 1 or not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("timing pilot loss must be finite and scalar")
    loss.backward()
    parameters = _trainable_parameters(model)
    if not parameters or any(parameter.grad is None for parameter in parameters):
        raise FloatingPointError("timing pilot requires gradients for every trainable parameter")
    clipper(parameters)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    synchronize()
    return _nonnegative_finite(_finite_clock(clock()) - started, "timing sample")


def _answer_loss(model: object, batch: Mapping[str, object]) -> torch.Tensor:
    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    mode = batch.get("mode")
    method = getattr(model, "answer_loss", None)
    if not callable(method):
        raise TypeError("timing pilot model must provide answer_loss")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor) or not isinstance(mode, str):
        raise ValueError("timing pilot batch requires tensor input_ids, tensor labels, and string mode")
    loss = method(input_ids, labels, mode)
    if not isinstance(loss, torch.Tensor):
        raise TypeError("timing pilot answer_loss must return a tensor")
    return loss


def _trainable_parameters(model: object) -> list[torch.Tensor]:
    method = getattr(model, "parameters", None)
    if not callable(method):
        raise TypeError("timing pilot model must provide parameters")
    return [parameter for parameter in method() if getattr(parameter, "requires_grad", False)]


def _clip_registered_gradients(parameters: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.nn.utils.clip_grad_norm_(parameters, 1.0)


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _validated_pilot_samples(values: Sequence[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != _PILOT_SAMPLES:
        raise ValueError(f"{name} must contain exactly 20 positive finite samples")
    result = tuple(_positive_finite(value, f"{name} sample") for value in values)
    return result


def _finite_clock(value: object) -> float:
    return _positive_or_zero_finite(value, "clock")


def _positive_finite(value: object, name: str) -> float:
    number = _positive_or_zero_finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _nonnegative_finite(value: object, name: str) -> float:
    number = _positive_or_zero_finite(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return number


def _positive_or_zero_finite(value: object, name: str) -> float:
    if not _finite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _nonnegative_integer(measurements: Mapping[str, object], key: str) -> int:
    value = measurements.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _positive_integer(measurements: Mapping[str, object], key: str) -> int:
    value = _nonnegative_integer(measurements, key)
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _nonnegative_measurement(measurements: Mapping[str, object], key: str) -> float:
    return _nonnegative_finite(measurements.get(key), key)


def summarize_exact_scores(
    records: Sequence[Mapping[str, object]],
    baseline_records: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Return pooled and unweighted family-macro exact-answer statistics."""
    rows = _validate_score_records(records)
    by_family: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_family.setdefault(str(row["family"]), []).append(row)
    families: dict[str, object] = {}
    for family in sorted(by_family):
        members = by_family[family]
        correct = sum(row["correct"] is True for row in members)
        invalid = sum(row.get("reason_code") == "invalid_format" for row in members)
        count = len(members)
        accuracy = correct / count
        invalid_rate = invalid / count
        families[family] = {
            "record_count": count,
            "correct": correct,
            "incorrect": count - correct,
            "invalid_format": invalid,
            "accuracy": accuracy,
            "invalid_rate": invalid_rate,
            "informative": is_informative_family(accuracy, invalid_rate),
        }
    total = len(rows)
    correct_total = sum(row["correct"] is True for row in rows)
    invalid_total = sum(row.get("reason_code") == "invalid_format" for row in rows)
    family_values = [float(value["accuracy"]) for value in families.values()]
    result: dict[str, object] = {
        "record_count": total,
        "correct": correct_total,
        "incorrect": total - correct_total,
        "invalid_format": invalid_total,
        "pooled_accuracy": correct_total / total,
        "global_accuracy": correct_total / total,
        "global_invalid_rate": invalid_total / total,
        "macro_accuracy": float(np.mean(family_values)),
        "families": families,
        "family_order": list(sorted(families)),
    }
    if baseline_records is not None:
        base = _validate_score_records(baseline_records)
        left = {str(row["example_id"]): row for row in base}
        right = {str(row["example_id"]): row for row in rows}
        if set(left) != set(right):
            raise ValueError("baseline and candidate ID sets must match")
        for identifier in sorted(left):
            if left[identifier]["family"] != right[identifier]["family"]:
                raise ValueError(f"family mismatch for {identifier}")
        wrong_to_right = sum(left[identifier]["correct"] is False and right[identifier]["correct"] is True for identifier in left)
        right_to_wrong = sum(left[identifier]["correct"] is True and right[identifier]["correct"] is False for identifier in left)
        result["transitions"] = {
            "wrong_to_right": wrong_to_right,
            "right_to_wrong": right_to_wrong,
            "unchanged": total - wrong_to_right - right_to_wrong,
        }
    return result


def is_informative_family(accuracy: float, invalid_rate: float) -> bool:
    """Apply the registered strict accuracy and inclusive invalid-rate bounds."""
    return (
        _finite(accuracy)
        and _finite(invalid_rate)
        and 0.05 < float(accuracy) < 0.95
        and 0.0 <= float(invalid_rate) <= 0.10
    )


def classify_informative_families(f0_summary: Mapping[str, object]) -> dict[str, object]:
    families = f0_summary.get("families")
    if not isinstance(families, Mapping):
        raise ValueError("F0 summary must contain a families mapping")
    informative: list[str] = []
    non_informative: list[str] = []
    for family in sorted(families):
        item = families[family]
        if not isinstance(item, Mapping):
            raise ValueError(f"invalid family summary for {family}")
        accuracy = _required_number(item, "accuracy")
        invalid_rate = _required_number(item, "invalid_rate")
        (informative if is_informative_family(accuracy, invalid_rate) else non_informative).append(str(family))
    global_accuracy = _required_number(f0_summary, "macro_accuracy")
    global_invalid = _required_number(f0_summary, "global_invalid_rate")
    family_set_passed = set(str(name) for name in families) == set(_REGISTERED_FAMILIES)
    global_passed = 0.05 < global_accuracy < 0.95 and global_invalid <= 0.10
    minimum_passed = len(informative) >= 2
    return {
        "families": {str(name): families[name] for name in sorted(families)},
        "informative_families": informative,
        "non_informative_families": non_informative,
        "global_macro_informative": global_passed,
        "global_invalid_rate_passed": global_invalid <= 0.10,
        "registered_family_set_passed": family_set_passed,
        "minimum_informative_families_passed": minimum_passed,
        "passes": global_passed and minimum_passed and family_set_passed,
    }


def paired_accuracy_bootstrap(
    records_by_arm_seed: Mapping[tuple[str, int], Sequence[Mapping[str, object]]],
    replicates: int = _REPLICATES,
    seed: int = 9047,
) -> dict[str, object]:
    """Compute paired, family-stratified accuracy intervals.

    Each example's correctness difference is first averaged over available
    seed pairings.  Every replicate then resamples IDs within each family and
    uses the same sampled indices for every arm comparison.
    """
    _require_replicates(replicates)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("bootstrap seed must be an integer")
    if not records_by_arm_seed:
        raise ValueError("accuracy bootstrap requires records")
    arms: dict[str, dict[int, list[Mapping[str, object]]]] = {}
    for key, records in records_by_arm_seed.items():
        if not isinstance(key, tuple) or len(key) != 2 or not isinstance(key[0], str) or not isinstance(key[1], int) or isinstance(key[1], bool):
            raise ValueError("records must be keyed by (arm, integer seed)")
        if key[1] in arms.setdefault(key[0], {}):
            raise ValueError("duplicate arm/seed records")
        arms[key[0]][key[1]] = _validate_score_records(records)
    _, family_ids = _validate_common_universe(arms)
    candidate = "A2" if "A2" in arms else sorted(arms)[0]
    references = [arm for arm in ("F0", "A1", "A3") if arm in arms and arm != candidate]
    if not references:
        references = [arm for arm in sorted(arms) if arm != candidate]
    rng = np.random.default_rng(seed)
    # Persist the matrix in the registered family order.  This order is part
    # of the evidence contract because the deterministic RNG stream is shared
    # across family draws.
    sample_indices = {
        family: rng.integers(0, len(family_ids[family]), size=(replicates, len(family_ids[family])))
        for family in _REGISTERED_FAMILIES
        if family in family_ids
    }
    pair_data: dict[str, dict[str, object]] = {}
    for reference in references:
        name = f"{candidate}-{reference}"
        differences, families, transitions = _paired_differences(arms[candidate], arms[reference])
        family_arrays = {
            family: np.asarray(families[family], dtype=np.float64)
            for family in sorted(family_ids)
        }
        observed_family = {family: float(np.mean(values)) for family, values in family_arrays.items()}
        point = float(np.mean(list(observed_family.values())))
        bootstrap = np.empty(replicates, dtype=np.float64)
        for index in range(replicates):
            bootstrap[index] = float(np.mean([float(values[sample_indices[family][index]].mean()) for family, values in family_arrays.items()]))
        result = {
            "candidate": candidate,
            "reference": reference,
            "replicates": replicates,
            "seed": seed,
            "point": point,
            "ci_low": float(np.percentile(bootstrap, _LOWER_PERCENTILE)),
            "ci_high": float(np.percentile(bootstrap, _UPPER_PERCENTILE)),
            "macro": point,
            "family_deltas": observed_family,
            "wrong_to_right": transitions["wrong_to_right"],
            "right_to_wrong": transitions["right_to_wrong"],
            "sample_indices": {family: matrix.tolist() for family, matrix in sample_indices.items()},
            "bootstrap": bootstrap.tolist(),
        }
        pair_data[name] = result
    output: dict[str, object] = {"replicates": replicates, "seed": seed, "comparisons": pair_data}
    for name, result in pair_data.items():
        candidate_name, reference_name = name.split("-", 1)
        prefix = f"{candidate_name.lower()}_minus_{reference_name.lower()}"
        output[f"{prefix}_macro"] = result["point"]
        output[f"{prefix}_family_deltas"] = result["family_deltas"]
        output[f"{prefix}_ci_low"] = result["ci_low"]
        output[f"{prefix}_ci_high"] = result["ci_high"]
    return output


def paired_retention_bootstrap(
    candidate_chunks: Sequence[Mapping[str, object]],
    f0_chunks: Sequence[Mapping[str, object]],
    replicates: int = _REPLICATES,
    seed: int = 9157,
) -> dict[str, object]:
    """Compute a paired chunk-bootstrap candidate/F0 perplexity ratio."""
    _require_replicates(replicates)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("bootstrap seed must be an integer")
    candidate = _validate_retention_chunks(candidate_chunks)
    f0 = _validate_retention_chunks(f0_chunks)
    if [row["chunk_index"] for row in candidate] != [row["chunk_index"] for row in f0]:
        raise ValueError("candidate and F0 chunk index sets must match")
    if any(left["scored_tokens"] != right["scored_tokens"] for left, right in zip(candidate, f0, strict=True)):
        raise ValueError("candidate and F0 scored token counts must match")
    point = _retention_ratio(candidate, f0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(candidate), size=(replicates, len(candidate)))
    ratios = np.empty(replicates, dtype=np.float64)
    for replicate, sampled in enumerate(indices):
        candidate_sample = [candidate[int(index)] for index in sampled]
        f0_sample = [f0[int(index)] for index in sampled]
        ratios[replicate] = _retention_ratio(candidate_sample, f0_sample)
    result = {
        "replicates": replicates,
        "seed": seed,
        "point": point,
        "ppl_ratio_point": point,
        "ci_low": float(np.percentile(ratios, _LOWER_PERCENTILE)),
        "ci_high": float(np.percentile(ratios, _UPPER_PERCENTILE)),
        "ppl_ratio_ci_low": float(np.percentile(ratios, _LOWER_PERCENTILE)),
        "ppl_ratio_ci_high": float(np.percentile(ratios, _UPPER_PERCENTILE)),
        "sample_indices": {"retention": indices.tolist()},
        "bootstrap": ratios.tolist(),
    }
    return result


def discovery_decision(
    scores: Mapping[str, object],
    retention: Mapping[str, object],
    timing: Mapping[str, object],
) -> dict[str, object]:
    """Apply every registered G4 predicate without rounding measured values."""
    threshold = 0.02
    regression = 0.02
    ratio_limit = 1.02
    gates: dict[str, dict[str, object]] = {}
    for name, value, passed in (
        ("a2_minus_f0_macro", _score_metric(scores, "a2", "f0"), None),
        ("a2_minus_a1_macro", _score_metric(scores, "a2", "a1"), None),
        ("a2_minus_a3_macro", _score_metric(scores, "a2", "a3"), None),
    ):
        gates[name] = {"value": value, "threshold": threshold, "passed": _finite(value) and value >= threshold}
    gates["every_a2_f0_family_delta"] = _family_gate(scores, "a2", "f0", -regression)
    gates["every_a2_a1_family_delta"] = _family_gate(scores, "a2", "a1", -regression)
    point = _retention_metric(retention, "point")
    high = _retention_metric(retention, "ci_high")
    gates["a2_f0_ppl_ratio_point"] = {"value": point, "threshold": ratio_limit, "passed": _finite(point) and point <= ratio_limit}
    gates["a2_f0_ppl_ratio_ci_high"] = {"value": high, "threshold": ratio_limit, "passed": _finite(high) and high <= ratio_limit}
    timing_value = _timing_metric(timing)
    gates["a3_a2_realized_training_time_ratio"] = {"value": timing_value, "lower": 0.90, "upper": 1.10, "passed": _finite(timing_value) and 0.90 <= timing_value <= 1.10}
    quality_names = tuple(name for name in gates if name.startswith("a2_minus_") or name.startswith("every_"))
    quality_passed = all(bool(gates[name]["passed"]) for name in quality_names)
    retention_passed = bool(gates["a2_f0_ppl_ratio_point"]["passed"] and gates["a2_f0_ppl_ratio_ci_high"]["passed"])
    timing_passed = bool(gates["a3_a2_realized_training_time_ratio"]["passed"])
    status, reason = _terminal_reason(quality_passed, retention_passed, timing_passed)
    return {"status": status, "reason": reason, "passed": status == "passed", "gates": gates}


def confirmation_decision(
    scores: Mapping[str, object],
    retention: Mapping[str, object],
    timing: Mapping[str, object],
) -> dict[str, object]:
    """Apply every registered G5 predicate for both confirmation seeds."""
    gates: dict[str, dict[str, object]] = {}
    per_seed = _registered_seed_mapping(scores.get("per_seed"))
    seed_values = sorted(per_seed)
    for reference in ("f0", "a1", "a3"):
        values = [_score_metric(per_seed[seed], "a2", reference) for seed in seed_values]
        gates[f"each_seed_a2_{reference}_positive"] = {"value": values, "passed": bool(values) and all(_finite(value) and value > 0.0 for value in values)}
        mean = float(np.mean(values)) if len(values) == len(_REGISTERED_CONFIRMATION_SEEDS) and all(_finite(value) for value in values) else float("nan")
        gates[f"two_seed_a2_{reference}_macro"] = {"value": mean, "threshold": 0.02, "passed": _finite(mean) and mean >= 0.02}
    for reference in ("f0", "a1"):
        family_means = _confirmation_family_means(per_seed, reference)
        gates[f"every_two_seed_a2_{reference}_family_delta"] = {
            "value": family_means if family_means is not None else {},
            "threshold": -0.02,
            "passed": family_means is not None and all(
                value >= -0.02 for value in family_means.values()
            ),
        }
    ci_values = _ci_lows(scores)
    gates["every_accuracy_ci_lower_strictly_positive"] = {"value": ci_values, "passed": bool(ci_values) and all(_finite(value) and value > 0.0 for value in ci_values)}
    retention_by_seed = _registered_seed_mapping(retention)
    timing_by_seed = _registered_seed_mapping(timing)
    retention_gates = []
    timing_gates = []
    for seed in _REGISTERED_CONFIRMATION_SEEDS:
        item = retention_by_seed.get(seed, {})
        point = _retention_metric(item, "point") if isinstance(item, Mapping) else float("nan")
        high = _retention_metric(item, "ci_high") if isinstance(item, Mapping) else float("nan")
        retention_gates.append(_finite(point) and point <= 1.02 and _finite(high) and high <= 1.02)
        time_item = timing_by_seed.get(seed, {})
        ratio = _timing_metric(time_item) if isinstance(time_item, Mapping) else float("nan")
        timing_gates.append(_finite(ratio) and 0.90 <= ratio <= 1.10)
    gates["each_seed_a2_f0_ppl"] = {"passed": bool(retention_gates), "value": retention_gates}
    gates["each_seed_a3_a2_timing"] = {"passed": bool(timing_gates), "value": timing_gates}
    quality_passed = all(bool(item["passed"]) for name, item in gates.items() if name != "each_seed_a2_f0_ppl" and name != "each_seed_a3_a2_timing")
    retention_passed = bool(retention_gates) and all(retention_gates)
    timing_passed = bool(timing_gates) and all(timing_gates)
    status, reason = _terminal_reason(quality_passed, retention_passed, timing_passed)
    return {"status": status, "reason": reason, "passed": status == "passed", "gates": gates}


def _validate_score_records(records: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    rows = list(records)
    if not rows:
        raise ValueError("exact score records cannot be empty")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("score records must be mappings")
        identifier = row.get("example_id")
        family = row.get("family")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("score records require unique example IDs")
        if not isinstance(family, str) or not family:
            raise ValueError("score records require a family")
        if row.get("correct") not in (True, False):
            raise ValueError(f"score record {identifier} has invalid correctness")
        seen.add(identifier)
    return rows


def _paired_differences(
    candidate: Mapping[int, Sequence[Mapping[str, object]]],
    reference: Mapping[int, Sequence[Mapping[str, object]]],
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, int]]:
    seeds = _seed_pairs(candidate, reference)
    per_id: dict[str, list[float]] = {}
    families: dict[str, str] = {}
    transitions = {"wrong_to_right": 0, "right_to_wrong": 0}
    for candidate_seed, reference_seed in seeds:
        candidate_rows = {str(row["example_id"]): row for row in candidate[candidate_seed]}
        reference_rows = {str(row["example_id"]): row for row in reference[reference_seed]}
        if set(candidate_rows) != set(reference_rows):
            raise ValueError("paired accuracy ID set mismatch")
        for identifier in sorted(candidate_rows):
            left = candidate_rows[identifier]
            right = reference_rows[identifier]
            if left["family"] != right["family"]:
                raise ValueError(f"family mismatch for {identifier}")
            family = str(left["family"])
            if identifier in families and families[identifier] != family:
                raise ValueError(f"family mismatch for {identifier}")
            families[identifier] = family
            candidate_correct = left["correct"] is True
            reference_correct = right["correct"] is True
            per_id.setdefault(identifier, []).append(float(candidate_correct) - float(reference_correct))
            if not reference_correct and candidate_correct:
                transitions["wrong_to_right"] += 1
            elif reference_correct and not candidate_correct:
                transitions["right_to_wrong"] += 1
    grouped: dict[str, list[float]] = {}
    for identifier, values in per_id.items():
        grouped.setdefault(families[identifier], []).append(float(np.mean(values)))
    return per_id, grouped, transitions


def _seed_pairs(candidate: Mapping[int, object], reference: Mapping[int, object]) -> list[tuple[int, int]]:
    candidate_seeds = sorted(candidate)
    reference_seeds = sorted(reference)
    if len(reference_seeds) == 1:
        return [(seed, reference_seeds[0]) for seed in candidate_seeds]
    if len(candidate_seeds) == 1:
        return [(candidate_seeds[0], seed) for seed in reference_seeds]
    common = sorted(set(candidate_seeds) & set(reference_seeds))
    if not common:
        raise ValueError("arm/seed sets have no common seed")
    return [(seed, seed) for seed in common]


def _validate_common_universe(
    arms: Mapping[str, Mapping[int, Sequence[Mapping[str, object]]]],
) -> tuple[list[str], dict[str, list[str]]]:
    """Validate and return one sorted ID/family universe for every vector.

    The bootstrap is only paired when every supplied arm/seed has exactly the
    same IDs and each ID has the same family label.  Returning the family ID
    lists also fixes the index order used by the single shared sample matrix.
    """
    vectors = [
        (arm, seed, rows)
        for arm, by_seed in arms.items()
        for seed, rows in by_seed.items()
    ]
    if not vectors:
        raise ValueError("accuracy bootstrap requires records")
    reference_rows = vectors[0][2]
    reference = {str(row["example_id"]): str(row["family"]) for row in reference_rows}
    if not reference:
        raise ValueError("accuracy bootstrap requires a non-empty ID universe")
    for arm, seed, rows in vectors:
        current = {str(row["example_id"]): str(row["family"]) for row in rows}
        if set(current) != set(reference):
            raise ValueError(f"common ID set/family universe mismatch for {arm}/{seed}")
        for identifier in sorted(reference):
            if current[identifier] != reference[identifier]:
                raise ValueError(f"common family universe mismatch for {identifier}")
    common_ids = sorted(reference)
    family_ids: dict[str, list[str]] = {}
    for identifier in common_ids:
        family_ids.setdefault(reference[identifier], []).append(identifier)
    if any(not family for family in family_ids):
        raise ValueError("common family universe contains an empty family")
    return common_ids, {family: family_ids[family] for family in sorted(family_ids)}


def _validate_retention_chunks(chunks: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    rows = list(chunks)
    if not rows:
        raise ValueError("retention chunks cannot be empty")
    indices: list[int] = []
    for row in rows:
        index = row.get("chunk_index")
        count = row.get("scored_tokens")
        nll = row.get("nll_sum")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("retention chunk index is invalid")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("retention scored token count must be positive")
        if not _finite(nll) or float(nll) < 0.0:
            raise ValueError("retention NLL must be finite and non-negative")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError("retention chunk indices must be unique")
    return sorted(rows, key=lambda row: int(row["chunk_index"]))


def _retention_ratio(candidate: Sequence[Mapping[str, object]], f0: Sequence[Mapping[str, object]]) -> float:
    candidate_tokens = sum(int(row["scored_tokens"]) for row in candidate)
    f0_tokens = sum(int(row["scored_tokens"]) for row in f0)
    if candidate_tokens <= 0 or f0_tokens <= 0:
        raise ValueError("retention bootstrap requires scored tokens")
    candidate_mean = sum(float(row["nll_sum"]) for row in candidate) / candidate_tokens
    f0_mean = sum(float(row["nll_sum"]) for row in f0) / f0_tokens
    ratio = math.exp(candidate_mean - f0_mean)
    if not _finite(ratio):
        raise ValueError("retention PPL ratio is nonfinite")
    return ratio


def _score_metric(scores: Mapping[str, object], candidate: str, reference: str) -> float:
    direct = f"{candidate.lower()}_minus_{reference.lower()}_macro"
    value = scores.get(direct)
    if _finite(value):
        return float(value)
    comparisons = scores.get("comparisons")
    if isinstance(comparisons, Mapping):
        item = comparisons.get(f"{candidate.upper()}-{reference.upper()}")
        if isinstance(item, Mapping):
            for key in ("point", "macro", "mean"):
                if _finite(item.get(key)):
                    return float(item[key])
    per_seed = scores.get("per_seed")
    if isinstance(per_seed, Mapping):
        values = [_score_metric(item, candidate, reference) for item in per_seed.values() if isinstance(item, Mapping)]
        if values and all(_finite(value) for value in values):
            return float(np.mean(values))
    return float("nan")


def _family_gate(scores: Mapping[str, object], candidate: str, reference: str, lower: float) -> dict[str, object]:
    key = f"{candidate.lower()}_family_deltas_vs_{reference.lower()}"
    values = scores.get(key)
    if values is None and isinstance(scores.get("comparisons"), Mapping):
        item = scores["comparisons"].get(f"{candidate.upper()}-{reference.upper()}")
        values = item.get("family_deltas") if isinstance(item, Mapping) else None
    numeric = _exact_family_values(values)
    passed = numeric is not None and all(value >= lower for value in numeric.values())
    return {"value": numeric, "threshold": lower, "passed": passed}


def _exact_family_values(value: object) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    if set(value) != set(_REGISTERED_FAMILIES) or not all(
        isinstance(name, str) for name in value
    ):
        return None
    if not all(_finite(value.get(name)) for name in _REGISTERED_FAMILIES):
        return None
    return {name: float(value[name]) for name in _REGISTERED_FAMILIES}


def _confirmation_family_means(
    per_seed: Mapping[int, object], reference: str
) -> dict[str, float] | None:
    key = f"a2_family_deltas_vs_{reference}"
    seed_values: list[dict[str, float]] = []
    for seed in _REGISTERED_CONFIRMATION_SEEDS:
        item = per_seed.get(seed)
        if not isinstance(item, Mapping):
            return None
        values = _exact_family_values(item.get(key))
        if values is None:
            return None
        seed_values.append(values)
    return {
        family: float(np.mean([values[family] for values in seed_values]))
        for family in _REGISTERED_FAMILIES
    }


def _retention_metric(retention: Mapping[str, object], kind: str) -> float:
    aliases = {"point": ("point", "ppl_ratio_point"), "ci_high": ("ci_high", "ppl_ratio_ci_high")}[kind]
    for key in aliases:
        if _finite(retention.get(key)):
            return float(retention[key])
    return float("nan")


def _timing_metric(timing: Mapping[str, object]) -> float:
    for key in ("a3_a2_realized_training_time_ratio", "ratio", "a3_over_a2"):
        if _finite(timing.get(key)):
            return float(timing[key])
    return float("nan")


def _ci_lows(scores: Mapping[str, object]) -> list[float]:
    values = scores.get("accuracy_ci_lows")
    if isinstance(values, Mapping):
        if set(values) != set(_REGISTERED_COMPARISONS):
            return []
        if not all(_finite(values.get(name)) for name in _REGISTERED_COMPARISONS):
            return []
        return [float(values[name]) for name in _REGISTERED_COMPARISONS]
    values = scores.get("ci_lows")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if len(values) != len(_REGISTERED_COMPARISONS) or not all(_finite(value) for value in values):
            return []
        return [float(value) for value in values]
    comparisons = scores.get("comparisons")
    if isinstance(comparisons, Mapping):
        if set(comparisons) != {name.upper().replace("-", "-") for name in _REGISTERED_COMPARISONS}:
            return []
        result: list[float] = []
        for name in _REGISTERED_COMPARISONS:
            item = comparisons.get(name.upper())
            if not isinstance(item, Mapping) or not _finite(item.get("ci_low")):
                return []
            result.append(float(item["ci_low"]))
        return result
    return []


def _registered_seed_mapping(value: object) -> dict[int, object]:
    """Normalize an exact `{47, 48}` mapping; malformed maps become empty."""
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[int, object] = {}
    for raw_seed, item in value.items():
        if isinstance(raw_seed, bool):
            return {}
        if isinstance(raw_seed, int):
            seed = raw_seed
        elif isinstance(raw_seed, str) and raw_seed in {str(item) for item in _REGISTERED_CONFIRMATION_SEEDS}:
            seed = int(raw_seed)
        else:
            return {}
        if seed in normalized:
            return {}
        normalized[seed] = item
    if set(normalized) != set(_REGISTERED_CONFIRMATION_SEEDS):
        return {}
    return normalized


def _terminal_reason(quality: bool, retention: bool, timing: bool) -> tuple[str, str]:
    if not quality:
        return "stopped_scientific", "scientific_quality_gate_failed"
    if not retention:
        return "stopped_scientific", "retention_uncertain"
    if not timing:
        return "stopped_scientific", "timing_inconclusive"
    return "passed", "passed"


def _required_number(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not _finite(value):
        raise ValueError(f"{key} must be finite")
    return float(value)


def _require_replicates(replicates: int) -> None:
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates != _REPLICATES:
        raise ValueError("production paired bootstraps require exactly 10,000 replicates")


def _finite(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(float(value))
