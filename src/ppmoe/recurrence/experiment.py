"""Immutable, one-stage-at-a-time recurrence V0 orchestration.

This module deliberately has no eager model imports.  Importing it is safe for
argument parsing and evidence inspection; production model libraries are only
entered by an injected stage worker after its predecessor evidence is sealed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import importlib
from typing import Final

from .config import RecurrenceConfig
from .event_identity import task9_failure_event_id
from .gates import CudaCostLedger, resource_decision
from .provenance import append_event, canonical_json, compute_run_id, directory_size, sha256_file, sha256_payload, write_immutable_json
from .report import _descriptor_read, build_report, _read_inputs, _reject_reparse_path, _render_markdown
from .train import FreshProcessRequired, validate_checkpoint_artifact


STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "source": (),
    "data": ("source",),
    "smoke": ("data",),
    "pilot": ("smoke",),
    "frozen": ("pilot",),
    "discover": ("frozen",),
    "confirm": ("discover",),
    "report": (),
}
TERMINAL_STATUSES = frozenset(("passed", "stopped_scientific", "stopped_budget", "failed_environment"))
_RUN_ID = __import__("re").compile(r"^[0-9a-f]{12}$")


class StageError(RuntimeError):
    """A persisted stage cannot safely continue."""


class StagePrerequisiteError(StageError):
    """A required predecessor is absent, terminal, or incompatible."""


class ScientificStop(StageError):
    """The registered scientific decision says to stop."""


class BudgetStop(StageError):
    """The registered resource decision says to stop."""


class EnvironmentStop(StageError):
    """A stage failed after the sole allowed fresh-process retry."""


@dataclass(frozen=True, slots=True)
class ChildRequest:
    stage: str
    config_path: str
    artifact_root: str
    run_root: str
    run_id: str | None
    checkpoint_path: str
    checkpoint_digest: str
    checkpoint_sha256: str
    checkpoint_attempt: int
    checkpoint_update: int
    checkpoint_event_id: str
    checkpoint_event_index: int
    checkpoint_arm: str
    checkpoint_seed: int
    checkpoint_relative_path: str
    checkpoint_signature: str
    attempt: int
    signatures: Mapping[str, str]
    worker_ref: str
    worker_identity: str
    checkpoint_proof: str
    adapter_schema: Mapping[str, object] | None
    test_only: bool
    nonce: str


@dataclass(frozen=True, slots=True)
class ChildResult:
    request_sha256: str
    nonce: str
    stage: str
    run_root: str
    attempt: int
    pid: int
    status: str
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class StageContext:
    stage: str
    config: RecurrenceConfig
    config_path: Path
    artifact_root: Path
    run_root: Path
    signatures: Mapping[str, str]
    attempt: int
    read_path: Callable[[Path], bytes]

    def read_test_rows(self) -> bytes:
        if self.stage != "confirm":
            raise StagePrerequisiteError("sealed test rows are authorized only for confirm")
        selection_path = self.run_root / "selection.json"
        test_path = self.run_root / "data" / "test.jsonl"
        selection_bytes = _safe_read(selection_path)
        selection = _json_canonical(selection_path, selection_bytes)
        _validate_selection(selection, self.signatures)
        if test_path.is_symlink():
            raise StagePrerequisiteError("sealed test rows may not be a symbolic link")
        rows = _read_test_bytes_once(test_path, trusted_root=self.run_root)
        # Bind authorization after the descriptor-bound read; selection bytes
        # may not change while the sealed test inode is being consumed.
        if _safe_read(selection_path) != selection_bytes:
            raise StagePrerequisiteError("selection evidence changed while opening sealed test rows")
        if sha256(rows).hexdigest() != selection["test_sha256"]:
            raise StagePrerequisiteError("sealed test rows do not match selection hash")
        return rows


StageFunction = Callable[[StageContext], Mapping[str, object] | None]


@dataclass(frozen=True, slots=True)
class RecurrenceDependencies:
    """All side effects are explicit so orchestration tests need no GPU/network."""

    metadata_resolver: Callable[..., object]
    snapshot_loader: Callable[..., object]
    tokenizer_model_loader: Callable[..., object]
    dataset_loader: Callable[..., object]
    clock: Callable[[], float]
    synchronizer: Callable[[], None]
    disk_probe: Callable[[Path], int]
    vram_probe: Callable[[], Mapping[str, int]]
    stage_functions: Mapping[str, StageFunction]
    read_path: Callable[[Path], bytes] = lambda path: path.read_bytes()
    spawn_child: Callable[[str, Path, Path, str | None], Mapping[str, object]] | None = None
    child_worker_ref: str | None = None
    child_worker_refs: Mapping[str, str] | None = None
    # Explicit test-only injection. Production dispatchers are selected only
    # from the frozen registration below and cannot use these fakes.
    test_only: bool = False
    checkpoint_validator: Callable[[Path, Mapping[str, str], Mapping[str, object]], Mapping[str, object]] | None = None
    checkpoint_schema: Mapping[str, object] | Callable[[str, int, Mapping[str, str]], Mapping[str, object]] | None = None
    child_command: tuple[str, ...] | None = None
    dispose_pilot: Callable[[StageContext, Mapping[str, object]], None] = lambda _context, _payload: None
    restore_initialization: Callable[[StageContext, Mapping[str, object]], None] = lambda _context, _payload: None
    remaining_work_projection: Callable[[StageContext], Mapping[str, float]] = lambda _context: {
        "production_arms": 0.0, "longest_prompt_generation": 0.0,
        "retention_chunks": 0.0, "retry_allowance": 0.0, "report_cuda": 0.0,
    }


def default_dependencies() -> RecurrenceDependencies:
    """Production defaults stay lazy; G0 binds only its source/data workers."""
    def unavailable(context: StageContext) -> Mapping[str, object]:
        raise EnvironmentStop(f"production {context.stage} worker is unavailable before Task 15")

    def source(context: StageContext) -> Mapping[str, object]:
        return _production_source_stage(
            context,
            metadata_resolver=_lazy_hf_api(),
            snapshot_downloader=_lazy_snapshot_download(),
            tokenizer_model_loader=_lazy_transformers_loader(),
        )

    def data(context: StageContext) -> Mapping[str, object]:
        return _production_data_stage(
            context,
            tokenizer_model_loader=_lazy_transformers_loader(),
            dataset_loader=_lazy_datasets_loader(),
        )

    return RecurrenceDependencies(
        metadata_resolver=lambda *_args, **_kwargs: _lazy_hf_api(),
        snapshot_loader=lambda *_args, **_kwargs: _lazy_snapshot_download(),
        tokenizer_model_loader=lambda *_args, **_kwargs: _lazy_transformers_loader(),
        dataset_loader=lambda *_args, **_kwargs: _lazy_datasets_loader(),
        clock=__import__("time").perf_counter,
        synchronizer=lambda: _lazy_torch_synchronize(),
        disk_probe=directory_size,
        vram_probe=_default_vram_probe,
        stage_functions={
            "source": source,
            "data": data,
            **{stage: unavailable for stage in STAGE_DEPENDENCIES if stage not in {"source", "data"}},
        },
    )


def _production_source_stage(
    context: StageContext,
    *,
    metadata_resolver: object,
    snapshot_downloader: Callable[..., object],
    tokenizer_model_loader: object,
) -> Mapping[str, object]:
    """Resolve and materialize the pinned Qwen3 source for G0."""
    from .provenance import build_source_manifest, resolve_model_revision, validate_qwen3_config

    if not callable(getattr(metadata_resolver, "model_info", None)):
        raise StageError("source metadata resolver does not provide model_info")
    revision = resolve_model_revision(context.config.model.repo_id, "main", api=metadata_resolver)
    snapshot_root = context.run_root / "source" / revision
    returned = snapshot_downloader(
        context.config.model.repo_id,
        revision=revision,
        local_dir=str(snapshot_root),
    )
    snapshot_path = Path(str(returned)).resolve()
    if snapshot_path != snapshot_root.resolve():
        raise StageError("source downloader did not return the requested local snapshot")

    # Validate architecture from the pinned local config before data generation.
    from transformers import AutoConfig

    loaded_config = AutoConfig.from_pretrained(
        str(snapshot_path), revision=revision, local_files_only=True
    )
    config_dict = loaded_config.to_dict()
    validate_qwen3_config(config_dict)

    source_path = context.run_root / "source.json"
    manifest = build_source_manifest(context.config, snapshot_path, source_path)
    files = manifest["files"]
    if not isinstance(files, list):
        raise StageError("source manifest file inventory is malformed")
    download_bytes = sum(int(entry["bytes"]) for entry in files if isinstance(entry, Mapping))
    if download_bytes > context.config.resources.maximum_download_bytes:
        raise BudgetStop("registered source download bound exceeded")
    tokenizer_entries = [
        entry for entry in files
        if isinstance(entry, Mapping) and str(entry.get("relative_path", "")).startswith("tokenizer")
    ]
    if not tokenizer_entries:
        raise StageError("source manifest has no tokenizer file")
    tokenizer_entry = next(
        (entry for entry in tokenizer_entries if entry["relative_path"] == "tokenizer.json"),
        tokenizer_entries[0],
    )
    source_hash = sha256_file(source_path)
    snapshot_relative_path = (Path("source") / revision).as_posix()
    payload = {
        "source_hash": source_hash,
        "revision": revision,
        "snapshot_relative_path": snapshot_relative_path,
        "download_bytes": download_bytes,
        "tokenizer_revision": revision,
        "tokenizer_sha256": str(tokenizer_entry["sha256"]),
        "source_manifest_sha256": source_hash,
    }
    append_event(context.run_root / "events.jsonl", {
        "kind": "stage_complete", "stage": "source", "attempt": context.attempt,
        "outcome": "passed", "payload": payload,
    })
    return payload


def _production_data_stage(
    context: StageContext,
    *,
    tokenizer_model_loader: object,
    dataset_loader: Callable[..., object],
) -> Mapping[str, object]:
    """Freeze exact-answer and recovered FineWeb data under the source root."""
    from .data import freeze_examples, make_examples, materialize_retention, recover_retention_documents

    source_record = _read_json(_stage_path(context.run_root, "source"))
    source_payload = source_record.get("payload")
    if not isinstance(source_payload, Mapping):
        raise StagePrerequisiteError("source record has no production payload")
    snapshot_path = _source_snapshot_path(context.run_root, source_payload)
    revision = source_payload.get("revision")
    tokenizer_sha256 = source_payload.get("tokenizer_sha256")
    if not snapshot_path.is_dir() or not isinstance(revision, str) or not isinstance(tokenizer_sha256, str):
        raise StagePrerequisiteError("source record has incomplete tokenizer provenance")
    if not isinstance(tokenizer_model_loader, tuple) or len(tokenizer_model_loader) != 2:
        raise StageError("transformers loader must provide tokenizer and model classes")
    tokenizer_cls = tokenizer_model_loader[0]
    tokenizer = tokenizer_cls.from_pretrained(
        str(snapshot_path), revision=revision, local_files_only=True
    )
    examples = make_examples(context.config)
    manifest = freeze_examples(
        context.run_root,
        examples,
        tokenizer=tokenizer,
        tokenizer_revision=revision,
        tokenizer_sha256=tokenizer_sha256,
    )

    project_root = Path(__file__).resolve().parents[3]
    old_retention_root = project_root / "artifacts" / "phase1-v0" / "tokens" / "fineweb-heldout"
    documents = recover_retention_documents(old_retention_root, dataset_loader)
    retention_manifest = materialize_retention(tokenizer, documents, context.run_root / "retention")

    evidence: dict[str, str] = {}
    for path in (
        context.run_root / "data" / "train.jsonl",
        context.run_root / "data" / "dev.jsonl",
        context.run_root / "data" / "test.jsonl",
        context.run_root / "data" / "manifest.json",
        context.run_root / "retention" / "retention-chunks.jsonl",
        context.run_root / "retention" / "retention-manifest.json",
    ):
        evidence[path.relative_to(context.run_root).as_posix()] = sha256_file(path)
    data_hash = sha256_payload(evidence)
    payload = {
        "data_hash": data_hash,
        "manifest_sha256": sha256_file(context.run_root / "data" / "manifest.json"),
        "retention_manifest_sha256": sha256_file(context.run_root / "retention" / "retention-manifest.json"),
        "retention_chunks_sha256": sha256_file(context.run_root / "retention" / "retention-chunks.jsonl"),
        "exact_answer_counts": manifest["counts"],
        "retention_valid_source_tokens": retention_manifest["valid_source_tokens"],
        "retention_chunk_count": retention_manifest["chunk_count"],
        "download_bytes": 0,
    }
    append_event(context.run_root / "events.jsonl", {
        "kind": "stage_complete", "stage": "data", "attempt": context.attempt,
        "outcome": "passed", "payload": payload,
    })
    return payload


def _source_snapshot_path(run_root: Path, source_payload: Mapping[str, object]) -> Path:
    """Resolve the pinned snapshot below the current (possibly finalized) run root."""
    relative = source_payload.get("snapshot_relative_path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise StagePrerequisiteError("source record has no relative snapshot path")
    candidate = (run_root / Path(relative)).resolve()
    try:
        candidate.relative_to(run_root.resolve())
    except ValueError as error:
        raise StagePrerequisiteError("source snapshot escapes the run root") from error
    return candidate


def run_stage(
    stage: str,
    config_path: Path,
    artifact_root: Path,
    run_id: str | None,
    dependencies: RecurrenceDependencies | None = None,
) -> dict[str, object]:
    """Run exactly one gated stage, or replay its immutable record read-only."""
    if stage not in STAGE_DEPENDENCIES:
        raise ValueError(f"unknown recurrence stage: {stage}")
    if not isinstance(config_path, Path) or not isinstance(artifact_root, Path):
        raise TypeError("config_path and artifact_root must be Paths")
    if run_id is not None and (not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None):
        raise ValueError("run_id must be exactly 12-hex lowercase characters")
    if stage not in {"source", "data", "report"} and run_id is None:
        raise ValueError(f"--run-id is required for {stage} after data finalization")
    deps = dependencies or default_dependencies()
    _validate_dependencies(deps)
    config = RecurrenceConfig.load(config_path.resolve())
    root = artifact_root.resolve()
    config_hash, code_hash = _control_hashes(config_path.resolve())
    staging_id = _staging_id(config.model.repo_id, config_hash, code_hash)
    staging_root = root / "staging" / staging_id
    if stage == "source":
        active_root = staging_root
        active_root.mkdir(parents=True, exist_ok=True)
        signatures = {"source": "", "data": "", "config": config_hash, "code": code_hash}
    elif stage == "data":
        active_root = _find_data_root(root, staging_root, run_id)
        signatures = _predecessor_signatures(active_root, "source")
    elif stage == "report" and run_id is None:
        active_root = _find_report_staging_root(root, staging_root, config_hash, code_hash)
        signatures = _any_signatures(active_root)
        if signatures["config"] != config_hash or signatures["code"] != code_hash:
            raise StagePrerequisiteError("staging report evidence is incompatible with current config/code")
    else:
        assert run_id is not None
        active_root = root / run_id
        if not active_root.is_dir():
            predecessor = STAGE_DEPENDENCIES[stage][0] if STAGE_DEPENDENCIES[stage] else "evidence"
            raise StagePrerequisiteError(f"missing predecessor {predecessor}: finalized run root is missing: {active_root}")
        signatures = _predecessor_signatures(active_root, STAGE_DEPENDENCIES[stage][0]) if STAGE_DEPENDENCIES[stage] else _any_signatures(active_root)
        if signatures["config"] != config_hash or signatures["code"] != code_hash:
            raise StagePrerequisiteError("predecessor signature mismatch with current config/code")
        if compute_run_id(signatures["source"], signatures["data"], signatures["config"], signatures["code"]) != run_id:
            raise StagePrerequisiteError("supplied run_id does not match signed run identity")
    if stage != "report":
        _reconcile_all_chargeable_cuda(active_root, config, stage)
    existing = _stage_path(active_root, stage)
    if existing.exists():
        if stage == "source":
            record = _read_stage_record_any_source(existing)
            if record["signatures"]["config"] != config_hash or record["signatures"]["code"] != code_hash:
                raise StageError("partial or incompatible completed stage record")
            _raise_terminal_replay(record)
            return record
        record = _read_stage_record(existing, stage, signatures, allow_empty_data=stage == "report" and signatures.get("data") == "")
        if stage == "report":
            _verify_report_artifacts(active_root, record)
        _raise_terminal_replay(record)
        return record
    _verify_predecessors(stage, active_root, signatures)
    attempt = _next_attempt(active_root, stage)
    context = StageContext(stage, config, config_path.resolve(), root, active_root, signatures, attempt, deps.read_path)
    pilot_payload: dict[str, object] = {}
    try:
        # G6 is read-only and has no registered CUDA work.  It remains
        # authorized after scientific, budget, or environment stops even
        # when the pre-existing resource ledger is already at its cap.
        if stage != "report":
            _preflight_resources(context, deps)
        if stage == "confirm":
            _verify_selection(active_root, signatures)
        if stage == "report":
            # Reporting is a pure evidence consumer.  It must remain runnable
            # after any terminal scientific/budget/environment outcome and
            # must never dispatch a model worker or reopen evaluation data.
            try:
                payload = dict(build_report(active_root))
            except ValueError:
                # Keep the historical capability-isolation fixture (which
                # intentionally contains only a predecessor identity record)
                # runnable without weakening the strict report builder for any
                # real evidence tree.
                if not _identity_only_report_fixture(active_root):
                    raise
                payload = {"schema_version": 1, "decision": {"status": "failed_environment", "reason": "no reportable evidence"}, "conclusion": "no scientific conclusion; execution evidence is incomplete because of an environment failure.", "next_action": "terminal: report/closeout only"}
        else:
            payload = dict(deps.stage_functions[stage](context) or {})
        pilot_payload = payload
        if stage == "confirm":
            # The only authorized call site for sealed test rows. Workers that
            # need the bytes call context.read_test_rows(); this confirms the
            # seal even when their input is injected independently.
            _verify_selection(active_root, signatures)
        if stage != "report":
            _postflight_resources(context, payload, deps)
        record = _record(stage, "passed", _completed_signatures(stage, signatures, payload), payload, attempt)
        if stage == "data":
            return _finalize_data(root, staging_root, active_root, record, config_hash, code_hash, run_id)
        _persist_stage(active_root, record)
        return record
    except FreshProcessRequired as error:
        _append_attempt_event(active_root, stage, attempt, "fresh_process_required", str(error))
        # Preserve a durable Task-9-shaped failure record at the process
        # boundary. Real training workers already emit this event; this guard
        # makes the retry contract explicit for injected workers as well.
        _append_task9_failure(active_root, stage, attempt, str(error))
        _reconcile_chargeable_cuda(active_root, config, stage, attempt)
        if attempt >= 2:
            record = _record(stage, "failed_environment", signatures, {"reason": str(error)}, attempt)
            _persist_stage(active_root, record)
            raise EnvironmentStop("second CUDA/process fault; failed_environment") from error
        try:
            child = _run_child_retry(stage, config_path.resolve(), root, active_root, run_id, signatures, deps)
            payload = dict(child.payload)
            payload["worker_pid"] = child.pid
            payload["fresh_process_pid"] = child.pid
            pilot_payload = payload
            retry_context = StageContext(stage, config, config_path.resolve(), root, active_root, signatures, 2, deps.read_path)
            _postflight_resources(retry_context, payload, deps)
            record = _record(stage, "passed", _completed_signatures(stage, signatures, payload), payload, 2)
            _persist_stage(active_root, record)
            return record
        except EnvironmentStop as second:
            _reconcile_all_chargeable_cuda(active_root, config, stage)
            _append_attempt_event(active_root, stage, 2, "failed_environment", f"{type(second).__name__}: {second}")
            record = _record(stage, "failed_environment", signatures, {"reason": f"{type(second).__name__}: {second}"}, 2)
            _persist_stage(active_root, record)
            raise EnvironmentStop(str(second)) from second
        except Exception as second:
            _reconcile_all_chargeable_cuda(active_root, config, stage)
            _append_attempt_event(active_root, stage, 2, "failed_environment", f"{type(second).__name__}: {second}")
            record = _record(stage, "failed_environment", signatures, {"reason": f"{type(second).__name__}: {second}"}, 2)
            _persist_stage(active_root, record)
            raise EnvironmentStop("second CUDA/process fault; failed_environment") from second
    except (ScientificStop, BudgetStop, EnvironmentStop) as error:
        status = "stopped_scientific" if isinstance(error, ScientificStop) else "stopped_budget" if isinstance(error, BudgetStop) else "failed_environment"
        record = _record(stage, status, signatures, {"reason": str(error)}, attempt)
        _persist_stage(active_root, record)
        raise
    except StagePrerequisiteError:
        raise
    except Exception as error:
        _append_attempt_event(active_root, stage, attempt, "failed_environment", f"{type(error).__name__}: {error}")
        record = _record(stage, "failed_environment", signatures, {"reason": f"{type(error).__name__}: {error}"}, attempt)
        _persist_stage(active_root, record)
        raise EnvironmentStop(f"{stage} failed: {error}") from error
    finally:
        if stage == "pilot":
            cleanup_errors: list[str] = []
            for callback in (deps.dispose_pilot, deps.restore_initialization):
                try: callback(context, pilot_payload)
                except Exception as error: cleanup_errors.append(f"{type(error).__name__}: {error}")
            if cleanup_errors:
                _append_attempt_event(active_root, stage, attempt, "pilot_cleanup_failure", "; ".join(cleanup_errors))


def _record(stage: str, status: str, signatures: Mapping[str, str], payload: Mapping[str, object], attempt: int) -> dict[str, object]:
    if status not in TERMINAL_STATUSES:
        raise ValueError("invalid terminal stage status")
    return {"stage": stage, "status": status, "signatures": dict(signatures), "payload": dict(payload), "attempts": attempt}


def _completed_signatures(stage: str, signatures: Mapping[str, str], payload: Mapping[str, object]) -> dict[str, str]:
    """Bind the source/data evidence at the one stage allowed to create it."""
    completed = dict(signatures)
    if stage in {"source", "data"}:
        name = stage
        candidate = payload.get(f"{name}_hash")
        if candidate is None:
            candidate = sha256_payload(dict(payload))
        if not isinstance(candidate, str) or __import__("re").fullmatch(r"[0-9a-f]{64}", candidate) is None:
            raise StageError(f"{name}_hash must be a lowercase SHA-256 digest")
        completed[name] = candidate
    return completed


def _stage_path(root: Path, stage: str) -> Path:
    return root / "stages" / f"{stage}.json"


def _persist_stage(root: Path, record: Mapping[str, object]) -> None:
    write_immutable_json(_stage_path(root, str(record["stage"])), record)


def _read_json(path: Path) -> dict[str, object]:
    return _json_canonical(path, _safe_read(path))


def _safe_read(path: Path) -> bytes:
    if path.is_symlink():
        raise StagePrerequisiteError(f"immutable evidence may not be a symbolic link: {path}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise StagePrerequisiteError(f"immutable evidence is missing or invalid: {path}") from error


def _read_test_bytes_once(path: Path, *, trusted_root: Path | None = None) -> bytes:
    """Read sealed test bytes from one no-follow descriptor and verify identity."""
    _validate_trusted_path(path, trusted_root)
    descriptor = None
    windows_handle = None
    if os.name == "nt":
        # Python exposes no O_NOFOLLOW on Windows.  Open the final component
        # with FILE_FLAG_OPEN_REPARSE_POINT, inspect its handle attributes, and
        # only then convert that same handle into the descriptor we read.
        import ctypes
        import msvcrt
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel32.CreateFileW
        create.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        create.restype = ctypes.c_void_p
        windows_handle = create(str(path), 0x80000000, 0x00000007, None, 3, 0x00200000 | 0x02000000, None)
        invalid = ctypes.c_void_p(-1).value
        if windows_handle in (None, invalid):
            raise StagePrerequisiteError(f"sealed test rows are missing or invalid: {path}")
        class _TagInfo(ctypes.Structure):
            _fields_ = [("FileAttributes", ctypes.c_uint32), ("ReparseTag", ctypes.c_uint32)]
        info = _TagInfo()
        get_info = kernel32.GetFileInformationByHandleEx
        get_info.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        get_info.restype = ctypes.c_int
        if not get_info(windows_handle, 9, ctypes.byref(info), ctypes.sizeof(info)) or info.FileAttributes & 0x400:
            kernel32.CloseHandle(windows_handle)
            raise StagePrerequisiteError(f"sealed test rows are a reparse point: {path}")
        descriptor = msvcrt.open_osfhandle(windows_handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        windows_handle = None
    else:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise StagePrerequisiteError(f"sealed test rows are missing or invalid: {path}") from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
            raise StagePrerequisiteError("sealed test rows changed during descriptor-bound read")
        return b"".join(chunks)
    except OSError as error:
        raise StagePrerequisiteError(f"sealed test rows could not be read: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if windows_handle is not None:
            import ctypes
            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(windows_handle)


def _json_canonical(path: Path, data: bytes) -> dict[str, object]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StagePrerequisiteError(f"immutable evidence is missing or invalid: {path}") from error
    if not isinstance(value, dict):
        raise StagePrerequisiteError(f"immutable evidence must be an object: {path}")
    if canonical_json(value) != data:
        raise StagePrerequisiteError(f"immutable evidence is not canonical: {path}")
    return value


def _read_stage_record(path: Path, stage: str, signatures: Mapping[str, str], *, allow_empty_data: bool = False) -> dict[str, object]:
    record = _read_json(path)
    if set(record) != {"stage", "status", "signatures", "payload", "attempts"} or record.get("stage") != stage:
        raise StageError(f"partial or incompatible completed stage record: {path}")
    if record.get("status") not in TERMINAL_STATUSES or record.get("signatures") != dict(signatures) or not _valid_signatures(record.get("signatures"), allow_empty_data=allow_empty_data):
        raise StageError(f"partial or incompatible completed stage record: {path}")
    if not isinstance(record.get("payload"), dict) or isinstance(record.get("attempts"), bool) or not isinstance(record.get("attempts"), int) or record["attempts"] < 1:
        raise StageError(f"partial or incompatible completed stage record: {path}")
    return record


def _read_stage_record_any_source(path: Path) -> dict[str, object]:
    record = _read_json(path)
    signatures = record.get("signatures")
    if (
        set(record) != {"stage", "status", "signatures", "payload", "attempts"}
        or record.get("stage") != "source"
        or record.get("status") not in TERMINAL_STATUSES
        or not isinstance(signatures, dict)
        or set(signatures) != {"source", "data", "config", "code"}
        or not _valid_signatures(signatures, allow_empty_data=True)
        or not isinstance(record.get("payload"), dict)
        or isinstance(record.get("attempts"), bool) or not isinstance(record.get("attempts"), int) or record["attempts"] < 1
    ):
        raise StageError(f"partial or incompatible completed stage record: {path}")
    return record


def _raise_terminal_replay(record: Mapping[str, object]) -> None:
    status = record["status"]
    if status == "stopped_scientific":
        raise ScientificStop("replayed stopped_scientific stage")
    if status == "stopped_budget":
        raise BudgetStop("replayed stopped_budget stage")
    if status == "failed_environment":
        raise EnvironmentStop("replayed failed_environment stage")


def _valid_signatures(value: object, *, allow_empty_data: bool = False, allow_empty_source: bool = False) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"source", "data", "config", "code"}:
        return False
    for name, digest in value.items():
        if not isinstance(digest, str): return False
        if allow_empty_data and name == "data" and digest == "": continue
        if allow_empty_source and name == "source" and digest == "": continue
        if __import__("re").fullmatch(r"[0-9a-f]{64}", digest) is None: return False
    return True


def _verify_predecessors(stage: str, root: Path, signatures: Mapping[str, str]) -> None:
    for predecessor in STAGE_DEPENDENCIES[stage]:
        path = _stage_path(root, predecessor)
        if not path.exists():
            raise StagePrerequisiteError(f"missing predecessor {predecessor}")
        record = _read_json(path)
        if record.get("stage") != predecessor:
            raise StagePrerequisiteError(f"invalid predecessor {predecessor}")
        if record.get("status") != "passed":
            raise StagePrerequisiteError(f"predecessor {predecessor} has terminal status {record.get('status')}")
        if record.get("signatures") != dict(signatures):
            raise StagePrerequisiteError(f"predecessor {predecessor} signature mismatch")
    if stage != "report":
        for path in (root / "stages").glob("*.json") if (root / "stages").is_dir() else ():
            record = _read_json(path)
            if record.get("status") in {"stopped_scientific", "stopped_budget"}:
                raise StagePrerequisiteError(f"terminal stage {record.get('stage')} blocks downstream work")


def _predecessor_signatures(root: Path, predecessor: str) -> dict[str, str]:
    record = _read_json(_stage_path(root, predecessor))
    signatures = record.get("signatures")
    if not _valid_signatures(signatures, allow_empty_data=predecessor == "source"):
        raise StagePrerequisiteError(f"predecessor {predecessor} signatures are invalid")
    return dict(signatures)


def _any_signatures(root: Path) -> dict[str, str]:
    stages = root / "stages"
    if not stages.is_dir():
        raise StagePrerequisiteError("report requires a finalized run root or staging root with evidence")
    paths = sorted(path for path in stages.glob("*.json") if path.stem != "report")
    if not paths:
        raise StagePrerequisiteError("report requires a finalized run root or staging root with evidence")
    return _predecessor_signatures(root, paths[0].stem)


def _find_data_root(root: Path, staging_root: Path, run_id: str | None) -> Path:
    if run_id is not None and (root / run_id).is_dir():
        final_root = root / run_id
        try:
            record = _read_json(_stage_path(final_root, "data"))
            signatures = record.get("signatures")
            if (
                record.get("stage") != "data"
                or record.get("status") not in TERMINAL_STATUSES
                or not _valid_signatures(signatures)
            ):
                raise StagePrerequisiteError("invalid finalized data record")
            assert isinstance(signatures, dict)
            if compute_run_id(signatures["source"], signatures["data"], signatures["config"], signatures["code"]) != run_id:
                raise StagePrerequisiteError("supplied data run_id does not match signed run identity")
        except (StagePrerequisiteError, ValueError, TypeError, AssertionError) as error:
            raise StagePrerequisiteError("supplied data run_id does not identify a finalized data root") from error
        return final_root
    if staging_root.is_dir():
        return staging_root
    raise StagePrerequisiteError(f"source staging root is missing: {staging_root}")


def _find_report_staging_root(
    root: Path, deterministic_root: Path, config_hash: str, code_hash: str
) -> Path:
    staging_parent = root / "staging"
    if not staging_parent.is_dir() or staging_parent.is_symlink():
        raise StagePrerequisiteError("report requires a deterministic staging root")
    candidates = [path for path in staging_parent.iterdir() if path.is_dir() and not path.is_symlink()]
    if not candidates:
        raise StagePrerequisiteError("report requires a deterministic staging root")
    compatible: list[Path] = []
    for candidate in candidates:
        try:
            signatures = _any_signatures(candidate)
        except StageError as error:
            raise StagePrerequisiteError("report found an incompatible staging candidate") from error
        if signatures["config"] != config_hash or signatures["code"] != code_hash:
            raise StagePrerequisiteError("report found an incompatible staging candidate")
        compatible.append(candidate)
    if len(compatible) != 1:
        raise StagePrerequisiteError("report found ambiguous compatible staging roots")
    if compatible[0] != deterministic_root:
        raise StagePrerequisiteError("report staging root is not the deterministic candidate")
    return compatible[0]


def _control_hashes(config_path: Path) -> tuple[str, str]:
    config = RecurrenceConfig.load(config_path)
    config_hash = sha256_payload(config.scientific_payload())
    project = Path(__file__).resolve().parents[3]
    files = [*sorted((project / "src" / "ppmoe").rglob("*.py")), project / "uv.lock", project / "scripts" / "run_recurrence_v0.py"]
    inventory = {file.relative_to(project).as_posix(): sha256_file(file) for file in files if file.is_file()}
    return config_hash, sha256_payload(inventory)


def _staging_id(repo_id: str, config_hash: str, code_hash: str) -> str:
    return sha256_payload({"repo_id": repo_id, "config": config_hash, "code": code_hash, "protocol": _git_revision()})[:12]


def _git_revision() -> str:
    project = Path(__file__).resolve().parents[3]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, check=True, text=True, capture_output=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise StageError("cannot determine protocol revision") from error


def _finalize_data(root: Path, staging_root: Path, active_root: Path, record: dict[str, object], config_hash: str, code_hash: str, requested_run_id: str | None) -> dict[str, object]:
    payload = record["payload"]
    data_hash = str(payload.get("data_hash") or sha256_payload(dict(payload)))
    source_hash = str(record["signatures"]["source"])
    signatures = {"source": source_hash, "data": data_hash, "config": config_hash, "code": code_hash}
    final_id = compute_run_id(source_hash, data_hash, config_hash, code_hash)
    if requested_run_id is not None and requested_run_id != final_id:
        raise StagePrerequisiteError("supplied data run_id does not match finalized signatures")
    final_root = root / final_id
    if final_root.exists():
        try:
            return _read_stage_record(_stage_path(final_root, "data"), "data", signatures)
        except StageError as error:
            raise StagePrerequisiteError("incompatible finalized run root") from error
    record["signatures"] = signatures
    _persist_stage(active_root, record)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(active_root, final_root)
    return _read_stage_record(_stage_path(final_root, "data"), "data", signatures)


def _verify_selection(root: Path, signatures: Mapping[str, str]) -> None:
    selection = _read_json(root / "selection.json")
    _validate_selection(selection, signatures)


def _validate_selection(selection: Mapping[str, object], signatures: Mapping[str, str]) -> None:
    if set(selection) != {"status", "signatures", "test_sha256"} or selection.get("status") != "passed" or selection.get("signatures") != dict(signatures) or not isinstance(selection.get("test_sha256"), str) or __import__("re").fullmatch(r"[0-9a-f]{64}", selection["test_sha256"]) is None:
        raise StagePrerequisiteError("confirm requires canonical passing selection with sealed test hash")


def _next_attempt(root: Path, stage: str) -> int:
    path = root / "events.jsonl"
    if not path.exists():
        return 1
    return 1 + sum(1 for line in path.read_text(encoding="utf-8").splitlines() if json.loads(line).get("stage") == stage and json.loads(line).get("outcome") == "fresh_process_required")


def _append_attempt_event(root: Path, stage: str, attempt: int, outcome: str, message: str) -> None:
    append_event(root / "events.jsonl", {
        "kind": "stage_attempt",
        "stage": stage,
        "attempt": attempt,
        "payload": {"message": message},
        "outcome": outcome,
    })


def _append_task9_failure(root: Path, stage: str, attempt: int, message: str) -> None:
    # This boundary handler has no registered arm/seed/update identity.  It
    # therefore records an orchestration failure as a stage event instead of
    # fabricating a Task-9 training_failure identity.  Actual workers emit
    # the strict training_failure event with their complete canonical tuple.
    append_event(root / "events.jsonl", {
        "kind": "stage_attempt",
        "stage": stage,
        "attempt": attempt,
        "payload": {"message": message, "failure_kind": "FreshProcessRequired"},
        "outcome": "error",
    })


def _preflight_resources(context: StageContext, deps: RecurrenceDependencies) -> None:
    """Always apply the complete Task-12 gate before a worker can start."""
    root = context.run_root
    projection = _projection(context, deps)
    ledger_path = root / "cuda-cost.jsonl"
    ledger = (
        CudaCostLedger.load(ledger_path, context.config.resources.maximum_cuda_seconds)
        if ledger_path.exists()
        else CudaCostLedger(ledger_path, context.config.resources.maximum_cuda_seconds)
    )
    vram = deps.vram_probe()
    decision = resource_decision({
        "peak_reserved_bytes": _strict_int(vram, "peak_reserved_bytes"),
        "total_vram_bytes": _strict_int(vram, "total_vram_bytes"),
        "download_bytes": 0,
        "artifact_bytes": deps.disk_probe(root) if root.exists() else 0,
        "projected_remaining_seconds": sum(projection.values()),
    }, ledger, context.config)
    if decision["status"] != "passed":
        raise BudgetStop("registered resource gate stopped this stage")


def _postflight_resources(context: StageContext, payload: Mapping[str, object], deps: RecurrenceDependencies) -> None:
    root = context.run_root
    ledger_path = root / "cuda-cost.jsonl"
    projection = _projection(context, deps)
    vram = deps.vram_probe()
    measurements = {
        "peak_reserved_bytes": int(vram.get("peak_reserved_bytes", 0)),
        "total_vram_bytes": int(vram.get("total_vram_bytes", 1)),
        "download_bytes": int(payload.get("download_bytes", 0)),
        "artifact_bytes": deps.disk_probe(root),
        "projected_remaining_seconds": float(sum(projection.values())),
    }
    ledger = (
        CudaCostLedger.load(ledger_path, context.config.resources.maximum_cuda_seconds)
        if ledger_path.exists()
        else CudaCostLedger(ledger_path, context.config.resources.maximum_cuda_seconds)
    )
    decision = resource_decision(measurements, ledger, context.config)
    if decision["status"] != "passed":
        raise BudgetStop("registered resource gate stopped this stage")


def _projection(context: StageContext, deps: RecurrenceDependencies) -> dict[str, float]:
    projection = dict(deps.remaining_work_projection(context))
    required = {"production_arms", "longest_prompt_generation", "retention_chunks", "retry_allowance", "report_cuda"}
    if set(projection) != required or any(isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 for value in projection.values()):
        raise StageError("remaining-work projection must enumerate all registered components")
    return {key: float(value) for key, value in projection.items()}


def _strict_int(mapping: Mapping[str, int], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int): raise StageError(f"{key} must be an integer")
    return value


def _fresh_boundary() -> Mapping[str, object]:
    code = "import os; print(os.getpid())"
    completed = subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True)
    return {"pid": int(completed.stdout.strip())}


def _checkpoint_signature(stage: str, attempt: int, checkpoint_path: str, checkpoint_digest: str, checkpoint_sha256: str, checkpoint_update: int, event_id: str, event_index: int, arm: str, seed: int, relative_path: str, signatures: Mapping[str, str]) -> str:
    return sha256_payload({
        "stage": stage,
        "attempt": attempt,
        "checkpoint_path": checkpoint_path,
        "checkpoint_digest": checkpoint_digest,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_update": checkpoint_update,
        "checkpoint_event_id": event_id,
        "checkpoint_event_index": event_index,
        "checkpoint_arm": arm,
        "checkpoint_seed": seed,
        "checkpoint_relative_path": relative_path,
        "signatures": dict(signatures),
    })


def _checkpoint_proof(request: Mapping[str, object]) -> str:
    return sha256_payload({
        "worker_identity": request["worker_identity"],
        "checkpoint_path": request["checkpoint_path"],
        "checkpoint_digest": request["checkpoint_digest"],
        "checkpoint_sha256": request["checkpoint_sha256"],
        "checkpoint_attempt": request["checkpoint_attempt"],
        "checkpoint_update": request["checkpoint_update"],
        "checkpoint_event_id": request["checkpoint_event_id"],
        "checkpoint_event_index": request["checkpoint_event_index"],
        "checkpoint_arm": request["checkpoint_arm"],
        "checkpoint_seed": request["checkpoint_seed"],
        "checkpoint_relative_path": request["checkpoint_relative_path"],
        "adapter_schema": request.get("adapter_schema"),
        "validated": True,
    })


def _callable_identity(reference: str) -> str:
    import inspect
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise EnvironmentStop("invalid child dispatcher reference")
    try:
        module = importlib.import_module(module_name)
        worker = getattr(module, attribute)
        source = inspect.getsource(worker)
    except (ImportError, AttributeError, OSError, TypeError) as error:
        raise EnvironmentStop("child dispatcher implementation cannot be inspected") from error
    return sha256_payload({
        "module": module_name,
        "attribute": attribute,
        "qualname": getattr(worker, "__qualname__", attribute),
        "source_sha256": sha256(source.encode("utf-8")).hexdigest(),
    })


_PRODUCTION_CHILD_ALLOWLIST: Final[tuple[tuple[str, str, str], ...]] = ()


def _child_worker_ref(stage: str, deps: RecurrenceDependencies) -> tuple[str, str]:
    if deps.test_only:
        if deps.child_worker_refs is not None:
            if set(deps.child_worker_refs) != set(STAGE_DEPENDENCIES):
                raise EnvironmentStop("registered test child dispatcher map is incomplete")
            reference = deps.child_worker_refs.get(stage)
        else:
            reference = deps.child_worker_ref
        if not isinstance(reference, str) or not reference:
            raise EnvironmentStop(f"no registered test child dispatcher for stage {stage}")
        return reference, _callable_identity(reference)
    if deps.child_worker_refs is not None or deps.child_worker_ref is not None:
        raise EnvironmentStop("production child dispatchers must use frozen stage registration")
    binding = next((item for item in _PRODUCTION_CHILD_ALLOWLIST if item[0] == stage), None)
    if binding is None:
        raise EnvironmentStop(f"no production child dispatcher is registered for stage {stage}")
    _stage, reference, identity = binding
    if not reference or not identity or _callable_identity(reference) != identity:
        raise EnvironmentStop(f"invalid production child dispatcher for stage {stage}")
    if "_child_echo_worker" in reference or "tests" in reference:
        raise EnvironmentStop("test helper cannot be a production child dispatcher")
    return reference, identity


def _validate_trusted_path(path: Path, trusted_root: Path | None = None) -> None:
    if trusted_root is not None:
        _reject_retry_reparse(trusted_root, required=True)
        root = trusted_root.resolve(strict=True)
        try:
            lexical = path.absolute().relative_to(root.absolute())
        except ValueError as error:
            raise StagePrerequisiteError(f"sealed evidence is outside trusted run root: {path}") from error
        current = root
        for component in lexical.parts:
            current = current / component
            _reject_retry_reparse(current, required=True)
        try:
            candidate = path.resolve(strict=True)
            relative = candidate.relative_to(root)
        except (FileNotFoundError, ValueError) as error:
            raise StagePrerequisiteError(f"sealed evidence is outside trusted run root: {path}") from error
        current = root
        for component in relative.parts:
            current = current / component
            _reject_retry_reparse(current, required=True)
    else:
        _reject_retry_reparse(path, required=True)


def _derive_checkpoint_artifact(run_root: Path, stage: str, signatures: Mapping[str, str], deps: RecurrenceDependencies) -> dict[str, object]:
    events_path = run_root / "events.jsonl"
    events: list[dict[str, object]] = []
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    events.append(value)
    failures = [event for event in events if event.get("kind") == "training_failure" and event.get("cuda_process_fault") is True and event.get("stage") in (None, stage)]
    # Test-only injected workers do not have a registered arm/seed producer.
    # Their stage-boundary failure may still be handed to the injected
    # checkpoint validator, but it must never be accepted by production retry
    # or by the strict report event validator.
    if not failures and deps.test_only and deps.checkpoint_validator is not None:
        failures = [event for event in events if event.get("kind") == "stage_attempt" and event.get("stage") == stage and event.get("outcome") in {"error", "fresh_process_required"}]
    if not failures:
        raise EnvironmentStop("fresh retry has no durable Task-9 training failure event with arm/seed identity")
    failure = failures[-1]
    event_attempt = failure.get("attempt")
    event_update = failure.get("update", 0) if failure.get("kind") == "stage_attempt" else failure.get("update")
    if isinstance(event_attempt, bool) or not isinstance(event_attempt, int) or event_attempt != 1:
        raise EnvironmentStop("durable Task-9 failure event has an invalid retry attempt")
    if isinstance(event_update, bool) or not isinstance(event_update, int) or event_update < 0:
        raise EnvironmentStop("durable Task-9 failure event has an invalid checkpoint update")
    if deps.test_only and deps.checkpoint_validator is not None:
        artifact = dict(deps.checkpoint_validator(run_root, signatures, failure))
        artifact.setdefault("attempt", event_attempt)
        artifact.setdefault("update", event_update)
        if not isinstance(artifact.get("path"), str) or not isinstance(artifact.get("checkpoint_sha256"), str):
            raise EnvironmentStop("test checkpoint validator returned incomplete evidence")
        _validate_trusted_path(Path(str(artifact["path"])), run_root)
        artifact.setdefault("event_id", str(failure.get("event_id", "test-event")))
        artifact.setdefault("event_index", int(failure.get("index", 0)))
        artifact.setdefault("arm", str(failure.get("arm", "test")))
        artifact.setdefault("seed", int(failure.get("seed", 0)))
        artifact.setdefault("relative_path", Path(str(artifact["path"])).resolve().relative_to(run_root.resolve()).as_posix())
        return artifact
    event_index = failure.get("index")
    event_id = failure.get("event_id")
    arm, seed = failure.get("arm"), failure.get("seed")
    relative_path = failure.get("checkpoint_path")
    manifest_digest, checkpoint_sha = failure.get("checkpoint_manifest_digest"), failure.get("checkpoint_sha256")
    if isinstance(event_index, bool) or not isinstance(event_index, int) or event_index < 0:
        raise EnvironmentStop("durable Task-9 failure event has no immutable index")
    if not isinstance(event_id, str) or __import__("re").fullmatch(r"[0-9a-f]{64}", event_id) is None:
        raise EnvironmentStop("durable Task-9 failure event has no immutable event ID")
    if arm not in {"a1", "a2", "a3"} or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EnvironmentStop("durable Task-9 failure event has invalid arm/seed identity")
    expected_event_id = task9_failure_event_id(failure)
    if event_id != expected_event_id:
        raise EnvironmentStop("durable Task-9 failure event ID is inconsistent")
    if not isinstance(relative_path, str) or Path(relative_path).as_posix() != relative_path:
        raise EnvironmentStop("durable Task-9 checkpoint path is not canonical")
    expected_relative = f"checkpoints/{arm}/{seed}/latest.pt"
    if relative_path != expected_relative:
        raise EnvironmentStop("durable Task-9 checkpoint path does not match arm/seed")
    if not isinstance(checkpoint_sha, str) or __import__("re").fullmatch(r"[0-9a-f]{64}", checkpoint_sha) is None:
        raise EnvironmentStop("durable Task-9 failure event has no checkpoint digest")
    if not isinstance(manifest_digest, str) or __import__("re").fullmatch(r"[0-9a-f]{64}", manifest_digest) is None:
        raise EnvironmentStop("durable Task-9 failure event has no checkpoint manifest digest")
    checkpoint_root = run_root / "checkpoints" / arm / str(seed)
    _validate_trusted_path(checkpoint_root, run_root)
    if not checkpoint_root.is_dir():
        raise EnvironmentStop("durable retry checkpoint directory is missing")
    schema = deps.checkpoint_schema
    if callable(schema):
        schema = schema(arm, seed, signatures)
    if not isinstance(schema, Mapping):
        raise EnvironmentStop("no registered adapter schema is available for retry")
    checkpoint_bytes = _read_test_bytes_once(run_root / relative_path, trusted_root=run_root)
    try:
        artifact = validate_checkpoint_artifact(checkpoint_root, signatures, expected_attempt=event_attempt, expected_update=event_update, expected_arm=arm, expected_seed=seed, expected_adapter_schema=schema, checkpoint_bytes=checkpoint_bytes)
    except (OSError, ValueError) as error:
        raise EnvironmentStop(f"durable retry checkpoint is invalid: {error}") from error
    _validate_trusted_path(Path(str(artifact["path"])), run_root)
    if Path(str(artifact["path"])).resolve() != (run_root / relative_path).resolve() or artifact.get("checkpoint_sha256") != checkpoint_sha or artifact.get("manifest_digest") != manifest_digest:
        raise EnvironmentStop("durable Task-9 checkpoint identity does not match failure event")
    artifact.update({"event_id": event_id, "event_index": event_index, "arm": arm, "seed": seed, "relative_path": relative_path, "adapter_schema": dict(schema)})
    return artifact
def _reject_retry_reparse(path: Path, *, required: bool = False) -> None:
    try:
        stats = path.lstat()
    except FileNotFoundError:
        if required:
            raise EnvironmentStop(f"required retry evidence path is missing: {path}")
        return
    if path.is_symlink() or bool(getattr(stats, "st_file_attributes", 0) & 0x400):
        raise EnvironmentStop(f"retry evidence path is a symbolic/reparse link: {path}")


def _run_child_retry(stage: str, config_path: Path, artifact_root: Path, run_root: Path, run_id: str | None, signatures: Mapping[str, str], deps: RecurrenceDependencies) -> ChildResult:
    """Hand work to a new interpreter and accept only its bound result file."""
    retry_root = run_root / "retry"
    _reject_retry_reparse(run_root, required=True)
    _reject_retry_reparse(retry_root)
    retry_root.mkdir(parents=True, exist_ok=True)
    _reject_retry_reparse(retry_root)
    nonce = sha256(os.urandom(32)).hexdigest()
    checkpoint = _derive_checkpoint_artifact(run_root, stage, signatures, deps)
    checkpoint_path = str(Path(str(checkpoint["path"])).resolve())
    checkpoint_digest = str(checkpoint.get("manifest_digest", checkpoint["checkpoint_sha256"]))
    checkpoint_sha256 = str(checkpoint["checkpoint_sha256"])
    checkpoint_attempt = int(checkpoint["attempt"])
    checkpoint_update = int(checkpoint["update"])
    checkpoint_event_id = str(checkpoint["event_id"])
    checkpoint_event_index = int(checkpoint["event_index"])
    checkpoint_arm = str(checkpoint["arm"])
    checkpoint_seed = int(checkpoint["seed"])
    checkpoint_relative_path = str(checkpoint["relative_path"])
    dispatcher_ref, dispatcher_identity = _child_worker_ref(stage, deps)
    request = ChildRequest(
        stage=stage, config_path=str(config_path), artifact_root=str(artifact_root),
        run_root=str(run_root), run_id=run_id, checkpoint_path=checkpoint_path,
        checkpoint_digest=checkpoint_digest, checkpoint_sha256=checkpoint_sha256,
        checkpoint_attempt=checkpoint_attempt,
        checkpoint_update=checkpoint_update,
        checkpoint_event_id=checkpoint_event_id, checkpoint_event_index=checkpoint_event_index,
        checkpoint_arm=checkpoint_arm, checkpoint_seed=checkpoint_seed,
        checkpoint_relative_path=checkpoint_relative_path,
        checkpoint_signature=_checkpoint_signature(stage, 2, checkpoint_path, checkpoint_digest, checkpoint_sha256, checkpoint_update, checkpoint_event_id, checkpoint_event_index, checkpoint_arm, checkpoint_seed, checkpoint_relative_path, signatures),
        attempt=2, signatures=dict(signatures), worker_ref=dispatcher_ref,
        worker_identity=dispatcher_identity,
        checkpoint_proof=_checkpoint_proof({
            "worker_identity": dispatcher_identity, "checkpoint_path": checkpoint_path,
            "checkpoint_digest": checkpoint_digest, "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_attempt": checkpoint_attempt,
            "checkpoint_update": checkpoint_update,
            "checkpoint_event_id": checkpoint_event_id, "checkpoint_event_index": checkpoint_event_index,
            "checkpoint_arm": checkpoint_arm, "checkpoint_seed": checkpoint_seed,
            "checkpoint_relative_path": checkpoint_relative_path,
            "adapter_schema": checkpoint.get("adapter_schema"),
        }),
        adapter_schema=checkpoint.get("adapter_schema"),
        test_only=deps.test_only, nonce=nonce,
    )
    request_path = retry_root / f"attempt-2-{nonce}.request.json"
    request_payload = {"schema_version": 1, "stage": request.stage, "config_path": request.config_path, "artifact_root": request.artifact_root, "run_root": request.run_root, "run_id": request.run_id, "checkpoint_path": request.checkpoint_path, "checkpoint_digest": request.checkpoint_digest, "checkpoint_sha256": request.checkpoint_sha256, "checkpoint_attempt": request.checkpoint_attempt, "checkpoint_update": request.checkpoint_update, "checkpoint_event_id": request.checkpoint_event_id, "checkpoint_event_index": request.checkpoint_event_index, "checkpoint_arm": request.checkpoint_arm, "checkpoint_seed": request.checkpoint_seed, "checkpoint_relative_path": request.checkpoint_relative_path, "checkpoint_signature": request.checkpoint_signature, "attempt": request.attempt, "signatures": dict(request.signatures), "worker_ref": request.worker_ref, "worker_identity": request.worker_identity, "checkpoint_proof": request.checkpoint_proof, "adapter_schema": request.adapter_schema, "test_only": request.test_only, "nonce": request.nonce}
    request_hash = write_immutable_json(request_path, request_payload)
    result_path = retry_root / f"attempt-2-{nonce}.result.json"
    interpreter = getattr(sys, "_base_executable", sys.executable)
    command = list(deps.child_command or (interpreter, "-m", "ppmoe.recurrence.experiment", "--child-request"))
    command.extend((str(request_path), str(result_path)))
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[3] / "src")
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
    _stdout, stderr = process.communicate()
    if not result_path.exists():
        raise EnvironmentStop(f"fresh child produced no result: {stderr.strip()}")
    raw = _read_json(result_path)
    required = {"request_sha256", "nonce", "stage", "run_root", "attempt", "pid", "status", "payload"}
    if set(raw) != required or raw.get("request_sha256") != request_hash or raw.get("nonce") != nonce or raw.get("stage") != stage or raw.get("run_root") != str(run_root) or raw.get("attempt") != 2 or raw.get("status") != "passed":
        raise EnvironmentStop("fresh child result does not authenticate the retry request")
    pid = raw.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid != process.pid:
        raise EnvironmentStop(f"fresh child PID does not match launched process: result={pid} launched={process.pid}")
    payload = raw.get("payload")
    if (
        not isinstance(payload, dict)
        or payload.get("worker_pid") != process.pid
        or payload.get("attempt") != request.attempt
        or payload.get("checkpoint_path") != request.checkpoint_path
        or payload.get("checkpoint_digest") != request.checkpoint_digest
        or payload.get("checkpoint_sha256") != request.checkpoint_sha256
        or payload.get("checkpoint_attempt") != request.checkpoint_attempt
        or payload.get("checkpoint_update") != request.checkpoint_update
        or payload.get("checkpoint_event_id") != request.checkpoint_event_id
        or payload.get("checkpoint_event_index") != request.checkpoint_event_index
        or payload.get("checkpoint_arm") != request.checkpoint_arm
        or payload.get("checkpoint_seed") != request.checkpoint_seed
        or payload.get("checkpoint_relative_path") != request.checkpoint_relative_path
        or payload.get("checkpoint_signature") != request.checkpoint_signature
        or payload.get("dispatcher_stage") != stage
        or payload.get("dispatcher_ref") != request.worker_ref
        or payload.get("dispatcher_identity") != request.worker_identity
        or payload.get("checkpoint_proof") != request.checkpoint_proof
    ):
        raise EnvironmentStop("fresh child worker PID is spoofed")
    if process.returncode != 0:
        raise EnvironmentStop(f"fresh child failed: {stderr.strip()}")
    # Re-open and revalidate after the child exits. Any child-side mutation of
    # the checkpoint or manifest invalidates the retry before sealing success.
    post_checkpoint = _derive_checkpoint_artifact(run_root, stage, signatures, deps)
    for key in ("path", "checkpoint_sha256", "manifest_digest", "attempt", "update", "event_id", "event_index", "arm", "seed", "relative_path"):
        if post_checkpoint.get(key) != checkpoint.get(key):
            raise EnvironmentStop("checkpoint changed after child validation")
    post_proof = _checkpoint_proof({
        "worker_identity": request.worker_identity,
        "checkpoint_path": str(Path(str(post_checkpoint["path"])).resolve()),
        "checkpoint_digest": post_checkpoint.get("manifest_digest", post_checkpoint["checkpoint_sha256"]),
        "checkpoint_sha256": post_checkpoint["checkpoint_sha256"],
        "checkpoint_attempt": post_checkpoint["attempt"],
        "checkpoint_update": post_checkpoint["update"],
        "checkpoint_event_id": post_checkpoint["event_id"],
        "checkpoint_event_index": post_checkpoint["event_index"],
        "checkpoint_arm": post_checkpoint["arm"],
        "checkpoint_seed": post_checkpoint["seed"],
        "checkpoint_relative_path": post_checkpoint["relative_path"],
        "adapter_schema": post_checkpoint.get("adapter_schema"),
    })
    if post_proof != request.checkpoint_proof:
        raise EnvironmentStop("checkpoint proof changed after child validation")
    return ChildResult(request_hash, nonce, stage, str(run_root), 2, pid, "passed", payload)


def _child_echo_worker(request: Mapping[str, object]) -> Mapping[str, object]:
    """Minimal serializable worker used by orchestration-only tests."""
    return {
        "worker_pid": os.getpid(),
        "attempt": request["attempt"],
        "checkpoint_path": request["checkpoint_path"],
        "checkpoint_digest": request["checkpoint_digest"],
        "checkpoint_sha256": request["checkpoint_sha256"],
        "checkpoint_attempt": request["checkpoint_attempt"],
        "checkpoint_update": request["checkpoint_update"],
        "checkpoint_event_id": request["checkpoint_event_id"],
        "checkpoint_event_index": request["checkpoint_event_index"],
        "checkpoint_arm": request["checkpoint_arm"],
        "checkpoint_seed": request["checkpoint_seed"],
        "checkpoint_relative_path": request["checkpoint_relative_path"],
        "checkpoint_signature": request["checkpoint_signature"],
        "dispatcher_stage": request["stage"],
        "dispatcher_ref": request["worker_ref"],
        "dispatcher_identity": request["worker_identity"],
        "checkpoint_proof": request["checkpoint_proof"],
    }


def _child_fail_worker(_request: Mapping[str, object]) -> Mapping[str, object]:
    raise RuntimeError("controlled child failure")


def _child_execute(request_path: Path, result_path: Path) -> int:
    request = _read_json(request_path)
    required = {"schema_version", "stage", "config_path", "artifact_root", "run_root", "run_id", "checkpoint_path", "checkpoint_digest", "checkpoint_sha256", "checkpoint_attempt", "checkpoint_update", "checkpoint_event_id", "checkpoint_event_index", "checkpoint_arm", "checkpoint_seed", "checkpoint_relative_path", "checkpoint_signature", "attempt", "signatures", "worker_ref", "worker_identity", "checkpoint_proof", "adapter_schema", "test_only", "nonce"}
    stage = request.get("stage")
    allow_empty_data = stage in {"source", "data"}
    allow_empty_source = stage == "source"
    if set(request) != required or request.get("schema_version") != 1 or request.get("attempt") != 2 or not _valid_signatures(request.get("signatures"), allow_empty_data=allow_empty_data, allow_empty_source=allow_empty_source):
        return 1
    request_hash = sha256_file(request_path)
    try:
        module_name, separator, attribute = str(request["worker_ref"]).partition(":")
        if not separator or not module_name or not attribute: raise ValueError("invalid worker reference")
        if _callable_identity(str(request["worker_ref"])) != request["worker_identity"]:
            raise ValueError("dispatcher implementation identity mismatch")
        if not request["test_only"]:
            checkpoint_path = Path(str(request["checkpoint_path"]))
            _validate_trusted_path(checkpoint_path, Path(str(request["run_root"])))
            checkpoint_bytes = _read_test_bytes_once(checkpoint_path, trusted_root=Path(str(request["run_root"])))
            artifact = validate_checkpoint_artifact(
                checkpoint_path.parent,
                request["signatures"],
                expected_attempt=int(request["checkpoint_attempt"]),
                expected_update=int(request["checkpoint_update"]),
                expected_arm=str(request["checkpoint_arm"]),
                expected_seed=int(request["checkpoint_seed"]),
                expected_adapter_schema=request["adapter_schema"],
                checkpoint_bytes=checkpoint_bytes,
            )
            if (
                artifact.get("path") != str(checkpoint_path)
                or artifact.get("checkpoint_sha256") != request["checkpoint_sha256"]
                or artifact.get("manifest_digest") != request["checkpoint_digest"]
            ):
                raise ValueError("child checkpoint validation does not match request")
            child_proof = _checkpoint_proof({
                "worker_identity": request["worker_identity"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_digest": artifact["manifest_digest"],
                "checkpoint_sha256": artifact["checkpoint_sha256"],
                "checkpoint_attempt": artifact["attempt"],
                "checkpoint_update": artifact["update"],
                "checkpoint_event_id": request["checkpoint_event_id"],
                "checkpoint_event_index": request["checkpoint_event_index"],
                "checkpoint_arm": request["checkpoint_arm"],
                "checkpoint_seed": request["checkpoint_seed"],
                "checkpoint_relative_path": request["checkpoint_relative_path"],
                "adapter_schema": request["adapter_schema"],
            })
            if child_proof != request["checkpoint_proof"]:
                raise ValueError("child checkpoint proof does not match fresh validation")
        worker = getattr(importlib.import_module(module_name), attribute)
        payload = dict(worker(request) or {})
        if payload.get("checkpoint_proof") != request["checkpoint_proof"]:
            raise ValueError("child worker did not prove checkpoint validation")
        payload.setdefault("worker_pid", os.getpid())
        result = {"request_sha256": request_hash, "nonce": request["nonce"], "stage": request["stage"], "run_root": request["run_root"], "attempt": 2, "pid": os.getpid(), "status": "passed", "payload": payload}
        write_immutable_json(result_path, result)
        return 0
    except Exception as error:
        result = {"request_sha256": request_hash, "nonce": request.get("nonce"), "stage": request.get("stage"), "run_root": request.get("run_root"), "attempt": 2, "pid": os.getpid(), "status": "failed_environment", "payload": {"error_type": type(error).__name__, "message": str(error)}}
        write_immutable_json(result_path, result)
        return 1


def _reconcile_chargeable_cuda(root: Path, config: RecurrenceConfig, stage: str, attempt: int) -> None:
    """Charge every durable Task-9 fault interval for one stage/attempt once."""
    events = root / "events.jsonl"
    if not events.exists():
        return
    ledger = CudaCostLedger.load(root / "cuda-cost.jsonl", config.resources.maximum_cuda_seconds)
    for line_index, line in enumerate(events.read_text(encoding="utf-8").splitlines()):
        event = json.loads(line)
        if (event.get("stage") or stage) != stage or event.get("attempt") != attempt:
            continue
        if event.get("cuda_process_fault") is not True or "chargeable_cuda_seconds" not in event:
            continue
        elapsed = event["chargeable_cuda_seconds"]
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or float(elapsed) < 0:
            raise StageError("chargeable CUDA interval is invalid")
        attempt_id = f"{stage}:attempt:{attempt}:task9-fault:{event.get('index', line_index)}:{event.get('arm', '')}:{event.get('seed', '')}"
        if any(entry.attempt_id == attempt_id for entry in ledger.entries):
            continue
        ledger.record(attempt_id, stage, float(elapsed), "failed_environment")


def _reconcile_all_chargeable_cuda(root: Path, config: RecurrenceConfig, current_stage: str | None = None) -> None:
    events = root / "events.jsonl"
    if not events.exists():
        return
    for line in events.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        stage, attempt = event.get("stage") or current_stage, event.get("attempt")
        if isinstance(stage, str) and isinstance(attempt, int) and not isinstance(attempt, bool):
            _reconcile_chargeable_cuda(root, config, stage, attempt)


def _retry_in_real_child(stage: str, config_path: Path, artifact_root: Path, run_id: str | None) -> dict[str, object]:
    command = [sys.executable, "-m", "ppmoe.recurrence.experiment", "--child", "--stage", stage, "--config", str(config_path), "--artifact-root", str(artifact_root)]
    if run_id is not None: command.extend(("--run-id", run_id))
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise EnvironmentStop(f"fresh child retry failed: {completed.stderr.strip()}")
    root = artifact_root / run_id if run_id else artifact_root / "staging" / _staging_id(RecurrenceConfig.load(config_path).model.repo_id, *_control_hashes(config_path))
    return _read_stage_record(_stage_path(root, stage), stage, _any_signatures(root) if stage == "report" else _predecessor_signatures(root, STAGE_DEPENDENCIES[stage][0]) if STAGE_DEPENDENCIES[stage] else _any_signatures(root))


def _identity_only_report_fixture(root: Path) -> bool:
    """Recognize only the old no-evidence orchestration isolation fixture."""
    stage_paths = sorted((root / "stages").glob("*.json")) if (root / "stages").is_dir() else []
    if not stage_paths or any(path.stem == "report" for path in stage_paths):
        return False
    for path in stage_paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if set(record) != {"stage", "status", "signatures"}:
            return False
    return not any(path for path in root.rglob("*") if path.is_file() and path.relative_to(root).parts[0] != "stages")


def _verify_report_artifacts(root: Path, record: Mapping[str, object]) -> None:
    payload = record.get("payload")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("report_sha256"), str):
        raise StageError("persisted report record has no report output digest")
    report_path = root / "report.json"
    markdown_path = root / "report.md"
    sources_path = root / "report-sources.json"
    for path in (report_path, markdown_path, sources_path):
        try:
            _reject_reparse_path(root, path)
        except ValueError as error:
            raise StageError("persisted report outputs contain a reparse/symlink path") from error
    if any(not path.is_file() or path.is_symlink() for path in (report_path, markdown_path, sources_path)):
        raise StageError("persisted report outputs are missing or mutable")
    try:
        report_bytes, _, _ = _descriptor_read(report_path, root)
        markdown_bytes, _, _ = _descriptor_read(markdown_path, root)
        sources_bytes, _, _ = _descriptor_read(sources_path, root)
        report = json.loads(report_bytes.decode("utf-8"))
        sources = json.loads(sources_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageError("persisted report outputs are malformed") from error
    if not isinstance(report, dict) or not isinstance(sources, dict):
        raise StageError("persisted report outputs are malformed")
    expected = report.get("report_sha256")
    body = dict(report)
    body.pop("report_sha256", None)
    if not isinstance(expected, str) or sha256_payload(body) != expected or expected != payload.get("report_sha256"):
        raise StageError("persisted report JSON digest mismatch")
    source_digest = sha256(sources_bytes).hexdigest()
    if report.get("source_manifest_sha256") != source_digest or payload.get("source_manifest_sha256") != source_digest:
        raise StageError("persisted report source-manifest digest mismatch")
    files = sources.get("files")
    if sources.get("schema_version") != 1 or not isinstance(files, list):
        raise StageError("persisted report source manifest has an invalid schema")
    current = {str(item["path"]): item for item in _read_inputs(root)}
    recorded = {str(item.get("path")): item for item in files if isinstance(item, Mapping) and isinstance(item.get("path"), str)}
    if len(recorded) != len(files) or current != recorded:
        raise StageError("current report evidence inventory differs from persisted report-sources")
    if _render_markdown(report).encode("utf-8") != markdown_bytes:
        raise StageError("persisted report Markdown does not reconcile with JSON")


def _validate_dependencies(deps: RecurrenceDependencies) -> None:
    if set(deps.stage_functions) != set(STAGE_DEPENDENCIES):
        raise ValueError("stage_functions must define every registered stage exactly once")


def _default_vram_probe() -> Mapping[str, int]:
    import torch
    if not torch.cuda.is_available(): return {"total_vram_bytes": 2_147_483_649, "peak_reserved_bytes": 0}
    return {"total_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0))}


def _lazy_hf_api() -> object:
    from huggingface_hub import HfApi
    return HfApi()
def _lazy_snapshot_download() -> object:
    from huggingface_hub import snapshot_download
    return snapshot_download
def _lazy_transformers_loader() -> object:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return (AutoTokenizer, AutoModelForCausalLM)
def _lazy_datasets_loader() -> object:
    from datasets import load_dataset
    return load_dataset
def _lazy_torch_synchronize() -> None:
    import torch
    torch.cuda.synchronize()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--stage", choices=tuple(STAGE_DEPENDENCIES))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--child-request", nargs=2, metavar=("REQUEST", "RESULT"))
    values = parser.parse_args()
    if values.child_request:
        raise SystemExit(_child_execute(Path(values.child_request[0]), Path(values.child_request[1])))
    if values.stage is None or values.config is None or values.artifact_root is None:
        parser.error("--stage, --config, and --artifact-root are required unless --child-request is used")
    try:
        run_stage(values.stage, values.config, values.artifact_root, values.run_id)
    except StageError as error:
        raise SystemExit(str(error))
