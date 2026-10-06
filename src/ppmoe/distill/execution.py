"""Immutable execution identities, authenticated events, and active-job clocks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import time
from typing import Any, Mapping


ACTIVE_RUNTIME_SECONDS = 172800
CUDA_SECONDS = 86400
DOWNLOAD_BYTES = 21474836480
ARTIFACT_BYTES = 21474836480
INHERITED_CUDA_SECONDS = 1168
INHERITED_DOWNLOAD_BYTES = 8066699435
TASK1_STOP_SHA256 = "65b9d9daf2e1af45ed58feff933f7cdb81e5326c460d0de57b5a02fa4620d996"
_GENESIS_DIGEST = "0" * 64


@dataclass(frozen=True, slots=True)
class ActiveJobContext:
    execution_id: str
    attempt_id: str
    stage: str
    prior_active_seconds: int
    started_monotonic_ns: int
    supervisor_pid: int
    supervisor_created_utc: str
    stage_signature: str


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(value: Any) -> str:
    return sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _checked_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.exists() or current.is_symlink():
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError(f"symlink/reparse path rejected: {current}")
    return absolute


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _ensure_disjoint(source_root: Path, output_root: Path) -> None:
    source = _checked_path(source_root).resolve(strict=True)
    output = _checked_path(output_root).resolve(strict=False)
    if _is_within(source, output) or _is_within(output, source):
        raise ValueError("source and execution roots overlap")


def _inventory(root: Path) -> tuple[list[dict[str, Any]], int]:
    entries: list[dict[str, Any]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.exists() and not path.is_symlink():
            continue
        _checked_path(path)
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        entries.append({"path": relative, "sha256": _sha_file(path), "size_bytes": size})
        total += size
    return entries, total


def _validate_old_terminal(source_root: Path, manifest: Mapping[str, Any], config_path: Path, repo_root: Path) -> tuple[Path, str]:
    from .study import TASK1_MANIFEST_SHA256, TASK1_PREPARE_SHA256

    terminal_path = source_root / "stages" / "calibrate.json"
    if not terminal_path.is_file():
        raise ValueError("source has no authenticated stopped_environment calibration closeout")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal_hash = _sha_file(terminal_path)
    if terminal_hash != TASK1_STOP_SHA256:
        raise ValueError("source terminal closeout hash differs from reviewed stopped_environment record")
    if terminal.get("status") != "stopped_environment":
        raise ValueError("source calibration closeout is not stopped_environment")
    if terminal.get("reason") != "global wall-time budget expired":
        raise ValueError("source calibration closeout reason is not the expired legacy wall budget")
    if terminal.get("stage") != "calibrate" or terminal.get("run_id") != source_root.name:
        raise ValueError("source calibration closeout identity mismatch")
    manifest_hash = _sha_file(source_root / "manifest.json")
    prepare_hash = _sha_file(source_root / "stages" / "prepare.json")
    if manifest_hash != TASK1_MANIFEST_SHA256 or prepare_hash != TASK1_PREPARE_SHA256:
        raise ValueError("source manifest/preparation hashes differ from reviewed Task 1 evidence")
    if terminal.get("task1_manifest_sha256") != manifest_hash or terminal.get("task1_prepare_sha256") != prepare_hash:
        raise ValueError("source terminal record does not bind manifest and preparation hashes")
    signature = terminal.get("signature")
    predecessors = signature.get("predecessors", {}) if isinstance(signature, dict) else {}
    if (
        predecessors.get("prepare_manifest_sha256") != manifest_hash
        or predecessors.get("prepare_stage_sha256") != prepare_hash
    ):
        raise ValueError("source terminal signature does not bind preparation hashes")
    try:
        config_rel = Path(config_path).resolve(strict=True).relative_to(Path(repo_root).resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise ValueError("config path must be inside repo root") from exc
    code_signatures = signature.get("code_signatures", {}) if isinstance(signature, dict) else {}
    if code_signatures.get(config_rel) != _sha_file(config_path):
        raise ValueError("source terminal record does not bind the unchanged scientific config")
    if manifest.get("config") is None:
        raise ValueError("prepared source scientific config is missing")
    raw_dir = source_root / "calibration" / "raw"
    if raw_dir.exists() and any(path.is_file() for path in raw_dir.rglob("*")):
        raise ValueError("source contains calibration model outputs; execution namespace reuse refused")
    return terminal_path, terminal_hash


def register_execution(
    source_root: Path,
    executions_root: Path,
    *,
    repo_root: Path,
    config_path: Path,
) -> Path:
    """Create one exclusive execution namespace bound to stopped source evidence."""
    from .study import validate_prepared_run

    source = _checked_path(Path(source_root)).resolve(strict=True)
    executions = _checked_path(Path(executions_root)).resolve(strict=False)
    repo = _checked_path(Path(repo_root)).resolve(strict=True)
    config = _checked_path(Path(config_path)).resolve(strict=True)
    _ensure_disjoint(source, executions)
    source_manifest = validate_prepared_run(source)
    runtime_identity = source_manifest.get("source_signature", {}).get("runtime_identity")
    if not isinstance(runtime_identity, dict) or not runtime_identity:
        raise ValueError("prepared source has no authenticated feasibility runtime identity")
    terminal_path, terminal_hash = _validate_old_terminal(source, source_manifest, config, repo)
    source_files, source_bytes = _inventory(source)

    feasibility_entries: list[dict[str, Any]] = []
    from .study import FEASIBILITY_ROOT
    feasibility_root = repo / FEASIBILITY_ROOT
    if feasibility_root.exists():
        feasibility_entries, _ = _inventory(feasibility_root)
        feasibility_entries = [
            {**entry, "path": (Path(FEASIBILITY_ROOT).as_posix() + "/" + entry["path"])}
            for entry in feasibility_entries
        ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "source_root": str(source),
        "source_run_id": source.name,
        "source_manifest_sha256": _sha_file(source / "manifest.json"),
        "source_prepare_sha256": _sha_file(source / "stages" / "prepare.json"),
        "source_terminal_sha256": terminal_hash,
        "config_path": str(config),
        "config_sha256": _sha_file(config),
        "lock_sha256": _sha_file(_checked_path(repo / "uv.lock")),
        "runtime_identity": runtime_identity,
        "policy": {
            "active_runtime_seconds": ACTIVE_RUNTIME_SECONDS,
            "cuda_seconds": CUDA_SECONDS,
            "download_bytes": DOWNLOAD_BYTES,
            "artifact_bytes": ARTIFACT_BYTES,
            "artifact_terminal_reserve_bytes": 1024 * 1024,
            "memory_reserve_bytes": 2 * 1024**3,
            "supported_stage": "calibrate",
        },
        "inherited_costs": {
            "cuda_seconds": INHERITED_CUDA_SECONDS,
            "download_bytes": INHERITED_DOWNLOAD_BYTES,
        },
        "artifact_baseline": {
            "source_files": source_files,
            "source_bytes": source_bytes,
            "feasibility_files": feasibility_entries,
        },
    }
    execution_id = _sha(manifest)
    manifest["execution_id"] = execution_id
    execution_root = executions / execution_id
    executions.mkdir(parents=True, exist_ok=True)
    _checked_path(executions)
    try:
        execution_root.mkdir()
    except FileExistsError:
        raise FileExistsError(f"execution already registered: {execution_id}") from None
    try:
        manifest_fd = os.open(
            execution_root / "execution.json",
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(manifest_fd, "wb") as stream:
            stream.write(_canonical(manifest) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        (execution_root / "events.jsonl").touch(exist_ok=False)
        (execution_root / "logs").mkdir()
        (execution_root / "stages").mkdir()
    except BaseException:
        # Retain partial registration evidence for inspection; never reuse it.
        raise
    return execution_root


def validate_execution(execution_root: Path, *, repo_root: Path, config_path: Path) -> dict[str, Any]:
    """Validate immutable execution, source, terminal, config, and lock identity."""
    root = _checked_path(Path(execution_root)).resolve(strict=True)
    manifest_path = _checked_path(root / "execution.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest_bytes != _canonical(manifest) + b"\n":
        raise ValueError("execution manifest is not canonical JSON")
    claimed_id = manifest.get("execution_id")
    without_id = {key: value for key, value in manifest.items() if key != "execution_id"}
    if not isinstance(claimed_id, str) or claimed_id != _sha(without_id) or root.name != claimed_id:
        raise ValueError("execution ID does not authenticate its manifest")
    source = _checked_path(Path(manifest["source_root"])).resolve(strict=True)
    _ensure_disjoint(source, root)
    repo = _checked_path(Path(repo_root)).resolve(strict=True)
    config = _checked_path(Path(config_path)).resolve(strict=True)
    if str(config) != manifest.get("config_path") or _sha_file(config) != manifest.get("config_sha256"):
        raise ValueError("execution scientific config identity changed")
    if _sha_file(_checked_path(repo / "uv.lock")) != manifest.get("lock_sha256"):
        raise ValueError("execution dependency lock identity changed")
    expected_policy = {
        "active_runtime_seconds": ACTIVE_RUNTIME_SECONDS,
        "cuda_seconds": CUDA_SECONDS,
        "download_bytes": DOWNLOAD_BYTES,
        "artifact_bytes": ARTIFACT_BYTES,
        "artifact_terminal_reserve_bytes": 1024 * 1024,
        "memory_reserve_bytes": 2 * 1024**3,
        "supported_stage": "calibrate",
    }
    if manifest.get("policy") != expected_policy:
        raise ValueError("execution resource policy differs from the registered fixed policy")
    if manifest.get("inherited_costs") != {
        "cuda_seconds": INHERITED_CUDA_SECONDS,
        "download_bytes": INHERITED_DOWNLOAD_BYTES,
    }:
        raise ValueError("execution inherited costs differ from reviewed feasibility")
    from .study import validate_prepared_run
    source_manifest = validate_prepared_run(source)
    if source.name != manifest.get("source_run_id"):
        raise ValueError("execution source run identity changed")
    if _sha_file(source / "manifest.json") != manifest.get("source_manifest_sha256"):
        raise ValueError("execution source manifest changed")
    if _sha_file(source / "stages" / "prepare.json") != manifest.get("source_prepare_sha256"):
        raise ValueError("execution source preparation changed")
    terminal_path, terminal_hash = _validate_old_terminal(source, source_manifest, config, repo)
    if terminal_hash != manifest.get("source_terminal_sha256"):
        raise ValueError("execution source terminal closeout changed")
    if _sha_file(terminal_path) != manifest.get("source_terminal_sha256"):
        raise ValueError("execution source terminal closeout changed")
    expected_source_files = manifest.get("artifact_baseline", {}).get("source_files")
    actual_source_files, actual_source_bytes = _inventory(source)
    if expected_source_files != actual_source_files or actual_source_bytes != manifest.get("artifact_baseline", {}).get("source_bytes"):
        raise ValueError("execution prepared-source artifact inventory changed")
    from .study import FEASIBILITY_ROOT
    feasibility_root = repo / FEASIBILITY_ROOT
    actual_feasibility, _ = _inventory(feasibility_root) if feasibility_root.exists() else ([], 0)
    actual_feasibility = [
        {**entry, "path": (Path(FEASIBILITY_ROOT).as_posix() + "/" + entry["path"])}
        for entry in actual_feasibility
    ]
    if actual_feasibility != manifest.get("artifact_baseline", {}).get("feasibility_files"):
        raise ValueError("execution feasibility artifact inventory changed")
    if not isinstance(manifest.get("runtime_identity"), dict) or not manifest["runtime_identity"]:
        raise ValueError("execution runtime identity is missing")
    read_events(root)
    return manifest


def _event_digest(sequence: int, previous_digest: str, payload: Mapping[str, Any]) -> str:
    return _sha({"sequence": sequence, "previous_digest": previous_digest, "payload": payload})


def read_events(execution_root: Path) -> list[dict[str, Any]]:
    path = _checked_path(Path(execution_root) / "events.jsonl")
    if not path.exists():
        return []
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        raise ValueError("execution event journal has a truncated tail")
    events: list[dict[str, Any]] = []
    previous = _GENESIS_DIGEST
    for expected_sequence, line in enumerate(content.splitlines()):
        try:
            event = json.loads(line)
            sequence = event["sequence"]
            payload = event["payload"]
            digest = event["payload_sha256"]
            prior = event["previous_digest"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("execution event journal is malformed") from exc
        if _canonical(event) != line:
            raise ValueError("execution event journal row is not canonical JSON")
        if sequence != expected_sequence or prior != previous:
            raise ValueError("execution event sequence or chain mismatch")
        if not isinstance(payload, dict) or digest != _event_digest(sequence, prior, payload):
            raise ValueError("execution event digest is invalid")
        events.append(event)
        previous = digest
    return events


def append_event(execution_root: Path, event: Mapping[str, Any]) -> None:
    """Append and fsync one canonical event in a contiguous digest chain."""
    root = _checked_path(Path(execution_root)).resolve(strict=True)
    path = _checked_path(root / "events.jsonl")
    existing = read_events(root)
    previous = existing[-1]["payload_sha256"] if existing else _GENESIS_DIGEST
    sequence = len(existing)
    payload = dict(event)
    digest = _event_digest(sequence, previous, payload)
    row = _canonical({
        "sequence": sequence,
        "previous_digest": previous,
        "payload": payload,
        "payload_sha256": digest,
    }) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        with os.fdopen(descriptor, "ab", closefd=False) as stream:
            stream.write(row)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def active_elapsed_seconds(context: ActiveJobContext, *, now_ns: int | None = None) -> int:
    now = time.monotonic_ns() if now_ns is None else now_ns
    elapsed_ns = max(0, now - context.started_monotonic_ns)
    return context.prior_active_seconds + (elapsed_ns + 999_999_999) // 1_000_000_000


def execution_stage_signature(execution_root: Path, stage: str, *, repo_root: Path) -> str:
    """Hash the prelaunch source, policy, config/lock, and implemented code surface."""
    if stage != "calibrate":
        raise ValueError(f"execution stage {stage} is not implemented for production")
    root = Path(execution_root)
    manifest = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    repo = Path(repo_root)
    from .study import CALIBRATION_FILES
    files = sorted(set(CALIBRATION_FILES) | {
        "src/ppmoe/distill/execution.py",
        "tests/test_distill_execution.py",
        "tests/test_distill_evaluate.py",
        "tests/test_distill_study.py",
    })
    code_signatures: dict[str, str] = {}
    for relative in files:
        path = _checked_path(repo / relative)
        if not path.is_file():
            raise ValueError(f"execution stage active file missing: {relative}")
        code_signatures[relative] = _sha_file(path)
    payload = {
        "stage": stage,
        "execution_id": manifest.get("execution_id"),
        "source_run_id": manifest.get("source_run_id"),
        "source_manifest_sha256": manifest.get("source_manifest_sha256"),
        "source_prepare_sha256": manifest.get("source_prepare_sha256"),
        "source_terminal_sha256": manifest.get("source_terminal_sha256"),
        "config_sha256": manifest.get("config_sha256"),
        "lock_sha256": manifest.get("lock_sha256"),
        "policy": manifest.get("policy"),
        "code_signatures": code_signatures,
    }
    return _sha(payload)


def process_creation_utc(pid: int) -> str | None:
    """Read a Windows process creation time without signaling or opening for write."""
    if os.name != "nt" or not isinstance(pid, int) or pid <= 0:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME),
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            creation = FILETIME()
            exit_time = FILETIME()
            kernel_time = FILETIME()
            user_time = FILETIME()
            if not kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                            ctypes.byref(kernel_time), ctypes.byref(user_time)):
                return None
            ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
            timestamp = datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=ticks // 10)
            return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
        finally:
            kernel32.CloseHandle(handle)
    except (AttributeError, OSError, OverflowError, ValueError, TypeError):
        return None


def load_active_job(execution_root: Path, attempt_id: str, *, stage: str) -> ActiveJobContext:
    """Load only a journal-authenticated active job owned by the live supervisor."""
    root = _checked_path(Path(execution_root)).resolve(strict=True)
    manifest_path = _checked_path(root / "execution.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest_bytes != _canonical(manifest) + b"\n":
        raise ValueError("execution manifest is not canonical JSON")
    execution_id = manifest.get("execution_id")
    unsigned_manifest = {key: value for key, value in manifest.items() if key != "execution_id"}
    if execution_id != _sha(unsigned_manifest) or root.name != execution_id:
        raise ValueError("execution manifest identity mismatch")
    events = read_events(root)
    starts = [
        event["payload"] for event in events
        if event["payload"].get("type") == "job_started"
        and event["payload"].get("attempt_id") == attempt_id
    ]
    if len(starts) != 1:
        raise ValueError("active job start event is missing or duplicated")
    payload = starts[0]
    if payload.get("execution_id") != execution_id or payload.get("stage") != stage:
        raise ValueError("active job execution or stage identity mismatch")
    job_path = _checked_path(root / "current-job.json")
    try:
        record_bytes = job_path.read_bytes()
        record = json.loads(record_bytes)
        job_payload = record["payload"]
        signature = record["payload_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("current active-job record is missing or malformed") from exc
    if record_bytes != _canonical(record) + b"\n":
        raise ValueError("current active-job record is not canonical JSON")
    if signature != _sha(job_payload) or job_payload != payload:
        raise ValueError("current active-job record signature mismatch")
    if any(
        event["payload"].get("type") in {"job_settled", "job_failed"}
        and event["payload"].get("attempt_id") == attempt_id
        for event in events
    ):
        raise ValueError("active job is already settled")
    active_attempts: set[str] = set()
    settled_attempts: set[str] = set()
    for event in events:
        event_payload = event["payload"]
        event_attempt = event_payload.get("attempt_id")
        if event_payload.get("type") == "job_started" and isinstance(event_attempt, str):
            active_attempts.add(event_attempt)
        elif event_payload.get("type") in {"job_settled", "job_failed"} and isinstance(event_attempt, str):
            settled_attempts.add(event_attempt)
    if active_attempts - settled_attempts != {attempt_id}:
        raise ValueError("execution has no unique current active job")
    pid = payload.get("supervisor_pid")
    created = payload.get("supervisor_created_utc")
    if type(pid) is not int or pid <= 0 or not isinstance(created, str):
        raise ValueError("active job supervisor identity is incomplete")
    if process_creation_utc(pid) != created:
        raise ValueError("active job supervisor is not the recorded live process")
    stage_pid = payload.get("stage_pid")
    stage_created = payload.get("stage_created_utc")
    if type(stage_pid) is not int or stage_pid != os.getpid() or not isinstance(stage_created, str):
        raise ValueError("active job is not owned by this contained stage process")
    if process_creation_utc(stage_pid) != stage_created:
        raise ValueError("active job stage process identity does not match this process")
    numeric = (payload.get("prior_active_seconds"), payload.get("started_monotonic_ns"))
    if any(type(value) is not int or value < 0 for value in numeric):
        raise ValueError("active job runtime fields are invalid")
    stage_signature = payload.get("stage_signature")
    if not isinstance(stage_signature, str) or len(stage_signature) != 64:
        raise ValueError("active job stage signature is invalid")
    return ActiveJobContext(
        execution_id=execution_id,
        attempt_id=attempt_id,
        stage=stage,
        prior_active_seconds=numeric[0],
        started_monotonic_ns=numeric[1],
        supervisor_pid=pid,
        supervisor_created_utc=created,
        stage_signature=stage_signature,
    )


__all__ = [
    "ActiveJobContext", "active_elapsed_seconds", "append_event", "execution_stage_signature",
    "load_active_job", "process_creation_utc", "read_events", "register_execution",
    "validate_execution",
]
