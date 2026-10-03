"""Minimal one-stage preflight/prepare entry points for distillation v1."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any, Mapping, Sequence

from .config import canonical_config, load_config
from .data import _manifest_jsonl, load_split, prepare_splits, sha256_file
from .evaluate import (
    RawRowJournal, ResourceLedger, ResourceLimitError, WorkerLease, calibrate_candidates,
    export_review_packets, preflight_disk,
)
from .model import WorkerOwnershipError, authenticate_model_snapshot, isolated_worker_factory, pinned_eos_token_ids
from .model import persist_model_inventory


DEFAULT_CONFIG = Path("configs/distillation-v1.toml")
FEASIBILITY_ROOT = Path("artifacts/distillation-v1/feasibility-20261003-recovery1")
FEASIBILITY_RESULT_SHA256 = "118a6c6733f5a7fd8fedcc47d2a4952bea6cfe7885059f87816703b560d06f57"
IMPLEMENTED_STAGES = {"preflight", "prepare", "calibrate"}
KNOWN_STAGES = IMPLEMENTED_STAGES | {
    "calibrate", "qualify", "generate", "audit", "train47", "evaluate47",
    "train48", "evaluate48", "seal", "report",
}
ACTIVE_FILES = (
    "src/ppmoe/distill/__init__.py",
    "src/ppmoe/distill/config.py",
    "src/ppmoe/distill/data.py",
    "src/ppmoe/distill/study.py",
    "scripts/run_distillation_v1.py",
    "tests/test_distill_config.py",
    "tests/test_distill_data.py",
    "tests/test_distill_study.py",
    "configs/distillation-v1.toml",
    "pyproject.toml",
    "uv.lock",
)
CALIBRATION_FILES = (
    "src/ppmoe/distill/__init__.py",
    "src/ppmoe/distill/config.py",
    "src/ppmoe/distill/data.py",
    "src/ppmoe/distill/study.py",
    "src/ppmoe/distill/model.py",
    "src/ppmoe/distill/answers.py",
    "src/ppmoe/distill/evaluate.py",
    "configs/distillation-v1.toml",
    "scripts/run_distillation_v1.py",
    "tests/test_distill_answers.py",
    "tests/test_distill_model.py",
    "tests/test_distill_evaluate.py",
    "tests/test_distill_study.py",
    "pyproject.toml",
    "uv.lock",
)
TASK1_RUN_ID = "acd8dfc400c8f079d0403da5d4b981b8aac8e94226db8760f632c8307ff785c2"
TASK1_MANIFEST_SHA256 = "bf0233bc65cd7f6554c7ab5269f17e051c31ff3903751ef7a24297ee203370ed"
TASK1_PREPARE_SHA256 = "970f1f669326ac26bb08af7c929180a61ba72915e5a1fb081b69914835ccfe9e"
TASK1_ORIGINATING_COMMIT = "4bd6d6ab0a66f4753799d45a6bc6f5bb4fb70c63"


class StageError(RuntimeError):
    """A registered stage is unavailable or its prerequisites are invalid."""


@dataclass(frozen=True, slots=True)
class StageRecord:
    stage: str
    run_id: str
    run_root: str
    status: str
    split_counts: dict[str, int]
    source_counts: dict[str, int]
    exclusion_counts: dict[str, int]
    details: dict[str, Any] | None = None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def stage_path(run_root: Path, stage: str) -> Path:
    if stage not in KNOWN_STAGES:
        raise StageError(f"unknown stage {stage}")
    return Path(run_root) / "stages" / f"{stage}.json"


def safe_output_path(root: Path, target: Path) -> Path:
    """Return an absolute output path after rejecting symlink/reparse traversal."""
    root_abs = Path(os.path.abspath(root))
    target_abs = Path(os.path.abspath(target))
    try:
        relative = target_abs.relative_to(root_abs)
    except ValueError as exc:
        raise ValueError("output path escapes run root") from exc
    current = Path(root_abs.anchor)
    for part in root_abs.parts[1:]:
        current /= part
        if current.exists() or current.is_symlink():
            _reject_link(current)
    current = root_abs
    for part in relative.parts:
        current /= part
        if current.exists() or current.is_symlink():
            _reject_link(current)
    return target_abs


def _reject_link(path: Path) -> None:
    info = path.lstat()
    reparse = bool(getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    if stat.S_ISLNK(info.st_mode) or reparse:
        raise ValueError(f"symlink/reparse output path rejected: {path.name}")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically commit one immutable JSON artifact without replacing an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    lock = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise FileExistsError(f"stage write already reserved: {path}") from None
    os.close(descriptor)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_canonical_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def validate_prepared_run(
    run_root: Path, *, runtime_identity: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Authenticate immutable prepare evidence without parsing any split labels."""
    root = Path(run_root)
    manifest_path = safe_output_path(root, root / "manifest.json")
    stage_file = safe_output_path(root, root / "stages" / "prepare.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage = json.loads(stage_file.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or stage.get("stage") != "prepare" or stage.get("status") != "passed":
        raise StageError("prepared run stage schema/status mismatch")
    if manifest.get("run_id") != root.name or stage.get("run_id") != root.name:
        raise StageError("prepared run ID/path mismatch")
    if not isinstance(manifest.get("source_signature"), dict):
        raise StageError("prepared source signature is missing")
    expected_run_id = sha256(_canonical_bytes(manifest["source_signature"])).hexdigest()
    if expected_run_id != root.name:
        raise StageError("prepared run ID does not hash its source signature")
    if stage.get("manifest_sha256") != sha256_file(manifest_path):
        raise StageError("prepared manifest bytes changed")
    expected_signature = sha256(_canonical_bytes(manifest.get("source_signature"))).hexdigest()
    if stage.get("source_signature_sha256") != expected_signature:
        raise StageError("prepared source signature mismatch")
    frozen_runtime = manifest.get("source_signature", {}).get("runtime_identity")
    if runtime_identity is not None and runtime_identity != frozen_runtime:
        raise StageError("prepared run Python/CUDA/device identity mismatch")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise StageError("prepared split manifest is missing")
    for entry in splits.values():
        relative = Path(entry["path"])
        path = safe_output_path(root, root / relative)
        if sha256_file(path) != entry["sha256"]:
            raise StageError(f"prepared bytes changed: {relative.name}")
        if len(path.read_bytes().splitlines()) != entry["count"]:
            raise StageError(f"prepared row count changed: {relative.name}")
        if "ordered_ids" in entry and len(entry["ordered_ids"]) != entry["count"]:
            raise StageError(f"prepared ordered-ID map changed: {relative.name}")
    if stage.get("retention_chunks_sha256") != manifest.get("retention", {}).get("chunks_sha256"):
        raise StageError("prepared retention signature mismatch")
    return manifest


def _file_signatures(repo_root: Path) -> dict[str, str]:
    signatures: dict[str, str] = {}
    for relative in ACTIVE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise StageError(f"active source file missing: {relative}")
        signatures[relative] = sha256_file(path)
    return signatures


def calibration_stage_signature(
    repo_root: Path, *, source_run_id: str, manifest_sha256: str, prepare_sha256: str,
    model_inventory_sha256: dict[str, str] | None = None,
    effective_eos_token_ids: Sequence[int] | None = None,
    originating_git_commit: str | None = None,
) -> dict[str, Any]:
    """Bind Task 2 execution code to the authenticated immutable Task 1 checkpoint."""
    code_signatures: dict[str, str] = {}
    for relative in CALIBRATION_FILES:
        path = Path(repo_root) / relative
        if not path.is_file():
            raise StageError(f"calibration source file missing: {relative}")
        code_signatures[relative] = sha256_file(path)
    signature: dict[str, Any] = {
        "stage": "calibrate",
        "source_run_id": source_run_id,
        "predecessors": {
            "prepare_manifest_sha256": manifest_sha256,
            "prepare_stage_sha256": prepare_sha256,
        },
        "model_inventory_sha256": model_inventory_sha256 or {},
        "effective_eos_token_ids": list(effective_eos_token_ids or ()),
        "task1_originating_git_commit": TASK1_ORIGINATING_COMMIT,
        "originating_git_commit": originating_git_commit or _git_commit(repo_root),
        "code_signatures": code_signatures,
    }
    signature["stage_signature"] = sha256(_canonical_bytes(signature)).hexdigest()
    return signature


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.lower()):
        raise StageError("current Git commit identity is invalid")
    return commit


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _authenticated_private_gold(root: Path, manifest: Mapping[str, Any], expected_ids: Sequence[str]) -> dict[str, str]:
    """Read calibration labels only after budget selection, from authenticated prepared bytes."""
    entry = manifest.get("splits", {}).get("calibration", {})
    relative = Path(str(entry.get("path", "")))
    path = safe_output_path(root, root / relative)
    if not path.is_file() or sha256_file(path) != entry.get("sha256"):
        raise StageError("authenticated calibration source bytes changed before private gold mapping")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if [record.get("id") for record in records] != list(expected_ids):
        raise StageError("private calibration gold IDs differ from authenticated ID order")
    if any(not isinstance(record.get("gold"), str) for record in records):
        raise StageError("authenticated calibration source lacks private gold values")
    return {str(record["id"]): str(record["gold"]) for record in records}


def _append_stage_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(_canonical_bytes(event) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_calibration_stage(
    run_root: Path, config: Any, *, repo_root: Path, config_path: Path,
    worker_loader: Any | None = None,
) -> dict[str, Any]:
    """Run only the gold-free blind calibration against the reviewed Task 1 run."""
    root = Path(run_root)
    if root.name != TASK1_RUN_ID:
        raise StageError("calibrate requires the reviewed Task 1 prepared run ID")
    if stage_path(root, "calibrate").exists():
        raise FileExistsError("calibrate stage is immutable and already exists")
    reservation_path = safe_output_path(root, root / "stages" / "calibrate-reservations.jsonl")
    history: list[dict[str, Any]] = []
    prior = []
    if reservation_path.exists():
        history = [json.loads(line) for line in reservation_path.read_text(encoding="utf-8").splitlines()]
        if any(item.get("event_type") not in {"reservation", "terminal", "download_reservation", "download_complete"} for item in history):
            raise StageError("calibration reservation journal has an invalid event")
        prior = [item for item in history if item.get("event_type") == "reservation"]
        if len(prior) > config.resources.phase_retries:
            # A terminal stop is still persisted below; no worker is loaded.
            exhausted_before_worker = True
        else:
            exhausted_before_worker = False
    else:
        exhausted_before_worker = False
    origin_event = next((item for item in history if item.get("originating_git_commit")), None)
    originating_commit = origin_event.get("originating_git_commit") if origin_event else _git_commit(repo_root)
    if not isinstance(originating_commit, str):
        raise StageError("calibration originating Git commit is missing")
    started_at = _utc_now()
    # This authenticates bytes against Task 1's stored signature; no source scan
    # or run ID recomputation from the evolving Task 2 files occurs here.
    runtime_identity = feasibility_runtime_identity(
        repo_root, vram_reserve_bytes=config.resources.vram_reserve_bytes
    )
    manifest = validate_prepared_run(root, runtime_identity=runtime_identity)
    manifest_path = root / "manifest.json"
    prepare_path = stage_path(root, "prepare")
    manifest_hash = sha256_file(manifest_path)
    prepare_hash = sha256_file(prepare_path)
    if manifest_hash != TASK1_MANIFEST_SHA256 or prepare_hash != TASK1_PREPARE_SHA256:
        raise StageError("calibration Task 1 manifest/prepare hashes do not match reviewed evidence")
    if manifest.get("config") != canonical_config(config):
        raise StageError("calibration config differs from prepared Task 1 config")
    calibration_entry = manifest.get("splits", {}).get("calibration")
    if not isinstance(calibration_entry, dict) or calibration_entry.get("count") != 60:
        raise StageError("prepared manifest does not register exactly 60 calibration IDs")
    expected_ids = calibration_entry.get("ordered_ids")
    if not isinstance(expected_ids, list) or len(expected_ids) != 60 or len(set(expected_ids)) != 60:
        raise StageError("prepared calibration ordered-ID map is invalid")
    preflight_disk(root, reserve_bytes=2 * 1024**3)
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    inventory_directory = safe_output_path(root, root / "calibration" / "model-inventory")
    inventories: dict[str, dict[str, Any]] = {}
    snapshot_paths: dict[str, Path] = {}
    inventory_hashes: dict[str, str] = {}
    downloaded_bytes = 0
    for role, pin in (("student", config.student), ("teacher", config.teacher)):
        metadata = api.model_info(pin.repo_id, revision=pin.revision, files_metadata=True)
        if getattr(metadata, "sha", None) != pin.revision:
            raise StageError(f"Hub pin authentication failed for {role}")
        if role == "student":
            expected_names = {item.rfilename for item in metadata.siblings}
            cache_path = (
                repo_root / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots" / pin.revision
            )
            cache_names = {
                path.relative_to(cache_path).as_posix()
                for path in cache_path.rglob("*") if path.is_file()
            } if cache_path.is_dir() else set()
            if cache_names != expected_names:
                estimated = sum(int(getattr(item, "size", 0) or 0) for item in metadata.siblings)
                if ResourceLedger.initial().download_bytes + downloaded_bytes + estimated > config.resources.download_bytes:
                    raise StageError("pinned student download would exceed the registered download cap")
                if shutil.disk_usage(repo_root).free < estimated + 2 * 1024**3:
                    raise StageError("insufficient disk for the pinned student model snapshot")
                downloaded_bytes += estimated
                download_started = _utc_now()
                _append_stage_event(reservation_path, {
                    "event_type": "download_reservation", "status": "reserved",
                    "event_id": uuid4().hex, "stage": "calibrate", "role": role,
                    "repo_id": pin.repo_id, "revision": pin.revision,
                    "reserved_bytes": estimated, "started_at_utc": download_started,
                    "originating_git_commit": originating_commit,
                    "code_signatures": _file_signatures(repo_root),
                })
            snapshot = Path(snapshot_download(
                repo_id=pin.repo_id, revision=pin.revision, local_files_only=(cache_names == expected_names),
                cache_dir=str(repo_root / ".cache" / "huggingface" / "hub"),
            ))
            if cache_names != expected_names:
                _append_stage_event(reservation_path, {
                    "event_type": "download_complete", "status": "completed",
                    "event_id": uuid4().hex, "stage": "calibrate", "role": role,
                    "repo_id": pin.repo_id, "revision": pin.revision,
                    "started_at_utc": download_started, "ended_at_utc": _utc_now(),
                    "originating_git_commit": originating_commit,
                })
        else:
            snapshot = (
                repo_root / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots" / pin.revision
            )
            if not snapshot.is_dir():
                raise StageError("pinned teacher snapshot is unavailable in the registered local cache")
        inventory = authenticate_model_snapshot(
            pin.repo_id, pin.revision, snapshot, metadata=metadata,
        )
        snapshot_paths[role] = snapshot
        inventories[role] = inventory
        inventory_hashes[role] = sha256(_canonical_bytes(inventory)).hexdigest()
    eos_ids = pinned_eos_token_ids(snapshot_paths["student"])
    if pinned_eos_token_ids(snapshot_paths["teacher"]) != eos_ids:
        raise StageError("student and teacher pinned generation EOS IDs differ")
    signature_record = calibration_stage_signature(
        repo_root, source_run_id=TASK1_RUN_ID,
        manifest_sha256=manifest_hash, prepare_sha256=prepare_hash,
        model_inventory_sha256=inventory_hashes,
        effective_eos_token_ids=eos_ids,
        originating_git_commit=originating_commit,
    )
    current_active_signatures = _file_signatures(repo_root)
    for event in history:
        frozen_active = event.get("code_signatures")
        if isinstance(frozen_active, dict) and frozen_active != current_active_signatures:
            raise StageError("calibration active code/config bytes drifted from the first durable reservation")
    if prior:
        frozen_signature = prior[0].get("signature")
        if not isinstance(frozen_signature, dict):
            raise StageError("calibration recovery lacks its frozen source signature")
        if frozen_signature.get("code_signatures") != signature_record.get("code_signatures"):
            raise StageError("calibration source/config bytes drifted from first reservation")
        if frozen_signature.get("model_inventory_sha256") != signature_record.get("model_inventory_sha256"):
            raise StageError("calibration pinned model bytes drifted from first reservation")
        if frozen_signature.get("stage_signature") != signature_record.get("stage_signature"):
            raise StageError("calibration recovery signature differs from first reservation")
    stage_signature = signature_record["stage_signature"]
    free_before = preflight_disk(root, reserve_bytes=2 * 1024**3)
    rows = load_split(root, "calibration")
    if len(rows) != 60 or [row["id"] for row in rows] != expected_ids:
        raise StageError("loaded calibration rows differ from the authenticated manifest ID order")
    if any(set(row) != {"id", "dataset", "source_split", "question"} for row in rows):
        raise StageError("calibration loader returned unexpected fields")

    raw_directory = safe_output_path(root, root / "calibration" / "raw")
    raw_directory.mkdir(parents=True, exist_ok=True)
    attempts_path = safe_output_path(root, root / "stages" / "calibrate-attempts.jsonl")
    reservation = {
        "status": "reserved", "event_type": "reservation", "event_id": uuid4().hex, "stage": "calibrate", "stage_signature": stage_signature,
        "predecessor_hashes": signature_record["predecessors"],
        "started_at_utc": started_at,
        "originating_git_commit": originating_commit,
        "signature": signature_record,
        "initial_cuda_seconds": 1168, "initial_download_bytes": 8066699435,
        "download_bytes": downloaded_bytes,
        "cuda_cap_seconds": config.resources.cuda_seconds,
        "wall_cap_seconds": config.resources.wall_seconds,
        "artifact_cap_bytes": config.resources.artifact_bytes,
        "disk_free_bytes": free_before,
    }
    _append_stage_event(reservation_path, reservation)
    for role, inventory in inventories.items():
        persist_model_inventory(inventory, inventory_directory / f"{role}.json")

    def journal_for(role: str, candidate: Sequence[int]) -> RawRowJournal:
        name = f"{int(candidate[0])}-{int(candidate[1])}-{role}.jsonl"
        return RawRowJournal(
            raw_directory / name,
            expected_ids=expected_ids,
            stage_signature=stage_signature,
        )

    def row_sink(row: Mapping[str, Any]) -> None:
        if row.get("status") in {"failed", "worker_load_failed"}:
            role = str(row.get("role", "worker"))
            candidate = row.get("candidate", (0, 0))
            journal_for(role, candidate).record_failure(row)
        else:
            journal_for(str(row["role"]), row["candidate"]).append(row)

    class FailureSink:
        def record_attempt(self, row: Mapping[str, Any]) -> None:
            role = str(row["role"])
            journal_for(role, row["candidate"]).record_attempt(row)

        def record_failure(self, row: Mapping[str, Any]) -> None:
            role = str(row.get("role", "worker"))
            candidate = row.get("candidate", (0, 0))
            journal_for(role, candidate).record_failure(row)

    def row_loader(role: str, candidate: Sequence[int]):
        return journal_for(role, candidate).read()

    def failure_count_loader(role: str, candidate: Sequence[int]) -> int:
        # Retry budget is aggregate per role over all candidates and recovery.
        total = 0
        for path in raw_directory.glob(f"*-*-{role}.jsonl"):
            journal = RawRowJournal(path, expected_ids=expected_ids, stage_signature=stage_signature)
            total += journal.failure_count()
        return total

    recovered_ledger = ResourceLedger.initial()
    prior_downloads = sum(int(item.get("reserved_bytes", 0)) for item in history if item.get("event_type") == "download_reservation") + downloaded_bytes
    for raw_path in raw_directory.glob("*.jsonl"):
        if raw_path.name.endswith(".attempts.jsonl"):
            continue
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            startup = float(raw.get("worker_startup_wall_seconds", 0.0))
            if startup:
                recovered_ledger = recovered_ledger.charge_worker(
                    cuda_seconds=None, wall_seconds=startup, download_bytes=0, artifact_bytes=0,
                )
            recovered_ledger = recovered_ledger.charge_worker(
                cuda_seconds=raw.get("cuda_seconds"),
                wall_seconds=float(raw.get("wall_seconds", 0.0)),
                download_bytes=0,
                artifact_bytes=len(line.encode("utf-8")) + 1,
            )
    for attempt_path in raw_directory.glob("*.attempts.jsonl"):
        raw_path = Path(str(attempt_path)[:-len(".attempts.jsonl")])
        parts = raw_path.stem.split("-")
        if len(parts) < 3 or parts[-1] not in {"student", "teacher"}:
            raise StageError("calibration attempt journal filename is invalid")
        candidate = [int(parts[0]), int(parts[1])]
        journal = RawRowJournal(raw_path, expected_ids=expected_ids, stage_signature=stage_signature)
        events = journal.attempts()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for index, event in enumerate(events):
            grouped.setdefault(str(event.get("attempt_id", f"legacy-{index}")), []).append(event)
        for attempt_events in grouped.values():
            if any(event.get("status") == "worker_complete" for event in attempt_events):
                continue
            failed = [event for event in attempt_events if event.get("status") in {"failed", "worker_load_failed"}]
            if failed:
                event = failed[-1]
                recovered_ledger = recovered_ledger.charge_worker(
                    cuda_seconds=event.get("cuda_seconds"),
                    wall_seconds=float(event.get("worker_startup_wall_seconds", 0.0)) + float(event.get("wall_seconds", 0.0)),
                    download_bytes=0, artifact_bytes=0,
                )
                continue
            reservation_event = next((event for event in attempt_events if event.get("status") == "worker_reserved"), None)
            if reservation_event is None:
                raise StageError("incomplete calibration attempt lacks a durable reservation")
            reserved = datetime.fromisoformat(str(reservation_event["reserved_at_utc"]).replace("Z", "+00:00"))
            elapsed = max(0.0, (datetime.now(timezone.utc) - reserved).total_seconds())
            known = 0.0
            raw_startup = 0.0
            for raw in journal.read():
                if raw.get("attempt_id") == reservation_event.get("attempt_id"):
                    raw_startup += float(raw.get("worker_startup_wall_seconds", 0.0))
                    known += float(raw.get("wall_seconds", 0.0))
            known += raw_startup
            if not raw_startup:
                known += sum(float(event.get("startup_wall_seconds", 0.0)) for event in attempt_events if event.get("status") == "worker_started")
            uncommitted = max(0.0, elapsed - known)
            if uncommitted:
                recovered_ledger = recovered_ledger.charge_worker(
                    cuda_seconds=None, wall_seconds=uncommitted,
                    download_bytes=0, artifact_bytes=0,
                )
    recovered_ledger = recovered_ledger.charge_worker(
        cuda_seconds=0, wall_seconds=0, download_bytes=prior_downloads, artifact_bytes=0,
    )
    current_artifacts = sum(
        path.stat().st_size
        for base in (root / "calibration", root / "stages")
        for path in (base.rglob("*") if base.exists() else ())
        if path.is_file() and (base == root / "calibration" or path.name.startswith("calibrate"))
    )
    recovered_ledger = replace(recovered_ledger, artifact_bytes=current_artifacts)

    try:
        with WorkerLease(root / "stages" / "calibrate.worker.lock") as lease:
            loader = worker_loader or (
                lambda role, candidate: isolated_worker_factory(
                    role, config, repo_root=repo_root, snapshot_path=snapshot_paths[role],
                    tokenizer_snapshot_path=snapshot_paths["student"], eos_token_ids=eos_ids,
                    child_started=lease.record_child,
                )
            )
            if exhausted_before_worker:
                result = {"status": "stopped_environment", "candidate": None, "attempts": [], "reason": "aggregate calibration phase retry budget exhausted"}
            else:
                result = calibrate_candidates(
                    rows,
                    candidates=config.prompt.candidates,
                    worker_factory=loader,
                    minimum_eos=config.qualification.minimum_calibration_eos,
                    stage_signature=stage_signature,
                    row_sink=row_sink,
                    row_loader=row_loader,
                    failure_count_loader=failure_count_loader,
                    failure_sink=FailureSink(),
                    ledger=recovered_ledger,
                    max_retries=config.resources.phase_retries,
                )
    except (ResourceLimitError, WorkerOwnershipError) as exc:
        result = {
            "status": "stopped_environment", "candidate": None, "attempts": [],
            "reason": f"{type(exc).__name__}: {exc}",
            "resource_ledger": {
                name: getattr(recovered_ledger, name)
                for name in recovered_ledger.__dataclass_fields__
            },
        }
    except Exception as exc:
        _append_stage_event(attempts_path, {
            "status": "failed", "stage_signature": stage_signature,
            "error_type": type(exc).__name__,
        })
        raise

    attempts_metadata = []
    for attempt in result.get("attempts", []):
        candidate = attempt["candidate"]
        role_rows = []
        for role in ("student", "teacher"):
            journal = journal_for(role, candidate)
            if journal.path.exists():
                role_rows.append({"role": role, "path": str(journal.path.relative_to(root)), "sha256": sha256_file(journal.path), "count": len(journal.read())})
        attempts_metadata.append({key: value for key, value in attempt.items() if key != "rows"} | {"raw_rows": role_rows})
    result["attempts"] = attempts_metadata
    result["stage_signature"] = stage_signature
    result["signature"] = signature_record
    result["task1_originating_commit"] = TASK1_ORIGINATING_COMMIT
    result["originating_git_commit"] = originating_commit
    result["effective_eos_token_ids"] = list(eos_ids)
    result["resource_caps"] = {
        "cuda_seconds": config.resources.cuda_seconds,
        "wall_seconds": config.resources.wall_seconds,
        "download_bytes": config.resources.download_bytes,
        "artifact_bytes": config.resources.artifact_bytes,
    }
    _append_stage_event(attempts_path, {
        "status": result["status"], "stage_signature": stage_signature,
        "attempt_count": len(attempts_metadata),
    })
    if result["status"] == "awaiting_review" and result.get("candidate"):
        selected_rows = []
        for role in ("student", "teacher"):
            selected_rows.extend(journal_for(role, result["candidate"]).read())
        review_directory = safe_output_path(root, root / "calibration" / "review")
        private_gold = _authenticated_private_gold(root, manifest, expected_ids)
        review = export_review_packets(rows, selected_rows, review_directory, seed=stage_signature, private_gold_by_id=private_gold)
        result["review_packet"] = str(Path(review["packets"][0]).relative_to(root))
        result["review_packet_sha256"] = review["packet_sha256"]
        result["private_review_map"] = str(Path(review["private_map"]).relative_to(root))
    if "resource_ledger" not in result:
        result["resource_ledger"] = {
            name: getattr(recovered_ledger, name)
            for name in recovered_ledger.__dataclass_fields__
        }
    actual_artifacts = sum(
        path.stat().st_size
        for base in (root / "calibration", root / "stages")
        for path in (base.rglob("*") if base.exists() else ())
        if path.is_file() and (base == root / "calibration" or path.name.startswith("calibrate"))
    )
    projected = actual_artifacts + len(_canonical_bytes(result)) + 1
    while True:
        result["resource_ledger"]["artifact_bytes"] = projected
        updated = actual_artifacts + len(_canonical_bytes(result)) + 1
        if updated == projected:
            break
        projected = updated
    if projected > config.resources.artifact_bytes:
        raise StageError("calibration artifacts exceed the registered artifact-byte cap")
    if result["resource_ledger"]["cuda_seconds"] > config.resources.cuda_seconds:
        raise StageError("calibration CUDA use exceeds the registered cap")
    if result["resource_ledger"]["wall_seconds"] > config.resources.wall_seconds:
        raise StageError("calibration wall time exceeds the registered cap")
    if result["resource_ledger"]["download_bytes"] > config.resources.download_bytes:
        raise StageError("calibration downloads exceed the registered cap")
    if result["status"] != "interrupted":
        ended_at = _utc_now()
        _append_stage_event(reservation_path, {
            "status": result["status"], "event_type": "terminal", "event_id": uuid4().hex,
            "stage_signature": stage_signature, "originating_git_commit": originating_commit,
            "started_at_utc": started_at, "ended_at_utc": ended_at,
        })
        result["started_at_utc"] = started_at
        result["ended_at_utc"] = ended_at
        atomic_write_json(stage_path(root, "calibrate"), result)
    return result


def validate_runtime_metadata(
    environment: dict[str, Any], preflight: dict[str, Any], *, vram_reserve_bytes: int
) -> dict[str, Any]:
    """Select stable Python/CUDA/device fields and validate the registered reserve."""
    torch_environment = environment.get("torch_environment", {})
    device = environment.get("device_snapshot", {})
    device_fields = str(device.get("stdout", "")).split(",")
    capacity = torch_environment.get("physical_capacity_bytes")
    memory_cap = preflight.get("memory_cap_bytes")
    if device.get("status") != "available" or not torch_environment.get("cuda_available"):
        raise StageError("feasibility environment has no available CUDA device")
    if not isinstance(capacity, int) or not isinstance(memory_cap, int):
        raise StageError("feasibility environment memory metadata is incomplete")
    if capacity - vram_reserve_bytes != memory_cap:
        raise StageError("feasibility VRAM cap does not match the registered reserve")
    if len(device_fields) < 5 or not torch_environment.get("gpu_name"):
        raise StageError("feasibility device identity is incomplete")
    packages = environment.get("packages", {})
    if packages.get("torch") != torch_environment.get("version"):
        raise StageError("feasibility torch package identities disagree")
    return {
        "python": environment.get("python"),
        "platform": environment.get("platform"),
        "packages": packages,
        "cuda_runtime": torch_environment.get("cuda_runtime"),
        "cuda_available": torch_environment.get("cuda_available"),
        "compute_capability": torch_environment.get("capability"),
        "bf16_supported": torch_environment.get("bf16_supported"),
        "gpu_name": torch_environment["gpu_name"],
        "physical_capacity_bytes": capacity,
        "memory_cap_bytes": memory_cap,
        "vram_reserve_bytes": vram_reserve_bytes,
        "driver_version": device_fields[4].strip(),
    }


def feasibility_runtime_identity(repo_root: Path, *, vram_reserve_bytes: int) -> dict[str, Any]:
    """Load authenticated environment metadata from the passed Task 0 feasibility run."""
    root = Path(repo_root) / FEASIBILITY_ROOT
    result_path = root / "result.json"
    environment_path = root / "environment.json"
    preflight_path = root / "preflight.json"
    if sha256_file(result_path) != FEASIBILITY_RESULT_SHA256:
        raise StageError("passed feasibility result hash changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if result.get("status") != "passed":
        raise StageError("Task 0 feasibility did not pass")
    environment_sha = sha256_file(environment_path)
    preflight_sha = sha256_file(preflight_path)
    if result.get("environment_sha256") != environment_sha or result.get("preflight_sha256") != preflight_sha:
        raise StageError("Task 0 environment evidence hash mismatch")
    identity = validate_runtime_metadata(environment, preflight, vram_reserve_bytes=vram_reserve_bytes)
    return {
        **identity,
        "feasibility_result_sha256": FEASIBILITY_RESULT_SHA256,
        "environment_sha256": environment_sha,
        "preflight_sha256": preflight_sha,
    }


def _prepared_source(config_path: Path, repo_root: Path) -> tuple[str, dict[str, Any], dict[str, Any], Any]:
    config = load_config(config_path)
    source_metadata, split_manifest = prepare_splits(config, repo_root)
    signatures = _file_signatures(repo_root)
    runtime_identity = feasibility_runtime_identity(
        repo_root, vram_reserve_bytes=config.resources.vram_reserve_bytes
    )
    split_hashes = {
        name: {
            "count": len(rows),
            "ordered_ids": [record.id for record in rows],
            "ordered_id_sha256": sha256("\n".join(record.id for record in rows).encode()).hexdigest(),
        }
        for name, rows in split_manifest.splits.items()
    }
    source_signature = {
        "config": canonical_config(config),
        "sources": source_metadata["sources"],
        "retention": source_metadata["retention"],
        "runtime_identity": runtime_identity,
        "source_counts": source_metadata["source_counts"],
        "exclusion_counts": source_metadata["exclusion_counts"],
        "splits": split_hashes,
        "code_signatures": signatures,
    }
    run_id = sha256(_canonical_bytes(source_signature)).hexdigest()
    return run_id, source_signature, source_metadata, split_manifest


def run_stage(
    stage: str,
    config_path: Path = DEFAULT_CONFIG,
    run_root: Path | None = None,
) -> StageRecord:
    if stage not in KNOWN_STAGES:
        raise StageError(f"unknown stage {stage}")
    if stage not in IMPLEMENTED_STAGES:
        raise StageError(f"stage {stage} is registered but not implemented")
    if stage == "prepare" and run_root is None:
        raise StageError("prepare requires an explicit run-root")
    repo_root = Path(__file__).resolve().parents[3]
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    if stage == "calibrate":
        if run_root is None:
            raise StageError("calibrate requires an explicit run-root")
        selected_root = Path(run_root)
        if not selected_root.is_absolute():
            selected_root = repo_root / selected_root
        selected_root = safe_output_path(selected_root.parent, selected_root)
        config = load_config(config_path)
        result = run_calibration_stage(
            selected_root, config, repo_root=repo_root, config_path=config_path,
        )
        manifest = json.loads((selected_root / "manifest.json").read_text(encoding="utf-8"))
        return StageRecord(
            stage, selected_root.name, str(selected_root), result["status"],
            manifest.get("split_counts", {}), manifest.get("source_counts", {}),
            manifest.get("exclusion_counts", {}), result,
        )
    run_id, signature, source_metadata, splits = _prepared_source(config_path, repo_root)
    default_root = repo_root / "artifacts" / "distillation-v1" / run_id
    selected_root = Path(run_root) if run_root is not None else default_root
    if not selected_root.is_absolute():
        selected_root = repo_root / selected_root
    selected_root = safe_output_path(selected_root.parent, selected_root)
    if selected_root.name != run_id:
        raise StageError("explicit run-root must end in the prepared full run ID")
    counts = {name: len(rows) for name, rows in splits.splits.items()}
    source_counts = dict(source_metadata["source_counts"])
    exclusion_counts = dict(source_metadata["exclusion_counts"])
    if stage == "preflight":
        return StageRecord(stage, run_id, str(selected_root), "preflight_ok", counts, source_counts, exclusion_counts)
    if selected_root.exists():
        raise FileExistsError(f"prepared run already exists: {selected_root}")
    selected_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=selected_root.parent))
    try:
        entries = {
            name: _manifest_jsonl(temporary_root, name, rows)
            for name, rows in splits.splits.items()
        }
        retention_source = repo_root / load_config(config_path).retention_chunks
        retention_copy = temporary_root / "retention" / "retention-chunks.jsonl"
        retention_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(retention_source, retention_copy)
        if sha256_file(retention_copy) != source_metadata["retention"]["chunks_sha256"]:
            raise StageError("copied retention chunks do not match authenticated source")
        entries["retention_chunks"] = {
            "path": "retention/retention-chunks.jsonl",
            "sha256": source_metadata["retention"]["chunks_sha256"],
            "count": source_metadata["retention"]["chunk_count"],
            "ordered_ids": list(range(source_metadata["retention"]["chunk_count"])),
        }
        exclusion_path = temporary_root / "data" / "exclusions.jsonl"
        exclusion_path.parent.mkdir(parents=True, exist_ok=True)
        with exclusion_path.open("xb") as stream:
            for item in splits.exclusions:
                stream.write(_canonical_bytes({"id": item.id, "dataset": item.dataset, "reason": item.reason}) + b"\n")
        entries["exclusions"] = {
            "path": "data/exclusions.jsonl",
            "sha256": sha256_file(exclusion_path),
            "count": len(splits.exclusions),
            "ordered_ids": [item.id for item in splits.exclusions],
        }
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "config": signature["config"],
            "source_signature": signature,
            "sources": source_metadata["sources"],
            "retention": source_metadata["retention"],
            "source_counts": source_counts,
            "exclusion_counts": exclusion_counts,
            "splits": entries,
            "split_counts": counts,
        }
        atomic_write_json(temporary_root / "manifest.json", manifest)
        manifest_sha = sha256_file(temporary_root / "manifest.json")
        atomic_write_json(
            stage_path(temporary_root, "prepare"),
            {
                "stage": "prepare",
                "status": "passed",
                "run_id": run_id,
                "manifest_sha256": manifest_sha,
                "source_signature_sha256": sha256(_canonical_bytes(signature)).hexdigest(),
                "retention_chunks_sha256": source_metadata["retention"]["chunks_sha256"],
                "source_counts": source_counts,
                "split_counts": counts,
                "exclusion_counts": exclusion_counts,
            },
        )
        # Atomic directory rename is the prepare checkpoint commit.
        os.rename(temporary_root, selected_root)
    except Exception:
        for child in sorted(temporary_root.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        temporary_root.rmdir()
        raise
    validate_prepared_run(selected_root)
    return StageRecord(stage, run_id, str(selected_root), "prepared", counts, source_counts, exclusion_counts)
