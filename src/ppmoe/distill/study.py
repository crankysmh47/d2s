"""Minimal one-stage preflight/prepare entry points for distillation v1."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

from .config import canonical_config, load_config
from .data import _manifest_jsonl, prepare_splits, sha256_file


DEFAULT_CONFIG = Path("configs/distillation-v1.toml")
FEASIBILITY_ROOT = Path("artifacts/distillation-v1/feasibility-20261003-recovery1")
FEASIBILITY_RESULT_SHA256 = "118a6c6733f5a7fd8fedcc47d2a4952bea6cfe7885059f87816703b560d06f57"
IMPLEMENTED_STAGES = {"preflight", "prepare"}
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
    repo_root = Path(__file__).resolve().parents[3]
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
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
