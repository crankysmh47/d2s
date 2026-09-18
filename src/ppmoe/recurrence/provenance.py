"""Canonical, immutable provenance primitives for recurrence V0."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from threading import Lock
from typing import Mapping

from .config import RecurrenceConfig


PROTOCOL_REVISION = "0615d0d9eac985c7ab5b0e019ab1afcea1cbe979"

_PATH_LOCKS_GUARD = Lock()
_PATH_LOCKS: dict[Path, Lock] = {}
_FULL_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_FULL_REVISION = re.compile(r"[0-9a-fA-F]{40}\Z")
_EXPECTED_QWEN3_CONFIG: dict[str, object] = {
    "architectures": ["Qwen3ForCausalLM"],
    "num_hidden_layers": 28,
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One exact file captured from a pinned source snapshot."""

    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """The source inputs and host metadata that define a recurrence run."""

    schema_version: int
    repo_id: str
    revision: str
    model_class: str
    files: tuple[SourceFile, ...]
    license_relative_path: str
    license_sha256: str
    python_version: str
    torch_version: str
    transformers_version: str
    gpu_name: str
    gpu_vram_bytes: int
    gpu_driver_version: str
    protocol_revision: str
    config_sha256: str
    code_sha256: str
    code_files: tuple[SourceFile, ...]
    recovery_revision: str | None


def canonical_json(payload: Mapping[str, object]) -> bytes:
    """Return the one unambiguous JSON representation used in evidence hashes."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_payload(payload: Mapping[str, object]) -> str:
    """Hash canonical JSON rather than caller-dependent mapping order."""
    return sha256(canonical_json(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of exact on-disk evidence bytes."""
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_revision(value: str) -> None:
    """Reject mutable model refs; reproducible sources require a complete commit SHA."""
    if not isinstance(value, str) or _FULL_REVISION.fullmatch(value) is None:
        raise ValueError("model revision must be a full immutable 40-hex commit SHA")


def resolve_model_revision(
    repo_id: str, requested_revision: str, *, api: object | None = None
) -> str:
    """Resolve a requested ref once, before any model loader may observe it."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    model_info = getattr(api, "model_info")
    metadata = model_info(repo_id, revision=requested_revision)
    revision = getattr(metadata, "sha", None)
    validate_revision(revision)
    return revision.lower()


def validate_qwen3_config(config_dict: Mapping[str, object]) -> None:
    """Assert the loaded model's public architecture matches V0 exactly."""
    for field, expected in _EXPECTED_QWEN3_CONFIG.items():
        if config_dict.get(field) != expected:
            raise ValueError(
                f"Qwen3 source config mismatch for {field}: "
                f"expected {expected!r}, got {config_dict.get(field)!r}"
            )
    embedding_vocab_size = config_dict.get("embedding_vocab_size")
    if embedding_vocab_size is not None and embedding_vocab_size != config_dict["vocab_size"]:
        raise ValueError(
            "Qwen3 source config mismatch for embedding_vocab_size: "
            f"expected {config_dict['vocab_size']!r}, got {embedding_vocab_size!r}"
        )


def compute_run_id(
    source_hash: str, data_hash: str, config_hash: str, code_hash: str
) -> str:
    """Derive the stable short identifier from all four finalized evidence hashes."""
    hashes = {
        "source_hash": source_hash,
        "data_hash": data_hash,
        "config_hash": config_hash,
        "code_hash": code_hash,
    }
    for name, value in hashes.items():
        if not isinstance(value, str) or _FULL_SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be a full 64-hex SHA-256 hash")
    return sha256_payload(hashes)[:12]


def build_source_manifest(
    config: RecurrenceConfig,
    snapshot_dir: Path,
    output_path: Path,
    *,
    recovery_revision: str | None = None,
) -> dict[str, object]:
    """Hash a resolved local model snapshot and persist its immutable source record."""
    if config.model.repo_id != "Qwen/Qwen3-1.7B":
        raise ValueError("source manifest only supports Qwen/Qwen3-1.7B")
    if config.model.model_class != "Qwen3ForCausalLM":
        raise ValueError("source manifest only supports Qwen3ForCausalLM")
    if snapshot_dir.is_symlink():
        raise ValueError("source snapshot must be materialized, not a symbolic link")
    snapshot_root = snapshot_dir.resolve(strict=True)
    if not snapshot_root.is_dir():
        raise ValueError(f"snapshot path is not a directory: {snapshot_root}")
    revision = snapshot_root.name
    try:
        validate_revision(revision)
    except ValueError as error:
        raise ValueError("snapshot path does not identify its immutable revision") from error
    destination = output_path.resolve()
    try:
        destination.relative_to(snapshot_root)
    except ValueError:
        pass
    else:
        raise ValueError("source manifest output cannot be inside the model snapshot")
    if recovery_revision is not None:
        validate_revision(recovery_revision)

    files = _snapshot_files(snapshot_root)
    _validate_snapshot_inventory(files)
    config_path = snapshot_root / "config.json"
    if not config_path.is_file():
        raise ValueError("source snapshot is missing config.json")
    try:
        source_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("source snapshot contains invalid config.json") from error
    if not isinstance(source_config, Mapping):
        raise ValueError("source snapshot config.json must be a JSON object")
    validate_qwen3_config(source_config)

    license_file = next(
        (entry for entry in files if Path(entry.relative_path).name.casefold().startswith("license")),
        None,
    )
    if license_file is None:
        raise ValueError("source snapshot is missing a license file")
    code_files = _runtime_code_files()
    gpu = _gpu_metadata()
    manifest = SourceManifest(
        schema_version=1,
        repo_id="Qwen/Qwen3-1.7B",
        revision=revision.lower(),
        model_class="Qwen3ForCausalLM",
        files=files,
        license_relative_path=license_file.relative_path,
        license_sha256=license_file.sha256,
        python_version=sys.version,
        torch_version=_torch_version(),
        transformers_version=_transformers_version(),
        gpu_name=gpu["name"],
        gpu_vram_bytes=gpu["vram_bytes"],
        gpu_driver_version=gpu["driver_version"],
        protocol_revision=_git_revision(),
        config_sha256=sha256_payload(config.scientific_payload()),
        code_sha256=sha256_payload({entry.relative_path: entry.sha256 for entry in code_files}),
        code_files=code_files,
        recovery_revision=recovery_revision.lower() if recovery_revision is not None else None,
    )
    payload = json.loads(canonical_json(asdict(manifest)))
    write_immutable_json(output_path, payload)
    return payload


def _snapshot_files(snapshot_root: Path) -> tuple[SourceFile, ...]:
    records: list[SourceFile] = []
    for path in snapshot_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                f"source snapshot must be materialized; symbolic link found: "
                f"{path.relative_to(snapshot_root).as_posix()}"
            )
        if not path.is_file():
            continue
        relative_path = path.relative_to(snapshot_root)
        if _is_transient_local_download_metadata(relative_path):
            continue
        records.append(
            SourceFile(
                relative_path=relative_path.as_posix(),
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    records.sort(key=lambda entry: entry.relative_path)
    if len({entry.relative_path for entry in records}) != len(records):
        raise ValueError("source snapshot contains duplicate relative file paths")
    return tuple(records)


def _is_transient_local_download_metadata(relative_path: Path) -> bool:
    """Exclude only Hugging Face local-dir download metadata, never model files."""
    return (
        relative_path.parts[:3] == (".cache", "huggingface", "download")
        and relative_path.name.endswith(".metadata")
    )


def _validate_snapshot_inventory(files: tuple[SourceFile, ...]) -> None:
    """Require the exact minimum materialized files needed for a V0 model load."""
    paths = {entry.relative_path for entry in files}
    for required in ("config.json", "generation_config.json", "tokenizer_config.json"):
        if required not in paths:
            raise ValueError(f"source snapshot is missing required file {required}")
    tokenizer_json = "tokenizer.json" in paths
    sentencepiece_model = "tokenizer.model" in paths or "spiece.model" in paths
    bpe_pair = {"vocab.json", "merges.txt"}.issubset(paths)
    if not tokenizer_json and not sentencepiece_model and not bpe_pair:
        raise ValueError(
            "source snapshot is missing tokenizer data "
            "(need tokenizer.json, tokenizer.model/spiece.model, or vocab.json plus merges.txt)"
        )
    weight_files = [path for path in paths if path.endswith(".safetensors")]
    if not weight_files:
        raise ValueError("source snapshot is missing a safetensors weight file")
    if len(weight_files) > 1 and "model.safetensors.index.json" not in paths:
        raise ValueError(
            "source snapshot with multiple safetensors weights requires "
            "model.safetensors.index.json"
        )


def _runtime_code_files(project_root: Path | None = None) -> tuple[SourceFile, ...]:
    """Hash all code that can control recurrence production stages, if present."""
    root = (project_root or Path(__file__).resolve().parents[3]).resolve()
    candidates = [
        *sorted((root / "src" / "ppmoe").rglob("*.py")),
        root / "uv.lock",
    ]
    entrypoint = root / "scripts" / "run_recurrence_v0.py"
    if entrypoint.exists():
        candidates.append(entrypoint)
    records: list[SourceFile] = []
    for path in candidates:
        if not path.is_file():
            raise ValueError(f"runtime source file is missing: {path}")
        records.append(
            SourceFile(
                relative_path=path.relative_to(root).as_posix(),
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return tuple(sorted(records, key=lambda entry: entry.relative_path))


def _torch_version() -> str:
    import torch

    return str(torch.__version__)


def _transformers_version() -> str:
    import transformers

    return str(transformers.__version__)


def _gpu_metadata() -> dict[str, str | int]:
    import torch

    if not torch.cuda.is_available():
        return {"name": "unavailable", "vram_bytes": 0, "driver_version": "unavailable"}
    properties = torch.cuda.get_device_properties(0)
    driver = _nvidia_driver_version()
    return {
        "name": torch.cuda.get_device_name(0),
        "vram_bytes": int(properties.total_memory),
        "driver_version": driver,
    }


def _nvidia_driver_version() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    driver = completed.stdout.strip().splitlines()
    return driver[0] if driver else "unavailable"


def _git_revision() -> str:
    """Return the immutable Task 0 protocol freeze revision."""
    validate_revision(PROTOCOL_REVISION)
    return PROTOCOL_REVISION


def write_immutable_json(path: Path, payload: Mapping[str, object]) -> str:
    """Persist a completed JSON record once, allowing only byte-identical replay."""
    encoded = canonical_json(payload)
    destination = path.resolve()
    with _path_lock(destination):
        if destination.exists():
            if destination.read_bytes() == encoded:
                return sha256_file(destination)
            raise FileExistsError(f"immutable evidence already exists: {destination}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return sha256(encoded).hexdigest()


def append_event(path: Path, event: Mapping[str, object]) -> int:
    """Append one durable, monotonically indexed event to an evidence ledger."""
    if "index" in event or "timestamp" in event:
        raise ValueError("event may not override index or timestamp")
    destination = path.resolve()
    with _path_lock(destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        index = _last_event_index(destination) + 1
        needs_separator = destination.exists() and destination.stat().st_size > 0 and not destination.read_bytes().endswith(b"\n")
        payload = dict(event)
        payload["index"] = index
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        with destination.open("ab") as stream:
            if needs_separator:
                stream.write(b"\n")
            stream.write(canonical_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
    return index


def directory_size(path: Path) -> int:
    """Return the total size of files beneath an artifact directory."""
    if path.is_file():
        return path.stat().st_size
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _last_event_index(path: Path) -> int:
    if not path.exists():
        return -1
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("malformed final event line") from error
    if not lines:
        return -1
    try:
        final = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise ValueError("malformed final event line") from error
    if not isinstance(final, dict) or isinstance(final.get("index"), bool) or not isinstance(final.get("index"), int):
        raise ValueError("malformed final event line")
    return final["index"]


def _path_lock(path: Path) -> Lock:
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(path, Lock())
