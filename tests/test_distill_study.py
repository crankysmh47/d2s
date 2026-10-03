from __future__ import annotations

import json
from pathlib import Path

import pytest

from ppmoe.distill.study import (
    StageError,
    atomic_write_json,
    run_stage,
    safe_output_path,
    stage_path,
    validate_prepared_run,
    validate_runtime_metadata,
)
from ppmoe.distill.data import sha256_file


def test_stage_paths_and_validation_are_explicit(tmp_path: Path) -> None:
    assert stage_path(tmp_path, "prepare") == tmp_path / "stages" / "prepare.json"
    with pytest.raises(StageError, match="not implemented"):
        run_stage("calibrate", tmp_path)
    with pytest.raises(StageError, match="unknown"):
        run_stage("made-up", tmp_path)


def test_atomic_json_writer_creates_once_and_rejects_finalized_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "stage.json"
    atomic_write_json(path, {"status": "prepared"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "prepared"}
    with pytest.raises(FileExistsError):
        atomic_write_json(path, {"status": "replacement"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "prepared"}


def test_output_path_rejects_symlink_or_reparse_traversal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symlink|reparse"):
        safe_output_path(root, link / "result.json")


def test_output_path_rejects_escape_from_run_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        safe_output_path(root, tmp_path / "outside.json")


def test_runtime_identity_binds_python_cuda_device_and_two_gib_reserve() -> None:
    environment = {
        "python": "3.11.9",
        "platform": "Windows-test",
        "packages": {"torch": "2.13.0+cu130"},
        "device_snapshot": {"status": "available", "stdout": "GPU,10240,5000,5240,616.92"},
        "torch_environment": {
            "cuda_available": True,
            "cuda_runtime": "13.0",
            "capability": [12, 0],
            "bf16_supported": True,
            "gpu_name": "GPU",
            "physical_capacity_bytes": 10 * 1024**3,
            "version": "2.13.0+cu130",
        },
    }
    preflight = {"memory_cap_bytes": 8 * 1024**3}

    identity = validate_runtime_metadata(environment, preflight, vram_reserve_bytes=2 * 1024**3)

    assert identity["python"] == "3.11.9"
    assert identity["gpu_name"] == "GPU"
    assert identity["memory_cap_bytes"] == 8 * 1024**3
    with pytest.raises(StageError, match="reserve"):
        validate_runtime_metadata(environment, preflight, vram_reserve_bytes=3 * 1024**3)


def test_prepared_run_rejects_tampered_split_bytes_without_parsing_labels(tmp_path: Path) -> None:
    run_id = "a" * 64
    root = tmp_path / run_id
    data = root / "data"
    data.mkdir(parents=True)
    split = data / "calibration.jsonl"
    split.write_bytes(b'{"id":"id-1","gold":"private"}\n')
    chunks = root / "retention" / "retention-chunks.jsonl"
    chunks.parent.mkdir()
    chunks.write_bytes(b'{"index":0}\n')
    signature = {"config": "frozen"}
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "source_signature": signature,
        "retention": {"chunks_sha256": sha256_file(chunks)},
        "splits": {
            "calibration": {
                "path": "data/calibration.jsonl",
                "sha256": sha256_file(split),
                "count": 1,
                "ordered_ids": ["id-1"],
            },
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    from hashlib import sha256

    signature_hash = sha256(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic_write_json(
        root / "stages" / "prepare.json",
        {
            "stage": "prepare",
            "status": "passed",
            "run_id": run_id,
            "manifest_sha256": sha256_file(manifest_path),
            "source_signature_sha256": signature_hash,
            "retention_chunks_sha256": sha256_file(chunks),
        },
    )

    assert validate_prepared_run(root)["run_id"] == run_id
    split.write_bytes(b'{"id":"id-1","gold":"tampered"}\n')
    with pytest.raises(StageError, match="bytes changed"):
        validate_prepared_run(root)
