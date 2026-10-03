from __future__ import annotations

import json
import importlib.util
from hashlib import sha256
from pathlib import Path

import pytest

from ppmoe.distill.study import (
    StageError,
    StageRecord,
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
    with pytest.raises(StageError, match="explicit run-root"):
        run_stage("calibrate", tmp_path)
    with pytest.raises(StageError, match="unknown"):
        run_stage("made-up", tmp_path)


def test_calibrate_requires_explicit_run_root_before_authenticating(monkeypatch, tmp_path: Path) -> None:
    def validation_must_not_run(*args, **kwargs):
        raise AssertionError("calibrate did not reject a missing run root before validation")

    monkeypatch.setattr("ppmoe.distill.study.run_calibration_stage", validation_must_not_run)
    with pytest.raises(StageError, match="run-root"):
        run_stage("calibrate", Path("configs/distillation-v1.toml"))


def test_calibration_dispatch_does_not_rederive_prepared_run_id(monkeypatch, tmp_path: Path) -> None:
    seen = {}

    def fake_calibrate(run_root, config, *, repo_root, config_path, worker_loader=None):
        seen["root"] = run_root
        run_root.mkdir(parents=True)
        (run_root / "manifest.json").write_text(json.dumps({"split_counts": {}, "source_counts": {}, "exclusion_counts": {}}), encoding="utf-8")
        return {"status": "awaiting_review", "stage_signature": "s" * 64}

    monkeypatch.setattr("ppmoe.distill.study.run_calibration_stage", fake_calibrate)
    monkeypatch.setattr("ppmoe.distill.study.load_config", lambda path: object())
    monkeypatch.setattr("ppmoe.distill.study.__file__", str(tmp_path / "a" / "b" / "c" / "study.py"))
    root = tmp_path / ("a" * 64)
    record = run_stage("calibrate", tmp_path / "config.toml", root)
    assert record.status == "awaiting_review"
    assert seen["root"] == root


def test_calibration_code_signature_binds_task1_and_task2_transitive_files(tmp_path: Path) -> None:
    from ppmoe.distill.study import calibration_stage_signature

    repo_root = Path(__file__).parents[1]
    signature = calibration_stage_signature(repo_root, source_run_id="r" * 64, manifest_sha256="m" * 64, prepare_sha256="p" * 64)
    assert signature["stage"] == "calibrate"
    assert signature["originating_git_commit"] == "4bd6d6ab0a66f4753799d45a6bc6f5bb4fb70c63"
    assert set(signature["code_signatures"]) >= {
        "src/ppmoe/distill/config.py", "src/ppmoe/distill/data.py",
        "src/ppmoe/distill/study.py", "src/ppmoe/distill/model.py",
        "src/ppmoe/distill/answers.py", "src/ppmoe/distill/evaluate.py", "uv.lock",
    }


def test_prepare_requires_explicit_run_root_before_source_scan(monkeypatch) -> None:
    def source_scan_must_not_run(*args, **kwargs):
        raise AssertionError("prepare checked source data before validating its required run root")

    monkeypatch.setattr("ppmoe.distill.study._prepared_source", source_scan_must_not_run)
    with pytest.raises(StageError, match="explicit run-root"):
        run_stage("prepare", Path("configs/distillation-v1.toml"))


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
    signature = {"config": "frozen"}
    run_id = sha256(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    root = tmp_path / run_id
    data = root / "data"
    data.mkdir(parents=True)
    split = data / "calibration.jsonl"
    split.write_bytes(b'{"id":"id-1","gold":"private"}\n')
    chunks = root / "retention" / "retention-chunks.jsonl"
    chunks.parent.mkdir()
    chunks.write_bytes(b'{"index":0}\n')
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


def test_prepared_run_rejects_directory_id_not_derived_from_signature(tmp_path: Path) -> None:
    signature = {"config": "frozen"}
    run_id = "f" * 64
    root = tmp_path / run_id
    root.mkdir()
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "source_signature": signature,
        "retention": {"chunks_sha256": "0" * 64},
        "splits": {},
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    signature_hash = sha256(json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic_write_json(
        root / "stages" / "prepare.json",
        {
            "stage": "prepare",
            "status": "passed",
            "run_id": run_id,
            "manifest_sha256": sha256_file(manifest_path),
            "source_signature_sha256": signature_hash,
            "retention_chunks_sha256": "0" * 64,
        },
    )

    with pytest.raises(StageError, match="run ID.*source signature"):
        validate_prepared_run(root)


def test_cli_accepts_positional_stages_and_rejects_missing_required_options(monkeypatch, capsys) -> None:
    script_path = Path(__file__).parents[1] / "scripts" / "run_distillation_v1.py"
    spec = importlib.util.spec_from_file_location("run_distillation_v1_test", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls: list[tuple[str, Path, Path | None]] = []

    def fake_run_stage(stage: str, config_path: Path, run_root: Path | None):
        calls.append((stage, config_path, run_root))
        return StageRecord(stage, "a" * 64, "computed", "preflight_ok", {}, {}, {})

    monkeypatch.setattr(cli, "run_stage", fake_run_stage)
    for stage in ("preflight", "prepare", "calibrate"):
        args = [stage, "--config", "configs/distillation-v1.toml"]
        if stage in {"prepare", "calibrate"}:
            args += ["--run-root", "artifacts/distillation-v1/" + "a" * 64]
        assert cli.main(args) == 0
    assert '"stage":"preflight"' in capsys.readouterr().out
    assert [call[0] for call in calls] == ["preflight", "prepare", "calibrate"]

    with pytest.raises(SystemExit):
        cli.main(["preflight"])
    with pytest.raises(SystemExit):
        cli.main(["prepare", "--config", "configs/distillation-v1.toml"])
    with pytest.raises(SystemExit):
        cli.main(["calibrate", "--config", "configs/distillation-v1.toml"])
    assert len(calls) == 3
