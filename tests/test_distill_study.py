from __future__ import annotations

import json
import os
import importlib.util
from hashlib import sha256
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import time

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
    assert len(signature["originating_git_commit"]) == 40
    assert signature["task1_originating_git_commit"] == "4bd6d6ab0a66f4753799d45a6bc6f5bb4fb70c63"
    assert set(signature["code_signatures"]) >= {
        "src/ppmoe/distill/config.py", "src/ppmoe/distill/data.py", "configs/distillation-v1.toml",
        "src/ppmoe/distill/study.py", "src/ppmoe/distill/model.py",
        "src/ppmoe/distill/answers.py", "src/ppmoe/distill/evaluate.py", "uv.lock",
    }


def _copy_calibration_test_repo(repo_root: Path) -> Path:
    source_repo = Path(__file__).parents[1]
    from ppmoe.distill.study import ACTIVE_FILES, CALIBRATION_FILES

    active = set(ACTIVE_FILES) | set(CALIBRATION_FILES) | {
        "src/ppmoe/distill/execution.py",
        "tests/test_distill_execution.py",
        "tests/test_distill_evaluate.py",
        "tests/test_distill_study.py",
        "scripts/run_distillation_v1.py",
    }
    for relative in active | {"uv.lock", "pyproject.toml"}:
        source = source_repo / relative
        target = repo_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return repo_root / "configs/distillation-v1.toml"


def _make_execution_source(tmp_path: Path, config, *, with_terminal: bool = True) -> Path:
    from ppmoe.distill.config import canonical_config
    from ppmoe.distill.data import sha256_file

    source_signature = {
        "config": canonical_config(config),
        "runtime_identity": {"python": "3.11-fixture", "cuda": "fixture-authenticated"},
    }
    run_id = sha256(json.dumps(source_signature, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    root = tmp_path / run_id
    root.mkdir(parents=True)
    rows = [
        {"id": f"id-{i}", "dataset": "gsm8k", "source_split": "train",
         "question": f"Unique calibration question {i}?", "gold": str(i), "solution": "fixture"}
        for i in range(60)
    ]
    data_path = root / "data" / "calibration.jsonl"
    data_path.parent.mkdir()
    data_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    retention = root / "retention" / "retention-chunks.jsonl"
    retention.parent.mkdir()
    retention.write_text('{"index":0}\n', encoding="utf-8")
    manifest = {
        "schema_version": 1, "run_id": run_id,
        "config": canonical_config(config), "source_signature": source_signature,
        "retention": {"chunks_sha256": sha256_file(retention)},
        "splits": {
            "calibration": {"path": "data/calibration.jsonl", "sha256": sha256_file(data_path),
                            "count": 60, "ordered_ids": [row["id"] for row in rows]},
            "retention_chunks": {"path": "retention/retention-chunks.jsonl", "sha256": sha256_file(retention),
                                 "count": 1, "ordered_ids": [0]},
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    prepare_path = root / "stages" / "prepare.json"
    prepare_path.parent.mkdir()
    signature_hash = sha256(json.dumps(source_signature, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    prepare_path.write_text(json.dumps({
        "stage": "prepare", "status": "passed", "run_id": run_id,
        "manifest_sha256": sha256_file(manifest_path),
        "source_signature_sha256": signature_hash,
        "retention_chunks_sha256": sha256_file(retention),
    }, sort_keys=True), encoding="utf-8")
    if with_terminal:
        terminal_path = root / "stages" / "calibrate.json"
        terminal_path.write_text(json.dumps({
            "stage": "calibrate", "status": "stopped_environment",
            "reason": "global wall-time budget expired", "run_id": run_id,
            "task1_manifest_sha256": sha256_file(manifest_path),
            "task1_prepare_sha256": sha256_file(prepare_path),
            "signature": {
                "predecessors": {"prepare_manifest_sha256": sha256_file(manifest_path),
                                 "prepare_stage_sha256": sha256_file(prepare_path)},
                "code_signatures": {"configs/distillation-v1.toml": sha256_file(Path(__file__).parents[1] / "configs/distillation-v1.toml")},
            },
        }, sort_keys=True), encoding="utf-8")
    return root


def test_execution_calibration_keeps_source_read_only_and_accounts_only_active_time(monkeypatch, tmp_path: Path) -> None:
    from ppmoe.distill.config import load_config
    from ppmoe.distill import execution
    from ppmoe.distill.execution import ActiveJobContext, append_event, execution_stage_signature, register_execution
    from ppmoe.distill.study import run_calibration_stage

    repo_root = tmp_path / "repo"
    config_path = _copy_calibration_test_repo(repo_root)
    config = load_config(config_path)
    source_root = _make_execution_source(tmp_path / "source", config)
    monkeypatch.setattr("ppmoe.distill.study.TASK1_RUN_ID", source_root.name)
    monkeypatch.setattr("ppmoe.distill.study.TASK1_MANIFEST_SHA256", sha256((source_root / "manifest.json").read_bytes()).hexdigest())
    monkeypatch.setattr("ppmoe.distill.study.TASK1_PREPARE_SHA256", sha256((source_root / "stages/prepare.json").read_bytes()).hexdigest())
    monkeypatch.setattr(execution, "TASK1_STOP_SHA256", sha256((source_root / "stages/calibrate.json").read_bytes()).hexdigest())
    runtime_identity = json.loads((source_root / "manifest.json").read_text())["source_signature"]["runtime_identity"]
    monkeypatch.setattr("ppmoe.distill.study.feasibility_runtime_identity", lambda *args, **kwargs: runtime_identity)
    monkeypatch.setattr("ppmoe.distill.study.preflight_disk", lambda *args, **kwargs: 100 * 1024**3)
    monkeypatch.setattr("ppmoe.distill.study._git_commit", lambda *args, **kwargs: "a" * 40)

    for model_dir in (
        repo_root / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots" / config.student.revision,
        repo_root / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots" / config.teacher.revision,
    ):
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

    class FakeApi:
        def model_info(self, repo_id, *, revision, files_metadata):
            return SimpleNamespace(sha=revision, siblings=[SimpleNamespace(rfilename="config.json", size=2)])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=FakeApi,
        snapshot_download=lambda **kwargs: pytest.fail("complete local model cache must avoid downloads"),
        hf_hub_download=lambda **kwargs: pytest.fail("complete local model cache must avoid downloads"),
    ))
    monkeypatch.setattr("ppmoe.distill.study.authenticate_model_snapshot", lambda *args, **kwargs: {"files": []})
    monkeypatch.setattr("ppmoe.distill.study.pinned_eos_token_ids", lambda *args, **kwargs: (2,))

    execution_root = register_execution(source_root, tmp_path / "executions", repo_root=repo_root, config_path=config_path)
    signature = execution_stage_signature(execution_root, "calibrate", repo_root=repo_root)
    supervisor_created = "2026-10-07T00:00:00.000000Z"
    stage_created = "2026-10-07T00:00:01.000000Z"
    started_ns = time.monotonic_ns() - 1_000_000_000
    job_payload = {
        "type": "job_started", "execution_id": execution_root.name,
        "attempt_id": "attempt-1", "stage": "calibrate",
        "prior_active_seconds": 500, "started_monotonic_ns": started_ns,
        "supervisor_pid": 123, "supervisor_created_utc": supervisor_created,
        "stage_pid": os.getpid(), "stage_created_utc": stage_created,
        "stage_signature": signature,
    }
    append_event(execution_root, job_payload)
    job_bytes = json.dumps(job_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    (execution_root / "current-job.json").write_bytes((json.dumps({
        "payload": job_payload, "payload_sha256": sha256(job_bytes).hexdigest(),
    }, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    monkeypatch.setattr("ppmoe.distill.execution.process_creation_utc", lambda pid: {
        123: supervisor_created, os.getpid(): stage_created,
    }.get(pid))
    context = ActiveJobContext(
        execution_root.name, "attempt-1", "calibrate", 500, started_ns, 123,
        supervisor_created, signature,
    )

    class Worker:
        startup_wall_seconds = 0
        def generate(self, row, candidate):
            return {"text": "Final answer: 1", "stop_reason": "eos", "generated_tokens": 2,
                    "wall_seconds": 0.001, "cuda_seconds": 0}
        def close(self):
            pass

    source_before = {path.relative_to(source_root): path.read_bytes() for path in source_root.rglob("*") if path.is_file()}
    result = run_calibration_stage(
        source_root, config, repo_root=repo_root, config_path=config_path,
        execution_root=execution_root, active_job=context,
        worker_loader=lambda *args: Worker(),
    )
    source_after = {path.relative_to(source_root): path.read_bytes() for path in source_root.rglob("*") if path.is_file()}
    assert result["status"] == "awaiting_review"
    assert result["wall_mode"] == "active_job"
    assert result["resource_ledger"]["wall_seconds"] < 600
    assert result["resource_ledger"]["cuda_seconds"] >= 1168
    assert result["resource_ledger"]["download_bytes"] == 8066699435
    assert source_after == source_before
    assert (execution_root / "calibration" / "raw").is_dir()
    assert (execution_root / "calibration" / "review").is_dir()
    assert (execution_root / "stages" / "calibrate.json").is_file()


@pytest.mark.parametrize("wall_mode", ["legacy_calendar", "active_job"])
def test_wall_refresh_crossing_cap_persists_terminal_stop(monkeypatch, tmp_path: Path, wall_mode: str) -> None:
    from datetime import datetime, timedelta, timezone
    from dataclasses import replace
    from ppmoe.distill.config import load_config
    from ppmoe.distill.evaluate import ResourceLedger
    from ppmoe.distill.study import _refresh_pre_model_ledger

    config = load_config(Path(__file__).parents[1] / "configs/distillation-v1.toml")
    if wall_mode == "active_job":
        ledger = replace(
            ResourceLedger(wall_seconds=172799, wall_cap=172800),
            wall_mode="active_job", settled_wall_seconds=172799,
            current_job_started_monotonic_ns=1_000_000_000,
        )
        monkeypatch.setattr("ppmoe.distill.evaluate.time.monotonic_ns", lambda: 2_000_000_001)
    else:
        started = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat().replace("+00:00", "Z")
        ledger = ResourceLedger(wall_seconds=172799, wall_cap=172800, wall_started_utc=started)
    output = tmp_path / wall_mode
    (output / "stages").mkdir(parents=True)
    updated, stop = _refresh_pre_model_ledger(
        output, output / "stages" / "calibrate-reservations.jsonl",
        config=config, manifest_hash="m" * 64, prepare_hash="p" * 64,
        originating_commit="a" * 40, started_at="2026-10-07T00:00:00.000000Z",
        signature_record={"stage_signature": "s" * 64, "predecessors": {}},
        ledger=ledger, detail={"phase": "preflight"},
    )
    saved = json.loads((output / "stages" / "calibrate.json").read_text(encoding="utf-8"))
    assert stop is not None and stop["status"] == "stopped_environment"
    assert updated.wall_seconds > updated.wall_cap
    assert saved["status"] == "stopped_environment"
    assert "wall-time" in saved["reason"]
    assert saved["resource_ledger"]["wall_mode"] == wall_mode


def test_legacy_expired_calibration_stops_before_hub_or_worker(monkeypatch, tmp_path: Path) -> None:
    import sys
    from ppmoe.distill.config import load_config
    from ppmoe.distill.study import run_calibration_stage

    repo_root = tmp_path / "repo"
    config_path = _copy_calibration_test_repo(repo_root)
    config = load_config(config_path)
    source_root = _make_execution_source(tmp_path / "source", config, with_terminal=False)
    monkeypatch.setattr("ppmoe.distill.study.TASK1_RUN_ID", source_root.name)
    monkeypatch.setattr("ppmoe.distill.study.TASK1_MANIFEST_SHA256", sha256((source_root / "manifest.json").read_bytes()).hexdigest())
    monkeypatch.setattr("ppmoe.distill.study.TASK1_PREPARE_SHA256", sha256((source_root / "stages/prepare.json").read_bytes()).hexdigest())
    monkeypatch.setattr("ppmoe.distill.study._git_commit", lambda *args, **kwargs: "a" * 40)
    runtime_identity = json.loads((source_root / "manifest.json").read_text())["source_signature"]["runtime_identity"]
    monkeypatch.setattr("ppmoe.distill.study.feasibility_runtime_identity", lambda *args, **kwargs: runtime_identity)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda: pytest.fail("expired legacy run reached Hub metadata"),
        snapshot_download=lambda **kwargs: pytest.fail("expired legacy run reached download"),
        hf_hub_download=lambda **kwargs: pytest.fail("expired legacy run reached download"),
    ))
    result = run_calibration_stage(
        source_root, config, repo_root=repo_root, config_path=config_path,
        worker_loader=lambda *args: pytest.fail("expired legacy run constructed a worker"),
    )
    assert result["status"] == "stopped_environment"
    assert "wall-time" in result["reason"]
    assert not (source_root / "calibration" / "raw").exists()


def test_separate_execution_cli_requires_paired_args_before_validation(tmp_path: Path) -> None:
    with pytest.raises(StageError, match="supplied together"):
        run_stage("calibrate", tmp_path / "config.toml", tmp_path / ("a" * 64), execution_root=tmp_path / "exec")


@pytest.mark.parametrize("has_completed_row", [False, True])
def test_execution_replay_bounds_incomplete_worker_to_settled_job_interval(monkeypatch, tmp_path: Path, has_completed_row: bool) -> None:
    from ppmoe.distill.config import load_config
    from ppmoe.distill.evaluate import RawRowJournal
    from ppmoe.distill.execution import ActiveJobContext, append_event
    from ppmoe.distill.study import _execution_job_intervals, _replay_calibration_resources

    root = tmp_path / "execution"
    (root / "calibration" / "raw").mkdir(parents=True)
    (root / "stages").mkdir()
    (root / "execution.json").write_text('{"execution_id":"execution-id"}', encoding="utf-8")
    (root / "events.jsonl").touch()
    old_start = 1_000_000_000
    old_end = 5_000_000_000
    append_event(root, {
        "type": "job_started", "execution_id": "execution-id", "attempt_id": "old-job",
        "stage": "calibrate", "prior_active_seconds": 10,
        "started_monotonic_ns": old_start,
        "supervisor_pid": 123, "supervisor_created_utc": "2026-10-07T00:00:00.000000Z",
        "stage_pid": 456, "stage_created_utc": "2026-10-07T00:00:01.000000Z",
        "stage_signature": "s" * 64,
    })
    append_event(root, {
        "type": "job_settled", "execution_id": "execution-id", "attempt_id": "old-job",
        "stage": "calibrate", "started_monotonic_ns": old_start,
        "ended_monotonic_ns": old_end, "settled_active_seconds": 14,
        "supervisor_pid": 123, "supervisor_created_utc": "2026-10-07T00:00:00.000000Z",
        "stage_pid": 456, "stage_created_utc": "2026-10-07T00:00:01.000000Z",
        "stage_signature": "s" * 64,
    })
    current = ActiveJobContext(
        "execution-id", "new-job", "calibrate", 14, 100_000_000_000,
        123, "2026-10-07T00:00:00.000000Z", "t" * 64,
    )
    append_event(root, {
        "type": "job_started", "execution_id": "execution-id", "attempt_id": "new-job",
        "stage": "calibrate", "prior_active_seconds": 14,
        "started_monotonic_ns": current.started_monotonic_ns,
        "supervisor_pid": current.supervisor_pid,
        "supervisor_created_utc": current.supervisor_created_utc,
        "stage_pid": 456, "stage_created_utc": "2026-10-07T00:00:01.000000Z",
        "stage_signature": current.stage_signature,
    })
    journal = RawRowJournal(
        root / "calibration" / "raw" / "256-768-student.jsonl",
        expected_ids=["id-1"], stage_signature="s" * 64,
    )
    journal.record_attempt({
        "attempt_id": "worker-1", "job_attempt_id": "old-job",
        "reserved_monotonic_ns": 2_000_000_000,
        "reserved_at_utc": "2026-10-01T00:00:00.000000Z",
        "status": "worker_reserved", "role": "student", "candidate": [256, 768],
        "stage_signature": "s" * 64,
    })
    journal.record_attempt({
        "attempt_id": "worker-1", "job_attempt_id": "old-job",
        "status": "worker_started", "startup_wall_seconds": 1.0,
        "stage_signature": "s" * 64,
    })
    if has_completed_row:
        journal.append({
            "source_id": "id-1", "role": "student", "candidate": [256, 768],
            "stage_signature": "s" * 64, "attempt_id": "worker-1",
            "wall_seconds": 1.0, "cuda_seconds": None,
            "worker_startup_wall_seconds": 1.0,
        })
    monkeypatch.setattr("ppmoe.distill.study.time.monotonic_ns", lambda: 101_000_000_000)
    intervals = _execution_job_intervals(root, current)
    config = load_config(Path(__file__).parents[1] / "configs/distillation-v1.toml")

    ledger = _replay_calibration_resources(
        root, [], expected_ids=["id-1"], stage_signature="s" * 64,
        config=config, active_job=current,
        settled_job_intervals=intervals,
        inherited_artifact_bytes=50,
    )
    replayed_again = _replay_calibration_resources(
        root, [], expected_ids=["id-1"], stage_signature="s" * 64,
        config=config, active_job=current,
        settled_job_intervals=intervals,
        inherited_artifact_bytes=50,
    )
    assert ledger.cuda_seconds == replayed_again.cuda_seconds == 1171
    assert ledger.download_bytes == replayed_again.download_bytes == 8066699435
    assert ledger.wall_seconds == replayed_again.wall_seconds == 15
    assert ledger.artifact_bytes == replayed_again.artifact_bytes == 50 + sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    journal.record_attempt({
        "attempt_id": "worker-2", "job_attempt_id": "unsettled-job",
        "reserved_monotonic_ns": 2_000_000_000,
        "reserved_at_utc": "2026-10-01T00:00:00.000000Z",
        "status": "worker_reserved", "role": "student", "candidate": [256, 768],
        "stage_signature": "s" * 64,
    })
    with pytest.raises(StageError, match="no settled owner interval"):
        _replay_calibration_resources(
            root, [], expected_ids=["id-1"], stage_signature="s" * 64,
            config=config, active_job=current,
            settled_job_intervals=intervals, inherited_artifact_bytes=50,
        )


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


def test_cpu_calibration_stage_recovers_prefix_exports_blind_packet_and_is_terminal(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace
    from datetime import datetime, timezone
    from hashlib import sha256
    import json
    import ppmoe.distill.study as study
    from ppmoe.distill.evaluate import RawRowJournal

    repo_root = tmp_path / "repo"
    import shutil
    source_repo = Path(__file__).parents[1]
    for relative in set(study.CALIBRATION_FILES) | set(study.ACTIVE_FILES):
        source = source_repo / relative
        target = repo_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    root = tmp_path / ("r" * 64)
    root.mkdir()
    ids = [f"cal-{index:03}" for index in range(60)]
    questions = [{"id": row_id, "dataset": "synthetic", "source_split": "calibration", "question": f"Question {index}?"} for index, row_id in enumerate(ids)]
    full_rows = [dict(row, gold=str(index), solution=f"private {index}") for index, row in enumerate(questions)]
    data_path = root / "data" / "calibration.jsonl"
    data_path.parent.mkdir()
    data_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in full_rows), encoding="utf-8")
    split_hash = sha256_file(data_path)
    config_value = {"synthetic": "frozen"}
    manifest = {
        "config": config_value,
        "splits": {"calibration": {"path": "data/calibration.jsonl", "sha256": split_hash, "count": 60, "ordered_ids": ids}},
        "split_counts": {"calibration": 60}, "source_counts": {}, "exclusion_counts": {},
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    prepare_path = root / "stages" / "prepare.json"
    prepare_path.parent.mkdir()
    prepare_path.write_text('{"status":"passed"}\n', encoding="utf-8")
    manifest_digest, prepare_digest = sha256_file(manifest_path), sha256_file(prepare_path)
    (root / "stages" / "calibrate").unlink(missing_ok=True)
    monkeypatch.setattr(study, "TASK1_RUN_ID", root.name)
    monkeypatch.setattr(study, "TASK1_MANIFEST_SHA256", manifest_digest)
    monkeypatch.setattr(study, "TASK1_PREPARE_SHA256", prepare_digest)
    monkeypatch.setattr(study, "_git_commit", lambda path: "c" * 40)
    monkeypatch.setattr(study, "feasibility_runtime_identity", lambda *args, **kwargs: {"synthetic": True})
    monkeypatch.setattr(study, "validate_prepared_run", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(study, "canonical_config", lambda config: config_value)
    monkeypatch.setattr(study, "preflight_disk", lambda *args, **kwargs: 20 * 1024**3)
    from ppmoe.distill.evaluate import ResourceLedger
    simulated_origin = study._utc_now()
    monkeypatch.setattr(
        study, "_initial_resource_ledger",
        lambda: ResourceLedger(wall_seconds=0, wall_started_utc=simulated_origin), raising=False,
    )
    monkeypatch.setattr(study, "load_split", lambda *args, **kwargs: questions)
    inventory_by_role = {
        role: {"repo_id": role, "revision": f"{role}-pin", "files": [{"path": "config.json", "size_bytes": 1, "sha256": "d" * 64, "hub_blob_id": "blob"}]}
        for role in ("student", "teacher")
    }
    authenticated_paths = []
    def fake_auth(repo, rev, path, metadata=None, **kwargs):
        authenticated_paths.append((repo, Path(path), kwargs.get("excluded_metadata_subtree")))
        return inventory_by_role["student" if repo == "student" else "teacher"]
    monkeypatch.setattr(study, "authenticate_model_snapshot", fake_auth)
    monkeypatch.setattr(study, "pinned_eos_token_ids", lambda path: (2,))
    class FakeApi:
        def model_info(self, repo_id, *, revision, files_metadata):
            names = ["config.json"] if repo_id == "student" else ["config.json", "README.md", ".gitattributes"]
            sizes = {"config.json": 1, "README.md": 2, ".gitattributes": 3}
            return SimpleNamespace(sha=revision, siblings=[SimpleNamespace(rfilename=name, size=sizes[name]) for name in names])
    snapshot_calls = []
    def fake_snapshot_download(**kwargs):
        snapshot_calls.append(kwargs["local_files_only"])
        return tmp_path / kwargs["revision"]
    def fake_hf_hub_download(**kwargs):
        assert any(item.get("event_type") == "download_reservation" and item.get("role") == "teacher"
                   for item in map(json.loads, (root / "stages" / "calibrate-reservations.jsonl").read_text().splitlines()))
        target = tmp_path / "downloads" / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * {"README.md": 2, ".gitattributes": 3}[kwargs["filename"]])
        return str(target)
    fake_hub = SimpleNamespace(HfApi=FakeApi, snapshot_download=fake_snapshot_download, hf_hub_download=fake_hf_hub_download)
    import sys
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    artifact_student = repo_root / "artifacts/recurrence-v0/129d39646933/source/student-pin"
    (artifact_student / ".cache" / "metadata").mkdir(parents=True)
    (artifact_student / "config.json").write_text("x", encoding="utf-8")
    (artifact_student / ".cache" / "metadata" / "local.json").write_text("ignored local metadata", encoding="utf-8")
    teacher_cache = repo_root / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/teacher-pin"
    teacher_cache.mkdir(parents=True)
    student_cache = repo_root / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/student-pin"
    student_cache.mkdir(parents=True)
    (student_cache / "config.json").write_text("x", encoding="utf-8")
    (teacher_cache / "config.json").write_text("x", encoding="utf-8")
    config = SimpleNamespace(
        resources=SimpleNamespace(vram_reserve_bytes=2 * 1024**3, phase_retries=2, download_bytes=21474836480,
                                  artifact_bytes=21474836480, cuda_seconds=86400, wall_seconds=172800),
        prompt=SimpleNamespace(candidates=((256, 768), (512, 1024), (768, 1536))),
        qualification=SimpleNamespace(minimum_calibration_eos=57),
        student=SimpleNamespace(repo_id="student", revision="student-pin"),
        teacher=SimpleNamespace(repo_id="teacher", revision="teacher-pin"),
    )
    original_signature = study.calibration_stage_signature
    # Seed a digest-authenticated single-row completed prefix to model crash recovery.
    student_inventory_hash = sha256(json.dumps(inventory_by_role["student"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    teacher_inventory_hash = sha256(json.dumps(inventory_by_role["teacher"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    prepared_signature = original_signature(
        repo_root, source_run_id=root.name, manifest_sha256=manifest_digest, prepare_sha256=prepare_digest,
        model_inventory_sha256={"student": student_inventory_hash, "teacher": teacher_inventory_hash},
        effective_eos_token_ids=(2,), originating_git_commit="c" * 40,
    )
    stage_signature = prepared_signature["stage_signature"]
    candidates = (256, 768)
    journal = RawRowJournal(root / "calibration" / "raw" / "256-768-student.jsonl", expected_ids=ids, stage_signature=stage_signature)
    journal.append({"source_id": ids[0], "role": "student", "candidate": list(candidates), "question": questions[0]["question"],
                    "text": "Final answer: 0", "stop_reason": "eos", "generated_tokens": 2, "wall_seconds": 0.01,
                    "cuda_seconds": 0.01, "worker_startup_wall_seconds": 0.01, "attempt_id": "old-attempt", "stage_signature": stage_signature})
    reserved_at = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    journal.record_attempt({"attempt_id": "old-attempt", "status": "worker_reserved", "role": "student", "candidate": list(candidates),
                            "stage_signature": stage_signature, "reserved_at_utc": reserved_at})
    journal.record_attempt({"attempt_id": "old-attempt", "status": "worker_started", "role": "student", "candidate": list(candidates),
                            "stage_signature": stage_signature, "startup_wall_seconds": 0.01, "reserved_at_utc": reserved_at})
    reservation_history = root / "stages" / "calibrate-reservations.jsonl"
    reservation_history.write_text(json.dumps({
        "event_type": "reservation", "status": "reserved", "stage": "calibrate",
        "stage_signature": stage_signature, "signature": prepared_signature,
        "code_signatures": study._file_signatures(repo_root),
        "originating_git_commit": "c" * 40, "started_at_utc": reserved_at,
    }, sort_keys=True) + "\n", encoding="utf-8")
    generated = {"student": 0, "teacher": 0}
    class FakeWorker:
        startup_wall_seconds = 0.02
        def __init__(self, role): self.role = role
        def generate(self, row, candidate):
            assert "gold" not in row and "solution" not in row
            generated[self.role] += 1
            return {"text": "Work 1+2. Final answer: 0", "stop_reason": "eos", "generated_tokens": 4,
                    "wall_seconds": 0.01, "cuda_seconds": 0.01, "peak_allocated_bytes": 100, "peak_reserved_bytes": 200}
        def close(self): pass
    result = study.run_calibration_stage(root, config, repo_root=repo_root, config_path=tmp_path / "config.toml",
                                         worker_loader=lambda role, candidate: FakeWorker(role))
    assert result["status"] == "awaiting_review"
    assert generated == {"student": 59, "teacher": 60}
    packet = root / result["review_packet"]
    packet_text = packet.read_text(encoding="utf-8")
    assert '"gold"' not in packet_text and '"role"' not in packet_text and "private" not in packet_text
    assert result["originating_git_commit"] == "c" * 40
    assert result["started_at_utc"] < result["ended_at_utc"]
    assert snapshot_calls == []
    assert ("student", artifact_student, ".cache") in authenticated_paths
    assert any(repo == "teacher" and path == teacher_cache and excluded is None for repo, path, excluded in authenticated_paths)
    reservation_events = [json.loads(line) for line in (root / "stages" / "calibrate-reservations.jsonl").read_text().splitlines()]
    teacher_reservation = next(item for item in reservation_events if item.get("event_type") == "download_reservation" and item.get("role") == "teacher")
    assert teacher_reservation["reserved_bytes"] == 5
    assert teacher_reservation["missing_files"] == [".gitattributes", "README.md"]
    with pytest.raises(FileExistsError, match="immutable"):
        study.run_calibration_stage(root, config, repo_root=repo_root, config_path=tmp_path / "config.toml",
                                    worker_loader=lambda role, candidate: pytest.fail("terminal stage must not rerun"))

    failure_root = tmp_path / ("f" * 64)
    (failure_root / "stages").mkdir(parents=True)
    (failure_root / "data").mkdir()
    shutil.copyfile(manifest_path, failure_root / "manifest.json")
    shutil.copyfile(prepare_path, failure_root / "stages" / "prepare.json")
    shutil.copyfile(data_path, failure_root / "data" / "calibration.jsonl")
    monkeypatch.setattr(study, "TASK1_RUN_ID", failure_root.name)
    failed_result = study.run_calibration_stage(
        failure_root, config, repo_root=repo_root, config_path=tmp_path / "config.toml",
        worker_loader=lambda role, candidate: (_ for _ in ()).throw(RuntimeError("synthetic startup failure")),
    )
    assert failed_result["status"] == "stopped_environment"
    assert (failure_root / "stages" / "calibrate.json").is_file()
    failed_row_paths = [
        Path(str(path)[:-len(".attempts.jsonl")])
        for path in (failure_root / "calibration" / "raw").glob("*.attempts.jsonl")
    ]
    assert failed_row_paths
    assert all(not path.exists() or not path.read_text(encoding="utf-8").strip() for path in failed_row_paths)
    failed_attempts = [
        json.loads(line)
        for path in (failure_root / "calibration" / "raw").glob("*.attempts.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(item["status"] == "worker_load_failed" for item in failed_attempts)
    assert not any(item["status"] == "worker_complete" for item in failed_attempts)
    with pytest.raises(FileExistsError, match="immutable"):
        study.run_calibration_stage(
            failure_root, config, repo_root=repo_root, config_path=tmp_path / "config.toml",
            worker_loader=lambda role, candidate: pytest.fail("terminal failure must not be retried"),
        )

    overcap_root = tmp_path / ("o" * 64)
    (overcap_root / "stages").mkdir(parents=True)
    (overcap_root / "data").mkdir()
    shutil.copyfile(manifest_path, overcap_root / "manifest.json")
    shutil.copyfile(prepare_path, overcap_root / "stages" / "prepare.json")
    shutil.copyfile(data_path, overcap_root / "data" / "calibration.jsonl")
    monkeypatch.setattr(study, "TASK1_RUN_ID", overcap_root.name)
    overcap_config = SimpleNamespace(**vars(config))
    overcap_config.resources = SimpleNamespace(**vars(config.resources))
    overcap_config.resources.artifact_bytes = 1
    overcap_result = study.run_calibration_stage(
        overcap_root, overcap_config, repo_root=repo_root, config_path=tmp_path / "config.toml",
        worker_loader=lambda role, candidate: pytest.fail("replayed artifact cap must stop before model work"),
    )
    assert overcap_result["status"] == "stopped_environment"
    saved_terminal = json.loads((overcap_root / "stages" / "calibrate.json").read_text(encoding="utf-8"))
    assert saved_terminal["resource_ledger"]["artifact_bytes"] > overcap_config.resources.artifact_bytes
    measured = sum(
        path.stat().st_size
        for base in (overcap_root / "calibration", overcap_root / "stages")
        for path in (base.rglob("*") if base.exists() else ())
        if path.is_file() and (base == overcap_root / "calibration" or path.name.startswith("calibrate"))
    )
    assert saved_terminal["resource_ledger"]["artifact_bytes"] == measured


def _early_resource_stage(monkeypatch, tmp_path: Path):
    from types import SimpleNamespace
    import shutil
    import ppmoe.distill.study as study

    repo_root = tmp_path / "repo"
    source_repo = Path(__file__).parents[1]
    for relative in set(study.CALIBRATION_FILES) | set(study.ACTIVE_FILES):
        source = source_repo / relative
        target = repo_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    root = tmp_path / ("e" * 64)
    (root / "stages").mkdir(parents=True)
    ids = [f"early-{i:03}" for i in range(60)]
    config_value = {"synthetic": "signed"}
    manifest = {"config": config_value, "splits": {"calibration": {"count": 60, "ordered_ids": ids}}}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    prepare_path = root / "stages" / "prepare.json"
    prepare_path.write_text('{"status":"passed"}\n', encoding="utf-8")
    monkeypatch.setattr(study, "TASK1_RUN_ID", root.name)
    monkeypatch.setattr(study, "TASK1_MANIFEST_SHA256", sha256(manifest_path.read_bytes()).hexdigest())
    monkeypatch.setattr(study, "TASK1_PREPARE_SHA256", sha256(prepare_path.read_bytes()).hexdigest())
    monkeypatch.setattr(study, "_git_commit", lambda path: "f" * 40)
    monkeypatch.setattr(study, "feasibility_runtime_identity", lambda *args, **kwargs: {"synthetic": True})
    monkeypatch.setattr(study, "validate_prepared_run", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(study, "canonical_config", lambda config: config_value)
    config = SimpleNamespace(
        resources=SimpleNamespace(
            vram_reserve_bytes=2 * 1024**3, phase_retries=2,
            download_bytes=21474836480, artifact_bytes=21474836480,
            cuda_seconds=86400, wall_seconds=172800,
        ),
        student=SimpleNamespace(repo_id="student", revision="student-test-pin"),
        teacher=SimpleNamespace(repo_id="teacher", revision="teacher-test-pin"),
        prompt=SimpleNamespace(candidates=((256, 768),)),
        qualification=SimpleNamespace(minimum_calibration_eos=57),
    )
    return study, repo_root, root, config


def test_expired_wall_persists_premodel_stop_without_hub_disk_or_worker(monkeypatch, tmp_path: Path) -> None:
    import sys
    from ppmoe.distill.evaluate import ResourceLedger
    from ppmoe.distill.evaluate import RawRowJournal

    study, repo_root, root, config = _early_resource_stage(monkeypatch, tmp_path)
    # Use the frozen production origin; on the current date it is genuinely expired.
    assert ResourceLedger.initial().wall_seconds > config.resources.wall_seconds
    history_path = root / "stages" / "calibrate-reservations.jsonl"
    history_path.write_text(json.dumps({
        "event_type": "download_reservation", "status": "reserved", "reserved_bytes": 321,
        "originating_git_commit": "c" * 40,
    }) + "\n", encoding="utf-8")
    manifest_hash = sha256((root / "manifest.json").read_bytes()).hexdigest()
    prepare_hash = sha256((root / "stages" / "prepare.json").read_bytes()).hexdigest()
    prior_signature = study.calibration_stage_signature(
        repo_root, source_run_id=root.name, manifest_sha256=manifest_hash,
        prepare_sha256=prepare_hash, originating_git_commit="c" * 40,
    )
    stage_signature = prior_signature["stage_signature"]
    raw_journal = RawRowJournal(
        root / "calibration" / "raw" / "256-768-student.jsonl",
        expected_ids=[f"early-{i:03}" for i in range(60)], stage_signature=stage_signature,
    )
    raw_journal.append({
        "source_id": "early-000", "role": "student", "candidate": [256, 768],
        "question": "synthetic", "text": "not inspected", "stop_reason": "eos",
        "generated_tokens": 1, "wall_seconds": 2.0, "cuda_seconds": 7.5,
        "worker_startup_wall_seconds": 3.0, "attempt_id": "old-complete",
        "stage_signature": stage_signature,
    })
    old_started = study._utc_now()
    for state, extras in (
        ("worker_reserved", {"reserved_at_utc": old_started}),
        ("worker_started", {"startup_wall_seconds": 3.0}),
        ("worker_complete", {"startup_wall_seconds": 3.0}),
    ):
        raw_journal.record_attempt({
            "attempt_id": "old-complete", "status": state, "role": "student",
            "candidate": [256, 768], "stage_signature": stage_signature, **extras,
        })
    history_path.write_text(history_path.read_text(encoding="utf-8") + json.dumps({
        "event_type": "reservation", "status": "reserved", "stage_signature": stage_signature,
        "signature": prior_signature, "originating_git_commit": "c" * 40,
        "code_signatures": study._file_signatures(repo_root),
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(study, "preflight_disk", lambda *args, **kwargs: pytest.fail("expired stop checked disk"))
    class PoisonHub:
        def __getattr__(self, name):
            raise AssertionError(f"expired stop accessed Hub API {name}")
    monkeypatch.setitem(sys.modules, "huggingface_hub", PoisonHub())

    result = study.run_calibration_stage(
        root, config, repo_root=repo_root, config_path=tmp_path / "config.toml",
        worker_loader=lambda *args: pytest.fail("expired stop constructed worker"),
    )
    saved = json.loads((root / "stages" / "calibrate.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    assert result["status"] == saved["status"] == "stopped_environment"
    assert "wall" in saved["reason"].lower() or "expired" in saved["reason"].lower()
    assert saved["phase_state"] == "pre_model_stopped"
    assert saved["model_inventory_status"] == "not_authenticated"
    assert "model_inventory" not in saved
    assert saved["resource_ledger"]["wall_seconds"] > config.resources.wall_seconds
    assert saved["resource_ledger"]["download_bytes"] == 8066699435 + 321
    assert saved["resource_ledger"]["cuda_seconds"] == 1179
    assert saved["originating_git_commit"] == "c" * 40
    assert saved["started_at_utc"] < saved["ended_at_utc"]
    assert any(item.get("event_type") == "terminal" and item.get("status") == "stopped_environment" for item in events)
    with pytest.raises(FileExistsError, match="immutable"):
        study.run_calibration_stage(root, config, repo_root=repo_root, config_path=tmp_path / "config.toml")


def test_low_disk_persists_premodel_stop_before_hub_or_worker(monkeypatch, tmp_path: Path) -> None:
    import sys
    from ppmoe.distill.evaluate import ResourceLedger, ResourceLimitError

    study, repo_root, root, config = _early_resource_stage(monkeypatch, tmp_path)
    now = study._utc_now()
    monkeypatch.setattr(study, "_initial_resource_ledger", lambda: ResourceLedger(wall_seconds=0, wall_started_utc=now), raising=False)
    monkeypatch.setattr(study, "preflight_disk", lambda *args, **kwargs: (_ for _ in ()).throw(ResourceLimitError("insufficient free disk: need 10, have 2")))
    class PoisonHub:
        def __getattr__(self, name): raise AssertionError(f"low-disk stop accessed Hub API {name}")
    monkeypatch.setitem(sys.modules, "huggingface_hub", PoisonHub())

    result = study.run_calibration_stage(
        root, config, repo_root=repo_root, config_path=tmp_path / "config.toml",
        worker_loader=lambda *args: pytest.fail("low-disk stop constructed worker"),
    )
    saved = json.loads((root / "stages" / "calibrate.json").read_text(encoding="utf-8"))
    assert result["status"] == saved["status"] == "stopped_environment"
    assert "disk" in saved["reason"].lower()
    assert saved["resource_ledger"]["download_bytes"] == ResourceLedger.initial().download_bytes
    assert saved["model_inventory_status"] == "not_authenticated"


@pytest.mark.parametrize("over_role", ["student", "teacher"])
def test_projected_download_overcap_stops_without_materializing_or_changing_sources(monkeypatch, tmp_path: Path, over_role: str) -> None:
    import sys
    from types import SimpleNamespace
    from ppmoe.distill.evaluate import ResourceLedger

    study, repo_root, root, config = _early_resource_stage(monkeypatch, tmp_path)
    now = study._utc_now()
    base_download = ResourceLedger.initial().download_bytes
    monkeypatch.setattr(study, "_initial_resource_ledger", lambda: ResourceLedger(wall_seconds=0, wall_started_utc=now), raising=False)
    config.resources.download_bytes = base_download + 5
    monkeypatch.setattr(study, "preflight_disk", lambda *args, **kwargs: 16 * 1024**3)
    original_bytes: dict[Path, bytes] = {}
    if over_role == "teacher":
        student_source = repo_root / "artifacts/recurrence-v0/129d39646933/source" / config.student.revision
        student_source.mkdir(parents=True)
        student_file = student_source / "config.json"
        student_file.write_bytes(b"{}")
        original_bytes[student_file] = student_file.read_bytes()
        teacher_snapshot = repo_root / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots" / config.teacher.revision
        teacher_snapshot.mkdir(parents=True)
        teacher_file = teacher_snapshot / "config.json"
        teacher_file.write_bytes(b"{}")
        original_bytes[teacher_file] = teacher_file.read_bytes()
    else:
        student_snapshot = repo_root / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots" / config.student.revision
        student_snapshot.mkdir(parents=True)
        student_file = student_snapshot / "existing-cache-bytes.bin"
        student_file.write_bytes(b"keep this pre-existing cache content unchanged")
        original_bytes[student_file] = student_file.read_bytes()

    class FakeApi:
        def model_info(self, repo_id, *, revision, files_metadata):
            names = ["config.json"]
            if over_role == "teacher" and repo_id == config.teacher.repo_id:
                names.append("missing.bin")
            return SimpleNamespace(
                sha=revision,
                siblings=[SimpleNamespace(rfilename=name, size=10) for name in names],
            )
    class PoisonDownload:
        def __call__(self, **kwargs):
            raise AssertionError("over-cap estimate materialized model files")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=FakeApi, snapshot_download=PoisonDownload(), hf_hub_download=PoisonDownload(),
    ))
    monkeypatch.setattr(study, "authenticate_model_snapshot", lambda *args, **kwargs: {"files": []})
    monkeypatch.setattr(study, "pinned_eos_token_ids", lambda *args: (2,))

    result = study.run_calibration_stage(
        root, config, repo_root=repo_root, config_path=tmp_path / "config.toml",
        worker_loader=lambda *args: pytest.fail("over-cap estimate constructed worker"),
    )
    saved = json.loads((root / "stages" / "calibrate.json").read_text(encoding="utf-8"))
    assert result["status"] == saved["status"] == "stopped_environment"
    assert over_role in saved["reason"]
    assert saved["resource_ledger"]["download_bytes"] == base_download + 10
    assert all(path.read_bytes() == content for path, content in original_bytes.items())
    stop_event = next(
        json.loads(line) for line in (root / "stages" / "calibrate-reservations.jsonl").read_text().splitlines()
        if json.loads(line).get("event_type") == "resource_stop"
    )
    assert stop_event["attempted_download_bytes"] == saved["resource_ledger"]["download_bytes"]
