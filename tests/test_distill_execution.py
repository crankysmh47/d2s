from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ppmoe.distill.execution import (
    ActiveJobContext,
    active_elapsed_seconds,
    append_event,
    execution_stage_signature,
    read_events,
    register_execution,
    validate_execution,
    load_active_job,
    process_creation_utc,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _prepared_source(root: Path, config: Path, *, stopped: bool = True) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    source_sig = {
        "config": {"scientific": "frozen"}, "run_id": "fixture",
        "runtime_identity": {"python": "fixture", "gpu": "verified-fixture"},
    }
    run_id = sha256(_canonical(source_sig)).hexdigest()
    source_root = root / run_id
    source_root.mkdir()
    payload = b'{"id":"id-1","question":"Q","gold":"private"}\n'
    (source_root / "data").mkdir()
    (source_root / "data" / "calibration.jsonl").write_bytes(payload)
    (source_root / "retention").mkdir()
    (source_root / "retention" / "retention-chunks.jsonl").write_bytes(b'{"index":0}\n')
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "source_signature": source_sig,
        "config": source_sig["config"],
        "retention": {"chunks_sha256": sha256(b'{"index":0}\n').hexdigest()},
        "splits": {
            "calibration": {
                "path": "data/calibration.jsonl",
                "sha256": sha256(payload).hexdigest(),
                "count": 1,
                "ordered_ids": ["id-1"],
            },
        },
    }
    manifest_path = source_root / "manifest.json"
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    (source_root / "stages").mkdir()
    prepare = {
        "stage": "prepare",
        "status": "passed",
        "run_id": run_id,
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "source_signature_sha256": sha256(_canonical(source_sig)).hexdigest(),
        "retention_chunks_sha256": manifest["retention"]["chunks_sha256"],
    }
    (source_root / "stages" / "prepare.json").write_bytes(_canonical(prepare) + b"\n")
    stop = {
        "stage": "calibrate",
        "status": "stopped_environment" if stopped else "awaiting_review",
        "reason": "global wall-time budget expired",
        "run_id": run_id,
        "task1_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "task1_prepare_sha256": sha256((source_root / "stages" / "prepare.json").read_bytes()).hexdigest(),
        "signature": {
            "predecessors": {
                "prepare_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
                "prepare_stage_sha256": sha256((source_root / "stages" / "prepare.json").read_bytes()).hexdigest(),
            },
            "code_signatures": {"config.toml": sha256(config.read_bytes()).hexdigest()},
        },
    }
    (source_root / "stages" / "calibrate.json").write_bytes(_canonical(stop) + b"\n")
    return source_root, source_root / "stages" / "calibrate.json"


def _repo_and_config(tmp_path: Path, monkeypatch, *, stopped: bool = True):
    from ppmoe.distill import execution
    from ppmoe.distill import study

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "uv.lock").write_text("version=1\n", encoding="utf-8")
    config = repo_root / "config.toml"
    config.write_text("scientific = 'unchanged'\n", encoding="utf-8")
    source_root, _ = _prepared_source(tmp_path / "source", config, stopped=stopped)
    monkeypatch.setattr(study, "TASK1_MANIFEST_SHA256", sha256((source_root / "manifest.json").read_bytes()).hexdigest())
    monkeypatch.setattr(study, "TASK1_PREPARE_SHA256", sha256((source_root / "stages" / "prepare.json").read_bytes()).hexdigest())
    monkeypatch.setattr(execution, "TASK1_STOP_SHA256", sha256((source_root / "stages" / "calibrate.json").read_bytes()).hexdigest())
    return source_root, repo_root, config


def test_active_runtime_excludes_between_job_gap():
    first = ActiveJobContext("e", "a", "calibrate", 0, 100_000_000_000,
                             1, "2026-10-07T00:00:00Z", "s")
    assert active_elapsed_seconds(first, now_ns=105_100_000_000) == 6
    second = ActiveJobContext("e", "b", "calibrate", 6, 900_000_000_000,
                              1, "2026-10-09T00:00:00Z", "s")
    assert active_elapsed_seconds(second, now_ns=902_000_000_000) == 8


def test_active_runtime_never_charges_negative_elapsed():
    context = ActiveJobContext("e", "a", "calibrate", 9, 100, 1,
                               "2026-10-07T00:00:00Z", "s")
    assert active_elapsed_seconds(context, now_ns=0) == 9


def test_registration_is_exclusive_and_carries_inherited_costs(tmp_path: Path, monkeypatch):
    from ppmoe.distill import execution

    source_root, repo_root, config = _repo_and_config(tmp_path, monkeypatch)
    monkeypatch.setattr(execution, "_utc_now", lambda: "2026-10-07T00:00:00.000000Z")
    executions_root = tmp_path / "executions"

    execution_root = register_execution(
        source_root, executions_root, repo_root=repo_root, config_path=config,
    )
    manifest = validate_execution(execution_root, repo_root=repo_root, config_path=config)
    assert manifest["source_run_id"] == source_root.name
    assert manifest["inherited_costs"] == {
        "cuda_seconds": 1168,
        "download_bytes": 8066699435,
    }
    assert manifest["policy"]["active_runtime_seconds"] == 172800
    with pytest.raises(FileExistsError):
        register_execution(source_root, executions_root, repo_root=repo_root, config_path=config)


def test_registration_requires_reviewed_environment_stop(tmp_path: Path, monkeypatch):
    source_root, repo_root, config = _repo_and_config(tmp_path, monkeypatch, stopped=False)
    with pytest.raises(ValueError, match="stopped_environment"):
        register_execution(source_root, tmp_path / "executions", repo_root=repo_root, config_path=config)


def test_event_chain_rejects_tampered_payload(tmp_path: Path, monkeypatch):
    source_root, repo_root, config = _repo_and_config(tmp_path, monkeypatch)
    execution_root = register_execution(source_root, tmp_path / "executions", repo_root=repo_root, config_path=config)
    append_event(execution_root, {"type": "registered", "at_utc": datetime.now(timezone.utc).isoformat()})
    assert read_events(execution_root)[0]["sequence"] == 0
    journal = execution_root / "events.jsonl"
    event = json.loads(journal.read_text(encoding="utf-8"))
    event["payload"]["type"] = "running"
    journal.write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical|digest"):
        read_events(execution_root)


def test_stage_signature_rejects_unimplemented_stage(tmp_path: Path):
    with pytest.raises(ValueError, match="not implemented"):
        execution_stage_signature(tmp_path, "train47", repo_root=tmp_path)


def test_load_active_job_requires_supervisor_and_contained_stage_identity(tmp_path: Path, monkeypatch):
    from ppmoe.distill import execution

    source_root, repo_root, config = _repo_and_config(tmp_path, monkeypatch)
    execution_root = register_execution(
        source_root, tmp_path / "executions", repo_root=repo_root, config_path=config,
    )
    supervisor_pid = 123
    supervisor_created = "2026-10-07T00:00:00.000000Z"
    stage_created = "2026-10-07T00:00:01.000000Z"
    payload = {
        "type": "job_started",
        "execution_id": execution_root.name,
        "attempt_id": "attempt-1",
        "stage": "calibrate",
        "prior_active_seconds": 0,
        "started_monotonic_ns": 900,
        "supervisor_pid": supervisor_pid,
        "supervisor_created_utc": supervisor_created,
        "stage_pid": __import__("os").getpid(),
        "stage_created_utc": stage_created,
        "stage_signature": "s" * 64,
    }
    append_event(execution_root, payload)
    (execution_root / "current-job.json").write_bytes(_canonical({
        "payload": payload,
        "payload_sha256": sha256(_canonical(payload)).hexdigest(),
    }) + b"\n")
    monkeypatch.setattr(execution, "process_creation_utc", lambda pid: {
        supervisor_pid: supervisor_created,
        payload["stage_pid"]: stage_created,
    }.get(pid))
    assert load_active_job(execution_root, "attempt-1", stage="calibrate").prior_active_seconds == 0

    monkeypatch.setattr(execution, "process_creation_utc", lambda pid: {
        supervisor_pid: supervisor_created,
        payload["stage_pid"]: "2026-10-07T00:00:02.000000Z",
    }.get(pid))
    with pytest.raises(ValueError, match="stage process"):
        load_active_job(execution_root, "attempt-1", stage="calibrate")


def test_load_active_job_rejects_dead_or_reused_supervisor(tmp_path: Path, monkeypatch):
    from ppmoe.distill import execution

    source_root, repo_root, config = _repo_and_config(tmp_path, monkeypatch)
    execution_root = register_execution(
        source_root, tmp_path / "executions", repo_root=repo_root, config_path=config,
    )
    payload = {
        "type": "job_started", "execution_id": execution_root.name,
        "attempt_id": "attempt-1", "stage": "calibrate", "prior_active_seconds": 0,
        "started_monotonic_ns": 1, "supervisor_pid": 123,
        "supervisor_created_utc": "2026-10-07T00:00:00.000000Z",
        "stage_pid": __import__("os").getpid(),
        "stage_created_utc": "2026-10-07T00:00:01.000000Z", "stage_signature": "s" * 64,
    }
    append_event(execution_root, payload)
    (execution_root / "current-job.json").write_bytes(_canonical({
        "payload": payload, "payload_sha256": sha256(_canonical(payload)).hexdigest(),
    }) + b"\n")
    monkeypatch.setattr(execution, "process_creation_utc", lambda pid: "2026-10-07T00:00:02.000000Z")
    with pytest.raises(ValueError, match="supervisor"):
        load_active_job(execution_root, "attempt-1", stage="calibrate")


def test_registration_rejects_overlapping_roots_and_preserves_source(tmp_path: Path, monkeypatch):
    source_root, repo_root, config = _repo_and_config(tmp_path, monkeypatch)
    before = {p.relative_to(source_root): p.read_bytes() for p in source_root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="overlap"):
        register_execution(source_root, source_root / "executions", repo_root=repo_root, config_path=config)
    after = {p.relative_to(source_root): p.read_bytes() for p in source_root.rglob("*") if p.is_file()}
    assert after == before




@pytest.mark.parametrize("tamper", ["source", "config", "terminal"])
def test_execution_validation_rejects_source_config_or_terminal_drift(tmp_path: Path, monkeypatch, tamper: str):
    source_root, repo_root, config = _repo_and_config(tmp_path, monkeypatch)
    execution_root = register_execution(
        source_root, tmp_path / "executions", repo_root=repo_root, config_path=config,
    )
    if tamper == "source":
        (source_root / "data" / "calibration.jsonl").write_bytes(b"tampered\n")
    elif tamper == "config":
        config.write_text("scientific = 'changed'\n", encoding="utf-8")
    else:
        terminal = source_root / "stages" / "calibrate.json"
        record = json.loads(terminal.read_text(encoding="utf-8"))
        record["task1_prepare_sha256"] = "0" * 64
        terminal.write_bytes(_canonical(record) + b"\n")
    with pytest.raises((ValueError, RuntimeError)):
        validate_execution(execution_root, repo_root=repo_root, config_path=config)


def test_event_reader_rejects_truncated_tail(tmp_path: Path, monkeypatch):
    source_root, repo_root, config = _repo_and_config(tmp_path, monkeypatch)
    execution_root = register_execution(
        source_root, tmp_path / "executions", repo_root=repo_root, config_path=config,
    )
    append_event(execution_root, {"type": "registered"})
    (execution_root / "events.jsonl").write_bytes((execution_root / "events.jsonl").read_bytes().rstrip())
    with pytest.raises(ValueError, match="truncated"):
        read_events(execution_root)


def test_load_active_job_rejects_unrelated_stage_pid(tmp_path: Path, monkeypatch):
    from ppmoe.distill import execution

    source_root, repo_root, config = _repo_and_config(tmp_path, monkeypatch)
    execution_root = register_execution(
        source_root, tmp_path / "executions", repo_root=repo_root, config_path=config,
    )
    payload = {
        "type": "job_started", "execution_id": execution_root.name,
        "attempt_id": "attempt-1", "stage": "calibrate", "prior_active_seconds": 0,
        "started_monotonic_ns": 1, "supervisor_pid": 123,
        "supervisor_created_utc": "2026-10-07T00:00:00.000000Z",
        "stage_pid": 999999, "stage_created_utc": "2026-10-07T00:00:01.000000Z",
        "stage_signature": "s" * 64,
    }
    append_event(execution_root, payload)
    (execution_root / "current-job.json").write_bytes(_canonical({
        "payload": payload, "payload_sha256": sha256(_canonical(payload)).hexdigest(),
    }) + b"\n")
    monkeypatch.setattr(execution, "process_creation_utc", lambda pid: {
        123: payload["supervisor_created_utc"],
        999999: payload["stage_created_utc"],
    }.get(pid))
    with pytest.raises(ValueError, match="contained stage process"):
        load_active_job(execution_root, "attempt-1", stage="calibrate")


def test_process_creation_utc_requires_live_process_and_reads_current_process():
    assert process_creation_utc(__import__("os").getpid()) is not None
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    assert child.wait(timeout=10) == 0
    assert process_creation_utc(child.pid) is None


def _lifecycle_start(execution_id: str, attempt_id: str, *, prior: int, started: int) -> dict[str, object]:
    return {
        "type": "job_started", "execution_id": execution_id, "attempt_id": attempt_id,
        "stage": "calibrate", "prior_active_seconds": prior,
        "started_monotonic_ns": started,
        "supervisor_pid": 123, "supervisor_created_utc": "2026-10-07T00:00:00.000000Z",
        "stage_pid": 456, "stage_created_utc": "2026-10-07T00:00:01.000000Z",
        "stage_signature": "a" * 64,
    }


def _lifecycle_settlement(start: dict[str, object], *, ended: int) -> dict[str, object]:
    duration = (ended - int(start["started_monotonic_ns"]) + 999_999_999) // 1_000_000_000
    return {
        **start, "type": "job_settled", "ended_monotonic_ns": ended,
        "settled_active_seconds": int(start["prior_active_seconds"]) + duration,
    }


def _execution_with_events(tmp_path: Path, events: list[dict[str, object]]) -> Path:
    root = tmp_path / "execution"
    root.mkdir()
    (root / "execution.json").write_bytes(_canonical({"execution_id": "execution-id"}) + b"\n")
    (root / "events.jsonl").touch()
    for event in events:
        append_event(root, event)
    return root


def test_validate_job_intervals_returns_ordered_cumulative_state(tmp_path: Path):
    from ppmoe.distill.execution import validate_job_intervals

    first = _lifecycle_start("execution-id", "first", prior=0, started=1_000_000_000)
    second = _lifecycle_start("execution-id", "second", prior=2, started=5_000_000_000)
    current = _lifecycle_start("execution-id", "current", prior=3, started=8_000_000_000)
    root = _execution_with_events(tmp_path, [
        first, _lifecycle_settlement(first, ended=3_000_000_000),
        second, _lifecycle_settlement(second, ended=6_000_000_000), current,
    ])

    intervals = validate_job_intervals(root)

    assert intervals["first"]["settled_active_seconds"] == 2
    assert intervals["second"]["prior_active_seconds"] == 2
    assert intervals["second"]["settled_active_seconds"] == 3
    assert intervals["current"]["prior_active_seconds"] == 3
    assert intervals["current"]["settlement"] is None


@pytest.mark.parametrize("corruption", [
    "reset_prior", "inflated_prior", "overlap", "duplicate_id", "out_of_order_settlement",
    "changed_identity", "settled_mismatch", "monotonic_regression",
])
def test_validate_job_intervals_rejects_broken_lifecycle_chain(tmp_path: Path, corruption: str):
    from ppmoe.distill.execution import validate_job_intervals

    first = _lifecycle_start("execution-id", "first", prior=0, started=1_000_000_000)
    settlement = _lifecycle_settlement(first, ended=3_000_000_000)
    second_prior = 0 if corruption == "reset_prior" else 3 if corruption == "inflated_prior" else 2
    second_started = 1_500_000_000 if corruption == "overlap" else 3_000_000_000
    second = _lifecycle_start("execution-id", "second", prior=second_prior, started=second_started)
    if corruption == "duplicate_id":
        second["attempt_id"] = "first"
    if corruption == "changed_identity":
        settlement["stage_signature"] = "b" * 64
    if corruption == "settled_mismatch":
        settlement["settled_active_seconds"] = 9
    if corruption == "monotonic_regression":
        second_started = 500_000_000
        second["started_monotonic_ns"] = second_started
    events = [first]
    if corruption == "out_of_order_settlement":
        events.extend([_lifecycle_settlement(second, ended=second_started + 1_000_000_000), settlement, second])
    else:
        events.extend([settlement, second])
    root = _execution_with_events(tmp_path, events)

    with pytest.raises(ValueError):
        validate_job_intervals(root)
