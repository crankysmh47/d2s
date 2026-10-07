from __future__ import annotations

import json
import os
import importlib.util
from pathlib import Path
import subprocess
import sys
import time

import pytest

from ppmoe.distill import execution, study
from ppmoe.distill.background import execution_status, launch_stage, supervise_stage
from ppmoe.distill import background
from ppmoe.distill.windows_job import WindowsJob

CPU_FIXTURE = Path(__file__).with_name("distill_background_cpu_fixture.py")


def _assert_process_identities_exited(owners: list[tuple[int, str]], *, timeout_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    pending = list(owners)
    while pending and time.monotonic() < deadline:
        pending = [
            (pid, created) for pid, created in pending
            if execution.process_creation_utc(pid) == created
        ]
        if pending:
            time.sleep(0.05)
    assert not pending, f"owned process identities survived the {timeout_seconds:g}-second exit deadline: {pending}"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _registered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location(
        "test_distill_execution_fixture", Path(__file__).with_name("test_distill_execution.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _repo_and_config = module._repo_and_config

    source_root, repo_root, config = _repo_and_config(tmp_path, monkeypatch)
    root = execution.register_execution(source_root, tmp_path / "executions", repo_root=repo_root, config_path=config)
    return root, source_root, repo_root, config


def test_registered_future_stage_is_signed_but_not_launchable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, _, repo, _ = _registered(tmp_path, monkeypatch)
    assert "src/ppmoe/distill/background.py" in study.execution_stage_files("calibrate")
    with pytest.raises(ValueError, match="not implemented"):
        execution.execution_stage_signature(root, "qualify", repo_root=repo)
    with pytest.raises(study.StageError, match="not implemented"):
        launch_stage(root, "qualify", repo_root=repo, config_path=repo / "config.toml")


def _launch_ack_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, ack_value=None):
    root, _source, repo, config = _registered(tmp_path, monkeypatch)
    popen_calls: list[object] = []

    class Child:
        pid = 41234

        def poll(self):
            return None

    child = Child()

    def popen(command, **_kwargs):
        popen_calls.append(command)
        attempt_id = command[command.index("--attempt-id") + 1]
        value = ack_value or {
            "attempt_id": attempt_id, "supervisor_pid": child.pid,
            "supervisor_created_utc": "2026-10-07T00:00:00.000000Z", "state": "running",
        }
        path = root / "launches" / f"{attempt_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
        return child

    monkeypatch.setattr(background.subprocess, "Popen", popen)
    monkeypatch.setattr(background, "direct_venv_launch", lambda command: (command, None, {}))
    monkeypatch.setattr(background, "process_creation_utc", lambda _pid: "2026-10-07T00:00:00.000000Z")
    return root, repo, config, child, popen_calls


def test_launch_stage_retries_transient_ack_read_failure_without_respawning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, repo, config, _child, popen_calls = _launch_ack_fixture(tmp_path, monkeypatch)
    original_read_text = Path.read_text
    ack_reads = 0

    def read_text(path: Path, *args, **kwargs):
        nonlocal ack_reads
        if path.parent == root / "launches" and path.suffix == ".json":
            ack_reads += 1
            if ack_reads == 1:
                raise FileNotFoundError(path)
            if ack_reads == 2:
                raise PermissionError(13, "temporarily unavailable", str(path))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(background.time, "sleep", lambda _seconds: None)

    result = launch_stage(root, "calibrate", repo_root=repo, config_path=config)

    assert result["state"] == "running"
    assert len(popen_calls) == 1
    assert ack_reads == 3


def test_launch_stage_persistent_ack_read_failure_reaches_deadline_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, repo, config, _child, popen_calls = _launch_ack_fixture(tmp_path, monkeypatch)
    original_read_text = Path.read_text
    now = 0.0

    def read_text(path: Path, *args, **kwargs):
        if path.parent == root / "launches" and path.suffix == ".json":
            raise PermissionError(13, "temporarily unavailable", str(path))
        return original_read_text(path, *args, **kwargs)

    def sleep(seconds: float):
        nonlocal now
        now += seconds

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(background.time, "monotonic", lambda: now)
    monkeypatch.setattr(background.time, "sleep", sleep)

    with pytest.raises(RuntimeError, match="did not acknowledge owned startup within 10 seconds"):
        launch_stage(root, "calibrate", repo_root=repo, config_path=config)

    assert len(popen_calls) == 1


@pytest.mark.parametrize("ack_value, error", [
    ({"state": "running"}, RuntimeError),
    ("{", json.JSONDecodeError),
])
def test_launch_stage_still_rejects_invalid_acknowledgment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ack_value, error):
    root, repo, config, _child, popen_calls = _launch_ack_fixture(tmp_path, monkeypatch, ack_value=ack_value)
    monkeypatch.setattr(background.time, "sleep", lambda _seconds: None)

    with pytest.raises(error):
        launch_stage(root, "calibrate", repo_root=repo, config_path=config)

    assert len(popen_calls) == 1


def test_status_ignores_stale_snapshot_and_explains_corrupt_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, _, repo, config = _registered(tmp_path, monkeypatch)
    assert json.loads((root / "execution.json").read_text(encoding="utf-8"))["schema_version"] == 1
    assert execution_status(root, repo_root=repo, config_path=config)["runtime_used_seconds"] == 0
    (root / "status.json").write_text('{"state":"completed"}\n', encoding="utf-8")
    status = execution_status(root, repo_root=repo, config_path=config)
    assert status["state"] == "registered"
    assert status["snapshot_stale"] is True
    (root / "events.jsonl").write_bytes(b"{truncated")
    broken = execution_status(root, repo_root=repo, config_path=config)
    assert broken["state"] == "stopped_environment"
    assert "truncated" in broken["diagnostic"].lower()


def test_settled_clock_chain_allows_future_registered_stage_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, _, _, _ = _registered(tmp_path, monkeypatch)
    owner = {"supervisor_pid": os.getpid(), "supervisor_created_utc": "2026-10-07T00:00:00.000000Z"}
    for attempt, stage, start, end, prior, settled in (
        ("a", "calibrate", 10, 1_000_000_010, 0, 1),
        ("b", "train47", 9_000_000_010, 10_000_000_010, 1, 2),
    ):
        common = {
            "execution_id": root.name, "attempt_id": attempt, "stage": stage,
            "prior_active_seconds": prior, "started_monotonic_ns": start,
            **owner, "stage_pid": os.getpid(), "stage_created_utc": "2026-10-07T00:00:01.000000Z",
            "stage_signature": "f" * 64,
        }
        execution.append_event(root, {"type": "job_started", **common})
        execution.append_event(root, {
            "type": "job_settled", **common, "ended_monotonic_ns": end,
            "settled_active_seconds": settled,
        })
    jobs = execution.validate_job_intervals(root)
    assert jobs["b"]["settled_active_seconds"] == 2


class _FakeJob:
    def __init__(self, root: Path, result: dict[str, object] | None, *, log_bytes: int = 0) -> None:
        self.root = root
        self.result = result
        self.log_bytes = log_bytes
        self.active_processes = 0
        self.terminated = False
        self.poll_calls = 0

    def start(self, _argv, *, stdout_path: Path, stderr_path: Path, cwd: Path):
        del cwd
        stdout_path.write_bytes(b"x" * self.log_bytes)
        stderr_path.write_bytes(b"")
        return {"pid": os.getpid(), "created_utc": "2026-10-07T00:00:00.000000Z"}

    def resume(self):
        if self.result is not None:
            (self.root / "stages" / "calibrate.json").write_bytes(_canonical(self.result) + b"\n")

    def poll(self):
        self.poll_calls += 1
        return 0

    def terminate_and_wait(self, timeout_seconds=10):
        self.terminated = True
        self.active_processes = 0
        return True

    def close(self):
        pass


def _supervisor_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, result_status: str | None = "awaiting_review", artifact_cap: int = 10_000_000, log_bytes: int = 0):
    root = tmp_path / ("e" * 64)
    root.mkdir()
    (root / "events.jsonl").touch()
    (root / "execution.json").write_bytes(_canonical({"execution_id": root.name, "schema_version": 1}) + b"\n")
    (root / "stages").mkdir()
    (root / "logs").mkdir()
    manifest = {
        "execution_id": root.name, "schema_version": 1, "source_root": str(tmp_path / "source"),
        "policy": {"active_runtime_seconds": 172800, "cuda_seconds": 86400,
                   "download_bytes": 21474836480, "artifact_bytes": artifact_cap},
        "artifact_baseline": {"source_bytes": 0, "feasibility_files": []},
        "source_run_id": "source-fixture",
        "source_manifest_sha256": "b" * 64,
        "source_prepare_sha256": "c" * 64,
        "inherited_costs": {"cuda_seconds": 1168, "download_bytes": 8066699435},
    }
    monkeypatch.setattr(background, "validate_execution", lambda *_a, **_k: manifest)
    signature = "a" * 64
    monkeypatch.setattr(background, "execution_stage_signature", lambda *_a, **_k: signature)
    policy_sha256 = __import__("hashlib").sha256(_canonical(manifest["policy"])).hexdigest()
    signature_record = study.calibration_stage_signature(
        Path(__file__).parents[1], source_run_id=manifest["source_run_id"],
        manifest_sha256=manifest["source_manifest_sha256"],
        prepare_sha256=manifest["source_prepare_sha256"],
        execution_stage_signature=signature, execution_policy_sha256=policy_sha256,
    )
    review = root / "calibration" / "review" / "packet.csv"
    review.parent.mkdir(parents=True)
    review.write_bytes(b"opaque-row,answer\n")
    result = None if result_status is None else {
        "execution_id": root.name, "stage": "calibrate", "status": result_status,
        "reason": "fixture outcome", "stage_signature": signature_record["stage_signature"],
        "signature": signature_record,
        "resource_ledger": {
            "cuda_seconds": 1175, "download_bytes": 8066699999,
            "artifact_bytes": 4096,
        },
        "resource_caps": {"cuda_seconds": 86400, "wall_seconds": 172800,
                          "download_bytes": 21474836480,
                          "artifact_bytes": artifact_cap},
        "inherited_artifact_bytes": 0,
        "review_packet": "calibration/review/packet.csv",
        "review_packet_sha256": __import__("hashlib").sha256(review.read_bytes()).hexdigest(),
        "private_review_map": "calibration/review/private-map.json",
    }
    job = _FakeJob(root, result, log_bytes=log_bytes)
    monkeypatch.setattr(background, "process_creation_utc", lambda _pid: "2026-10-07T00:00:00.000000Z")
    return root, manifest, job


@pytest.mark.parametrize(
    ("result_status", "expected_state", "terminal"),
    [("awaiting_review", "awaiting_review", False),
     ("completed", "completed", False),
     ("stopped_scientific", "stopped_scientific", True),
     (None, "stopped_environment", True)],
)
def test_stage_outcome_is_authenticated_and_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    result_status: str | None, expected_state: str, terminal: bool,
):
    root, _manifest, job = _supervisor_fixture(tmp_path, monkeypatch, result_status=result_status)
    result = supervise_stage(
        root, "calibrate", repo_root=tmp_path, config_path=tmp_path / "config.toml",
        _job_factory=lambda: job, _argv_builder=lambda *_a: ["cpu-fixture"],
        _attempt_id="test-attempt", _clock_ns=lambda: 10_000_000_000,
    )
    assert result in {0, 2}
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == expected_state
    assert (root / "terminal.json").exists() is terminal
    events = execution.read_events(root)
    assert any(event["payload"].get("type") == "job_settled" for event in events)
    assert job.poll_calls == 1
    reconstructed = execution_status(root, repo_root=tmp_path, config_path=tmp_path / "config.toml")
    if expected_state == "awaiting_review":
        assert reconstructed["costs"]["complete"] is True
        assert reconstructed["costs"]["cuda_seconds"]["total"] == 1175
        assert reconstructed["costs"]["download_bytes"]["total"] == 8066699999
        assert reconstructed["inherited_costs"]["cuda_seconds"] == 1168
        assert reconstructed["artifact_usage"]["total_bytes"] >= reconstructed["artifact_usage"]["new_subtree_bytes"]
        assert reconstructed["review_packet_paths"] == ["calibration/review/packet.csv"]
        assert reconstructed["cost_provenance"]["stage_artifact_sha256"] == next(
            event["payload"]["artifact_sha256"] for event in events
            if event["payload"].get("type") == "attempt_result"
        )
        markdown = (root / "status.md").read_text(encoding="utf-8")
        assert "CUDA" in markdown and "Download" in markdown and "Review packet" in markdown
        assert "private_review_map" not in markdown
        handoff = next(event["payload"] for event in events if event["payload"].get("type") == "stage_stopped")
        assert handoff["review_packet_paths"] == ["calibration/review/packet.csv"]
    elif result_status is None:
        assert reconstructed["costs"]["complete"] is False
        assert reconstructed["costs"]["cuda_seconds"]["total"] is None
        assert reconstructed["inherited_costs"]["cuda_seconds"] == 1168


@pytest.mark.parametrize("tamper", ["missing", "digest", "context", "stage"])
def test_producer_shaped_result_signature_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str,
):
    root, _manifest, job = _supervisor_fixture(tmp_path, monkeypatch)
    assert job.result is not None
    signature = job.result["signature"]
    if tamper == "missing":
        job.result.pop("signature")
    elif tamper == "digest":
        signature["stage_signature"] = "f" * 64
    elif tamper == "context":
        signature["execution_stage_signature"] = "e" * 64
    else:
        job.result["stage"] = "prepare"
    job.result = dict(job.result)
    supervise_stage(
        root, "calibrate", repo_root=tmp_path, config_path=tmp_path / "config.toml",
        _job_factory=lambda: job, _argv_builder=lambda *_a: ["cpu-fixture"],
        _attempt_id=f"bad-signature-{tamper}", _clock_ns=lambda: 10_000_000_000,
    )
    events = execution.read_events(root)
    outcome = next(event["payload"] for event in events if event["payload"].get("type") == "attempt_result")
    assert outcome["state"] == "stopped_environment"
    assert outcome["artifact_sha256"] is None


def test_status_rejects_tampered_stage_artifact_and_withholds_review_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root, _manifest, job = _supervisor_fixture(tmp_path, monkeypatch)
    supervise_stage(
        root, "calibrate", repo_root=tmp_path, config_path=tmp_path / "config.toml",
        _job_factory=lambda: job, _argv_builder=lambda *_a: ["cpu-fixture"],
        _attempt_id="tamper-after-settlement", _clock_ns=lambda: 10_000_000_000,
    )
    artifact = root / "stages" / "calibrate.json"
    artifact.write_bytes(artifact.read_bytes() + b" ")
    status = execution_status(root, repo_root=tmp_path, config_path=tmp_path / "config.toml")
    assert status["state"] == "stopped_environment"
    assert status["costs"]["complete"] is False
    assert status["review_packet_paths"] == []
    assert "changed" in status["diagnostic"].lower()


def test_unsafe_review_packet_is_not_settled_as_awaiting_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root, _manifest, job = _supervisor_fixture(tmp_path, monkeypatch)
    assert job.result is not None
    job.result["review_packet"] = "..\\outside.csv"
    result = supervise_stage(
        root, "calibrate", repo_root=tmp_path, config_path=tmp_path / "config.toml",
        _job_factory=lambda: job, _argv_builder=lambda *_a: ["cpu-fixture"],
        _attempt_id="unsafe-review-packet", _clock_ns=lambda: 10_000_000_000,
    )
    assert result == 2
    events = execution.read_events(root)
    outcome = next(event["payload"] for event in events if event["payload"].get("type") == "attempt_result")
    assert outcome["state"] == "stopped_environment"
    assert outcome["artifact_sha256"] is None
    assert not any(event["payload"].get("type") == "stage_stopped" for event in events)
    status = execution_status(root, repo_root=tmp_path, config_path=tmp_path / "config.toml")
    assert status["review_packet_paths"] == []
    assert "unsafe" in status["reason"]


def test_log_growth_triggers_owned_termination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root, _manifest, job = _supervisor_fixture(
        tmp_path, monkeypatch, result_status=None, artifact_cap=1024 * 1024 + 100, log_bytes=4096,
    )
    supervise_stage(
        root, "calibrate", repo_root=tmp_path, config_path=tmp_path / "config.toml",
        _job_factory=lambda: job, _argv_builder=lambda *_a: ["cpu-fixture"],
        _attempt_id="growth-attempt", _clock_ns=lambda: 10_000_000_000,
    )
    assert job.terminated is True
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "stopped_environment"
    assert "artifact" in status["reason"]


def test_settled_or_reviewed_stage_cannot_launch_twice_or_skip_to_future_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root, _manifest, job = _supervisor_fixture(tmp_path, monkeypatch)
    kwargs = {
        "repo_root": tmp_path, "config_path": tmp_path / "config.toml",
        "_job_factory": lambda: job, "_argv_builder": lambda *_a: ["cpu-fixture"],
        "_attempt_id": "first", "_clock_ns": lambda: 20_000_000_000,
    }
    supervise_stage(root, "calibrate", **kwargs)
    with pytest.raises(RuntimeError, match="already has a settled attempt"):
        supervise_stage(root, "calibrate", **{**kwargs, "_attempt_id": "second"})
    with pytest.raises(study.StageError, match="not implemented"):
        supervise_stage(root, "train47", **kwargs)
    assert job.poll_calls == 1


def test_invalid_policy_or_stage_signature_refuses_before_child_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root, _manifest, _job = _supervisor_fixture(tmp_path, monkeypatch)
    def forbidden_factory():
        raise AssertionError("worker must not be created after signature validation fails")
    monkeypatch.setattr(background, "execution_stage_signature", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("signature drift")))
    with pytest.raises(ValueError, match="signature drift"):
        supervise_stage(root, "calibrate", repo_root=tmp_path, config_path=tmp_path / "config.toml",
                        _job_factory=forbidden_factory, _argv_builder=lambda *_a: ["unused"])


@pytest.mark.skipif(os.name != "nt", reason="watchdog process-tree test requires Windows Job Objects")
def test_watchdog_deadline_terminates_real_cpu_child_and_grandchild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root, _manifest, _job = _supervisor_fixture(tmp_path, monkeypatch, result_status=None)
    work = tmp_path / "watchdog child path with spaces"
    work.mkdir()
    child = work / "parent.py"
    marker = work / "grandchild marker.txt"
    child.write_text(
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable, '-c', \"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).write_text('escaped')\"])\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )

    class AdvancingClock:
        def __init__(self):
            self.value = time.monotonic_ns()

        def __call__(self):
            return self.value

        def advance(self, _seconds: float):
            self.value += 172800 * 1_000_000_000

    clock = AdvancingClock()
    result = supervise_stage(
        root, "calibrate", repo_root=tmp_path, config_path=tmp_path / "config.toml",
        _job_factory=WindowsJob,
        _argv_builder=lambda *_a: [sys.executable, str(child)],
        _clock_ns=clock, _sleep=clock.advance, _attempt_id="watchdog-attempt",
    )
    assert result == 2
    status = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    assert status["state"] == "stopped_environment"
    assert "runtime" in status["reason"]
    time.sleep(1.2)
    assert not marker.exists()


@pytest.mark.skipif(os.name != "nt", reason="requires detached Windows supervisor process")
def test_detached_cpu_launcher_exits_before_stage_and_other_invocation_reads_status(tmp_path: Path):
    workspace = tmp_path / "cpu fixture path with spaces"
    prepared = subprocess.run(
        [sys.executable, str(CPU_FIXTURE), "prepare", "--workspace", str(workspace)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15,
    )
    assert prepared.returncode == 0, prepared.stderr
    launched = subprocess.run(
        [sys.executable, str(CPU_FIXTURE), "start", "--workspace", str(workspace)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15,
    )
    assert launched.returncode == 0, launched.stderr
    start_record = json.loads(launched.stdout)
    stdout_path = Path(start_record["stdout_path"])
    assert "finished" not in stdout_path.read_text(encoding="utf-8")

    deadline = time.monotonic() + 10
    status: dict[str, object] = {}
    while time.monotonic() < deadline:
        result = subprocess.run(
            [sys.executable, str(CPU_FIXTURE), "status", "--workspace", str(workspace)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr
        status = json.loads(result.stdout)
        if status["state"] == "awaiting_review":
            break
        time.sleep(0.2)
    assert status["state"] == "awaiting_review"
    assert "finished" in stdout_path.read_text(encoding="utf-8")
    assert status["runtime_remaining_seconds"] < 172800


@pytest.mark.skipif(os.name != "nt", reason="requires detached Windows supervisor process")
def test_detached_recovery_job_carries_31_second_clock_and_exits_cleanly(tmp_path: Path):
    workspace = tmp_path / "recovery cpu fixture path with spaces"
    prepared = subprocess.run(
        [sys.executable, str(CPU_FIXTURE), "prepare-recovery", "--workspace", str(workspace)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
    )
    assert prepared.returncode == 0, prepared.stderr
    fixture = json.loads(prepared.stdout)
    root = Path(fixture["execution_root"])
    initial = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert initial["runtime_used_seconds"] == 31
    assert initial["runtime_remaining_seconds"] == 172769

    launched = subprocess.run(
        [sys.executable, str(CPU_FIXTURE), "start", "--workspace", str(workspace)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20,
    )
    assert launched.returncode == 0, launched.stderr
    launch = json.loads(launched.stdout)
    stdout_path = Path(launch["stdout_path"])
    assert "finished" not in stdout_path.read_text(encoding="utf-8")

    deadline = time.monotonic() + 12
    status: dict[str, object] = {}
    while time.monotonic() < deadline:
        result = subprocess.run(
            [sys.executable, str(CPU_FIXTURE), "status", "--workspace", str(workspace)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20,
        )
        assert result.returncode == 0, result.stderr
        status = json.loads(result.stdout)
        if status["state"] == "awaiting_review":
            break
        time.sleep(0.2)
    assert status["state"] == "awaiting_review"
    stage = json.loads((root / "stages" / "calibrate.json").read_text(encoding="utf-8"))
    assert stage["stage"] == "calibrate"
    assert stage["started_at_utc"]
    assert stage["ended_at_utc"]
    journal = [json.loads(line) for line in (root / "stages" / "calibrate-reservations.jsonl").read_text(encoding="utf-8").splitlines()]
    terminal = [row for row in journal if row.get("event_type") == "terminal"]
    assert len(terminal) == 1
    assert terminal[0]["started_at_utc"] == stage["started_at_utc"]
    assert terminal[0]["ended_at_utc"] == stage["ended_at_utc"]
    events = execution.read_events(root)
    payloads = [event["payload"] for event in events]
    started = next(item for item in payloads if item.get("type") == "job_started")
    settled = next(item for item in payloads if item.get("type") == "job_settled")
    assert started["prior_active_seconds"] == 31
    assert settled["settled_active_seconds"] > 31
    assert settled["settled_active_seconds"] == 31 + (settled["ended_monotonic_ns"] - settled["started_monotonic_ns"] + 999_999_999) // 1_000_000_000
    assert "finished" in stdout_path.read_text(encoding="utf-8")
    _assert_process_identities_exited([
        (started["stage_pid"], started["stage_created_utc"]),
        (started["supervisor_pid"], started["supervisor_created_utc"]),
    ])


@pytest.mark.skipif(os.name != "nt", reason="requires detached Windows supervisor process")
def test_detached_repair_job_carries_60_second_clock_and_cumulative_artifact_baseline(tmp_path: Path):
    workspace = tmp_path / "repair cpu fixture path with spaces"
    prepared = subprocess.run(
        [sys.executable, str(CPU_FIXTURE), "prepare-repair", "--workspace", str(workspace)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
    )
    assert prepared.returncode == 0, prepared.stderr
    fixture = json.loads(prepared.stdout)
    root = Path(fixture["execution_root"])
    manifest = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    initial = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert fixture["schema_version"] == 3
    assert initial["runtime_used_seconds"] == 60
    assert initial["runtime_remaining_seconds"] == 172740
    projected = subprocess.run(
        [sys.executable, str(CPU_FIXTURE), "status", "--workspace", str(workspace)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
    )
    assert projected.returncode == 0, projected.stderr
    projection = json.loads(projected.stdout)
    assert projection["runtime_used_seconds"] == 60
    assert projection["artifact_usage"]["inherited_artifact_baseline_bytes"] == (
        2047868 + 42270 + 4744 + manifest["artifact_baseline"]["repair_authorization_reservation"]["size_bytes"]
    )

    launched = subprocess.run(
        [sys.executable, str(CPU_FIXTURE), "start", "--workspace", str(workspace)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20,
    )
    assert launched.returncode == 0, launched.stderr
    launch = json.loads(launched.stdout)
    stdout_path = Path(launch["stdout_path"])
    deadline = time.monotonic() + 12
    status: dict[str, object] = {}
    while time.monotonic() < deadline:
        result = subprocess.run(
            [sys.executable, str(CPU_FIXTURE), "status", "--workspace", str(workspace)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20,
        )
        assert result.returncode == 0, result.stderr
        status = json.loads(result.stdout)
        if status["state"] == "awaiting_review":
            break
        time.sleep(0.2)
    assert status["state"] == "awaiting_review"
    assert status["runtime_used_seconds"] >= 60
    assert status["runtime_remaining_seconds"] <= 172740
    events = execution.read_events(root)
    payloads = [event["payload"] for event in events]
    started = next(item for item in payloads if item.get("type") == "job_started")
    settled = next(item for item in payloads if item.get("type") == "job_settled")
    assert started["prior_active_seconds"] == 60
    assert settled["settled_active_seconds"] >= 60
    assert "finished" in stdout_path.read_text(encoding="utf-8")
    _assert_process_identities_exited([
        (started["stage_pid"], started["stage_created_utc"]),
        (started["supervisor_pid"], started["supervisor_created_utc"]),
    ])


@pytest.mark.skipif(os.name != "nt", reason="requires real concurrent detached Windows supervisors")
def test_concurrent_detached_launchers_have_one_attempt_owner(tmp_path: Path):
    workspace = tmp_path / "concurrent cpu fixture"
    prepared = subprocess.run(
        [sys.executable, str(CPU_FIXTURE), "prepare", "--workspace", str(workspace)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15,
    )
    assert prepared.returncode == 0, prepared.stderr
    command = [sys.executable, str(CPU_FIXTURE), "start", "--workspace", str(workspace)]
    launches = [subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for _ in range(2)]
    results = [process.communicate(timeout=15) for process in launches]
    assert sorted(process.returncode for process in launches) == [0, 1]
    succeeded = json.loads(next(stdout for process, (stdout, _stderr) in zip(launches, results) if process.returncode == 0))
    fixture = json.loads((workspace / "fixture.json").read_text(encoding="utf-8"))
    root = Path(fixture["execution_root"])
    events = execution.read_events(root)
    starts = [event["payload"] for event in events if event["payload"].get("type") == "job_started"]
    assert len(starts) == 1
    assert starts[0]["attempt_id"] == succeeded["attempt_id"]
