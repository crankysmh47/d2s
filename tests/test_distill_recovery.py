from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest

from ppmoe.distill.execution import append_event, validate_job_intervals
from ppmoe.distill import background


PREDECESSOR_ID = "fc901bb0766a9c79fb205302d220fe727a99ebbd0b315e6ee4ad4e66e6733634"
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "distillation-v1.toml"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _predecessor_copy(tmp_path: Path) -> tuple[Path, Path]:
    source = REPO_ROOT / "artifacts" / "distillation-v1-executions" / PREDECESSOR_ID
    parent = tmp_path / "executions"
    parent.mkdir()
    target = parent / PREDECESSOR_ID
    shutil.copytree(source, target)
    before = {path.relative_to(target).as_posix(): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    return target, before


def _patch_disposable_pins(monkeypatch: pytest.MonkeyPatch, previous: Path) -> None:
    from hashlib import sha256
    from ppmoe.distill import recovery

    reviewed = {
        path.relative_to(previous).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in previous.rglob("*") if path.is_file() and path.name not in {"status.json", "status.md"}
    }
    monkeypatch.setattr(recovery, "REVIEWED_IMMUTABLE_FILES", reviewed)


def test_recovery_registration_module_exposes_single_linked_recovery_api():
    spec = importlib.util.find_spec("ppmoe.distill.recovery")
    assert spec is not None
    module = __import__("ppmoe.distill.recovery", fromlist=["register_recovery"])
    assert callable(getattr(module, "register_recovery", None))
    assert callable(getattr(module, "validate_recovery_manifest", None))


def test_registration_carries_authenticated_clock_and_costs_without_touching_predecessor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ppmoe.distill.recovery import register_recovery
    from ppmoe.distill.execution import artifact_baseline_bytes, validate_execution

    previous, before = _predecessor_copy(tmp_path)
    _patch_disposable_pins(monkeypatch, previous)
    root = register_recovery(previous, previous.parent, repo_root=REPO_ROOT, config_path=CONFIG)
    manifest = validate_execution(root, repo_root=REPO_ROOT, config_path=CONFIG)
    after = {path.relative_to(previous).as_posix(): path.read_bytes() for path in previous.rglob("*") if path.is_file()}

    assert after == before
    assert manifest["schema_version"] == 2
    assert manifest["inherited_active_seconds"] == 31
    assert manifest["inherited_costs"] == {"cuda_seconds": 1199, "download_bytes": 8066717862}
    assert manifest["source_root"] == json.loads((previous / "execution.json").read_text())["source_root"]
    assert manifest["policy"]["active_runtime_seconds"] == 172800
    assert manifest["recovery"]["predecessor_execution_id"] == PREDECESSOR_ID
    assert manifest["recovery"]["one_hop"] is True
    reservation = manifest["recovery"]["authorization_reservation"]
    assert artifact_baseline_bytes(manifest) == 2047565 + reservation["size_bytes"]
    assert manifest["recovery"]["accounting_rationale"]["producer_reported_cuda_seconds"] == 1190
    assert validate_job_intervals(root) == {}


def test_recovery_status_uses_inherited_clock_before_any_new_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ppmoe.distill.recovery import register_recovery

    previous, _before = _predecessor_copy(tmp_path)
    _patch_disposable_pins(monkeypatch, previous)
    root = register_recovery(previous, previous.parent, repo_root=REPO_ROOT, config_path=CONFIG)
    append_event(root, {"type": "registered", "execution_id": root.name})
    status = background.execution_status(root, repo_root=REPO_ROOT, config_path=CONFIG)

    assert status["runtime_used_seconds"] == 31
    assert status["runtime_remaining_seconds"] == 172769


def test_recovery_refuses_a_second_attempt_even_after_registration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ppmoe.distill.recovery import register_recovery

    previous, _before = _predecessor_copy(tmp_path)
    _patch_disposable_pins(monkeypatch, previous)
    first = register_recovery(previous, previous.parent, repo_root=REPO_ROOT, config_path=CONFIG)
    with pytest.raises((FileExistsError, ValueError), match="recover|authoriz|exist|registered"):
        register_recovery(previous, previous.parent, repo_root=REPO_ROOT, config_path=CONFIG)
    assert first.is_dir()
    assert len([path for path in previous.parent.iterdir() if path.is_dir()]) == 2


def test_first_job_starts_from_inherited_31_and_settles_cumulatively(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from hashlib import sha256
    from ppmoe.distill.execution import artifact_baseline_bytes, read_events
    from ppmoe.distill.recovery import register_recovery
    from ppmoe.distill.study import calibration_stage_signature

    previous, _before = _predecessor_copy(tmp_path)
    _patch_disposable_pins(monkeypatch, previous)
    root = register_recovery(previous, previous.parent, repo_root=REPO_ROOT, config_path=CONFIG)
    manifest = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    execution_signature = __import__("ppmoe.distill.execution", fromlist=["execution_stage_signature"]).execution_stage_signature(
        root, "calibrate", repo_root=REPO_ROOT,
    )
    signature = calibration_stage_signature(
        REPO_ROOT, source_run_id=manifest["source_run_id"],
        manifest_sha256=manifest["source_manifest_sha256"],
        prepare_sha256=manifest["source_prepare_sha256"],
        execution_stage_signature=execution_signature,
        execution_policy_sha256=sha256(_canonical(manifest["policy"])).hexdigest(),
    )
    packet = root / "calibration" / "review" / "packet.csv"
    packet.parent.mkdir(parents=True)
    packet.write_bytes(b"opaque_id,answer\n")
    artifact = {
        "execution_id": root.name, "stage": "calibrate", "status": "awaiting_review",
        "reason": "CPU recovery fixture", "stage_signature": signature["stage_signature"],
        "signature": signature,
        "resource_ledger": {"cuda_seconds": 1200, "download_bytes": 8066718863, "artifact_bytes": packet.stat().st_size},
        "resource_caps": {"cuda_seconds": 86400, "wall_seconds": 172800, "download_bytes": 21474836480, "artifact_bytes": 21474836480},
        "inherited_artifact_bytes": artifact_baseline_bytes(manifest),
        "review_packet": "calibration/review/packet.csv",
        "review_packet_sha256": sha256(packet.read_bytes()).hexdigest(),
    }

    class CompletedCpuJob:
        active_processes = 0

        def start(self, _argv, *, stdout_path: Path, stderr_path: Path, cwd: Path):
            del cwd
            stdout_path.write_bytes(b"")
            stderr_path.write_bytes(b"")
            return {"pid": os.getpid(), "created_utc": "2026-10-07T00:00:00.000000Z"}

        def resume(self):
            (root / "stages" / "calibrate.json").write_bytes(_canonical(artifact) + b"\n")

        def poll(self):
            return 0

        def terminate_and_wait(self, timeout_seconds=10):
            del timeout_seconds
            return True

        def close(self):
            pass

    monkeypatch.setattr(background, "process_creation_utc", lambda _pid: "2026-10-07T00:00:00.000000Z")
    clock = iter((100, 100, 100, 1_000_000_100))
    result = background.supervise_stage(
        root, "calibrate", repo_root=REPO_ROOT, config_path=CONFIG,
        _job_factory=CompletedCpuJob, _argv_builder=lambda *_args: ["cpu-fixture"],
        _attempt_id="inherited-clock-proof", _clock_ns=lambda: next(clock),
    )
    assert result == 0, (root / "terminal.json").read_text(encoding="utf-8") if (root / "terminal.json").exists() else (root / "status.json").read_text(encoding="utf-8")

    events = [row["payload"] for row in read_events(root)]
    started = next(row for row in events if row.get("type") == "job_started")
    settled = next(row for row in events if row.get("type") == "job_settled")
    assert started["prior_active_seconds"] == 31
    assert settled["settled_active_seconds"] == 32
    assert background.execution_status(root, repo_root=REPO_ROOT, config_path=CONFIG)["runtime_used_seconds"] == 32


def test_recovery_baseline_rejects_mutated_immutable_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ppmoe.distill.recovery import register_recovery

    previous, _before = _predecessor_copy(tmp_path)
    _patch_disposable_pins(monkeypatch, previous)
    with (previous / "logs" / "dd45bbdd2ecf45cb8d8fcc653132ee0a.stdout.log").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="pinned evidence|inventory"):
        register_recovery(previous, previous.parent, repo_root=REPO_ROOT, config_path=CONFIG)
    assert not list(previous.parent.glob("recovery-authorization-*.json"))


def test_mutable_predecessor_status_uses_frozen_observed_reserve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ppmoe.distill.execution import validate_execution
    from ppmoe.distill.recovery import register_recovery

    previous, _before = _predecessor_copy(tmp_path)
    _patch_disposable_pins(monkeypatch, previous)
    root = register_recovery(previous, previous.parent, repo_root=REPO_ROOT, config_path=CONFIG)
    (previous / "status.json").write_bytes(b"status was refreshed with a longer snapshot\n")
    manifest = validate_execution(root, repo_root=REPO_ROOT, config_path=CONFIG)
    assert manifest["artifact_baseline"]["predecessor_status_reserve_bytes"] == 4482


def test_two_concurrent_registrations_consume_one_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ppmoe.distill.recovery import register_recovery

    previous, _before = _predecessor_copy(tmp_path)
    _patch_disposable_pins(monkeypatch, previous)
    def attempt():
        try:
            return register_recovery(previous, previous.parent, repo_root=REPO_ROOT, config_path=CONFIG)
        except (FileExistsError, ValueError):
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: attempt(), range(2)))
    assert sum(result is not None for result in results) == 1
    assert len(list(previous.parent.glob("recovery-authorization-*.json"))) == 1
    assert len([path for path in previous.parent.iterdir() if path.is_dir()]) == 2


def test_failed_registration_keeps_exclusive_reservation_consumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ppmoe.distill import recovery

    previous, _before = _predecessor_copy(tmp_path)
    _patch_disposable_pins(monkeypatch, previous)
    original = recovery._sha_file
    def fail_authorization(path: Path) -> str:
        if path.name.startswith("recovery-authorization-"):
            raise OSError("fixture failure after exclusive reservation")
        return original(path)
    monkeypatch.setattr(recovery, "_sha_file", fail_authorization)
    with pytest.raises(OSError, match="after exclusive reservation"):
        recovery.register_recovery(previous, previous.parent, repo_root=REPO_ROOT, config_path=CONFIG)
    monkeypatch.setattr(recovery, "_sha_file", original)
    with pytest.raises(FileExistsError, match="reservation already exists"):
        recovery.register_recovery(previous, previous.parent, repo_root=REPO_ROOT, config_path=CONFIG)


def test_recovery_refuses_any_preexisting_partial_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ppmoe.distill.recovery import register_recovery

    previous, _before = _predecessor_copy(tmp_path)
    _patch_disposable_pins(monkeypatch, previous)
    (previous.parent / "recovery-authorization-partial.json").write_bytes(b"partial")
    with pytest.raises(FileExistsError, match="reservation already exists"):
        register_recovery(previous, previous.parent, repo_root=REPO_ROOT, config_path=CONFIG)
    assert not (previous.parent / f"recovery-authorization-{PREDECESSOR_ID}.json").exists()


def test_recovery_refuses_a_live_predecessor_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ppmoe.distill import execution
    from ppmoe.distill.recovery import register_recovery

    previous, _before = _predecessor_copy(tmp_path)
    _patch_disposable_pins(monkeypatch, previous)
    terminal = json.loads((previous / "terminal.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(
        execution, "process_creation_utc",
        lambda pid: terminal["supervisor_created_utc"] if pid == terminal["supervisor_pid"] else None,
    )
    with pytest.raises(ValueError, match="live process owner"):
        register_recovery(previous, previous.parent, repo_root=REPO_ROOT, config_path=CONFIG)
    assert not list(previous.parent.glob("recovery-authorization-*.json"))


@pytest.mark.parametrize("bad_value", [True, -1, 32])
def test_recovery_manifest_rejects_clock_refunds_even_when_rehashed(bad_value: int | bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ppmoe.distill.recovery import register_recovery, validate_recovery_manifest

    previous, _before = _predecessor_copy(tmp_path)
    _patch_disposable_pins(monkeypatch, previous)
    root = register_recovery(previous, previous.parent, repo_root=REPO_ROOT, config_path=CONFIG)
    manifest = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    manifest["inherited_active_seconds"] = bad_value
    with pytest.raises(ValueError, match="clock/cost baseline"):
        validate_recovery_manifest(manifest, repo_root=REPO_ROOT, config_path=CONFIG)


def test_recovery_manifest_rejects_rehashed_cost_refunds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ppmoe.distill.recovery import register_recovery, validate_recovery_manifest

    previous, _before = _predecessor_copy(tmp_path)
    _patch_disposable_pins(monkeypatch, previous)
    root = register_recovery(previous, previous.parent, repo_root=REPO_ROOT, config_path=CONFIG)
    manifest = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    manifest["inherited_costs"]["cuda_seconds"] = 1190
    with pytest.raises(ValueError, match="clock/cost baseline"):
        validate_recovery_manifest(manifest, repo_root=REPO_ROOT, config_path=CONFIG)


def test_recovery_cli_registers_schema2_and_rejects_budget_override_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    import importlib.util

    previous, _before = _predecessor_copy(tmp_path)
    _patch_disposable_pins(monkeypatch, previous)
    spec = importlib.util.spec_from_file_location("distillation_background_cli", REPO_ROOT / "scripts" / "run_distillation_background.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = [
        "register-recovery", "--previous-execution-root", str(previous),
        "--executions-root", str(previous.parent), "--config", str(CONFIG),
        "--repo-root", str(REPO_ROOT),
    ]
    assert module.main(argv) == 0
    output = json.loads(capsys.readouterr().out)
    root = Path(output["execution_root"])
    assert json.loads((root / "execution.json").read_text(encoding="utf-8"))["schema_version"] == 2
    assert json.loads((root / "status.json").read_text(encoding="utf-8"))["runtime_used_seconds"] == 31
    with pytest.raises(SystemExit):
        module.main(argv + ["--active-runtime-seconds", "172800"])
