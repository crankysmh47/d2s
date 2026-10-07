from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest


REPO_ROOT = Path(__file__).parents[1]
CONFIG_PATH = REPO_ROOT / "configs/distillation-v1.toml"
PREDECESSOR_ID = "b0c082cba66a37f6f49bacc4bee182102b926ac5ac3af4c0a657823213e1802f"
PREDECESSOR = REPO_ROOT / "artifacts/distillation-v1-executions" / PREDECESSOR_ID


def _detached_schema2_predecessor(tmp_path: Path) -> tuple[Path, Path]:
    executions = tmp_path / "executions"
    executions.mkdir()
    copied = executions / PREDECESSOR_ID
    shutil.copytree(PREDECESSOR, copied)
    return copied, executions


def test_repair_registration_authenticates_pinned_schema2_and_accounts_once(tmp_path: Path) -> None:
    from ppmoe.distill.execution import artifact_baseline_bytes, validate_execution
    from ppmoe.distill.repair_recovery import register_repair

    previous, executions = _detached_schema2_predecessor(tmp_path)
    old_manifest_bytes = (previous / "execution.json").read_bytes()
    old_status_bytes = (previous / "status.json").read_bytes()
    root = register_repair(
        previous, executions, repo_root=REPO_ROOT, config_path=CONFIG_PATH,
    )
    manifest = validate_execution(root, repo_root=REPO_ROOT, config_path=CONFIG_PATH)

    assert manifest["schema_version"] == 3
    assert manifest["inherited_active_seconds"] == 60
    assert manifest["inherited_costs"] == {"cuda_seconds": 1228, "download_bytes": 8066717862}
    assert manifest["repair"]["predecessor_execution_id"] == PREDECESSOR_ID
    assert len(manifest["repair"]["predecessor_inventory"]) == 13
    assert manifest["artifact_baseline"]["repair_predecessor_immutable_bytes"] == 42270
    assert manifest["artifact_baseline"]["repair_predecessor_status_reserve_bytes"] == 4744
    reservation = manifest["artifact_baseline"]["repair_authorization_reservation"]
    assert artifact_baseline_bytes(manifest) == 2047868 + 42270 + 4744 + reservation["size_bytes"]
    assert (previous / "execution.json").read_bytes() == old_manifest_bytes
    assert (previous / "status.json").read_bytes() == old_status_bytes
    assert not list(executions.glob("recovery-authorization-*.json"))
    assert validate_execution(previous, repo_root=REPO_ROOT, config_path=CONFIG_PATH)["schema_version"] == 2
    with pytest.raises(FileExistsError, match="single-use"):
        register_repair(previous, executions, repo_root=REPO_ROOT, config_path=CONFIG_PATH)


def _real_schema3_execution(tmp_path: Path):
    from ppmoe.distill.execution import validate_execution
    from ppmoe.distill.repair_recovery import register_repair

    previous, executions = _detached_schema2_predecessor(tmp_path)
    root = register_repair(previous, executions, repo_root=REPO_ROOT, config_path=CONFIG_PATH)
    manifest = validate_execution(root, repo_root=REPO_ROOT, config_path=CONFIG_PATH)
    return root, manifest


def test_schema3_pre_model_stop_uses_real_manifest_clock_and_cost_seed(tmp_path: Path, monkeypatch) -> None:
    import time
    from ppmoe.distill import execution, study
    from ppmoe.distill.config import load_config
    from ppmoe.distill.evaluate import ResourceLimitError
    from ppmoe.distill.execution import ActiveJobContext
    from ppmoe.distill.background import _base_status

    root, manifest = _real_schema3_execution(tmp_path)
    reservation_history = root / "stages" / "calibrate-reservations.jsonl"
    reservation_history.write_text(json.dumps({
        "event_type": "download_reservation", "status": "reserved",
        "reserved_bytes": 37, "stage": "calibrate", "role": "student",
    }) + "\n", encoding="utf-8")
    projection = _base_status(root, manifest, {}, events=[])
    assert projection["runtime_used_seconds"] == 60
    assert projection["runtime_remaining_seconds"] == 172740
    assert projection["artifact_usage"]["inherited_artifact_baseline_bytes"] == (
        2047868 + 42270 + 4744 + manifest["artifact_baseline"]["repair_authorization_reservation"]["size_bytes"]
    )
    signature = execution.execution_stage_signature(root, "calibrate", repo_root=REPO_ROOT)
    context = ActiveJobContext(
        execution_id=root.name, attempt_id="repair-pre-model", stage="calibrate",
        prior_active_seconds=60, started_monotonic_ns=time.monotonic_ns(),
        supervisor_pid=1, supervisor_created_utc="2026-10-07T00:00:00.000000Z",
        stage_signature=signature,
    )
    monkeypatch.setattr(execution, "load_active_job", lambda *_a, **_k: context)
    monkeypatch.setattr(study, "_execution_job_intervals", lambda *_a, **_k: {})
    monkeypatch.setattr(
        study, "preflight_disk",
        lambda *_a, **_k: (_ for _ in ()).throw(ResourceLimitError("synthetic pre-model stop")),
    )
    result = study.run_calibration_stage(
        Path(manifest["source_root"]), load_config(CONFIG_PATH), repo_root=REPO_ROOT,
        config_path=CONFIG_PATH, execution_root=root, active_job=context,
        worker_loader=lambda *_a: pytest.fail("pre-model stop constructed a worker"),
    )

    assert result["status"] == "stopped_environment"
    assert result["resource_ledger"]["cuda_seconds"] == 1228
    assert result["resource_ledger"]["download_bytes"] == 8066717862 + 37
    reservations = [json.loads(line) for line in reservation_history.read_text(encoding="utf-8").splitlines()]
    stage_reservation = next(item for item in reservations if item.get("event_type") == "reservation")
    assert stage_reservation["initial_cuda_seconds"] == 1228
    assert stage_reservation["initial_download_bytes"] == 8066717862
    resource_stop = next(item for item in reservations if item.get("event_type") == "resource_stop")
    assert resource_stop["resource_ledger"]["download_bytes"] == 8066717862 + 37
    saved = json.loads((root / "stages/calibrate.json").read_text(encoding="utf-8"))
    assert saved["inherited_artifact_bytes"] == 2094882 + manifest["artifact_baseline"]["repair_authorization_reservation"]["size_bytes"]
    assert saved["resource_ledger"]["wall_seconds"] >= 60


def test_initial_active_clock_is_explicit_for_schemas_one_two_and_three_and_rejects_bool() -> None:
    from ppmoe.distill.execution import initial_active_seconds

    assert initial_active_seconds({"schema_version": 1}) == 0
    assert initial_active_seconds({"schema_version": 2, "inherited_active_seconds": 31, "recovery": {"inherited_active_seconds": 31}}) == 31
    assert initial_active_seconds({"schema_version": 3, "inherited_active_seconds": 60, "repair": {"inherited_active_seconds": 60}}) == 60
    with pytest.raises(ValueError, match="schema"):
        initial_active_seconds({"schema_version": True})
    with pytest.raises(ValueError, match="clock"):
        initial_active_seconds({"schema_version": 3, "inherited_active_seconds": True, "repair": {"inherited_active_seconds": True}})
    with pytest.raises(ValueError, match="clock"):
        initial_active_seconds({"schema_version": 3, "inherited_active_seconds": 60, "repair": {"inherited_active_seconds": 60.0}})


@pytest.mark.parametrize("mutation", ["changed", "extra", "missing"])
def test_repair_registration_rejects_nonexact_predecessor_inventory_before_consuming_slot(
    tmp_path: Path, mutation: str,
) -> None:
    from ppmoe.distill.repair_recovery import register_repair

    previous, executions = _detached_schema2_predecessor(tmp_path)
    target = previous / "terminal.json"
    if mutation == "changed":
        target.write_bytes(target.read_bytes() + b" ")
    elif mutation == "extra":
        (previous / "unexpected.txt").write_text("extra", encoding="utf-8")
    else:
        target.unlink()
    with pytest.raises((OSError, ValueError)):
        register_repair(previous, executions, repo_root=REPO_ROOT, config_path=CONFIG_PATH)
    assert not list(executions.glob("repair-authorization-*.json"))


def test_repair_registration_rejects_wrong_namespace_chain_and_live_owner(tmp_path: Path, monkeypatch) -> None:
    from ppmoe.distill import execution
    from ppmoe.distill.repair_recovery import register_repair

    previous, executions = _detached_schema2_predecessor(tmp_path)
    with pytest.raises(ValueError, match="parent"):
        register_repair(previous, tmp_path / "other-executions", repo_root=REPO_ROOT, config_path=CONFIG_PATH)
    chained = executions / "not-the-reviewed-id"
    chained.mkdir()
    with pytest.raises(ValueError, match="only the pinned"):
        register_repair(chained, executions, repo_root=REPO_ROOT, config_path=CONFIG_PATH)

    terminal = json.loads((previous / "terminal.json").read_text(encoding="utf-8"))
    live_identity = terminal["supervisor_created_utc"]
    monkeypatch.setattr(execution, "process_creation_utc", lambda _pid: live_identity)
    with pytest.raises(ValueError, match="live process owner"):
        register_repair(previous, executions, repo_root=REPO_ROOT, config_path=CONFIG_PATH)
    assert not list(executions.glob("repair-authorization-*.json"))


@pytest.mark.parametrize("state", ["stopped_scientific", "unknown_terminal"])
def test_repair_registration_rejects_nonreviewed_stop_before_slot_creation(tmp_path: Path, state: str) -> None:
    from ppmoe.distill.repair_recovery import register_repair

    previous, executions = _detached_schema2_predecessor(tmp_path)
    terminal_path = previous / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["state"] = state
    terminal_path.write_text(json.dumps(terminal, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory|pin|terminal"):
        register_repair(previous, executions, repo_root=REPO_ROOT, config_path=CONFIG_PATH)
    assert not list(executions.glob("repair-authorization-*.json"))


def test_repair_registration_rejects_unsettled_job_before_slot_creation(tmp_path: Path, monkeypatch) -> None:
    from ppmoe.distill import execution
    from ppmoe.distill.repair_recovery import register_repair

    previous, executions = _detached_schema2_predecessor(tmp_path)
    monkeypatch.setattr(execution, "validate_job_intervals", lambda *_a, **_k: {
        "unsettled": {"settlement": None, "prior_active_seconds": 31, "settled_active_seconds": None},
    })
    with pytest.raises(ValueError, match="settle"):
        register_repair(previous, executions, repo_root=REPO_ROOT, config_path=CONFIG_PATH)
    assert not list(executions.glob("repair-authorization-*.json"))


def test_repair_registration_rejects_predecessor_symlink_before_slot_creation(tmp_path: Path) -> None:
    from ppmoe.distill.repair_recovery import register_repair

    previous, executions = _detached_schema2_predecessor(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("linked", encoding="utf-8")
    try:
        (previous / "unexpected-link.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Windows symlink creation is unavailable: {exc}")
    with pytest.raises(ValueError, match="symlink/reparse"):
        register_repair(previous, executions, repo_root=REPO_ROOT, config_path=CONFIG_PATH)
    assert not list(executions.glob("repair-authorization-*.json"))


def test_concurrent_repair_registration_consumes_one_slot_and_creates_one_execution(tmp_path: Path) -> None:
    from ppmoe.distill.repair_recovery import register_repair

    previous, executions = _detached_schema2_predecessor(tmp_path)
    def register():
        try:
            return register_repair(previous, executions, repo_root=REPO_ROOT, config_path=CONFIG_PATH)
        except (FileExistsError, OSError) as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _index: register(), range(2)))
    successes = [value for value in outcomes if isinstance(value, Path)]
    failures = [value for value in outcomes if isinstance(value, (FileExistsError, OSError))]
    assert len(successes) == len(failures) == 1
    assert len(list(executions.glob("repair-authorization-*.json"))) == 1
    assert len([path for path in executions.iterdir() if path.is_dir() and path.name != PREDECESSOR_ID]) == 1


@pytest.mark.parametrize("mutation", ["cost_refund", "clock_refund", "old_artifact_carry", "new_inventory_carry"])
def test_rehashed_schema3_manifest_rejects_refunds_and_changed_artifact_carry(
    tmp_path: Path, mutation: str,
) -> None:
    from ppmoe.distill import execution

    root, _manifest = _real_schema3_execution(tmp_path)
    copied_parent = tmp_path / "rebound"
    copied_parent.mkdir()
    copied = copied_parent / root.name
    shutil.copytree(root, copied)
    value = json.loads((copied / "execution.json").read_text(encoding="utf-8"))
    if mutation == "cost_refund":
        value["inherited_costs"]["cuda_seconds"] = 1199
        value["repair"]["inherited_costs"]["cuda_seconds"] = 1199
    elif mutation == "clock_refund":
        value["inherited_active_seconds"] = 31
        value["repair"]["inherited_active_seconds"] = 31
    elif mutation == "old_artifact_carry":
        value["artifact_baseline"]["predecessor_immutable_bytes"] -= 1
    else:
        value["artifact_baseline"]["repair_predecessor_immutable_bytes"] = 0
    value["execution_id"] = execution._sha({key: item for key, item in value.items() if key != "execution_id"})
    rebound = copied_parent / value["execution_id"]
    (copied / "execution.json").write_bytes(execution._canonical(value) + b"\n")
    copied.rename(rebound)
    with pytest.raises(ValueError, match="inherited|clock|baseline|carry|accounting"):
        execution.validate_execution(rebound, repo_root=REPO_ROOT, config_path=CONFIG_PATH)


@pytest.mark.parametrize(("container", "field", "float_value"), [
    ((), "cuda_seconds", 1228.0),
    ((), "download_bytes", 8066717862.0),
    (("repair",), "cuda_seconds", 1228.0),
    (("repair",), "download_bytes", 8066717862.0),
    (("repair",), "inherited_active_seconds", 60.0),
])
def test_rehashed_schema3_manifest_rejects_equal_valued_float_accounting(
    tmp_path: Path, container: tuple[str, ...], field: str, float_value: float,
) -> None:
    from ppmoe.distill import execution

    root, _manifest = _real_schema3_execution(tmp_path)
    copied_parent = tmp_path / "float-rebound"
    copied_parent.mkdir()
    copied = copied_parent / root.name
    shutil.copytree(root, copied)
    value = json.loads((copied / "execution.json").read_text(encoding="utf-8"))
    target = value
    for key in container:
        target = target[key]
    if field in {"cuda_seconds", "download_bytes"}:
        target = target["inherited_costs"]
    target[field] = float_value
    value["execution_id"] = execution._sha({key: item for key, item in value.items() if key != "execution_id"})
    rebound = copied_parent / value["execution_id"]
    (copied / "execution.json").write_bytes(execution._canonical(value) + b"\n")
    copied.rename(rebound)

    with pytest.raises(ValueError, match="inherited|clock|accounting"):
        execution.validate_execution(rebound, repo_root=REPO_ROOT, config_path=CONFIG_PATH)


@pytest.mark.parametrize("field", ["source", "config", "policy", "ancestor"])
def test_rehashed_schema3_manifest_rejects_changed_source_config_policy_or_ancestor(
    tmp_path: Path, field: str,
) -> None:
    from ppmoe.distill import execution

    root, _manifest = _real_schema3_execution(tmp_path)
    copied_parent = tmp_path / "rebound"
    copied_parent.mkdir()
    copied = copied_parent / root.name
    shutil.copytree(root, copied)
    value = json.loads((copied / "execution.json").read_text(encoding="utf-8"))
    if field == "source":
        value["source_run_id"] = "changed-source"
    elif field == "config":
        value["config_sha256"] = "0" * 64
    elif field == "policy":
        value["policy"]["active_runtime_seconds"] -= 1
    else:
        value["recovery"]["predecessor_execution_id"] = "changed-ancestor"
    value["execution_id"] = execution._sha({key: item for key, item in value.items() if key != "execution_id"})
    rebound = copied_parent / value["execution_id"]
    (copied / "execution.json").write_bytes(execution._canonical(value) + b"\n")
    copied.rename(rebound)
    with pytest.raises(ValueError):
        execution.validate_execution(rebound, repo_root=REPO_ROOT, config_path=CONFIG_PATH)


def test_repair_slot_remains_consumed_after_manifest_creation_failure(tmp_path: Path, monkeypatch) -> None:
    from ppmoe.distill.repair_recovery import PREDECESSOR_ID, register_repair

    previous, executions = _detached_schema2_predecessor(tmp_path)
    original_mkdir = Path.mkdir

    def fail_new_execution_dir(path: Path, *args, **kwargs):
        if path.parent == executions and path.name != PREDECESSOR_ID and len(path.name) == 64:
            raise OSError("synthetic manifest-directory failure")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_new_execution_dir)
    with pytest.raises(OSError, match="manifest-directory"):
        register_repair(previous, executions, repo_root=REPO_ROOT, config_path=CONFIG_PATH)
    slots = list(executions.glob("repair-authorization-*.json"))
    assert len(slots) == 1
    assert slots[0].stat().st_size > 0
    with pytest.raises(FileExistsError, match="consumed"):
        register_repair(previous, executions, repo_root=REPO_ROOT, config_path=CONFIG_PATH)


def test_repair_cli_registers_schema3_without_budget_override_controls(tmp_path: Path) -> None:
    previous, executions = _detached_schema2_predecessor(tmp_path)
    script = REPO_ROOT / "scripts/run_distillation_background.py"
    command = [
        sys.executable, str(script), "register-repair",
        "--previous-execution-root", str(previous), "--executions-root", str(executions),
        "--config", str(CONFIG_PATH), "--repo-root", str(REPO_ROOT),
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    registered = json.loads(result.stdout)
    manifest = json.loads((Path(registered["execution_root"]) / "execution.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    status = json.loads((Path(registered["execution_root"]) / "status.json").read_text(encoding="utf-8"))
    assert status["runtime_used_seconds"] == 60

    override_workspace = tmp_path / "override"
    override_workspace.mkdir()
    other_previous, other_executions = _detached_schema2_predecessor(override_workspace)
    override_command = [
        sys.executable, str(script), "register-repair",
        "--previous-execution-root", str(other_previous), "--executions-root", str(other_executions),
        "--config", str(CONFIG_PATH), "--repo-root", str(REPO_ROOT),
    ]
    overridden = subprocess.run(
        [*override_command, "--cuda-seconds", "1"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert overridden.returncode == 2
    assert not list(other_executions.glob("repair-authorization-*.json"))


def test_schema3_exhausted_worker_path_uses_real_validated_execution_and_cumulative_seed(
    tmp_path: Path, monkeypatch,
) -> None:
    import sys
    import time
    from types import SimpleNamespace
    from ppmoe.distill import execution, study
    from ppmoe.distill.config import load_config
    from ppmoe.distill.execution import ActiveJobContext

    root, manifest = _real_schema3_execution(tmp_path)
    signature = execution.execution_stage_signature(root, "calibrate", repo_root=REPO_ROOT)
    context = ActiveJobContext(
        execution_id=root.name, attempt_id="repair-exhausted-worker", stage="calibrate",
        prior_active_seconds=60, started_monotonic_ns=time.monotonic_ns(),
        supervisor_pid=1, supervisor_created_utc="2026-10-07T00:00:00.000000Z",
        stage_signature=signature,
    )
    monkeypatch.setattr(execution, "load_active_job", lambda *_a, **_k: context)
    monkeypatch.setattr(study, "_execution_job_intervals", lambda *_a, **_k: {})
    monkeypatch.setattr(study, "preflight_disk", lambda *_a, **_k: 20 * 1024**3)
    student_snapshot = REPO_ROOT / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots" / load_config(CONFIG_PATH).student.revision
    def local_names(path: Path) -> list[str]:
        return [
            file.relative_to(path).as_posix()
            for file in path.rglob("*") if file.is_file()
            and file.relative_to(path).parts[0] != ".cache"
        ] if path.is_dir() else []
    names_by_role = {"student": local_names(student_snapshot), "teacher": local_names(teacher_snapshot := REPO_ROOT / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots" / load_config(CONFIG_PATH).teacher.revision)}
    names_by_repo = {load_config(CONFIG_PATH).student.repo_id: names_by_role["student"], load_config(CONFIG_PATH).teacher.repo_id: names_by_role["teacher"]}
    fake_hub = SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=lambda repo_id, *, revision, files_metadata: SimpleNamespace(
            sha=revision, siblings=[SimpleNamespace(rfilename=name, size=0) for name in names_by_repo[repo_id]],
        )),
        snapshot_download=lambda **_kwargs: tmp_path / "fake-student-snapshot",
        hf_hub_download=lambda **_kwargs: pytest.fail("no download is allowed in the pinned exhausted-worker path"),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    original_is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda self: True if self == teacher_snapshot else original_is_dir(self))
    monkeypatch.setattr(study, "authenticate_model_snapshot", lambda repo_id, revision, *_a, **_k: {
        "repo_id": repo_id, "revision": revision, "files": [],
    })
    monkeypatch.setattr(study, "pinned_eos_token_ids", lambda _path: (151643, 151645))
    result = study.run_calibration_stage(
        Path(manifest["source_root"]), load_config(CONFIG_PATH), repo_root=REPO_ROOT,
        config_path=CONFIG_PATH, execution_root=root, active_job=context,
        worker_loader=lambda *_a: (_ for _ in ()).throw(RuntimeError("synthetic exhausted worker load")),
    )

    assert result["status"] == "stopped_environment"
    assert result["resource_ledger"]["cuda_seconds"] >= 1228
    assert result["resource_ledger"]["download_bytes"] == 8066717862
    attempts = [
        json.loads(line)
        for path in (root / "calibration/raw").glob("*.attempts.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    failed_loads = [row for row in attempts if row["status"] == "worker_load_failed"]
    assert len(failed_loads) == 3
    assert not any(row["status"] == "worker_complete" for row in attempts)
    assert all(row["error_message"] == "synthetic exhausted worker load" for row in failed_loads)
    assert all(not path.read_text(encoding="utf-8").strip() for path in (root / "calibration/raw").glob("*.jsonl") if not path.name.endswith(".attempts.jsonl"))
    saved = json.loads((root / "stages/calibrate.json").read_text(encoding="utf-8"))
    assert saved["resource_ledger"]["cuda_seconds"] == result["resource_ledger"]["cuda_seconds"]
    assert saved["resource_ledger"]["download_bytes"] == 8066717862


def test_schema3_normal_calibration_producer_reaches_authenticated_review_handoff(tmp_path: Path, monkeypatch) -> None:
    import sys
    import time
    from types import SimpleNamespace
    from ppmoe.distill import execution, study
    from ppmoe.distill.config import load_config
    from ppmoe.distill.execution import ActiveJobContext

    root, manifest = _real_schema3_execution(tmp_path)
    config = load_config(CONFIG_PATH)
    context = ActiveJobContext(
        execution_id=root.name, attempt_id="repair-normal-producer", stage="calibrate",
        prior_active_seconds=60, started_monotonic_ns=time.monotonic_ns(),
        supervisor_pid=1, supervisor_created_utc="2026-10-07T00:00:00.000000Z",
        stage_signature=execution.execution_stage_signature(root, "calibrate", repo_root=REPO_ROOT),
    )
    monkeypatch.setattr(execution, "load_active_job", lambda *_a, **_k: context)
    monkeypatch.setattr(study, "_execution_job_intervals", lambda *_a, **_k: {})
    monkeypatch.setattr(study, "preflight_disk", lambda *_a, **_k: 20 * 1024**3)
    student_snapshot = REPO_ROOT / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots" / config.student.revision
    teacher_snapshot = REPO_ROOT / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots" / config.teacher.revision
    def local_names(path: Path) -> list[str]:
        return [
            file.relative_to(path).as_posix()
            for file in path.rglob("*") if file.is_file()
            and file.relative_to(path).parts[0] != ".cache"
        ] if path.is_dir() else []
    names_by_repo = {
        config.student.repo_id: local_names(student_snapshot),
        config.teacher.repo_id: local_names(teacher_snapshot),
    }
    monkeypatch.setattr(study, "authenticate_model_snapshot", lambda repo_id, revision, *_a, **_k: {
        "repo_id": repo_id, "revision": revision, "files": [],
    })
    monkeypatch.setattr(study, "pinned_eos_token_ids", lambda _path: (151643, 151645))
    fake_hub = SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=lambda repo_id, *, revision, files_metadata: SimpleNamespace(
            sha=revision, siblings=[SimpleNamespace(rfilename=name, size=0) for name in names_by_repo[repo_id]],
        )),
        snapshot_download=lambda **_kwargs: tmp_path / "fake-student-snapshot",
        hf_hub_download=lambda **_kwargs: pytest.fail("successful local fixture must not download model files"),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    original_is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda self: True if self == teacher_snapshot else original_is_dir(self))

    class FakeWorker:
        startup_wall_seconds = 0.0

        def generate(self, _row, _candidate):
            return {
                "text": "Work. Final answer: 42.", "stop_reason": "eos", "generated_tokens": 4,
                "wall_seconds": 0.01, "cuda_seconds": 0.01,
                "peak_allocated_bytes": 0, "peak_reserved_bytes": 0,
            }

    result = study.run_calibration_stage(
        Path(manifest["source_root"]), config, repo_root=REPO_ROOT,
        config_path=CONFIG_PATH, execution_root=root, active_job=context,
        worker_loader=lambda role, _candidate: FakeWorker(),
    )

    assert result["status"] == "awaiting_review"
    assert result["execution_id"] == manifest["execution_id"]
    assert result["inherited_artifact_bytes"] == (
        2094882 + manifest["artifact_baseline"]["repair_authorization_reservation"]["size_bytes"]
    )
    assert result["resource_ledger"]["cuda_seconds"] >= 1228
    assert result["resource_ledger"]["download_bytes"] == 8066717862
    reservations = [
        json.loads(line)
        for line in (root / "stages/calibrate-reservations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    stage_reservation = next(row for row in reservations if row.get("event_type") == "reservation")
    assert stage_reservation["initial_cuda_seconds"] == 1228
    assert stage_reservation["initial_download_bytes"] == 8066717862
    assert (root / result["review_packet"]).is_file()
    assert json.loads((root / "stages/calibrate.json").read_text(encoding="utf-8"))["status"] == "awaiting_review"
