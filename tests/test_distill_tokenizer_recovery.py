from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from ppmoe.distill import execution
from ppmoe.distill.background import execution_status


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "distillation-v1.toml"
PREDECESSOR_ID = "3116768bd90444328e1ec0e4f8b518a18c238ed74e643d24ae0a69affd0683eb"
PREDECESSOR = REPO_ROOT / "artifacts" / "distillation-v1-executions" / PREDECESSOR_ID


def _copy_predecessor(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    executions = tmp_path / "executions"
    executions.mkdir()
    target = executions / PREDECESSOR_ID
    shutil.copytree(PREDECESSOR, target)
    before = {
        path.relative_to(target).as_posix(): path.read_bytes()
        for path in target.rglob("*") if path.is_file()
    }
    return target, before


def _register(previous: Path) -> Path:
    from ppmoe.distill.tokenizer_recovery import register_tokenizer_recovery

    return register_tokenizer_recovery(
        previous, previous.parent, repo_root=REPO_ROOT, config_path=CONFIG,
    )


def _real_schema4_execution(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    previous, _before = _copy_predecessor(tmp_path)
    root = _register(previous)
    manifest = execution.validate_execution(root, repo_root=REPO_ROOT, config_path=CONFIG)
    return root, manifest


def _active_context(root: Path, *, attempt_id: str):
    from ppmoe.distill.execution import ActiveJobContext

    return ActiveJobContext(
        execution_id=root.name, attempt_id=attempt_id, stage="calibrate",
        prior_active_seconds=91, started_monotonic_ns=time.monotonic_ns(),
        supervisor_pid=1, supervisor_created_utc="2026-10-07T00:00:00.000000Z",
        stage_signature=execution.execution_stage_signature(root, "calibrate", repo_root=REPO_ROOT),
    )


def _status_for_persisted_stage(root: Path, context) -> dict[str, object]:
    """Attach a valid settled lifecycle/result envelope to the real producer artifact."""
    from hashlib import sha256
    from ppmoe.distill.execution import append_event

    stage_path = root / "stages" / "calibrate.json"
    artifact_bytes = stage_path.read_bytes()
    artifact = json.loads(artifact_bytes)
    started = context.started_monotonic_ns
    ended = started + 1_000_000_000
    start = {
        "type": "job_started", "execution_id": root.name, "attempt_id": context.attempt_id,
        "stage": "calibrate", "prior_active_seconds": context.prior_active_seconds,
        "started_monotonic_ns": started, "supervisor_pid": context.supervisor_pid,
        "supervisor_created_utc": context.supervisor_created_utc,
        "stage_pid": context.supervisor_pid, "stage_created_utc": context.supervisor_created_utc,
        "stage_signature": context.stage_signature,
    }
    append_event(root, start)
    append_event(root, {
        key: value for key, value in start.items() if key != "type"
    } | {
        "type": "job_settled", "ended_monotonic_ns": ended,
        "settled_active_seconds": context.prior_active_seconds + 1,
    })
    append_event(root, {
        "type": "attempt_result", "execution_id": root.name,
        "attempt_id": context.attempt_id, "stage": "calibrate",
        "state": artifact["status"], "reason": artifact.get("reason"),
        "artifact_sha256": sha256(artifact_bytes).hexdigest(),
    })
    return execution_status(root, repo_root=REPO_ROOT, config_path=CONFIG)


def _rehash(manifest: dict[str, object]) -> None:
    manifest["execution_id"] = execution._sha({key: value for key, value in manifest.items() if key != "execution_id"})


def test_registers_exact_schema4_without_mutating_or_invalidating_schema3(tmp_path: Path) -> None:
    from ppmoe.distill.execution import artifact_baseline_bytes, validate_execution

    previous, before = _copy_predecessor(tmp_path)
    old_manifest = json.loads((previous / "execution.json").read_text(encoding="utf-8"))
    root = _register(previous)
    manifest = validate_execution(root, repo_root=REPO_ROOT, config_path=CONFIG)

    assert manifest["schema_version"] == 4
    assert manifest["inherited_active_seconds"] == 91
    assert manifest["inherited_costs"] == {"cuda_seconds": 1250, "download_bytes": 8066717862}
    assert manifest["tokenizer_recovery"]["protocol"] == "distillation-tokenizer-contract-recovery-v1"
    assert manifest["tokenizer_recovery"]["predecessor_execution_id"] == PREDECESSOR_ID
    assert manifest["recovery"] == old_manifest["recovery"]
    assert manifest["repair"] == old_manifest["repair"]
    assert manifest["artifact_baseline"]["tokenizer_predecessor_immutable_bytes"] == 53674
    assert manifest["artifact_baseline"]["tokenizer_predecessor_status_reserve_bytes"] == 4736
    reservation = manifest["artifact_baseline"]["tokenizer_authorization_reservation"]
    assert artifact_baseline_bytes(manifest) == (
        old_manifest["artifact_baseline"]["source_bytes"]
        + sum(row["size_bytes"] for row in old_manifest["artifact_baseline"]["feasibility_files"])
        + old_manifest["artifact_baseline"]["predecessor_immutable_bytes"]
        + old_manifest["artifact_baseline"]["predecessor_status_reserve_bytes"]
        + old_manifest["artifact_baseline"]["authorization_reservation"]["size_bytes"]
        + old_manifest["artifact_baseline"]["repair_predecessor_immutable_bytes"]
        + old_manifest["artifact_baseline"]["repair_predecessor_status_reserve_bytes"]
        + old_manifest["artifact_baseline"]["repair_authorization_reservation"]["size_bytes"]
        + 53674 + 4736 + reservation["size_bytes"]
    )
    assert execution.validate_execution(previous, repo_root=REPO_ROOT, config_path=CONFIG)["schema_version"] == 3
    assert {path.relative_to(previous).as_posix(): path.read_bytes() for path in previous.rglob("*") if path.is_file()} == before
    assert execution_status(root, repo_root=REPO_ROOT, config_path=CONFIG)["runtime_used_seconds"] == 91


def test_registration_rejects_rehashed_unknown_top_level_or_context_keys(tmp_path: Path) -> None:
    previous, _before = _copy_predecessor(tmp_path)
    root = _register(previous)
    manifest = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    manifest["caller_extension"] = "not part of schema 4"
    _rehash(manifest)
    with pytest.raises(ValueError, match="schema|key|field"):
        from ppmoe.distill.tokenizer_recovery import validate_tokenizer_recovery_manifest

        validate_tokenizer_recovery_manifest(manifest, repo_root=REPO_ROOT, config_path=CONFIG)
    manifest = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    manifest["tokenizer_recovery"]["caller_extension"] = "not part of schema 4"
    _rehash(manifest)
    with pytest.raises(ValueError, match="schema|key|field"):
        validate_tokenizer_recovery_manifest(manifest, repo_root=REPO_ROOT, config_path=CONFIG)


@pytest.mark.parametrize("mutation", [
    "top_cost_float", "context_cost_bool", "context_clock_float", "context_status_float",
    "pinned_size_float", "passthrough_policy_float", "unknown_baseline",
])
def test_manifest_rejects_rehashed_cost_or_baseline_mutations(tmp_path: Path, mutation: str) -> None:
    previous, _before = _copy_predecessor(tmp_path)
    root = _register(previous)
    manifest = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    if mutation == "top_cost_float":
        manifest["inherited_costs"]["cuda_seconds"] = 1250.0
    elif mutation == "context_cost_bool":
        manifest["tokenizer_recovery"]["inherited_costs"]["download_bytes"] = True
    elif mutation == "context_clock_float":
        manifest["tokenizer_recovery"]["inherited_active_seconds"] = 91.0
    elif mutation == "context_status_float":
        manifest["tokenizer_recovery"]["predecessor_status_reserve_bytes"] = 4736.0
    elif mutation == "pinned_size_float":
        manifest["tokenizer_recovery"]["pinned_files"]["calibration/model-inventory/student.json"]["size_bytes"] = 2790.0
    elif mutation == "passthrough_policy_float":
        manifest["policy"]["active_runtime_seconds"] = 172800.0
    else:
        manifest["artifact_baseline"]["extension"] = "extra baseline field"
    _rehash(manifest)
    with pytest.raises(ValueError, match="cost|clock|baseline|field|context|exact"):
        from ppmoe.distill.tokenizer_recovery import validate_tokenizer_recovery_manifest

        validate_tokenizer_recovery_manifest(manifest, repo_root=REPO_ROOT, config_path=CONFIG)

@pytest.mark.parametrize("field,bad_value", [
    ("inherited_active_seconds", True),
    ("inherited_active_seconds", 90),
])
def test_rehashed_manifest_rejects_clock_changes(tmp_path: Path, field: str, bad_value: object) -> None:
    previous, _before = _copy_predecessor(tmp_path)
    root = _register(previous)
    manifest = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    manifest[field] = bad_value
    _rehash(manifest)
    with pytest.raises(ValueError, match="clock|inherited|schema"):
        from ppmoe.distill.tokenizer_recovery import validate_tokenizer_recovery_manifest

        validate_tokenizer_recovery_manifest(manifest, repo_root=REPO_ROOT, config_path=CONFIG)


def test_predecessor_pin_mutation_does_not_consume_authorization_slot(tmp_path: Path) -> None:
    previous, _before = _copy_predecessor(tmp_path)
    with (previous / "events.jsonl").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError):
        _register(previous)
    assert not list(previous.parent.glob("tokenizer-authorization-*.json"))


def test_duplicate_registration_consumes_one_slot(tmp_path: Path) -> None:
    previous, _before = _copy_predecessor(tmp_path)
    root = _register(previous)
    with pytest.raises((FileExistsError, ValueError), match="slot|authorization|exist|registered"):
        _register(previous)
    assert root.is_dir()
    assert len(list(previous.parent.glob("tokenizer-authorization-*.json"))) == 1


def test_partial_preexisting_slot_is_not_reused(tmp_path: Path) -> None:
    previous, _before = _copy_predecessor(tmp_path)
    partial = previous.parent / "tokenizer-authorization-partial.json"
    partial.write_bytes(b"partial slot")
    with pytest.raises(FileExistsError, match="slot|exists"):
        _register(previous)
    assert partial.read_bytes() == b"partial slot"
    assert not list(previous.parent.glob(f"tokenizer-authorization-{PREDECESSOR_ID}.json"))


def test_tampered_authorization_reservation_invalidates_schema4(tmp_path: Path) -> None:
    from ppmoe.distill.execution import validate_execution

    previous, _before = _copy_predecessor(tmp_path)
    root = _register(previous)
    slot = previous.parent / f"tokenizer-authorization-{PREDECESSOR_ID}.json"
    slot.write_bytes(slot.read_bytes().replace(b"distillation-tokenizer-contract-recovery-v1", b"changed-tokenizer-protocol"))
    with pytest.raises(ValueError, match="authorization|reservation|path|hash"):
        validate_execution(root, repo_root=REPO_ROOT, config_path=CONFIG)


def test_schema4_rejects_extra_tokenizer_slot_without_invalidating_old_namespaces(tmp_path: Path) -> None:
    from ppmoe.distill.execution import validate_execution

    previous, _before = _copy_predecessor(tmp_path)
    root = _register(previous)
    (previous.parent / "tokenizer-authorization-extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the fixed predecessor slot"):
        validate_execution(root, repo_root=REPO_ROOT, config_path=CONFIG)
    assert validate_execution(previous, repo_root=REPO_ROOT, config_path=CONFIG)["schema_version"] == 3


def test_stage_signature_registries_bind_new_recovery_and_task1_integration() -> None:
    from ppmoe.distill.study import CALIBRATION_FILES, EXECUTION_COMMON_FILES

    assert "tests/test_distill_tokenizer_integration.py" in CALIBRATION_FILES
    assert "src/ppmoe/distill/tokenizer_recovery.py" in EXECUTION_COMMON_FILES
    assert "tests/test_distill_tokenizer_recovery.py" in EXECUTION_COMMON_FILES


def test_concurrent_registration_creates_one_child(tmp_path: Path) -> None:
    previous, _before = _copy_predecessor(tmp_path)

    def attempt() -> Path | None:
        try:
            return _register(previous)
        except (FileExistsError, ValueError):
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _index: attempt(), range(2)))
    assert sum(value is not None for value in outcomes) == 1
    assert len(list(previous.parent.glob("tokenizer-authorization-*.json"))) == 1
    assert len([path for path in previous.parent.iterdir() if path.is_dir()]) == 2


def test_schema4_initial_clock_and_artifact_accounting_are_exact() -> None:
    from ppmoe.distill.execution import artifact_baseline_bytes, initial_active_seconds

    manifest = {
        "schema_version": 4,
        "inherited_active_seconds": 91,
        "inherited_costs": {"cuda_seconds": 1250, "download_bytes": 8066717862},
        "tokenizer_recovery": {"inherited_active_seconds": 91},
        "artifact_baseline": {
            "source_bytes": 9, "feasibility_files": [],
            "predecessor_immutable_bytes": 10, "predecessor_status_reserve_bytes": 11,
            "authorization_reservation": {"size_bytes": 12},
            "repair_predecessor_immutable_bytes": 13, "repair_predecessor_status_reserve_bytes": 14,
            "repair_authorization_reservation": {"size_bytes": 15},
            "tokenizer_predecessor_immutable_bytes": 16, "tokenizer_predecessor_status_reserve_bytes": 17,
            "tokenizer_authorization_reservation": {"size_bytes": 18},
        },
    }
    assert initial_active_seconds(manifest) == 91
    assert artifact_baseline_bytes(manifest) == 9 + 10 + 11 + 12 + 13 + 14 + 15 + 16 + 17 + 18
    with pytest.raises(ValueError, match="clock"):
        initial_active_seconds({**manifest, "inherited_active_seconds": True})


def test_schema4_pre_model_stop_uses_real_manifest_reservation_and_full_baseline(tmp_path: Path, monkeypatch) -> None:
    from ppmoe.distill import study
    from ppmoe.distill.config import load_config
    from ppmoe.distill.evaluate import ResourceLimitError

    root, manifest = _real_schema4_execution(tmp_path)
    context = _active_context(root, attempt_id="tokenizer-pre-model")
    monkeypatch.setattr(execution, "load_active_job", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(study, "_execution_job_intervals", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        study, "preflight_disk",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ResourceLimitError("fixture pre-model resource stop")),
    )
    result = study.run_calibration_stage(
        Path(manifest["source_root"]), load_config(CONFIG), repo_root=REPO_ROOT,
        config_path=CONFIG, execution_root=root, active_job=context,
        worker_loader=lambda *_args: pytest.fail("pre-model stop must not construct a worker"),
    )

    assert result["status"] == "stopped_environment"
    assert result["resource_ledger"]["cuda_seconds"] == 1250
    assert result["resource_ledger"]["download_bytes"] == 8066717862
    assert result["inherited_artifact_bytes"] == 2153595 + manifest["artifact_baseline"]["tokenizer_authorization_reservation"]["size_bytes"]
    reservations = [json.loads(line) for line in (root / "stages/calibrate-reservations.jsonl").read_text(encoding="utf-8").splitlines()]
    initial = next(row for row in reservations if row.get("event_type") == "reservation")
    assert initial["initial_cuda_seconds"] == 1250
    assert initial["initial_download_bytes"] == 8066717862
    saved = json.loads((root / "stages/calibrate.json").read_text(encoding="utf-8"))
    assert saved["resource_ledger"]["cuda_seconds"] == 1250
    assert saved["resource_ledger"]["download_bytes"] == 8066717862


def _patch_local_model_boundaries(monkeypatch, tmp_path: Path, config) -> None:
    from types import SimpleNamespace
    from ppmoe.distill import study

    student_snapshot = REPO_ROOT / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots" / config.student.revision
    teacher_snapshot = REPO_ROOT / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots" / config.teacher.revision

    def local_names(path: Path) -> list[str]:
        return [
            file.relative_to(path).as_posix()
            for file in path.rglob("*") if file.is_file() and file.relative_to(path).parts[0] != ".cache"
        ] if path.is_dir() else []

    names_by_repo = {
        config.student.repo_id: local_names(student_snapshot),
        config.teacher.repo_id: local_names(teacher_snapshot),
    }
    monkeypatch.setattr(study, "authenticate_model_snapshot", lambda repo_id, revision, *_args, **_kwargs: {
        "repo_id": repo_id, "revision": revision, "files": [],
    })
    monkeypatch.setattr(study, "pinned_eos_token_ids", lambda _path: (151643, 151645))
    fake_hub = SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=lambda repo_id, *, revision, files_metadata: SimpleNamespace(
            sha=revision, siblings=[SimpleNamespace(rfilename=name, size=0) for name in names_by_repo[repo_id]],
        )),
        snapshot_download=lambda **_kwargs: tmp_path / "fake-student-snapshot",
        hf_hub_download=lambda **_kwargs: pytest.fail("tokenizer recovery CPU fixture must not download models"),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    original_is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda self: True if self == teacher_snapshot else original_is_dir(self))


def test_schema4_exhausted_worker_path_persists_cumulative_costs(tmp_path: Path, monkeypatch) -> None:
    from ppmoe.distill import study
    from ppmoe.distill.config import load_config

    root, manifest = _real_schema4_execution(tmp_path)
    config = load_config(CONFIG)
    context = _active_context(root, attempt_id="tokenizer-exhausted-worker")
    monkeypatch.setattr(execution, "load_active_job", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(study, "_execution_job_intervals", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(study, "preflight_disk", lambda *_args, **_kwargs: 20 * 1024**3)
    _patch_local_model_boundaries(monkeypatch, tmp_path, config)
    result = study.run_calibration_stage(
        Path(manifest["source_root"]), config, repo_root=REPO_ROOT,
        config_path=CONFIG, execution_root=root, active_job=context,
        worker_loader=lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic exhausted worker load")),
    )

    assert result["status"] == "stopped_environment"
    assert result["resource_ledger"]["cuda_seconds"] == 1253
    assert result["resource_ledger"]["download_bytes"] == 8066717862
    attempts = [json.loads(line) for path in (root / "calibration/raw").glob("*.attempts.jsonl") for line in path.read_text(encoding="utf-8").splitlines()]
    failed = [row for row in attempts if row.get("status") == "worker_load_failed"]
    assert len(failed) == 3
    assert all(row.get("error_message") == "synthetic exhausted worker load" for row in failed)
    reservations = [json.loads(line) for line in (root / "stages/calibrate-reservations.jsonl").read_text(encoding="utf-8").splitlines()]
    initial = next(row for row in reservations if row.get("event_type") == "reservation")
    assert initial["initial_cuda_seconds"] == 1250
    assert initial["initial_download_bytes"] == 8066717862
    saved = json.loads((root / "stages/calibrate.json").read_text(encoding="utf-8"))
    assert saved["resource_ledger"]["cuda_seconds"] == 1253
    assert saved["resource_ledger"]["download_bytes"] == 8066717862
    status = _status_for_persisted_stage(root, context)
    assert status["state"] == "stopped_environment"
    assert status["costs"]["complete"] is True
    assert status["costs"]["cuda_seconds"] == {"total": 1253, "complete": True}
    assert status["costs"]["download_bytes"] == {"total": 8066717862, "complete": True}
    assert status["artifact_usage"]["inherited_artifact_baseline_bytes"] == result["inherited_artifact_bytes"]


def test_schema4_normal_producer_reaches_review_handoff_with_one_cumulative_baseline(tmp_path: Path, monkeypatch) -> None:
    from ppmoe.distill import study
    from ppmoe.distill.config import load_config

    root, manifest = _real_schema4_execution(tmp_path)
    config = load_config(CONFIG)
    contexts = [_active_context(root, attempt_id="tokenizer-normal-producer")]
    monkeypatch.setattr(execution, "load_active_job", lambda *_args, **_kwargs: contexts[0])
    monkeypatch.setattr(study, "_execution_job_intervals", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(study, "preflight_disk", lambda *_args, **_kwargs: 20 * 1024**3)
    _patch_local_model_boundaries(monkeypatch, tmp_path, config)

    class FakeWorker:
        startup_wall_seconds = 0.0
        generated = 0

        def generate(self, _row, _candidate):
            FakeWorker.generated += 1
            return {
                "text": "Work. Final answer: 42.", "stop_reason": "eos", "generated_tokens": 4,
                "wall_seconds": 0.01, "cuda_seconds": 0.01,
                "peak_allocated_bytes": 0, "peak_reserved_bytes": 0,
            }

    result = study.run_calibration_stage(
        Path(manifest["source_root"]), config, repo_root=REPO_ROOT,
        config_path=CONFIG, execution_root=root, active_job=contexts[0],
        worker_loader=lambda _role, _candidate: FakeWorker(),
    )
    assert result["status"] == "awaiting_review"
    assert result["inherited_artifact_bytes"] == 2153595 + manifest["artifact_baseline"]["tokenizer_authorization_reservation"]["size_bytes"]
    assert result["resource_ledger"]["cuda_seconds"] >= 1250
    assert FakeWorker.generated > 0
    assert result["resource_ledger"]["cuda_seconds"] == 1250 + FakeWorker.generated
    assert result["resource_ledger"]["download_bytes"] == 8066717862
    reservations = [json.loads(line) for line in (root / "stages/calibrate-reservations.jsonl").read_text(encoding="utf-8").splitlines()]
    initial = next(row for row in reservations if row.get("event_type") == "reservation")
    assert initial["initial_cuda_seconds"] == 1250
    assert initial["initial_download_bytes"] == 8066717862
    saved = json.loads((root / "stages/calibrate.json").read_text(encoding="utf-8"))
    assert saved["status"] == "awaiting_review"
    assert saved["resource_ledger"]["cuda_seconds"] == result["resource_ledger"]["cuda_seconds"]
    assert saved["resource_ledger"]["download_bytes"] == 8066717862
    assert (root / result["review_packet"]).is_file()
    status = _status_for_persisted_stage(root, contexts[0])
    assert status["state"] == "awaiting_review"
    assert status["costs"]["complete"] is True
    assert status["costs"]["cuda_seconds"] == {"total": 1250 + FakeWorker.generated, "complete": True}
    assert status["costs"]["download_bytes"] == {"total": 8066717862, "complete": True}
    assert status["artifact_usage"]["inherited_artifact_baseline_bytes"] == result["inherited_artifact_bytes"]

    generated_before_resume = FakeWorker.generated
    (root / "stages/calibrate.json").unlink()
    contexts[0] = _active_context(root, attempt_id="tokenizer-normal-resume")
    resumed = study.run_calibration_stage(
        Path(manifest["source_root"]), config, repo_root=REPO_ROOT,
        config_path=CONFIG, execution_root=root, active_job=contexts[0],
        worker_loader=lambda _role, _candidate: FakeWorker(),
    )
    assert resumed["status"] == "awaiting_review"
    assert resumed["resource_ledger"]["cuda_seconds"] == result["resource_ledger"]["cuda_seconds"]
    assert resumed["resource_ledger"]["download_bytes"] == 8066717862
    assert resumed["inherited_artifact_bytes"] == result["inherited_artifact_bytes"]
    assert FakeWorker.generated == generated_before_resume


def test_tokenizer_recovery_cli_has_only_exact_registration_arguments(tmp_path: Path) -> None:
    import importlib.util

    previous, _before = _copy_predecessor(tmp_path)
    spec = importlib.util.spec_from_file_location("distillation_background_cli", REPO_ROOT / "scripts" / "run_distillation_background.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = [
        "register-tokenizer-recovery", "--previous-execution-root", str(previous),
        "--executions-root", str(previous.parent), "--config", str(CONFIG), "--repo-root", str(REPO_ROOT),
    ]
    assert module.main(argv) == 0
    created = json.loads((previous.parent / f"tokenizer-authorization-{PREDECESSOR_ID}.json").read_text(encoding="utf-8"))
    assert created["protocol"] == "distillation-tokenizer-contract-recovery-v1"
    with pytest.raises(SystemExit):
        module.main(argv + ["--cuda-seconds", "1"])


@pytest.mark.skipif(os.name != "nt", reason="requires detached Windows supervisor process")
def test_tokenizer_cpu_fixture_uses_shared_terminal_and_exits_at_91_second_baseline(tmp_path: Path) -> None:
    workspace = tmp_path / "tokenizer cpu fixture path with spaces"
    fixture_script = Path(__file__).with_name("distill_background_cpu_fixture.py")
    prepared = subprocess.run(
        [sys.executable, str(fixture_script), "prepare-tokenizer-recovery", "--workspace", str(workspace)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert prepared.returncode == 0, prepared.stderr
    fixture = json.loads(prepared.stdout)
    root = Path(fixture["execution_root"])
    manifest = json.loads((root / "execution.json").read_text(encoding="utf-8"))
    initial = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert fixture["schema_version"] == 4
    assert initial["runtime_used_seconds"] == 91
    assert initial["runtime_remaining_seconds"] == 172709
    projected = subprocess.run(
        [sys.executable, str(fixture_script), "status", "--workspace", str(workspace)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert projected.returncode == 0, projected.stderr
    assert json.loads(projected.stdout)["artifact_usage"]["inherited_artifact_baseline_bytes"] == (
        2153595 + manifest["artifact_baseline"]["tokenizer_authorization_reservation"]["size_bytes"]
    )
    launched = subprocess.run(
        [sys.executable, str(fixture_script), "start", "--workspace", str(workspace)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=20,
    )
    assert launched.returncode == 0, launched.stderr
    launch = json.loads(launched.stdout)
    stdout_path = Path(launch["stdout_path"])
    deadline = time.monotonic() + 12
    status: dict[str, object] = {}
    while time.monotonic() < deadline:
        current = subprocess.run(
            [sys.executable, str(fixture_script), "status", "--workspace", str(workspace)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=20,
        )
        assert current.returncode == 0, current.stderr
        status = json.loads(current.stdout)
        if status["state"] == "awaiting_review":
            break
        time.sleep(0.2)
    assert status["state"] == "awaiting_review"
    stage = json.loads((root / "stages/calibrate.json").read_text(encoding="utf-8"))
    assert stage["resource_ledger"]["cuda_seconds"] == 1250
    assert stage["resource_ledger"]["download_bytes"] == 8066717862
    events = execution.read_events(root)
    payloads = [event["payload"] for event in events]
    started = next(item for item in payloads if item.get("type") == "job_started")
    settled = next(item for item in payloads if item.get("type") == "job_settled")
    assert started["prior_active_seconds"] == 91
    assert settled["settled_active_seconds"] >= 91
    assert "finished" in stdout_path.read_text(encoding="utf-8")
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline and any(
        execution.process_creation_utc(pid) == created for pid, created in (
            (started["stage_pid"], started["stage_created_utc"]),
            (started["supervisor_pid"], started["supervisor_created_utc"]),
        )
    ):
        time.sleep(0.1)
    assert all(
        execution.process_creation_utc(pid) != created for pid, created in (
            (started["stage_pid"], started["stage_created_utc"]),
            (started["supervisor_pid"], started["supervisor_created_utc"]),
        )
    )
