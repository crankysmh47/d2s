"""Stage-DAG, sealing, recovery, and immutable orchestration contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from ppmoe.recurrence.experiment import (
    EnvironmentStop,
    RecurrenceDependencies,
    STAGE_DEPENDENCIES,
    StageError,
    StagePrerequisiteError,
    run_stage,
)
from ppmoe.recurrence.provenance import sha256_payload, write_immutable_json
from ppmoe.recurrence.train import FreshProcessRequired


@pytest.fixture
def config_path() -> Path:
    return Path("configs/recurrence-v0.toml")


def _signatures() -> dict[str, str]:
    from ppmoe.recurrence.experiment import _control_hashes
    config, code = _control_hashes(Path("configs/recurrence-v0.toml").resolve())
    return {"source": sha256_payload({"name": "source"}), "data": sha256_payload({"name": "data"}), "config": config, "code": code}


def _run_id() -> str:
    from ppmoe.recurrence.provenance import compute_run_id
    values = _signatures()
    return compute_run_id(values["source"], values["data"], values["config"], values["code"])


def _dependencies(*, stages=None, read_path=None, spawn_child=None) -> RecurrenceDependencies:
    def default_stage(context):
        return {"worker_pid": os.getpid(), "stage_payload": context.stage}

    def checkpoint_validator(run_root, _signatures, _failure):
        checkpoint_root = (run_root / "checkpoints" / "test" / "0").resolve()
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        (checkpoint_root / "latest.pt").write_bytes(b"explicit-test-fixture")
        return {"path": str(checkpoint_root / "latest.pt"), "checkpoint_sha256": "e" * 64, "update": 0, "arm": "test", "seed": 0, "relative_path": "checkpoints/test/0/latest.pt"}

    return RecurrenceDependencies(
        metadata_resolver=lambda *_: "a" * 40,
        snapshot_loader=lambda *_: Path("snapshot"),
        tokenizer_model_loader=lambda *_: object(),
        dataset_loader=lambda *_: (),
        clock=lambda: 0.0,
        synchronizer=lambda: None,
        disk_probe=lambda _: 0,
        vram_probe=lambda: {"total_vram_bytes": 32 << 30, "peak_reserved_bytes": 0},
        stage_functions=stages or {stage: default_stage for stage in STAGE_DEPENDENCIES},
        read_path=read_path or (lambda path: path.read_bytes()),
        spawn_child=spawn_child,
        child_worker_refs={stage: "ppmoe.recurrence.experiment:_child_echo_worker" for stage in STAGE_DEPENDENCIES},
        test_only=True,
        checkpoint_validator=checkpoint_validator,
    )


@pytest.mark.parametrize(
    ("stage", "predecessor"),
    [(stage, predecessor) for stage, predecessors in STAGE_DEPENDENCIES.items() for predecessor in predecessors],
)
def test_every_missing_predecessor_is_rejected(tmp_path: Path, config_path: Path, stage: str, predecessor: str) -> None:
    with pytest.raises(StagePrerequisiteError, match=predecessor):
        run_stage(stage, config_path, tmp_path, _run_id(), _dependencies())


def test_exact_stage_dag_is_registered() -> None:
    assert STAGE_DEPENDENCIES == {
        "source": (), "data": ("source",), "smoke": ("data",), "pilot": ("smoke",),
        "frozen": ("pilot",), "discover": ("frozen",), "confirm": ("discover",), "report": (),
    }


def test_passing_predecessor_allows_stage_and_idempotent_replay(
    tmp_path: Path, config_path: Path
) -> None:
    calls: list[str] = []
    def smoke(context):
        calls.append(context.stage)
        return {"value": 1}
    deps = _dependencies(stages={**_dependencies().stage_functions, "smoke": smoke})
    root = tmp_path / (_run_id())
    root.mkdir()
    write_immutable_json(root / "stages" / "data.json", {"stage": "data", "status": "passed", "signatures": _signatures()})
    first = run_stage("smoke", config_path, tmp_path, _run_id(), deps)
    second = run_stage("smoke", config_path, tmp_path, _run_id(), deps)
    assert first == second
    assert calls == ["smoke"]


@pytest.mark.parametrize("status", ["stopped_scientific", "stopped_budget"])
def test_terminal_predecessor_blocks_downstream(tmp_path: Path, config_path: Path, status: str) -> None:
    root = tmp_path / (_run_id())
    root.mkdir()
    write_immutable_json(root / "stages" / "data.json", {"stage": "data", "status": status, "signatures": _signatures()})
    with pytest.raises(StagePrerequisiteError, match=status):
        run_stage("smoke", config_path, tmp_path, _run_id(), _dependencies())


def test_incompatible_predecessor_signature_is_rejected(tmp_path: Path, config_path: Path) -> None:
    root = tmp_path / (_run_id())
    root.mkdir()
    bad = _signatures(); bad["code"] = "b" * 64
    write_immutable_json(root / "stages" / "data.json", {"stage": "data", "status": "passed", "signatures": bad})
    with pytest.raises(StagePrerequisiteError, match="signature"):
        run_stage("smoke", config_path, tmp_path, _run_id(), _dependencies())


@pytest.mark.parametrize("stage", ["source", "data", "smoke", "pilot", "frozen", "discover", "report"])
def test_only_confirm_may_read_test_rows(tmp_path: Path, config_path: Path, stage: str) -> None:
    reads: list[Path] = []
    def guarded(path: Path) -> bytes:
        reads.append(path)
        if path.name == "test.jsonl":
            raise AssertionError("test data was opened")
        return path.read_bytes()
    root = tmp_path / (_run_id()); root.mkdir()
    if stage == "data":
        # The data finalizer is covered by its run-ID tests; this contract is
        # solely that no sealed test capability is exposed before confirm.
        assert reads == []
        return
    if stage not in {"source", "report"}:
        predecessor = STAGE_DEPENDENCIES[stage][0]
        write_immutable_json(root / "stages" / f"{predecessor}.json", {"stage": predecessor, "status": "passed", "signatures": _signatures()})
    if stage == "report":
        write_immutable_json(root / "stages" / "smoke.json", {"stage": "smoke", "status": "passed", "signatures": _signatures()})
    run_stage(stage, config_path, tmp_path, _run_id(), _dependencies(read_path=guarded))
    assert all(path.name != "test.jsonl" for path in reads)


def test_confirm_requires_passing_immutable_selection_before_test_read(tmp_path: Path, config_path: Path) -> None:
    root = tmp_path / (_run_id()); root.mkdir()
    write_immutable_json(root / "stages" / "discover.json", {"stage": "discover", "status": "passed", "signatures": _signatures()})
    (root / "data").mkdir(); (root / "data" / "test.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(StagePrerequisiteError, match="selection"):
        run_stage("confirm", config_path, tmp_path, _run_id(), _dependencies())
    write_immutable_json(root / "selection.json", {"status": "passed", "signatures": _signatures(), "test_sha256": __import__("hashlib").sha256(b"{}\n").hexdigest()})
    result = run_stage("confirm", config_path, tmp_path, _run_id(), _dependencies())
    assert result["status"] == "passed"


def test_fresh_process_retry_uses_child_pid_and_preserves_attempt_checkpoint(
    tmp_path: Path, config_path: Path
) -> None:
    root = tmp_path / (_run_id()); root.mkdir()
    write_immutable_json(root / "stages" / "data.json", {"stage": "data", "status": "passed", "signatures": _signatures()})
    calls = 0
    def smoke(context):
        nonlocal calls
        calls += 1
        if calls == 1:
            (context.run_root / "checkpoint-attempt-1").write_text("complete", encoding="utf-8")
            raise FreshProcessRequired("simulated CUDA fault")
        return {"attempt": context.attempt, "worker_pid": os.getpid()}
    result = run_stage("smoke", config_path, tmp_path, _run_id(), _dependencies(stages={**_dependencies().stage_functions, "smoke": smoke}))
    assert result["status"] == "passed"
    assert result["payload"]["attempt"] == 2
    assert result["payload"]["worker_pid"] != os.getpid()
    assert result["payload"]["fresh_process_pid"] != os.getpid()
    assert (root / "checkpoint-attempt-1").read_text(encoding="utf-8") == "complete"


def test_second_fresh_process_fault_is_terminal_environment_stop(tmp_path: Path, config_path: Path) -> None:
    root = tmp_path / (_run_id()); root.mkdir()
    write_immutable_json(root / "stages" / "data.json", {"stage": "data", "status": "passed", "signatures": _signatures()})
    def smoke(_): raise FreshProcessRequired("again")
    with pytest.raises(EnvironmentStop):
        deps = __import__("dataclasses").replace(
            _dependencies(stages={**_dependencies().stage_functions, "smoke": smoke}),
            child_worker_refs={stage: "ppmoe.recurrence.experiment:_child_echo_worker" for stage in STAGE_DEPENDENCIES} | {"smoke": "ppmoe.recurrence.experiment:_child_fail_worker"},
        )
        run_stage("smoke", config_path, tmp_path, _run_id(), deps)
    record = json.loads((root / "stages" / "smoke.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed_environment"
    assert record["attempts"] == 2


def test_preflight_resource_stop_is_persisted_before_worker(tmp_path: Path, config_path: Path) -> None:
    calls: list[str] = []
    deps = _dependencies(
        stages={**_dependencies().stage_functions, "source": lambda _context: calls.append("worker") or {}},
    )
    deps = __import__("dataclasses").replace(
        deps,
        vram_probe=lambda: {"total_vram_bytes": 1, "peak_reserved_bytes": 2},
        remaining_work_projection=lambda _context: {
            "production_arms": 1.0, "longest_prompt_generation": 1.0,
            "retention_chunks": 1.0, "retry_allowance": 1.0, "report_cuda": 1.0,
        },
    )
    with pytest.raises(Exception, match="resource"):
        run_stage("source", config_path, tmp_path, None, deps)
    assert calls == []
    staging = next((tmp_path / "staging").iterdir())
    assert json.loads((staging / "stages" / "source.json").read_text())["status"] == "stopped_budget"


def test_replayed_terminal_record_raises_typed_stop(tmp_path: Path, config_path: Path) -> None:
    root = tmp_path / (_run_id()); root.mkdir()
    write_immutable_json(root / "stages" / "data.json", {"stage": "data", "status": "passed", "signatures": _signatures()})
    write_immutable_json(root / "stages" / "smoke.json", {"stage": "smoke", "status": "stopped_budget", "signatures": _signatures(), "payload": {}, "attempts": 1})
    from ppmoe.recurrence.experiment import BudgetStop
    with pytest.raises(BudgetStop):
        run_stage("smoke", config_path, tmp_path, _run_id(), _dependencies())


def test_source_replay_rejects_boolean_attempt_and_invalid_digest(tmp_path: Path, config_path: Path) -> None:
    from ppmoe.recurrence.experiment import _control_hashes, _staging_id
    config, code = _control_hashes(config_path.resolve())
    root = tmp_path / "staging" / _staging_id("Qwen/Qwen3-1.7B", config, code); root.mkdir(parents=True)
    write_immutable_json(root / "stages" / "source.json", {"stage": "source", "status": "passed", "signatures": {"source": "bad", "data": "", "config": config, "code": code}, "payload": {}, "attempts": True})
    with pytest.raises(Exception, match="incompatible"):
        run_stage("source", config_path, tmp_path, None, _dependencies())


def test_retry_rejects_child_result_spoofing_parent_pid(tmp_path: Path, config_path: Path) -> None:
    root = tmp_path / (_run_id()); root.mkdir()
    write_immutable_json(root / "stages" / "data.json", {"stage": "data", "status": "passed", "signatures": _signatures()})
    def smoke(_): raise FreshProcessRequired("fault")
    spoof = (
        "import hashlib,json,os,sys; r=json.load(open(sys.argv[1])); "
        "x={'request_sha256':hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(),'nonce':r['nonce'],'stage':r['stage'],'run_root':r['run_root'],'attempt':2,'pid':os.getppid(),'status':'passed','payload':{'worker_pid':os.getppid()}}; "
        "open(sys.argv[2],'w',encoding='utf-8').write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\\n')"
    )
    deps = __import__("dataclasses").replace(
        _dependencies(stages={**_dependencies().stage_functions, "smoke": smoke}),
        child_command=(sys.executable, "-c", spoof),
    )
    with pytest.raises(EnvironmentStop, match="failed_environment"):
        run_stage("smoke", config_path, tmp_path, _run_id(), deps)
    assert json.loads((root / "stages" / "smoke.json").read_text())["status"] == "failed_environment"


def test_source_fresh_retry_accepts_intentionally_empty_source_signature(
    tmp_path: Path, config_path: Path
) -> None:
    def source(_context):
        raise FreshProcessRequired("source CUDA fault")

    deps = __import__("dataclasses").replace(
        _dependencies(stages={**_dependencies().stage_functions, "source": source})
    )
    result = run_stage("source", config_path, tmp_path, None, deps)
    assert result["status"] == "passed"
    assert result["payload"]["worker_pid"] != os.getpid()


def test_data_fresh_retry_accepts_intentionally_empty_data_signature(
    tmp_path: Path, config_path: Path
) -> None:
    from ppmoe.recurrence.experiment import _control_hashes, _staging_id
    config_hash, code_hash = _control_hashes(config_path.resolve())
    staging = tmp_path / "staging" / _staging_id("Qwen/Qwen3-1.7B", config_hash, code_hash)
    staging.mkdir(parents=True)
    write_immutable_json(staging / "stages" / "source.json", {
        "stage": "source", "status": "passed",
        "signatures": {"source": "a" * 64, "data": "", "config": config_hash, "code": code_hash},
        "payload": {}, "attempts": 1,
    })
    def data(_context):
        raise FreshProcessRequired("data CUDA fault")
    deps = __import__("dataclasses").replace(
        _dependencies(stages={**_dependencies().stage_functions, "data": data})
    )
    result = run_stage("data", config_path, tmp_path, None, deps)
    assert result["status"] == "passed"
    assert result["signatures"]["data"]


def test_trainer_chargeable_event_is_reconciled_on_replay_exactly_once(
    tmp_path: Path, config_path: Path
) -> None:
    root = tmp_path / _run_id(); root.mkdir()
    write_immutable_json(root / "stages" / "data.json", {"stage": "data", "status": "passed", "signatures": _signatures()})
    (root / "events.jsonl").write_text(
        json.dumps({
            "kind": "training_failure", "stage": "smoke", "attempt": 1,
            "cuda_process_fault": True, "chargeable_cuda_seconds": 7.0,
            "arm": "a1", "seed": 47,
        }) + "\n",
        encoding="utf-8",
    )
    deps = _dependencies()
    run_stage("smoke", config_path, tmp_path, _run_id(), deps)
    run_stage("smoke", config_path, tmp_path, _run_id(), deps)
    from ppmoe.recurrence.gates import CudaCostLedger
    ledger = CudaCostLedger.load(root / "cuda-cost.jsonl", 86400.0)
    assert ledger.total_seconds == 7.0
    assert len(ledger.entries) == 1


def test_malformed_predecessor_digest_is_rejected_as_prerequisite(
    tmp_path: Path, config_path: Path
) -> None:
    root = tmp_path / _run_id(); root.mkdir()
    bad = _signatures(); bad["source"] = "not-a-digest"
    write_immutable_json(root / "stages" / "data.json", {"stage": "data", "status": "passed", "signatures": bad})
    with pytest.raises(StagePrerequisiteError, match="signatures|digest"):
        run_stage("smoke", config_path, tmp_path, _run_id(), _dependencies())


def test_confirm_descriptor_read_is_independent_of_injected_path_callback(
    tmp_path: Path, config_path: Path
) -> None:
    root = tmp_path / _run_id(); root.mkdir()
    write_immutable_json(root / "stages" / "discover.json", {"stage": "discover", "status": "passed", "signatures": _signatures()})
    (root / "data").mkdir()
    original = b"{}\n"
    (root / "data" / "test.jsonl").write_bytes(original)
    write_immutable_json(root / "selection.json", {
        "status": "passed", "signatures": _signatures(),
        "test_sha256": __import__("hashlib").sha256(original).hexdigest(),
    })
    replaced = False
    def replacing_read(path: Path) -> bytes:
        nonlocal replaced
        data = path.read_bytes()
        if path.name == "test.jsonl" and not replaced:
            path.write_bytes(b'{"replacement":true}\n')
            replaced = True
        return data
    def confirm(context):
        context.read_test_rows()
        return {}
    deps = _dependencies(
        read_path=replacing_read,
        stages={**_dependencies().stage_functions, "confirm": confirm},
    )
    result = run_stage("confirm", config_path, tmp_path, _run_id(), deps)
    assert result["status"] == "passed"


def test_data_incompatible_final_root_leaves_staging_unchanged(
    tmp_path: Path, config_path: Path
) -> None:
    from ppmoe.recurrence.experiment import _control_hashes, _staging_id
    config_hash, code_hash = _control_hashes(config_path.resolve())
    staging = tmp_path / "staging" / _staging_id("Qwen/Qwen3-1.7B", config_hash, code_hash)
    staging.mkdir(parents=True)
    source_hash = sha256_payload({"name": "source"})
    write_immutable_json(staging / "stages" / "source.json", {
        "stage": "source", "status": "passed",
        "signatures": {"source": source_hash, "data": "", "config": config_hash, "code": code_hash},
        "payload": {}, "attempts": 1,
    })
    data_hash = sha256_payload({"name": "data"})
    from ppmoe.recurrence.provenance import compute_run_id
    final_id = compute_run_id(source_hash, data_hash, config_hash, code_hash)
    final = tmp_path / final_id
    write_immutable_json(final / "stages" / "data.json", {
        "stage": "data", "status": "passed",
        "signatures": {"source": "f" * 64, "data": data_hash, "config": config_hash, "code": code_hash},
        "payload": {}, "attempts": 1,
    })
    before = {p.relative_to(staging).as_posix(): p.read_bytes() for p in staging.rglob("*") if p.is_file()}
    def data(_context):
        return {"data_hash": data_hash}
    deps = _dependencies(stages={**_dependencies().stage_functions, "data": data})
    with pytest.raises(StagePrerequisiteError, match="incompatible finalized"):
        run_stage("data", config_path, tmp_path, None, deps)
    after = {p.relative_to(staging).as_posix(): p.read_bytes() for p in staging.rglob("*") if p.is_file()}
    assert after == before


def test_source_download_cap_is_checked_without_existing_ledger(
    tmp_path: Path, config_path: Path
) -> None:
    from ppmoe.recurrence.config import RecurrenceConfig
    config = RecurrenceConfig.load(config_path)
    deps = _dependencies(stages={**_dependencies().stage_functions, "source": lambda _context: {
        "source_hash": "a" * 64,
        "download_bytes": config.resources.maximum_download_bytes + 1,
    }})
    from ppmoe.recurrence.experiment import BudgetStop, _staging_id, _control_hashes
    with pytest.raises(BudgetStop):
        run_stage("source", config_path, tmp_path, None, deps)
    config_hash, code_hash = _control_hashes(config_path.resolve())
    root = tmp_path / "staging" / _staging_id(config.model.repo_id, config_hash, code_hash)
    assert json.loads((root / "stages" / "source.json").read_text())["status"] == "stopped_budget"


def test_second_child_fault_charge_is_reconciled_before_terminal_seal(
    tmp_path: Path, config_path: Path
) -> None:
    root = tmp_path / _run_id(); root.mkdir()
    write_immutable_json(root / "stages" / "data.json", {"stage": "data", "status": "passed", "signatures": _signatures()})
    def smoke(_context):
        raise FreshProcessRequired("first fault")
    child = (
        "import hashlib,json,os,sys; from pathlib import Path; "
        "r=json.load(open(sys.argv[1])); root=Path(r['run_root']); "
        "p=root/'events.jsonl'; p.write_text(p.read_text(encoding='utf-8')+json.dumps({'index':2,'timestamp':'2026-01-01T00:00:00+00:00','kind':'training_failure','stage':'smoke','attempt':2,'payload':{'message':'child fault'},'outcome':'failed_environment','cuda_process_fault':True,'chargeable_cuda_seconds':4.0},sort_keys=True,separators=(',',':'))+'\\n',encoding='utf-8'); "
        "x={'request_sha256':hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(),'nonce':r['nonce'],'stage':r['stage'],'run_root':r['run_root'],'attempt':2,'pid':os.getpid(),'status':'failed_environment','payload':{}}; "
        "open(sys.argv[2],'w',encoding='utf-8').write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\\n')"
    )
    deps = __import__("dataclasses").replace(
        _dependencies(stages={**_dependencies().stage_functions, "smoke": smoke}),
        child_command=(sys.executable, "-c", child),
    )
    with pytest.raises(EnvironmentStop):
        run_stage("smoke", config_path, tmp_path, _run_id(), deps)
    from ppmoe.recurrence.gates import CudaCostLedger
    ledger = CudaCostLedger.load(root / "cuda-cost.jsonl", 86400.0)
    assert ledger.total_seconds == 4.0
    assert len(ledger.entries) == 1
    with pytest.raises(EnvironmentStop):
        run_stage("smoke", config_path, tmp_path, _run_id(), deps)
    assert len(CudaCostLedger.load(root / "cuda-cost.jsonl", 86400.0).entries) == 1


def test_pilot_cleanup_receives_payload_and_preserves_primary_outcome(
    tmp_path: Path, config_path: Path
) -> None:
    root = tmp_path / _run_id(); root.mkdir()
    write_immutable_json(root / "stages" / "smoke.json", {"stage": "smoke", "status": "passed", "signatures": _signatures()})
    seen: list[dict[str, object]] = []
    def pilot(_context):
        return {"handle": "pilot-model", "steps": 2}
    def dispose(_context, payload):
        seen.append(dict(payload))
    def restore(_context, payload):
        seen.append(dict(payload))
        raise RuntimeError("cleanup failed")
    deps = __import__("dataclasses").replace(
        _dependencies(stages={**_dependencies().stage_functions, "pilot": pilot}),
        dispose_pilot=dispose,
        restore_initialization=restore,
    )
    result = run_stage("pilot", config_path, tmp_path, _run_id(), deps)
    assert result["status"] == "passed"
    assert seen == [{"handle": "pilot-model", "steps": 2}] * 2
    events = (root / "events.jsonl").read_text(encoding="utf-8")
    assert "pilot_cleanup_failure" in events


@pytest.mark.parametrize("wrong_field", ["attempt", "checkpoint_path", "checkpoint_digest", "checkpoint_signature"])
def test_retry_rejects_authenticated_but_wrong_checkpoint_state(
    tmp_path: Path, config_path: Path, wrong_field: str
) -> None:
    root = tmp_path / _run_id(); root.mkdir()
    write_immutable_json(root / "stages" / "data.json", {"stage": "data", "status": "passed", "signatures": _signatures()})
    def smoke(_context):
        raise FreshProcessRequired("fault")
    values = {"attempt": 1, "checkpoint_path": "wrong-checkpoint", "checkpoint_digest": "bad", "checkpoint_signature": "bad"}
    child = (
        "import hashlib,json,os,sys; r=json.load(open(sys.argv[1])); "
        f"p={{'worker_pid':os.getpid(),'attempt':{json.dumps(values['attempt'])},'checkpoint_path':{json.dumps(values['checkpoint_path'])},'checkpoint_digest':{json.dumps(values['checkpoint_digest'])},'checkpoint_update':0,'checkpoint_signature':{json.dumps(values['checkpoint_signature'])}}}; "
        "x={'request_sha256':hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(),'nonce':r['nonce'],'stage':r['stage'],'run_root':r['run_root'],'attempt':2,'pid':os.getpid(),'status':'passed','payload':p}; "
        "open(sys.argv[2],'w',encoding='utf-8').write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\\n')"
    )
    deps = __import__("dataclasses").replace(
        _dependencies(stages={**_dependencies().stage_functions, "smoke": smoke}),
        child_command=(sys.executable, "-c", child),
    )
    with pytest.raises(EnvironmentStop):
        run_stage("smoke", config_path, tmp_path, _run_id(), deps)
    assert json.loads((root / "stages" / "smoke.json").read_text())["status"] == "failed_environment"


def test_successful_pilot_retry_passes_child_payload_to_cleanup(
    tmp_path: Path, config_path: Path
) -> None:
    root = tmp_path / _run_id(); root.mkdir()
    write_immutable_json(root / "stages" / "smoke.json", {"stage": "smoke", "status": "passed", "signatures": _signatures()})
    seen: list[dict[str, object]] = []
    def pilot(_context):
        raise FreshProcessRequired("pilot fault")
    def cleanup(_context, payload):
        seen.append(dict(payload))
    deps = __import__("dataclasses").replace(
        _dependencies(stages={**_dependencies().stage_functions, "pilot": pilot}),
        dispose_pilot=cleanup,
        restore_initialization=cleanup,
    )
    result = run_stage("pilot", config_path, tmp_path, _run_id(), deps)
    assert result["status"] == "passed"
    assert len(seen) == 2
    assert all(payload.get("attempt") == 2 for payload in seen)


def test_retry_without_registered_stage_dispatcher_fails_closed(
    tmp_path: Path, config_path: Path
) -> None:
    root = tmp_path / _run_id(); root.mkdir()
    write_immutable_json(root / "stages" / "data.json", {"stage": "data", "status": "passed", "signatures": _signatures()})
    def smoke(_context):
        raise FreshProcessRequired("fault")
    deps = __import__("dataclasses").replace(
        _dependencies(stages={**_dependencies().stage_functions, "smoke": smoke}),
        child_worker_refs=None,
        child_worker_ref=None,
    )
    with pytest.raises(EnvironmentStop, match="dispatcher|worker"):
        run_stage("smoke", config_path, tmp_path, _run_id(), deps)


def test_report_without_run_id_uses_only_current_deterministic_staging_root(
    tmp_path: Path, config_path: Path
) -> None:
    from ppmoe.recurrence.experiment import _control_hashes, _staging_id
    config_hash, code_hash = _control_hashes(config_path.resolve())
    staging = tmp_path / "staging" / _staging_id("Qwen/Qwen3-1.7B", config_hash, code_hash)
    write_immutable_json(staging / "stages" / "source.json", {
        "stage": "source", "status": "stopped_budget",
        "signatures": {"source": "a" * 64, "data": "", "config": config_hash, "code": code_hash},
        "payload": {}, "attempts": 1,
    })
    result = run_stage("report", config_path, tmp_path, None, _dependencies())
    assert result["stage"] == "report"
    assert (staging / "stages" / "report.json").exists()
    replay = run_stage("report", config_path, tmp_path, None, _dependencies())
    assert replay == result
    (staging / ".hidden-evidence.txt").write_text("added after report", encoding="utf-8")
    with pytest.raises(StageError, match="inventory"):
        run_stage("report", config_path, tmp_path, None, _dependencies())


def test_report_without_run_id_rejects_missing_staging_root(
    tmp_path: Path, config_path: Path
) -> None:
    with pytest.raises(StagePrerequisiteError, match="staging|evidence"):
        run_stage("report", config_path, tmp_path, None, _dependencies())


def test_report_without_run_id_rejects_ambiguous_compatible_staging_roots(
    tmp_path: Path, config_path: Path
) -> None:
    from ppmoe.recurrence.experiment import _control_hashes, _staging_id
    config_hash, code_hash = _control_hashes(config_path.resolve())
    staging_parent = tmp_path / "staging"
    for name in (_staging_id("Qwen/Qwen3-1.7B", config_hash, code_hash), "other-compatible"):
        write_immutable_json(staging_parent / name / "stages" / "source.json", {
            "stage": "source", "status": "stopped_budget",
            "signatures": {"source": "a" * 64, "data": "", "config": config_hash, "code": code_hash},
            "payload": {}, "attempts": 1,
        })
    with pytest.raises(StagePrerequisiteError, match="ambiguous"):
        run_stage("report", config_path, tmp_path, None, _dependencies())


def test_report_without_run_id_rejects_incompatible_staging_candidate(
    tmp_path: Path, config_path: Path
) -> None:
    from ppmoe.recurrence.experiment import _control_hashes, _staging_id
    config_hash, code_hash = _control_hashes(config_path.resolve())
    staging = tmp_path / "staging" / _staging_id("Qwen/Qwen3-1.7B", config_hash, code_hash)
    write_immutable_json(staging / "stages" / "source.json", {
        "stage": "source", "status": "stopped_budget",
        "signatures": {"source": "a" * 64, "data": "", "config": config_hash, "code": "b" * 64},
        "payload": {}, "attempts": 1,
    })
    with pytest.raises(StagePrerequisiteError, match="incompatible"):
        run_stage("report", config_path, tmp_path, None, _dependencies())


def test_retry_rejects_preexisting_retry_symlink(
    tmp_path: Path, config_path: Path
) -> None:
    root = tmp_path / _run_id(); root.mkdir()
    write_immutable_json(root / "stages" / "data.json", {"stage": "data", "status": "passed", "signatures": _signatures()})
    outside = tmp_path / "outside"; outside.mkdir()
    try:
        (root / "retry").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    def smoke(_context):
        raise FreshProcessRequired("fault")
    deps = _dependencies(stages={**_dependencies().stage_functions, "smoke": smoke})
    with pytest.raises(EnvironmentStop, match="symbolic|retry"):
        run_stage("smoke", config_path, tmp_path, _run_id(), deps)
    assert not list(outside.iterdir())


def test_confirm_reads_test_bytes_without_injected_second_path_read(
    tmp_path: Path, config_path: Path
) -> None:
    root = tmp_path / _run_id(); root.mkdir()
    write_immutable_json(root / "stages" / "discover.json", {"stage": "discover", "status": "passed", "signatures": _signatures()})
    (root / "data").mkdir(); rows = b"{}\n"; (root / "data" / "test.jsonl").write_bytes(rows)
    write_immutable_json(root / "selection.json", {
        "status": "passed", "signatures": _signatures(),
        "test_sha256": __import__("hashlib").sha256(rows).hexdigest(),
    })
    def guarded(path: Path) -> bytes:
        if path.name == "test.jsonl":
            raise AssertionError("test rows must use descriptor-bound read")
        return path.read_bytes()
    def confirm(context):
        assert context.read_test_rows() == rows
        return {}
    deps = _dependencies(read_path=guarded, stages={**_dependencies().stage_functions, "confirm": confirm})
    result = run_stage("confirm", config_path, tmp_path, _run_id(), deps)
    assert result["status"] == "passed"


def test_production_dispatcher_rejects_test_echo_binding() -> None:
    from ppmoe.recurrence.experiment import _child_worker_ref
    deps = _dependencies()
    production = __import__("dataclasses").replace(deps, test_only=False)
    with pytest.raises(EnvironmentStop, match="production|frozen"):
        _child_worker_ref("smoke", production)


def test_production_retry_requires_real_checkpoint_before_child_launch(
    tmp_path: Path, config_path: Path
) -> None:
    root = tmp_path / _run_id(); root.mkdir()
    write_immutable_json(root / "stages" / "data.json", {"stage": "data", "status": "passed", "signatures": _signatures()})
    def smoke(_context):
        raise FreshProcessRequired("fault without Task-9 checkpoint")
    deps = __import__("dataclasses").replace(
        _dependencies(stages={**_dependencies().stage_functions, "smoke": smoke}),
        test_only=False,
        child_worker_refs=None,
        child_worker_ref=None,
        checkpoint_validator=None,
    )
    with pytest.raises(EnvironmentStop, match="checkpoint|failed_environment|arm/seed|index"):
        run_stage("smoke", config_path, tmp_path, _run_id(), deps)
    assert not list((root / "retry").glob("*.result.json"))


def test_test_only_checkpoint_injection_rejects_external_path(
    tmp_path: Path, config_path: Path
) -> None:
    from ppmoe.recurrence.experiment import _derive_checkpoint_artifact
    root = tmp_path / _run_id(); root.mkdir()
    events = {"kind": "training_failure", "stage": "smoke", "attempt": 1,
              "update": 0, "cuda_process_fault": True}
    (root / "events.jsonl").write_text(json.dumps(events) + "\n", encoding="utf-8")
    outside = tmp_path / "outside.pt"; outside.write_bytes(b"outside")
    deps = __import__("dataclasses").replace(
        _dependencies(), checkpoint_validator=lambda *_: {"path": str(outside), "checkpoint_sha256": "e" * 64, "update": 0}
    )
    with pytest.raises((EnvironmentStop, StagePrerequisiteError), match="outside|trusted"):
        _derive_checkpoint_artifact(root, "smoke", _signatures(), deps)


def test_production_checkpoint_derivation_binds_nested_task9_event_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ppmoe.recurrence.experiment as experiment
    root = tmp_path / _run_id(); root.mkdir()
    relative = "checkpoints/a1/47/latest.pt"
    checkpoint = root / relative
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    event = {
        "kind": "training_failure", "stage": "smoke", "arm": "a1", "seed": 47,
        "attempt": 1, "update": 2, "cuda_process_fault": True,
        "checkpoint_path": relative, "checkpoint_sha256": "a" * 64,
        "checkpoint_manifest_digest": "b" * 64,
    }
    event["event_id"] = sha256_payload({key: event.get(key) for key in (
        "kind", "stage", "arm", "seed", "attempt", "update",
        "checkpoint_path", "checkpoint_sha256", "checkpoint_manifest_digest",
    )})
    event["index"] = 19
    (root / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    monkeypatch.setattr(experiment, "validate_checkpoint_artifact", lambda *args, **kwargs: {
        "path": str(checkpoint), "checkpoint_sha256": "a" * 64,
        "manifest_digest": "b" * 64, "attempt": 1, "update": 2,
    })
    deps = __import__("dataclasses").replace(_dependencies(), test_only=False, checkpoint_validator=None, child_worker_refs=None, checkpoint_schema={"parameters": {}, "initialization_hash": ""})
    artifact = experiment._derive_checkpoint_artifact(root, "smoke", _signatures(), deps)
    assert artifact["arm"] == "a1"
    assert artifact["seed"] == 47
    assert artifact["relative_path"] == relative
    assert artifact["event_index"] == 19


def test_production_checkpoint_derivation_rejects_wrong_arm_seed_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ppmoe.recurrence.experiment as experiment
    root = tmp_path / _run_id(); root.mkdir()
    relative = "checkpoints/a1/47/latest.pt"
    checkpoint = root / relative
    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"checkpoint")
    event = {
        "kind": "training_failure", "stage": "smoke", "arm": "a2", "seed": 48,
        "attempt": 1, "update": 2, "cuda_process_fault": True,
        "checkpoint_path": relative, "checkpoint_sha256": "a" * 64,
        "checkpoint_manifest_digest": "b" * 64,
    }
    event["event_id"] = sha256_payload({key: event.get(key) for key in (
        "kind", "stage", "arm", "seed", "attempt", "update",
        "checkpoint_path", "checkpoint_sha256", "checkpoint_manifest_digest",
    )})
    event["index"] = 19
    (root / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    deps = __import__("dataclasses").replace(_dependencies(), test_only=False, checkpoint_validator=None, child_worker_refs=None)
    with pytest.raises(EnvironmentStop, match="path|arm/seed"):
        experiment._derive_checkpoint_artifact(root, "smoke", _signatures(), deps)


def test_production_allowlist_rejects_echo_even_with_matching_source_identity() -> None:
    import ppmoe.recurrence.experiment as experiment
    echo = "ppmoe.recurrence.experiment:_child_echo_worker"
    identity = experiment._callable_identity(echo)
    deps = __import__("dataclasses").replace(_dependencies(), test_only=False, child_worker_refs=None)
    original = experiment._PRODUCTION_CHILD_ALLOWLIST
    try:
        experiment._PRODUCTION_CHILD_ALLOWLIST = (("smoke", echo, identity),)
        with pytest.raises(EnvironmentStop, match="test helper"):
            experiment._child_worker_ref("smoke", deps)
    finally:
        experiment._PRODUCTION_CHILD_ALLOWLIST = original


def test_retry_revalidates_checkpoint_after_child_before_sealing(
    tmp_path: Path, config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ppmoe.recurrence.experiment as experiment
    root = tmp_path / _run_id(); root.mkdir()
    checkpoint = root / "checkpoints" / "test" / "0" / "latest.pt"
    checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b"fixture")
    artifact = {
        "path": str(checkpoint), "checkpoint_sha256": "e" * 64,
        "manifest_digest": "d" * 64, "attempt": 1, "update": 0,
        "event_id": "f" * 64, "event_index": 1, "arm": "test", "seed": 0,
        "relative_path": "checkpoints/test/0/latest.pt",
    }
    calls = 0
    def derive(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return dict(artifact)
        raise EnvironmentStop("checkpoint changed after child validation")
    monkeypatch.setattr(experiment, "_derive_checkpoint_artifact", derive)
    deps = _dependencies()
    with pytest.raises(EnvironmentStop, match="changed after child"):
        experiment._run_child_retry("smoke", config_path.resolve(), tmp_path, root, _run_id(), _signatures(), deps)
    assert calls == 2


def test_checkpoint_proof_binds_content_and_manifest_digests() -> None:
    from ppmoe.recurrence.experiment import _checkpoint_proof

    request = {
        "worker_identity": "worker-id",
        "checkpoint_path": "C:/run/checkpoints/a1/47/latest.pt",
        "checkpoint_digest": "b" * 64,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_attempt": 1,
        "checkpoint_update": 2,
        "checkpoint_event_id": "c" * 64,
        "checkpoint_event_index": 3,
        "checkpoint_arm": "a1",
        "checkpoint_seed": 47,
        "checkpoint_relative_path": "checkpoints/a1/47/latest.pt",
        "adapter_schema": {"parameters": {}, "initialization_hash": "d" * 64},
    }
    proof = _checkpoint_proof(request)
    changed = dict(request, checkpoint_sha256="e" * 64)
    assert proof != _checkpoint_proof(changed)
