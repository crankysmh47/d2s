from __future__ import annotations

from pathlib import Path
import hashlib
from dataclasses import replace

import pytest

from ppmoe.recurrence.config_v1 import load_v1_config
from ppmoe.recurrence.provenance import canonical_json
from ppmoe.recurrence.study_v1 import (
    ResourceLimitReached,
    _budget_check,
    _derive_run_id,
    _development_gate,
    _prepare,
    _read_stage,
    _report,
    _select_pilot_difficulties,
    _pilot_selection_status,
    _stage_record,
    _verify_calibration_runtime_code,
    _verify_v1_split,
    run_v1_stage,
)


def test_stage_order_rejects_unimplemented_or_unprepared_stages(tmp_path: Path) -> None:
    config = Path("configs/recurrence-v1.toml")
    with pytest.raises(ValueError, match="unsupported V1 stage"):
        run_v1_stage("discover", config)
    with pytest.raises(ValueError, match="run root"):
        run_v1_stage("calibrate", config)


def test_run_id_binds_config_and_frozen_hashes() -> None:
    signatures = {
        "source": "a" * 64,
        "model": "b" * 64,
        "tokenizer": "c" * 64,
        "train": "d" * 64,
        "dev": "e" * 64,
        "scorer": "2f138e6fe106c81dd03712d0a49a3a916cec2596a4acfa8f4e111f6463bab719",
        "generator": "f" * 64,
    }
    assert _derive_run_id(b"exact config bytes", signatures) == _derive_run_id(b"exact config bytes", signatures)
    assert _derive_run_id(b"changed config bytes", signatures) != _derive_run_id(b"exact config bytes", signatures)


def test_score_records_reject_duplicate_or_missing_ids(tmp_path: Path) -> None:
    from ppmoe.recurrence.v1_runtime import _validate_exact_rows

    expected = ["one", "two"]
    valid = [
        {"example_id": "one", "family": "expressions", "correct": True, "reason_code": "correct"},
        {"example_id": "two", "family": "expressions", "correct": False, "reason_code": "wrong_value"},
    ]
    _validate_exact_rows(valid, expected)
    with pytest.raises(ValueError, match="exactly once"):
        _validate_exact_rows([valid[0], valid[0]], expected)
    with pytest.raises(ValueError, match="exactly once"):
        _validate_exact_rows(valid[:1], expected)


def test_invalid_normalization_preserves_failure_detail() -> None:
    from ppmoe.recurrence.v1_runtime import _normalize_score

    result = _normalize_score({"correct": False, "reason_code": "length_cap", "failure_detail": "partial"})
    assert result["reason_code"] == "invalid_format"
    assert result["failure_detail"] == "partial"


def test_candidate_choice_is_first_eligible_and_fallback_is_fixed() -> None:
    table = [
        {"family": "expressions", "candidate": 1, "eligible": False},
        {"family": "expressions", "candidate": 2, "eligible": True},
        {"family": "lists", "candidate": 1, "eligible": True},
        {"family": "lists", "candidate": 2, "eligible": True},
        {"family": "graphs", "candidate": 1, "eligible": False},
        {"family": "graphs", "candidate": 2, "eligible": False},
    ]
    choices, fallback = _select_pilot_difficulties(table)
    assert choices == {"expressions": 2, "lists": 1, "graphs": 1}
    assert fallback == ["graphs"]
    choices, fallback = _select_pilot_difficulties([row for row in table if row["family"] == "graphs"])
    assert choices == {}
    assert fallback == ["expressions", "lists", "graphs"]


def test_split_hash_from_stage_record_rejects_paired_split_manifest_tampering(tmp_path: Path) -> None:
    split = tmp_path / "pilot.jsonl"
    manifest = tmp_path / "pilot.jsonl.manifest.json"
    original = b'{"id":"one"}\n'
    digest = hashlib.sha256(original).hexdigest()
    split.write_bytes(original)
    manifest.write_bytes(canonical_json({"sha256": digest, "count": 1}))
    assert _verify_v1_split(split, digest, 1)[0]["id"] == "one"
    split.write_bytes(b'{"id":"two"}\n')
    manifest.write_bytes(canonical_json({"sha256": hashlib.sha256(split.read_bytes()).hexdigest(), "count": 1}))
    with pytest.raises(ValueError, match="immutable stage record"):
        _verify_v1_split(split, digest, 1)


def test_prepare_is_exclusive_and_hash_record_is_self_checked(tmp_path: Path) -> None:
    config_path = Path("configs/recurrence-v1.toml").resolve()
    config = load_v1_config(config_path)
    root = tmp_path / "run-a"
    run_id = "a" * 64
    inputs = {
        "config_bytes": config_path.read_bytes(), "signatures": {"source": "b" * 64},
        "source_root": tmp_path / "source", "snapshot": tmp_path / "source" / "snapshot",
        "forbidden_ids": {"v0-a", "v0-b"}, "train_count": 8000, "dev_count": 600,
        "manifest_sha256": "c" * 64,
    }
    result = _prepare(root, run_id, config_path, config, inputs)
    assert result["status"] == "passed"
    assert _read_stage(root / "stages" / "prepare.json", "prepare")["stage_sha256"] == result["stage_sha256"]
    with pytest.raises(FileExistsError):
        _prepare(root, run_id, config_path, config, inputs)


def test_budget_stop_is_resource_status_and_artifact_cap_counts_files(tmp_path: Path) -> None:
    from ppmoe.recurrence.gates import CudaCostLedger
    config = load_v1_config(Path("configs/recurrence-v1.toml"))
    root = tmp_path / "run"
    root.mkdir()
    (root / "evidence.bin").write_bytes(b"evidence")
    limited = replace(config, resources=replace(config.resources, maximum_artifact_bytes=1))
    ledger = CudaCostLedger.load(root / "resources" / "cuda-cost.jsonl", 86400.0)
    with pytest.raises(ResourceLimitReached, match="artifact byte cap"):
        _budget_check(root, ledger, limited)


def test_report_waits_until_terminal_no_go_and_never_reads_sealed_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ppmoe.recurrence.study_v1 import _runtime_code_files
    config_path = tmp_path / "recurrence-v1.toml"
    config_path.write_bytes(Path("configs/recurrence-v1.toml").read_bytes())
    config = load_v1_config(config_path)
    config = replace(config, data=replace(config.data, output_root=str(tmp_path / "runs")))
    signatures = {key: chr(97 + index) * 64 for index, key in enumerate(("source", "model", "tokenizer", "train", "dev", "scorer", "generator"))}
    run_id = _derive_run_id(config_path.read_bytes(), signatures)
    root = tmp_path / "runs" / run_id
    stages = root / "stages"
    stages.mkdir(parents=True)
    (root / "data").mkdir()
    (root / "data" / "sealed.jsonl").write_bytes(b"poison; must remain unopened")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    prepare_payload = {"run_id": run_id, "run_root": str(root), "config_sha256": config_hash, "signatures": signatures}
    prepare = _stage_record("prepare", "passed", prepare_payload, {})
    (stages / "prepare.json").write_bytes(canonical_json(prepare))
    passed = _stage_record("calibrate", "passed", {"run_id": run_id}, {"prepare": prepare["stage_sha256"]})
    (stages / "calibrate.json").write_bytes(canonical_json(passed))
    with pytest.raises(ValueError, match="reserved until later study stages"):
        _report(root, config, config_path)
    stopped_payload = {"run_id": run_id, "pilot_scores": {}, "runtime_code_files": _runtime_code_files(Path.cwd()), "cuda_cost_ledger_sha256": None, "cuda_cost_ledger_total_seconds": 0.0, "calibration_cuda_active_seconds": 0.0}
    stopped = _stage_record("calibrate", "stopped_scientific", stopped_payload, {"prepare": prepare["stage_sha256"]})
    (stages / "calibrate.json").write_bytes(canonical_json(stopped))
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.name == "sealed.jsonl":
            raise AssertionError("report opened sealed confirmation values")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    result = _report(root, config, config_path)
    assert result["status"] == "stopped_scientific"
    assert _report(root, config, config_path)["stage_sha256"] == result["stage_sha256"]
    config_path.write_bytes(config_path.read_bytes() + b"\n# drift\n")
    with pytest.raises(ValueError, match="config hash|run ID"):
        _report(root, config, config_path)


def test_runtime_code_hash_change_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ppmoe.recurrence import study_v1

    root = tmp_path / "run"
    stage_dir = root / "stages"
    stage_dir.mkdir(parents=True)
    payload = {"runtime_code_files": {"src": "old"}}
    record = _stage_record("calibrate", "passed", payload, {})
    (stage_dir / "calibrate.json").write_bytes(canonical_json(record))
    monkeypatch.setattr(study_v1, "_runtime_code_files", lambda _root: {"src": "new"})
    with pytest.raises(ValueError, match="runtime code hashes changed"):
        _verify_calibration_runtime_code(root, tmp_path)


def test_development_gate_uses_strict_accuracy_and_invalid_caps() -> None:
    summary = {
        "macro_accuracy": 0.50, "global_invalid_rate": 0.0,
        "families": {
            "expressions": {"accuracy": 0.05, "invalid_rate": 0.0},
            "graphs": {"accuracy": 0.50, "invalid_rate": 0.10},
            "lists": {"accuracy": 0.50, "invalid_rate": 0.0},
        },
    }
    result = _development_gate(summary)
    assert result["passes"] is True
    summary["global_invalid_rate"] = 0.1001
    assert _development_gate(summary)["passes"] is False


def test_fewer_than_two_eligible_families_is_terminal_scientific_stop() -> None:
    choices, _ = _select_pilot_difficulties([
        {"family": "expressions", "candidate": 1, "eligible": True},
        {"family": "lists", "candidate": 1, "eligible": False},
        {"family": "graphs", "candidate": 1, "eligible": False},
    ])
    assert _pilot_selection_status(choices) == "stopped_scientific"


def test_prepare_verifies_train_dev_but_never_opens_v0_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    from ppmoe.recurrence import study_v1

    revision = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    source_root = tmp_path / "source-run"
    snapshot = source_root / "source" / revision
    snapshot.mkdir(parents=True)
    tokenizer_bytes, weights_bytes = b"tokenizer", b"weights"
    (snapshot / "tokenizer.json").write_bytes(tokenizer_bytes)
    (snapshot / "weights.safetensors").write_bytes(weights_bytes)
    files = [
        {"relative_path": "tokenizer.json", "bytes": len(tokenizer_bytes), "sha256": hashlib.sha256(tokenizer_bytes).hexdigest()},
        {"relative_path": "weights.safetensors", "bytes": len(weights_bytes), "sha256": hashlib.sha256(weights_bytes).hexdigest()},
    ]
    metadata_dir = snapshot / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    for entry in files:
        (metadata_dir / f"{Path(str(entry['relative_path'])).name}.metadata").write_bytes(b"etag\ncommit")
    source_manifest = {"repo_id": "Qwen/Qwen3-1.7B", "revision": revision, "model_class": "Qwen3ForCausalLM", "files": files}
    source_raw = canonical_json(source_manifest)
    (source_root / "source.json").write_bytes(source_raw)
    source_hash = hashlib.sha256(source_raw).hexdigest()
    (source_root / "stages").mkdir()
    (source_root / "stages" / "source.json").write_bytes(canonical_json({
        "stage": "source", "status": "passed", "payload": {
            "revision": revision, "snapshot_relative_path": f"source/{revision}",
            "source_manifest_sha256": source_hash, "tokenizer_revision": revision,
            "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        }, "signatures": {"source": source_hash},
    }))
    data_root = source_root / "data"
    data_root.mkdir()
    train_rows = [{"id": f"train-{i}", "split": "train"} for i in range(8000)]
    dev_rows = [{"id": f"dev-{i}", "split": "dev"} for i in range(600)]
    train_raw = b"".join(canonical_json(row) for row in train_rows)
    dev_raw = b"".join(canonical_json(row) for row in dev_rows)
    (data_root / "train.jsonl").write_bytes(train_raw)
    (data_root / "dev.jsonl").write_bytes(dev_raw)
    train_ids = sorted(row["id"] for row in train_rows)
    dev_ids = sorted(row["id"] for row in dev_rows)
    from ppmoe.recurrence.provenance import sha256_payload
    manifest = {
        "counts": {"train": 8000, "dev": 600},
        "split_sha256": {"train": hashlib.sha256(train_raw).hexdigest(), "dev": hashlib.sha256(dev_raw).hexdigest()},
        "split_id_sha256": {"train": sha256_payload({"ids": train_ids}), "dev": sha256_payload({"ids": dev_ids})},
    }
    manifest_raw = canonical_json(manifest)
    (data_root / "manifest.json").write_bytes(manifest_raw)
    data_hash = "d" * 64
    (source_root / "stages" / "data.json").write_bytes(canonical_json({
        "stage": "data", "status": "passed", "payload": {
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(), "data_hash": data_hash,
        }, "signatures": {"source": source_hash, "data": data_hash},
    }))
    poison_test = data_root / "test.jsonl"
    poison_test.write_bytes(b"poison")
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.name == "test.jsonl":
            raise AssertionError("V0 sealed test data was opened")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    config = replace(load_v1_config(Path("configs/recurrence-v1.toml")), source_run=str(source_root))
    result = study_v1._verify_source(config, Path("configs/recurrence-v1.toml").resolve(), Path.cwd())
    assert result["train_count"] == 8000
    assert result["dev_count"] == 600
    assert len(result["forbidden_ids"]) == 8600


def test_snapshot_rejects_unlisted_regular_files(tmp_path: Path) -> None:
    import hashlib
    from ppmoe.recurrence.study_v1 import _verify_snapshot

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "weights.safetensors").write_bytes(b"listed")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    manifest = {"files": [{
        "relative_path": "weights.safetensors", "bytes": 6,
        "sha256": hashlib.sha256(b"listed").hexdigest(),
    }]}
    metadata_dir = snapshot / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "weights.safetensors.metadata").write_bytes(b"metadata")
    with pytest.raises(ValueError, match="inventory differs"):
        _verify_snapshot(snapshot, manifest)


def test_snapshot_allows_only_inventory_derived_huggingface_metadata(tmp_path: Path) -> None:
    import hashlib
    from ppmoe.recurrence.study_v1 import _verify_snapshot

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    listed = snapshot / "model.safetensors"
    listed.write_bytes(b"weights")
    metadata_dir = snapshot / ".cache" / "huggingface" / "download"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "model.safetensors.metadata").write_bytes(b"etag\ncommit")
    manifest = {"files": [{
        "relative_path": "model.safetensors", "bytes": len(b"weights"),
        "sha256": hashlib.sha256(b"weights").hexdigest(),
    }]}
    _verify_snapshot(snapshot, manifest)
    (metadata_dir / "unlisted-config.json.metadata").write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="inventory differs"):
        _verify_snapshot(snapshot, manifest)


def test_run_root_binding_rejects_copied_prepare_stage(tmp_path: Path) -> None:
    import hashlib
    from ppmoe.recurrence.study_v1 import _validate_run_binding

    config_path = Path("configs/recurrence-v1.toml").resolve()
    config = load_v1_config(config_path)
    signatures = {key: chr(97 + index) * 64 for index, key in enumerate(("source", "model", "tokenizer", "train", "dev", "scorer", "generator"))}
    run_id = _derive_run_id(config_path.read_bytes(), signatures)
    expected = tmp_path / "runs" / run_id
    expected.mkdir(parents=True)
    prepare = _stage_record("prepare", "passed", {
        "run_id": run_id, "run_root": str(expected),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "signatures": signatures,
    }, {})
    local_config = replace(config, data=replace(config.data, output_root="runs"))
    _validate_run_binding(expected, prepare, local_config, config_path, tmp_path)
    copied = tmp_path / "copied" / run_id
    copied.mkdir(parents=True)
    with pytest.raises(ValueError, match="configured output root"):
        _validate_run_binding(copied, prepare, local_config, config_path, tmp_path)


def test_terminal_report_rejects_tampered_score_split_and_sidecar(tmp_path: Path) -> None:
    import hashlib
    from ppmoe.recurrence.study_v1 import _validate_calibration_files

    root = tmp_path / "run"
    data_dir, score_dir = root / "data", root / "scores"
    data_dir.mkdir(parents=True)
    score_dir.mkdir()
    split = data_dir / "pilot-expressions-c1.jsonl"
    split_raw = b'{"id":"one","split":"pilot","family":"expressions"}\n'
    split.write_bytes(split_raw)
    split_sha = hashlib.sha256(split_raw).hexdigest()
    manifest = {"sha256": split_sha, "count": 1, "ordered_ids": ["one"]}
    manifest_path = Path(str(split) + ".manifest.json")
    manifest_path.write_bytes(canonical_json(manifest))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    score = score_dir / "pilot-expressions-c1.jsonl"
    score_raw = canonical_json({"example_id": "one", "family": "expressions", "correct": True, "reason_code": "correct"})
    score.write_bytes(score_raw)
    summary = {"record_count": 1, "correct": 1, "incorrect": 0, "invalid_format": 0, "pooled_accuracy": 1.0, "global_accuracy": 1.0, "global_invalid_rate": 0.0, "macro_accuracy": 1.0, "families": {"expressions": {"record_count": 1, "correct": 1, "incorrect": 0, "invalid_format": 0, "accuracy": 1.0, "invalid_rate": 0.0, "informative": False}}, "family_order": ["expressions"]}
    payload = {"pilot_scores": {"pilot-expressions-c1": {
        "split_sha256": split_sha, "split_count": 1, "split_manifest_sha256": manifest_sha,
        "score_sha256": hashlib.sha256(score_raw).hexdigest(), "score_count": 1, "summary": summary,
    }}, "partial_failure": None, "dev": None}
    (data_dir / "sealed.jsonl").write_bytes(b"sealed payload")
    (data_dir / "sealed.jsonl.manifest.json").write_bytes(b"sealed manifest")
    _validate_calibration_files(root, payload)
    split.write_bytes(b'{"id":"two","split":"pilot","family":"expressions"}\n')
    manifest_path.write_bytes(canonical_json({"sha256": hashlib.sha256(split.read_bytes()).hexdigest(), "count": 1, "ordered_ids": ["two"]}))
    with pytest.raises(ValueError, match="stage-bound split hash"):
        _validate_calibration_files(root, payload)


def test_interrupted_phase_report_seals_failure_and_never_resumes(tmp_path: Path) -> None:
    import hashlib
    import time
    from datetime import datetime, timezone
    from ppmoe.recurrence.study_v1 import _assert_calibration_not_started, _recover_interrupted_calibration

    config_path = Path("configs/recurrence-v1.toml").resolve()
    config = load_v1_config(config_path)
    config = replace(config, data=replace(config.data, output_root=str(tmp_path / "runs")))
    signatures = {key: chr(97 + index) * 64 for index, key in enumerate(("source", "model", "tokenizer", "train", "dev", "scorer", "generator"))}
    run_id = _derive_run_id(config_path.read_bytes(), signatures)
    root = tmp_path / "runs" / run_id
    (root / "stages").mkdir(parents=True)
    (root / "resources").mkdir()
    prepare_payload = {"run_id": run_id, "run_root": str(root), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(), "signatures": signatures}
    prepare = _stage_record("prepare", "passed", prepare_payload, {})
    (root / "stages" / "prepare.json").write_bytes(canonical_json(prepare))
    attempt = _stage_record("calibration-attempt", "running", {
        "run_id": run_id, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "wall_started_monotonic": time.monotonic() - 20.0,
        "process_id": 99999999,
    }, {"prepare": prepare["stage_sha256"]})
    (root / "stages" / "calibration-attempt.json").write_bytes(canonical_json(attempt))
    (root / "resources" / "progress.jsonl").write_bytes(canonical_json({
        "event": "phase_started", "phase_id": "lost-score", "phase": "score", "process_id": 99999999,
        "path": "scores/pilot-expressions-c1.jsonl", "wall_started_utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_started": time.monotonic() - 20.0,
    }))
    result = _recover_interrupted_calibration(root, config, config_path, Path.cwd())
    assert result["status"] in {"failed_environment", "stopped_resource"}
    assert result["resources"]["cuda_budget_state"] == "unknown_after_process_loss"
    with pytest.raises(FileExistsError, match="attempt already exists"):
        _assert_calibration_not_started(root)


def test_cuda_budget_ledger_tampering_fails_closed(tmp_path: Path) -> None:
    from ppmoe.recurrence.gates import CudaCostLedger
    from ppmoe.recurrence.study_v1 import _load_cost_ledger

    root = tmp_path / "run"
    root.mkdir()
    (root / "resources").mkdir()
    ledger_path = root / "resources" / "cuda-cost.jsonl"
    ledger = CudaCostLedger.load(ledger_path, 86400.0)
    ledger.record("attempt-a", "calibration-score", 2.0, "completed")
    assert _load_cost_ledger(root, 86400.0).total_seconds == pytest.approx(2.0)
    original = ledger_path.read_bytes()
    ledger_path.write_bytes(original.replace(b'"elapsed_seconds":2.0', b'"elapsed_seconds":3.0'))
    with pytest.raises(ValueError, match="integrity|anchor"):
        _load_cost_ledger(root, 86400.0)
    ledger_path.write_bytes(original)
    ledger_path.write_bytes(ledger_path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="integrity|truncated|anchor"):
        _load_cost_ledger(root, 86400.0)


def test_failed_measured_model_load_is_charged_to_anchored_ledger(tmp_path: Path) -> None:
    from ppmoe.recurrence.gates import CudaCostLedger
    from ppmoe.recurrence.study_v1 import _record_phase_cost

    root = tmp_path / "run"
    (root / "resources").mkdir(parents=True)
    ledger = CudaCostLedger.load(root / "resources" / "cuda-cost.jsonl", 86400.0)
    _record_phase_cost(root, ledger, "failed-load", "calibration-model-load", 1.75, "failed_environment")
    reloaded = CudaCostLedger.load(root / "resources" / "cuda-cost.jsonl", 86400.0)
    assert reloaded.total_seconds == pytest.approx(1.75)
    assert reloaded.entries[-1].outcome == "failed_environment"
