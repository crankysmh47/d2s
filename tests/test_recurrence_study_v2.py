from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from ppmoe.recurrence.config_v2 import load_v2_config
from ppmoe.recurrence.data import Example
from ppmoe.recurrence.data_v2 import make_training_view_v2
from ppmoe.recurrence.provenance import canonical_json, sha256_payload
from ppmoe.recurrence.study_v1 import _runtime_code_files, _stage_record as v1_stage_record
from ppmoe.recurrence import study_v2
from ppmoe.recurrence.study_v2 import verify_v2_sources


_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
_SOURCE_REL = "artifacts/recurrence-v0/129d39646933"
_V1_RUN_REL = "artifacts/recurrence-v1"
_V1_ID = "4f1ddbedf7b4cbd11d73699d579a3e299a049aa8fb2b036e9124842d695eb2ca"
_RUNTIME_FILES = (
    "src/ppmoe/recurrence/study_v1.py",
    "src/ppmoe/recurrence/v1_runtime.py",
    "src/ppmoe/recurrence/model.py",
    "src/ppmoe/recurrence/evaluate.py",
    "src/ppmoe/recurrence/data.py",
    "src/ppmoe/recurrence/human_review.py",
    "src/ppmoe/recurrence/gates.py",
)
_CODE_FILES = (
    "scripts/run_recurrence_v2.py",
    "src/ppmoe/recurrence/answer_v1.py",
    "src/ppmoe/recurrence/data_v1.py",
    "src/ppmoe/recurrence/config_v1.py",
    "src/ppmoe/recurrence/v1_runtime.py",
    "src/ppmoe/recurrence/study_v1.py",
    "src/ppmoe/recurrence/model.py",
    "src/ppmoe/recurrence/evaluate.py",
    "src/ppmoe/recurrence/data.py",
    "src/ppmoe/recurrence/human_review.py",
    "src/ppmoe/recurrence/gates.py",
    "src/ppmoe/recurrence/answer_v2.py",
    "src/ppmoe/recurrence/config_v2.py",
    "src/ppmoe/recurrence/data_v2.py",
    "src/ppmoe/recurrence/study_v2.py",
    "src/ppmoe/recurrence/v2_runtime.py",
    "src/ppmoe/recurrence/provenance.py",
)


def _stage(stage: str, status: str, payload: dict[str, object], prerequisites: dict[str, str] | None = None) -> dict[str, object]:
    unsigned = {
        "schema_version": 1,
        "stage": stage,
        "status": status,
        "prerequisite_stage_sha256": prerequisites or {},
        "payload": payload,
    }
    return {**unsigned, "stage_sha256": sha256_payload(unsigned)}


@pytest.fixture
def v2_source_fixture(tmp_path: Path) -> tuple[object, Path, Path]:
    repo = Path.cwd()
    for relative in _CODE_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, destination)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    v1_config_path = config_dir / "recurrence-v1.toml"
    v1_config_path.write_bytes((repo / "configs/recurrence-v1.toml").read_bytes())
    (config_dir / "recurrence-v2.toml").write_bytes((repo / "configs/recurrence-v2.toml").read_bytes())

    source_root = tmp_path / _SOURCE_REL
    snapshot = source_root / "source" / _REVISION
    snapshot.mkdir(parents=True)
    tokenizer_bytes, weights_bytes = b"exact tokenizer bytes", b"local model bytes"
    (snapshot / "tokenizer.json").write_bytes(tokenizer_bytes)
    (snapshot / "weights.safetensors").write_bytes(weights_bytes)
    (snapshot / ".cache/huggingface/download").mkdir(parents=True)
    for name in ("tokenizer.json", "weights.safetensors"):
        (snapshot / ".cache/huggingface/download" / f"{name}.metadata").write_bytes(b"etag\ncommit")
    inventory = [
        {"relative_path": "tokenizer.json", "bytes": len(tokenizer_bytes), "sha256": hashlib.sha256(tokenizer_bytes).hexdigest()},
        {"relative_path": "weights.safetensors", "bytes": len(weights_bytes), "sha256": hashlib.sha256(weights_bytes).hexdigest()},
    ]
    source_manifest = {
        "repo_id": "Qwen/Qwen3-1.7B", "revision": _REVISION,
        "model_class": "Qwen3ForCausalLM", "files": inventory,
    }
    source_raw = canonical_json(source_manifest)
    (source_root / "source.json").write_bytes(source_raw)
    source_hash = hashlib.sha256(source_raw).hexdigest()
    stages = source_root / "stages"
    stages.mkdir()
    (stages / "source.json").write_bytes(canonical_json({
        "stage": "source", "status": "passed", "payload": {
            "revision": _REVISION, "snapshot_relative_path": f"source/{_REVISION}",
            "source_manifest_sha256": source_hash, "tokenizer_revision": _REVISION,
            "tokenizer_sha256": inventory[0]["sha256"],
        }, "signatures": {"source": source_hash},
    }))

    data_dir = source_root / "data"
    data_dir.mkdir()
    train = [{"id": f"train-{index}", "split": "train"} for index in range(8000)]
    dev = [{"id": f"dev-{index}", "split": "dev"} for index in range(600)]
    train_raw = b"".join(canonical_json(row) for row in train)
    dev_raw = b"".join(canonical_json(row) for row in dev)
    (data_dir / "train.jsonl").write_bytes(train_raw)
    (data_dir / "dev.jsonl").write_bytes(dev_raw)
    (data_dir / "test.jsonl").write_bytes(b"poisoned V0 sealed path")
    manifest = {
        "counts": {"train": 8000, "dev": 600},
        "split_sha256": {"train": hashlib.sha256(train_raw).hexdigest(), "dev": hashlib.sha256(dev_raw).hexdigest()},
        "split_id_sha256": {
            "train": sha256_payload({"ids": sorted(row["id"] for row in train)}),
            "dev": sha256_payload({"ids": sorted(row["id"] for row in dev)}),
        },
    }
    manifest_raw = canonical_json(manifest)
    (data_dir / "manifest.json").write_bytes(manifest_raw)
    data_hash = "d" * 64
    (stages / "data.json").write_bytes(canonical_json({
        "stage": "data", "status": "passed", "payload": {
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(), "data_hash": data_hash,
        }, "signatures": {"source": source_hash, "data": data_hash},
    }))

    from ppmoe.recurrence.config_v1 import load_v1_config
    from ppmoe.recurrence.study_v1 import _derive_run_id as derive_v1_run_id, _verify_source

    v1_config = load_v1_config(v1_config_path)
    verified = _verify_source(v1_config, v1_config_path, tmp_path)
    v1_run_id = derive_v1_run_id(v1_config_path.read_bytes(), verified["signatures"])
    v1_root = tmp_path / _V1_RUN_REL / v1_run_id
    (v1_root / "stages").mkdir(parents=True)
    (v1_root / "data").mkdir()
    prepare = v1_stage_record("prepare", "passed", {
        "run_id": v1_run_id, "run_root": str(v1_root),
        "config_sha256": hashlib.sha256(v1_config_path.read_bytes()).hexdigest(),
        "signatures": verified["signatures"],
        "source_root": str(source_root),
        "snapshot_relative_path": f"source/{_REVISION}",
        "source_manifest_sha256": verified["signatures"]["source"],
        "data_manifest_sha256": verified["manifest_sha256"],
        "v0_train_count": 8000,
        "v0_dev_count": 600,
        "v0_train_dev_forbidden_id_count": 8600,
    }, {})
    (v1_root / "stages/prepare.json").write_bytes(canonical_json(prepare))
    pilots: dict[str, object] = {}
    for family in ("expressions", "lists", "graphs"):
        for candidate in (1, 2):
            key = f"pilot-{family}-c{candidate}"
            ids = [f"{family}-c{candidate}-{index}" for index in range(30)]
            split = v1_root / "data" / f"{key}.jsonl"
            raw = b"".join(canonical_json({"id": identifier, "split": "pilot"}) for identifier in ids)
            split.write_bytes(raw)
            sidecar = Path(f"{split}.manifest.json")
            sidecar.write_bytes(canonical_json({
                "sha256": hashlib.sha256(raw).hexdigest(), "count": 30, "ordered_ids": ids,
            }))
            pilots[key] = {
                "split_sha256": hashlib.sha256(raw).hexdigest(),
                "split_manifest_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                "split_count": 30,
            }
    runtime_files = _runtime_code_files(tmp_path)
    signatures = {**verified["signatures"], "runtime_code": sha256_payload(runtime_files)}
    calibration = v1_stage_record("calibrate", "stopped_scientific", {
        "run_id": v1_run_id, "signatures": signatures,
        "runtime_code_files": runtime_files, "pilot_scores": pilots,
    }, {"prepare": prepare["stage_sha256"]})
    (v1_root / "stages/calibrate.json").write_bytes(canonical_json(calibration))
    (v1_root / "data/sealed.jsonl").write_bytes(b"poisoned V1 sealed path")

    config = replace(
        load_v2_config(repo / "configs/recurrence-v2.toml"),
        v1_pilot_root=(Path(_V1_RUN_REL) / v1_run_id).as_posix(),
    )
    return config, tmp_path, v1_root / "data/pilot-expressions-c1.jsonl"


def test_verifies_actual_registered_row_counts_and_all_v1_pilot_ids(v2_source_fixture: tuple[object, Path, Path]) -> None:
    config, project_root, _ = v2_source_fixture
    result = verify_v2_sources(config, project_root)
    assert result["train_count"] == 8000
    assert result["dev_count"] == 600
    assert len(result["v0_train_dev_ids"]) == 8600
    assert result["v0_train_ids"] == tuple(f"train-{index}" for index in range(8000))
    assert result["v0_dev_ids"] == tuple(f"dev-{index}" for index in range(600))
    assert len(result["v1_pilot_ids"]) == 180
    assert len(result["forbidden_ids"]) == 8780


def test_changed_v1_pilot_prevents_prepare_before_v2_root_creation(v2_source_fixture: tuple[object, Path, Path]) -> None:
    config, project_root, split = v2_source_fixture
    split.write_bytes(split.read_bytes() + b" ")
    with pytest.raises(ValueError, match="V1 pilot.*hash"):
        verify_v2_sources(config, project_root)
    assert not (project_root / "artifacts/recurrence-v2").exists()


def test_run_prepare_rejects_tampered_v1_pilot_before_creating_v2_root(
    v2_source_fixture: tuple[object, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, project_root, split = v2_source_fixture
    split.write_bytes(split.read_bytes() + b" ")
    config_path = project_root / "configs/recurrence-v2.toml"
    monkeypatch.setattr(study_v2, "__file__", str(project_root / "src/ppmoe/recurrence/study_v2.py"))
    monkeypatch.setattr(study_v2, "load_v2_config", lambda _path: config)
    with pytest.raises(ValueError, match="V1 pilot.*hash"):
        study_v2.run_v2_stage("prepare", config_path)
    assert not (project_root / "artifacts/recurrence-v2").exists()


def test_verification_never_opens_v0_or_v1_sealed_data(
    v2_source_fixture: tuple[object, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, project_root, _ = v2_source_fixture
    original_read_bytes = Path.read_bytes
    forbidden = {"test.jsonl", "sealed.jsonl"}

    def guarded_read_bytes(path: Path) -> bytes:
        if path.name in forbidden:
            raise AssertionError(f"sealed input was opened: {path}")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    assert len(verify_v2_sources(config, project_root)["v1_pilot_ids"]) == 180


def test_duplicate_id_in_stage_bound_v1_pilot_is_rejected(v2_source_fixture: tuple[object, Path, Path]) -> None:
    config, project_root, split = v2_source_fixture
    rows = [json.loads(line) for line in split.read_bytes().splitlines()]
    rows[1]["id"] = rows[0]["id"]
    raw = b"".join(canonical_json(row) for row in rows)
    split.write_bytes(raw)
    sidecar = Path(f"{split}.manifest.json")
    sidecar_raw = canonical_json({
        "sha256": hashlib.sha256(raw).hexdigest(), "count": 30,
        "ordered_ids": [row["id"] for row in rows],
    })
    sidecar.write_bytes(sidecar_raw)
    calibrate_path = split.parents[1] / "stages/calibrate.json"
    calibrate = json.loads(calibrate_path.read_bytes())
    entry = calibrate["payload"]["pilot_scores"]["pilot-expressions-c1"]
    entry["split_sha256"] = hashlib.sha256(raw).hexdigest()
    entry["split_manifest_sha256"] = hashlib.sha256(sidecar_raw).hexdigest()
    calibrate_path.write_bytes(canonical_json(_stage(
        "calibrate", calibrate["status"], calibrate["payload"], calibrate["prerequisite_stage_sha256"],
    )))
    with pytest.raises(ValueError, match="duplicate|unique"):
        verify_v2_sources(config, project_root)


def test_changed_local_tokenizer_bytes_fail_inventory_authentication(v2_source_fixture: tuple[object, Path, Path]) -> None:
    config, project_root, _ = v2_source_fixture
    tokenizer = project_root / _SOURCE_REL / "source" / _REVISION / "tokenizer.json"
    tokenizer.write_bytes(tokenizer.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="tokenizer|model file hash"):
        verify_v2_sources(config, project_root)


def test_v0_source_manifest_hash_is_authenticated(v2_source_fixture: tuple[object, Path, Path]) -> None:
    config, project_root, _ = v2_source_fixture
    source_manifest = project_root / _SOURCE_REL / "source.json"
    source_manifest.write_bytes(source_manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source stage.*manifest"):
        verify_v2_sources(config, project_root)


def test_v0_data_manifest_hash_is_authenticated(v2_source_fixture: tuple[object, Path, Path]) -> None:
    config, project_root, _ = v2_source_fixture
    data_manifest = project_root / _SOURCE_REL / "data/manifest.json"
    data_manifest.write_bytes(data_manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="data stage.*manifest"):
        verify_v2_sources(config, project_root)


def test_duplicate_v0_train_ids_fail_after_rebinding_its_manifest(v2_source_fixture: tuple[object, Path, Path]) -> None:
    config, project_root, _ = v2_source_fixture
    source_root = project_root / _SOURCE_REL
    train_path = source_root / "data/train.jsonl"
    rows = [json.loads(line) for line in train_path.read_bytes().splitlines()]
    rows[1]["id"] = rows[0]["id"]
    train_raw = b"".join(canonical_json(row) for row in rows)
    train_path.write_bytes(train_raw)
    manifest_path = source_root / "data/manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["split_sha256"]["train"] = hashlib.sha256(train_raw).hexdigest()
    manifest["split_id_sha256"]["train"] = sha256_payload({"ids": sorted(row["id"] for row in rows)})
    manifest_raw = canonical_json(manifest)
    manifest_path.write_bytes(manifest_raw)
    stage_path = source_root / "stages/data.json"
    stage = json.loads(stage_path.read_bytes())
    stage["payload"]["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    stage_path.write_bytes(canonical_json(stage))
    with pytest.raises(ValueError, match="IDs must be non-empty and unique"):
        verify_v2_sources(config, project_root)


def test_v1_pilot_order_must_match_the_authenticated_sidecar(v2_source_fixture: tuple[object, Path, Path]) -> None:
    config, project_root, split = v2_source_fixture
    rows = [json.loads(line) for line in split.read_bytes().splitlines()]
    raw = split.read_bytes()
    sidecar = Path(f"{split}.manifest.json")
    sidecar_raw = canonical_json({
        "sha256": hashlib.sha256(raw).hexdigest(), "count": 30,
        "ordered_ids": [row["id"] for row in reversed(rows)],
    })
    sidecar.write_bytes(sidecar_raw)
    calibrate_path = split.parents[1] / "stages/calibrate.json"
    calibrate = json.loads(calibrate_path.read_bytes())
    calibrate["payload"]["pilot_scores"]["pilot-expressions-c1"]["split_manifest_sha256"] = hashlib.sha256(sidecar_raw).hexdigest()
    calibrate_path.write_bytes(canonical_json(_stage(
        "calibrate", calibrate["status"], calibrate["payload"], calibrate["prerequisite_stage_sha256"],
    )))
    with pytest.raises(ValueError, match="ordered IDs differ"):
        verify_v2_sources(config, project_root)


def test_training_view_preserves_every_real_v0_train_row_in_order() -> None:
    source = Path("artifacts/recurrence-v0/129d39646933/data/train.jsonl")
    rows = [Example(**json.loads(line)) for line in source.read_bytes().splitlines()]
    assert len(rows) == 8000
    view = make_training_view_v2(rows)
    assert len(view) == len(rows)
    assert [row.id for row in view] == [row.id for row in rows]
    assert [row.answer for row in view] == [row.answer for row in rows]
    assert [row.structure_id for row in view] == [row.structure_id for row in rows]
    for before, after in zip(rows, view, strict=True):
        if before.family == "lists":
            assert after.prompt != before.prompt
        else:
            assert after.prompt == before.prompt


class _TinyTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages: list[dict[str, str]], **_kwargs: object) -> list[int]:
        return list(range(max(1, len(messages[0]["content"].split()))))

    def __call__(self, answer: str, **_kwargs: object) -> dict[str, list[int]]:
        return {"input_ids": [1] * max(1, len(answer.split()))}


def _prepare_inputs() -> tuple[list[dict[str, object]], dict[str, object]]:
    source = Path("artifacts/recurrence-v0/129d39646933/data/train.jsonl")
    rows = [json.loads(line) for line in source.read_bytes().splitlines()]
    return rows, {
        "signatures": {
            "v0_source_manifest": "a" * 64,
            "v0_data_manifest": "b" * 64,
            "tokenizer": "f" * 64,
            "train": "c" * 64,
            "dev": "d" * 64,
        },
        "source_root": source.parents[1],
        "snapshot": Path("artifacts/recurrence-v0/129d39646933/source") / _REVISION,
        "manifest_sha256": "e" * 64,
        "train_count": 8000,
        "dev_count": 600,
        "forbidden_ids": {f"id-{index}" for index in range(8780)},
        "train_records": tuple(rows),
        "v0_train_ids": tuple(str(row["id"]) for row in rows),
    }


def _run_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = replace(
        load_v2_config(Path("configs/recurrence-v2.toml")),
        data=replace(load_v2_config(Path("configs/recurrence-v2.toml")).data, output_root=str(tmp_path / "runs")),
    )
    _, inputs = _prepare_inputs()
    monkeypatch.setattr(study_v2, "load_v2_config", lambda _path: config)
    monkeypatch.setattr(study_v2, "verify_v2_sources", lambda _config, _root: inputs)
    monkeypatch.setattr(study_v2, "_load_v2_tokenizer", lambda _snapshot, _revision: _TinyTokenizer())
    result = study_v2.run_v2_stage("prepare", Path("configs/recurrence-v2.toml"))
    assert result["status"] == "passed", result.get("failure")
    return Path(str(result["run_root"]))


def test_prepare_run_id_is_64_hex_and_stage_binds_config_sources_and_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _run_prepare(tmp_path, monkeypatch)
    record = study_v2.read_v2_stage(root / "stages/prepare.json", "prepare")
    payload = record["payload"]
    assert isinstance(payload, dict)
    run_id = str(payload["run_id"])
    assert len(run_id) == 64 and all(character in "0123456789abcdef" for character in run_id)
    assert root.name == run_id
    assert payload["signatures"]["config"] == hashlib.sha256(Path("configs/recurrence-v2.toml").read_bytes()).hexdigest()
    assert payload["signatures"]["train_view"] == hashlib.sha256((root / "data/train-view.jsonl").read_bytes()).hexdigest()
    manifest = json.loads((root / "data/train-view.jsonl.manifest.json").read_bytes())
    assert manifest["count"] == 8000
    assert len(manifest["ordered_ids"]) == 8000
    assert manifest["tokenizer_sha256"] == "f" * 64


def test_prepare_never_overwrites_an_existing_run_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _run_prepare(tmp_path, monkeypatch)
    stage_before = (root / "stages/prepare.json").read_bytes()
    with pytest.raises(FileExistsError):
        study_v2.run_v2_stage("prepare", Path("configs/recurrence-v2.toml"))
    assert (root / "stages/prepare.json").read_bytes() == stage_before


def test_prepare_binding_rejects_config_or_code_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _run_prepare(tmp_path, monkeypatch)
    record = study_v2.read_v2_stage(root / "stages/prepare.json", "prepare")
    payload = record["payload"]
    assert isinstance(payload, dict)
    frozen_signatures = payload["signatures"]
    assert isinstance(frozen_signatures, dict)
    source_signatures = {key: value for key, value in frozen_signatures.items() if key != "config"}
    config_bytes = Path("configs/recurrence-v2.toml").read_bytes()
    with pytest.raises(ValueError, match="signature changed|config hash changed"):
        study_v2.validate_v2_prepare_binding(record, config_bytes + b"\n", source_signatures)
    drifted = {**source_signatures, "code:source": "0" * 64}
    with pytest.raises(ValueError, match="signature changed"):
        study_v2.validate_v2_prepare_binding(record, config_bytes, drifted)


def test_prepare_failure_after_root_creation_records_failed_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(
        load_v2_config(Path("configs/recurrence-v2.toml")),
        data=replace(load_v2_config(Path("configs/recurrence-v2.toml")).data, output_root=str(tmp_path / "runs")),
    )
    _, inputs = _prepare_inputs()
    monkeypatch.setattr(study_v2, "load_v2_config", lambda _path: config)
    monkeypatch.setattr(study_v2, "verify_v2_sources", lambda _config, _root: inputs)
    monkeypatch.setattr(study_v2, "_load_v2_tokenizer", lambda _snapshot, _revision: _TinyTokenizer())
    original_write = study_v2._write_new

    def fail_manifest(path: Path, raw: bytes) -> None:
        if path.name == "train-view.jsonl.manifest.json":
            raise OSError("simulated manifest write failure")
        original_write(path, raw)

    monkeypatch.setattr(study_v2, "_write_new", fail_manifest)
    result = study_v2.run_v2_stage("prepare", Path("configs/recurrence-v2.toml"))
    assert result["status"] == "failed_environment"
    record = study_v2.read_v2_stage(Path(str(result["run_root"])) / "stages/prepare.json", "prepare")
    assert record["status"] == "failed_environment"


def test_calibrate_rehashes_prepared_train_view_before_any_gpu_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _run_prepare(tmp_path, monkeypatch)
    train_path = root / "data/train-view.jsonl"
    train_path.write_bytes(train_path.read_bytes() + b" ")
    model_loaded = False

    def load_model(*_args):
        nonlocal model_loaded
        model_loaded = True
        raise AssertionError("model load must not start after prepared data drift")

    monkeypatch.setattr(study_v2, "_load_v2_model_measured", load_model, raising=False)
    with pytest.raises(ValueError, match="train-view.*hash|signature changed"):
        study_v2.run_v2_stage("calibrate", Path("configs/recurrence-v2.toml"), root)
    assert model_loaded is False


def test_report_after_calibration_stop_does_not_read_sealed_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _enable_fake_calibration_sources(tmp_path, monkeypatch)
    _fake_v2_score_factory(monkeypatch, {"expressions"}, [])
    stopped = study_v2.run_v2_stage("calibrate", Path("configs/recurrence-v2.toml"), root)
    assert stopped["status"] == "stopped_scientific"
    sealed = root / "data/sealed.jsonl"
    sealed.write_bytes(b"poisoned; any read raises")

    def forbidden_read(path: Path, description: str = ""):
        if Path(path) == sealed:
            raise AssertionError("sealed bytes must remain unopened after a terminal stop")
        return Path(path).read_bytes()

    monkeypatch.setattr(study_v2, "_read_bytes", lambda path, description="": forbidden_read() if Path(path) == sealed else Path(path).read_bytes())
    result = study_v2.run_v2_stage("report", Path("configs/recurrence-v2.toml"), root)
    assert result["next_stage"] == "complete"


def _enable_fake_calibration_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, object]]:
    config = replace(
        load_v2_config(Path("configs/recurrence-v2.toml")),
        data=replace(load_v2_config(Path("configs/recurrence-v2.toml")).data, output_root=str(tmp_path / "runs")),
    )
    _, inputs = _prepare_inputs()
    monkeypatch.setattr(study_v2, "load_v2_config", lambda _path: config)
    monkeypatch.setattr(study_v2, "verify_v2_sources", lambda _config, _root: inputs)
    monkeypatch.setattr(study_v2, "_load_v2_tokenizer", lambda _snapshot, _revision: _TinyTokenizer())
    root = Path(str(study_v2.run_v2_stage("prepare", Path("configs/recurrence-v2.toml"))["run_root"]))
    monkeypatch.setattr(study_v2, "_load_v2_model_measured", lambda *_args: (object(), _TinyTokenizer(), 0.0))
    return root, {"config": config, "inputs": inputs}


def _fake_v2_score_factory(monkeypatch: pytest.MonkeyPatch, eligible: set[str], calls: list[str]) -> None:
    def fake_score(_model, _tokenizer, examples, _mode, output_path, _signatures, **_kwargs):
        from ppmoe.recurrence.gates import summarize_exact_scores

        rows = []
        for index, row in enumerate(examples):
            calls.append(f"{row.split}:{row.family}:{row.structure_id}")
            if row.split == "dev":
                correct = index % 5 == 0
            else:
                family = row.family
                candidate = output_path.stem.rsplit("-c", 1)[1]
                chosen = candidate == "2" if family == "expressions" else candidate == "1"
                correct = family in eligible and chosen and index % 6 == 0
            rows.append({"example_id": row.id, "family": row.family, "correct": correct, "reason_code": "correct" if correct else "wrong_value"})
        raw = b"".join(canonical_json(row) for row in rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(raw)
        return {"validated_count": len(rows), "raw_sha256": hashlib.sha256(raw).hexdigest(), "summary": summarize_exact_scores(rows), "cuda_active_seconds": 0.0}

    monkeypatch.setattr(study_v2, "score_v2_examples", fake_score)


def test_calibration_scores_six_pilots_once_selects_first_eligible_and_candidate_one_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _enable_fake_calibration_sources(tmp_path, monkeypatch)
    calls: list[str] = []
    _fake_v2_score_factory(monkeypatch, {"expressions", "lists"}, calls)
    result = study_v2.run_v2_stage("calibrate", Path("configs/recurrence-v2.toml"), root)
    assert result["status"] == "passed", result.get("failure")
    assert len([call for call in calls if call.startswith("pilot:")]) == 180
    assert result["selected_difficulty"] == {"expressions": 2, "lists": 1, "graphs": 1}
    assert result["fallback_families"] == ["graphs"]
    assert result["next_stage"] == "frozen"
    assert (root / "human-review/review.md").is_file()
    assert not (root / "stages/frozen.json").exists()


def test_calibration_with_fewer_than_two_eligible_families_stops_before_dev_or_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _enable_fake_calibration_sources(tmp_path, monkeypatch)
    calls: list[str] = []
    _fake_v2_score_factory(monkeypatch, {"expressions"}, calls)
    result = study_v2.run_v2_stage("calibrate", Path("configs/recurrence-v2.toml"), root)
    assert result["status"] == "stopped_scientific", result
    assert result["next_stage"] == "report"
    assert len([call for call in calls if call.startswith("pilot:")]) == 180
    assert not any(call.startswith("dev:") for call in calls)
    assert not (root / "data/dev.jsonl").exists()
    assert not (root / "data/sealed.jsonl").exists()
    assert not (root / "human-review").exists()


def test_interrupted_open_phase_is_terminal_with_unknown_cuda_duration_and_no_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _enable_fake_calibration_sources(tmp_path, monkeypatch)
    prepare = study_v2.read_v2_stage(root / "stages/prepare.json", "prepare")
    attempt = study_v2._stage_record("calibration-attempt", "running", {
        "run_id": prepare["payload"]["run_id"],
        "config_sha256": hashlib.sha256(Path("configs/recurrence-v2.toml").read_bytes()).hexdigest(),
        "started_utc": "2026-09-27T00:00:00Z", "wall_started_monotonic": 1.0, "process_id": 99999999,
    }, {"prepare": prepare["stage_sha256"]})
    (root / "stages/calibration-attempt.json").write_bytes(canonical_json(attempt))
    study_v2.study_v1._append_ledger(root, {
        "event": "phase_started", "phase_id": "interrupted", "phase": "score", "process_id": 99999999,
        "monotonic_started": 1.0, "path": "scores/pilot-expressions-c1.jsonl", "score_key": "pilot-expressions-c1",
    })
    with pytest.raises(FileExistsError, match="already started"):
        study_v2.run_v2_stage("calibrate", Path("configs/recurrence-v2.toml"), root)
    report = study_v2.run_v2_stage("report", Path("configs/recurrence-v2.toml"), root)
    calibration = study_v2.read_v2_stage(root / "stages/calibrate.json", "calibrate")
    resources = calibration["payload"]["resources"]
    assert report["status"] == "failed_environment"
    assert resources["cuda_budget_state"] == "unknown_after_process_loss"
    assert resources["open_phase_count"] == 1
    assert not (root / "stages/frozen.json").exists()


def test_report_rejects_corrupted_pilot_score_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _ = _enable_fake_calibration_sources(tmp_path, monkeypatch)
    calls: list[str] = []
    _fake_v2_score_factory(monkeypatch, {"expressions"}, calls)
    study_v2.run_v2_stage("calibrate", Path("configs/recurrence-v2.toml"), root)
    score = root / "scores/pilot-expressions-c1.jsonl"
    score.write_bytes(score.read_bytes() + b" ")
    with pytest.raises(ValueError, match="score hash/count mismatch"):
        study_v2.run_v2_stage("report", Path("configs/recurrence-v2.toml"), root)


def test_exhausted_calibration_cuda_budget_is_a_resource_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(
        load_v2_config(Path("configs/recurrence-v2.toml")),
        data=replace(load_v2_config(Path("configs/recurrence-v2.toml")).data, output_root=str(tmp_path / "runs")),
        resources=replace(load_v2_config(Path("configs/recurrence-v2.toml")).resources, maximum_calibration_cuda_seconds=0),
    )
    _, inputs = _prepare_inputs()
    monkeypatch.setattr(study_v2, "load_v2_config", lambda _path: config)
    monkeypatch.setattr(study_v2, "verify_v2_sources", lambda _config, _root: inputs)
    monkeypatch.setattr(study_v2, "_load_v2_tokenizer", lambda _snapshot, _revision: _TinyTokenizer())
    root = Path(str(study_v2.run_v2_stage("prepare", Path("configs/recurrence-v2.toml"))["run_root"]))
    result = study_v2.run_v2_stage("calibrate", Path("configs/recurrence-v2.toml"), root)
    assert result["status"] == "stopped_resource"
    assert "CUDA-active time cap" in result["failure"]
    assert not list((root / "scores").glob("*.jsonl"))


def test_v2_cli_requires_full_run_id_for_calibration_and_emits_one_json_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import run_recurrence_v2 as cli

    config = load_v2_config(Path("configs/recurrence-v2.toml"))
    monkeypatch.setattr(cli, "load_v2_config", lambda _path: config)
    monkeypatch.setattr(cli, "run_v2_stage", lambda stage, config_path, root: {"status": "stopped_scientific", "stage": stage, "run_root": str(root)})
    with pytest.raises(SystemExit) as missing:
        cli.main(["--stage", "calibrate", "--config", "configs/recurrence-v2.toml"])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as malformed:
        cli.main(["--stage", "report", "--config", "configs/recurrence-v2.toml", "--run-id", "abc"])
    assert malformed.value.code == 2
    assert cli.main(["--stage", "calibrate", "--config", "configs/recurrence-v2.toml", "--run-id", "a" * 64]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "stopped_scientific"
    with pytest.raises(ValueError, match="implemented stages"):
        study_v2.run_v2_stage("frozen", Path("configs/recurrence-v2.toml"))


def test_partial_prepare_stage_write_does_not_claim_failure_was_durably_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        load_v2_config(Path("configs/recurrence-v2.toml")),
        data=replace(load_v2_config(Path("configs/recurrence-v2.toml")).data, output_root=str(tmp_path / "runs")),
    )
    _, inputs = _prepare_inputs()
    monkeypatch.setattr(study_v2, "load_v2_config", lambda _path: config)
    monkeypatch.setattr(study_v2, "verify_v2_sources", lambda _config, _root: inputs)
    monkeypatch.setattr(study_v2, "_load_v2_tokenizer", lambda _snapshot, _revision: _TinyTokenizer())
    original_write = study_v2._write_new

    def fail_after_partial_stage_write(path: Path, raw: bytes) -> None:
        if path.name == "prepare.json":
            path.write_bytes(raw[:24])
            raise OSError("simulated partial prepare stage write")
        original_write(path, raw)

    monkeypatch.setattr(study_v2, "_write_new", fail_after_partial_stage_write)
    with pytest.raises(RuntimeError, match="failed_environment.*durably"):
        study_v2.run_v2_stage("prepare", Path("configs/recurrence-v2.toml"))
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    stage_path = run_dirs[0] / "stages/prepare.json"
    assert stage_path.read_bytes() != b""
    assert len(stage_path.read_bytes()) == 24
