"""Authenticated prepare stage for the isolated recurrence V2 study."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

from .config_v1 import load_v1_config
from .config_v2 import V2Config, load_v2_config
from .data import Example, _example_payload, render_example
from .data_v2 import make_training_view_v2
from .provenance import canonical_json, sha256_file, sha256_payload
from . import study_v1


_FULL_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_V1_PILOT_KEYS = tuple(
    f"pilot-{family}-c{candidate}"
    for family in ("expressions", "lists", "graphs")
    for candidate in (1, 2)
)
_V2_CODE_FILES = (
    "src/ppmoe/recurrence/answer_v2.py",
    "src/ppmoe/recurrence/config_v2.py",
    "src/ppmoe/recurrence/data_v2.py",
    "src/ppmoe/recurrence/study_v2.py",
    "src/ppmoe/recurrence/data.py",
    "src/ppmoe/recurrence/provenance.py",
)
_V1_CODE_FILES = (
    "src/ppmoe/recurrence/answer_v1.py",
    "src/ppmoe/recurrence/data_v1.py",
    "src/ppmoe/recurrence/config_v1.py",
    "src/ppmoe/recurrence/v1_runtime.py",
    "src/ppmoe/recurrence/study_v1.py",
    "src/ppmoe/recurrence/model.py",
    "src/ppmoe/recurrence/evaluate.py",
    "src/ppmoe/recurrence/human_review.py",
    "src/ppmoe/recurrence/gates.py",
)


def verify_v2_sources(config: V2Config, project_root: Path) -> dict[str, object]:
    """Authenticate the V0 train/dev source and the six V1 pilot ID splits.

    This function is read-only. In particular, it never opens V0 test data or
    any V1 sealed split, score file, or report.
    """
    project_root = Path(project_root).resolve(strict=True)
    v1_config_path = project_root / "configs" / "recurrence-v1.toml"
    v1_config = load_v1_config(v1_config_path)
    if v1_config.source_run != config.source_run:
        raise ValueError("V1 and V2 configs must reference the same pinned V0 source run")
    if (v1_config.model_repo_id, v1_config.model_revision) != (config.model_repo_id, config.model_revision):
        raise ValueError("V1 and V2 configs must use the same pinned model and tokenizer revision")

    source = study_v1._verify_source(v1_config, v1_config_path, project_root)
    source_root = Path(source["source_root"])
    train_path = source_root / "data" / "train.jsonl"
    train_raw = _read_bytes(train_path, "V0 train split")
    if hashlib.sha256(train_raw).hexdigest() != source["signatures"]["train"]:
        raise ValueError("V0 train split changed after source authentication")
    train_records = _parse_rows(train_raw, "train", 8000, "V0")
    dev_path = source_root / "data" / "dev.jsonl"
    dev_raw = _read_bytes(dev_path, "V0 development split")
    if hashlib.sha256(dev_raw).hexdigest() != source["signatures"]["dev"]:
        raise ValueError("V0 dev split changed after source authentication")
    dev_records = _parse_rows(dev_raw, "dev", 600, "V0")
    train_ids = [str(row["id"]) for row in train_records]
    dev_ids = [str(row["id"]) for row in dev_records]
    v0_ids = set(train_ids) | set(dev_ids)
    if len(train_ids) != 8000 or len(dev_ids) != 600 or len(v0_ids) != 8600:
        raise ValueError("V0 train/dev rows must contain 8,600 distinct IDs in total")
    if source["train_count"] != 8000 or source["dev_count"] != 600:
        raise ValueError("authenticated V0 source counts differ from the registered 8,000/600")

    v1_root = _resolve(config.v1_pilot_root, project_root)
    study_v1._assert_no_reparse(v1_root)
    if not v1_root.is_dir():
        raise ValueError("registered V1 pilot run is missing")
    prepare = study_v1._read_stage(v1_root / "stages" / "prepare.json", "prepare")
    prepare_payload = _mapping(prepare.get("payload"), "V1 prepare payload")
    study_v1._validate_run_binding(v1_root, prepare, v1_config, v1_config_path, project_root)
    if prepare.get("status") != "passed":
        raise ValueError("V1 pilot source requires a passed prepare stage")
    if Path(str(prepare_payload.get("source_root", ""))).resolve(strict=True) != source_root.resolve(strict=True):
        raise ValueError("V1 prepare source root differs from the authenticated V0 source")
    expected_snapshot_relative = Path(source["snapshot"]).relative_to(source_root).as_posix()
    recorded_snapshot = Path(str(prepare_payload.get("snapshot_relative_path", ""))).as_posix()
    if recorded_snapshot != expected_snapshot_relative:
        raise ValueError("V1 prepare snapshot path differs from the authenticated pinned snapshot")
    if (
        prepare_payload.get("v0_train_count") != 8000
        or prepare_payload.get("v0_dev_count") != 600
        or prepare_payload.get("v0_train_dev_forbidden_id_count") != 8600
    ):
        raise ValueError("V1 prepare does not bind the registered V0 train/dev counts and ID exclusion")

    calibrate = study_v1._read_stage(v1_root / "stages" / "calibrate.json", "calibrate")
    if calibrate.get("status") not in {
        "passed", "stopped_scientific", "stopped_resource", "failed_environment",
    }:
        raise ValueError("V1 pilot calibration stage is not durably terminal")
    prerequisites = _mapping(calibrate.get("prerequisite_stage_sha256"), "V1 calibration prerequisites")
    if prerequisites != {"prepare": prepare.get("stage_sha256")}:
        raise ValueError("V1 calibration stage does not bind its prepare stage")
    calibration_payload = _mapping(calibrate.get("payload"), "V1 calibration payload")
    if calibration_payload.get("run_id") != prepare_payload.get("run_id"):
        raise ValueError("V1 calibration and prepare run IDs differ")
    prepared_signatures = _mapping(prepare_payload.get("signatures"), "V1 prepared signatures")
    calibrated_signatures = _mapping(calibration_payload.get("signatures"), "V1 calibration signatures")
    runtime_code_files = study_v1._runtime_code_files(project_root)
    study_v1._verify_calibration_runtime_code(v1_root, project_root)
    expected_calibrated = {**dict(prepared_signatures), "runtime_code": sha256_payload(runtime_code_files)}
    if dict(calibrated_signatures) != expected_calibrated:
        raise ValueError("V1 calibration signatures differ from its authenticated prepare and runtime code")

    pilot_scores = _mapping(calibration_payload.get("pilot_scores"), "V1 pilot score manifest")
    v1_ids: list[str] = []
    pilot_evidence: dict[str, dict[str, object]] = {}
    v1_data = v1_root / "data"
    for key in _V1_PILOT_KEYS:
        raw_entry = pilot_scores.get(key)
        entry = _mapping(raw_entry, f"V1 pilot binding {key}")
        if entry.get("split_count") != 30:
            raise ValueError(f"V1 pilot {key} count differs from the registered 30")
        split_path = v1_data / f"{key}.jsonl"
        sidecar_path = Path(f"{split_path}.manifest.json")
        _assert_safe_input(v1_root, split_path)
        _assert_safe_input(v1_root, sidecar_path)
        split_raw = _read_bytes(split_path, f"V1 pilot {key} split")
        sidecar_raw = _read_bytes(sidecar_path, f"V1 pilot {key} sidecar")
        split_hash, sidecar_hash = hashlib.sha256(split_raw).hexdigest(), hashlib.sha256(sidecar_raw).hexdigest()
        if split_hash != entry.get("split_sha256"):
            raise ValueError(f"V1 pilot {key} split hash differs from calibrate.json")
        if sidecar_hash != entry.get("split_manifest_sha256"):
            raise ValueError(f"V1 pilot {key} sidecar hash differs from calibrate.json")
        sidecar = _parse_json(sidecar_raw, f"V1 pilot {key} sidecar")
        if sidecar.get("sha256") != split_hash or sidecar.get("count") != 30:
            raise ValueError(f"V1 pilot {key} sidecar disagrees with its stage-bound split")
        ordered_ids = sidecar.get("ordered_ids")
        if not isinstance(ordered_ids, list) or len(ordered_ids) != 30 or any(
            not isinstance(identifier, str) or not identifier for identifier in ordered_ids
        ):
            raise ValueError(f"V1 pilot {key} ordered IDs are invalid")
        rows = _parse_rows(split_raw, "pilot", 30, f"V1 pilot {key}")
        split_ids = [str(row["id"]) for row in rows]
        if split_ids != ordered_ids:
            raise ValueError(f"V1 pilot {key} ordered IDs differ from its split sidecar")
        if len(set(split_ids)) != 30:
            raise ValueError(f"V1 pilot {key} contains duplicate IDs")
        v1_ids.extend(split_ids)
        pilot_evidence[key] = {"split_sha256": split_hash, "sidecar_sha256": sidecar_hash}

    if len(v1_ids) != 180 or len(set(v1_ids)) != 180:
        raise ValueError("V1 pilot IDs must contain 180 unique IDs across all six candidates")
    v1_id_set = set(v1_ids)
    if v0_ids.intersection(v1_id_set):
        raise ValueError("V1 pilot IDs overlap authenticated V0 train/dev IDs")

    signatures = {
        "v0_source_manifest": str(source["signatures"]["source"]),
        "v0_source_stage": sha256_file(source_root / "stages" / "source.json"),
        "v0_data_stage": sha256_file(source_root / "stages" / "data.json"),
        "v0_data_manifest": str(source["manifest_sha256"]),
        "v0_model_inventory": str(source["signatures"]["model"]),
        "tokenizer": str(source["signatures"]["tokenizer"]),
        "v0_train": str(source["signatures"]["train"]),
        "v0_dev": str(source["signatures"]["dev"]),
        "v0_train_order": sha256_payload({"ordered_ids": train_ids}),
        "v0_dev_order": sha256_payload({"ordered_ids": dev_ids}),
        "v1_prepare": str(prepare["stage_sha256"]),
        "v1_calibrate": str(calibrate["stage_sha256"]),
        "v1_pilot_ids": sha256_payload({"ids": sorted(v1_id_set)}),
    }
    for key, evidence in pilot_evidence.items():
        signatures[f"v1_{key}_split"] = str(evidence["split_sha256"])
        signatures[f"v1_{key}_sidecar"] = str(evidence["sidecar_sha256"])
    for relative in (*_V1_CODE_FILES, *_V2_CODE_FILES):
        signatures[f"code:{relative}"] = sha256_file(project_root / Path(relative))
    for name, value in signatures.items():
        if not isinstance(value, str) or _FULL_SHA256.fullmatch(value) is None:
            raise ValueError(f"invalid full SHA-256 for source signature {name}")

    if prepare_payload.get("source_manifest_sha256") != signatures["v0_source_manifest"]:
        raise ValueError("V1 prepare does not bind the authenticated V0 source manifest")
    if prepare_payload.get("data_manifest_sha256") != signatures["v0_data_manifest"]:
        raise ValueError("V1 prepare does not bind the authenticated V0 data manifest")
    return {
        "signatures": signatures,
        "source_root": source_root,
        "snapshot": Path(source["snapshot"]),
        "manifest_sha256": source["manifest_sha256"],
        "train_count": len(train_records),
        "dev_count": len(dev_ids),
        "train_records": tuple(train_records),
        "v0_train_ids": tuple(train_ids),
        "v0_dev_ids": tuple(dev_ids),
        "v0_train_dev_ids": v0_ids,
        "v1_pilot_ids": v1_id_set,
        "forbidden_ids": v0_ids | v1_id_set,
        "pilot_evidence": pilot_evidence,
    }


def run_v2_stage(stage: str, config_path: Path, run_root: Path | None = None) -> dict[str, object]:
    """Run the V2 prepare stage; calibration/report are implemented by later tasks."""
    if stage != "prepare":
        raise ValueError("unsupported V2 stage; only prepare is implemented")
    config_path = Path(config_path).resolve(strict=True)
    config = load_v2_config(config_path)
    project_root = Path(__file__).resolve().parents[3]
    config_bytes = _read_bytes(config_path, "V2 config")

    inputs = verify_v2_sources(config, project_root)
    train_records = inputs["train_records"]
    if not isinstance(train_records, tuple) or len(train_records) != 8000:
        raise ValueError("authenticated V0 training view must contain exactly 8,000 rows")
    train_rows = tuple(_example_from_record(row, "V0 train") for row in train_records)
    tokenizer = _load_v2_tokenizer(Path(str(inputs["snapshot"])), config.model_revision)
    view, train_raw, train_manifest = _build_training_view(
        train_rows, tokenizer, config.model_revision, str(inputs["signatures"]["tokenizer"]), config,
    )
    signatures = dict(_mapping(inputs["signatures"], "authenticated V2 source signatures"))
    signatures["config"] = hashlib.sha256(config_bytes).hexdigest()
    signatures["train_view"] = str(train_manifest["sha256"])
    signatures["train_view_manifest"] = sha256_payload(train_manifest)
    run_id = _derive_run_id(config_bytes, signatures)
    output_root = _resolve(config.data.output_root, project_root)
    expected_root = output_root / run_id
    if run_root is not None and Path(run_root).absolute() != expected_root.absolute():
        raise ValueError("prepare run root must equal configured output root plus deterministic run ID")
    payload: dict[str, object] = {
        "run_id": run_id,
        "run_root": str(expected_root),
        "status": "passed",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "signatures": signatures,
        "source_root": str(inputs["source_root"]),
        "snapshot_relative_path": str(Path(inputs["snapshot"]).relative_to(Path(inputs["source_root"]))),
        "source_manifest_sha256": signatures["v0_source_manifest"],
        "data_manifest_sha256": inputs["manifest_sha256"],
        "v0_train_count": 8000,
        "v0_dev_count": 600,
        "v0_train_dev_forbidden_id_count": 8600,
        "v1_pilot_id_count": 180,
        "forbidden_id_count": len(inputs["forbidden_ids"]),
        "train_view_sha256": train_manifest["sha256"],
        "train_view_manifest_sha256": signatures["train_view_manifest"],
        "train_view_count": len(view),
        "created_utc": _utc_now(),
    }
    _assert_no_reparse(expected_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse(expected_root)
    expected_root.mkdir(parents=False, exist_ok=False)
    try:
        for directory in (expected_root / "stages", expected_root / "data", expected_root / "scores", expected_root / "resources"):
            _assert_no_reparse(directory)
            directory.mkdir()
        _write_new(expected_root / "data" / "train-view.jsonl", train_raw)
        _write_new(expected_root / "data" / "train-view.jsonl.manifest.json", canonical_json(train_manifest))
        record = _stage_record("prepare", "passed", payload, {})
        _write_new(expected_root / "stages" / "prepare.json", canonical_json(record))
    except BaseException as error:
        failed_payload = {**payload, "status": "failed_environment", "failure": f"{type(error).__name__}: {error}"}
        try:
            (expected_root / "stages").mkdir(exist_ok=True)
            failure_record = _stage_record("prepare", "failed_environment", failed_payload, {})
            _write_new(expected_root / "stages" / "prepare.json", canonical_json(failure_record))
        except BaseException:
            pass
        return {
            "run_id": run_id, "status": "failed_environment", "next_stage": "report",
            "run_root": str(expected_root), "failure": str(error),
        }
    return {
        "run_id": run_id, "status": "passed", "next_stage": "calibrate",
        "run_root": str(expected_root), "stage_sha256": record["stage_sha256"],
    }


def _build_training_view(
    train_rows: tuple[Example, ...], tokenizer: object, revision: str,
    tokenizer_sha256: str, config: V2Config,
) -> tuple[tuple[Example, ...], bytes, dict[str, object]]:
    view = make_training_view_v2(train_rows)
    encoded_rows: list[bytes] = []
    maximum_prompt = 0
    maximum_sequence = 0
    for row in view:
        rendered = render_example(tokenizer, row)
        prompt_length = len(rendered.prompt_input_ids)
        sequence_length = len(rendered.train_input_ids)
        if prompt_length + config.max_new_tokens > config.max_sequence_tokens:
            raise ValueError("V2 training prompt leaves insufficient room for the registered generation budget")
        maximum_prompt = max(maximum_prompt, prompt_length)
        maximum_sequence = max(maximum_sequence, sequence_length)
        encoded_rows.append(canonical_json(_example_payload(row)))
    raw = b"".join(encoded_rows)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "split": "train",
        "view": "v2-list-prompt-repair",
        "count": len(view),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "ordered_ids": [row.id for row in view],
        "tokenizer_revision": revision,
        "tokenizer_sha256": tokenizer_sha256,
        "max_prompt_tokens": maximum_prompt,
        "max_prompt_plus_answer_tokens": maximum_sequence,
        "max_prompt_plus_generation_tokens": config.max_new_tokens + maximum_prompt,
    }
    return view, raw, manifest


def read_v2_stage(path: Path, expected_stage: str) -> dict[str, object]:
    """Read and authenticate one V2 stage record."""
    record = _read_json(Path(path))
    if record.get("schema_version") != 1 or record.get("stage") != expected_stage:
        raise ValueError(f"{expected_stage} stage record has an unsupported schema")
    digest = record.get("stage_sha256")
    unsigned = {key: value for key, value in record.items() if key != "stage_sha256"}
    if not isinstance(digest, str) or sha256_payload(unsigned) != digest:
        raise ValueError(f"{expected_stage} stage hash mismatch")
    return record


def validate_v2_prepare_binding(
    record: Mapping[str, object], config_bytes: bytes, signatures: Mapping[str, str],
) -> None:
    """Reject continuation when config bytes or any prepared signature drift."""
    if record.get("stage") != "prepare" or record.get("status") != "passed":
        raise ValueError("V2 continuation requires a passed prepare stage")
    payload = _mapping(record.get("payload"), "V2 prepare payload")
    expected = dict(signatures)
    expected["config"] = hashlib.sha256(config_bytes).hexdigest()
    frozen = _mapping(payload.get("signatures"), "V2 prepared signatures")
    if dict(frozen) != expected:
        raise ValueError("V2 source, code, config, or train-view signature changed after prepare")
    if payload.get("config_sha256") != expected["config"]:
        raise ValueError("V2 config hash changed after prepare")
    if payload.get("run_id") != _derive_run_id(config_bytes, expected):
        raise ValueError("V2 prepare run ID does not match its config and frozen signatures")


def _derive_run_id(config_bytes: bytes, signatures: Mapping[str, str]) -> str:
    if any(not isinstance(value, str) or _FULL_SHA256.fullmatch(value) is None for value in signatures.values()):
        raise ValueError("run signatures must be full SHA-256 values")
    return sha256_payload({
        "config_bytes_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "signatures": dict(signatures),
    })


def _stage_record(stage: str, status: str, payload: Mapping[str, object], prerequisites: Mapping[str, str]) -> dict[str, object]:
    unsigned = {
        "schema_version": 1, "stage": stage, "status": status,
        "prerequisite_stage_sha256": dict(prerequisites), "payload": dict(payload),
    }
    return {**unsigned, "stage_sha256": sha256_payload(unsigned)}


def _load_v2_tokenizer(snapshot: Path, revision: str) -> object:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        snapshot, revision=revision, local_files_only=True, trust_remote_code=False,
    )


def _example_from_record(row: Mapping[str, object], description: str) -> Example:
    required = {"id", "family", "split", "structure_id", "prompt", "answer", "latent_spec"}
    if set(row) != required:
        raise ValueError(f"{description} row has an unexpected schema")
    try:
        return Example(**row)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} row cannot be converted to an Example") from error


def _parse_rows(raw: bytes, split: str, expected_count: int, description: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{description} {split} row {line_number} is invalid JSON") from error
        if not isinstance(value, dict) or value.get("split") != split:
            raise ValueError(f"{description} {split} row {line_number} has invalid schema or split")
        identifier = value.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"{description} {split} IDs must be non-empty and unique")
        seen.add(identifier)
        rows.append(value)
    if len(rows) != expected_count:
        raise ValueError(f"{description} {split} row count is {len(rows)}, expected {expected_count}")
    return rows


def _assert_safe_input(root: Path, path: Path) -> None:
    study_v1._assert_no_reparse(path)
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError("V1 pilot evidence path escapes its registered run") from error


def _read_bytes(path: Path, description: str) -> bytes:
    study_v1._assert_no_reparse(path)
    try:
        return Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"required {description} is unreadable") from error


def _read_json(path: Path) -> dict[str, object]:
    raw = _read_bytes(path, f"V2 stage file {path}")
    return _parse_json(raw, f"V2 stage file {path}")


def _parse_json(raw: bytes, description: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).absolute()


def _assert_no_reparse(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise ValueError(f"symlink or reparse point is not allowed: {current}")


def _write_new(path: Path, raw: bytes) -> None:
    _assert_no_reparse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
