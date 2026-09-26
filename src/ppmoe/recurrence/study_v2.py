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

import torch

from .config_v1 import load_v1_config
from .config_v2 import V2Config, load_v2_config
from .data import FAMILIES, Example, _example_payload, render_example
from .data_v2 import freeze_v2_split, make_pilot_v2, make_selected_splits_v2, make_training_view_v2
from .gates import CudaCostLedger, is_informative_family, summarize_exact_scores
from .provenance import canonical_json, sha256_file, sha256_payload
from . import study_v1
from .v2_runtime import ScoreExecutionError, load_v2_model, score_v2_examples


_FULL_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_V1_PILOT_KEYS = tuple(
    f"pilot-{family}-c{candidate}"
    for family in ("expressions", "lists", "graphs")
    for candidate in (1, 2)
)
_V2_CODE_FILES = (
    "scripts/run_recurrence_v2.py",
    "src/ppmoe/recurrence/answer_v2.py",
    "src/ppmoe/recurrence/config_v2.py",
    "src/ppmoe/recurrence/data_v2.py",
    "src/ppmoe/recurrence/v2_runtime.py",
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
    """Run exactly one V2 stage; later model-adaptation stages are unavailable."""
    if stage not in {"prepare", "calibrate", "report"}:
        raise ValueError("unsupported V2 stage; implemented stages are prepare, calibrate, report")
    config_path = Path(config_path).resolve(strict=True)
    config = load_v2_config(config_path)
    project_root = Path(__file__).resolve().parents[3]
    if stage != "prepare":
        if run_root is None:
            raise ValueError(f"{stage} requires a run root")
        root = Path(run_root).absolute()
        _assert_no_reparse(root)
        if stage == "calibrate":
            return _calibrate_v2(root, config, config_path, project_root)
        return _report_v2(root, config, config_path, project_root)
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
        except BaseException as record_error:
            raise RuntimeError(
                "failed_environment stage could not be durably recorded after "
                f"prepare failure: {record_error}"
            ) from record_error
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


def _calibrate_v2(root: Path, config: V2Config, config_path: Path, project_root: Path) -> dict[str, object]:
    prepare_path = root / "stages" / "prepare.json"
    prepare = read_v2_stage(prepare_path, "prepare")
    if prepare.get("status") != "passed":
        raise ValueError("calibrate requires a passed prepare stage")
    if (root / "stages" / "calibrate.json").exists() or (root / "stages" / "calibration-attempt.json").exists():
        raise FileExistsError("calibration has already started; retries require a new run identity")
    fresh = _validate_v2_prepared_run(root, prepare, config, config_path, project_root)
    signatures = dict(_mapping(_mapping(prepare.get("payload"), "prepare payload").get("signatures"), "prepared signatures"))
    runtime_files = _runtime_code_files(project_root)
    signatures["runtime_code"] = sha256_payload(runtime_files)
    payload = _mapping(prepare.get("payload"), "prepare payload")
    snapshot = Path(str(fresh["snapshot"]))
    study_v1._write_attempt_marker(root, prepare, config_path)
    started = time.monotonic()
    deadline = started + float(config.resources.maximum_calibration_wall_seconds)
    cost_ledger = _load_v2_cost_ledger(root, float(config.resources.maximum_study_cuda_seconds))
    score_manifest: dict[str, object] = {}
    pilot_table: list[dict[str, object]] = []
    active_phase: tuple[str, str] | None = None
    active_score: Path | None = None
    try:
        _budget_check_v2(root, cost_ledger, config, started)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        phase_id = study_v1._start_phase(root, "model_load", metadata={"run_id": payload["run_id"]})
        active_phase = (phase_id, "calibration-model-load")
        try:
            model, tokenizer, load_seconds = _load_v2_model_measured(snapshot, config.model_revision)
        except MeasuredV2PhaseFailure as error:
            _record_v2_phase(root, cost_ledger, phase_id, "calibration-model-load", error.cuda_active_seconds, "failed_environment")
            active_phase = None
            raise error.cause
        except Exception:
            _record_v2_phase(root, cost_ledger, phase_id, "calibration-model-load", 0.0, "failed_environment")
            active_phase = None
            raise
        _record_v2_phase(root, cost_ledger, phase_id, "calibration-model-load", load_seconds, "completed")
        active_phase = None
        forbidden = set(fresh["forbidden_ids"])
        examples_by_key: dict[str, tuple[Example, ...]] = {}
        for family in FAMILIES:
            for candidate in (1, 2):
                _budget_check_v2(root, cost_ledger, config, started)
                examples = make_pilot_v2(config, family, candidate, forbidden)
                examples_by_key[f"pilot-{family}-c{candidate}"] = examples
                forbidden.update(row.id for row in examples)
                split_path = root / "data" / f"pilot-{family}-c{candidate}.jsonl"
                manifest = freeze_v2_split(
                    split_path, examples, tokenizer, revision=config.model_revision,
                    tokenizer_sha256=str(signatures["tokenizer"]),
                )
                _verify_v2_split(split_path, manifest, examples)
                key = f"pilot-{family}-c{candidate}"
                entry = {
                    "split_sha256": manifest["sha256"],
                    "split_manifest_sha256": sha256_file(Path(f"{split_path}.manifest.json")),
                    "split_count": len(examples),
                }
                score_manifest[key] = entry
                study_v1._append_ledger(root, {"event": "split_frozen", "score_key": key, "split_path": str(split_path.relative_to(root)), **entry})
                score_path = root / "scores" / f"{key}.jsonl"
                active_score = score_path
                phase_id = study_v1._start_phase(root, "score", str(score_path.relative_to(root)), {"score_key": key, "ordered_ids": [row.id for row in examples]})
                active_phase = (phase_id, "calibration-pilot")
                try:
                    result = score_v2_examples(model, tokenizer, examples, "single", score_path, signatures, deadline=deadline)
                except ScoreExecutionError as error:
                    outcome = "stopped_resource" if error.deadline_exceeded else "failed_environment"
                    _record_v2_phase(root, cost_ledger, phase_id, "calibration-pilot", error.cuda_active_seconds, outcome, {"row_count": error.row_count, "raw_sha256": error.raw_sha256})
                    active_phase = None
                    raise
                _record_v2_phase(root, cost_ledger, phase_id, "calibration-pilot", float(result["cuda_active_seconds"]), "completed", {"row_count": result["validated_count"], "raw_sha256": result["raw_sha256"]})
                active_phase = None
                summary = _validate_v2_score_file(score_path, examples, result)
                score_manifest[key] = {**entry, "score_sha256": result["raw_sha256"], "score_count": result["validated_count"], "summary": summary}
                family_result = _mapping(_mapping(summary.get("families"), "pilot family summary").get(family), "pilot result")
                accuracy, invalid = float(family_result["accuracy"]), float(family_result["invalid_rate"])
                pilot_table.append({
                    "family": family, "candidate": candidate, "record_count": int(summary["record_count"]),
                    "correct": int(summary["correct"]), "invalid_format": int(summary["invalid_format"]),
                    "accuracy": accuracy, "invalid_rate": invalid,
                    "eligible": is_informative_family(accuracy, invalid),
                })
                _budget_check_v2(root, cost_ledger, config, started)
        choices, fallback = study_v1._select_pilot_difficulties(pilot_table)
        eligible_families = sorted(family for family in FAMILIES if any(
            row["family"] == family and row["eligible"] is True for row in pilot_table
        ))
        if len(eligible_families) < 2:
            result_payload = {
                "run_id": payload["run_id"], "status": "stopped_scientific", "pilot_table": pilot_table,
                "pilot_scores": score_manifest, "eligible_families": eligible_families,
                "selected_difficulty": {}, "fallback_families": list(FAMILIES), "dev": None, "sealed": None,
                "pilot_eligibility_state": "complete",
                "resources": _v2_resource_summary(root, cost_ledger, config, started),
                "signatures": signatures, "runtime_code_files": runtime_files, "next_stage": "report",
            }
            return _finish_v2_calibration(root, result_payload, "stopped_scientific", cost_ledger)
        for row in pilot_table:
            row["selected"] = row["candidate"] == choices[str(row["family"])]
            row["selection_kind"] = "selected_first_eligible" if row["family"] not in fallback and row["selected"] else "prespecified_candidate_1_fallback" if row["family"] in fallback and row["selected"] else "not_selected"
        splits = make_selected_splits_v2(config, choices, forbidden)
        dev_examples, sealed_examples = splits["dev"], splits["sealed"]
        dev_path = root / "data" / "dev.jsonl"
        dev_manifest = freeze_v2_split(dev_path, dev_examples, tokenizer, revision=config.model_revision, tokenizer_sha256=str(signatures["tokenizer"]))
        sealed_path = root / "data" / "sealed.jsonl"
        sealed_manifest = freeze_v2_split(sealed_path, sealed_examples, tokenizer, revision=config.model_revision, tokenizer_sha256=str(signatures["tokenizer"]))
        sealed_meta = {
            "sha256": sealed_manifest["sha256"], "manifest_sha256": sha256_file(Path(f"{sealed_path}.manifest.json")),
            "count": len(sealed_examples),
        }
        dev_entry = {
            "split_sha256": dev_manifest["sha256"],
            "split_manifest_sha256": sha256_file(Path(f"{dev_path}.manifest.json")), "split_count": len(dev_examples),
        }
        score_manifest["dev-f0"] = dev_entry
        study_v1._append_ledger(root, {"event": "split_frozen", "score_key": "dev-f0", "split_path": str(dev_path.relative_to(root)), **dev_entry})
        score_path = root / "scores" / "dev-f0.jsonl"
        active_score = score_path
        phase_id = study_v1._start_phase(root, "score", str(score_path.relative_to(root)), {"score_key": "dev-f0", "ordered_ids": [row.id for row in dev_examples]})
        active_phase = (phase_id, "calibration-dev")
        try:
            result = score_v2_examples(model, tokenizer, dev_examples, "single", score_path, signatures, deadline=deadline)
        except ScoreExecutionError as error:
            outcome = "stopped_resource" if error.deadline_exceeded else "failed_environment"
            _record_v2_phase(root, cost_ledger, phase_id, "calibration-dev", error.cuda_active_seconds, outcome, {"row_count": error.row_count, "raw_sha256": error.raw_sha256})
            active_phase = None
            raise
        _record_v2_phase(root, cost_ledger, phase_id, "calibration-dev", float(result["cuda_active_seconds"]), "completed", {"row_count": result["validated_count"], "raw_sha256": result["raw_sha256"]})
        active_phase = None
        summary = _validate_v2_score_file(score_path, dev_examples, result)
        score_manifest["dev-f0"] = {**dev_entry, "score_sha256": result["raw_sha256"], "score_count": result["validated_count"], "summary": summary}
        _budget_check_v2(root, cost_ledger, config, started)
        gate = _development_gate_v2(summary)
        review = _export_v2_review(score_path, dev_examples, root)
        _budget_check_v2(root, cost_ledger, config, started)
        status = "passed" if gate["passes"] else "stopped_scientific"
        result_payload = {
            "run_id": payload["run_id"], "status": status, "pilot_table": pilot_table,
            "pilot_scores": score_manifest, "eligible_families": eligible_families,
            "selected_difficulty": choices, "fallback_families": fallback,
            "pilot_eligibility_state": "complete",
            "dev": {**dev_entry, "scores": summary, "gate": gate}, "sealed": sealed_meta,
            "human_review": review, "resources": _v2_resource_summary(root, cost_ledger, config, started),
            "signatures": signatures, "runtime_code_files": runtime_files,
            "next_stage": "frozen" if status == "passed" else "report",
        }
        return _finish_v2_calibration(root, result_payload, status, cost_ledger)
    except ResourceLimitReachedV2 as error:
        failure_status, failure = "stopped_resource", str(error)
    except Exception as error:
        failure_status, failure = "failed_environment", f"{type(error).__name__}: {error}"
        if isinstance(error, ScoreExecutionError) and error.deadline_exceeded:
            failure_status = "stopped_resource"
        if active_score is not None and active_score.is_file():
            key = "dev-f0" if active_score.stem == "dev-f0" else active_score.stem
            if key in score_manifest:
                score_manifest[key] = {
                    **_mapping(score_manifest[key], "partial score evidence"),
                    "score_sha256": sha256_file(active_score), "score_count": _raw_line_count(active_score), "partial": True,
                }
        if active_phase is not None:
            phase_id, phase_name = active_phase
            elapsed = error.cuda_active_seconds if isinstance(error, ScoreExecutionError) else 0.0
            outcome = "stopped_resource" if failure_status == "stopped_resource" else "failed_environment"
            _record_v2_phase(root, cost_ledger, phase_id, phase_name, elapsed, outcome)
    failure_payload = {
        "run_id": payload["run_id"], "status": failure_status, "failure": failure,
        "pilot_table": pilot_table, "pilot_scores": score_manifest,
        "pilot_eligibility_state": _pilot_eligibility_state(score_manifest),
        "dev": None, "sealed": None, "signatures": signatures, "runtime_code_files": runtime_files,
        "resources": _v2_resource_summary(root, cost_ledger, config, started), "next_stage": "report",
    }
    return _finish_v2_calibration(root, failure_payload, failure_status, cost_ledger)


class ResourceLimitReachedV2(RuntimeError):
    """Registered V2 calibration or artifact budget was reached."""


class MeasuredV2PhaseFailure(RuntimeError):
    def __init__(self, cause: BaseException, cuda_active_seconds: float):
        super().__init__(str(cause))
        self.cause = cause
        self.cuda_active_seconds = cuda_active_seconds


def _validate_v2_prepared_run(root: Path, prepare: Mapping[str, object], config: V2Config, config_path: Path, project_root: Path) -> dict[str, object]:
    payload = _mapping(prepare.get("payload"), "prepare payload")
    if payload.get("config_sha256") != sha256_file(config_path):
        raise ValueError("V2 config hash changed after prepare")
    signatures = dict(_mapping(payload.get("signatures"), "prepared signatures"))
    run_id = _derive_run_id(_read_bytes(config_path, "V2 config"), signatures)
    if payload.get("run_id") != run_id or root.resolve(strict=True) != (_resolve(config.data.output_root, project_root) / run_id).resolve(strict=True):
        raise ValueError("V2 run root or deterministic run ID does not match prepare")
    if Path(str(payload.get("run_root"))).resolve(strict=True) != root.resolve(strict=True):
        raise ValueError("V2 prepare payload run root changed")
    fresh = verify_v2_sources(config, project_root)
    train_path = _safe_v2_run_file(root, Path("data/train-view.jsonl"))
    manifest_path = _safe_v2_run_file(root, Path("data/train-view.jsonl.manifest.json"))
    train_raw, manifest_raw = _read_bytes(train_path, "prepared V2 train-view"), _read_bytes(manifest_path, "prepared V2 train-view manifest")
    manifest = _parse_json(manifest_raw, "prepared V2 train-view manifest")
    train_hash = hashlib.sha256(train_raw).hexdigest()
    if train_hash != signatures.get("train_view") or train_hash != payload.get("train_view_sha256") or manifest.get("sha256") != train_hash:
        raise ValueError("prepared train-view raw hash changed")
    if sha256_payload(manifest) != signatures.get("train_view_manifest") or manifest_raw != canonical_json(manifest):
        raise ValueError("prepared train-view manifest hash or canonical bytes changed")
    rows = _parse_rows(train_raw, "train", 8000, "prepared V2")
    ids = [str(row["id"]) for row in rows]
    source_ids = list(fresh.get("v0_train_ids", ()))
    if len(ids) != 8000 or ids != source_ids or manifest.get("ordered_ids") != ids or manifest.get("count") != 8000:
        raise ValueError("prepared train-view must contain the exact 8,000 ordered source IDs")
    if manifest.get("tokenizer_revision") != config.model_revision or manifest.get("tokenizer_sha256") != signatures.get("tokenizer"):
        raise ValueError("prepared train-view tokenizer revision/hash changed")
    expected = dict(fresh["signatures"])
    expected.update({
        "config": hashlib.sha256(_read_bytes(config_path, "V2 config")).hexdigest(),
        "train_view": train_hash,
        "train_view_manifest": sha256_payload(manifest),
    })
    validate_v2_prepare_binding(prepare, _read_bytes(config_path, "V2 config"), expected)
    if signatures != expected:
        raise ValueError("V2 source, scorer, runtime, config, or train-view signature changed after prepare")
    return fresh


def _load_v2_model_measured(snapshot: Path, revision: str) -> tuple[object, object, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("V2 calibration requires CUDA")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        model, tokenizer = load_v2_model(snapshot, revision)
    except BaseException as error:
        end.record()
        end.synchronize()
        raise MeasuredV2PhaseFailure(error, float(start.elapsed_time(end)) / 1000.0) from error
    end.record()
    end.synchronize()
    return model, tokenizer, float(start.elapsed_time(end)) / 1000.0


def _record_v2_phase(root: Path, ledger: CudaCostLedger, phase_id: str, stage: str, seconds: float, outcome: str, evidence: Mapping[str, object] | None = None) -> None:
    ledger.record(phase_id, stage, seconds, outcome)
    study_v1._complete_phase(root, phase_id, stage, seconds, outcome, evidence)


def _load_v2_cost_ledger(root: Path, cap: float) -> CudaCostLedger:
    return study_v1._load_cost_ledger(root, cap)


def _v2_calibration_seconds(ledger: CudaCostLedger) -> float:
    return sum(entry.elapsed_seconds for entry in ledger.entries if entry.stage.startswith("calibration-"))


def _budget_check_v2(root: Path, ledger: CudaCostLedger, config: V2Config, started: float) -> None:
    if time.monotonic() - started >= config.resources.maximum_calibration_wall_seconds:
        raise ResourceLimitReachedV2("calibration wall-clock cap reached")
    if _v2_calibration_seconds(ledger) >= config.resources.maximum_calibration_cuda_seconds:
        raise ResourceLimitReachedV2("calibration CUDA-active time cap reached")
    if ledger.total_seconds >= config.resources.maximum_study_cuda_seconds:
        raise ResourceLimitReachedV2("whole-study CUDA-active time cap reached")
    if study_v1._tree_bytes(root) >= config.resources.maximum_artifact_bytes:
        raise ResourceLimitReachedV2("new artifact byte cap reached")


def _v2_resource_summary(root: Path, ledger: CudaCostLedger, config: V2Config, started: float) -> dict[str, object]:
    return {
        "calibration_cuda_active_seconds": _v2_calibration_seconds(ledger),
        "study_cuda_active_seconds": ledger.total_seconds,
        "calibration_wall_seconds": max(0.0, time.monotonic() - started),
        "maximum_calibration_cuda_seconds": config.resources.maximum_calibration_cuda_seconds,
        "maximum_calibration_wall_seconds": config.resources.maximum_calibration_wall_seconds,
        "maximum_study_cuda_seconds": config.resources.maximum_study_cuda_seconds,
        "artifact_bytes": study_v1._tree_bytes(root),
        "maximum_artifact_bytes": config.resources.maximum_artifact_bytes,
        "cuda": study_v1._cuda_resource_snapshot(),
    }


def _verify_v2_split(path: Path, manifest: Mapping[str, object], examples: tuple[Example, ...]) -> None:
    raw_path = _safe_v2_run_file(path.parents[1], path.relative_to(path.parents[1]))
    if sha256_file(raw_path) != manifest.get("sha256") or _raw_line_count(raw_path) != len(examples):
        raise ValueError("V2 split raw hash/count differs from freeze manifest")
    sidecar = _safe_v2_run_file(path.parents[1], Path(f"{path.relative_to(path.parents[1])}.manifest.json"))
    if sha256_file(sidecar) != sha256_payload(manifest):
        raise ValueError("V2 split sidecar hash differs from freeze manifest")
    rows = _parse_rows(_read_bytes(raw_path, "V2 split"), examples[0].split, len(examples), "V2")
    ids = [str(row["id"]) for row in rows]
    if ids != [row.id for row in examples] or manifest.get("ordered_ids") != ids:
        raise ValueError("V2 split ordered IDs differ from generated examples")


def _validate_v2_score_file(path: Path, examples: tuple[Example, ...], result: Mapping[str, object]) -> dict[str, object]:
    if sha256_file(path) != result.get("raw_sha256") or result.get("validated_count") != len(examples):
        raise ValueError("V2 score file hash/count differs from runtime validation")
    rows = _read_jsonl(path)
    ids = [row.get("example_id") for row in rows]
    if ids != [row.id for row in examples] or len(set(ids)) != len(examples):
        raise ValueError("V2 score IDs do not match frozen ordered split")
    summary = summarize_exact_scores(rows)
    if summary != result.get("summary"):
        raise ValueError("V2 score summary differs from raw records")
    return summary


def _development_gate_v2(summary: Mapping[str, object]) -> dict[str, object]:
    families = _mapping(summary.get("families"), "development family summaries")
    family_set = set(families) == set(FAMILIES)
    informative = []
    for family, raw in families.items():
        row = _mapping(raw, "development family result")
        if is_informative_family(float(row["accuracy"]), float(row["invalid_rate"])):
            informative.append(family)
    macro, invalid = float(summary.get("macro_accuracy", -1)), float(summary.get("global_invalid_rate", 1))
    return {
        "family_summaries": dict(families), "macro_accuracy": macro, "global_invalid_rate": invalid,
        "informative_families": sorted(informative), "macro_pass": 0.05 < macro < 0.95,
        "invalid_pass": invalid <= 0.10, "family_set_pass": family_set,
        "minimum_informative_families_pass": len(informative) >= 2,
        "passes": family_set and 0.05 < macro < 0.95 and invalid <= 0.10 and len(informative) >= 2,
    }


def _export_v2_review(score_path: Path, examples: tuple[Example, ...], root: Path) -> dict[str, object]:
    from .human_review import export_review
    import random

    rows = _read_jsonl(score_path)
    by_id = {example.id: example for example in examples}
    rng = random.Random(9158)
    sampled = []
    for family in FAMILIES:
        members = [row for row in rows if row.get("family") == family]
        sampled.extend(rng.sample(members, 10))
    records = []
    for row in sampled:
        example = by_id[str(row["example_id"])]
        records.append({**row, "arm": "F0", "split": "dev", "prompt": example.prompt, "answer": example.answer})
    output = export_review(records, root / "human-review", seed=9158)
    return {
        "sample_count": len(records), "seed": 9158,
        "review_sha256": sha256_file(Path(output["review_path"])),
        "manifest_sha256": sha256_file(Path(output["manifest_path"])),
        "review_path": str(Path(output["review_path"]).relative_to(root)),
        "manifest_path": str(Path(output["manifest_path"]).relative_to(root)),
        "reviewer": "unassigned",
    }


def _finish_v2_calibration(root: Path, payload: Mapping[str, object], status: str, ledger: CudaCostLedger) -> dict[str, object]:
    if (root / "stages/calibrate.json").exists():
        raise FileExistsError("calibrate stage already exists")
    value = dict(payload)
    value["cuda_cost_ledger_sha256"] = study_v1._optional_sha256(root / "resources/cuda-cost.jsonl")
    value["cuda_cost_ledger_total_seconds"] = ledger.total_seconds
    value["calibration_cuda_active_seconds"] = _v2_calibration_seconds(ledger)
    prepare = read_v2_stage(root / "stages/prepare.json", "prepare")
    record = _stage_record("calibrate", status, value, {"prepare": str(prepare["stage_sha256"])})
    _write_new(root / "stages/calibrate.json", canonical_json(record))
    return {**value, "run_id": value.get("run_id"), "status": status, "run_root": str(root), "stage_sha256": record["stage_sha256"], "next_stage": value.get("next_stage", "report")}


def _pilot_eligibility_state(evidence: Mapping[str, object]) -> str:
    """Eligibility is complete only after all six pilots have complete score files."""
    for family in FAMILIES:
        for candidate in (1, 2):
            entry = evidence.get(f"pilot-{family}-c{candidate}")
            if not isinstance(entry, Mapping) or entry.get("partial") is True or entry.get("score_sha256") is None:
                return "incomplete"
    return "complete"


def _safe_v2_run_file(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("V2 run evidence path is unsafe")
    path = root / relative
    _assert_no_reparse(path)
    resolved = path.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    if not resolved.is_file():
        raise ValueError("V2 run evidence must be a regular file")
    return resolved


def _report_v2(root: Path, config: V2Config, config_path: Path, project_root: Path) -> dict[str, object]:
    prepare = read_v2_stage(root / "stages/prepare.json", "prepare")
    _validate_v2_prepared_run(root, prepare, config, config_path, project_root)
    calibration_path = root / "stages/calibrate.json"
    if not calibration_path.exists():
        _recover_v2_interrupted(root, prepare, config, config_path)
        calibration = read_v2_stage(calibration_path, "calibrate")
    else:
        calibration = read_v2_stage(calibration_path, "calibrate")
    prerequisites = _mapping(calibration.get("prerequisite_stage_sha256"), "V2 calibration prerequisites")
    if prerequisites != {"prepare": prepare.get("stage_sha256")}:
        raise ValueError("V2 calibration stage does not bind prepare")
    if calibration.get("status") not in {"passed", "stopped_scientific", "stopped_resource", "failed_environment"}:
        raise ValueError("report requires a terminal V2 calibration stage")
    cal_payload = _mapping(calibration.get("payload"), "V2 calibration payload")
    if cal_payload.get("run_id") != _mapping(prepare.get("payload"), "prepare payload").get("run_id"):
        raise ValueError("V2 calibration run ID differs from prepare")
    evidence_keys = set(_mapping(cal_payload.get("pilot_scores", {}), "V2 calibration evidence"))
    expected_pilots = {f"pilot-{family}-c{candidate}" for family in FAMILIES for candidate in (1, 2)}
    if calibration.get("status") in {"passed", "stopped_scientific"} and not expected_pilots.issubset(evidence_keys):
        raise ValueError("terminal scientific calibration must bind all six pilot splits and scores")
    if cal_payload.get("dev") is not None and "dev-f0" not in evidence_keys:
        raise ValueError("calibration development summary is missing its bound raw score")
    if calibration.get("status") == "passed":
        dev = _mapping(cal_payload.get("dev"), "passed development evidence")
        if not _mapping(dev.get("gate"), "passed development gate").get("passes") or cal_payload.get("next_stage") != "frozen":
            raise ValueError("passed calibration does not bind the registered development gate")
    recomputed_summaries = _validate_v2_calibration_files(root, cal_payload)
    _validate_v2_calibration_decisions(cal_payload, recomputed_summaries, str(calibration.get("status")))
    expected_runtime = _runtime_code_files(project_root)
    if cal_payload.get("runtime_code_files") != expected_runtime:
        raise ValueError("V2 calibration runtime code hashes changed")
    _validate_v2_cost_binding(root, cal_payload, config)
    report_path = root / "stages/report.json"
    if report_path.exists():
        report = read_v2_stage(report_path, "report")
        if report.get("prerequisite_stage_sha256") != {"prepare": prepare["stage_sha256"], "calibrate": calibration["stage_sha256"]}:
            raise ValueError("V2 report prerequisite hashes changed")
        report_payload = _mapping(report.get("payload"), "V2 report payload")
        if report_payload.get("config_sha256") != sha256_file(config_path):
            raise ValueError("V2 report config hash changed")
        return {**dict(report_payload), "stage_sha256": report["stage_sha256"], "run_root": str(root)}
    status = str(calibration["status"])
    payload = {
        "run_id": _mapping(prepare.get("payload"), "prepare payload").get("run_id"),
        "status": status, "prepare_stage_sha256": prepare["stage_sha256"],
        "calibration_stage_sha256": calibration["stage_sha256"],
        "config_sha256": sha256_file(config_path), "calibration": dict(cal_payload),
        "next_stage": "complete" if status in {"stopped_scientific", "stopped_resource", "failed_environment"} else "frozen",
    }
    record = _stage_record("report", status, payload, {"prepare": str(prepare["stage_sha256"]), "calibrate": str(calibration["stage_sha256"])})
    _write_new(report_path, canonical_json(record))
    return {**payload, "stage_sha256": record["stage_sha256"], "run_root": str(root)}


def _recover_v2_interrupted(root: Path, prepare: Mapping[str, object], config: V2Config, config_path: Path) -> dict[str, object]:
    attempt_path = root / "stages/calibration-attempt.json"
    if not attempt_path.exists():
        raise FileNotFoundError("no V2 calibration attempt exists to report")
    attempt = read_v2_stage(attempt_path, "calibration-attempt")
    attempt_payload = _mapping(attempt.get("payload"), "V2 calibration attempt payload")
    if attempt_payload.get("run_id") != _mapping(prepare.get("payload"), "prepare payload").get("run_id"):
        raise ValueError("V2 attempt marker run ID changed")
    if _mapping(attempt.get("prerequisite_stage_sha256"), "V2 attempt prerequisites") != {"prepare": prepare.get("stage_sha256")}:
        raise ValueError("V2 attempt marker prerequisite changed")
    if attempt_payload.get("config_sha256") != sha256_file(config_path):
        raise ValueError("V2 attempt marker config hash changed")
    pid = int(attempt_payload.get("process_id", -1))
    if pid != os.getpid() and study_v1._process_is_alive(pid):
        raise RuntimeError("V2 calibration process is still active; report will not race it")
    ledger = _load_v2_cost_ledger(root, float(config.resources.maximum_study_cuda_seconds))
    events_path = root / "resources/progress.jsonl"
    events = _read_jsonl(events_path) if events_path.exists() else []
    started = {str(event.get("phase_id")): event for event in events if event.get("event") == "phase_started"}
    completed = {str(event.get("phase_id")): event for event in events if event.get("event") == "phase_completed"}
    open_phases = [event for phase_id, event in started.items() if phase_id not in completed]
    score_entries: dict[str, object] = {}
    partial: dict[str, object] | None = None
    for event in events:
        if event.get("event") == "split_frozen":
            key = str(event.get("score_key"))
            score_entries[key] = {name: event[name] for name in ("split_sha256", "split_manifest_sha256", "split_count")}
    for phase_id, event in started.items():
        if event.get("phase") != "score":
            continue
        key, relative = str(event.get("score_key", "")), str(event.get("path", ""))
        if key not in score_entries or not relative:
            continue
        candidate = _safe_v2_run_file(root, Path(relative)) if (root / Path(relative)).exists() else root / Path(relative)
        if not candidate.is_file():
            continue
        raw_hash, count = sha256_file(candidate), _raw_line_count(candidate)
        rows = _read_jsonl(candidate)
        expected_ids = event.get("ordered_ids")
        score_ids = [row.get("example_id") for row in rows]
        completion = completed.get(phase_id)
        is_partial = completion is None
        if completion is not None:
            if completion.get("raw_sha256") != raw_hash or completion.get("row_count") != count:
                raise ValueError("completed V2 score phase evidence differs from its raw file")
            if count != int(_mapping(score_entries[key], "V2 split evidence")["split_count"]):
                raise ValueError("completed V2 score phase has the wrong row count")
        if not isinstance(expected_ids, list) or score_ids != expected_ids[:count]:
            raise ValueError("recovered V2 score rows do not match the frozen ordered IDs")
        entry = {**_mapping(score_entries[key], "V2 split evidence"), "score_sha256": raw_hash, "score_count": count}
        if is_partial:
            entry["partial"] = True
            partial = {"path": relative, "raw_sha256": raw_hash, "row_count": count}
        if rows:
            entry["summary"] = summarize_exact_scores(rows)
        score_entries[key] = entry
    elapsed = max(0.0, time.monotonic() - float(attempt_payload.get("wall_started_monotonic", time.monotonic())))
    payload = {
        "run_id": attempt_payload.get("run_id"), "status": "failed_environment",
        "failure": "process ended before calibration stage was durably completed",
        "pilot_scores": score_entries, "partial_failure": partial,
        "pilot_eligibility_state": _pilot_eligibility_state(score_entries),
        "resources": {
            "calibration_cuda_active_seconds": _v2_calibration_seconds(ledger),
            "cuda_budget_state": "unknown_after_process_loss",
            "open_phase_count": len(open_phases), "wall_elapsed_upper_bound_seconds": elapsed,
            "artifact_bytes": study_v1._tree_bytes(root),
        }, "signatures": _mapping(_mapping(prepare.get("payload"), "prepare payload").get("signatures"), "prepared signatures"),
        "runtime_code_files": _runtime_code_files(Path(__file__).resolve().parents[3]), "next_stage": "report",
    }
    for event in open_phases:
        study_v1._append_ledger(root, {"event": "phase_recovered", "phase_id": event.get("phase_id"), "recovered_utc": study_v1._utc_now(), "cuda_budget_state": "unknown_after_process_loss"})
    return _finish_v2_calibration(root, payload, "failed_environment", ledger)


def _validate_v2_calibration_files(root: Path, payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    evidence = _mapping(payload.get("pilot_scores", {}), "V2 calibration evidence")
    expected_data: set[str] = {"data/train-view.jsonl", "data/train-view.jsonl.manifest.json"}
    expected_scores: set[str] = set()
    summaries: dict[str, dict[str, object]] = {}
    for key, raw_entry in evidence.items():
        if key != "dev-f0" and key not in {f"pilot-{family}-c{candidate}" for family in FAMILIES for candidate in (1, 2)}:
            raise ValueError(f"unknown V2 calibration evidence key {key}")
        entry = _mapping(raw_entry, f"V2 evidence {key}")
        split_relative = Path("data/dev.jsonl") if key == "dev-f0" else Path("data") / f"{key}.jsonl"
        score_relative = Path("scores/dev-f0.jsonl") if key == "dev-f0" else Path("scores") / f"{key}.jsonl"
        split_path = _safe_v2_run_file(root, split_relative)
        sidecar_path = _safe_v2_run_file(root, Path(f"{split_relative}.manifest.json"))
        expected_data.update((split_relative.as_posix(), Path(f"{split_relative}.manifest.json").as_posix()))
        if sha256_file(split_path) != entry.get("split_sha256") or _raw_line_count(split_path) != entry.get("split_count"):
            raise ValueError(f"V2 stage-bound split hash/count mismatch for {key}")
        if sha256_file(sidecar_path) != entry.get("split_manifest_sha256"):
            raise ValueError(f"V2 split sidecar hash mismatch for {key}")
        manifest = _parse_json(_read_bytes(sidecar_path, f"V2 {key} sidecar"), f"V2 {key} sidecar")
        if manifest.get("sha256") != entry.get("split_sha256") or manifest.get("count") != entry.get("split_count"):
            raise ValueError(f"V2 sidecar content mismatch for {key}")
        split_rows = _parse_rows(_read_bytes(split_path, f"V2 {key} split"), "dev" if key == "dev-f0" else "pilot", int(entry["split_count"]), f"V2 {key}")
        ids = [str(row["id"]) for row in split_rows]
        if manifest.get("ordered_ids") != ids:
            raise ValueError(f"V2 split ordered IDs mismatch for {key}")
        if entry.get("score_sha256") is not None:
            score_path = _safe_v2_run_file(root, score_relative)
            expected_scores.add(score_relative.as_posix())
            if sha256_file(score_path) != entry.get("score_sha256") or _raw_line_count(score_path) != entry.get("score_count"):
                raise ValueError(f"V2 score hash/count mismatch for {key}")
            rows = _read_jsonl(score_path)
            score_ids = [row.get("example_id") for row in rows]
            if entry.get("partial") is True:
                if score_ids != ids[:len(score_ids)]:
                    raise ValueError(f"V2 partial score ordered IDs mismatch for {key}")
            elif score_ids != ids:
                raise ValueError(f"V2 score ordered IDs mismatch for {key}")
            summary = summarize_exact_scores(rows)
            if isinstance(entry.get("summary"), Mapping) and summary != entry["summary"]:
                raise ValueError(f"V2 score summary mismatch for {key}")
            if entry.get("partial") is not True:
                summaries[key] = summary
    partial = payload.get("partial_failure")
    if isinstance(partial, Mapping) and partial.get("path"):
        relative = Path(str(partial["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("V2 partial score path is unsafe")
        partial_path = _safe_v2_run_file(root, relative)
        if sha256_file(partial_path) != partial.get("raw_sha256") or _raw_line_count(partial_path) != partial.get("row_count"):
            raise ValueError("V2 partial score hash/count mismatch")
        expected_scores.add(relative.as_posix())
    for folder, expected, pattern in ((root / "data", expected_data, "*.jsonl"), (root / "scores", expected_scores, "*.jsonl")):
        actual: set[str] = set()
        if folder.exists():
            for candidate in folder.rglob(pattern):
                relative = candidate.relative_to(root).as_posix()
                if relative in {"data/sealed.jsonl", "data/sealed.jsonl.manifest.json"}:
                    continue
                actual.add(relative)
            if folder.name == "data":
                actual.update(candidate.relative_to(root).as_posix() for candidate in folder.rglob("*.manifest.json") if candidate.name != "sealed.jsonl.manifest.json")
        if actual != expected:
            raise ValueError(f"unbound or missing non-sealed V2 files: {sorted(actual ^ expected)[:1]}")
    review = payload.get("human_review")
    if isinstance(review, Mapping):
        for field, hash_field in (("review_path", "review_sha256"), ("manifest_path", "manifest_sha256")):
            relative = Path(str(review[field]))
            path = _safe_v2_run_file(root, relative)
            if sha256_file(path) != review[hash_field]:
                raise ValueError("V2 human review export hash changed")
    return summaries


def _validate_v2_calibration_decisions(
    payload: Mapping[str, object], summaries: Mapping[str, Mapping[str, object]], status: str,
) -> None:
    """Recompute pilot choices and the development gate from authenticated raw scores."""
    expected_pilot_keys = {f"pilot-{family}-c{candidate}" for family in FAMILIES for candidate in (1, 2)}
    complete_pilots = expected_pilot_keys.issubset(summaries)
    expected_eligibility_state = "complete" if complete_pilots else "incomplete"
    if payload.get("pilot_eligibility_state") != expected_eligibility_state:
        raise ValueError("V2 pilot eligibility state differs from authenticated complete pilot scores")
    pilot_rows: list[dict[str, object]] = []
    for family in FAMILIES:
        for candidate in (1, 2):
            key = f"pilot-{family}-c{candidate}"
            summary = summaries.get(key)
            if summary is None:
                continue
            family_values = _mapping(summary.get("families"), f"{key} family summaries")
            family_summary = _mapping(family_values.get(family), f"{key} family result")
            accuracy, invalid = float(family_summary["accuracy"]), float(family_summary["invalid_rate"])
            pilot_rows.append({
                "family": family, "candidate": candidate,
                "record_count": int(summary["record_count"]), "correct": int(summary["correct"]),
                "invalid_format": int(summary["invalid_format"]), "accuracy": accuracy,
                "invalid_rate": invalid, "eligible": is_informative_family(accuracy, invalid),
            })
    recorded_table = payload.get("pilot_table")
    if recorded_table is not None:
        if not isinstance(recorded_table, list):
            raise ValueError("V2 pilot table must be a list")
        recorded_by_key = {
            (row.get("family"), row.get("candidate")): row
            for row in recorded_table if isinstance(row, Mapping)
        }
        if len(recorded_by_key) != len(recorded_table):
            raise ValueError("V2 pilot table has malformed or duplicate rows")
        expected_by_key = {(str(row["family"]), int(row["candidate"])): row for row in pilot_rows}
        if set(recorded_by_key) != set(expected_by_key):
            raise ValueError("V2 pilot table does not match available authenticated pilot scores")
        for key, expected in expected_by_key.items():
            recorded = recorded_by_key[key]
            if any(recorded.get(field) != expected[field] for field in expected):
                raise ValueError("V2 pilot eligibility or summary differs from authenticated raw scores")

    if status in {"passed", "stopped_scientific"} and not expected_pilot_keys.issubset(summaries):
        raise ValueError("scientific V2 decision requires all six authenticated pilot score summaries")
    if status in {"passed", "stopped_scientific"}:
        if not isinstance(recorded_table, list) or not isinstance(payload.get("eligible_families"), list):
            raise ValueError("scientific V2 decision must retain its pilot table and eligible-family set")
        if "selected_difficulty" not in payload:
            raise ValueError("scientific V2 decision must retain its selected difficulty decision")
    if len(expected_pilot_keys.intersection(summaries)) == 6:
        choices, fallback = study_v1._select_pilot_difficulties(pilot_rows)
        eligible_families = sorted(family for family in FAMILIES if any(
            row["family"] == family and row["eligible"] is True for row in pilot_rows
        ))
        if "selected_difficulty" in payload:
            stored_choices = _mapping(payload.get("selected_difficulty"), "stored V2 pilot choices")
            expected_choices = choices if len(eligible_families) >= 2 else {}
            if dict(stored_choices) != expected_choices:
                raise ValueError("V2 selected difficulty differs from recomputed pilot selection")
        if "eligible_families" in payload and payload.get("eligible_families") != eligible_families:
            raise ValueError("V2 eligible families differ from authenticated pilot summaries")
        if len(eligible_families) < 2:
            if status != "stopped_scientific" or payload.get("dev") is not None or payload.get("next_stage") != "report":
                raise ValueError("V2 pilot no-go must stop scientifically before development")
        else:
            if status in {"passed", "stopped_scientific"} and "fallback_families" not in payload:
                raise ValueError("scientific V2 decision must retain its candidate-1 fallback set")
            if "fallback_families" in payload and payload.get("fallback_families") != fallback:
                raise ValueError("V2 candidate-1 fallback differs from authenticated pilot selection")
            recorded_rows = payload.get("pilot_table")
            if isinstance(recorded_rows, list):
                for recorded in recorded_rows:
                    if not isinstance(recorded, Mapping):
                        continue
                    family, candidate = str(recorded["family"]), int(recorded["candidate"])
                    selected = candidate == choices[family]
                    kind = (
                        "selected_first_eligible" if family not in fallback and selected
                        else "prespecified_candidate_1_fallback" if family in fallback and selected
                        else "not_selected"
                    )
                    has_selection = "selected" in recorded or "selection_kind" in recorded
                    if status in {"passed", "stopped_scientific"} and not has_selection:
                        raise ValueError("scientific V2 calibration is missing its persisted pilot selection table")
                    if has_selection and (recorded.get("selected") is not selected or recorded.get("selection_kind") != kind):
                        raise ValueError("V2 pilot selection table differs from recomputed first-eligible choice")

    dev_summary = summaries.get("dev-f0")
    dev = payload.get("dev")
    if dev is None:
        if dev_summary is not None and status in {"passed", "stopped_scientific"}:
            raise ValueError("V2 development raw scores exist without a stored development decision")
        if status == "passed":
            raise ValueError("passed V2 calibration requires a development decision")
        eligible_count = len([family for family in FAMILIES if any(
            row["family"] == family and row["eligible"] is True for row in pilot_rows
        )])
        if status == "stopped_scientific" and eligible_count >= 2:
            raise ValueError("V2 scientific stop after eligible pilots requires a development gate")
        return
    dev_record = _mapping(dev, "V2 development decision")
    if dev_summary is None:
        raise ValueError("V2 development decision has no authenticated raw score summary")
    if dev_record.get("scores") != dev_summary:
        raise ValueError("V2 development summary differs from authenticated raw scores")
    expected_gate = _development_gate_v2(dev_summary)
    if dev_record.get("gate") != expected_gate:
        raise ValueError("V2 development gate differs from recomputed raw-score gate")
    if status == "passed":
        if not expected_gate["passes"] or payload.get("next_stage") != "frozen":
            raise ValueError("passed V2 calibration requires a passing recomputed development gate")
    elif status == "stopped_scientific":
        if expected_gate["passes"] or payload.get("next_stage") != "report":
            raise ValueError("V2 development no-go status differs from recomputed gate")


def _validate_v2_cost_binding(root: Path, payload: Mapping[str, object], config: V2Config) -> None:
    ledger = _load_v2_cost_ledger(root, float(config.resources.maximum_study_cuda_seconds))
    if payload.get("cuda_cost_ledger_sha256") != study_v1._optional_sha256(root / "resources/cuda-cost.jsonl"):
        raise ValueError("V2 CUDA cost ledger hash differs from calibration binding")
    if float(payload.get("cuda_cost_ledger_total_seconds", -1)) != ledger.total_seconds:
        raise ValueError("V2 CUDA cost total differs from calibration binding")
    if float(payload.get("calibration_cuda_active_seconds", -1)) != _v2_calibration_seconds(ledger):
        raise ValueError("V2 calibration CUDA time differs from cost ledger")


def _raw_line_count(path: Path) -> int:
    return sum(1 for _ in Path(path).open("rb"))


def _runtime_code_files(project_root: Path) -> dict[str, str]:
    files = {
        relative: sha256_file(project_root / Path(relative))
        for relative in (*_V1_CODE_FILES, *_V2_CODE_FILES)
    }
    files["aggregate_sha256"] = sha256_payload(files)
    return files


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            rows.append(value)
    return rows


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
