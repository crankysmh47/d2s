"""Immutable prepare/calibrate/report stages for the separate recurrence V1 study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import time
from typing import Any
import uuid

import torch

from .config_v1 import V1Config, load_v1_config
from .data import Example
from .data_v1 import FAMILIES, freeze_v1_split, make_pilot, make_selected_splits
from .gates import is_informative_family, summarize_exact_scores
from .gates import CudaCostLedger
from .provenance import canonical_json, sha256_file, sha256_payload
from .v1_runtime import ScoreExecutionError, load_v1_model, score_v1_examples


_FROZEN_SCORER_SHA256 = "2f138e6fe106c81dd03712d0a49a3a916cec2596a4acfa8f4e111f6463bab719"
_STAGES = {"prepare", "calibrate", "report"}
_TERMINAL = {"stopped_scientific", "stopped_resource", "failed_environment", "passed"}
_WALL_SAFETY_SECONDS = 7200.0


class ResourceLimitReached(RuntimeError):
    """A frozen V1 compute or artifact budget has been reached."""


class MeasuredPhaseFailure(RuntimeError):
    """Failed CUDA phase carrying time accrued before its exception."""

    def __init__(self, cause: BaseException, cuda_active_seconds: float):
        super().__init__(str(cause))
        self.cuda_active_seconds = cuda_active_seconds


def run_v1_stage(stage: str, config_path: Path, run_root: Path | None = None) -> dict[str, object]:
    """Run exactly one V1 stage. Later study stages remain deliberately unavailable."""
    if stage not in _STAGES:
        raise ValueError(f"unsupported V1 stage {stage!r}; implemented stages are prepare, calibrate, report")
    config_path = Path(config_path).resolve(strict=True)
    config = load_v1_config(config_path)
    project_root = Path(__file__).resolve().parents[3]
    if stage == "prepare":
        prepared = _verify_source(config, config_path, project_root)
        run_id = _derive_run_id(prepared["config_bytes"], prepared["signatures"])
        expected_root = _resolve(config.data.output_root, project_root) / run_id
        if run_root is not None and Path(run_root).absolute() != expected_root.absolute():
            raise ValueError("prepare run root must equal configured output root plus deterministic run ID")
        return _prepare(expected_root, run_id, config_path, config, prepared)
    if run_root is None:
        raise ValueError(f"{stage} requires a run root")
    root = Path(run_root).absolute()
    _assert_no_reparse(root)
    if stage == "calibrate":
        try:
            return _calibrate(root, config, config_path, project_root)
        except Exception as error:
            calibration_path = root / "stages" / "calibrate.json"
            if calibration_path.exists():
                raise
            prepare = _read_stage(root / "stages" / "prepare.json", "prepare")
            prepare_payload = _mapping(prepare.get("payload"), "prepare payload")
            payload = {
                "run_id": prepare_payload.get("run_id"), "status": "failed_environment",
                "failure": f"preflight {type(error).__name__}: {error}",
                "signatures": dict(_mapping(prepare_payload.get("signatures"), "prepared signatures")),
                "resources": {"calibration_cuda_active_seconds": _ledger_seconds(root), "artifact_bytes": _tree_bytes(root)},
                "runtime_code_files": _runtime_code_files(project_root),
                "next_stage": "report",
            }
            return _finish_calibration(root, payload, "failed_environment")
    return _report(root, config, config_path)


def _derive_run_id(config_bytes: bytes, signatures: Mapping[str, str]) -> str:
    return sha256_payload({"config_bytes_sha256": hashlib.sha256(config_bytes).hexdigest(), "signatures": dict(signatures)})


def _verify_source(config: V1Config, config_path: Path, project_root: Path) -> dict[str, Any]:
    source_root = _resolve(config.source_run, project_root)
    _assert_no_reparse(source_root)
    if not source_root.is_dir():
        raise ValueError("V0 source run is missing")
    source_record = _read_json(source_root / "stages" / "source.json")
    data_record = _read_json(source_root / "stages" / "data.json")
    if source_record.get("stage") != "source" or source_record.get("status") != "passed":
        raise ValueError("V0 source stage must be passed")
    if data_record.get("stage") != "data" or data_record.get("status") != "passed":
        raise ValueError("V0 data stage must be passed")
    source_manifest_bytes = _read_bytes(source_root / "source.json")
    source = _parse_json(source_manifest_bytes, "V0 source manifest")
    source_hash = hashlib.sha256(source_manifest_bytes).hexdigest()
    source_payload = _mapping(source_record.get("payload"), "V0 source payload")
    source_signatures = _mapping(source_record.get("signatures"), "V0 source signatures")
    data_payload = _mapping(data_record.get("payload"), "V0 data payload")
    data_signatures = _mapping(data_record.get("signatures"), "V0 data signatures")
    if (
        source.get("repo_id") != config.model_repo_id
        or source.get("model_class") != "Qwen3ForCausalLM"
        or source.get("revision") != config.model_revision
        or source_payload.get("revision") != config.model_revision
        or source_payload.get("source_manifest_sha256") != source_hash
        or source_signatures.get("source") != source_hash
        or data_signatures.get("source") != source_hash
    ):
        raise ValueError("V0 source stage does not bind the pinned Qwen source manifest")
    relative_snapshot = source_payload.get("snapshot_relative_path")
    if not isinstance(relative_snapshot, str) or Path(relative_snapshot).is_absolute() or ".." in Path(relative_snapshot).parts:
        raise ValueError("pinned source snapshot path is invalid")
    snapshot = source_root / relative_snapshot
    if snapshot.name != config.model_revision:
        raise ValueError("pinned model snapshot directory name does not match the immutable revision")
    _verify_snapshot(snapshot, source)
    tokenizer_hash = source_payload.get("tokenizer_sha256")
    if source_payload.get("tokenizer_revision") != config.model_revision:
        raise ValueError("pinned tokenizer revision mismatch")
    inventory = source.get("files")
    tokenizer_entries = [row for row in inventory if isinstance(row, Mapping) and row.get("relative_path") == "tokenizer.json"] if isinstance(inventory, list) else []
    if len(tokenizer_entries) != 1 or tokenizer_entries[0].get("sha256") != tokenizer_hash:
        raise ValueError("tokenizer hash is not bound by the pinned source inventory")

    manifest_bytes = _read_bytes(source_root / "data" / "manifest.json")
    manifest = _parse_json(manifest_bytes, "V0 data manifest")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if data_payload.get("manifest_sha256") != manifest_hash or data_payload.get("data_hash") != data_signatures.get("data"):
        raise ValueError("V0 data stage does not bind its manifest and data signature")
    counts = _mapping(manifest.get("counts"), "V0 split counts")
    if counts.get("train") != 8000 or counts.get("dev") != 600:
        raise ValueError("V0 train/dev source counts differ from the registered 8,000/600")
    split_hashes = _mapping(manifest.get("split_sha256"), "V0 split hashes")
    train_path, dev_path = source_root / "data" / "train.jsonl", source_root / "data" / "dev.jsonl"
    train_bytes, dev_bytes = _read_bytes(train_path), _read_bytes(dev_path)
    train_hash, dev_hash = hashlib.sha256(train_bytes).hexdigest(), hashlib.sha256(dev_bytes).hexdigest()
    if split_hashes.get("train") != train_hash or split_hashes.get("dev") != dev_hash:
        raise ValueError("V0 train/dev split raw hash differs from its data manifest")
    train_rows = _parse_rows(train_bytes, "train", 8000)
    dev_rows = _parse_rows(dev_bytes, "dev", 600)
    forbidden = {str(row["id"]) for row in train_rows} | {str(row["id"]) for row in dev_rows}
    if len(forbidden) != 8600:
        raise ValueError("V0 train/dev source IDs are not unique")
    if sha256_payload({"ids": sorted(row["id"] for row in train_rows)}) != _mapping(manifest.get("split_id_sha256"), "V0 split ID hashes").get("train"):
        raise ValueError("V0 training IDs do not match manifest split ID hash")
    if sha256_payload({"ids": sorted(row["id"] for row in dev_rows)}) != _mapping(manifest.get("split_id_sha256"), "V0 split ID hashes").get("dev"):
        raise ValueError("V0 development IDs do not match manifest split ID hash")

    config_bytes = config_path.read_bytes()
    scorer_path = project_root / "src" / "ppmoe" / "recurrence" / "answer_v1.py"
    generator_path = project_root / "src" / "ppmoe" / "recurrence" / "data_v1.py"
    scorer_hash, generator_hash = sha256_file(scorer_path), sha256_file(generator_path)
    if scorer_hash != _FROZEN_SCORER_SHA256:
        raise ValueError("V1 scorer SHA256 differs from the frozen specification")
    signatures = {
        "source": source_hash,
        "model": sha256_payload({"revision": config.model_revision, "files": inventory}),
        "tokenizer": str(tokenizer_hash),
        "train": train_hash,
        "dev": dev_hash,
        "scorer": scorer_hash,
        "generator": generator_hash,
    }
    return {
        "config_bytes": config_bytes,
        "signatures": signatures,
        "source": source,
        "source_root": source_root,
        "snapshot": snapshot.resolve(strict=True),
        "forbidden_ids": forbidden,
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "manifest_sha256": manifest_hash,
    }


def _verify_snapshot(snapshot: Path, source: Mapping[str, object]) -> None:
    _assert_no_reparse(snapshot)
    if not snapshot.is_dir():
        raise ValueError("pinned local model snapshot is missing")
    entries = source.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source file inventory is missing")
    expected_files: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("source file inventory entry is invalid")
        relative, expected_size, expected_hash = entry.get("relative_path"), entry.get("bytes"), entry.get("sha256")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("source file inventory path is invalid")
        if relative in expected_files:
            raise ValueError("source file inventory paths must be unique")
        expected_files[relative] = entry
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or not isinstance(expected_hash, str):
            raise ValueError(f"source file inventory metadata is invalid: {relative}")
    metadata_sources = [relative for relative in expected_files if not relative.startswith(".cache/")]
    metadata_names = [Path(relative).name + ".metadata" for relative in metadata_sources]
    if len(set(metadata_names)) != len(metadata_names):
        raise ValueError("source inventory has ambiguous Hugging Face metadata basenames")
    metadata_directory = ".cache/huggingface/download"
    expected_metadata = {f"{metadata_directory}/{name}" for name in metadata_names}
    actual_files: set[str] = set()
    actual_metadata: set[str] = set()
    for directory, dirs, files in os.walk(snapshot, followlinks=False):
        parent = Path(directory)
        for name in [*dirs, *files]:
            target = parent / name
            metadata = target.lstat()
            if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise ValueError(f"symlink or reparse point in model snapshot: {target}")
            if name in dirs:
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ValueError(f"model snapshot contains a non-directory path: {target}")
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"model snapshot contains a non-regular file: {target}")
            relative = target.relative_to(snapshot).as_posix()
            if relative.startswith(metadata_directory + "/"):
                actual_metadata.add(relative)
            else:
                actual_files.add(relative)
    if actual_files != set(expected_files) or actual_metadata != expected_metadata:
        extra = sorted((actual_files - set(expected_files)) | (actual_metadata - expected_metadata))
        missing = sorted((set(expected_files) - actual_files) | (expected_metadata - actual_metadata))
        raise ValueError(f"snapshot file inventory differs from manifest (extra={extra[:1]}, missing={missing[:1]})")
    for relative, entry in expected_files.items():
        target = snapshot / Path(relative)
        if target.stat().st_size != entry["bytes"] or sha256_file(target) != entry["sha256"]:
            raise ValueError(f"pinned model file hash mismatch: {relative}")


def _prepare(root: Path, run_id: str, config_path: Path, config: V1Config, inputs: Mapping[str, Any]) -> dict[str, object]:
    _assert_no_reparse(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=False, exist_ok=False)
    signatures = dict(inputs["signatures"])
    payload: dict[str, object] = {
        "run_id": run_id,
        "run_root": str(root),
        "config_sha256": hashlib.sha256(inputs["config_bytes"]).hexdigest(),
        "source_root": str(inputs["source_root"]),
        "snapshot_relative_path": str(Path(inputs["snapshot"]).relative_to(inputs["source_root"])),
        "source_manifest_sha256": signatures["source"],
        "data_manifest_sha256": inputs["manifest_sha256"],
        "signatures": signatures,
        "v0_train_count": inputs["train_count"],
        "v0_dev_count": inputs["dev_count"],
        "v0_train_dev_forbidden_id_count": len(inputs["forbidden_ids"]),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        for directory in (root / "stages", root / "data", root / "scores", root / "resources"):
            _assert_no_reparse(directory)
            directory.mkdir()
        _append_ledger(root, {"event": "prepare", "cumulative_cuda_active_seconds": 0.0, "artifact_bytes": 0})
        record = _stage_record("prepare", "passed", payload, {})
        _write_new(root / "stages" / "prepare.json", record)
    except BaseException as error:
        record = _stage_record("prepare", "failed_environment", {**payload, "failure": f"{type(error).__name__}: {error}"}, {})
        _write_new(root / "stages" / "prepare.json", record)
        return {"run_id": run_id, "status": "failed_environment", "next_stage": "report", "run_root": str(root), "failure": str(error)}
    return {"run_id": run_id, "status": "passed", "next_stage": "calibrate", "run_root": str(root), "stage_sha256": record["stage_sha256"]}


def _calibrate(root: Path, config: V1Config, config_path: Path, project_root: Path) -> dict[str, object]:
    prepare_path = root / "stages" / "prepare.json"
    prepare_record = _read_stage(prepare_path, "prepare")
    _validate_run_binding(root, prepare_record, config, config_path, project_root)
    if prepare_record.get("status") != "passed":
        raise ValueError("calibrate requires a passed prepare stage")
    if (root / "stages" / "calibrate.json").exists():
        raise FileExistsError("calibration stage already exists; retries require a new run identity")
    _assert_calibration_not_started(root)
    payload = _mapping(prepare_record.get("payload"), "prepare payload")
    if payload.get("config_sha256") != sha256_file(config_path):
        raise ValueError("config hash changed after prepare")
    fresh = _verify_source(config, config_path, project_root)
    frozen = _mapping(payload.get("signatures"), "prepared signatures")
    if dict(frozen) != fresh["signatures"]:
        raise ValueError("source, tokenizer, data, scorer, or generator hash changed after prepare")
    signatures = dict(frozen)
    runtime_code_files = _runtime_code_files(project_root)
    runtime_code_sha256 = sha256_payload(runtime_code_files)
    signatures["runtime_code"] = runtime_code_sha256
    snapshot = Path(str(fresh["snapshot"]))
    if Path(str(payload.get("source_root"))).resolve(strict=True) != Path(fresh["source_root"]).resolve(strict=True):
        raise ValueError("source run path changed after prepare")
    _write_attempt_marker(root, prepare_record, config_path)
    deadline = time.monotonic() + _WALL_SAFETY_SECONDS
    cost_ledger = _load_cost_ledger(root, float(config.resources.maximum_study_cuda_seconds))
    cumulative = _calibration_cost_seconds(cost_ledger)
    score_manifest: dict[str, object] = {}
    pilot_table: list[dict[str, object]] = []
    dev_summary: dict[str, object] | None = None
    dev_split_evidence: dict[str, object] | None = None
    active_score_path: Path | None = None
    sealed_meta: dict[str, object] | None = None
    status = "passed"
    failure: str | None = None
    try:
        _budget_check(root, cost_ledger, config)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        phase_id = _start_phase(root, "model_load")
        try:
            model, tokenizer, model_load_seconds = _load_calibration_runtime_measured(snapshot, config.model_revision)
        except MeasuredPhaseFailure as error:
            _record_phase_cost(root, cost_ledger, phase_id, "calibration-model-load", error.cuda_active_seconds, "failed_environment")
            raise
        except Exception:
            _complete_phase(root, phase_id, "calibration-model-load", 0.0, "failed_environment")
            raise
        cost_ledger.record(phase_id, "calibration-model-load", model_load_seconds, "completed")
        _complete_phase(root, phase_id, "calibration-model-load", model_load_seconds, "completed")
        cumulative = _calibration_cost_seconds(cost_ledger)
        forbidden = set(fresh["forbidden_ids"])
        pilot_examples: dict[tuple[str, int], tuple[Example, ...]] = {}
        for family in FAMILIES:
            for candidate in (1, 2):
                _budget_check(root, cost_ledger, config)
                examples = make_pilot(config, family, candidate, forbidden)
                pilot_examples[(family, candidate)] = examples
                forbidden.update(row.id for row in examples)
                split_path = root / "data" / f"pilot-{family}-c{candidate}.jsonl"
                split_manifest = freeze_v1_split(
                    split_path, examples, tokenizer, revision=config.model_revision,
                    tokenizer_sha256=str(signatures["tokenizer"]),
                )
                _verify_v1_split(split_path, str(split_manifest["sha256"]), len(examples))
                score_key = f"pilot-{family}-c{candidate}"
                score_manifest[score_key] = {
                    "split_sha256": split_manifest["sha256"],
                    "split_manifest_sha256": sha256_file(Path(str(split_path) + ".manifest.json")),
                    "split_count": len(examples),
                }
                _append_ledger(root, {"event": "split_frozen", "score_key": score_key, "split_path": str(split_path.relative_to(root)), **score_manifest[score_key]})
                score_path = root / "scores" / f"pilot-{family}-c{candidate}.jsonl"
                active_score_path = score_path
                phase_id = _start_phase(root, "score", str(score_path.relative_to(root)), {"score_key": score_key, "ordered_ids": [row.id for row in examples]})
                try:
                    score_result = score_v1_examples(
                        model, tokenizer, examples, "single", score_path, signatures,
                        max_output_tokens=256, max_total_tokens=512, deadline=deadline,
                    )
                except ScoreExecutionError as error:
                    outcome = "stopped_resource" if error.deadline_exceeded else "failed_environment"
                    cost_ledger.record(phase_id, "calibration-pilot", error.cuda_active_seconds, outcome)
                    _complete_phase(root, phase_id, "calibration-pilot", error.cuda_active_seconds, outcome, {"row_count": error.row_count, "raw_sha256": error.raw_sha256})
                    raise
                cost_ledger.record(phase_id, "calibration-pilot", float(score_result["cuda_active_seconds"]), "completed")
                _complete_phase(root, phase_id, "calibration-pilot", float(score_result["cuda_active_seconds"]), "completed", {"row_count": score_result["validated_count"], "raw_sha256": score_result["raw_sha256"]})
                cumulative = _calibration_cost_seconds(cost_ledger)
                summary = _validate_score_file(score_path, examples, score_result)
                score_manifest[score_key] = {
                    **score_manifest[score_key],
                    "score_sha256": score_result["raw_sha256"],
                    "score_count": score_result["validated_count"],
                    "summary": summary,
                }
                _budget_check(root, cost_ledger, config)
                table_row = _pilot_summary(family, candidate, summary)
                pilot_table.append(table_row)
        choices, fallback = _select_pilot_difficulties(pilot_table)
        if len(choices) < 2:
            status = "stopped_scientific"
            result_payload: dict[str, object] = {
                "run_id": payload["run_id"], "status": status, "pilot_table": pilot_table,
                "pilot_scores": score_manifest, "eligible_families": sorted(choices),
                "selected_difficulty": {}, "dev": None, "sealed": None,
                "cuda_active_seconds": cumulative, "artifact_bytes": _tree_bytes(root),
                "signatures": signatures, "next_stage": "report",
                "runtime_code_files": runtime_code_files,
                "resources": {"calibration_cuda_active_seconds": cumulative, "artifact_bytes": _tree_bytes(root), "cuda": _cuda_resource_snapshot()},
            }
            return _finish_calibration(root, result_payload, status)
        for row in pilot_table:
            row["selected"] = row["candidate"] == choices[str(row["family"])]
            row["selection_kind"] = "selected_first_eligible" if row["family"] not in fallback and row["selected"] else "prespecified_candidate_1_fallback" if row["family"] in fallback and row["selected"] else "not_selected"
        _budget_check(root, cost_ledger, config)
        splits = make_selected_splits(config, choices, forbidden)
        dev_examples = splits["dev"]
        sealed_examples = splits["sealed"]
        dev_path = root / "data" / "dev.jsonl"
        dev_manifest = freeze_v1_split(dev_path, dev_examples, tokenizer, revision=config.model_revision, tokenizer_sha256=str(signatures["tokenizer"]))
        _verify_v1_split(dev_path, str(dev_manifest["sha256"]), len(dev_examples))
        sealed_path = root / "data" / "sealed.jsonl"
        sealed_manifest = freeze_v1_split(sealed_path, sealed_examples, tokenizer, revision=config.model_revision, tokenizer_sha256=str(signatures["tokenizer"]))
        sealed_meta = {"sha256": sealed_manifest["sha256"], "count": len(sealed_examples)}
        dev_split_evidence = {"sha256": dev_manifest["sha256"], "count": len(dev_examples)}
        score_manifest["dev-f0"] = {
            "split_sha256": dev_manifest["sha256"],
            "split_manifest_sha256": sha256_file(Path(str(dev_path) + ".manifest.json")),
            "split_count": len(dev_examples),
        }
        _append_ledger(root, {"event": "split_frozen", "score_key": "dev-f0", "split_path": str(dev_path.relative_to(root)), **score_manifest["dev-f0"]})
        dev_score_path = root / "scores" / "dev-f0.jsonl"
        active_score_path = dev_score_path
        phase_id = _start_phase(root, "score", str(dev_score_path.relative_to(root)), {"score_key": "dev-f0", "ordered_ids": [row.id for row in dev_examples]})
        try:
            dev_result = score_v1_examples(model, tokenizer, dev_examples, "single", dev_score_path, signatures, max_output_tokens=256, max_total_tokens=512, deadline=deadline)
        except ScoreExecutionError as error:
            outcome = "stopped_resource" if error.deadline_exceeded else "failed_environment"
            cost_ledger.record(phase_id, "calibration-dev", error.cuda_active_seconds, outcome)
            _complete_phase(root, phase_id, "calibration-dev", error.cuda_active_seconds, outcome, {"row_count": error.row_count, "raw_sha256": error.raw_sha256})
            raise
        cost_ledger.record(phase_id, "calibration-dev", float(dev_result["cuda_active_seconds"]), "completed")
        _complete_phase(root, phase_id, "calibration-dev", float(dev_result["cuda_active_seconds"]), "completed", {"row_count": dev_result["validated_count"], "raw_sha256": dev_result["raw_sha256"]})
        cumulative = _calibration_cost_seconds(cost_ledger)
        dev_summary = _validate_score_file(dev_score_path, dev_examples, dev_result)
        score_manifest["dev-f0"] = {
            **score_manifest["dev-f0"], "score_sha256": dev_result["raw_sha256"],
            "score_count": dev_result["validated_count"], "summary": dev_summary,
        }
        _budget_check(root, cost_ledger, config)
        selected_gate = _development_gate(dev_summary)
        review = _export_blinded_review(dev_score_path, dev_examples, root)
        status = "passed" if selected_gate["passes"] else "stopped_scientific"
        result_payload = {
            "run_id": payload["run_id"], "status": status, "pilot_table": pilot_table,
            "pilot_scores": score_manifest, "eligible_families": sorted(row for row in FAMILIES if row not in fallback),
            "selected_difficulty": choices, "fallback_families": fallback,
            "dev": {"split_sha256": dev_manifest["sha256"], "split_manifest_sha256": sha256_file(Path(str(dev_path) + ".manifest.json")), "split_count": len(dev_examples), "scores": dev_summary, "gate": selected_gate},
            "sealed": sealed_meta,
            "resources": {"calibration_cuda_active_seconds": cumulative, "artifact_bytes": _tree_bytes(root), "cuda": _cuda_resource_snapshot(), "wall_safety_deadline_seconds": _WALL_SAFETY_SECONDS, "maximum_calibration_cuda_seconds": config.resources.maximum_calibration_cuda_seconds, "maximum_study_cuda_seconds": config.resources.maximum_study_cuda_seconds, "maximum_artifact_bytes": config.resources.maximum_artifact_bytes},
            "human_review": review, "signatures": signatures,
            "runtime_code_files": runtime_code_files,
            "next_stage": "frozen" if status == "passed" else "report",
        }
        return _finish_calibration(root, result_payload, status)
    except ResourceLimitReached as error:
        status, failure = "stopped_resource", str(error)
    except Exception as error:
        status, failure = "failed_environment", f"{type(error).__name__}: {error}"
        if isinstance(error, ScoreExecutionError) and error.deadline_exceeded:
            status = "stopped_resource"
        if isinstance(error, MeasuredPhaseFailure):
            cumulative = _calibration_cost_seconds(cost_ledger)
        if isinstance(error, ScoreExecutionError):
            score_manifest["partial_failure"] = {
                "path": str(active_score_path.relative_to(root)) if active_score_path is not None else None,
                "row_count": error.row_count, "raw_sha256": error.raw_sha256,
            }
            if active_score_path is not None:
                key = active_score_path.stem
                if key in score_manifest:
                    score_manifest[key] = {**_mapping(score_manifest[key], "partial score split evidence"), "score_sha256": error.raw_sha256, "score_count": error.row_count, "partial": True}
            if error.deadline_exceeded:
                status = "stopped_resource"
        if "wall safety deadline" in str(error).lower():
            status = "stopped_resource"
        if active_score_path is not None and active_score_path.is_file() and "partial_failure" not in score_manifest:
            key = active_score_path.stem
            raw_hash, row_count = sha256_file(active_score_path), _raw_line_count(active_score_path)
            score_manifest["partial_failure"] = {
                "path": str(active_score_path.relative_to(root)),
                "row_count": row_count,
                "raw_sha256": raw_hash,
            }
            if key in score_manifest:
                score_manifest[key] = {**_mapping(score_manifest[key], "partial score split evidence"), "score_sha256": raw_hash, "score_count": row_count, "partial": True}
    cumulative = _calibration_cost_seconds(_load_cost_ledger(root, float(config.resources.maximum_study_cuda_seconds)))
    _append_ledger(root, {"event": "calibration_failed", "artifact_bytes": _tree_bytes(root), "status": status, "failure": failure, "resource_bound_fired": "wall_safety_deadline" if status == "stopped_resource" and "wall safety deadline" in str(failure).lower() else "cuda_or_artifact_cap" if status == "stopped_resource" else None})
    result_payload = {
        "run_id": payload.get("run_id"), "status": status, "failure": failure,
        "pilot_table": pilot_table, "pilot_scores": score_manifest,
        "dev": None if dev_summary is None and dev_split_evidence is None else {**(dev_split_evidence or {}), "scores": dev_summary},
        "sealed": sealed_meta,
        "signatures": signatures, "next_stage": "report",
        "runtime_code_files": _runtime_code_files(project_root),
        "resources": {"calibration_cuda_active_seconds": cumulative, "artifact_bytes": _tree_bytes(root), "cuda": _cuda_resource_snapshot(), "wall_safety_deadline_seconds": _WALL_SAFETY_SECONDS, "resource_bound_fired": "wall_safety_deadline" if status == "stopped_resource" and "wall safety deadline" in str(failure).lower() else "cuda_or_artifact_cap" if status == "stopped_resource" else None},
    }
    return _finish_calibration(root, result_payload, status)


def _load_calibration_runtime(snapshot: Path, revision: str) -> tuple[object, object]:
    return load_v1_model(snapshot, revision)


def _runtime_code_files(project_root: Path) -> dict[str, str]:
    paths = (
        "src/ppmoe/recurrence/study_v1.py",
        "src/ppmoe/recurrence/v1_runtime.py",
        "src/ppmoe/recurrence/model.py",
        "src/ppmoe/recurrence/evaluate.py",
        "src/ppmoe/recurrence/data.py",
        "src/ppmoe/recurrence/human_review.py",
        "src/ppmoe/recurrence/gates.py",
    )
    output = {relative: sha256_file(project_root / Path(relative)) for relative in paths}
    output["aggregate_sha256"] = sha256_payload(output)
    return output


def _cuda_resource_snapshot() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"available": False}
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "available": True,
        "device": torch.cuda.get_device_name(index),
        "index": index,
        "total_vram_bytes": int(properties.total_memory),
        "allocated_vram_bytes": int(torch.cuda.memory_allocated(index)),
        "reserved_vram_bytes": int(torch.cuda.memory_reserved(index)),
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(index)),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(index)),
    }


def _verify_calibration_runtime_code(root: Path, project_root: Path) -> None:
    calibration = _read_stage(root / "stages" / "calibrate.json", "calibrate")
    payload = _mapping(calibration.get("payload"), "calibration payload")
    expected = _mapping(payload.get("runtime_code_files"), "calibration runtime code signatures")
    current = _runtime_code_files(project_root)
    if dict(expected) != current:
        raise ValueError("calibration runtime code hashes changed")


def _load_calibration_runtime_measured(snapshot: Path, revision: str) -> tuple[object, object, float]:
    if not torch.cuda.is_available():
        model, tokenizer = _load_calibration_runtime(snapshot, revision)
        return model, tokenizer, 0.0
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        model, tokenizer = _load_calibration_runtime(snapshot, revision)
    except BaseException as error:
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end)) / 1000.0
        raise MeasuredPhaseFailure(error, elapsed) from error
    end.record()
    end.synchronize()
    return model, tokenizer, float(start.elapsed_time(end)) / 1000.0


def _pilot_summary(family: str, candidate: int, summary: Mapping[str, object]) -> dict[str, object]:
    families = _mapping(summary.get("families"), "pilot family summary")
    family_summary = _mapping(families.get(family), "pilot family result")
    accuracy, invalid_rate = float(family_summary["accuracy"]), float(family_summary["invalid_rate"])
    return {
        "family": family, "candidate": candidate,
        "record_count": int(summary["record_count"]), "correct": int(summary["correct"]),
        "invalid_format": int(summary["invalid_format"]), "accuracy": accuracy,
        "invalid_rate": invalid_rate, "eligible": is_informative_family(accuracy, invalid_rate),
    }


def _validate_score_file(path: Path, examples: Sequence[Example], result: Mapping[str, object]) -> dict[str, object]:
    raw_hash = sha256_file(path)
    if raw_hash != result.get("raw_sha256") or result.get("validated_count") != len(examples):
        raise ValueError("score file hash/count does not match runtime validation")
    rows = _read_jsonl(path)
    expected_ids = [row.id for row in examples]
    actual_ids = [row.get("example_id") for row in rows]
    if actual_ids != expected_ids or len(set(actual_ids)) != len(expected_ids):
        raise ValueError("score file IDs must match the frozen ordered split exactly once")
    summary = summarize_exact_scores(rows)
    if summary != result.get("summary"):
        raise ValueError("score summary does not match raw score records")
    return summary


def _development_gate(summary: Mapping[str, object]) -> dict[str, object]:
    families = _mapping(summary.get("families"), "development family summaries")
    family_set = set(families) == set(FAMILIES)
    info = [family for family, value in families.items() if isinstance(value, Mapping) and is_informative_family(float(value["accuracy"]), float(value["invalid_rate"]))]
    macro = float(summary.get("macro_accuracy", -1))
    invalid = float(summary.get("global_invalid_rate", 1))
    return {
        "family_summaries": dict(families), "macro_accuracy": macro,
        "global_invalid_rate": invalid, "informative_families": sorted(info),
        "macro_pass": 0.05 < macro < 0.95, "invalid_pass": invalid <= 0.10,
        "family_set_pass": family_set, "minimum_informative_families_pass": len(info) >= 2,
        "passes": family_set and 0.05 < macro < 0.95 and invalid <= 0.10 and len(info) >= 2,
    }


def _select_pilot_difficulties(pilot_table: Sequence[Mapping[str, object]]) -> tuple[dict[str, int], list[str]]:
    choices: dict[str, int] = {}
    for family in FAMILIES:
        eligible = [row for row in pilot_table if row.get("family") == family and row.get("eligible") is True]
        eligible.sort(key=lambda row: int(row["candidate"]))
        if eligible:
            choices[family] = int(eligible[0]["candidate"])
    fallback = [family for family in FAMILIES if family not in choices]
    if len(choices) >= 2:
        for family in fallback:
            choices[family] = 1
    return choices, fallback


def _pilot_selection_status(choices: Mapping[str, int]) -> str:
    return "stopped_scientific" if len(choices) < 2 else "continue"


def _verify_v1_split(path: Path, expected_sha256: str, expected_count: int) -> list[dict[str, object]]:
    """Check stage-bound bytes before parsing; sidecars cannot authenticate themselves."""
    _assert_no_reparse(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("split raw hash differs from immutable stage record")
    raw = _read_bytes(path)
    if len(raw.splitlines()) != expected_count:
        raise ValueError("split raw count differs from immutable stage record")
    rows = _read_jsonl(path)
    identifiers = [row.get("id") for row in rows]
    if any(not isinstance(identifier, str) for identifier in identifiers) or len(set(identifiers)) != expected_count:
        raise ValueError("split IDs must be unique and complete")
    return rows


def _export_blinded_review(score_path: Path, examples: Sequence[Example], root: Path) -> dict[str, object]:
    from .human_review import export_review

    rows = _read_jsonl(score_path)
    by_id = {example.id: example for example in examples}
    sampled: list[dict[str, object]] = []
    rng = random.Random(9158)
    for family in FAMILIES:
        members = [row for row in rows if row.get("family") == family]
        if len(members) < 10:
            raise ValueError("development set has fewer than ten examples in a family")
        sampled.extend(rng.sample(members, 10))
    records = []
    for row in sampled:
        example = by_id[str(row["example_id"])]
        records.append({
            **row, "arm": "F0", "split": "dev", "prompt": example.prompt,
            "answer": example.answer,
        })
    output = export_review(records, root / "human-review", seed=9158)
    return {
        "sample_count": len(records), "seed": 9158,
        "review_sha256": sha256_file(Path(output["review_path"])),
        "manifest_sha256": sha256_file(Path(output["manifest_path"])),
        "review_path": str(output["review_path"].relative_to(root)),
        "manifest_path": str(output["manifest_path"].relative_to(root)),
        "reviewer": "unassigned; parent-run Luna audit pending",
    }


def _finish_calibration(root: Path, payload: Mapping[str, object], status: str) -> dict[str, object]:
    if status not in _TERMINAL:
        raise ValueError(f"invalid calibration status {status}")
    ledger = _load_cost_ledger(root, 86400.0)
    frozen_payload = dict(payload)
    frozen_payload["cuda_cost_ledger_sha256"] = _optional_sha256(root / "resources" / "cuda-cost.jsonl")
    frozen_payload["cuda_cost_ledger_total_seconds"] = ledger.total_seconds
    frozen_payload["calibration_cuda_active_seconds"] = _calibration_cost_seconds(ledger)
    prepare = _read_stage(root / "stages" / "prepare.json", "prepare")
    record = _stage_record("calibrate", status, frozen_payload, {"prepare": str(prepare["stage_sha256"])})
    _write_new(root / "stages" / "calibrate.json", record)
    result = dict(frozen_payload)
    result.update({"run_id": frozen_payload.get("run_id"), "status": status, "next_stage": frozen_payload.get("next_stage", "report"), "run_root": str(root), "stage_sha256": record["stage_sha256"]})
    return result


def _report(root: Path, config: V1Config, config_path: Path) -> dict[str, object]:
    if (root / "stages" / "report.json").exists():
        record = _read_stage(root / "stages" / "report.json", "report")
        prepare = _read_stage(root / "stages" / "prepare.json", "prepare")
        calibration = _read_stage(root / "stages" / "calibrate.json", "calibrate")
        _validate_run_binding(root, prepare, config, config_path, Path(__file__).resolve().parents[3])
        if calibration.get("status") == "passed":
            raise ValueError("calibration passed; final report is reserved until later study stages complete")
        _validate_calibration_files(root, _mapping(calibration.get("payload"), "calibration payload"))
        _verify_calibration_runtime_code(root, Path(__file__).resolve().parents[3])
        _validate_cost_binding(root, _mapping(calibration.get("payload"), "calibration payload"), config)
        if payload := _mapping(record.get("payload"), "report payload"):
            if payload.get("config_sha256") != sha256_file(config_path):
                raise ValueError("idempotent report config hash changed")
        prerequisites = _mapping(record.get("prerequisite_stage_sha256"), "report prerequisites")
        if prerequisites != {"prepare": prepare.get("stage_sha256"), "calibrate": calibration.get("stage_sha256")}:
            raise ValueError("report prerequisite stage hashes changed")
        return {**_mapping(record.get("payload"), "report payload"), "stage_sha256": record["stage_sha256"], "run_root": str(root)}
    prepare = _read_stage(root / "stages" / "prepare.json", "prepare")
    if not (root / "stages" / "calibrate.json").exists():
        _recover_interrupted_calibration(root, config, config_path, Path(__file__).resolve().parents[3])
    calibration = _read_stage(root / "stages" / "calibrate.json", "calibrate")
    _validate_run_binding(root, prepare, config, config_path, Path(__file__).resolve().parents[3])
    if calibration.get("status") == "passed":
        raise ValueError("calibration passed; final report is reserved until later study stages complete")
    _validate_calibration_files(root, _mapping(calibration.get("payload"), "calibration payload"))
    _verify_calibration_runtime_code(root, Path(__file__).resolve().parents[3])
    _validate_cost_binding(root, _mapping(calibration.get("payload"), "calibration payload"), config)
    if calibration.get("status") not in _TERMINAL:
        raise ValueError("report requires a terminal calibration record")
    if calibration.get("status") == "passed":
        raise ValueError("calibration passed; final report is reserved until later study stages complete")
    prepared_payload = _mapping(prepare.get("payload"), "prepare payload")
    if prepared_payload.get("config_sha256") != sha256_file(config_path):
        raise ValueError("report config hash differs from prepared config")
    cal_payload = _mapping(calibration.get("payload"), "calibration payload")
    payload = {
        "run_id": prepared_payload.get("run_id"), "status": calibration.get("status"),
        "prepare_stage_sha256": prepare["stage_sha256"], "calibration_stage_sha256": calibration["stage_sha256"],
        "config_sha256": sha256_file(config_path),
        "calibration": dict(cal_payload), "next_stage": "complete" if calibration.get("status") in {"stopped_scientific", "stopped_resource", "failed_environment"} else "frozen",
    }
    record = _stage_record("report", str(calibration.get("status")), payload, {"prepare": str(prepare["stage_sha256"]), "calibrate": str(calibration["stage_sha256"])})
    _write_new(root / "stages" / "report.json", record)
    return {**payload, "stage_sha256": record["stage_sha256"], "run_root": str(root)}


def _validate_run_binding(
    root: Path, prepare_record: Mapping[str, object], config: V1Config,
    config_path: Path, project_root: Path,
) -> None:
    payload = _mapping(prepare_record.get("payload"), "prepare payload")
    signatures = _mapping(payload.get("signatures"), "prepare signatures")
    normalized = {str(key): str(value) for key, value in signatures.items()}
    run_id = _derive_run_id(config_path.read_bytes(), normalized)
    if payload.get("run_id") != run_id:
        raise ValueError("prepare run ID does not match config and frozen signatures")
    expected = _resolve(config.data.output_root, project_root) / run_id
    _assert_no_reparse(root)
    resolved_root = root.resolve(strict=True)
    if resolved_root != expected.resolve(strict=True):
        raise ValueError("run root does not match the configured output root and prepare run ID")
    if Path(str(payload.get("run_root"))).resolve(strict=True) != resolved_root:
        raise ValueError("prepare payload run root does not match the active run root")
    if payload.get("config_sha256") != sha256_file(config_path):
        raise ValueError("config hash differs from prepare record")


def _validate_calibration_files(root: Path, payload: Mapping[str, object]) -> None:
    """Reauthenticate every stage-bound non-sealed split and score before reporting."""
    pilot_scores = _mapping(payload.get("pilot_scores", {}), "calibration pilot scores")
    for key, raw_entry in pilot_scores.items():
        if key != "dev-f0" and not (isinstance(key, str) and key.startswith("pilot-")):
            raise ValueError(f"unknown non-sealed calibration evidence key: {key}")
        entry = _mapping(raw_entry, f"calibration evidence {key}")
        if key == "dev-f0":
            split_relative = Path("data/dev.jsonl")
            score_relative = Path("scores/dev-f0.jsonl")
        else:
            split_relative = Path("data") / f"{key}.jsonl"
            score_relative = Path("scores") / f"{key}.jsonl"
        split_path = _safe_run_file(root, split_relative)
        sidecar_path = _safe_run_file(root, Path(str(split_relative) + ".manifest.json"))
        if sha256_file(split_path) != entry.get("split_sha256"):
            raise ValueError(f"stage-bound split hash mismatch for {key}")
        if _raw_line_count(split_path) != entry.get("split_count"):
            raise ValueError(f"stage-bound split count mismatch for {key}")
        if sha256_file(sidecar_path) != entry.get("split_manifest_sha256"):
            raise ValueError(f"stage-bound split sidecar hash mismatch for {key}")
        manifest = _read_json(sidecar_path)
        if manifest.get("sha256") != entry.get("split_sha256") or manifest.get("count") != entry.get("split_count"):
            raise ValueError(f"split sidecar does not match stage binding for {key}")
        ordered_ids = manifest.get("ordered_ids")
        if not isinstance(ordered_ids, list) or len(ordered_ids) != entry.get("split_count") or any(not isinstance(identifier, str) for identifier in ordered_ids):
            raise ValueError(f"split sidecar ordered IDs are invalid for {key}")
        split_rows = _read_jsonl(split_path)
        split_ids = [row.get("id") for row in split_rows]
        if split_ids != ordered_ids or len(set(split_ids)) != len(split_ids):
            raise ValueError(f"stage-bound split ordered IDs mismatch for {key}")
        score_hash, score_count = entry.get("score_sha256"), entry.get("score_count")
        if score_hash is None and score_count is None:
            continue
        score_path = _safe_run_file(root, score_relative)
        if sha256_file(score_path) != score_hash or _raw_line_count(score_path) != score_count:
            raise ValueError(f"stage-bound score hash/count mismatch for {key}")
        score_rows = _read_jsonl(score_path)
        score_ids = [row.get("example_id") for row in score_rows]
        if entry.get("partial") is True:
            if score_ids != ordered_ids[:len(score_ids)]:
                raise ValueError(f"stage-bound partial score ordered IDs mismatch for {key}")
        elif score_ids != ordered_ids:
            raise ValueError(f"stage-bound score ordered IDs mismatch for {key}")
        summary = entry.get("summary")
        if isinstance(summary, Mapping) and summarize_exact_scores(score_rows) != summary:
            raise ValueError(f"stage-bound score summary mismatch for {key}")
    partial = payload.get("partial_failure")
    if partial is not None:
        partial = _mapping(partial, "partial score evidence")
        relative = Path(str(partial.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("partial score evidence path is unsafe")
        partial_path = _safe_run_file(root, relative)
        if sha256_file(partial_path) != partial.get("raw_sha256") or _raw_line_count(partial_path) != partial.get("row_count"):
            raise ValueError("partial score evidence hash/count mismatch")
    expected_splits = {"data/dev.jsonl"} if "dev-f0" in pilot_scores else set()
    expected_scores = set()
    for key, raw_entry in pilot_scores.items():
        entry = _mapping(raw_entry, f"calibration evidence {key}")
        split = "data/dev.jsonl" if key == "dev-f0" else f"data/{key}.jsonl"
        score = "scores/dev-f0.jsonl" if key == "dev-f0" else f"scores/{key}.jsonl"
        expected_splits.add(split)
        if split != "data/sealed.jsonl":
            expected_splits.add(split + ".manifest.json")
        if entry.get("score_sha256") is not None:
            expected_scores.add(score)
    if isinstance(partial, Mapping) and partial.get("path"):
        expected_scores.add(str(partial["path"]))
    for folder, expected in ((root / "data", expected_splits), (root / "scores", expected_scores)):
        actual: set[str] = set()
        if folder.exists():
            for item in folder.rglob("*.jsonl"):
                relative = item.relative_to(root).as_posix()
                if relative in {"data/sealed.jsonl", "data/sealed.jsonl.manifest.json"}:
                    continue
                actual.add(relative)
            if folder.name == "data":
                actual.update(
                    relative for item in folder.rglob("*.manifest.json")
                    if (relative := item.relative_to(root).as_posix()) != "data/sealed.jsonl.manifest.json"
                )
        if actual != expected:
            raise ValueError(f"unbound or missing non-sealed calibration files: {sorted(actual ^ expected)[:1]}")


def _safe_run_file(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("run evidence path escapes the V1 run root")
    path = root / relative
    _assert_no_reparse(path)
    resolved = path.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    if not resolved.is_file():
        raise ValueError("run evidence path must be a regular file")
    return resolved


def _budget_check(root: Path, ledger: CudaCostLedger, config: V1Config) -> None:
    artifacts = _tree_bytes(root)
    if _calibration_cost_seconds(ledger) >= config.resources.maximum_calibration_cuda_seconds:
        raise ResourceLimitReached("calibration CUDA-active time cap reached")
    if ledger.total_seconds >= config.resources.maximum_study_cuda_seconds:
        raise ResourceLimitReached("whole-study CUDA-active time cap reached")
    if artifacts >= config.resources.maximum_artifact_bytes:
        raise ResourceLimitReached("new artifact byte cap reached")


def _assert_calibration_not_started(root: Path) -> None:
    if (root / "stages" / "calibration-attempt.json").exists():
        raise FileExistsError("calibration attempt already exists; retries require a new run identity")


def _write_attempt_marker(root: Path, prepare: Mapping[str, object], config_path: Path) -> None:
    path = root / "stages" / "calibration-attempt.json"
    payload = {
        "run_id": _mapping(prepare.get("payload"), "prepare payload").get("run_id"),
        "config_sha256": sha256_file(config_path),
        "started_utc": _utc_now(), "wall_started_monotonic": time.monotonic(),
        "process_id": os.getpid(),
    }
    _write_new(path, _stage_record("calibration-attempt", "running", payload, {"prepare": str(prepare["stage_sha256"])}))
    _append_ledger(root, {"event": "calibration_attempt_started", **payload})


def _start_phase(root: Path, phase: str, path: str | None = None, metadata: Mapping[str, object] | None = None) -> str:
    phase_id = str(uuid.uuid4())
    _append_ledger(root, {
        "event": "phase_started", "phase_id": phase_id, "phase": phase,
        "process_id": os.getpid(), "wall_started_utc": _utc_now(),
        "monotonic_started": time.monotonic(), "path": path, **dict(metadata or {}),
    })
    return phase_id


def _complete_phase(
    root: Path, phase_id: str, stage: str, cuda_seconds: float, outcome: str,
    evidence: Mapping[str, object] | None = None,
) -> None:
    _append_ledger(root, {
        "event": "phase_completed", "phase_id": phase_id, "stage": stage,
        "cuda_active_seconds": float(cuda_seconds), "outcome": outcome,
        "completed_utc": _utc_now(), **dict(evidence or {}),
    })


def _record_phase_cost(
    root: Path, ledger: CudaCostLedger, phase_id: str, stage: str,
    cuda_seconds: float, outcome: str,
) -> None:
    ledger.record(phase_id, stage, cuda_seconds, outcome)
    _complete_phase(root, phase_id, stage, cuda_seconds, outcome)


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _recover_interrupted_calibration(
    root: Path, config: V1Config, config_path: Path, project_root: Path,
) -> dict[str, object]:
    prepare = _read_stage(root / "stages" / "prepare.json", "prepare")
    _validate_run_binding(root, prepare, config, config_path, project_root)
    _assert_no_reparse(root / "resources" / "progress.jsonl")
    events = _read_jsonl(root / "resources" / "progress.jsonl") if (root / "resources" / "progress.jsonl").exists() else []
    started = {str(row.get("phase_id")): row for row in events if row.get("event") == "phase_started"}
    completed = {str(row.get("phase_id")) for row in events if row.get("event") in {"phase_completed", "phase_recovered"}}
    open_phases = [event for phase_id, event in started.items() if phase_id not in completed]
    attempt = _read_stage(root / "stages" / "calibration-attempt.json", "calibration-attempt") if (root / "stages" / "calibration-attempt.json").exists() else None
    if attempt is None and not open_phases:
        raise FileNotFoundError("no durable calibration attempt exists to recover")
    if attempt is not None:
        attempt_payload = _mapping(attempt.get("payload"), "calibration attempt payload")
        prepare_payload = _mapping(prepare.get("payload"), "prepare payload")
        if attempt_payload.get("run_id") != prepare_payload.get("run_id") or attempt_payload.get("config_sha256") != sha256_file(config_path):
            raise ValueError("calibration attempt marker does not bind the prepared run and config")
        if _mapping(attempt.get("prerequisite_stage_sha256"), "attempt prerequisites") != {"prepare": prepare.get("stage_sha256")}:
            raise ValueError("calibration attempt marker prerequisite hash changed")
        pid = int(attempt_payload.get("process_id", -1))
        if pid != os.getpid() and _process_is_alive(pid):
            raise RuntimeError("calibration process is still active; report will not race it")
    ledger = _load_cost_ledger(root, float(config.resources.maximum_study_cuda_seconds))
    now_mono = time.monotonic()
    wall_upper = 0.0
    for event in open_phases:
        try:
            wall_upper = max(wall_upper, max(0.0, now_mono - float(event["monotonic_started"])))
        except (KeyError, TypeError, ValueError):
            wall_upper = _WALL_SAFETY_SECONDS
    if attempt is not None:
        try:
            wall_upper = max(wall_upper, max(0.0, now_mono - float(_mapping(attempt.get("payload"), "attempt payload")["wall_started_monotonic"])))
        except (KeyError, TypeError, ValueError):
            wall_upper = _WALL_SAFETY_SECONDS
    status = "stopped_resource" if wall_upper >= _WALL_SAFETY_SECONDS else "failed_environment"
    score_entries: dict[str, object] = {}
    partial: dict[str, object] | None = None
    for event in events:
        if event.get("event") == "split_frozen":
            key = str(event["score_key"])
            score_entries[key] = {key_name: event[key_name] for key_name in ("split_sha256", "split_manifest_sha256", "split_count")}
        if event.get("event") == "phase_started" and event.get("phase") == "score":
            key, rel = str(event.get("score_key", "")), str(event.get("path", ""))
            if key and key in score_entries:
                score_path = root / Path(rel)
                if score_path.exists():
                    partial = {"path": rel, "raw_sha256": sha256_file(score_path), "row_count": _raw_line_count(score_path)}
                    score_entries[key] = {**_mapping(score_entries[key], "split evidence"), "score_sha256": partial["raw_sha256"], "score_count": partial["row_count"], "partial": True}
    payload: dict[str, object] = {
        "run_id": _mapping(prepare.get("payload"), "prepare payload").get("run_id"),
        "status": status, "failure": "process ended before calibration stage was durably completed",
        "pilot_scores": score_entries, "partial_failure": partial,
        "resources": {
            "calibration_cuda_active_seconds": _calibration_cost_seconds(ledger),
            "cuda_budget_state": "unknown_after_process_loss",
            "wall_elapsed_upper_bound_seconds": wall_upper,
            "resource_bound_fired": "wall_safety_deadline" if status == "stopped_resource" else "process_interruption",
            "artifact_bytes": _tree_bytes(root),
        }, "next_stage": "report",
        "runtime_code_files": _runtime_code_files(project_root),
    }
    for event in open_phases:
        _append_ledger(root, {"event": "phase_recovered", "phase_id": event.get("phase_id"), "recovered_utc": _utc_now(), "cuda_budget_state": "unknown_after_process_loss"})
    return _finish_calibration(root, payload, status)


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _append_ledger(root: Path, item: Mapping[str, object]) -> None:
    path = root / "resources" / "progress.jsonl"
    _assert_no_reparse(path)
    encoded = canonical_json(dict(item))
    with path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _ledger_seconds(root: Path) -> float:
    return _calibration_cost_seconds(_load_cost_ledger(root, 86400.0))


def _load_cost_ledger(root: Path, cap_seconds: float) -> CudaCostLedger:
    path = root / "resources" / "cuda-cost.jsonl"
    _assert_no_reparse(path)
    ledger = CudaCostLedger.load(path, cap_seconds)
    if path.exists():
        encoded = _read_bytes(path)
        if encoded and not encoded.endswith(b"\n"):
            raise ValueError("CUDA cost ledger is truncated (missing final newline)")
    return ledger


def _calibration_cost_seconds(ledger: CudaCostLedger) -> float:
    return sum(entry.elapsed_seconds for entry in ledger.entries if entry.stage.startswith("calibration-"))


def _validate_cost_binding(root: Path, payload: Mapping[str, object], config: V1Config) -> None:
    ledger = _load_cost_ledger(root, float(config.resources.maximum_study_cuda_seconds))
    expected_hash = payload.get("cuda_cost_ledger_sha256")
    if expected_hash != _optional_sha256(root / "resources" / "cuda-cost.jsonl"):
        raise ValueError("anchored CUDA cost ledger hash differs from calibration stage")
    if float(payload.get("cuda_cost_ledger_total_seconds", -1)) != ledger.total_seconds:
        raise ValueError("anchored CUDA cost ledger total differs from calibration stage")
    if float(payload.get("calibration_cuda_active_seconds", -1)) != _calibration_cost_seconds(ledger):
        raise ValueError("anchored calibration CUDA total differs from calibration stage")


def _optional_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def _tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        _assert_no_reparse(path)
        if path.is_file():
            total += path.stat().st_size
    return total


def _stage_record(stage: str, status: str, payload: Mapping[str, object], prerequisites: Mapping[str, str]) -> dict[str, object]:
    base = {"schema_version": 1, "stage": stage, "status": status, "prerequisite_stage_sha256": dict(prerequisites), "payload": dict(payload)}
    return {**base, "stage_sha256": sha256_payload(base)}


def _read_stage(path: Path, expected_stage: str) -> dict[str, object]:
    record = _read_json(path)
    if record.get("stage") != expected_stage or record.get("schema_version") != 1:
        raise ValueError(f"{expected_stage} stage record has an unsupported schema")
    digest = record.get("stage_sha256")
    unsigned = {key: value for key, value in record.items() if key != "stage_sha256"}
    if not isinstance(digest, str) or sha256_payload(unsigned) != digest:
        raise ValueError(f"{expected_stage} stage hash mismatch")
    return record


def _write_new(path: Path, payload: Mapping[str, object]) -> None:
    _assert_no_reparse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical_json(dict(payload)))
        stream.flush()
        os.fsync(stream.fileno())


def _parse_rows(raw: bytes, split: str, expected_count: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"V0 {split} row {number} is invalid JSON") from error
        if not isinstance(value, dict) or value.get("split") != split:
            raise ValueError(f"V0 {split} row {number} has invalid schema or split")
        identifier = value.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"V0 {split} IDs must be non-empty and unique")
        seen.add(identifier)
        rows.append(value)
    if len(rows) != expected_count:
        raise ValueError(f"V0 {split} count is {len(rows)}, expected {expected_count}")
    return rows


def _resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).absolute()


def _read_bytes(path: Path) -> bytes:
    _assert_no_reparse(path)
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"required V1 source input is unreadable: {path}") from error


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(_read_bytes(path))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _parse_json(raw: bytes, description: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _assert_no_reparse(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise ValueError(f"symlink or reparse point is not allowed: {current}")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            rows.append(row)
    return rows


def _raw_line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)
